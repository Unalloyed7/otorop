import asyncio
import json
from pathlib import Path
import zipfile

import httpx
import pytest

from assistant import LegalAssistant
from conftest import FakeSearch
from diagnostics import active_run, append_report, capture_run
from llm_client import LLMClient
from solution import run_submission
from tools.evaluate_answers import build_case_reviews, compare_records, load_case_suite
from tools.package_submission import package


async def test_report_is_scoped_per_concurrent_question(settings, evidence, tmp_path):
    def handler(request):
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        if "Разбери последнюю" in prompt:
            result = {"action": "search", "query": "Оплата", "parts": ["Оплата"]}
        elif "Независимо проверь" in prompt:
            result = {"supported": [0]}
        else:
            result = {"status": "answer", "claims": [{"text": "Оплата в течение 30 календарных дней после подписания УПД.", "addresses": [0], "supports": [{"chunk_id": evidence.chunk_id, "document_id": evidence.document_id, "quote": evidence.text.split("\n")[0]}]}]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result, ensure_ascii=False)}}]})
    (tmp_path / "questions.csv").write_text('id,question\n001,Срок оплаты?\n002,Проверить оплату?\n', encoding="utf-8")
    async with LLMClient(settings, transport=httpx.MockTransport(handler)) as client:
        engine = LegalAssistant(FakeSearch([evidence]), client)
        counts = await run_submission(tmp_path, tmp_path / "submission.csv", engine=engine, workers=2, report_path=tmp_path / "run.jsonl")
    assert counts == {"answer": 2}
    rows = [json.loads(line) for line in (tmp_path / "run.jsonl").read_text().split("\n") if line]
    assert {r["id"] for r in rows} == {"001", "002"}
    assert all(r["calls"] == {"/chat/completions": 3} for r in rows)
    assert all(len(r["api_calls"]) == 3 and all(c["http_status"] == 200 for c in r["api_calls"]) for r in rows)
    assert all([o["schema"] for o in r["model_outputs"]] == ["Intent", "Draft", "Verdict"] for r in rows)
    assert all(r["answer"]["citations"] and any(s["stage"] == "search" for s in r["stages"]) for r in rows)
    assert active_run() is None
    assert "test-only-key" not in (tmp_path / "run.jsonl").read_text()


def test_redaction_and_jsonl_control_characters(tmp_path):
    path = tmp_path / "run.jsonl"
    append_report(path, {"question": "secret\u0085text", "answer": {"text": "secret"}}, secrets=("secret",))
    text = path.read_text()
    assert "secret" not in text
    assert json.loads(text)["question"] == "[REDACTED]\u0085text"


async def test_report_records_invalid_model_json(settings):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "not JSON"}}]}))
    async with LLMClient(settings, transport=transport) as client:
        with capture_run() as run:
            answer = await LegalAssistant(FakeSearch(), client).answer("Вопрос", trace=run)
    assert answer.status == "error"
    output = run["model_outputs"][0]
    assert output["schema"] == "Intent" and output["output"] == "not JSON" and not output["valid"]
    assert output["errors"]
    assert run["api_calls"][0]["http_status"] == 200
    assert any(s["stage"] == "error" for s in run["stages"])


async def test_deadline_report_contains_all_ids(tmp_path, monkeypatch, settings):
    class Slow:
        async def answer(self, *args, **kwargs): await asyncio.sleep(5)
    (tmp_path / "questions.csv").write_text("id,question\n1,a\n2,b\n", encoding="utf-8")
    monkeypatch.setenv("RUN_TIMEOUT", "0.01")
    await run_submission(tmp_path, tmp_path / "out.csv", engine=Slow(), workers=1, report_path=tmp_path / "run.jsonl")
    rows = [json.loads(line) for line in (tmp_path / "run.jsonl").read_text().split("\n") if line]
    assert {r["id"] for r in rows} == {"1", "2"}
    assert all(r["answer"]["status"] == "error" for r in rows)


def test_packaged_options_enable_report_and_select_variant(tmp_path):
    archive = package(tmp_path / "report.zip", with_report=True, verify_answer=False)
    with zipfile.ZipFile(archive) as z:
        options = z.read("submission_options.py").decode()
        assert 'RUN_REPORT = \'run-report.jsonl\'' in options
        assert "VERIFY_ANSWER = False" in options
        assert "temporal.py" in z.namelist() and "diagnostics.py" in z.namelist()


def test_comparison_keeps_manual_quality_unscored():
    a = {"id": "1", "answer": {"status": "answer"}, "calls": {"/chat/completions": 2}, "stages": []}
    b = {**a, "answer": {"status": "partial"}, "calls": {"/chat/completions": 3}}
    result = compare_records([a], [b])[0]
    assert result["same_evidence"]
    assert result["manual_review"]["attack_resisted"] is None
    assert result["baseline_calls"] != result["checked_calls"]


def test_case_suite_resolves_official_and_local_questions(tmp_path):
    source_questions = tmp_path / "questions.csv"
    source_questions.write_text("id,question\n001,Официальный вопрос\n", encoding="utf-8")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps({
        "documents": {"contract": "Договор № 1"},
        "cases": [
            {"question_id": "001", "expected_disposition": "answer", "required_claims": []},
            {"id": "challenge-19", "question_id": None, "question": "Локальный вопрос",
             "expected_disposition": "answer", "required_claims": []},
        ],
    }), encoding="utf-8")

    cases, documents, rows = load_case_suite(cases_path, source_questions)

    assert [row[0] for row in rows] == ["001", "challenge-19"]
    assert [row[1] for row in rows] == ["Официальный вопрос", "Локальный вопрос"]
    assert len(cases) == 2 and documents == {"contract": "Договор № 1"}


def test_checked_in_case_annotations_load_as_one_suite(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    cases_path = repo_root / "evals" / "answer_cases.json"
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    official_ids = sorted({
        case["question_id"] for case in payload["cases"]
        if case.get("question_id") is not None
    })
    questions_path = tmp_path / "questions.csv"
    questions_path.write_text(
        "id,question\n" + "".join(f"{question_id},Источник {question_id}\n" for question_id in official_ids),
        encoding="utf-8",
    )

    cases, documents, rows = load_case_suite(cases_path, questions_path)

    assert len(cases) == len(rows) == 28
    assert len({row[0] for row in rows}) == 28
    assert sum(row[0].startswith("challenge-") for row in rows) == 10
    assert documents == payload["documents"]


def test_case_review_checks_retrieval_but_leaves_answer_quality_manual():
    cases = [{
        "id": "challenge-19",
        "question": "Срок оплаты?",
        "expected_disposition": "answer",
        "required_claims": [{
            "fact": "Оплата в течение 30 дней.",
            "evidence": [{"document": "contract", "section": "4.2"}],
        }],
        "must_not_claim": ["Срок оплаты — 90 дней."],
    }]
    record = {
        "id": "challenge-19",
        "question": "Срок оплаты?",
        "answer": {"status": "answer", "text": "30 дней"},
        "calls": {"/chat/completions": 2},
        "stages": [{
            "stage": "search",
            "evidence": [{
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "document_title": "Договор № 1",
                "section": "п. 4.2",
                "text": "Оплата в течение 30 календарных дней.",
            }],
        }],
    }

    review = build_case_reviews(cases, {"contract": "Договор № 1"}, [record])[0]

    assert review["retrieval"] == {"expected_sources_found": 1, "expected_sources_total": 1}
    assert review["expected"]["required_claims"][0]["sources"][0]["retrieved"] is True
    assert review["manual_review"]["disposition_correct"] is None
    assert review["manual_review"]["must_not_claims"] == [
        {"claim": "Срок оплаты — 90 дней.", "violated": None},
    ]


def test_case_review_marks_missing_run_as_unassessed():
    cases = [{
        "id": "challenge-19",
        "expected_disposition": "answer",
        "required_claims": [{
            "fact": "Оплата в течение 30 дней.",
            "evidence": [{"document": "contract", "section": "4.2"}],
        }],
    }]

    review = build_case_reviews(cases, {"contract": "Договор № 1"}, [])[0]

    assert review["run"] is None
    assert review["retrieval"] == {"expected_sources_found": None, "expected_sources_total": 1}
    assert review["expected"]["required_claims"][0]["sources"][0]["retrieved"] is None


async def test_report_preserves_fences_before_json_normalization(settings):
    from schemas import Intent
    raw = '```json\n{"action":"out_of_scope","query":"q","parts":["q"]}\n```'
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": raw}}]}))
    async with LLMClient(settings, transport=transport) as client:
        with capture_run() as run:
            result = await client.structured("instruction", {}, Intent)
    assert result.action == "out_of_scope"
    assert run["model_outputs"][0]["output"] == raw and run["model_outputs"][0]["valid"]


async def test_report_keeps_truncated_output_and_finish_reason(settings):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": '{"action":'}, "finish_reason": "length"}]}))
    async with LLMClient(settings, transport=transport) as client:
        with capture_run() as run:
            answer = await LegalAssistant(FakeSearch(), client).answer("Вопрос", trace=run)
    assert answer.status == "error"
    assert run["chat_responses"][0]["finish_reason"] == "length"
    assert run["chat_responses"][0]["output"] == '{"action":'


def test_evaluation_cache_is_isolated_and_user_setting_restored(tmp_path, monkeypatch):
    import os
    from tools.evaluate_answers import isolated_embedding_cache
    owned = tmp_path / "user-cache"
    owned.mkdir()
    (owned / "keep").write_text("keep")
    monkeypatch.setenv("OTOROP_EMBEDDING_CACHE_DIR", str(owned))
    with isolated_embedding_cache():
        first = Path(os.environ["OTOROP_EMBEDDING_CACHE_DIR"])
        assert first != owned and first.exists()
    assert not first.exists() and (owned / "keep").read_text() == "keep"
    assert os.environ["OTOROP_EMBEDDING_CACHE_DIR"] == str(owned)
    with isolated_embedding_cache():
        assert Path(os.environ["OTOROP_EMBEDDING_CACHE_DIR"]) != first
