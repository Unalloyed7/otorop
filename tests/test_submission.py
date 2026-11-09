import csv
from pathlib import Path

import pytest

from assistant import LegalAssistant
from conftest import FakeClient, FakeSearch, draft, intent
from schemas import Answer
from solution import SubmissionError, load_questions, render_answer, run_submission, validate_submission, write_submission


async def test_csv_same_engine_real_file(settings, evidence, tmp_path):
    rows = [("001", 'Договор, "условия"\nСрок оплаты?'), ("002", "Иной договор?")]
    with (tmp_path / "questions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "question"]); w.writerows(rows)
    client = FakeClient(settings, intent(), draft(evidence), {"supported": [0]}, intent().model_copy(update={"action": "clarify", "clarification": "Какой договор?"}))
    engine = LegalAssistant(FakeSearch([evidence]), client)
    counts = await run_submission(tmp_path, tmp_path / "submission.csv", engine=engine, workers=1)
    validate_submission(tmp_path / "submission.csv", ["001", "002"])
    with (tmp_path / "submission.csv").open(encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    assert counts == {"answer": 1, "clarify": 1}
    assert evidence.document_title in records[0]["answer"] and "4.2" in records[0]["answer"]
    assert len(client.requests[3][1]["history"]) == 1
    assert "001" not in str(client.requests)


def test_quotes_commas_newlines_unicode_and_length(tmp_path):
    texts = ['Первый, "цитата"\nВторая строка\u0085не перевод CSV', "а" * 10000]
    path = tmp_path / "submission.csv"
    write_submission(path, [("01", "q"), ("02", "q")], texts)
    validate_submission(path, ["01", "02"])
    with path.open(encoding="utf-8", newline="") as f:
        assert [r["answer"] for r in csv.DictReader(f)] == texts


@pytest.mark.parametrize("content", ['id,question\n1,a\n1,b\n', 'id,question\n,a\n', 'id,question\n1,\n', 'id,question\n1,a,extra\n', 'id,query\n1,a\n', 'id,question\n'])
def test_invalid_inputs(tmp_path, content):
    p = tmp_path / "questions.csv"; p.write_text(content)
    with pytest.raises(SubmissionError): load_questions(p)


@pytest.mark.parametrize("texts", [[""], [" "], ["x" * 10001], []])
def test_reject_invalid_submission(tmp_path, texts):
    with pytest.raises(SubmissionError): write_submission(tmp_path / "submission.csv", [("1", "q")], texts)


async def test_errors_still_produce_one_answer_per_id(tmp_path, monkeypatch):
    (tmp_path / "questions.csv").write_text("id,question\n01,Вопрос\n02,Другой\n", encoding="utf-8")
    for name in ["LLM_BASE_URL", "OPENAI_BASE_URL"]: monkeypatch.delenv(name, raising=False)
    counts = await run_submission(tmp_path, tmp_path / "submission.csv")
    assert counts == {"error": 2}
    validate_submission(tmp_path / "submission.csv", ["01", "02"])
    assert "Техническая" in (tmp_path / "submission.csv").read_text()


def test_large_answer_is_not_cut_mid_condition(evidence):
    from schemas import Citation
    c = Citation(document_id=evidence.document_id, chunk_id=evidence.chunk_id, document_title=evidence.document_title, section=evidence.section, quote="я" * 4000, coordinates=evidence.coordinates, edition="Тест", claim_index=0)
    with pytest.raises(SubmissionError): render_answer(Answer(status="answer", text="я" * 7000, citations=[c]))


async def test_run_budget_preserves_completed_answers_and_all_ids(tmp_path, monkeypatch):
    import asyncio
    class SlowEngine:
        async def answer(self, question):
            if question == "slow": await asyncio.sleep(5)
            return Answer(status="out_of_scope", text="Ответ на " + question)
    (tmp_path / "questions.csv").write_text("id,question\n1,fast\n2,slow\n3,later\n", encoding="utf-8")
    monkeypatch.setenv("RUN_TIMEOUT", "0.01")
    counts = await run_submission(tmp_path, tmp_path / "submission.csv", engine=SlowEngine(), workers=1)
    assert counts == {"out_of_scope": 1, "error": 2}
    validate_submission(tmp_path / "submission.csv", ["1", "2", "3"])
