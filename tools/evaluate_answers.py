"""Сохранить реальные ответы и сравнить VERIFY_ANSWER=0/1 без автоматического скора."""
from __future__ import annotations

import argparse
import asyncio
import csv
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assistant import LegalAssistant
from llm_client import LLMClient, ModelAPIError, Settings, env_value
from search_adapter import build_search
from solution import load_questions, run_submission


def read_records(path):
    # U+0085 внутри текста не является границей JSONL.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def case_questions_path(data_dir: Path) -> Path:
    for filename in ("questions.csv", "demo_questions.csv"):
        candidate = data_dir / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"В {data_dir} нужен questions.csv или demo_questions.csv из материалов организаторов"
    )


def load_case_suite(cases_path: Path, questions_path: Path):
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("Файл кейсов должен содержать массив cases")
    documents = payload.get("documents")
    if not isinstance(documents, dict):
        raise ValueError("Файл кейсов должен содержать словарь documents")

    source_questions = dict(load_questions(questions_path))
    cases = payload["cases"]
    rows = []
    seen = set()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Кейс {index} должен быть объектом")
        question_id = case.get("question_id")
        case_id = case.get("id")
        if question_id is not None:
            if not isinstance(question_id, str) or question_id not in source_questions:
                raise ValueError(f"Для кейса {case_id or index} нет вопроса в questions.csv")
            run_id, question = question_id, source_questions[question_id]
        else:
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(f"У локального кейса {index} отсутствует id")
            run_id, question = case_id, case.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"У локального кейса {case_id} отсутствует question")
        if run_id in seen:
            raise ValueError(f"Повторяющийся ID кейса: {run_id}")
        seen.add(run_id)
        for claim in case.get("required_claims", []):
            for source in claim.get("evidence", []):
                if source.get("document") not in documents:
                    raise ValueError(f"Неизвестный алиас документа в кейсе {run_id}")
        rows.append((run_id, question))
    if not rows:
        raise ValueError("Файл кейсов не содержит сценариев")
    return cases, documents, rows


def _normalized_source(value: str) -> str:
    value = re.sub(r"^\s*(?:пункт|п\.|раздел|статья|ст\.)\s*", "", value, flags=re.I)
    return " ".join(value.casefold().replace("ё", "е").split())


def build_case_reviews(cases, documents, records, questions=None):
    records_by_id = {}
    for record in records:
        record_id = record.get("id")
        if not isinstance(record_id, str) or record_id in records_by_id:
            raise ValueError("В отчёте отсутствует ID или он повторяется")
        records_by_id[record_id] = record

    reviews = []
    for case in cases:
        case_id = case.get("id") or case.get("question_id")
        record = records_by_id.get(case_id)
        search_evidence = [
            evidence
            for stage in (record or {}).get("stages", [])
            if stage.get("stage") == "search"
            for evidence in stage.get("evidence", [])
        ]
        claim_reviews = []
        for claim in case.get("required_claims", []):
            source_reviews = []
            for expected in claim.get("evidence", []):
                title = documents[expected["document"]]
                matches = [
                    evidence
                    for evidence in search_evidence
                    if _normalized_source(evidence.get("document_title", "")) == _normalized_source(title)
                    and _normalized_source(evidence.get("section", "")) == _normalized_source(expected["section"])
                ]
                source_reviews.append({
                    "document": expected["document"],
                    "document_title": title,
                    "section": expected["section"],
                    "retrieved": bool(matches) if record is not None else None,
                    "matching_chunks": [
                        {key: evidence.get(key) for key in ("document_id", "chunk_id", "document_title", "section", "text")}
                        for evidence in matches
                    ],
                })
            claim_reviews.append({
                "fact": claim["fact"],
                "sources": source_reviews,
                "supported": None,
                "citation_correct": None,
            })
        reviews.append({
            "id": case_id,
            "question": (
                record.get("question") if record else None
            ) or (questions or {}).get(case_id) or case.get("question"),
            "expected": {
                "disposition": case["expected_disposition"],
                "required_claims": claim_reviews,
                "required_limitation": case.get("required_limitation"),
                "must_not_claim": case.get("must_not_claim", []),
            },
            "retrieval": {
                "expected_sources_found": None if record is None else sum(
                    source["retrieved"] is True for claim in claim_reviews for source in claim["sources"]
                ),
                "expected_sources_total": sum(len(claim["sources"]) for claim in claim_reviews),
            },
            "run": None if record is None else {
                "answer": record.get("answer"),
                "csv_answer": record.get("csv_answer"),
                "duration_ms": record.get("duration_ms"),
                "calls": record.get("calls"),
                "model_outputs": record.get("model_outputs", []),
                "search_evidence": search_evidence,
            },
            "manual_review": {
                "disposition_correct": None,
                "limitation_correct": None,
                "must_not_claims": [
                    {"claim": text, "violated": None}
                    for text in case.get("must_not_claim", [])
                ],
                "notes": "",
            },
        })
    return reviews


def write_questions_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["id", "question"])
        writer.writerows(rows)


@contextmanager
def isolated_embedding_cache():
    """Оба режима начинают с холодного кэша; пользовательский кэш не удаляется."""
    key = "OTOROP_EMBEDDING_CACHE_DIR"
    previous = os.environ.get(key)
    with tempfile.TemporaryDirectory(prefix="otorop-eval-embeddings-") as directory:
        os.environ[key] = directory
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def compare_records(baseline, checked):
    a, b = {r["id"]: r for r in baseline}, {r["id"]: r for r in checked}
    if set(a) != set(b):
        raise ValueError("Наборы ID в двух прогонах различаются")
    def sources(record):
        return sorted({(e["document_id"], e["chunk_id"], e["text"]) for s in record.get("stages", []) if s["stage"] == "search" for e in s["evidence"]})
    rows = []
    for qid in a:
        left, right = a[qid], b[qid]
        rows.append({"id": qid, "baseline_status": left["answer"]["status"], "checked_status": right["answer"]["status"], "baseline_calls": left["calls"], "checked_calls": right["calls"], "same_evidence": sources(left) == sources(right), "manual_review": {"baseline_supported": None, "checked_supported": None, "unnecessary_refusal": None, "attack_resisted": None}})
    return rows


async def evaluate(data_dir: Path, output: Path, mode: str, cases_path: Path | None = None):
    settings = Settings.from_env()  # Никаких ключей/адресов в manifest и stdout.
    questions_path = case_questions_path(data_dir) if cases_path else data_dir / "questions.csv"
    case_suite = load_case_suite(cases_path, questions_path) if cases_path else None
    output.mkdir(parents=True, exist_ok=False)
    variants = [False, True] if mode == "both" else [mode == "1"]
    manifest = {"model": settings.model, "embedding_model": settings.embedding_model, "reranker_model": settings.reranker_model, "temperature": settings.temperature, "max_tokens": settings.max_tokens, "search_retries": settings.search_retries, "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest(), "evaluation": "independent runs; quality requires manual review; not an official score", "case_suite": {"count": len(case_suite[0]), "sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(), "automatic_semantic_scoring": False} if case_suite else None, "variants": {}}
    records = {}
    for verify in variants:
        label = "verify-1" if verify else "verify-0"
        with isolated_embedding_cache():
            async with LLMClient(settings) as client:
                backend = await build_search(client, data_dir=data_dir)
                engine = LegalAssistant(backend, client, verify=verify)
                report = output / (label + ".jsonl")
                if case_suite:
                    with tempfile.TemporaryDirectory(prefix="otorop-case-questions-") as directory:
                        case_data_dir = Path(directory)
                        write_questions_csv(case_data_dir / "questions.csv", case_suite[2])
                        counts = await run_submission(case_data_dir, output / (label + ".csv"), engine=engine, workers=settings.max_workers, report_path=report)
                else:
                    counts = await run_submission(data_dir, output / (label + ".csv"), engine=engine, workers=settings.max_workers, report_path=report)
                records[label] = read_records(report)
                calls = Counter()
                for record in records[label]: calls.update(record.get("calls", {}))
                manifest["variants"][label] = {"statuses": counts, "calls": dict(calls), "embedding_cache": "isolated cold cache per variant", "index_fingerprint": getattr(getattr(backend, "index", None), "fingerprint", None)}
                if case_suite:
                    reviews = build_case_reviews(
                        case_suite[0], case_suite[1], records[label], dict(case_suite[2])
                    )
                    review_path = output / f"case-review-{label}.jsonl"
                    review_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reviews), encoding="utf-8")
    if mode == "both":
        (output / "comparison.json").write_text(json.dumps(compare_records(records["verify-0"], records["verify-1"]), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Model IDs выдаёт оператор; страховка от случайного повторения ключа в них.
    raw = json.dumps(manifest, ensure_ascii=False, indent=2)
    raw = raw.replace(settings.api_key.get_secret_value(), "[REDACTED]")
    (output / "manifest.json").write_text(raw + "\n", encoding="utf-8")
    print("Прогон сохранён: " + str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(env_value("DATA_DIR", default="data")))
    parser.add_argument("--cases", type=Path, help="Запустить размеченный набор из evals/answer_cases.json")
    parser.add_argument("--output", type=Path, default=Path("reports") / datetime.now(timezone(timedelta(hours=3))).strftime("run-%Y%m%d-%H%M%S"))
    parser.add_argument("--verify", choices=["0", "1", "both"], default="both")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        asyncio.run(evaluate(args.data_dir, args.output, args.verify, args.cases))
    except ModelAPIError:
        print("Доступ к API не настроен. Официальный прокси выдаётся только при запуске на платформе; локально нужны собственные разрешённые реквизиты.", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"Ошибка входных файлов или записи отчёта: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
