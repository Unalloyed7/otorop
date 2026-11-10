"""Сохранить реальные ответы и сравнить VERIFY_ANSWER=0/1 без автоматического скора."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assistant import LegalAssistant
from llm_client import LLMClient, ModelAPIError, Settings, env_value
from search_adapter import build_search
from solution import run_submission


def read_records(path):
    # U+0085 внутри текста не является границей JSONL.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


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


async def evaluate(data_dir: Path, output: Path, mode: str):
    settings = Settings.from_env()  # Никаких ключей/адресов в manifest и stdout.
    output.mkdir(parents=True, exist_ok=False)
    variants = [False, True] if mode == "both" else [mode == "1"]
    manifest = {"model": settings.model, "embedding_model": settings.embedding_model, "reranker_model": settings.reranker_model, "temperature": settings.temperature, "max_tokens": settings.max_tokens, "search_retries": settings.search_retries, "questions_sha256": hashlib.sha256((data_dir / "questions.csv").read_bytes()).hexdigest(), "evaluation": "independent runs; quality requires manual review; not an official score", "variants": {}}
    records = {}
    for verify in variants:
        label = "verify-1" if verify else "verify-0"
        with isolated_embedding_cache():
            async with LLMClient(settings) as client:
                backend = await build_search(client)
                engine = LegalAssistant(backend, client, verify=verify)
                report = output / (label + ".jsonl")
                counts = await run_submission(data_dir, output / (label + ".csv"), engine=engine, workers=settings.max_workers, report_path=report)
                records[label] = read_records(report)
                calls = Counter()
                for record in records[label]: calls.update(record.get("calls", {}))
                manifest["variants"][label] = {"statuses": counts, "calls": dict(calls), "embedding_cache": "isolated cold cache per variant", "index_fingerprint": getattr(getattr(backend, "index", None), "fingerprint", None)}
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
    parser.add_argument("--output", type=Path, default=Path("reports") / datetime.now(timezone(timedelta(hours=3))).strftime("run-%Y%m%d-%H%M%S"))
    parser.add_argument("--verify", choices=["0", "1", "both"], default="both")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        asyncio.run(evaluate(args.data_dir, args.output, args.verify))
    except ModelAPIError:
        print("Доступ к API не настроен. Официальный прокси выдаётся только при запуске на платформе; локально нужны собственные разрешённые реквизиты.", file=sys.stderr)
        return 2
    except (OSError, ValueError):
        print("Проверьте входные файлы и выберите новый каталог для отчёта.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
