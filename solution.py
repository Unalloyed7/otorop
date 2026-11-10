"""Конкурсная точка входа. DATA_DIR/questions.csv -> ./submission.csv."""
from __future__ import annotations

import asyncio
import csv
import logging
import importlib
import time
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

from assistant import LegalAssistant
from diagnostics import append_report, capture_run
from llm_client import LLMClient, Settings, env_value
from schemas import Answer
from search_adapter import build_search


class SubmissionError(ValueError):
    pass


def load_questions(path: Path) -> list[tuple[str, str]]:
    csv.field_size_limit(1_000_000)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, strict=True)
        if reader.fieldnames != ["id", "question"]:
            raise SubmissionError("Нужен заголовок id,question")
        rows = []
        seen = set()
        for row in reader:
            qid, question = row.get("id"), row.get("question")
            if None in row or not qid or not qid.strip() or not question or not question.strip():
                raise SubmissionError("Пустой ID/вопрос или неверное число колонок")
            if qid in seen:
                raise SubmissionError("Повторяющийся ID")
            seen.add(qid)
            rows.append((qid, question))
    if not rows:
        raise SubmissionError("Нет вопросов")
    return rows


def render_answer(answer: Answer) -> str:
    # Номера [n] соответствуют индексам выводов; веб использует те же Citation.
    references = []
    for c in answer.citations:
        reference = f'[{c.claim_index + 1}] {c.document_title}, {c.section} ({c.edition}). «{c.quote}»'
        if reference not in references:
            references.append(reference)
    text = answer.text + ("\nОснования:\n" + "\n".join(references) if references else "")
    if not text.strip() or len(text) > 10000:
        # Не обрезаем условие/исключение вместе с концом длинного ответа.
        raise SubmissionError("Ответ превышает лимит CSV")
    return text


def write_submission(path: Path, rows: list[tuple[str, str]], answers: list[str]):
    if len(rows) != len(answers) or len({r[0] for r in rows}) != len(rows):
        raise SubmissionError("Неполный набор ID")
    if any(not isinstance(a, str) or not a.strip() or len(a) > 10000 for a in answers):
        raise SubmissionError("Пустой или слишком длинный ответ")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=".submission-", suffix=".csv", delete=False) as f:
            temporary = Path(f.name)
            writer = csv.writer(f, lineterminator="\n", quoting=csv.QUOTE_ALL)
            writer.writerow(["id", "answer"])
            writer.writerows((row[0], text) for row, text in zip(rows, answers))
        validate_submission(temporary, [r[0] for r in rows])
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def validate_submission(path: Path, expected_ids: list[str]):
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, strict=True)
        if reader.fieldnames != ["id", "answer"]:
            raise SubmissionError("Неверный заголовок сабмита")
        rows = list(reader)
    ids = [r.get("id") for r in rows]
    if ids != expected_ids or len(ids) != len(set(ids)):
        raise SubmissionError("ID сабмита не совпадают со входом")
    if any(None in r or not r.get("answer", "").strip() or len(r["answer"]) > 10000 for r in rows):
        raise SubmissionError("Неверный ответ в сабмите")


def _runtime_options():
    try:
        return importlib.import_module("submission_options")
    except ModuleNotFoundError as exc:
        if exc.name != "submission_options":
            raise
        return None


async def _process(rows, engine, workers, budget=1120, *, report_path=None):
    if not 1 <= workers <= 8 or not 0 < budget <= 1170:
        raise SubmissionError("Некорректный параллелизм или бюджет времени")
    results = [None] * len(rows)
    written = set()
    client = getattr(engine, "client", None)
    settings = getattr(client, "settings", None)
    secrets = (settings.api_key.get_secret_value(),) if settings else ()
    queue = asyncio.Queue()
    for i, (_, question) in enumerate(rows):
        queue.put_nowait((i, question))

    def record(i, result, text, run, elapsed):
        if report_path:
            append_report(report_path, {"id": rows[i][0], "question": rows[i][1], "verify": getattr(engine, "verify", None), "answer": result.model_dump(mode="json"), "csv_answer": text, "duration_ms": round(elapsed * 1000, 2), "calls": dict(Counter(c["endpoint"] for c in run["api_calls"])), **run}, secrets=secrets)
            written.add(i)
            invalid = sum(not output["valid"] for output in run["model_outputs"])
            print(f"diagnostics: case={i+1}, status={result.status}, api_calls={len(run['api_calls'])}, invalid_json={invalid}", flush=True)

    async def worker():
        while not queue.empty():
            i, question = queue.get_nowait()
            started = time.perf_counter()
            with capture_run() as run:
                try:
                    # ID не передаётся ни движку, ни модели.
                    result = await engine.answer(question, trace=run) if report_path else await engine.answer(question)
                    text = render_answer(result)
                except asyncio.CancelledError:
                    result = Answer(status="error", text="Техническая ошибка: исчерпан лимит времени обработки. Основания в базе не проверены.")
                    run["stages"].append({"stage": "error", "kind": "RunTimeout"})
                    results[i] = (result.status, result.text)
                    record(i, result, result.text, run, time.perf_counter() - started)
                    raise
                except Exception as exc:
                    result = Answer(status="error", text="Техническая ошибка обработки ответа. Повторите запрос позднее.")
                    text = result.text
                    run["stages"].append({"stage": "error", "kind": type(exc).__name__})
                results[i] = (result.status, text)
                record(i, result, text, run, time.perf_counter() - started)
                queue.task_done()
    try:
        async with asyncio.timeout(budget):
            await asyncio.gather(*(worker() for _ in range(min(workers, len(rows)))))
    except TimeoutError:
        logging.getLogger(__name__).warning("submission: исчерпан бюджет времени")
    for i, result in enumerate(results):
        if result is None:
            results[i] = ("error", "Техническая ошибка: исчерпан лимит времени обработки. Основания в базе не проверены.")
        if report_path and i not in written:
            error = Answer(status=results[i][0], text=results[i][1])
            record(i, error, error.text, {"api_calls": [], "model_outputs": [], "stages": [{"stage": "error", "kind": "NotProcessedBeforeDeadline"}]}, 0)
    return results


async def run_submission(data_dir: Path, output: Path = Path("submission.csv"), *, engine=None, workers: int = 2, report_path: Path | None = None) -> dict[str, int]:
    rows = load_questions(data_dir / "questions.csv")
    budget = float(env_value("RUN_TIMEOUT", default="1120"))
    options = _runtime_options()
    report = report_path or env_value("RUN_REPORT") or getattr(options, "RUN_REPORT", None)
    report = Path(report) if report else None
    if report:
        if report.suffix != ".jsonl" or report.resolve() == output.resolve():
            raise SubmissionError("Отчёт должен быть отдельным файлом .jsonl")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("", encoding="utf-8")
    if engine is not None:
        results = await _process(rows, engine, workers, budget, report_path=report)
    else:
        client = None
        try:
            settings = Settings.from_env()
            override = getattr(options, "VERIFY_ANSWER", None)
            if type(override) is bool:
                settings = settings.model_copy(update={"verify": override})
            # Регламент гарантирует запись в рабочий каталог и /tmp, не в HOME.
            if not env_value("OTOROP_EMBEDDING_CACHE_DIR"):
                os.environ["OTOROP_EMBEDDING_CACHE_DIR"] = str(Path(tempfile.gettempdir()) / "otorop" / "embeddings")
            client = LLMClient(settings)
            search = await build_search(client)
            results = await _process(rows, LegalAssistant(search, client), settings.max_workers, budget, report_path=report)
        except Exception as exc:
            logging.getLogger(__name__).error("submission: ошибка конфигурации API или индекса")
            results = [("error", "Техническая ошибка настройки API или поискового индекса. Основания в базе не проверены.") for _ in rows]
            if report:
                report.write_text("", encoding="utf-8")
                for (qid, question), (_, text) in zip(rows, results):
                    append_report(report, {"id": qid, "question": question, "answer": Answer(status="error", text=text).model_dump(), "calls": {}, "stages": [{"stage": "error", "kind": type(exc).__name__, "message": "configuration/index"}]}, secrets=(settings.api_key.get_secret_value(),) if client else ())
        finally:
            if client:
                await client.aclose()
    write_submission(output, rows, [text for _, text in results])
    return dict(Counter(status for status, _ in results))


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    # HTTPX на INFO печатает URL; наш запуск не включает подробные логи клиентов.
    try:
        counts = asyncio.run(run_submission(Path(env_value("DATA_DIR", default="data"))))
    except (OSError, ValueError, csv.Error):
        print("Ошибка входного CSV или записи submission.csv", file=sys.stderr)
        return 1
    print("Создан submission.csv; статусы: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
