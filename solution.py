"""Конкурсная точка входа. DATA_DIR/questions.csv -> ./submission.csv."""
from __future__ import annotations

import asyncio
import csv
import logging
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

from assistant import LegalAssistant
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


async def _process(rows, engine, workers, budget=1120):
    if not 1 <= workers <= 8 or not 0 < budget <= 1170:
        raise SubmissionError("Некорректный параллелизм или бюджет времени")
    results = [None] * len(rows)
    queue = asyncio.Queue()
    for i, (_, question) in enumerate(rows):
        queue.put_nowait((i, question))

    async def worker():
        while not queue.empty():
            i, question = queue.get_nowait()
            # Только question; ID не передаётся ни движку, ни модели.
            try:
                result = await engine.answer(question)
                text = render_answer(result)
            except Exception:
                result = Answer(status="error", text="Техническая ошибка обработки ответа. Повторите запрос позднее.")
                text = result.text
            results[i] = (result.status, text)
            queue.task_done()
    try:
        async with asyncio.timeout(budget):
            await asyncio.gather(*(worker() for _ in range(min(workers, len(rows)))))
    except TimeoutError:
        logging.getLogger(__name__).warning("submission: исчерпан бюджет времени")
    for i, result in enumerate(results):
        if result is None:
            results[i] = ("error", "Техническая ошибка: исчерпан лимит времени обработки. Основания в базе не проверены.")
    return results


async def run_submission(data_dir: Path, output: Path = Path("submission.csv"), *, engine=None, workers: int = 2) -> dict[str, int]:
    rows = load_questions(data_dir / "questions.csv")
    budget = float(env_value("RUN_TIMEOUT", default="1120"))
    if engine is not None:
        results = await _process(rows, engine, workers, budget)
    else:
        client = None
        try:
            settings = Settings.from_env()
            client = LLMClient(settings)
            search = await build_search(client)
            results = await _process(rows, LegalAssistant(search, client), settings.max_workers, budget)
        except Exception:
            logging.getLogger(__name__).error("submission: ошибка конфигурации API или индекса")
            results = [("error", "Техническая ошибка настройки API или поискового индекса. Основания в базе не проверены.") for _ in rows]
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
