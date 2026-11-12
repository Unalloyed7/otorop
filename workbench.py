"""Устойчивые задания и локальная диагностика рабочего места юриста."""
from __future__ import annotations

import asyncio
from collections import Counter
from datetime import date
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import secrets
import sqlite3
import threading
import time

from fastapi import HTTPException, Request, Response
from pydantic import Field, model_validator

from diagnostics import _redact, capture_run, runtime_configuration
from schemas import Answer, Message, Model, ScenarioDates

TERMINAL = {"ready", "error", "cancelled"}


class JobRequest(Model):
    idempotency_key: str = Field(min_length=8, max_length=120, pattern=r"^[A-Za-z0-9_-]+$")
    messages: list[Message] = Field(min_length=1, max_length=40)
    as_of: date | None = None
    scenario: ScenarioDates | None = None

    @model_validator(mode="after")
    def valid_history(self):
        from assistant import parse_messages
        parse_messages(self.messages)
        return self


class ImportRun(Model):
    name: str = Field(min_length=1, max_length=200)
    records: list[dict] = Field(min_length=1, max_length=1000)
    configuration: dict = Field(default_factory=dict)


class Rating(Model):
    correctness: str = Field(default="unknown", pattern=r"^(correct|partial|wrong|unknown)$")
    unnecessary_refusal: bool | None = None
    note: str = Field(default="", max_length=2000)


class OfficialScore(Model):
    archive: str = Field(min_length=1, max_length=200)
    score: float = Field(allow_inf_nan=False)
    duration_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    note: str = Field(default="", max_length=2000)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


class Store:
    def __init__(self, path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        if str(path) != ":memory:":
            Path(path).chmod(0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, owner TEXT NOT NULL, request_key TEXT NOT NULL,
          fingerprint TEXT NOT NULL, payload TEXT NOT NULL, state TEXT NOT NULL, events TEXT NOT NULL,
          answer TEXT, report TEXT, created REAL NOT NULL, updated REAL NOT NULL, UNIQUE(owner, request_key));
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL,
          records TEXT NOT NULL, configuration TEXT NOT NULL, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS ratings(owner TEXT NOT NULL, run_id TEXT NOT NULL, case_id TEXT NOT NULL,
          value TEXT NOT NULL, PRIMARY KEY(owner,run_id,case_id));
        CREATE TABLE IF NOT EXISTS scores(id TEXT PRIMARY KEY, owner TEXT NOT NULL, value TEXT NOT NULL, created REAL NOT NULL);
        """)

    def rows(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def write(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).rowcount

    def create_job(self, owner, body, capacity):
        payload = body.model_dump(mode="json")
        fingerprint = hashlib.sha256(encoded(payload).encode()).hexdigest()
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                old = self.rows("SELECT * FROM jobs WHERE owner=? AND request_key=?", (owner, body.idempotency_key))
                if old:
                    if old[0]["fingerprint"] != fingerprint:
                        raise HTTPException(409, "Ключ задания уже использован для другого запроса.")
                    return old[0], False
                count = self.rows("SELECT count(*) AS n FROM jobs WHERE state NOT IN ('ready','error','cancelled')")[0]["n"]
                if count >= capacity:
                    raise HTTPException(429, "Очередь заполнена. Повторите позже.")
                job_id, now = secrets.token_hex(16), time.time()
                self.write("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?)", (job_id, owner, body.idempotency_key, fingerprint, encoded(payload), "queued", "[]", None, None, now, now))
                return self.rows("SELECT * FROM jobs WHERE id=?", (job_id,))[0], True
            finally:
                self.db.execute("COMMIT")

    def owned(self, job_id, owner):
        rows = self.rows("SELECT * FROM jobs WHERE id=? AND owner=?", (job_id, owner))
        if not rows:
            raise HTTPException(404, "Задание не найдено.")
        return rows[0]

    def event(self, job_id, event):
        with self.lock:
            rows = self.rows("SELECT state,events FROM jobs WHERE id=?", (job_id,))
            if not rows or rows[0]["state"] in TERMINAL:
                return
            events = json.loads(rows[0]["events"])
            events.append({**event, "time": time.time()})
            state = rows[0]["state"]
            if event["stage"] in {"checking", "generation", "repairing", "rejected_claim", "attempt_result"}:
                state = "checking"
            elif event["stage"] in {"searching", "search", "context", "intent"}:
                state = "searching"
            self.write("UPDATE jobs SET state=?,events=?,updated=? WHERE id=?", (state, encoded(events[-100:]), time.time(), job_id))

    def finish(self, job_id, answer, report):
        self.write("UPDATE jobs SET state=?,answer=?,report=?,updated=? WHERE id=? AND state!='cancelled'",
                   ("error" if answer.status == "error" else "ready", encoded(answer.model_dump(mode="json")), encoded(report), time.time(), job_id))

    @staticmethod
    def view(row):
        return {"job_id": row["id"], "state": row["state"], "events": json.loads(row["events"]),
                "answer": json.loads(row["answer"]) if row["answer"] and row["state"] in {"ready", "error"} else None,
                "created": row["created"], "updated": row["updated"]}


class JobManager:
    def __init__(self, store, engine, *, workers=2, capacity=4):
        self.store, self.engine, self.capacity = store, engine, capacity
        self.queue, self.active, self.tasks = asyncio.Queue(), {}, []
        self.workers, self.stopping = workers, False

    async def start(self):
        # Одно ASGI-приложение на БД. Прерванное выполнение не запускается повторно:
        # вызов модели мог уже завершиться до перезапуска процесса.
        interrupted = Answer(status="error", text="Выполнение прервано перезапуском сервера. Повторите запрос новым заданием.")
        self.store.write("UPDATE jobs SET state='error',answer=?,updated=? WHERE state IN ('searching','checking')", (encoded(interrupted.model_dump(mode="json")), time.time()))
        self.tasks = [asyncio.create_task(self.worker()) for _ in range(self.workers)]
        for row in self.store.rows("SELECT id FROM jobs WHERE state='queued' ORDER BY created"):
            self.queue.put_nowait(row["id"])

    async def close(self):
        self.stopping = True
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def worker(self):
        while True:
            job_id = await self.queue.get()
            if not self.store.write("UPDATE jobs SET state='searching',updated=? WHERE id=? AND state='queued'", (time.time(), job_id)):
                self.queue.task_done()
                continue
            task = asyncio.create_task(self.execute(job_id))
            self.active[job_id] = task
            try:
                await task
            except asyncio.CancelledError:
                if self.stopping:
                    raise
            finally:
                self.active.pop(job_id, None)
                self.queue.task_done()

    async def execute(self, job_id):
        row = self.store.rows("SELECT * FROM jobs WHERE id=?", (job_id,))[0]
        body = JobRequest.model_validate_json(row["payload"])
        started = time.perf_counter()
        with capture_run() as report:
            try:
                if self.engine is None:
                    answer = Answer(status="error", text="Модель или база документов пока не подключены.")
                else:
                    parameters = inspect.signature(self.engine.answer).parameters
                    kwargs = {"as_of": body.as_of}
                    if "scenario" in parameters:
                        kwargs["scenario"] = body.scenario
                    if "trace" in parameters:
                        kwargs["trace"] = report
                    if "on_progress" in parameters:
                        kwargs["on_progress"] = lambda event: self.store.event(job_id, event)
                    answer = await self.engine.answer(body.messages, **kwargs)
            except asyncio.CancelledError:
                answer = Answer(status="error", text="Выполнение прервано. Непроверенный ответ не опубликован.")
                self.store.finish(job_id, answer, {})
                raise
            except Exception:
                answer = Answer(status="error", text="Техническая ошибка задания. Непроверенный ответ не опубликован.")
            report = {**report, "id": job_id, "question": body.messages[-1].content, "answer": answer.model_dump(mode="json"),
                      "configuration": runtime_configuration(getattr(self.engine, "client", None)),
                      "calls": dict(Counter(c["endpoint"] for c in report["api_calls"])),
                      "duration_ms": round((time.perf_counter()-started)*1000, 2)}
            settings = getattr(getattr(self.engine, "client", None), "settings", None)
            secret = settings.api_key.get_secret_value() if settings else ""
            report = _redact(report, (secret,))
            answer = Answer.model_validate(_redact(answer.model_dump(), (secret,)))
            self.store.finish(job_id, answer, report)

    def cancel(self, job_id):
        self.store.write("UPDATE jobs SET state='cancelled',answer=NULL,updated=? WHERE id=? AND state NOT IN ('ready','error','cancelled')", (time.time(), job_id))
        if job_id in self.active:
            self.active[job_id].cancel()


def owner(request: Request, response: Response):
    if request.method in {"POST", "PATCH", "DELETE"}:
        from urllib.parse import urlparse
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            raise HTTPException(403, "Запрос должен поступать из этого приложения.")
    token = request.cookies.get("psb_owner", "")
    import re
    if not re.fullmatch(r"[a-f0-9]{64}", token):
        token = secrets.token_hex(32)
        response.set_cookie("psb_owner", token, httponly=True, samesite="strict", secure=request.url.scheme == "https", max_age=604800)
    return hashlib.sha256(token.encode()).hexdigest()


def run_view(store, row):
    records = json.loads(row["records"])
    ratings = {r["case_id"]: json.loads(r["value"]) for r in store.rows("SELECT case_id,value FROM ratings WHERE owner=? AND run_id=?", (row["owner"], row["id"]))}
    return {"run_id": row["id"], "name": row["name"], "configuration": json.loads(row["configuration"]), "created": row["created"], "records": [{**r, "rating": ratings.get(str(r["id"]), Rating().model_dump())} for r in records]}


def normalize_records(records):
    result, seen = [], set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("stages", []), list) or not isinstance(record.get("model_outputs", []), list):
            raise HTTPException(422, "Неизвестная структура диагностического отчёта.")
        duration = record.get("duration_ms")
        calls = record.get("calls", {})
        if duration is not None and (type(duration) not in {float, int} or not math.isfinite(duration) or duration < 0):
            raise HTTPException(422, "Некорректная длительность в отчёте.")
        if not isinstance(calls, dict) or any(type(v) is not int or v < 0 for v in calls.values()):
            raise HTTPException(422, "Некорректные счётчики вызовов.")
        if any(isinstance(s, dict) and s.get("stage") == "search" and not isinstance(s.get("evidence", []), list) for s in record.get("stages", [])):
            raise HTTPException(422, "Некорректная выдача поиска в отчёте.")
        case_id = str(record.get("id", ""))
        if not case_id or case_id in seen:
            raise HTTPException(422, "В отчёте нужны непустые уникальные ID.")
        seen.add(case_id)
        try:
            answer = Answer.model_validate(record["answer"])
        except Exception:
            raise HTTPException(422, "Отчёт содержит ответ неизвестного формата.") from None
        # Принимаем данные, а не пути на сервере. Настройки ключей/адресов не импортируем.
        result.append({"id": case_id, "question": str(record.get("question", ""))[:60000], "answer": answer.model_dump(mode="json"),
            "duration_ms": record.get("duration_ms"), "calls": record.get("calls", {}),
            "rejections": [{"index": s.get("index"), "reason": str(s.get("reason", ""))[:1500]} for s in record.get("stages", []) if isinstance(s, dict) and s.get("stage") == "rejected_claim"],
            "errors": [{"kind": s.get("kind"), "message": s.get("message")} for s in record.get("stages", []) if isinstance(s, dict) and s.get("stage") == "error"],
            "format_errors": sum(not x.get("valid", True) for x in record.get("model_outputs", []) if isinstance(x, dict)),
            "format_details": [{"schema": x.get("schema"), "errors": x.get("errors", []), "output": str(x.get("output", ""))[:4000]} for x in record.get("model_outputs", []) if isinstance(x, dict) and not x.get("valid", True)],
            "sources": [{**{k: e.get(k) for k in ("document_id", "chunk_id", "document_title", "section", "text")}, "edition": (e.get("applicability") or {}).get("edition", e.get("edition"))} for s in record.get("stages", []) if isinstance(s, dict) and s.get("stage") == "search" for e in s.get("evidence", []) if isinstance(e, dict)]})
    return result


def install_routes(app):
    def store():
        return app.state.job_manager.store

    @app.post("/api/jobs")
    async def create_job(body: JobRequest, request: Request, response: Response):
        from assistant import parse_messages
        parse_messages(body.messages)
        identity = owner(request, response)
        row, created = store().create_job(identity, body, app.state.job_manager.capacity)
        if created:
            app.state.job_manager.queue.put_nowait(row["id"])
            response.status_code = 202
        return Store.view(row)

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str, request: Request, response: Response):
        return Store.view(store().owned(job_id, owner(request, response)))

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, request: Request, response: Response):
        identity = owner(request, response)
        store().owned(job_id, identity)
        app.state.job_manager.cancel(job_id)
        return Store.view(store().owned(job_id, identity))

    @app.post("/api/jobs/{job_id}/save-run")
    async def save_job_run(job_id: str, request: Request, response: Response):
        identity = owner(request, response)
        row = store().owned(job_id, identity)
        if row["state"] not in {"ready", "error"} or not row["report"]:
            raise HTTPException(409, "Задание ещё не завершено.")
        report = json.loads(row["report"])
        existing = store().rows("SELECT * FROM runs WHERE id=? AND owner=?", (job_id, identity))
        if not existing:
            store().write("INSERT INTO runs VALUES(?,?,?,?,?,?)", (job_id, identity, "Задание " + job_id[:8], encoded(normalize_records([report])), encoded(report.get("configuration", {})), time.time()))
        return {"run_id": job_id}

    @app.post("/api/runs/import")
    async def import_run(body: ImportRun, request: Request, response: Response):
        identity = owner(request, response)
        if len(store().rows("SELECT id FROM runs WHERE owner=?", (identity,))) >= 100:
            raise HTTPException(429, "Достигнут лимит сохранённых запусков.")
        config = body.configuration or body.records[0].get("configuration", {})
        if not isinstance(config, dict):
            raise HTTPException(422, "Настройки запуска должны быть объектом.")
        safe_config = {k: config[k] for k in ("code_revision", "engine_sha256", "model", "embedding_model", "reranker_model", "temperature", "max_tokens", "verify", "max_workers", "max_model_calls", "max_search_calls", "question_timeout", "search_retries", "timeout", "retries", "top_k", "reranker_top_n") if k in config and type(config[k]) in {str, int, float, bool, type(None)}}
        run_id = secrets.token_hex(16)
        store().write("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, identity, body.name, encoded(normalize_records(body.records)), encoded(safe_config), time.time()))
        return {"run_id": run_id}

    @app.get("/api/runs")
    async def runs(request: Request, response: Response):
        identity = owner(request, response)
        return {"runs": [{k: row[k] for k in ("id", "name", "created")} for row in store().rows("SELECT id,name,created FROM runs WHERE owner=? ORDER BY created DESC", (identity,))]}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str, request: Request, response: Response):
        rows = store().rows("SELECT * FROM runs WHERE id=? AND owner=?", (run_id, owner(request, response)))
        if not rows:
            raise HTTPException(404, "Запуск не найден.")
        return run_view(store(), rows[0])

    @app.patch("/api/runs/{run_id}/cases/{case_id:path}/rating")
    async def rate(run_id: str, case_id: str, body: Rating, request: Request, response: Response):
        identity = owner(request, response)
        rows = store().rows("SELECT * FROM runs WHERE id=? AND owner=?", (run_id, identity))
        if not rows or case_id not in {str(r["id"]) for r in json.loads(rows[0]["records"])}:
            raise HTTPException(404, "Сценарий не найден.")
        store().write("INSERT OR REPLACE INTO ratings VALUES(?,?,?,?)", (identity, run_id, case_id, encoded(body.model_dump())))
        return body

    @app.get("/api/official-scores")
    async def scores(request: Request, response: Response):
        identity = owner(request, response)
        return {"scores": [json.loads(r["value"]) for r in store().rows("SELECT value FROM scores WHERE owner=? ORDER BY created", (identity,))]}

    @app.post("/api/official-scores")
    async def add_score(body: OfficialScore, request: Request, response: Response):
        identity = owner(request, response)
        store().write("INSERT INTO scores VALUES(?,?,?,?)", (secrets.token_hex(16), identity, encoded(body.model_dump()), time.time()))
        return body
