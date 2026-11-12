"""Диагностика одного запроса через ContextVar; ключи/адреса API не записываются."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import hashlib
import os
from pathlib import Path
from typing import Any

_active: ContextVar[dict | None] = ContextVar("legal_run_diagnostics", default=None)


def active_run() -> dict | None:
    return _active.get()


def runtime_configuration(client=None):
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("schemas.py", "assistant.py", "llm_client.py", "retrieval.py", "documents.py", "structural_chunking.py", "knowledge_pipeline.py", "search_adapter.py", "temporal.py", "date_context.py", "verification.py", "execution.py", "solution.py"):
        path = root / name
        if path.is_file():
            digest.update(name.encode() + b"\0" + path.read_bytes())
    settings = getattr(client, "settings", None)
    result = {"code_revision": os.environ.get("CODE_VERSION"), "engine_sha256": digest.hexdigest()}
    if settings:
        result.update({key: getattr(settings, key, None) for key in ("model", "embedding_model", "reranker_model", "temperature", "max_tokens", "verify", "max_workers", "max_model_calls", "max_search_calls", "question_timeout", "search_retries", "timeout", "retries", "top_k", "reranker_top_n")})
    return result


@contextmanager
def capture_run():
    run = {"api_calls": [], "chat_responses": [], "model_outputs": [], "stages": []}
    token = _active.set(run)
    try:
        yield run
    finally:
        _active.reset(token)


def _redact(value: Any, secrets: tuple[str, ...]):
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {_redact(str(k), secrets): _redact(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v, secrets) for v in value]
    return value


def append_report(path: Path, record: dict, *, secrets: tuple[str, ...] = ()):
    # Один синхронный write: записи параллельных asyncio-воркеров не перемешиваются.
    with path.open("a", encoding="utf-8", newline="") as f:
        f.write(json.dumps(_redact(record, secrets), ensure_ascii=False, allow_nan=False) + "\n")
