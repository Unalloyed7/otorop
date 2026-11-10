"""Диагностика одного запроса через ContextVar; ключи/адреса API не записываются."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
from typing import Any

_active: ContextVar[dict | None] = ContextVar("legal_run_diagnostics", default=None)


def active_run() -> dict | None:
    return _active.get()


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
