"""FastAPI-оболочка общего движка. Запуск: uvicorn web_app:app --port 8000."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import Field, model_validator

from assistant import LegalAssistant, parse_messages
from llm_client import LLMClient, ModelAPIError, Settings
from schemas import Answer, Message, Model
from search_adapter import build_search

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_BODY_BYTES = 300_000


class ChatRequest(Model):
    messages: list[Message] = Field(min_length=1, max_length=40)
    as_of: date | None = None

    @model_validator(mode="after")
    def valid_history(self):
        parse_messages(self.messages)
        return self


class BodyLimit:
    """Ограничивает фактическое тело до JSON-разбора, включая запросы без длины."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        data = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            data.extend(event.get("body", b""))
            if len(data) > MAX_BODY_BYTES:
                return await JSONResponse({"detail": "Слишком большой запрос."}, status_code=413)(scope, receive, send)
            if not event.get("more_body", False):
                break
        consumed = False
        async def replay():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": bytes(data), "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


def _documents(backend) -> list[dict]:
    index = getattr(backend, "index", backend)
    documents = getattr(index, "documents", None)
    if isinstance(documents, dict):
        return list(documents.values())
    # Контракт просмотра — get_document/get_chunk и индекс документов №1.
    raise ValueError("Backend не предоставляет каталог документов")


def _document_summary(document: dict) -> dict:
    return {key: document.get(key) for key in (
        "document_id", "title", "document_type", "edition", "signed_date", "valid_from", "valid_to", "source_file", "contract_number", "counterparty",
    )} | {"char_count": len(document.get("text", ""))}


def create_app(*, engine: Any = None, backend: Any = None) -> FastAPI:
    """Явная инъекция компонентов — для тестов; рабочий запуск использует общий factory."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = None
        app.state.backend = backend
        app.state.engine = engine
        app.state.model_configured = engine is not None
        app.state.issue = None
        app.state.pending = 0
        app.state.capacity = 4
        try:
            if engine is None:
                try:
                    settings = Settings.from_env()
                    client = LLMClient(settings)
                    app.state.model_configured = True
                    app.state.capacity = settings.max_workers * 2
                except ModelAPIError:
                    app.state.issue = "model_configuration"
                if app.state.backend is None:
                    try:
                        app.state.backend = await build_search(client)
                    except Exception:
                        app.state.issue = "knowledge_base"
                        log.warning("web: не удалось открыть базу документов")
                if app.state.backend is not None and client is not None:
                    app.state.engine = LegalAssistant(app.state.backend, client)
            elif app.state.backend is None:
                app.state.backend = getattr(engine, "search", None)
            yield
        finally:
            if client is not None:
                await client.aclose()

    app = FastAPI(title="ПСБ · Юридический ассистент", version="0.2.0", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url="/api/openapi.json")
    app.add_middleware(BodyLimit)

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Не возвращаем содержимое пользовательского ввода в ошибке.
        return JSONResponse({"detail": "Проверьте формат запроса и историю диалога."}, status_code=422)

    @app.get("/healthz", include_in_schema=False)
    async def health():
        return {"status": "ok"}

    @app.get("/api/status")
    async def status():
        available = app.state.backend is not None
        try:
            count = len(_documents(app.state.backend)) if available else 0
        except ValueError:
            count = 0
        return {
            "ready": available and app.state.engine is not None,
            "documents_available": available,
            "document_count": count,
            "model_configured": app.state.model_configured,
            "issue": app.state.issue,
        }

    @app.get("/readyz", include_in_schema=False)
    async def ready():
        data = await status()
        return JSONResponse(data, status_code=200 if data["ready"] else 503)

    @app.post("/api/chat", response_model=Answer)
    async def chat(body: ChatRequest):
        if app.state.engine is None:
            return Answer(status="error", text="Сервис ответов пока не подключён. Документы можно просматривать; попробуйте задать вопрос после подключения модели и базы.")
        if app.state.pending >= app.state.capacity:
            return JSONResponse({"detail": "Сервис занят. Повторите запрос немного позже."}, status_code=429, headers={"Retry-After": "3"})
        app.state.pending += 1
        try:
            return await app.state.engine.answer(body.messages, as_of=body.as_of)
        finally:
            app.state.pending -= 1

    def require_backend():
        if app.state.backend is None:
            raise HTTPException(503, "База документов пока недоступна.")
        return app.state.backend

    def document_by_id(document_id: str):
        source = require_backend()
        if not callable(getattr(source, "get_document", None)):
            raise HTTPException(503, "Просмотр документов не поддерживается этим поиском.")
        try:
            return source.get_document(document_id)
        except (KeyError, ValueError):
            raise HTTPException(404, "Документ не найден.") from None

    @app.get("/api/documents")
    async def documents():
        try:
            rows = _documents(require_backend())
        except ValueError:
            raise HTTPException(503, "Просмотр документов не поддерживается этим поиском.") from None
        return {"documents": sorted((_document_summary(d) for d in rows), key=lambda d: (d.get("document_type") or "", d.get("title") or ""))}

    @app.get("/api/documents/{document_id}")
    async def get_document(document_id: str):
        doc = document_by_id(document_id)
        return _document_summary(doc) | {"text": doc["text"]}

    @app.get("/api/documents/{document_id}/download")
    async def download_document(document_id: str):
        doc = document_by_id(document_id)
        filename = Path(doc["source_file"]).stem + ".txt"
        return Response(doc["text"], media_type="text/plain; charset=utf-8", headers={"Content-Disposition": "attachment; filename*=UTF-8''" + quote(filename, safe="")})

    @app.get("/api/chunks/{chunk_id}")
    async def get_chunk(chunk_id: str, window: int = Query(default=1, ge=0, le=3)):
        source = require_backend()
        if not callable(getattr(source, "get_chunk", None)) or not callable(getattr(source, "get_neighbors", None)):
            raise HTTPException(503, "Просмотр фрагментов не поддерживается этим поиском.")
        try:
            chunk = source.get_chunk(chunk_id)
            neighbors = source.get_neighbors(chunk_id, window=window)
        except (KeyError, ValueError):
            raise HTTPException(404, "Фрагмент не найден.") from None
        def view(item):
            return {key: item.get(key) for key in ("chunk_id", "document_id", "section_reference", "text", "start_char", "end_char")}
        return {"chunk": view(chunk), "neighbors": [view(item) for item in neighbors]}

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")
    return app


app = create_app()
