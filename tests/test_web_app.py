from __future__ import annotations

import asyncio
import zipfile

import httpx
import pytest
from fastapi.testclient import TestClient

from retrieval import create_search
from schemas import Answer
from web_app import create_app


@pytest.fixture
def source(tmp_path):
    with zipfile.ZipFile(tmp_path / "knowledge-base.zip", "w") as archive:
        archive.writestr("документы компании__Договор Т-1-2026.txt", 'ДОГОВОР ПОСТАВКИ № Т-1/2026\nООО «Тестовый поставщик»\n🙂 Текст до пункта.\n4.2. Оплата 30 календарных дней.\n<script>window.compromised=true</script>\n')
    return tmp_path


@pytest.fixture
def backend(source):
    return create_search(source)


class RecordingEngine:
    def __init__(self):
        self.calls = []

    async def answer(self, messages, as_of=None):
        self.calls.append((messages, as_of))
        return Answer(status="out_of_scope", text="Ответ общего движка")


def test_chat_passes_history_and_date_to_same_engine(backend):
    engine = RecordingEngine()
    app = create_app(engine=engine, backend=backend)
    with TestClient(app) as client:
        history = [{"role": "user", "content": "Договор Т-1/2026"}, {"role": "assistant", "content": "Что уточним?"}, {"role": "user", "content": "Срок оплаты?"}]
        response = client.post("/api/chat", json={"messages": history, "as_of": "2026-02-01"})
        assert response.status_code == 200
        assert response.json()["text"] == "Ответ общего движка"
        assert [m.model_dump() for m in engine.calls[0][0]] == history
        assert engine.calls[0][1].isoformat() == "2026-02-01"
        client.post("/api/chat", json={"messages": [{"role": "user", "content": "Другой вопрос"}]})
        assert len(engine.calls[1][0]) == 1


def test_documents_and_citation_coordinates_are_original_text(backend):
    app = create_app(engine=RecordingEngine(), backend=backend)
    with TestClient(app) as client:
        rows = client.get("/api/documents").json()["documents"]
        assert len(rows) == 1 and "text" not in rows[0]
        doc = client.get("/api/documents/" + rows[0]["document_id"]).json()
        clause = next(c for c in backend.index.chunks if c["section_reference"] == "4.2")
        response = client.get("/api/chunks/" + clause["chunk_id"])
        assert response.status_code == 200
        fragment = response.json()["chunk"]
        assert doc["text"][fragment["start_char"]:fragment["end_char"]] == fragment["text"]
        assert len(response.json()["neighbors"]) <= 3
        download = client.get("/api/documents/" + doc["document_id"] + "/download")
        assert download.text == doc["text"]
        assert download.headers["content-type"].startswith("text/plain")
        assert "filename*=UTF-8''" in download.headers["content-disposition"]
        assert download.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("path", ["/api/documents/unknown", "/api/chunks/unknown", "/.env", "/assets/%2e%2e/.env", "/api/documents/%2e%2e%2f.env"])
def test_cannot_read_arbitrary_files(backend, path):
    with TestClient(create_app(engine=RecordingEngine(), backend=backend)) as client:
        response = client.get(path)
        assert response.status_code == 404


@pytest.mark.parametrize("body", [
    {"messages": []},
    {"messages": [{"role": "system", "content": "PRIVATE_MARKER"}]},
    {"messages": [{"role": "assistant", "content": "PRIVATE_MARKER"}]},
    {"messages": [{"role": "user", "content": "PRIVATE_MARKER"}], "as_of": "bad-date"},
    {"messages": [{"role": "user", "content": "x" * 40000}, {"role": "user", "content": "x" * 30000}]},
])
def test_invalid_request_does_not_reflect_user_content(backend, body):
    with TestClient(create_app(engine=RecordingEngine(), backend=backend)) as client:
        response = client.post("/api/chat", json=body)
        assert response.status_code == 422
        assert "PRIVATE_MARKER" not in response.text


def test_oversized_body_rejected_before_json(backend):
    with TestClient(create_app(engine=RecordingEngine(), backend=backend)) as client:
        response = client.post("/api/chat", content=b"x" * 300001, headers={"Content-Type": "application/json"})
        assert response.status_code == 413


def test_no_api_config_allows_document_browsing(source, monkeypatch):
    for name in ["LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_API_KEY", "OPENAI_API_KEY", "LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL", "SEARCH_FACTORY"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATA_DIR", str(source))
    monkeypatch.setenv("WEB_STATE_DIR", str(source / "state"))
    with TestClient(create_app()) as client:
        status = client.get("/api/status").json()
        assert status["documents_available"] and status["document_count"] == 1
        assert not status["ready"] and not status["model_configured"]
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        assert client.get("/api/documents").status_code == 200
        answer = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Оплата?"}]}).json()
        assert answer["status"] == "error" and not answer["citations"]


def test_lifespan_closes_owned_client(backend, settings, monkeypatch):
    class OwnedClient:
        def __init__(self, value): self.settings = value; self.closed = False
        async def aclose(self): self.closed = True
    client = OwnedClient(settings)
    monkeypatch.setattr("web_app.Settings.from_env", lambda: settings)
    monkeypatch.setattr("web_app.LLMClient", lambda settings: client)
    app = create_app(backend=backend)
    with TestClient(app) as test:
        assert test.get("/readyz").status_code == 200
        assert not client.closed
    assert client.closed


def test_static_and_security_headers(backend):
    with TestClient(create_app(engine=RecordingEngine(), backend=backend)) as client:
        for path in ["/", "/assets/styles.css", "/assets/app.js", "/assets/favicon.svg"]:
            response = client.get(path)
            assert response.status_code == 200
            assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        assert "Юридический ассистент" in client.get("/").text
        assert client.get("/api/status").headers["cache-control"] == "no-store"
        assert client.get("/api/openapi.json").status_code == 200


async def test_queue_is_bounded(backend):
    release = asyncio.Event()
    started = asyncio.Event()
    class SlowEngine:
        calls = 0
        async def answer(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 4: started.set()
            await release.wait()
            return Answer(status="out_of_scope", text="Готово")
    app = create_app(engine=SlowEngine(), backend=backend)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            tasks = [asyncio.create_task(client.post("/api/chat", json={"messages": [{"role": "user", "content": "q"}]})) for _ in range(4)]
            await asyncio.wait_for(started.wait(), 2)
            blocked = await client.post("/api/chat", json={"messages": [{"role": "user", "content": "q"}]})
            assert blocked.status_code == 429
            release.set()
            assert all(r.status_code == 200 for r in await asyncio.gather(*tasks))
            assert app.state.pending == 0
