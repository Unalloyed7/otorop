import asyncio
import json

import httpx
import pytest

from llm_client import LLMClient, ModelAPIError, ModelProtocolError, Settings
from schemas import Intent


def transport(data, status=200):
    return httpx.MockTransport(lambda request: httpx.Response(status, json=data))


async def test_all_three_endpoints_and_models(settings):
    requests = []
    def handle(request):
        payload = json.loads(request.content)
        requests.append((request, payload))
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 0]}]})
        if request.url.path.endswith("/rerank"):
            return httpx.Response(200, json={"results": [{"index": 1}, {"index": 0}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Ответ"}, "finish_reason": "stop"}]})
    async with LLMClient(settings, transport=httpx.MockTransport(handle)) as client:
        assert await client.chat([{"role": "user", "content": "Вопрос"}]) == "Ответ"
        assert await client.embeddings(["первый", "второй"]) == [[1, 0], [0, 1]]
        assert await client.rerank("query", ["a", "b"], 2) == [1, 0]
        assert sum(client.calls.values()) == 3
    assert [p["model"] for _, p in requests] == [settings.model, settings.embedding_model, settings.reranker_model]
    assert all(r.headers["authorization"] == "Bearer test-only-key" for r, _ in requests)
    assert "response_format" not in requests[0][1]


@pytest.mark.parametrize("status,retry", [(429, True), (503, True), (401, False), (400, False)])
async def test_bounded_retries_and_safe_diagnostics(settings, monkeypatch, status, retry):
    monkeypatch.setattr("llm_client.asyncio.sleep", lambda _: _immediate())
    settings = settings.model_copy(update={"retries": 2})
    calls = 0
    def handle(request):
        nonlocal calls
        calls += 1
        return httpx.Response(status, text="test-only-key private prompt")
    async with LLMClient(settings, transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ModelAPIError) as error:
            await client.chat([])
    assert calls == (3 if retry else 1)
    assert "test-only-key" not in str(error.value) and "private" not in str(error.value)
    assert str(status) in str(error.value)


async def _immediate(): pass


async def test_timeout_retry_recovers(settings, monkeypatch):
    monkeypatch.setattr("llm_client.asyncio.sleep", lambda _: _immediate())
    calls = 0
    def handle(request):
        nonlocal calls
        calls += 1
        if calls == 1: raise httpx.ReadTimeout("secret", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ОК"}}]})
    async with LLMClient(settings.model_copy(update={"retries": 1}), transport=httpx.MockTransport(handle)) as client:
        assert await client.chat([]) == "ОК"
        assert calls == 2


async def test_concurrency_limit_applies_to_all_endpoints(settings):
    active, maximum = 0, 0
    async def handle(request):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(.005)
        active -= 1
        if request.url.path.endswith("embeddings"):
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1]}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    async with LLMClient(settings, transport=httpx.MockTransport(handle)) as client:
        await asyncio.gather(*(client.chat([]) if i % 2 else client.embeddings(["a"]) for i in range(12)))
    assert maximum == settings.max_workers


@pytest.mark.parametrize("data", [[], {}, {"choices": []}, {"choices": [{"message": {"content": ""}}]}, {"choices": [{"message": {"content": ["bad"]}}]}, {"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]}])
async def test_malformed_chat_response(settings, data):
    async with LLMClient(settings, transport=transport(data)) as client:
        with pytest.raises(ModelProtocolError): await client.chat([])


@pytest.mark.parametrize("items", [[], [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [1]}], [{"index": 0, "embedding": [0]}, {"index": 1, "embedding": [1]}], [{"index": -1, "embedding": [1]}, {"index": 1, "embedding": [1]}], [{"index": 0, "embedding": [1, 2]}, {"index": 1, "embedding": [1]}], [{"index": 0, "embedding": [True]}, {"index": 1, "embedding": [1]}]])
async def test_bad_embedding_indices_vectors(settings, items):
    async with LLMClient(settings, transport=transport({"data": items})) as client:
        with pytest.raises(ModelProtocolError): await client.embeddings(["a", "b"])


@pytest.mark.parametrize("items", [[], [{"index": 4}], [{"index": 0}, {"index": 0}], [{"index": True}], [{}]])
async def test_invalid_rerank(settings, items):
    async with LLMClient(settings, transport=transport({"results": items})) as client:
        with pytest.raises(ModelProtocolError): await client.rerank("q", ["a", "b"], 2)


async def test_json_content_is_validated(settings):
    async with LLMClient(settings, transport=transport({"choices": [{"message": {"content": '{"action": "invented"}'}}]})) as client:
        with pytest.raises(ModelProtocolError): await client.structured("s", {}, Intent)


async def test_missing_models_are_not_guessed(settings):
    async with LLMClient(settings.model_copy(update={"embedding_model": None, "reranker_model": None}), transport=transport({})) as client:
        with pytest.raises(ModelAPIError): await client.embeddings(["a"])
        with pytest.raises(ModelAPIError): await client.rerank("q", ["a"])
        assert not client.calls


def test_env_aliases(monkeypatch):
    for name in ["LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "OPENAI_MODEL", "EMB_MODEL", "RERANKER_MODEL"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setenv("GEN_MODEL", "organizer-id")
    s = Settings.from_env()
    assert s.model == "organizer-id" and not s.reranker_model
    assert "test-only-key" not in repr(s)


def test_invalid_settings_do_not_leak_input(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://user:test-only-key@proxy.test")
    with pytest.raises(ModelAPIError) as e: Settings.from_env()
    assert "test-only-key" not in str(e.value)
