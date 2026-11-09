from __future__ import annotations

from collections import deque

import pytest
from pydantic import SecretStr

from llm_client import ModelAPIError, Settings
from schemas import Applicability, Coordinates, Evidence, Intent, SearchResult


@pytest.fixture
def settings():
    return Settings(base_url="https://proxy.test/v1", api_key=SecretStr("test-only-key"), model="organizer-generation-id", embedding_model="organizer-embedding-id", reranker_model="organizer-reranker-id", retries=0)


@pytest.fixture
def evidence():
    text = "4.2. Покупатель оплачивает товар в течение 30 календарных дней после подписания УПД.\n7.2. Неустойка составляет 0,1% в день, но не более 10% от неоплаченной суммы."
    return Evidence(document_id="contract-1", chunk_id="contract-1:0", document_title="Тестовый договор № Т-1/2026", section="пп. 4.2, 7.2", text=text, coordinates=Coordinates(source_path="contract.txt", char_start=0, char_end=len(text)), applicability=Applicability(edition="Первоначальная редакция", resolved=True), contract_number="Т-1/2026", counterparty="Тестовый поставщик")


def intent(**kwargs):
    return Intent(action="search", query="Срок оплаты по договору Т-1/2026", parts=["Срок оплаты"], **kwargs)


def draft(e, text="Товар оплачивается в течение 30 календарных дней после подписания УПД.", **kwargs):
    return {"status": "answer", "claims": [{"text": text, "supports": [{"chunk_id": e.chunk_id, "document_id": e.document_id, "quote": e.text.split('\n')[0]}], "addresses": [0]}], "missing_info": [], **kwargs}


class FakeClient:
    def __init__(self, settings, *responses):
        self.settings = settings
        self.responses = deque(responses)
        self.requests = []

    async def structured(self, instruction, data, schema):
        self.requests.append((instruction, data, schema))
        if not self.responses:
            raise AssertionError("Лишний вызов модели")
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return schema.model_validate(response)


class FakeSearch:
    def __init__(self, evidence=(), **kwargs):
        self.result = SearchResult(evidence=list(evidence), **kwargs)
        self.requests = []

    async def search(self, query, filters):
        self.requests.append((query, filters))
        return self.result
