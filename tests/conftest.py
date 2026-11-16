from __future__ import annotations

from collections import deque

import pytest
from pydantic import SecretStr

from llm_client import ModelAPIError, Settings
from schemas import Applicability, Coordinates, Evidence, Intent, SearchResult, Verdict


def with_coverage(response, data):
    """Complete the default mock verdict; explicit coverage tests supply their own.

    This is fixture data, not a semantic evaluator. Live quality is tested separately.
    """
    response = dict(response)
    if "checks" not in response and "supported" in response:
        selected = response["supported"]
        count = len(data.get("claims", []))
        response["checks"] = [{"index": i, "status": "confirmed" if i in selected else "insufficient"}
                              for i in range(count)]
        seen = set()
        for i in selected:
            if type(i) is not int or i < 0 or i >= count or i in seen:
                response["checks"].append({"index": i, "status": "confirmed"})
            seen.add(i)
    if "checks" in response:
        response["checks"] = [{"relevance": "direct", "excluded_fragments": [], **c} for c in response["checks"]]
    if "answer_claim_indices" not in response:
        response["answer_claim_indices"] = [c["index"] for c in response.get("checks", [])
            if c["status"] in {"confirmed", "conditional"} and c["relevance"] != "unrequested" and not c["excluded_fragments"]]
    if "coverage" in response:
        return response
    selected = {c["index"] for c in response["checks"] if c["status"] in {"confirmed", "conditional"}} if response.get("checks") else set(response.get("supported", []))
    coverage = []
    for index, _ in enumerate(data.get("parts", [])):
        claims = [i for i, c in enumerate(data.get("claims", [])) if i in selected and index in c["addresses"]]
        coverage.append({"part_index": index, "status": "complete" if claims else "missing", "claim_indices": claims})
    return {**response, "coverage": coverage}


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
        if schema is Verdict and isinstance(response, dict):
            response = with_coverage(response, data)
        return schema.model_validate(response)


class FakeSearch:
    def __init__(self, evidence=(), **kwargs):
        self.result = SearchResult(evidence=list(evidence), **kwargs)
        self.requests = []

    async def search(self, query, filters):
        self.requests.append((query, filters))
        return self.result
