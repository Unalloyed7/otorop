from __future__ import annotations

from unittest.mock import AsyncMock
import zipfile

import pytest

from assistant import LegalAssistant
from conftest import FakeClient, draft, intent
from retrieval import SchemaSearchBackend, SearchIndex, create_search
from documents import load_documents
from schemas import SearchFilters


@pytest.fixture
def corpus(tmp_path):
    with zipfile.ZipFile(tmp_path / "knowledge-base.zip", "w") as archive:
        archive.writestr("Договор Т-3-2026.txt", "ДОГОВОР № Т-3/2026\n"
                         "3.2. Доставка на Старый склад.\n4.2. Оплата в течение 30 дней.\n")
        archive.writestr("Доп соглашение Т-3-2026.txt", "ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ № 1\n"
                         "к договору № Т-3/2026\n"
                         "Изложить пункт 3.2 договора в следующей редакции:\n"
                         "3.2. Доставка на Новый склад.\n"
                         "Новый адрес применяется к партиям, спецификации на которые подписаны после 1 октября 2026 г.\n")
    return tmp_path


async def test_engine_uses_bundle_and_sends_complete_amendment_to_generator(corpus, settings):
    backend = create_search(corpus)
    backend.search_bundle = AsyncMock(wraps=backend.search_bundle)
    filters = SearchFilters(contract_number="Т-3/2026", event_date="2026-10-02", date_kind="specification_signed")
    found = await backend.search("адрес доставки", filters)
    clause = next(e for e in found.evidence if "Новый склад" in e.text)
    plan = intent(filters=filters).model_copy(update={"query": "адрес доставки Т-3/2026", "parts": ["Адрес доставки", "Условие изменения адреса"]})
    response = draft(clause, text="Доставка на Новый склад.")
    response["claims"][0]["addresses"] = [0, 1]
    client = FakeClient(settings, plan, response, {"supported": [0]})
    trace = {"stages": []}
    result = await LegalAssistant(backend, client).answer("Куда доставлять и когда меняется адрес?", trace=trace)
    assert result.status == "answer"
    backend.search_bundle.assert_awaited_once()
    assert backend.search_bundle.call_args.args[0] == plan.parts
    evidence = client.requests[1][1]["evidence"]
    assert any("Старый склад" in e["text"] for e in evidence)
    assert any("Новый склад" in e["text"] for e in evidence)
    assert any("после 1 октября" in e["text"] for e in evidence)
    assert any(e["metadata"]["applicability_status"] == "applicable" for e in evidence)
    bundle_trace = next(s for s in trace["stages"] if s["stage"] == "search_bundle")
    assert {r["type"] for r in bundle_trace["relations"]} >= {"changes_provision", "defines_applicability"}
    assert not bundle_trace["truncated"]


async def test_bundle_retrieves_all_long_clause_parts_and_parent_intro(tmp_path):
    clause = "2.1. " + " ".join(f"Условие {i} регулирует передачу оригиналов." for i in range(24))
    source = ("ДОГОВОР № Т-9/2026\n2. Правила применяются только к согласованным заказам:\n"
              + clause + "\nЗа исключением партий с предоплатой.\n2.2. Иное положение.\n")
    (tmp_path / "Договор Т-9-2026.txt").write_text(source, encoding="utf-8")
    backend = SchemaSearchBackend(SearchIndex(load_documents(tmp_path), max_chunk_chars=300))
    bundle = await backend.search_bundle(["передача оригиналов"], SearchFilters(contract_number="Т-9/2026"),
                                         top_k=1, max_context_chars=4000, max_context_chunks=20)
    chunks = bundle["evidence"]
    assert any("только к согласованным заказам" in c["text"] for c in chunks)
    assert any("За исключением" in c["text"] for c in chunks)
    for chunk in chunks:
        assert set(chunk.get("provision_chunk_ids", [])) <= {c["chunk_id"] for c in chunks}
    assert not bundle["truncated"]
    for span in bundle["context_spans"]:
        assert source[span["start_char"]:span["end_char"]] == span["text"]


async def test_engine_reports_dependency_budget_limit_instead_of_dropping_condition(corpus, settings):
    settings = settings.model_copy(update={"top_k": 1, "search_retries": 0})
    backend = create_search(corpus, client=None)
    plan = intent(filters=SearchFilters(contract_number="Т-3/2026")).model_copy(update={"query": "новый адрес доставки", "parts": ["Адрес доставки"]})
    client = FakeClient(settings, plan)
    trace = {"stages": []}
    result = await LegalAssistant(backend, client).answer("Какой адрес?", trace=trace)
    assert result.status == "insufficient"
    assert any("бюджет контекста" in gap for gap in result.missing_info)
    assert len(client.requests) == 1
    assert next(s for s in trace["stages"] if s["stage"] == "search_bundle")["truncated"]


async def test_recovery_budget_includes_already_admitted_evidence(corpus):
    backend = create_search(corpus)
    first = await backend.search_bundle(["оплата"], SearchFilters(contract_number="Т-3/2026"), top_k=1)
    existing = tuple(e["chunk_id"] for e in first["evidence"])
    second = await backend.search_bundle(["новый адрес доставки"], SearchFilters(contract_number="Т-3/2026"), top_k=1,
                                          existing_chunk_ids=existing, max_context_chunks=len(existing))
    assert {e["chunk_id"] for e in second["evidence"]} == set(existing)
    assert second["truncated"]


async def test_bundle_budgets_overlap_as_actually_sent_to_generator(tmp_path):
    source = "ДОГОВОР № Т-9/2026\n2.1. " + " ".join(f"Условие {i} регулирует передачу оригиналов." for i in range(24))
    (tmp_path / "Договор Т-9-2026.txt").write_text(source, encoding="utf-8")
    backend = SchemaSearchBackend(SearchIndex(load_documents(tmp_path), max_chunk_chars=300))
    filters = SearchFilters(contract_number="Т-9/2026")
    complete = await backend.search_bundle(["передача оригиналов"], filters, top_k=1, max_context_chars=5000)
    assert complete["evidence_chars"] > complete["context_chars"]
    budget = complete["context_chars"]
    limited = await backend.search_bundle(["передача оригиналов"], filters, top_k=1, max_context_chars=budget)
    assert limited["truncated"]
    assert limited["evidence_chars"] <= budget
    assert not limited["evidence"]  # complete provision or an explicit gap


async def test_reference_to_long_clause_is_not_treated_as_ambiguous(tmp_path):
    source = ("1.1. Срок оплаты определяется пунктом 2.1 договора.\n2.1. "
              + " ".join(f"Условие {i} регулирует передачу оригиналов." for i in range(24)))
    (tmp_path / "Договор Т-9-2026.txt").write_text(source, encoding="utf-8")
    backend = SchemaSearchBackend(SearchIndex(load_documents(tmp_path), max_chunk_chars=300))
    referring = next(c for c in backend.index.chunks if c["section_reference"] == "1.1")
    assert any(r["type"] == "references_provision" for r in referring["provision_relations"])
    assert not any(g["kind"] == "referenced_provision_ambiguous" for g in referring["provision_gaps"])
    bundle = await backend.search_bundle(["срок оплаты"], SearchFilters(), top_k=1, max_context_chars=5000)
    ids = {c["chunk_id"] for c in bundle["evidence"]}
    assert {c["chunk_id"] for c in backend.index.chunks if c["section_reference"] == "2.1"} <= ids
