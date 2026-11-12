from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock
import zipfile

import pytest

from assistant import LegalAssistant
from conftest import FakeClient, draft, intent
from retrieval import SchemaSearchBackend, SearchIndex, create_search
from documents import load_documents
from schemas import SearchFilters, SearchResult


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


async def test_unrelated_graph_gap_does_not_downgrade_verified_answer(tmp_path, settings):
    (tmp_path / "Трудовые правила.txt").write_text(
        "Статья 70. Испытание\n"
        "В период испытания на работника распространяются положения трудового законодательства.\n"
        "Особенности компенсации определяются статьёй 999 настоящего Кодекса.\n",
        encoding="utf-8",
    )
    backend = create_search(tmp_path)
    backend.search = AsyncMock(return_value=SearchResult(evidence=backend._as_evidence(backend.index.chunks)))
    backend.search_bundle = AsyncMock(wraps=backend.search_bundle)
    rule = next(e for e in backend._as_evidence(backend.index.chunks) if "В период испытания" in e.text)
    plan = intent().model_copy(update={"query": "трудовые правила в период испытания", "parts": ["Распространяются ли трудовые правила?"]})
    response = draft(rule, text="На работника в период испытания распространяется трудовое законодательство.")
    client = FakeClient(settings, plan, response, {"supported": [0], "missing_info": []})
    trace = {}
    result = await LegalAssistant(backend, client).answer("При испытании действуют трудовые правила?", trace=trace)
    bundle = next(s for s in trace["stages"] if s["stage"] == "search_bundle")
    assert bundle["groups"][0]["coverage"]["gaps"]["items"]
    assert bundle["groups"][0]["coverage"]["status"] != "covered"
    assert any(e["metadata"].get("provision_gaps") for e in client.requests[2][1]["evidence"])
    assert result.status == "answer" and not result.missing_info
    assert "Матрица" not in result.text and "Не хватает" not in result.text
    assert len(client.requests) == 3
    backend.search_bundle.assert_awaited_once()


async def test_material_missing_reference_remains_partial(tmp_path, settings):
    (tmp_path / "Правило.txt").write_text(
        "1.1. Работник вправе получить компенсацию при условиях, указанных в пункте 9.9 договора.\n",
        encoding="utf-8",
    )
    backend = create_search(tmp_path)
    rule = backend._as_evidence(backend.index.chunks)[0]
    response = draft(rule, text="Право на компенсацию зависит от условий пункта 9.9 договора.",
                     status="partial", missing_info=["Пункт 9.9 договора с условиями компенсации"])
    plan = intent().model_copy(update={"query": "условия компенсации работнику", "parts": ["Когда положена компенсация?"]})
    client = FakeClient(settings, plan, response, {"supported": [0], "missing_info": ["Пункт 9.9 договора с условиями компенсации"]})
    result = await LegalAssistant(backend, client).answer("Когда положена компенсация?")
    assert result.status == "partial" and result.citations
    assert result.missing_info == ["Пункт 9.9 договора с условиями компенсации"]
    assert "Матрица" not in result.text


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "data/knowledge-base.zip").is_file(),
                    reason="Нужна открытая база организаторов")
async def test_probation_question_with_real_corpus_and_confirmed_verdict(settings):
    backend = create_search(Path(__file__).resolve().parents[1] / "data")
    backend.search_bundle = AsyncMock(wraps=backend.search_bundle)
    quote = ("В период испытания на работника распространяются положения трудового законодательства "
             "и иных нормативных правовых актов, содержащих нормы трудового права, коллективного "
             "договора, соглашений, локальных нормативных актов.")
    source = next(c for c in backend.index.chunks if quote in c["text"])
    rule = backend._as_evidence([source])[0]
    plan = intent(temporal_mode="current").model_copy(update={
        "query": "распространение трудового законодательства на работника в период испытательного срока",
        "parts": ["Распространяются ли общие нормы трудового права на работника, находящегося на испытательном сроке?"],
    })
    response = draft(rule, text="На работника в период испытания распространяется трудовое законодательство.")
    response["claims"][0]["supports"][0]["quote"] = quote
    # This checks real retrieval and status assembly with a fixed model verdict.
    # It does not measure the quality of a live model.
    client = FakeClient(settings, plan, response, {"supported": [0], "missing_info": []})
    trace = {}
    answer = await LegalAssistant(backend, client).answer(
        "Сотрудник работает на испытательном сроке. На него распространяются обычные трудовые правила?",
        trace=trace,
    )
    assert answer.status == "answer" and answer.missing_info == []
    assert answer.citations[0].quote == quote
    assert next(s for s in trace["stages"] if s["stage"] == "search_bundle")["gaps"]
    backend.search_bundle.assert_awaited_once()
