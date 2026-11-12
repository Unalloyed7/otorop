from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from knowledge_pipeline import (
    LLM_PROMPT_VERSION,
    KnowledgePipeline,
    RelationExtraction,
    resolve_candidates,
)
from documents import load_documents
from retrieval import SchemaSearchBackend, SearchIndex, build_index
from schemas import SearchResult
from knowledge_pipeline import validate_relation_semantics


def _write_archive(path: Path, contents: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in contents.items():
            archive.writestr(name, text)


def _corpus(extra_text: str = "2.1. Согласование выполняет юридический отдел.\n") -> dict[str, str]:
    return {
        "регламент__Регламент А.txt": (
            "РЕГЛАМЕНТ А\n"
            "2.1. Согласование выполняет юридический отдел.\n"
            "2.2. " + extra_text
        ),
        "письма__Письмо Б.txt": (
            "ПИСЬМО Б\n"
            "При применении порядка учитываются пункты 2.1, 2.2 и 4.2.\n"
            "См. пункты 3–5 настоящего документа.\n"
        ),
    }


def test_pipeline_resumes_cached_stages_and_invalidates_changed_document(tmp_path):
    archive = tmp_path / "knowledge-base.zip"
    cache = tmp_path / "pipeline.sqlite3"
    _write_archive(archive, _corpus())

    first = build_index(archive, pipeline_cache_path=cache)
    first_ids = {chunk["chunk_id"] for chunk in first.chunks}
    assert first.pipeline_diagnostics["status"] == "ready"
    assert first.pipeline_diagnostics["stages"]["structure"]["processed"] == 2
    assert first.pipeline_diagnostics["reference_count"] >= 6

    second = build_index(archive, pipeline_cache_path=cache)
    assert {chunk["chunk_id"] for chunk in second.chunks} == first_ids
    assert second.pipeline_diagnostics["stages"]["structure"]["cached"] == 2
    assert second.pipeline_diagnostics["stages"]["candidates"]["cached"] == 2
    assert second.pipeline_diagnostics["stages"]["graph"]["cached"] == 1

    _write_archive(archive, _corpus("2.1. Согласование выполняет директор.\n"))
    changed = build_index(archive, pipeline_cache_path=cache)
    assert changed.pipeline_diagnostics["stages"]["structure"]["cached"] == 1
    assert changed.pipeline_diagnostics["stages"]["structure"]["processed"] == 1
    assert changed.pipeline_diagnostics["stages"]["graph"]["processed"] == 1


def test_llm_enriched_graph_is_restored_from_cache(tmp_path):
    archive = tmp_path / "knowledge-base.zip"
    cache = tmp_path / "pipeline.sqlite3"
    _write_archive(archive, {
        "письма__Письмо.txt": (
            "Письмо о порядке.\n"
            "При условии, указанном в пункте 9.9, применяется особый порядок.\n"
        ),
    })

    class Client:
        settings = SimpleNamespace(model="test-extractor-v1")
        calls = 0

        async def structured(self, instruction, data, schema):
            self.calls += 1
            return schema.model_validate({
                "relations": [{
                    "relation_type": "condition_for",
                    "target_reference": "9.9",
                    "quote": "При условии, указанном в пункте 9.9",
                }],
            })

    client = Client()
    first_index = build_index(archive, pipeline_cache_path=cache, retain_pipeline=True)
    first_backend = SchemaSearchBackend(
        first_index,
        client,
        knowledge_pipeline=first_index.knowledge_pipeline,
        llm_relations=True,
    )
    asyncio.run(first_backend.prepare_knowledge_graph())
    first_gap_count = len(first_index.provision_gaps)
    assert first_index.pipeline_diagnostics["llm_relations"]["status"] == "complete"
    assert first_gap_count > 0

    second_index = build_index(archive, pipeline_cache_path=cache, retain_pipeline=True)
    second_backend = SchemaSearchBackend(
        second_index,
        client,
        knowledge_pipeline=second_index.knowledge_pipeline,
        llm_relations=True,
    )
    asyncio.run(second_backend.prepare_knowledge_graph())
    assert second_index.pipeline_diagnostics["llm_relations"]["status"] == "cached"
    assert len(second_index.provision_gaps) == first_gap_count
    assert client.calls == 1
    first_index.knowledge_pipeline.close()
    second_index.knowledge_pipeline.close()


def test_entity_mentions_for_changed_or_removed_documents_are_pruned(tmp_path):
    archive = tmp_path / "knowledge-base.zip"
    cache = tmp_path / "pipeline.sqlite3"
    _write_archive(archive, {
        "договоры__Договор А.txt": "ДОГОВОР № А-1/2026. Контрагент Альфа-Трейд.\n",
        "письма__Письмо Б.txt": "Письмо для Бета.\n",
    })
    first = build_index(archive, pipeline_cache_path=cache)
    removed_id = next(
        doc["document_id"] for doc in first.documents.values()
        if doc["source_file"].endswith("Письмо Б.txt")
    )
    _write_archive(archive, {
        "договоры__Договор А.txt": "ДОГОВОР № А-1/2026. Контрагент Альфа-Трейдинг.\n",
    })
    build_index(archive, pipeline_cache_path=cache)
    import sqlite3
    connection = sqlite3.connect(cache)
    try:
        assert connection.execute(
            "SELECT count(*) FROM entity_mentions WHERE document_id=?",
            (removed_id,),
        ).fetchone()[0] == 0
    finally:
        connection.close()


def test_relation_type_is_checked_separately_from_quote_provenance(tmp_path):
    archive = tmp_path / "knowledge-base.zip"
    _write_archive(archive, {
        "договоры__Договор Т-4.txt": "ДОГОВОР № Т-4/2026\n3.2. Старый адрес.\n",
        "допсоглашения__Доп соглашение Т-4.txt": (
            "Дополнительное соглашение к договору № Т-4/2026.\n"
            "Изложить пункт 3.2 договора в следующей редакции:\n"
            "3.2. Новый адрес.\n"
        ),
    })
    index = build_index(archive, pipeline_cache_path=tmp_path / "pipeline.sqlite3")
    relation = next(
        item for item in index.provision_relations
        if item["type"] == "changes_provision"
    )
    assert relation["semantic_validation"]["status"] == "supported_by_surface_cues"
    assert relation["checks"]["quote"] == "verified"

    wrong_type = dict(relation)
    wrong_type["type"] = "exception_to"
    wrong_type["semantic_validation"] = validate_relation_semantics(wrong_type, index)
    assert wrong_type["semantic_validation"]["status"] == "contradicted"
    assert relation["semantic_validation"]["status"] != "verified"

    overbroad_type = dict(relation)
    overbroad_type["type"] = "supplements"
    overbroad_type["quote"] = "Требование применяется в дополнение к порядку, установленному п. 5.4."
    overbroad_type["semantic_validation"] = validate_relation_semantics(overbroad_type, index)
    assert overbroad_type["semantic_validation"]["status"] == "contradicted"
    assert overbroad_type["semantic_validation"]["conflicting_type"] == "supplements_procedure"


@pytest.mark.asyncio
async def test_bundle_recovery_searches_missing_part_with_bounded_budget(tmp_path):
    archive = tmp_path / "knowledge-base.zip"
    _write_archive(archive, {
        "договоры__Договор Т-5.txt": (
            "ДОГОВОР № Т-5/2026\n"
            "4.2. Покупатель оплачивает товар в течение 30 дней после получения УПД.\n"
        ),
    })
    backend = SchemaSearchBackend(
        SearchIndex(load_documents(archive)),
        top_k=2,
    )
    chunk = backend.index.chunks[0]
    evidence = backend._as_evidence([chunk])[0]
    calls = []

    async def recover(query, filters):
        calls.append(query)
        return SearchResult(evidence=[] if len(calls) == 1 else [evidence])

    backend.search = recover
    bundle = await backend.search_bundle(
        ["порядок оплаты по договору"],
        {},
        top_k=1,
        recovery_rounds=2,
        recovery_query_budget=1,
    )
    group = bundle["groups"][0]
    assert len(calls) == 2
    assert bundle["recovery"]["queries_used"] == 1
    assert bundle["recovery"]["stop_reason"] == "coverage_satisfied"
    assert group["recovered_chunk_ids"] == [chunk["chunk_id"]]
    assert group["coverage"]["basis"]["status"] == "found"
    assert group["coverage"]["status"] == "covered"


@pytest.mark.asyncio
async def test_bundle_reports_explicit_ambiguity_resolution_and_coverage_matrix(tmp_path):
    archive = tmp_path / "knowledge-base.zip"
    _write_archive(archive, {
        "письма__Письмо.txt": (
            "Письмо. Для расчёта стоимости применяют пункт 2.1.\n"
        ),
        "договоры__Альфа.txt": "ДОГОВОР № Т-6/2026\n2.1. Цена для Альфа-Трейд.\n",
        "договоры__Бета.txt": "ДОГОВОР № Т-6/2026\n2.1. Цена для Бета.\n",
    })
    index = SearchIndex(load_documents(archive))
    source = next(chunk for chunk in index.chunks if "применяют пункт 2.1" in chunk["text"])
    candidates = [
        chunk for chunk in index.chunks
        if chunk.get("section_reference") == "2.1"
    ]
    assert len(candidates) == 2
    candidates[0]["counterparties"] = ["Альфа-Трейд"]
    candidates[1]["counterparties"] = ["Бета"]
    index.provision_gaps.append({
        "kind": "referenced_provision_ambiguous",
        "candidate_id": "ambiguous-1",
        "source_chunk_id": source["chunk_id"],
        "target_reference": "2.1",
        "candidate_chunk_ids": [chunk["chunk_id"] for chunk in candidates],
        "message": "Две цели.",
    })
    backend = SchemaSearchBackend(index, top_k=2)
    source_evidence = backend._as_evidence([source])[0]

    async def source_only(query, filters):
        return SearchResult(evidence=[source_evidence])

    backend.search = source_only
    bundle = await backend.search_bundle(
        ["пункт 2.1 цены"],
        {"counterparty": "Альфа-Трейд"},
        top_k=2,
        recovery_rounds=1,
        recovery_query_budget=1,
    )
    group = bundle["groups"][0]
    assert group["gap_resolutions"][0]["status"] == "resolved_by_explicit_filter"
    assert group["coverage"]["basis"]["status"] == "found"
    assert group["coverage"]["gaps"]["status"] == "none_detected_in_indexed_graph"
    assert group["coverage"]["conditions"]["status"] == "not_explicitly_requested"
    assert group["coverage"]["exceptions"]["status"] == "not_explicitly_requested"
    assert candidates[0]["chunk_id"] in group["evidence_chunk_ids"]


@pytest.mark.asyncio
async def test_bundle_resolves_ambiguous_document_with_explicit_document_filter(tmp_path):
    archive = tmp_path / "knowledge-base.zip"
    _write_archive(archive, {
        "письма__Письмо.txt": "Письмо. Применяется порядок из связанного документа.\n",
        "регламенты__Регламент А.txt": "Регламент А. 1.1. Порядок согласования А.\n",
        "регламенты__Регламент Б.txt": "Регламент Б. 1.1. Порядок согласования Б.\n",
    })
    index = SearchIndex(load_documents(archive))
    source = next(chunk for chunk in index.chunks if "связанного документа" in chunk["text"])
    candidate_documents = [
        document for document in index.documents.values()
        if document["source_file"].endswith(("Регламент А.txt", "Регламент Б.txt"))
    ]
    assert len(candidate_documents) == 2
    chosen_document = next(
        document for document in candidate_documents
        if document["source_file"].endswith("Регламент А.txt")
    )
    index.provision_gaps.append({
        "kind": "referenced_document_ambiguous",
        "candidate_id": "ambiguous-documents",
        "source_chunk_id": source["chunk_id"],
        "target_reference": "регламент",
        "candidate_document_ids": [document["document_id"] for document in candidate_documents],
        "message": "Два подходящих регламента.",
    })
    backend = SchemaSearchBackend(index, top_k=2)
    source_evidence = backend._as_evidence([source])[0]

    async def source_only(query, filters):
        return SearchResult(evidence=[source_evidence])

    backend.search = source_only
    bundle = await backend.search_bundle(
        ["порядок согласования"],
        {"document_id": chosen_document["document_id"]},
        top_k=2,
        recovery_rounds=1,
        recovery_query_budget=1,
    )
    group = bundle["groups"][0]
    resolution = group["gap_resolutions"][0]
    assert resolution["kind"] == "referenced_document_ambiguous"
    assert resolution["status"] == "resolved_by_explicit_filter"
    assert resolution["chunk_id"] in group["evidence_chunk_ids"]
    assert resolution["chunk_id"] in group["coverage"]["basis"]["chunk_ids"]


def test_reference_parser_expands_lists_and_bounded_ranges_with_source_offsets():
    from knowledge_pipeline import extract_references

    source = "Соблюдать пункты 2.1, 2.3 и 4.2, а также пункты 3–5."
    refs = extract_references(source, absolute_start=100)
    assert [item["reference"] for item in refs] == ["2.1", "2.3", "4.2", "3", "4", "5"]
    for item in refs:
        start = item["start_char"] - 100
        end = item["end_char"] - 100
        assert source[start:end] == item["quote"]
    assert all(item["end_char"] > item["start_char"] for item in refs)


def test_ambiguous_reference_stays_a_gap_instead_of_choosing_first(tmp_path):
    archive = tmp_path / "knowledge-base.zip"
    _write_archive(archive, {
        "договоры__Т-9-А.txt": "ДОГОВОР № Т-9/2026\n2.1. Первый пункт.\n",
        "договоры__Т-9-Б.txt": "ДОГОВОР № Т-9/2026\n2.1. Другой пункт.\n",
        "допсоглашения__Дополнительное соглашение № 1 к Т-9.txt": (
            "Дополнительное соглашение к договору № Т-9/2026.\n"
            "Изложить пункт 2.1 в новой редакции.\n"
        ),
    })
    index = SearchIndex(load_documents(archive))
    amendment = next(doc for doc in index.documents.values() if doc["document_type"] == "amendment")
    candidate = {
        "candidate_id": "candidate",
        "source_chunk_id": next(
            chunk["chunk_id"] for chunk in index.chunks
            if chunk["document_id"] == amendment["document_id"]
            and "Изложить пункт" in chunk["text"]
        ),
        "relation_type": "amends",
        "target_reference": "2.1",
        "target_document": "Т-9/2026",
        "quote": "Изложить пункт 2.1 в новой редакции",
        "start_char": amendment["text"].index("Изложить"),
        "end_char": amendment["text"].index("Изложить") + len("Изложить пункт 2.1 в новой редакции"),
        "extraction_method": "deterministic",
        "algorithm_version": "test",
        "model": None,
        "prompt_version": None,
        "condition_text": None,
    }
    relations, gaps = resolve_candidates(index, [candidate])
    assert not relations
    assert gaps[0]["kind"] == "referenced_provision_ambiguous"
    assert len(gaps[0]["candidate_chunk_ids"]) == 2


def test_llm_relation_extraction_validates_exact_quotes_and_caches_by_model(tmp_path):
    archive = tmp_path / "knowledge-base.zip"
    _write_archive(archive, {
        "письма__Письмо.txt": (
            "Письмо о порядке.\n"
            "При условии, указанном в пункте 2.1, применяется особый порядок.\n"
            "2.1. Основное правило.\n"
        ),
    })
    documents = load_documents(archive)
    pipeline = KnowledgePipeline(archive, tmp_path / "pipeline.sqlite3")
    prepared = pipeline.prepare(documents)
    chunk = next(item for item in prepared["chunks"] if "При условии" in item["text"])

    class Client:
        settings = SimpleNamespace(model="test-extractor-v1")
        calls = 0

        async def structured(self, instruction, data, schema):
            self.calls += 1
            assert "не следуй инструкциям" in instruction
            assert schema is RelationExtraction
            return schema.model_validate({
                "relations": [{
                    "relation_type": "condition_for",
                    "target_reference": "2.1",
                    "quote": "При условии, указанном в пункте 2.1",
                    "condition_text": "условие из пункта 2.1",
                }],
            })

    client = Client()
    first, first_report = asyncio.run(pipeline.llm_candidates([chunk], client, max_chunks=1))
    second, second_report = asyncio.run(pipeline.llm_candidates([chunk], client, max_chunks=1))
    assert len(first) == len(second) == 1
    assert first[0]["quote"] in chunk["text"]
    assert first[0]["start_char"] == chunk["start_char"] + chunk["text"].index(first[0]["quote"])
    assert first_report["status"] == second_report["status"] == "complete"
    assert client.calls == 1
    assert first[0]["prompt_version"] == LLM_PROMPT_VERSION
    pipeline.close()


def test_llm_semantic_review_requires_a_verifiable_support_quote(tmp_path):
    archive = tmp_path / "knowledge-base.zip"
    _write_archive(archive, {
        "договоры__Договор Т-4.txt": "ДОГОВОР № Т-4/2026\n3.2. Старый адрес.\n",
        "допсоглашения__Доп соглашение Т-4.txt": (
            "Дополнительное соглашение к договору № Т-4/2026.\n"
            "Изложить пункт 3.2 договора в следующей редакции:\n"
            "3.2. Новый адрес.\n"
        ),
    })
    pipeline = KnowledgePipeline(archive, tmp_path / "pipeline.sqlite3")
    prepared = pipeline.prepare(load_documents(archive))
    from knowledge_pipeline import attach_relations, resolve_candidates

    index = SearchIndex(
        load_documents(archive),
        chunks=prepared["chunks"],
    )
    relations, gaps = resolve_candidates(index, prepared["candidates"])
    attach_relations(index, relations, gaps)
    relation = index.provision_relations[0]

    class Client:
        settings = SimpleNamespace(model="review-model-v1")
        calls = 0

        async def structured(self, instruction, data, schema):
            self.calls += 1
            assert "Точного совпадения цитаты недостаточно" in instruction
            assert schema.__name__ == "RelationReviewBatch"
            return schema.model_validate({
                "reviews": [
                    {
                        "relation_id": item["relation_id"],
                        "decision": "supported",
                        "rationale": "Цитата задаёт изменение указанного пункта.",
                        "evidence_quote": item["support_quote"],
                    }
                    for item in data["relations"]
                ],
            })

    client = Client()
    report = asyncio.run(
        pipeline.review_relation_types(index, client, max_relations=32)
    )
    cached = asyncio.run(
        pipeline.review_relation_types(index, client, max_relations=32)
    )
    assert report["decisions"]["supported"] == report["reviewed"]
    assert cached["cached"] == report["reviewed"]
    assert client.calls == 1
    relation = next(
        item for item in index.provision_relations
        if item["relation_id"] == relation["relation_id"]
    )
    assert relation["semantic_validation"].get("model_review"), relation
    assert relation["semantic_validation"]["model_review"]["status"] == "supported"
    pipeline.close()


def test_invalid_llm_quotes_are_not_promoted_to_relations(tmp_path):
    archive = tmp_path / "knowledge-base.zip"
    _write_archive(archive, {
        "письма__Письмо.txt": "Письмо. При условии пункта 2.1 применяется правило.\n2.1. Правило.\n",
    })
    documents = load_documents(archive)
    pipeline = KnowledgePipeline(archive, tmp_path / "pipeline.sqlite3")
    prepared = pipeline.prepare(documents)
    chunk = next(item for item in prepared["chunks"] if "При условии" in item["text"])

    class Client:
        settings = SimpleNamespace(model="test-extractor-v1")

        async def structured(self, instruction, data, schema):
            return schema.model_validate({
                "relations": [{
                    "relation_type": "condition_for",
                    "target_reference": "2.1",
                    "quote": "Невыдуманная цитата, которой нет в документе",
                }],
            })

    candidates, report = asyncio.run(
        pipeline.llm_candidates([chunk], Client(), max_chunks=1)
    )
    assert candidates == []
    assert report["status"] == "partial"
    assert report["remaining_chunks"] == 1
    assert not report["coverage_complete"]
    pipeline.close()
