from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
import zipfile

from knowledge_pipeline import (
    LLM_PROMPT_VERSION,
    KnowledgePipeline,
    RelationExtraction,
    resolve_candidates,
)
from documents import load_documents
from retrieval import SchemaSearchBackend, SearchIndex, build_index


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
