from __future__ import annotations

from types import SimpleNamespace
import zipfile

import pytest

from documents import load_documents
from knowledge_pipeline import KnowledgePipeline, resolve_candidates
from retrieval import SchemaSearchBackend, SearchIndex, build_index, create_search
from schemas import SearchFilters


def corpus(tmp_path, sources):
    path = tmp_path / "knowledge-base.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in sources.items():
            archive.writestr(name, text)
    return path


def candidate(chunk, target_document):
    return {
        "candidate_id": "probe", "source_chunk_id": chunk["chunk_id"],
        "relation_type": "references", "target_reference": "2.1",
        "target_document": target_document, "quote": chunk["text"],
        "start_char": chunk["start_char"], "end_char": chunk["end_char"],
        "extraction_method": "llm", "algorithm_version": "test",
    }


def test_missing_explicit_document_never_falls_back_to_local_clause(tmp_path):
    path = corpus(tmp_path, {"Письмо.txt": "ПИСЬМО\n2.1. Локальное правило.\n"
                           "3.1. Согласно пункту 2.1 договора № Т-99/2026 выполняется проверка.\n"})
    index = build_index(path, pipeline_cache_path=tmp_path / "cache.sqlite3")
    source = next(c for c in index.chunks if c["section_reference"] == "3.1")
    assert not source["provision_relations"]  # legacy and new extraction agree
    assert any(g["kind"] == "referenced_document_not_found" for g in source["provision_gaps"])
    relations, gaps = resolve_candidates(index, [candidate(source, "Т-99/2026")])
    assert not relations
    assert gaps[0]["kind"] == "referenced_document_not_found"


@pytest.mark.parametrize("descriptor", ["Т-9/2026", "Договор № Т-9/2026"])
def test_explicit_document_narrows_scope_despite_local_duplicate(tmp_path, descriptor):
    path = corpus(tmp_path, {
        "Договор.txt": "ДОГОВОР № Т-9/2026\n2.1. Условие договора.\n",
        "Письмо.txt": "ПИСЬМО\n2.1. Локальное правило.\n"
                      "3.1. Согласно пункту 2.1 договора № Т-9/2026 выполняется проверка.\n",
    })
    index = build_index(path, pipeline_cache_path=tmp_path / "cache.sqlite3")
    source = next(c for c in index.chunks if c["section_reference"] == "3.1")
    relations, gaps = resolve_candidates(index, [candidate(source, descriptor)])
    assert not gaps and len(relations) == 1
    assert index.get_chunk(relations[0]["to_chunk_id"])["text"] == "2.1. Условие договора."
    assert all(r["from_document_id"] != r["to_document_id"] for r in source["provision_relations"])


def test_missing_explicit_law_does_not_resolve_to_local_article(tmp_path):
    path = corpus(tmp_path, {"Письмо.txt": "Статья 70. Локальное положение\n"
                           "3.1. Согласно статье 70 ТК РФ выполняется проверка.\n"})
    index = build_index(path, pipeline_cache_path=tmp_path / "cache.sqlite3")
    source = next(c for c in index.chunks if c["section_reference"] == "3.1")
    assert not source["provision_relations"]
    assert source["provision_gaps"]


def test_ambiguous_amendment_never_keeps_legacy_edges_including_cached_graph(tmp_path):
    path = corpus(tmp_path, {
        "Договор А.txt": "ДОГОВОР № Т-9/2026\n2.1. Первый пункт.\n",
        "Договор Б.txt": "ДОГОВОР № Т-9/2026\n2.1. Другой пункт.\n",
        "Доп соглашение.txt": "ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ к договору № Т-9/2026\n"
                              "Изложить пункт 2.1 в новой редакции.\n2.1. Новый порядок.\n",
    })
    cache = tmp_path / "cache.sqlite3"
    for index in (SearchIndex(load_documents(path)), build_index(path, pipeline_cache_path=cache),
                  build_index(path, pipeline_cache_path=cache)):
        assert not any(r["type"] in {"changes_provision", "amends", "defines_applicability"}
                       for r in index.provision_relations)
        assert any(g["kind"] == "referenced_provision_ambiguous" for g in index.provision_gaps)


class Extractor:
    def __init__(self, failing=None):
        self.settings = SimpleNamespace(model="extractor", embedding_model=None, reranker_model=None)
        self.calls = []
        self.failing = failing

    async def structured(self, instruction, data, schema):
        self.calls.append(data["chunk_id"])
        if data["chunk_id"] == self.failing:
            raise RuntimeError("unavailable")
        return schema.model_validate({"relations": []})


def implicit_corpus(tmp_path):
    # No regex markers: implicit dependencies must also be eligible for LLM.
    return corpus(tmp_path, {"Письмо.txt": "Первое положение.\n\nВторое положение.\n\nТретье положение.\n"})


async def test_queue_resumes_after_restart_without_repeating_successful_calls(tmp_path):
    path = implicit_corpus(tmp_path)
    docs = load_documents(path)
    cache = tmp_path / "cache.sqlite3"
    client = Extractor()
    with KnowledgePipeline(path, cache) as pipeline:
        chunks = pipeline.prepare(docs)["chunks"]
        _, first = await pipeline.llm_candidates(chunks, client, max_chunks=1)
        assert first["status"] == "partial" and first["remaining_chunks"] == 2
        assert first["attempted_chunks"] == 1 and first["budget_exhausted"]
    with KnowledgePipeline(path, cache) as pipeline:
        chunks = pipeline.prepare(docs)["chunks"]
        _, second = await pipeline.llm_candidates(chunks, client, max_chunks=1)
        assert second["completed_chunks"] == 2 and second["remaining_chunks"] == 1
        _, third = await pipeline.llm_candidates(chunks, client, max_chunks=1)
        assert third["status"] == "complete" and third["coverage_complete"]
        _, fourth = await pipeline.llm_candidates(chunks, client, max_chunks=1)
        assert fourth["attempted_chunks"] == 0 and fourth["cached_chunks"] == 3
    assert len(client.calls) == len(set(client.calls)) == 3


async def test_failed_first_chunk_does_not_starve_remaining_work(tmp_path):
    path = implicit_corpus(tmp_path)
    with KnowledgePipeline(path, tmp_path / "cache.sqlite3") as pipeline:
        chunks = pipeline.prepare(load_documents(path))["chunks"]
        client = Extractor(failing=chunks[0]["chunk_id"])
        for _ in range(3):
            _, report = await pipeline.llm_candidates(chunks, client, max_chunks=1)
        assert report["completed_chunks"] == 2 and report["remaining_chunks"] == 1
        assert set(client.calls) == {c["chunk_id"] for c in chunks}
        client.failing = None
        _, report = await pipeline.llm_candidates(chunks, client, max_chunks=1)
        assert report["coverage_complete"] and len(client.calls) == 4


async def test_bundle_budget_applies_once_for_all_question_parts_and_advances_next_time(tmp_path, monkeypatch):
    path = implicit_corpus(tmp_path)
    monkeypatch.setenv("KNOWLEDGE_LLM_MAX_CHUNKS", "1")
    index = build_index(path, pipeline_cache_path=tmp_path / "cache.sqlite3", retain_pipeline=True)
    client = Extractor()
    backend = SchemaSearchBackend(index, client, knowledge_pipeline=index.knowledge_pipeline, llm_relations=True)
    try:
        for completed in (1, 2, 3):
            result = await backend.search_bundle(["первое положение", "второе положение"], SearchFilters())
            assert len(client.calls) == completed
            assert result["pipeline"]["llm_relations"]["completed_chunks"] == completed
        assert result["pipeline"]["llm_relations"]["coverage_complete"]
        await backend.prepare_knowledge_graph()
        assert len(client.calls) == 3
    finally:
        index.knowledge_pipeline.close()


async def test_partial_graph_restores_all_prior_relations_then_caches_complete_coverage(tmp_path, monkeypatch):
    path = corpus(tmp_path, {"Письмо.txt": "1.1. Срок оплаты определяет пункт 2.1.\n"
                           "1.2. Оригиналы получает секретарь.\n2.1. Основное правило.\n"})
    cache = tmp_path / "cache.sqlite3"
    monkeypatch.setenv("KNOWLEDGE_LLM_MAX_CHUNKS", "1")

    class Client(Extractor):
        async def structured(self, instruction, data, schema):
            self.calls.append(data["chunk_id"])
            relations = [{"relation_type": "applies_to", "target_reference": "2.1",
                          "quote": data["source_text"]}] if data["section_reference"] == "1.1" else []
            return schema.model_validate({"relations": relations})

    client = Client()
    for step, completed in enumerate((1, 2, 3, 3)):
        if step == 3:
            monkeypatch.setenv("KNOWLEDGE_LLM_MAX_CHUNKS", "64")
        index = build_index(path, pipeline_cache_path=cache, retain_pipeline=True)
        backend = SchemaSearchBackend(index, client, knowledge_pipeline=index.knowledge_pipeline, llm_relations=True)
        try:
            await backend.prepare_knowledge_graph()
            report = index.pipeline_diagnostics["llm_relations"]
            assert report["completed_chunks"] == completed
            assert report["remaining_chunks"] == 3 - completed
            assert len(client.calls) == completed
            assert len([r for r in index.provision_relations if r.get("extraction_method") == "llm"]) == 1
        finally:
            index.knowledge_pipeline.close()
    assert report["status"] == "cached" and report["coverage_complete"]


def test_explicit_writable_cache_works_with_readonly_home(tmp_path, monkeypatch):
    home = tmp_path / "readonly-home"
    home.mkdir()
    home.chmod(0o555)
    monkeypatch.setenv("XDG_CACHE_HOME", str(home))
    monkeypatch.setenv("OTOROP_KNOWLEDGE_CACHE_DIR", str(tmp_path / "writable-tmp"))
    path = implicit_corpus(tmp_path)
    try:
        backend = create_search(path)
        assert backend.index.pipeline_diagnostics["status"] == "ready"
        assert list((tmp_path / "writable-tmp").glob("*.sqlite3"))
        assert not list(home.iterdir())
    finally:
        home.chmod(0o755)
