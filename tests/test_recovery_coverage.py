from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from documents import load_documents
from knowledge_pipeline import KnowledgePipeline, attach_relations
from retrieval import SchemaSearchBackend, SearchIndex
from schemas import SearchFilters, SearchResult
from tools.audit_retrieval import matches


def backend(tmp_path):
    archive = tmp_path / "knowledge-base.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("Правило.txt", "1.1. Оплата товара в течение 30 дней.\n")
        z.writestr("Исключение.txt", "За исключением согласованной предоплаты.\n")
        z.writestr("Условие.txt", "При условии подписания спецификации действует особый порядок.\n")
    search = SchemaSearchBackend(SearchIndex(load_documents(archive)))
    rule = next(c for c in search.index.chunks if c["section_reference"] == "1.1")
    exception = next(c for c in search.index.chunks if "За исключением" in c["text"])
    condition = next(c for c in search.index.chunks if "При условии" in c["text"])
    return search, rule, exception, condition


@pytest.mark.parametrize("part,kind", [("Исключения из срока оплаты", "exceptions"),
                                      ("Условия применения порядка оплаты", "conditions")])
async def test_recovery_is_driven_by_missing_coverage_not_only_empty_results(tmp_path, part, kind):
    search, rule, exception, condition = backend(tmp_path)
    calls = []
    async def find(query, filters):
        calls.append(query)
        source = rule if len(calls) == 1 else exception if kind == "exceptions" else condition
        return SearchResult(evidence=search._as_evidence([source]))
    search.search = find
    result = await search.search_bundle([part], SearchFilters(), recovery_query_budget=1)
    assert len(calls) == 2
    assert result["recovery"]["queries_used"] == 1
    assert result["recovery"]["stop_reason"] == "coverage_satisfied"
    assert result["groups"][0]["coverage"][kind]["status"] == "found"
    assert result["groups"][0]["coverage"]["status"] == "covered"


async def test_recovery_scores_cannot_evict_original_legal_basis(tmp_path):
    search, rule, exception, _ = backend(tmp_path)
    calls = []
    async def find(query, filters):
        calls.append(query)
        evidence = search._as_evidence([rule if len(calls) == 1 else exception])
        evidence[0].metadata["score"] = 1 if len(calls) == 1 else 1000000
        return SearchResult(evidence=evidence)
    search.search = find
    result = await search.search_bundle(["Исключения из срока оплаты"], SearchFilters(),
                                        max_context_chunks=1, recovery_query_budget=1)
    assert {e["chunk_id"] for e in result["evidence"]} == {rule["chunk_id"]}
    assert result["groups"][0]["coverage"]["status"] != "covered"
    for component in ("basis", "conditions", "exceptions"):
        assert set(result["groups"][0]["coverage"][component]["chunk_ids"]) <= {rule["chunk_id"]}


@pytest.mark.parametrize("rejection", ["surface", "model"])
async def test_rejected_relation_cannot_expand_context_or_satisfy_coverage(tmp_path, rejection):
    search, rule, _, condition = backend(tmp_path)
    edge = {"relation_id": "bad-edge", "type": "exception_to",
            "from_chunk_id": rule["chunk_id"], "to_chunk_id": condition["chunk_id"],
            "from_document_id": rule["document_id"], "to_document_id": condition["document_id"],
            "target_reference": "1.1", "quote": condition["text"],
            "supporting_chunk_ids": [condition["chunk_id"]],
            "supporting_ranges": [{"text": condition["text"]}], "checks": {}}
    attach_relations(search.index, [edge], [])
    if rejection == "model":
        edge["type_status"] = "candidate"
        edge["checks"]["relation_type"] = "supported_by_surface_cues"
        edge["semantic_validation"] = {"status": "supported_by_surface_cues",
                                       "model_review": {"status": "contradicted"}}
    async def find(query, filters):
        return SearchResult(evidence=search._as_evidence([rule]))
    search.search = find
    result = await search.search_bundle(["Исключения из срока оплаты"], SearchFilters())
    assert not result["relations"]
    assert all(not e.metadata.get("provision_relations") for e in search.bundle_result(result).evidence)
    assert result["rejected_relations"][0]["relation_id"] == "bad-edge"
    assert condition["chunk_id"] not in {e["chunk_id"] for e in result["evidence"]}
    assert result["groups"][0]["coverage"]["exceptions"]["status"] == "missing_in_retrieved_bundle"
    assert result["recovery"]["stop_reason"] == "no_new_evidence"


async def test_zero_budget_never_claims_missing_exception_was_covered(tmp_path):
    search, rule, _, _ = backend(tmp_path)
    calls = []
    async def find(query, filters):
        calls.append(query)
        return SearchResult(evidence=search._as_evidence([rule]))
    search.search = find
    result = await search.search_bundle(["Исключения из срока оплаты"], SearchFilters(), recovery_query_budget=0)
    assert len(calls) == 1 and result["recovery"]["queries_used"] == 0
    assert result["recovery"]["stop_reason"] == "query_budget_exhausted"
    assert result["groups"][0]["missing"]


def test_expanded_cases_are_distinct_and_have_non_vacuous_expectations():
    cases = json.loads((Path(__file__).resolve().parents[1] / "evals/retrieval_cases.json").read_text())["cases"]
    assert len(cases) >= 50
    assert len({c["id"] for c in cases}) == len(cases)
    assert len({c["query"] for c in cases}) == len(cases)
    assert {"date_boundary", "ambiguity", "missing_edition", "multiple_sources", "false_premise"} <= {c["category"] for c in cases}
    for case in cases:
        assert case.get("required") or case.get("ineligible") or case.get("expected_candidates")


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "data/knowledge-base.zip").is_file(),
                    reason="Нужна открытая база организаторов")
def test_all_extended_gold_targets_refer_to_real_source_chunks():
    root = Path(__file__).resolve().parents[1]
    search = SchemaSearchBackend(SearchIndex(load_documents(root / "data/knowledge-base.zip")))
    for case in json.loads((root / "evals/retrieval_cases.json").read_text())["cases"]:
        for target in case.get("required", []) + case.get("ineligible", []) + case.get("applicability_checks", []):
            assert any(matches(c, target) for c in search.index.chunks), (case["id"], target)


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "data/knowledge-base.zip").is_file(),
                    reason="Нужна открытая база организаторов")
async def test_recovery_preserves_original_18_source_expectations(tmp_path):
    from temporal import temporal_issue
    from datetime import date

    root = Path(__file__).resolve().parents[1]
    search = SchemaSearchBackend(SearchIndex(load_documents(root / "data/knowledge-base.zip")))
    cases = json.loads((root / "evals/retrieval_cases.json").read_text())["cases"][:18]
    for case in cases:
        filters = SearchFilters.model_validate(case.get("filters", {}))
        for rounds in (0, 2):
            bundle = await search.search_bundle([case["query"]], filters, top_k=12,
                                               max_context_chars=60000, max_context_chunks=24,
                                               recovery_rounds=rounds)
            result = search.bundle_result(bundle)
            eligible = [search.index.get_chunk(e.chunk_id) for e in result.evidence
                        if temporal_issue(e, filters, mode="auto", reference=date(2026, 11, 12)) is None]
            for target in case["required"]:
                assert any(matches(c, target) for c in eligible), (case["id"], rounds, target)
            assert not any(matches(c, target) for c in eligible for target in case.get("forbidden", []))


async def test_condition_recovery_can_search_inside_original_document(tmp_path):
    archive = tmp_path / "knowledge-base.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("Правило.txt", "1.1. Оплата товара в течение 30 дней.\n"
                                "2.1. При условии подписания спецификации действует особый порядок.\n")
    search = SchemaSearchBackend(SearchIndex(load_documents(archive)))
    rule = next(c for c in search.index.chunks if c["section_reference"] == "1.1")
    condition = next(c for c in search.index.chunks if c["section_reference"] == "2.1")
    calls = []
    async def find(query, filters):
        values = filters.model_dump(exclude_none=True) if hasattr(filters, "model_dump") else filters
        calls.append(values)
        # The global request is noisy; a document-scoped request finds the condition.
        source = condition if values.get("document_id") == rule["document_id"] else rule
        return SearchResult(evidence=search._as_evidence([source]))
    search.search = find
    result = await search.search_bundle(["Условия оплаты"], SearchFilters())
    assert len(calls) == 3
    assert calls[-1]["document_id"] == rule["document_id"]
    assert result["groups"][0]["coverage"]["conditions"]["status"] == "found"
    assert result["recovery"]["queries_used"] == 2


async def test_relation_review_resumes_beyond_first_batch_and_restores_cache(tmp_path):
    from types import SimpleNamespace

    search, rule, exception, condition = backend(tmp_path)
    edges = [{"relation_id": f"edge-{n}", "type": "exception_to",
              "from_chunk_id": rule["chunk_id"], "to_chunk_id": target["chunk_id"],
              "from_document_id": rule["document_id"], "to_document_id": target["document_id"],
              "supporting_ranges": [{"text": target["text"]}],
              "supporting_chunk_ids": [target["chunk_id"]], "quote": target["text"],
              "checks": {}} for n, target in enumerate((exception, condition))]
    attach_relations(search.index, edges, [])
    calls = []
    class Reviewer:
        settings = SimpleNamespace(model="review-test", base_url="test://endpoint")
        async def structured(self, instruction, data, schema):
            calls.extend(item["relation_id"] for item in data["relations"])
            return schema.model_validate({"reviews": [{"relation_id": item["relation_id"],
                "decision": "supported", "rationale": "Подтверждается цитатой.",
                "evidence_quote": item["to_text"]} for item in data["relations"]]})
    for expected in (1, 0, 0):
        # A new instance simulates restart; cached entries must not consume the batch.
        with KnowledgePipeline(tmp_path / "knowledge-base.zip", tmp_path / "review.sqlite3") as pipeline:
            report = await pipeline.review_relation_types(search.index, Reviewer(), max_relations=1)
        assert report["remaining_relations"] == expected
        assert report["coverage_complete"] == (expected == 0)
        assert report["status"] == ("partial" if expected else "complete")
    assert len(calls) == len(set(calls)) == 2
