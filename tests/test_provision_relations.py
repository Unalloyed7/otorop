from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from retrieval import build_index, create_search, evaluate_applicability
from schemas import SearchFilters

ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "data" / "knowledge-base.zip"
def _section(index, document_type: str, contract: str, section: str, contains: str | None = None):
    return next(
        chunk for chunk in index.chunks
        if chunk["document_type"] == document_type
        and chunk.get("contract_number") == contract
        and chunk.get("section_reference") == section
        and (contains is None or contains in chunk["text"])
    )


@pytest.mark.skipif(not KB_PATH.is_file(), reason="Локальная база организаторов не установлена")
def test_amendment_edges_identify_original_clause_replacement_and_exact_condition():
    index = build_index(KB_PATH)
    original = _section(index, "agreement", "П-17/2024", "3.2")
    replacement = _section(
        index,
        "amendment",
        "П-17/2024",
        "3.2",
        "Московская область, г. Химки",
    )
    condition = next(
        chunk for chunk in index.chunks
        if chunk["document_type"] == "amendment"
        and chunk.get("chunk_role") == "applicability_condition"
        and "Спецификации" in chunk["text"]
    )

    changed = next(
        edge for edge in index.provision_relations
        if edge["type"] == "changes_provision"
        and edge["from_chunk_id"] == original["chunk_id"]
        and edge["to_chunk_id"] == replacement["chunk_id"]
    )
    conditioned = next(
        edge for edge in index.provision_relations
        if edge["type"] == "defines_applicability"
        and edge["from_chunk_id"] == replacement["chunk_id"]
        and edge["to_chunk_id"] == condition["chunk_id"]
    )
    for relation in (changed, conditioned):
        support = index.get_chunk(relation["supporting_chunk_ids"][0])
        document = index.get_document(support["document_id"])
        assert document["text"][support["start_char"]:support["end_char"]] == relation["quote"]
        assert relation["confidence"] == "explicit"
        assert relation["target_reference"] == "3.2"

    condition_record = condition["applicability_conditions"][0]
    source = index.get_document(condition["document_id"])["text"]
    assert source[condition_record["start_char"]:condition_record["end_char"]] == condition_record["source_text"]
    assert evaluate_applicability(condition, "2024-10-01")["status"] == "not_applicable"
    assert evaluate_applicability(condition, "2024-10-02")["status"] == "applicable"

    unrelated = [
        chunk for chunk in index.chunks
        if chunk["document_id"] == condition["document_id"]
        and "Остальные условия договора" in chunk["text"]
    ]
    assert unrelated
    assert all(not chunk["applicability_conditions"] for chunk in unrelated)


@pytest.mark.asyncio
@pytest.mark.skipif(not KB_PATH.is_file(), reason="Локальная база организаторов не установлена")
async def test_search_bundle_keeps_rule_change_condition_and_exact_context_spans():
    backend = create_search(ROOT / "data")
    bundle = await backend.search_bundle(
        ["адрес доставки", "когда применяется новый адрес"],
        SearchFilters(
            contract_number="П-17/2024",
            event_date="2024-10-02",
            date_kind="specification_signed",
        ),
        top_k=5,
        max_context_chars=5000,
    )

    evidence = bundle["evidence"]
    assert any(item["document_type"] == "agreement" and item["section_reference"] == "3.2" for item in evidence)
    assert any(
        item["document_type"] == "amendment"
        and "Химки" in item["text"]
        and item["applicability"]["status"] == "applicable"
        for item in evidence
    )
    assert any(item.get("chunk_role") == "applicability_condition" for item in evidence)
    assert {edge["type"] for edge in bundle["relations"]} >= {
        "changes_provision",
        "defines_applicability",
    }
    assert all(not group["missing"] for group in bundle["groups"])
    assert not bundle["truncated"]
    for span in bundle["context_spans"]:
        document = backend.get_document(span["document_id"])
        assert document["text"][span["start_char"]:span["end_char"]] == span["text"]
        assert span["source_chunk_ids"]


@pytest.mark.asyncio
@pytest.mark.skipif(not KB_PATH.is_file(), reason="Локальная база организаторов не установлена")
async def test_search_bundle_reports_budget_truncation_without_partial_relation_pair():
    backend = create_search(ROOT / "data")
    bundle = await backend.search_bundle(
        ["адрес доставки"],
        SearchFilters(contract_number="П-17/2024"),
        top_k=5,
        max_context_chars=20,
    )

    assert bundle["truncated"]
    assert bundle["omitted_chunk_ids"]
    assert bundle["evidence"] == []
    assert bundle["groups"][0]["truncated"]


@pytest.mark.skipif(not KB_PATH.is_file(), reason="Локальная база организаторов не установлена")
def test_explicit_edition_and_procedure_relations_are_exposed():
    index = build_index(KB_PATH)
    assert any(
        edge["type"] == "supersedes_edition"
        and edge["confidence"] == "explicit"
        and edge["target_reference"] == "2.4"
        for edge in index.provision_relations
    )
    assert any(
        edge["type"] == "supplements_procedure"
        and edge["confidence"] == "explicit"
        and edge["target_reference"] == "5.4"
        for edge in index.provision_relations
    )


def test_missing_referenced_article_is_a_gap_not_synthesized(tmp_path):
    archive_path = tmp_path / "knowledge-base.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "документы компании__Письмо К-88.txt",
            "Письмо комплаенс-службы К-88.\n"
            "При оценке вопроса учитывается статья 67 Трудового кодекса РФ.\n",
        )
        archive.writestr(
            "законы__ТК_ст70_действующая.txt",
            "Источник: Статья 70. ТК РФ\nСтатья 70. Испытание при приеме на работу.\n",
        )
    index = build_index(archive_path)
    source = next(chunk for chunk in index.chunks if "статья 67" in chunk["text"].casefold())
    gap = next(
        item for item in index.provision_gaps
        if item["source_chunk_id"] == source["chunk_id"]
    )
    assert gap["kind"] == "referenced_provision_not_found"
    assert gap["target_reference"] == "67"
    assert "статья 67" in gap["source_text"].casefold()
    assert not any(
        chunk["document_type"] == "law" and chunk.get("section_reference") == "67"
        for chunk in index.chunks
    )


def test_line_wrapped_date_condition_preserves_source_and_clause_scope(tmp_path):
    archive_path = tmp_path / "knowledge-base.zip"
    amendment_text = (
        "ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ № 1\n"
        "к договору поставки № Т-9/2026\n"
        "Изложить пункт 3.2 договора в следующей редакции:\n"
        "3.2. Новый адрес доставки применяется к партиям, спецификации на которые подписаны после\n"
        "1 октября 2024 г.\n"
        "3.3. Иные условия не меняются.\n"
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "документы компании__Договор поставки Т-9-2026.txt",
            "ДОГОВОР ПОСТАВКИ № Т-9/2026\n3.2. Исходный адрес доставки.\n",
        )
        archive.writestr(
            "документы компании__Доп соглашение к Т-9-2026.txt",
            amendment_text,
        )
    index = build_index(archive_path)
    condition = next(
        chunk for chunk in index.chunks
        if chunk.get("chunk_role") == "applicability_condition"
    )
    condition_record = condition["applicability_conditions"][0]
    assert "\n" in condition_record["source_text"]
    assert amendment_text[condition_record["start_char"]:condition_record["end_char"]] == condition_record["source_text"]
    assert condition_record["target_reference"] == "3.2"
    assert condition_record["kind"] == "specification_signed_at"
    assert evaluate_applicability(condition, "2024-10-01")["status"] == "not_applicable"
    assert evaluate_applicability(condition, "2024-10-02")["status"] == "applicable"
    other_clause = next(chunk for chunk in index.chunks if chunk.get("section_reference") == "3.3")
    assert not other_clause["applicability_conditions"]
