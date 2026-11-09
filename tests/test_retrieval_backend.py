from __future__ import annotations

from pathlib import Path

import pytest

from retrieval import create_search
from schemas import SearchFilters

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "knowledge-base.zip").is_file(),
    reason="Локальная база организаторов не установлена",
)


@pytest.mark.asyncio
async def test_factory_returns_schema_evidence_with_exact_source_coordinates() -> None:
    backend = create_search(ROOT / "data")
    result = await backend.search(
        "срок оплаты по договору П-17/2024",
        SearchFilters(contract_number="П-17/2024"),
    )

    clause = next(item for item in result.evidence if item.section == "4.2")
    document = backend.get_document(clause.document_id)
    assert clause.text == document["text"][
        clause.coordinates.char_start:clause.coordinates.char_end
    ]
    assert clause.contract_number == "П-17/2024"
    assert clause.coordinates.source_path == document["source_file"]
    assert clause.metadata["score"] > 0


@pytest.mark.asyncio
async def test_factory_maps_specification_date_and_preserves_amendment_source() -> None:
    backend = create_search(ROOT / "data")
    result = await backend.search(
        "новый адрес доставки П-17/2024",
        SearchFilters(
            contract_number="П-17/2024",
            event_date="2024-10-02",
            date_kind="specification_signed",
        ),
    )

    amendment = next(
        item for item in result.evidence
        if item.metadata.get("amendment_number") == 1
    )
    assert amendment.applicability.resolved
    assert amendment.applicability.date_kind == "specification_signed_at"
    assert "после 1 октября 2024 г." in amendment.applicability.conditions[0]
    assert backend.get_chunk(amendment.chunk_id)["document_id"] == amendment.document_id


@pytest.mark.asyncio
async def test_partial_counterparty_returns_distinct_candidates_without_deciding_clarification() -> None:
    backend = create_search(ROOT / "data")
    result = await backend.search(
        "гарантийный срок",
        SearchFilters(counterparty="Альфа"),
    )

    assert result.clarification is None
    assert set(result.candidates) == {"Альфа-Трейд", "Альфа-Трейдинг"}
    assert {item.counterparty for item in result.evidence} == {"Альфа-Трейд", "Альфа-Трейдинг"}
