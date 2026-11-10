from datetime import date
import zipfile

from documents import load_documents
from retrieval import create_search
from schemas import SearchFilters
from temporal import rule_periods


def test_paragraph_expiry_does_not_expire_whole_law(tmp_path):
    with zipfile.ZipFile(tmp_path / "knowledge-base.zip", "w") as archive:
        archive.writestr("законы__НК_ст999_действующая.txt", "Источник: Статья 999. НК РФ\nДата выгрузки: 10.10.2026\n1. Ставка составляет 25 процентов.\n2. Утратил силу с 1 января 2023 года.\n3. Другое правило действует до 31 декабря 2026 года.\n")
    doc = load_documents(tmp_path / "knowledge-base.zip")[0]
    assert doc["valid_to"] is None
    assert doc["provision_date_sources"][0]["date"] == "2023-01-01"
    assert all(x["kind"] != "valid_to" for x in doc["date_sources"])


async def test_current_law_is_retrievable_on_date_different_from_snapshot(tmp_path):
    with zipfile.ZipFile(tmp_path / "knowledge-base.zip", "w") as archive:
        archive.writestr("законы__НК_ст999_действующая.txt", "Источник: Статья 999. НК РФ\nДата выгрузки: 10.10.2026\n1. Ставка составляет 25 процентов.\n2. Утратил силу с 1 января 2023 года.\n")
    result = await create_search(tmp_path).search("налоговая ставка", SearchFilters(event_date="2026-11-10", date_kind="event"))
    assert result.evidence
    assert any("25 процентов" in e.text for e in result.evidence)
    assert any(e.metadata["provision_date_sources"] for e in result.evidence)


def test_snapshot_does_not_create_rule_period():
    assert rule_periods("Дата выгрузки: 10.10.2026") == []
    assert rule_periods("Пункт изменен с 1 января 2026 года") == []


def test_explicit_end_is_exclusive_after_inclusive_po():
    periods = rule_periods("Правило применяется с 01.01.2025 по 31.12.2025 включительно.")
    assert periods == [{"valid_from": "2025-01-01", "valid_to": "2026-01-01", "source_text": "Правило применяется с 01.01.2025 по 31.12.2025 включительно."}]
