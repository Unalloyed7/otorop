"""Filter invariants: document class, not words in a counterparty description."""
from pathlib import Path
import pytest
from retrieval import create_search
from schemas import SearchFilters


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("OTOROP_KNOWLEDGE_CACHE_DIR", str(tmp_path / "cache"))
    for index in (1, 2):
        (tmp_path / f"Договор {index}.txt").write_text(
            f"ДОГОВОР ПОСТАВКИ № Р-{index}/2026\nООО «Клиент {index}» и ООО «Владелец»\n"
            "1.1. Проверка поставки и оплата в течение 30 дней.\n", encoding="utf-8")
    (tmp_path / "Регламент.txt").write_text(
        "РЕГЛАМЕНТ СОГЛАСОВАНИЯ ДОГОВОРОВ\nРедакция от 01.01.2025. Действует с 01.01.2025.\n"
        "1.1. Проверку договоров проводит служба контроля в течение 3 рабочих дней.\n", encoding="utf-8")
    (tmp_path / "Письмо.txt").write_text(
        "Письмо службы контроля от 01.01.2025\n"
        "Проверка договоров требуется независимо от суммы сделки.\n", encoding="utf-8")
    return create_search(tmp_path)


@pytest.mark.parametrize("party", ["Клиент 1", "Клиент 2", "Владелец", "Неназванный покупатель", "Компания-75319"])
async def test_party_cannot_remove_normative_documents(catalog, party):
    found = await catalog.search("проверка договоров службой контроля", SearchFilters(counterparty=party, document_scope="policy"))
    assert found.evidence
    assert {e.metadata["document_type"] for e in found.evidence} <= {"internal_policy", "letter"}
    assert any("3 рабочих дней" in e.text for e in found.evidence)
    assert not found.candidates


async def test_unknown_entity_does_not_select_some_other_contract(catalog):
    found = await catalog.search("проверка и оплата поставки", SearchFilters(counterparty="Несуществующий клиент"))
    assert all(e.metadata["document_type"] not in {"agreement", "amendment"} for e in found.evidence)
    assert any(e.metadata["entity_resolution"]["counterparty"] == "not_found" for e in found.evidence)


async def test_confirmed_party_restricts_only_contractual_sources(catalog):
    found = await catalog.search("проверка и оплата поставки", SearchFilters(counterparty="Клиент 1"))
    contracts = [e for e in found.evidence if e.metadata["document_type"] == "agreement"]
    assert contracts and all(e.contract_number == "Р-1/2026" for e in contracts)
    assert any(e.metadata["document_type"] == "internal_policy" for e in found.evidence)
