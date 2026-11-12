from collections import deque
from datetime import date

import pytest

from assistant import LegalAssistant
from conftest import FakeClient, FakeSearch, draft, intent
from schemas import Coordinates, SearchFilters, SearchResult


class SequentialSearch:
    def __init__(self, *results):
        self.results = deque(results)
        self.requests = []

    async def search(self, query, filters):
        self.requests.append((query, filters.model_dump(mode="json")))
        response = self.results.popleft()
        if isinstance(response, Exception): raise response
        return response


def claim(evidence, text=None, part=0):
    c = draft(evidence, **({"text": text} if text else {}))["claims"][0]
    c["addresses"] = [part]
    return c


def warranty(evidence):
    text = "5.2. Гарантийный срок составляет 24 месяца с даты поставки."
    return evidence.model_copy(update={"chunk_id": "warranty:1", "text": text, "section": "5.2", "coordinates": Coordinates(source_path="contract.txt", char_start=200, char_end=200+len(text))})


@pytest.mark.parametrize("invalid", ["chunk", "document", "quote", "number"])
async def test_one_bad_claim_does_not_erase_valid_claim(settings, evidence, invalid):
    good = claim(evidence)
    bad = claim(evidence, "Оплата составляет 90 дней.", part=1)
    if invalid == "chunk": bad["supports"][0]["chunk_id"] = "unknown"
    if invalid == "document": bad["supports"][0]["document_id"] = "another"
    if invalid == "quote": bad["supports"][0]["quote"] = "не существует"
    client = FakeClient(settings, intent().model_copy(update={"parts": ["Оплата", "Второй вывод"]}), {"status": "answer", "claims": [good, bad]}, {"supported": [0]})
    result = await LegalAssistant(FakeSearch([evidence]), client).answer("Два вопроса")
    assert result.status == "partial" and len(result.citations) == 1
    assert "30 календарных" in result.text and "90 дней" not in result.text
    assert result.citations[0].claim_index == 0
    assert any("Второй вывод" in x for x in result.missing_info)


async def test_invalid_first_claim_and_verifier_reindex(settings, evidence):
    bad = claim(evidence, "Не подтверждено", part=0)
    bad["supports"][0]["chunk_id"] = "unknown"
    good = claim(evidence, part=1)
    client = FakeClient(settings, intent().model_copy(update={"parts": ["Первый", "Оплата"]}), {"status": "answer", "claims": [bad, good]}, {"supported": [0]})
    answer = await LegalAssistant(FakeSearch([evidence]), client).answer("Два вопроса")
    assert answer.status == "partial"
    assert "[1]" in answer.text and "[2]" not in answer.text
    assert answer.citations[0].claim_index == 0
    assert len(client.requests[2][1]["claims"]) == 1


async def test_recovery_adds_new_claim_without_losing_verified_one(settings, evidence):
    extra = warranty(evidence)
    search = SequentialSearch(SearchResult(evidence=[evidence]), SearchResult(evidence=[extra]))
    plan = intent(filters=SearchFilters(contract_number="Т-1/2026")).model_copy(update={"parts": ["Оплата", "Гарантия"]})
    client = FakeClient(settings, plan,
        {"status": "partial", "claims": [claim(evidence)], "missing_info": ["Гарантийное условие"]}, {"supported": [0]},
        {"status": "answer", "claims": [claim(extra, "Гарантийный срок составляет 24 месяца с даты поставки.", part=1)]}, {"supported": [0, 1]})
    trace = {}
    result = await LegalAssistant(search, client).answer("Оплата и гарантия?", trace=trace)
    assert result.status == "answer" and len(result.citations) == 2
    assert "30" in result.text and "24" in result.text and not result.missing_info
    assert len(search.requests) == 2 and len(client.requests) == 5
    assert search.requests[1][1]["purpose"] == "limitations"
    assert search.requests[1][1]["anchor_document_ids"] == [evidence.document_id]
    assert search.requests[0][1]["contract_number"] == search.requests[1][1]["contract_number"]
    assert search.requests[0][0] != search.requests[1][0]
    assert "Гарантия" in search.requests[1][0]
    assert client.requests[3][1]["previous_claims"]
    assert len([s for s in trace["stages"] if s["stage"] == "search"]) == 2


async def test_empty_first_search_recovers(settings, evidence):
    search = SequentialSearch(SearchResult(), SearchResult(evidence=[evidence]))
    client = FakeClient(settings, intent(), draft(evidence), {"supported": [0]})
    result = await LegalAssistant(search, client).answer("Срок?")
    assert result.status == "answer" and len(search.requests) == 2


async def test_search_retry_can_be_disabled(settings):
    search = SequentialSearch(SearchResult())
    client = FakeClient(settings.model_copy(update={"search_retries": 0}), intent())
    result = await LegalAssistant(search, client).answer("Срок?")
    assert result.status == "insufficient" and len(search.requests) == 1


async def test_no_new_chunks_means_no_second_generation(settings, evidence):
    search = FakeSearch([evidence])
    client = FakeClient(settings, intent(), draft(evidence, status="partial", missing_info=["Гарантия"]), {"supported": [0]})
    result = await LegalAssistant(search, client).answer("Вопрос")
    assert result.status == "partial"
    assert len(search.requests) == 2 and len(client.requests) == 3


async def test_retry_error_preserves_confirmed_part_as_technical_partial(settings, evidence):
    search = SequentialSearch(SearchResult(evidence=[evidence]), RuntimeError("PRIVATE_API_KEY"))
    client = FakeClient(settings, intent(), draft(evidence, status="partial", missing_info=["Гарантия"]), {"supported": [0]})
    trace = {}
    result = await LegalAssistant(search, client).answer("Вопрос", trace=trace)
    assert result.status == "partial" and result.citations
    assert "Техническая ошибка" in result.text and "PRIVATE_API_KEY" not in str(trace)
    assert any(s["stage"] == "error" for s in trace["stages"])


async def test_semantic_rejection_preserves_other_claim(settings, evidence):
    extra = warranty(evidence)
    client = FakeClient(settings, intent().model_copy(update={"parts": ["Оплата", "Гарантия"]}),
        {"status": "answer", "claims": [claim(evidence), claim(extra, "Гарантия составляет 24 месяца с даты поставки.", 1)]}, {"supported": [1]})
    result = await LegalAssistant(FakeSearch([evidence, extra]), client).answer("Вопрос")
    assert result.status == "partial"
    assert "24 месяца" in result.text and "30 календарных" not in result.text
    assert all(c.claim_index == 0 for c in result.citations)


async def test_recovery_is_bounded_to_one_search(settings):
    search = SequentialSearch(SearchResult(), SearchResult())
    result = await LegalAssistant(search, FakeClient(settings, intent())).answer("Вопрос")
    assert result.status == "insufficient" and len(search.requests) == 2


def law(evidence, text):
    return evidence.model_copy(update={"text": text, "coordinates": Coordinates(source_path="law.txt", char_start=0, char_end=len(text)), "metadata": {"document_type": "law"}, "applicability": evidence.applicability.model_copy(update={"current_only": True, "snapshot_date": date(2026, 10, 10)})})


async def test_current_law_does_not_require_event_to_equal_snapshot(settings, evidence):
    e = law(evidence, "1. Ставка составляет 25 процентов.")
    plan = intent(filters=SearchFilters(event_date="2026-11-10", date_kind="event"), temporal_mode="current")
    result = await LegalAssistant(FakeSearch([e]), FakeClient(settings, plan, draft(e, text="Ставка составляет 25 процентов."), {"supported": [0]})).answer("Какая ставка сейчас?")
    assert result.status == "answer" and result.citations


@pytest.mark.parametrize("when", ["2024-01-01", "2026-10-10"])
async def test_snapshot_is_not_proof_of_historical_period(settings, evidence, when):
    e = law(evidence, "1. Ставка составляет 25 процентов.")
    client = FakeClient(settings, intent(filters=SearchFilters(event_date=when, date_kind="event"), temporal_mode="historical"))
    result = await LegalAssistant(FakeSearch([e]), client).answer("Историческая редакция?")
    assert result.status == "insufficient" and len(client.requests) == 1


async def test_explicit_historical_rule_period_can_be_used(settings, evidence):
    e = law(evidence, "Статья 1. Установить минимальный размер оплаты труда с 1 января 2026 года в сумме 27 093 рубля в месяц.")
    plan = intent(filters=SearchFilters(event_date="2026-09-01", date_kind="event"), temporal_mode="historical")
    client = FakeClient(settings, plan, draft(e, text="С 1 января 2026 года размер составляет 27 093 рубля в месяц."), {"supported": [0]})
    result = await LegalAssistant(FakeSearch([e]), client).answer("На сентябрь 2026?")
    assert result.status == "answer" and "1 января 2026" in result.citations[0].quote


async def test_historical_period_must_be_in_quote_too(settings, evidence):
    e = law(evidence, "Статья 1. Установить размер с 1 января 2026 года в сумме 27 093 рубля в месяц.")
    response = draft(e, text="Размер составляет 27 093 рубля в месяц.")
    response["claims"][0]["supports"][0]["quote"] = "27 093 рубля в месяц"
    client = FakeClient(settings, intent(filters=SearchFilters(event_date="2026-09-01", date_kind="event"), temporal_mode="historical"), response)
    result = await LegalAssistant(FakeSearch([e]), client).answer("На сентябрь 2026?")
    assert result.status == "insufficient" and not result.citations


async def test_future_provision_is_not_applied_as_current(settings, evidence):
    e = law(evidence, "С 1 января 2099 года ставка составляет 25 процентов.")
    client = FakeClient(settings, intent(temporal_mode="current"))
    result = await LegalAssistant(FakeSearch([e]), client).answer("Какая ставка сейчас?")
    assert result.status == "insufficient" and len(client.requests) == 1


async def test_future_not_inferred_from_open_ended_past_rule(settings, evidence):
    e = law(evidence, "С 1 января 2026 года ставка составляет 25 процентов.")
    client = FakeClient(settings, intent(filters=SearchFilters(event_date="2099-01-01", date_kind="event"), temporal_mode="future"))
    result = await LegalAssistant(FakeSearch([e]), client).answer("Какая ставка в 2099 году?")
    assert result.status == "insufficient"


async def test_explicit_year_period_is_usable(settings, evidence):
    e = law(evidence, "В 2025 году ставка составляет 12 процентов.")
    client = FakeClient(settings, intent(filters=SearchFilters(event_date="2025-08-01", date_kind="event"), temporal_mode="historical"), draft(e, text="В 2025 году ставка составляет 12 процентов."), {"supported": [0]})
    result = await LegalAssistant(FakeSearch([e]), client).answer("Какая ставка в августе 2025?")
    assert result.status == "answer"


async def test_known_part_with_user_clarification_still_scans_limitations(settings, evidence):
    search = SequentialSearch(SearchResult(evidence=[evidence]), SearchResult(evidence=[evidence]))
    client = FakeClient(settings, intent(), draft(evidence, status="clarify", missing_info=["Дата подписания спецификации"]), {"supported": [0]})
    result = await LegalAssistant(search, client).answer("Вопрос")
    assert result.status == "partial" and result.citations
    assert len(search.requests) == 2
    assert search.requests[1][1]["purpose"] == "limitations"
