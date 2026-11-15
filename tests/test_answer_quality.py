"""Regression of observed quality failures, independent of the model's skill."""
import asyncio
from time import perf_counter

import pytest

from assistant import LegalAssistant, _recover_quote
from conftest import FakeClient, FakeSearch, draft, intent
from schemas import Coordinates, SearchFilters
from solution import run_submission, validate_submission


async def test_known_party_is_searched_despite_intent_request_for_number(settings, evidence):
    plan = intent(filters=SearchFilters(counterparty=evidence.counterparty)).model_copy(update={
        "action": "clarify", "clarification": "Назовите номер договора.",
    })
    client = FakeClient(settings, plan, draft(evidence), {"supported": [0]})
    search = FakeSearch([evidence])
    trace = {}
    answer = await LegalAssistant(search, client).answer("Когда платить Тестовому поставщику?", trace=trace)
    assert answer.status == "answer" and answer.citations
    assert search.requests[0][1].counterparty == evidence.counterparty
    assert any(s["stage"] == "clarification_deferred" for s in trace["stages"])


async def test_ambiguous_search_still_requires_selection(settings):
    plan = intent(filters=SearchFilters(counterparty="Альфа")).model_copy(update={"action": "clarify"})
    search = FakeSearch(candidates=["Альфа-Трейд", "Альфа-Трейдинг"])
    client = FakeClient(settings, plan)
    answer = await LegalAssistant(search, client).answer("Срок оплаты Альфе?")
    assert answer.status == "clarify"
    assert len(search.requests) == len(client.requests) == 1


async def test_elided_list_is_recovered_and_checked_against_original(settings, evidence):
    source = "Испытание не устанавливается для:\nбеременных женщин;\nлиц, не достигших возраста восемнадцати лет;\nдругих лиц."
    e = evidence.model_copy(update={"text": source, "coordinates": Coordinates(source_path="rule.txt", char_start=0, char_end=len(source))})
    response = draft(e, text="Испытание для несовершеннолетнего устанавливать нельзя.")
    response["claims"][0]["supports"][0]["quote"] = "Испытание не устанавливается для лиц, не достигших возраста восемнадцати лет."
    client = FakeClient(settings, intent(), response, {"supported": [0]})
    trace = {}
    result = await LegalAssistant(FakeSearch([e]), client).answer("Допустимо ли испытание несовершеннолетнего?", trace=trace)
    assert result.status == "answer"
    citation = result.citations[0]
    assert source[citation.coordinates.char_start:citation.coordinates.char_end] == citation.quote
    assert "беременных женщин" in citation.quote  # real intervening source text, not a stitched quote
    assert client.requests[2][1]["claims"][0]["supports"][0]["quote"] == citation.quote
    assert any(s["stage"] == "quote_recovery" for s in trace["stages"])


@pytest.mark.parametrize("quote", [
    "Оплата производится в течение 90 календарных дней после подписания УПД.",
    "Оплата производится в течение 30 рабочих дней после подписания УПД.",
    "Оплата производится в течение 0.1 календарных дней после подписания УПД.",
])
def test_quote_recovery_does_not_substitute_words_numbers_or_units(quote):
    assert _recover_quote("Оплата производится в течение 30 календарных дней после подписания УПД.", quote) is None


async def test_recovered_citation_does_not_confirm_false_claim(settings, evidence):
    source = "Испытание не устанавливается для:\nбеременных женщин;\nлиц, не достигших возраста восемнадцати лет."
    e = evidence.model_copy(update={"text": source})
    response = draft(e, text="Испытание для несовершеннолетнего допускается.")
    response["claims"][0]["supports"][0]["quote"] = "Испытание не устанавливается для лиц, не достигших возраста восемнадцати лет."
    client = FakeClient(settings, intent(), response, {"supported": []})
    result = await LegalAssistant(FakeSearch([e]), client).answer("Можно установить испытание?")
    assert result.status == "insufficient" and not result.citations
    assert len(client.requests) == 3


def coverage(status, claims, *, available=False):
    return [{"part_index": 0, "status": status, "claim_indices": claims,
             "missing_info": [] if status == "complete" else ["Основная ставка"],
             "evidence_available": available}]


async def test_exception_only_is_not_a_complete_answer_and_gets_revision(settings, evidence):
    source = "1. При экспорте налоговая ставка составляет 0 процентов.\n7. По заявлению можно применять другие ставки."
    e = evidence.model_copy(update={"text": source, "section": "Статья о ставках"})
    partial = draft(e, text="По заявлению можно применять другие ставки.")
    partial["claims"][0]["supports"][0]["quote"] = source.splitlines()[1]
    fixed = draft(e, text="Основная ставка при экспорте — 0 процентов.")
    client = FakeClient(settings, intent().model_copy(update={"parts": ["Основная ставка при экспорте"]}),
        partial, {"supported": [0], "coverage": coverage("partial", [0], available=True)},
        fixed, {"supported": [0, 1], "coverage": coverage("complete", [1])})
    answer = await LegalAssistant(FakeSearch([e]), client).answer("Какая ставка при экспорте?")
    assert answer.status == "answer" and "0 процентов" in answer.text and not answer.missing_info
    assert answer.verification["repair_calls"] == 1
    assert len(client.requests) == 5
    assert client.requests[3][1]["repair"]["incomplete_parts"] == ["Основная ставка при экспорте"]


async def test_supported_claims_cannot_override_incomplete_coverage(settings, evidence):
    client = FakeClient(settings, intent(), draft(evidence),
                        {"supported": [0], "coverage": coverage("partial", [0])})
    answer = await LegalAssistant(FakeSearch([evidence]), client).answer("Нужны все сведения")
    assert answer.status == "partial" and "Основная ставка" in answer.missing_info
    assert answer.citations


async def test_missing_coverage_is_repaired_once_not_silently_accepted(settings, evidence):
    client = FakeClient(settings, intent(), draft(evidence),
                        {"supported": [0], "coverage": []},
                        {"supported": [0], "coverage": coverage("complete", [0])})
    answer = await LegalAssistant(FakeSearch([evidence]), client).answer("Срок оплаты?")
    assert answer.status == "answer" and len(client.requests) == 4
    assert "coverage" in client.requests[3][1]["repair"]["error"]


async def test_cold_initialization_is_within_total_submission_budget(tmp_path, settings, monkeypatch):
    import solution
    (tmp_path / "questions.csv").write_text("id,question\n1,Срок оплаты?\n", encoding="utf-8")
    monkeypatch.setenv("RUN_TIMEOUT", ".03")
    monkeypatch.setattr(solution.Settings, "from_env", lambda: settings)
    class Client:
        def __init__(self, settings): self.settings = settings
        async def aclose(self): pass
    monkeypatch.setattr(solution, "LLMClient", Client)
    async def slow_index(client):
        await asyncio.sleep(5)
    monkeypatch.setattr(solution, "build_search", slow_index)
    started = perf_counter()
    result = await asyncio.wait_for(run_submission(tmp_path, tmp_path / "submission.csv"), timeout=.5)
    assert perf_counter() - started < .5 and result == {"error": 1}
    validate_submission(tmp_path / "submission.csv", ["1"])


async def test_queue_only_receives_remaining_budget_after_index(tmp_path, settings, monkeypatch):
    import solution
    (tmp_path / "questions.csv").write_text("id,question\n1,Срок оплаты?\n", encoding="utf-8")
    monkeypatch.setenv("RUN_TIMEOUT", ".2")
    monkeypatch.setattr(solution.Settings, "from_env", lambda: settings)
    class Client:
        def __init__(self, settings): self.settings = settings
        async def aclose(self): pass
    monkeypatch.setattr(solution, "LLMClient", Client)
    async def index(client):
        await asyncio.sleep(.04)
        return FakeSearch()
    async def process(rows, engine, workers, budget, **kwargs):
        assert 0 < budget < .18
        return [("out_of_scope", "Тестовый ответ")]
    monkeypatch.setattr(solution, "build_search", index)
    monkeypatch.setattr(solution, "_process", process)
    result = await run_submission(tmp_path, tmp_path / "submission.csv")
    assert result == {"out_of_scope": 1}


async def test_verifier_can_correct_drafter_part_assignment(settings, evidence):
    plan = intent().model_copy(update={"parts": ["Срок оплаты", "От какого события считать срок"]})
    client = FakeClient(settings, plan, draft(evidence), {"supported": [0], "coverage": [
        {"part_index": 0, "status": "complete", "claim_indices": [0]},
        {"part_index": 1, "status": "complete", "claim_indices": [0]},
    ]})
    answer = await LegalAssistant(FakeSearch([evidence]), client).answer("Срок и начало отсчёта?")
    assert answer.status == "answer" and answer.verification["model_calls"] == 3


async def test_verifier_elided_redundant_reference_does_not_erase_confirmed_fact(settings, evidence):
    response = draft(evidence)
    ref = {**response["claims"][0]["supports"][0], "quote": "Отсутствующий в этом фрагменте заголовок: ... 30 календарных дней"}
    client = FakeClient(settings, intent(), response, {"checks": [
        {"index": 0, "status": "confirmed", "limitations": [ref]},
    ]})
    trace = {}
    answer = await LegalAssistant(FakeSearch([evidence]), client).answer("Срок оплаты?", trace=trace)
    assert answer.status == "answer" and answer.verification["repair_calls"] == 0
    assert any(s["stage"] == "auxiliary_quote_recovery" for s in trace["stages"])
    assert all(c.quote in evidence.text for c in answer.statements[0].limitations)


async def test_limitations_search_rechecks_existing_answer_without_repeating_generation(settings, evidence):
    extra = evidence.model_copy(update={"chunk_id": "context:extra", "text": "Дополнительные положения об ответственности."})
    class Search:
        async def search(self, query, filters):
            from schemas import SearchResult
            return SearchResult(evidence=[evidence] if filters.purpose == "grounds" else [evidence, extra])
    client = FakeClient(settings, intent(), draft(evidence), {"supported": [0]}, {"supported": [0]})
    result = await LegalAssistant(Search(), client).answer("Срок оплаты?")
    assert result.status == "answer" and len(result.citations) == 1
    assert [c[2].__name__ for c in client.requests] == ["Intent", "Draft", "Verdict", "Verdict"]


async def test_foreign_edition_as_ground_gets_bounded_repair(settings, evidence):
    from datetime import date
    old = evidence.model_copy(update={"metadata": {"document_type": "internal_policy"},
        "applicability": evidence.applicability.model_copy(update={"valid_to": date(2025, 3, 1), "date_kind": "event"})})
    new = old.model_copy(update={"document_id": "new", "chunk_id": "new:1",
        "applicability": old.applicability.model_copy(update={"valid_from": date(2025, 3, 1), "valid_to": None})})
    plan = intent(filters=SearchFilters(event_date="2025-02-20", date_kind="contract_signed"), temporal_mode="historical")
    client = FakeClient(settings, plan, draft(new), draft(old), {"supported": [0]})
    result = await LegalAssistant(FakeSearch([old, new]), client).answer("Какое правило в феврале 2025?")
    assert result.status == "answer" and result.citations[0].document_id == old.document_id
    assert result.verification["repair_calls"] == 1
