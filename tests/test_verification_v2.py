from datetime import date
import json
import asyncio

import httpx
import pytest

from assistant import LegalAssistant, validate_supports, CitationError
from conftest import FakeClient, FakeSearch, draft, intent
from date_context import parse_date_span, range_changes
from diagnostics import capture_run
from llm_client import LLMClient, ModelAPIError
from schemas import Coordinates, Draft, ScenarioDates, SearchFilters, SearchResult


@pytest.mark.parametrize("value,start,end,precision", [
    ("август 2024", "2024-08-01", "2024-09-01", "month"),
    ("в феврале 2024", "2024-02-01", "2024-03-01", "month"),
    ("2024", "2024-01-01", "2025-01-01", "year"),
    ("2024-08-02..2024-08-17", "2024-08-02", "2024-08-18", "range"),
    ("2024-02-29", "2024-02-29", "2024-03-01", "day"),
    ("март 2024", "2024-03-01", "2024-04-01", "month"),
    ("в мае 2024", "2024-05-01", "2024-06-01", "month"),
    ("2024 год", "2024-01-01", "2025-01-01", "year"),
    ("12.11.2026", "2026-11-12", "2026-11-13", "day"),
    ("12 ноября 2026", "2026-11-12", "2026-11-13", "day"),
    ("2024-08..2024-09", "2024-08-01", "2024-10-01", "range"),
])
def test_date_precision_is_an_interval(value, start, end, precision):
    span = parse_date_span(value)
    assert (str(span.start), str(span.end_exclusive), span.precision) == (start, end, precision)


def policy(evidence, year, amount, start=None, end=None):
    text = f"2.4. Договоры свыше {amount} рублей одобряет директор."
    return evidence.model_copy(update={"document_id": f"policy-{year}", "chunk_id": f"policy-{year}:2.4", "contract_number": None,
        "document_title": "Регламент согласования", "section": "2.4", "text": text,
        "coordinates": Coordinates(source_path=f"policy-{year}.txt", char_start=0, char_end=len(text)),
        "metadata": {"document_type": "internal_policy"},
        "applicability": evidence.applicability.model_copy(update={"edition": str(year), "valid_from": date.fromisoformat(start) if start else None, "valid_to": date.fromisoformat(end) if end else None, "date_kind": "event"})})


async def test_same_rule_during_month_does_not_require_a_day(settings, evidence):
    old = policy(evidence, 2023, "10 000 000", end="2025-03-01")
    newer = policy(evidence, 2025, "15 000 000", start="2025-03-01")
    filters = SearchFilters(scenario=ScenarioDates(contract_signed="август 2024"), governing_date="contract_signed")
    response = draft(old, text="Договор свыше 10 000 000 рублей требует одобрения директора.")
    client = FakeClient(settings, intent(filters=filters, temporal_mode="historical"), response, {"supported": [0]})
    answer = await LegalAssistant(FakeSearch([old, newer]), client).answer("Заключили в августе 2024", scenario={})
    assert answer.status == "answer" and "Уточните" not in answer.text
    assert answer.scenario.contract_signed.precision == "month"
    assert answer.statements[0].check.status == "confirmed"


async def test_changed_rule_inside_range_requires_clarification(settings, evidence):
    old = policy(evidence, 2023, "10 000 000", end="2025-03-01")
    new = policy(evidence, 2025, "15 000 000", start="2025-03-01")
    filters = SearchFilters(scenario=ScenarioDates(contract_signed="2025-02-01..2025-03-31"), governing_date="contract_signed")
    client = FakeClient(settings, intent(filters=filters), draft(old, text="Порог — 10 000 000 рублей."))
    answer = await LegalAssistant(FakeSearch([old, new]), client).answer("Подписали в феврале или марте")
    assert answer.status == "clarify" and not answer.citations
    assert any("Уточните дату" in r for s in answer.statements for r in s.check.reasons)


def test_identical_rule_across_editions_needs_both_citations(evidence):
    old = policy(evidence, 2023, "10 000 000", end="2025-03-01")
    new = policy(evidence, 2025, "10 000 000", start="2025-03-01")
    filters = SearchFilters(scenario=ScenarioDates(event="2025-02-01..2025-03-31"))
    response = draft(old, text="Порог — 10 000 000 рублей.")
    response["claims"][0]["supports"].append({"document_id": new.document_id, "chunk_id": new.chunk_id, "quote": new.text})
    assert len(validate_supports(Draft.model_validate(response), [old, new], filters, ["порог"])) == 2
    response["claims"][0]["supports"].pop()
    with pytest.raises(CitationError):
        validate_supports(Draft.model_validate(response), [old, new], filters, ["порог"])


def test_specification_range_is_not_replaced_by_contract_date(evidence):
    e = evidence.model_copy(update={"metadata": {"applicability_conditions": [{"kind": "specification_signed_at", "operator": "after", "date": "2024-10-01"}]}})
    filters = SearchFilters(scenario=ScenarioDates(contract_signed="2024-08-01", specification_signed="2024-10"))
    assert any("спецификации" in issue for issue in range_changes([e], filters))
    filters.scenario.specification_signed = parse_date_span("2024-10-02")
    assert not range_changes([e], filters)


async def test_one_format_repair_has_concrete_errors_and_same_engine(settings, evidence):
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        data = json.loads(payload["messages"][1]["content"])
        prompt = payload["messages"][0]["content"]
        requests.append(data)
        if "Разбери последнюю" in prompt:
            if "repair" not in data:
                reply = {"action": "search", "query": "оплата", "parts": ["оплата"], "filters": {"unknown": None}}
            else:
                assert data["repair"]["details"][0]["path"] == ["filters", "unknown"]
                reply = {"action": "search", "query": "оплата", "parts": ["оплата"]}
        elif "Независимо проверь" in prompt:
            reply = {"supported": [0]}
        else:
            e = data["evidence"][0]
            reply = draft(evidence)
            reply["claims"][0]["supports"][0].update(document_id=e["document_id"], chunk_id=e["chunk_id"])
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)}, "finish_reason": "stop"}]})
    async with LLMClient(settings, transport=httpx.MockTransport(handler)) as client:
        with capture_run() as trace:
            answer = await LegalAssistant(FakeSearch([evidence]), client).answer("Срок?", trace=trace)
    assert answer.status == "answer"
    assert len(requests) == 4 and answer.verification["repair_calls"] == 1
    assert [o["valid"] for o in trace["model_outputs"]] == [False, True, True, True]


async def test_verified_part_survives_failed_reference_repair(settings, evidence):
    response = draft(evidence)
    invalid = json.loads(json.dumps(response["claims"][0]))
    invalid["addresses"] = [1]
    invalid["supports"][0]["chunk_id"] = "missing"
    response["claims"].append(invalid)
    client = FakeClient(settings, intent().model_copy(update={"parts": ["оплата", "гарантия"]}), response, {"supported": [0]}, ModelAPIError("safe"))
    answer = await LegalAssistant(FakeSearch([evidence]), client).answer("Оплата и гарантия?")
    assert answer.status == "partial" and answer.citations
    assert answer.verification["repair_calls"] == 1
    assert any(s.published for s in answer.statements)
    assert any(not s.published and s.check.status == "insufficient" for s in answer.statements)


async def test_general_rule_exception_amendment_conditional_result(settings, evidence):
    def fragment(ident, text, kind):
        return evidence.model_copy(update={"document_id": ident, "chunk_id": ident+":1", "text": text,
            "coordinates": Coordinates(source_path=ident+".txt", char_start=0, char_end=len(text)), "metadata": {"document_type": kind}})
    general = fragment("general", "Договор согласуется с юридическим отделом.", "internal_policy")
    exception = fragment("exception", "При типовой форме без изменений договор согласуется только с главным бухгалтером.", "internal_policy")
    amendment = fragment("amendment", "Для договора Т-1/2026 применяется согласование по типовой форме.", "amendment")
    class Search:
        calls = []
        async def search(self, query, filters):
            self.calls.append(filters.purpose)
            return SearchResult(evidence=[general] if filters.purpose == "grounds" else [general, exception, amendment])
    first = draft(general, text=general.text)
    conditional = draft(exception, text="Если используется типовая форма без изменений, договор согласуется только с главным бухгалтером.")
    conditional["claims"][0]["conditions"] = [{"text": "Типовая форма без изменений", "key": "standard_form_unchanged", "expected": True}]
    ref = lambda e: {"document_id": e.document_id, "chunk_id": e.chunk_id, "quote": e.text}
    client = FakeClient(settings, intent(), first, {"supported": [0]}, conditional,
        {"checks": [{"index": 0, "status": "contradiction", "reasons": ["Есть специальное исключение"], "limitations": [ref(exception)]}, {"index": 1, "status": "conditional", "limitations": [ref(amendment)]}]})
    search = Search()
    answer = await LegalAssistant(search, client).answer("Кто согласует?")
    assert search.calls == ["grounds", "limitations"]
    assert answer.status == "partial"
    assert "Если используется" in answer.text and "Договор согласуется с юридическим отделом." not in answer.text
    assert {s.check.status for s in answer.statements} == {"contradiction", "conditional"}
    published = next(s for s in answer.statements if s.published)
    assert published.conditions[0].status == "unknown"
    assert published.limitations[0].role == "amendment"
    assert answer.verification["limitations_search_done"]


@pytest.mark.parametrize("field,value", [("duration", "90"), ("day_type", "working")])
async def test_structured_value_is_checked_by_code_even_when_text_is_correct(settings, evidence, field, value):
    response = draft(evidence)
    response["claims"][0]["value"] = {field: value}
    answer = await LegalAssistant(FakeSearch([evidence]), FakeClient(settings, intent(), response)).answer("Срок?")
    assert answer.status == "insufficient" and not answer.citations
    assert not answer.statements[0].check.mechanical_pass


async def test_format_repair_is_per_request_not_shared_across_concurrent_answers(settings, evidence):
    seen = {}
    async def handler(request):
        await asyncio.sleep(.001)
        body = json.loads(request.content)
        data = json.loads(body["messages"][1]["content"])
        question = data["history"][-1]["content"]
        system = body["messages"][0]["content"]
        if "Разбери последнюю" in system:
            seen[question] = seen.get(question, 0) + 1
            reply = {"action": "search", "query": "оплата", "parts": ["оплата"]}
            if seen[question] == 1:
                reply["unknown"] = None
        elif "Независимо проверь" in system:
            assert data["scenario"]["facts"]["case"] == question
            reply = {"supported": [0]}
        else:
            e = data["evidence"][0]
            reply = draft(evidence)
            reply["claims"][0]["supports"][0].update(document_id=e["document_id"], chunk_id=e["chunk_id"])
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)}}]})
    async with LLMClient(settings, transport=httpx.MockTransport(handler)) as client:
        engine = LegalAssistant(FakeSearch([evidence]), client)
        results = await asyncio.gather(*(engine.answer(q, scenario={"facts": {"case": q}}) for q in ["А", "Б"]))
    assert all(r.status == "answer" and r.verification["repair_calls"] == 1 for r in results)
    assert [r.scenario.facts["case"] for r in results] == ["А", "Б"]
    assert seen == {"А": 2, "Б": 2}


async def test_budget_preserves_prior_verified_part(settings, evidence):
    extra = evidence.model_copy(update={"chunk_id": "new", "text": "Другие дополнительные условия."})
    class Search:
        calls = 0
        async def search(self, query, filters):
            self.calls += 1
            return SearchResult(evidence=[evidence] if self.calls == 1 else [extra])
    client = FakeClient(settings.model_copy(update={"max_model_calls": 3}), intent(), draft(evidence), {"supported": [0]})
    answer = await LegalAssistant(Search(), client).answer("Срок?")
    assert answer.status == "partial" and answer.citations
    assert answer.verification["model_calls"] == 3
    assert any("лимит" in gap.lower() for gap in answer.missing_info)


async def test_deadline_preserves_already_checked_statement(settings, evidence):
    class SlowLimitations:
        async def search(self, query, filters):
            if filters.purpose == "limitations":
                await asyncio.Event().wait()
            from schemas import SearchResult
            return SearchResult(evidence=[evidence])

    client = FakeClient(settings.model_copy(update={"question_timeout": .05}), intent(), draft(evidence), {"supported": [0]})
    trace = {"stages": []}
    answer = await LegalAssistant(SlowLimitations(), client).answer("Срок оплаты?", trace=trace)
    assert answer.status == "partial" and answer.citations
    assert answer.statements[0].published and answer.statements[0].check.semantic_checked
    assert "прервана" in answer.text
    assert any(s.get("kind") == "TimeoutError" for s in trace["stages"])


async def test_json_schema_instead_of_instance_only_gets_one_repair(settings):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"type": "object", "properties": {}})}}]})
    async with LLMClient(settings, transport=httpx.MockTransport(handler)) as client:
        answer = await LegalAssistant(FakeSearch(), client).answer("Вопрос?")
    assert answer.status == "error" and len(requests) == 2
    assert answer.verification["repair_calls"] == 1
