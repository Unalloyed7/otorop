"""Regression of repeated facts, unsolicited consequences and answer composition."""
from datetime import date

import pytest

from assistant import LegalAssistant
from conftest import FakeClient, FakeSearch, draft, intent
from execution import RunControl, controlled_run
from schemas import Claim, SearchFilters, SearchResult
from solution import render_answer
from test_answer_recovery import SequentialSearch


def check(index, *, duplicate=None, relevance="direct", **kwargs):
    return {"index": index, "status": "confirmed", "duplicate_of": duplicate, "relevance": relevance, **kwargs}


async def test_paraphrased_duplicate_keeps_proofs_and_covers_both_parts(settings, evidence):
    other = evidence.model_copy(update={"chunk_id": "second-proof"})
    first = draft(evidence)["claims"][0]
    repeated = draft(other, text="После подписания УПД на оплату есть 30 календарных дней.")["claims"][0]
    repeated["addresses"] = [1]
    client = FakeClient(settings, intent().model_copy(update={"parts": ["Срок", "Начало отсчёта"]}),
        {"status": "answer", "claims": [first, repeated]},
        {"checks": [check(1, duplicate=0), check(0)], "coverage": [
            {"part_index": 0, "status": "complete", "claim_indices": [0]},
            {"part_index": 1, "status": "complete", "claim_indices": [1]},
        ]})
    trace = {}
    answer = await LegalAssistant(FakeSearch([evidence, other]), client).answer("Срок и начало отсчёта?", trace=trace)
    assert answer.status == "answer"
    assert repeated["text"] not in answer.text
    assert {c.chunk_id for c in answer.citations} == {evidence.chunk_id, other.chunk_id}
    assert {c.claim_index for c in answer.citations} == {0}
    assert len([s for s in answer.statements if s.published]) == 1
    assert answer.verification["model_calls"] == 3
    assert any(s["stage"] == "semantic_duplicate" for s in trace["stages"])


@pytest.mark.parametrize("difference", ["period", "value", "subject", "conditions"])
async def test_duplicate_hint_cannot_merge_distinct_legal_context(settings, evidence, difference):
    other = evidence.model_copy(update={"chunk_id": "other"})
    first = draft(evidence)["claims"][0]
    second = draft(other, text="Срок оплаты по этому правилу составляет 30 календарных дней.")["claims"][0]
    if difference == "period":
        other.applicability = other.applicability.model_copy(update={"valid_from": date(2020, 1, 1)})
    elif difference == "value":
        first["value"], second["value"] = {"amount": "100"}, {"amount": "200"}
    elif difference == "subject":
        first["subject"], second["subject"] = {"employee": "Первый сотрудник"}, {"employee": "Второй сотрудник"}
    else:
        first["conditions"] = [{"text": "Первое разрешение", "key": "first", "expected": True}]
        second["conditions"] = [{"text": "Второе разрешение", "key": "second", "expected": True}]
    plan = intent(filters=SearchFilters(scenario={"facts": {"first": True, "second": True}}))
    client = FakeClient(settings, plan, {"status": "answer", "claims": [first, second]},
                        {"checks": [check(0), check(1, duplicate=0)]})
    answer = await LegalAssistant(FakeSearch([evidence, other]), client).answer("Оба правила?")
    assert answer.status == "answer" and len(answer.citations) == 2
    assert first["text"] in answer.text and second["text"] in answer.text


@pytest.mark.parametrize("broken_recovery", [False, True])
async def test_unsolicited_consequence_is_not_published_or_recovered(settings, evidence, broken_recovery):
    main = draft(evidence)["claims"][0]
    extra = draft(evidence, text="За просрочку начисляется неустойка 0,1% в день.")["claims"][0]
    extra["supports"][0]["quote"] = evidence.text.splitlines()[1]
    search = SequentialSearch(SearchResult(evidence=[evidence]),
                              RuntimeError("recovery failed") if broken_recovery else SearchResult(evidence=[evidence]))
    client = FakeClient(settings, intent(), {"status": "answer", "claims": [main, extra]},
        {"checks": [check(0), check(1, relevance="unrequested")], "coverage": [
            {"part_index": 0, "status": "complete", "claim_indices": [0, 1]},
        ]})
    answer = await LegalAssistant(search, client).answer("Когда платить? Ответственность не спрашиваю.")
    assert answer.status == ("partial" if broken_recovery else "answer")
    assert "неустойка" not in answer.text
    assert all(s.text != extra["text"] for s in answer.statements if s.published)
    assert any(s.text == extra["text"] and s.check.status == "insufficient" for s in answer.statements)


async def test_quote_rejected_after_verdict_cannot_leave_false_complete_coverage(settings, evidence):
    a = draft(evidence)["claims"][0]
    b = draft(evidence, text="Второе необходимое положение.")["claims"][0]
    verdict = {"checks": [check(0), check(1, limitations=[{
        "document_id": "missing", "chunk_id": "missing", "quote": "Недоступное подтверждение",
    }])], "coverage": [{"part_index": 0, "status": "complete", "claim_indices": [0, 1]}]}
    client = FakeClient(settings, verdict)
    plan = intent()
    control = RunControl(max_models=1)
    with controlled_run(control):
        claims, _, gaps = await LegalAssistant(FakeSearch(), client)._checked(
            [Claim.model_validate(c) for c in (a, b)], [evidence],
            SearchFilters(), plan, date.today(), {"parts": plan.parts}, {})
    assert len(claims) == 1 and gaps
    assert control.incomplete_parts == [0]


async def test_one_follow_up_question_covers_shared_unknown_fact(settings, evidence):
    question = "Когда подписан документ, к которому относятся оба вопроса?"
    plan = intent().model_copy(update={"parts": ["Какой срок", "Какой порядок"]})
    client = FakeClient(settings, plan, draft(evidence), {"supported": [0], "coverage": [
        {"part_index": 0, "status": "partial", "claim_indices": [0], "missing_info": ["Дата документа"], "needs_clarification": True},
        {"part_index": 1, "status": "missing", "missing_info": ["Уточните дату подписания"], "needs_clarification": True},
    ], "clarifications": [{"question": question, "part_indices": [0, 1]}]})
    answer = await LegalAssistant(FakeSearch([evidence]), client).answer("Два вопроса по документу")
    assert answer.status == "partial" and answer.missing_info == [question]
    assert answer.text.endswith(question) and "Подтверждена только" not in answer.text


async def test_incomplete_follow_up_plan_does_not_hide_another_gap(settings, evidence):
    plan = intent().model_copy(update={"parts": ["Срок", "Порядок"]})
    client = FakeClient(settings, plan, draft(evidence), {"supported": [0], "coverage": [
        {"part_index": 0, "status": "partial", "claim_indices": [0], "missing_info": ["Дата"]},
        {"part_index": 1, "status": "missing", "missing_info": ["Подписанная форма"]},
    ], "clarifications": [{"question": "Когда подписано?", "part_indices": [0]}]})
    answer = await LegalAssistant(FakeSearch([evidence]), client).answer("Срок и порядок?")
    assert answer.status == "partial" and answer.missing_info == ["Дата", "Подписанная форма"]


async def test_readable_body_and_compact_sources_keep_full_quotes_in_data(settings, evidence):
    a = draft(evidence, text="Оплатить можно в течение 30 календарных дней после УПД.")["claims"][0]
    b = draft(evidence, text="При задержке действует договорная неустойка с установленным пределом.")["claims"][0]
    b["supports"][0]["quote"] = evidence.text.splitlines()[1]
    client = FakeClient(settings, intent(), {"status": "answer", "claims": [a, b]}, {"supported": [0, 1]})
    answer = await LegalAssistant(FakeSearch([evidence]), client).answer("Когда платить и что при задержке?")
    rendered = render_answer(answer)
    assert f'{a["text"]} [1] {b["text"]} [2]' == answer.text
    assert evidence.document_title in rendered and "4.2" in rendered
    assert a["supports"][0]["quote"] not in rendered
    assert a["supports"][0]["quote"] == answer.citations[0].quote
    assert answer.verification["model_calls"] == 3


async def test_mixed_claim_is_excluded_even_if_its_first_clause_answers_question(settings, evidence):
    main = draft(evidence)["claims"][0]
    mixed = draft(evidence, text=main["text"] + " За просрочку начисляется неустойка.")["claims"][0]
    mixed["supports"].append({**mixed["supports"][0], "quote": evidence.text.splitlines()[1]})
    client = FakeClient(settings, intent(), {"status": "answer", "claims": [main, mixed]},
        {"checks": [check(0), check(1, excluded_fragments=["За просрочку начисляется неустойка."])],
         "coverage": [{"part_index": 0, "status": "complete", "claim_indices": [0, 1]}]})
    answer = await LegalAssistant(FakeSearch([evidence]), client).answer("Какой срок оплаты?")
    assert answer.status == "answer" and "неустойка" not in answer.text
    assert len([s for s in answer.statements if s.published]) == 1


async def test_more_complete_verified_fact_replaces_weaker_version(settings, evidence):
    first = draft(evidence, text="На оплату есть 30 календарных дней.")["claims"][0]
    fuller = draft(evidence, text="На оплату есть 30 календарных дней с даты подписания УПД.")["claims"][0]
    client = FakeClient(settings, intent(), {"status": "answer", "claims": [first, fuller]},
        {"checks": [check(0), check(1, subsumes=0)]})
    answer = await LegalAssistant(FakeSearch([evidence]), client).answer("Срок и событие отсчёта?")
    assert answer.status == "answer" and answer.text == fuller["text"] + " [1]"
    assert len([s for s in answer.statements if s.published]) == 1


async def test_follow_up_cannot_append_unchecked_legal_explanation(settings, evidence):
    question = "Когда подписан документ?"
    client = FakeClient(settings, intent(), draft(evidence), {"supported": [0], "coverage": [
        {"part_index": 0, "status": "partial", "claim_indices": [0], "needs_clarification": True,
         "missing_info": ["Дата документа"]},
    ], "clarifications": [{"question": question + " Закон требует оплаты за 90 дней.", "part_indices": [0]}]})
    answer = await LegalAssistant(FakeSearch([evidence]), client).answer("Когда платить?")
    assert answer.status == "partial" and answer.missing_info == [question]
    assert answer.text.endswith(question) and "90 дней" not in answer.text


async def test_true_but_unselected_recommendation_is_not_published(settings, evidence):
    main = draft(evidence)["claims"][0]
    extra = draft(evidence, text="Для оплаты можно настроить отдельное платёжное поручение.")["claims"][0]
    client = FakeClient(settings, intent(), {"status": "answer", "claims": [main, extra]},
        {"answer_claim_indices": [0], "checks": [check(0), check(1)],
         "coverage": [{"part_index": 0, "status": "complete", "claim_indices": [0]}]})
    answer = await LegalAssistant(FakeSearch([evidence]), client).answer("Какой срок оплаты?")
    assert answer.status == "answer" and answer.text == main["text"] + " [1]"
    assert all(s.text != extra["text"] for s in answer.statements if s.published)
