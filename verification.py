"""Проверяемые утверждения: источники, условия и результат независимой проверки."""
from __future__ import annotations

import hashlib
import re

from date_context import governing_span, scenario_for
from schemas import Citation, Condition, RulePeriod, RuleValue, StatementCheck, Subject, VerifiedStatement


def linked_citations(supports, evidence, index=0, role="restriction"):
    from assistant import CitationError, _citation_section, _instruction_overlap, _quote_span
    by_id = {e.chunk_id: e for e in evidence}
    result = []
    for support in supports:
        source = by_id.get(support.chunk_id)
        if source is None or source.document_id != support.document_id:
            raise CitationError("Несуществующий источник ограничения или условия")
        start, end = _quote_span(source.text, support.quote)
        if _instruction_overlap(source.text, start, end):
            raise CitationError("Инструкция для ИИ не является юридическим основанием")
        relation = role
        if role == "restriction":
            if source.metadata.get("temporal_status") == "not_applicable":
                relation = "other_edition"
            elif source.metadata.get("document_type") == "amendment":
                relation = "amendment"
            elif re.search(r"типов|кроме|исключ|только", source.text, re.I):
                relation = "exception"
        result.append(Citation(document_id=source.document_id, chunk_id=source.chunk_id,
            document_title=source.document_title, section=_citation_section(source, start),
            quote=source.text[start:end], coordinates=source.coordinates.model_copy(update={"char_start": source.coordinates.char_start + start, "char_end": source.coordinates.char_start + end}),
            edition=source.applicability.edition, claim_index=index, role=relation, applicability=source.applicability))
    return result


def statement_key(claim):
    return "s-" + hashlib.sha256(claim.model_dump_json().encode()).hexdigest()[:16]


def build_statement(claim, evidence, filters, *, citations=None, mechanical=True, semantic=None, reasons=(), published=False):
    sources = {e.chunk_id: e for e in evidence}
    related = [sources[s.chunk_id] for s in claim.supports if s.chunk_id in sources]
    subject = claim.subject.model_copy()
    subject.contract_number = subject.contract_number or next((e.contract_number for e in related if e.contract_number), None) or filters.contract_number
    subject.counterparty = subject.counterparty or next((e.counterparty for e in related if e.counterparty), None) or filters.counterparty
    value = claim.value.model_copy()
    if not value.description:
        value.description = claim.text
    # Совместимость с коротким черновиком v1: извлекаем только буквально названные поля.
    duration = re.search(r"(\d+)\s+(?:(календарных|рабочих)\s+)?(дней|дня|день|месяцев|месяца|недель|недели)", claim.text)
    if duration and not value.duration:
        value.duration, value.unit = duration[1], duration[3]
        value.day_type = {"календарных": "calendar", "рабочих": "working"}.get(duration[2], "unspecified")
    anchor = re.search(r"(?:после|с даты|со дня)\s+(.+?)(?:\.[ \n]|\.$|$)", claim.text)
    if anchor and not value.counted_from:
        value.counted_from = anchor[0].rstrip(".")
    conditions = []
    scenario = scenario_for(filters)
    for item in claim.conditions:
        state = "unknown"
        if item.key and item.expected is not None and item.key in scenario.facts:
            state = "met" if scenario.facts[item.key] == item.expected else "not_met"
        conditions.append(Condition(text=item.text, key=item.key, expected=item.expected, status=state,
                                    citations=linked_citations(item.supports, evidence)))
    limitations = linked_citations(claim.limitations, evidence)
    if semantic:
        limitations += linked_citations(semantic.limitations, evidence, role="contradiction" if semantic.status == "contradiction" else "restriction")
    status = semantic.status if semantic else "insufficient"
    why = list(reasons) + (semantic.reasons if semantic else ["Смысловая проверка не выполнена"])
    if not mechanical:
        status = "insufficient"
    if status == "confirmed" and conditions:
        status = "conditional"
    if status == "conditional" and any(c.status != "met" for c in conditions) and not re.search(r"\bесли\b|при услов|\bв случае\b", claim.text, re.I):
        status = "insufficient"
        why.append("Неизвестное условие потеряно в безусловной формулировке")
    period = claim.period
    if period is None and related:
        periods = {(e.applicability.valid_from, e.applicability.valid_to, e.applicability.date_kind) for e in related}
        if len(periods) == 1:
            left, right, kind = next(iter(periods))
            period = RulePeriod(valid_from=left, valid_to=right, date_kind=kind)
    return VerifiedStatement(statement_id=statement_key(claim), text=claim.text, subject=subject, rule=claim.rule,
        value=value, conditions=conditions, period=period, supports=citations or [], limitations=limitations,
        check=StatementCheck(status=status, mechanical_pass=mechanical, semantic_checked=semantic is not None, reasons=why),
        published=published and mechanical and status in {"confirmed", "conditional"})
