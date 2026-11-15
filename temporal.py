"""Ограниченная проверка периода нормы по её собственному тексту, без правовой памяти."""
from __future__ import annotations

from datetime import date, timedelta
import re

from schemas import Evidence, SearchFilters

_MONTHS = dict(zip(("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"), range(1, 13)))
_DATE = r"(?:\d{1,2}[./]\d{1,2}[./]\d{4}|\d{1,2}\s+(?:" + "|".join(_MONTHS) + r")\s+\d{4})"


def _date(value: str) -> date | None:
    parts = re.fullmatch(r"(\d{1,2})[./](\d{1,2})[./](\d{4})", value)
    if parts:
        day, month, year = map(int, parts.groups())
    else:
        match = re.fullmatch(r"(\d{1,2})\s+([а-я]+)\s+(\d{4})", value.casefold())
        if not match or match[2] not in _MONTHS:
            return None
        day, month, year = int(match[1]), _MONTHS[match[2]], int(match[3])
    try:
        return date(year, month, day)
    except ValueError:
        return None


def rule_periods(text: str) -> list[dict]:
    """Только явно датированное правило в том же абзаце. Не дата выгрузки/изменения.

    Это узкая эвристика, а не толкование всех вариантов вступления норм в силу.
    Текст периода остаётся основанием и должен попадать в цитаты ответа.
    """
    result = []
    for line in text.split("\n"):
        if re.search(r"дата\s+(?:выгрузки|снимка)|ВНИМАНИЕ\s+ДЛЯ\s+ИИ", line, re.I):
            continue
        if not re.search(r"\b(?:установить|установлен[аоы]?|составля\w*|применя\w*|действует)\b", line, re.I):
            continue
        start = re.search(r"\bс\s+(" + _DATE + r")", line, re.I)
        end = re.search(r"\bпо\s+(" + _DATE + r")", line, re.I)
        year = re.search(r"\bв\s+(20\d{2})\s+(?:году|г\.)", line, re.I)
        valid_from = _date(start[1]) if start else (date(int(year[1]), 1, 1) if year else None)
        valid_to = (_date(end[1]) + timedelta(days=1)) if end and _date(end[1]) else (date(int(year[1]) + 1, 1, 1) if year and not start else None)
        if valid_from and (not valid_to or valid_to > valid_from):
            result.append({"valid_from": valid_from.isoformat(), "valid_to": valid_to.isoformat() if valid_to else None, "source_text": line.strip()})
    return result


def mode_for(filters: SearchFilters, requested: str, reference: date) -> str:
    from date_context import governing_span
    span = governing_span(filters)
    when = filters.event_date or (span.start if span else None)
    if requested == "current" and when and when > reference:
        return "future"
    if requested != "auto":
        return requested
    if filters.date_kind in {"law_snapshot", "snapshot", "snapshot_date"}:
        return "snapshot"
    if not when or when == reference:
        return "current"
    return "historical" if when < reference else "future"


def temporal_issue(e: Evidence, filters: SearchFilters, *, mode: str, reference: date) -> str | None:
    from date_context import governing_span
    from schemas import ScenarioDates
    span = governing_span(filters, e)
    if span:
        kind = filters.governing_date or filters.date_kind or "event"
        flat = filters.model_copy(update={"event_date": span.start, "date_kind": kind, "scenario": ScenarioDates()})
        issue = _temporal_issue_day(e, flat, mode=mode, reference=reference)
        if issue:
            return issue
        if span.precision != "day":
            flat.event_date = span.end_exclusive - timedelta(days=1)
            return _temporal_issue_day(e, flat, mode=mode, reference=reference)
        return None
    return _temporal_issue_day(e, filters, mode=mode, reference=reference)


def _temporal_issue_day(e: Evidence, filters: SearchFilters, *, mode: str, reference: date) -> str | None:
    a, when = e.applicability, filters.event_date
    if not a.current_only:
        if e.metadata.get("document_type") == "internal_policy":
            target = when or reference
            if (a.valid_from and target < a.valid_from) or (a.valid_to and target >= a.valid_to):
                return "Редакция регламента не действует на рассматриваемую дату"
        if when and a.date_kind in {"event", filters.date_kind} and ((a.valid_from and when < a.valid_from) or (a.valid_to and when >= a.valid_to)):
            return "Редакция не действует на дату события"
        return None
    selected = mode_for(filters, mode, reference)
    if selected == "current":
        # Выданный текущий текст не превращается в исторический из-за даты выгрузки.
        target = when or reference
        if (a.valid_from and target < a.valid_from) or (a.valid_to and target >= a.valid_to):
            return "Редакция не действует на рассматриваемую дату"
        periods = rule_periods(e.text)
        if len(periods) == 1 and periods[0]["source_text"] == e.text.strip():
            p = periods[0]
            if target < date.fromisoformat(p["valid_from"]) or (p["valid_to"] and target >= date.fromisoformat(p["valid_to"])):
                return "Период приведённого положения не включает рассматриваемую дату"
        return None
    if not when:
        return "Нужна дата применения нормы"
    if selected == "snapshot":
        return None if when == a.snapshot_date else "Нужен снимок источника на указанную дату"
    for period in rule_periods(e.text):
        start = date.fromisoformat(period["valid_from"])
        stop = date.fromisoformat(period["valid_to"]) if period["valid_to"] else None
        if when < start or (stop and when >= stop):
            continue
        # Не продолжаем бессрочное правило из прошлого в неизвестное будущее.
        if stop or (a.snapshot_date and when <= a.snapshot_date) or (selected == "future" and start > reference):
            return None
    return f"Редакция или прямо установленный период {e.document_title} на {when.isoformat()}"


def period_quoted(supports: list, evidence: list[Evidence], filters: SearchFilters, *, mode: str, reference: date) -> bool:
    """Для датированного прошлого/будущего сам текст периода тоже должен цитироваться."""
    from date_context import governing_span
    from schemas import ScenarioDates
    span = governing_span(filters)
    if span:
        filters = filters.model_copy(update={"event_date": span.start, "date_kind": filters.date_kind or "event", "scenario": ScenarioDates()})
    if not filters.event_date or mode_for(filters, mode, reference) in {"current", "snapshot"}:
        return True
    by_id = {e.chunk_id: e for e in evidence}
    for support in supports:
        e = by_id.get(support.chunk_id)
        if e and e.applicability.current_only:
            # Не позволяем сузить цитату до суммы, отбросив датированное условие.
            if temporal_issue(e.model_copy(update={"text": support.quote}), filters, mode=mode, reference=reference):
                return False
            if span and span.precision != "day":
                end_filters = filters.model_copy(update={"event_date": span.end_exclusive - timedelta(days=1)})
                if temporal_issue(e.model_copy(update={"text": support.quote}), end_filters, mode=mode, reference=reference):
                    return False
    return True
