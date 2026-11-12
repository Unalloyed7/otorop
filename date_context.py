"""Раздельные обстоятельства и точность дат. Интервалы всегда полуоткрытые."""
from __future__ import annotations

from datetime import date, timedelta
import re
from typing import Literal

from schemas import DateSpan, ScenarioDates, SearchFilters

_MONTHS = ["январ", "феврал", "март", "апрел", "ма", "июн", "июл", "август", "сентябр", "октябр", "ноябр", "декабр"]
_SignedDateKind = Literal["contract_signed", "specification_signed"]
_EXPLICIT_DATE = re.compile(
    r"(?<![\w/-])(?:"
    r"\d{1,2}\.\d{1,2}\.\d{4}"
    r"|\d{4}-\d{2}(?:-\d{2})?"
    r"|\d{1,2}\s+[а-яё]+\s+\d{4}(?:\s*(?:года|году|г\.?))?"
    r"|(?:в\s+)?[а-яё]+\s+\d{4}(?:\s*(?:года|году|г\.?))?"
    r"|(?:в\s+)?\d{4}(?:\s*(?:год(?:а|у)?|г\.?))?"
    r")(?![\w/-])",
    re.IGNORECASE,
)
_SIGNED_DATE_CUES: tuple[tuple[_SignedDateKind, re.Pattern[str]], ...] = (
    (
        "specification_signed",
        re.compile(
            r"спецификац[а-яё]*[\s\S]{0,80}(?:подпис|заключ)"
            r"|(?:подпис|заключ)[а-яё]*[\s\S]{0,80}спецификац",
            re.IGNORECASE,
        ),
    ),
    (
        "contract_signed",
        re.compile(
            r"(?:договор|контракт)[а-яё]*[\s\S]{0,80}(?:заключ|подпис)"
            r"|(?:заключ|подпис)[а-яё]*[\s\S]{0,80}(?:договор|контракт)",
            re.IGNORECASE,
        ),
    ),
)


def _month_number(word):
    matches = [i + 1 for i, stem in enumerate(_MONTHS) if (word in {"май", "мая", "мае"} if i == 4 else word.startswith(stem))]
    return matches[0] if len(matches) == 1 else None

def parse_date_span(value: str) -> DateSpan:
    text = value.strip().casefold()
    year_match = re.fullmatch(r"(?:в\s+)?(\d{4})(?:\s*(?:год|года|году|г\.?))?", text)
    if year_match:
        year = int(year_match[1])
        return DateSpan(start=date(year, 1, 1), end_exclusive=date(year + 1, 1, 1), precision="year", label=value)
    month = re.fullmatch(r"(\d{4})-(\d{2})", text)
    if not month:
        words = re.fullmatch(r"(?:в\s+)?([а-яё]+)\s+(\d{4})(?:\s*(?:года|году|г\.?))?", text)
        if words:
            number = _month_number(words[1])
            if number:
                month = (None, words[2], str(number))
    if month:
        year, number = int(month[1]), int(month[2])
        return DateSpan(start=date(year, number, 1), end_exclusive=date(year + (number == 12), number % 12 + 1, 1), precision="month", label=value)
    if ".." in text:
        if text.count("..") != 1:
            raise ValueError("Нужны две границы диапазона")
        start, end = [parse_date_span(x.strip()) for x in text.split("..", 1)]
        return DateSpan(start=start.start, end_exclusive=end.end_exclusive, precision="range", label=value)
    numeric = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", text)
    words = re.fullmatch(r"(\d{1,2})\s+([а-яё]+)\s+(\d{4})(?:\s*(?:года|г\.?))?", text)
    if numeric:
        day = date(int(numeric[3]), int(numeric[2]), int(numeric[1]))
    elif words and _month_number(words[2]):
        day = date(int(words[3]), _month_number(words[2]), int(words[1]))
    else:
        day = date.fromisoformat(text)
    return DateSpan(start=day, end_exclusive=day + timedelta(days=1), precision="day", label=value)


def infer_signed_date(text: str) -> tuple[_SignedDateKind, DateSpan] | None:
    candidates: dict[tuple[_SignedDateKind, date, date, str], tuple[_SignedDateKind, DateSpan]] = {}
    for match in _EXPLICIT_DATE.finditer(text):
        try:
            span = parse_date_span(match.group())
        except ValueError:
            continue
        left = max(text.rfind(separator, 0, match.start()) for separator in ".!?\n") + 1
        right = len(text)
        for separator in ".!?\n":
            position = text.find(separator, match.end())
            if position >= 0:
                right = min(right, position)
        sentence = text[left:right]
        kinds = [kind for kind, cue in _SIGNED_DATE_CUES if cue.search(sentence)]
        if len(kinds) == 1:
            candidates[(kinds[0], span.start, span.end_exclusive, span.precision)] = (kinds[0], span)
    if len(candidates) != 1:
        return None
    return next(iter(candidates.values()))


def scenario_for(filters: SearchFilters) -> ScenarioDates:
    result = filters.scenario.model_copy(deep=True)
    aliases = {"specification_signed_at": "specification_signed", "contract_signed_at": "contract_signed", "document_signed": "contract_signed", "delivered": "delivery"}
    kind = aliases.get(filters.date_kind, filters.date_kind)
    if filters.event_date:
        kind = kind if kind in ScenarioDates.model_fields and kind != "facts" else "event"
        old = getattr(result, kind)
        exact = parse_date_span(filters.event_date.isoformat())
        if old and not (old.start <= filters.event_date < old.end_exclusive):
            raise ValueError("Дата события противоречит указанному периоду")
        if not old:
            setattr(result, kind, exact)
    return result


def governing_span(filters: SearchFilters, evidence=None) -> DateSpan | None:
    scenario = scenario_for(filters)
    if evidence is not None:
        kind = {"specification_signed_at": "specification_signed", "contract_signed_at": "contract_signed", "document_signed": "contract_signed"}.get(evidence.applicability.date_kind, evidence.applicability.date_kind)
        if kind in {"specification_signed", "contract_signed", "delivery"}:
            return getattr(scenario, kind)
    if filters.governing_date:
        return getattr(scenario, filters.governing_date)
    return scenario.event or scenario.contract_signed or scenario.delivery or scenario.specification_signed


def condition_status(record: dict, scenario: ScenarioDates) -> str:
    kind = {"specification_signed_at": "specification_signed", "contract_signed_at": "contract_signed", "signed_at": "contract_signed"}.get(record.get("kind"), record.get("kind"))
    span = getattr(scenario, kind, None) if kind in ScenarioDates.model_fields else None
    if not span or not record.get("date"):
        return "unknown"
    pivot = date.fromisoformat(record["date"])
    operator = record.get("operator")
    if operator in {"after", "on_or_after"}:
        threshold = pivot + timedelta(days=operator == "after")
        return "applicable" if span.start >= threshold else ("not_applicable" if span.end_exclusive <= threshold else "varies")
    if operator in {"before", "on_or_before"}:
        threshold = pivot + timedelta(days=operator == "on_or_before")
        return "applicable" if span.end_exclusive <= threshold else ("not_applicable" if span.start >= threshold else "varies")
    return "unknown"


def range_changes(evidence, filters: SearchFilters) -> list[str]:
    """Обнаружить пересечение границы; одинаковые нормы могут покрыть интервал вместе."""
    span = governing_span(filters)
    scenario = scenario_for(filters)
    issues = []
    for e in evidence:
        if any(condition_status(c, scenario) == "varies" for c in e.metadata.get("applicability_conditions", [])):
            issues.append("Уточните дату подписания спецификации: внутри указанного периода меняется условие допсоглашения")
    if not span or span.precision == "day":
        return list(dict.fromkeys(issues))
    groups = {}
    for e in evidence:
        a = e.applicability
        if not a.valid_from and not a.valid_to:
            continue
        left, right = a.valid_from or date.min, a.valid_to or date.max
        if left >= span.end_exclusive or right <= span.start:
            continue
        signature = (e.contract_number or e.document_title.casefold(), e.section, re.sub(r"\s+", " ", e.text).strip())
        groups.setdefault(signature, []).append((left, right))
    for intervals in groups.values():
        cursor = span.start
        for left, right in sorted(intervals):
            if left <= cursor:
                cursor = max(cursor, right)
        if cursor < span.end_exclusive:
            issues.append("Уточните дату события: внутри указанного периода меняется применимая редакция или отсутствует её часть")
    return list(dict.fromkeys(issues))


def equivalent_coverage(evidence, filters: SearchFilters) -> set[str]:
    span = governing_span(filters)
    if not span or span.precision == "day":
        return set()
    groups = {}
    for e in evidence:
        if e.applicability.current_only:
            continue
        signature = (e.contract_number or e.document_title.casefold(), e.section, re.sub(r"\s+", " ", e.text).strip())
        groups.setdefault(signature, []).append(e)
    covered = set()
    for items in groups.values():
        cursor = span.start
        intervals = sorted(((e.applicability.valid_from or date.min, e.applicability.valid_to or date.max) for e in items))
        for left, right in intervals:
            if left <= cursor:
                cursor = max(cursor, right)
        if cursor >= span.end_exclusive:
            covered.update(e.chunk_id for e in items)
    return covered
