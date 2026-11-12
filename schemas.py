"""Общий контракт поиска и ответа v1. Координаты относятся к исходному тексту."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


DateKind = Literal["contract_signed", "specification_signed", "delivery", "event"]


class DateSpan(Model):
    """Календарный интервал [start, end_exclusive), а не выдуманная дата события."""
    start: date
    end_exclusive: date
    precision: Literal["day", "month", "year", "range"] = "range"
    label: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def valid_span(self):
        if self.end_exclusive <= self.start:
            raise ValueError("Пустой интервал даты")
        if self.precision == "day" and self.end_exclusive != self.start + timedelta(days=1):
            raise ValueError("День должен содержать ровно один день")
        if self.precision == "month":
            expected = date(self.start.year + (self.start.month == 12), self.start.month % 12 + 1, 1)
            if self.start.day != 1 or self.end_exclusive != expected:
                raise ValueError("Месяц должен охватывать весь календарный месяц")
        if self.precision == "year" and (self.start != date(self.start.year, 1, 1) or self.end_exclusive != date(self.start.year + 1, 1, 1)):
            raise ValueError("Год должен охватывать весь календарный год")
        return self


class ScenarioDates(Model):
    contract_signed: DateSpan | None = None
    specification_signed: DateSpan | None = None
    delivery: DateSpan | None = None
    event: DateSpan | None = None
    facts: dict[str, str | bool | int] = Field(default_factory=dict, max_length=30)

    @field_validator("contract_signed", "specification_signed", "delivery", "event", mode="before", json_schema_input_type=DateSpan | str | None)
    @classmethod
    def parse_span(cls, value):
        if isinstance(value, str):
            from date_context import parse_date_span
            return parse_date_span(value)
        return value


class Coordinates(Model):
    source_path: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    page: int | None = Field(default=None, ge=1)
    bbox: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def ordered(self):
        if self.char_end <= self.char_start:
            raise ValueError("Некорректный диапазон координат")
        return self


class Applicability(Model):
    edition: str = Field(min_length=1)
    valid_from: date | None = None
    valid_to: date | None = None  # правая граница исключается
    date_kind: str | None = None  # specification_signed, contract_signed, event ...
    conditions: list[str] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)  # chunk_id
    current_only: bool = False
    snapshot_date: date | None = None
    resolved: bool = False  # №1 установил применимость к переданным фильтрам

    @model_validator(mode="after")
    def ordered(self):
        if self.valid_from and self.valid_to and self.valid_to <= self.valid_from:
            raise ValueError("Пустой интервал действия редакции")
        return self


class Evidence(Model):
    document_id: str = Field(min_length=1, max_length=300)
    chunk_id: str = Field(min_length=1, max_length=300)
    document_title: str = Field(min_length=1, max_length=500)
    section: str = Field(min_length=1, max_length=250)
    text: str = Field(min_length=1, max_length=30000)
    coordinates: Coordinates
    applicability: Applicability
    contract_number: str | None = None
    counterparty: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchFilters(Model):
    contract_number: str | None = None
    counterparty: str | None = None
    document_id: str | None = None
    event_date: date | None = None
    date_kind: str | None = Field(default=None, description="Какое событие датировано, например specification_signed")
    scenario: ScenarioDates = Field(default_factory=ScenarioDates)
    governing_date: DateKind | None = None
    purpose: Literal["grounds", "limitations"] = "grounds"
    anchor_document_ids: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def date_has_meaning(self):
        if self.event_date and not self.date_kind:
            raise ValueError("Для даты события нужен date_kind")
        kind = {"specification_signed_at": "specification_signed", "document_signed": "contract_signed", "delivered": "delivery"}.get(self.date_kind, self.date_kind)
        span = getattr(self.scenario, kind, None) if kind in {"contract_signed", "specification_signed", "delivery", "event"} else None
        if span and self.event_date and not (span.start <= self.event_date < span.end_exclusive):
            raise ValueError("Дата противоречит периоду того же события")
        if self.governing_date and not getattr(self.scenario, self.governing_date) and not (self.event_date and kind == self.governing_date):
            raise ValueError("Для governing_date нужно соответствующее датированное обстоятельство")
        return self


class SearchResult(Model):
    evidence: list[Evidence] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    clarification: str | None = None
    candidates: list[str] = Field(default_factory=list)
    # Пустая выдача != отказ API. Технические сбои поиска следует выбрасывать.

    @model_validator(mode="after")
    def unique_chunks(self):
        ids = [e.chunk_id for e in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("Повторяющийся chunk_id в выдаче")
        return self


class Citation(Model):
    document_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    document_title: str = Field(min_length=1)
    section: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    coordinates: Coordinates
    edition: str = Field(min_length=1)
    claim_index: int = Field(ge=0)
    role: Literal["support", "restriction", "contradiction", "amendment", "exception", "other_edition", "procedure"] = "support"
    applicability: Applicability | None = None


class Subject(Model):
    contract_number: str | None = None
    counterparty: str | None = None
    employee: str | None = None


class RuleValue(Model):
    description: str = Field(default="", max_length=1600)
    amount: str | None = None
    currency: str | None = None
    duration: str | None = None
    unit: str | None = None
    day_type: Literal["calendar", "working", "unspecified"] = "unspecified"
    counted_from: str | None = None
    action: str | None = None


class RulePeriod(Model):
    valid_from: date | None = None
    valid_to: date | None = None
    date_kind: str | None = None

    @model_validator(mode="after")
    def ordered(self):
        if self.valid_from and self.valid_to and self.valid_to <= self.valid_from:
            raise ValueError("Пустой период правила")
        return self


CheckStatus = Literal["confirmed", "conditional", "contradiction", "insufficient"]


class StatementCheck(Model):
    status: CheckStatus
    mechanical_pass: bool
    semantic_checked: bool
    reasons: list[str] = Field(default_factory=list)


class Condition(Model):
    text: str = Field(min_length=1, max_length=1200)
    key: str | None = None
    expected: str | bool | int | None = None
    status: Literal["met", "not_met", "unknown"] = "unknown"
    citations: list[Citation] = Field(default_factory=list)


class VerifiedStatement(Model):
    statement_id: str
    text: str
    subject: Subject
    rule: Literal["duty", "permission", "restriction", "information"] = "information"
    value: RuleValue
    conditions: list[Condition] = Field(default_factory=list)
    period: RulePeriod | None = None
    supports: list[Citation] = Field(default_factory=list)
    limitations: list[Citation] = Field(default_factory=list)
    check: StatementCheck
    published: bool = False

    @model_validator(mode="after")
    def publish_only_checked(self):
        if self.published and (self.check.status not in {"confirmed", "conditional"} or not self.check.mechanical_pass or not self.check.semantic_checked or not self.supports):
            raise ValueError("Публикуемое утверждение требует проверок и подтверждающих ссылок")
        return self


AnswerStatus = Literal["answer", "partial", "clarify", "insufficient", "out_of_scope", "refuse", "error"]


class Answer(Model):
    status: AnswerStatus
    text: str = Field(min_length=1, max_length=8000)
    citations: list[Citation] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    schema_version: Literal[2] = 2
    statements: list[VerifiedStatement] = Field(default_factory=list)
    scenario: ScenarioDates = Field(default_factory=ScenarioDates)
    verification: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def consistent(self):
        if not self.text.strip():
            raise ValueError("Пустой ответ")
        if self.status in {"answer", "partial"} and not self.citations:
            raise ValueError("Содержательный ответ требует ссылок")
        if self.status in {"partial", "clarify", "insufficient"} and not self.missing_info:
            raise ValueError("Укажите недостающую информацию")
        if self.status == "answer" and self.missing_info:
            raise ValueError("Полный ответ не должен скрывать пробелы")
        return self


class Message(Model):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=60000)


# Внутренний результат генерации. Названия источников модель не генерирует.
class Support(Model):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class ClaimCondition(Model):
    text: str = Field(min_length=1, max_length=1200)
    key: str | None = None
    expected: str | bool | int | None = None
    status: Literal["met", "not_met", "unknown"] = "unknown"
    supports: list[Support] = Field(default_factory=list, max_length=8)


class Claim(Model):
    text: str = Field(min_length=1, max_length=1600)
    supports: list[Support] = Field(min_length=1, max_length=8)
    addresses: list[StrictInt] = Field(min_length=1)  # индексы частей вопроса из Intent.parts
    subject: Subject = Field(default_factory=Subject)
    rule: Literal["duty", "permission", "restriction", "information"] = "information"
    value: RuleValue = Field(default_factory=RuleValue)
    conditions: list[ClaimCondition] = Field(default_factory=list, max_length=15)
    period: RulePeriod | None = None
    limitations: list[Support] = Field(default_factory=list, max_length=12)


class Draft(Model):
    status: Literal["answer", "partial", "clarify", "insufficient"]
    claims: list[Claim] = Field(default_factory=list, max_length=10)
    missing_info: list[str] = Field(default_factory=list, max_length=10)


class Intent(Model):
    action: Literal["search", "clarify", "out_of_scope", "refuse"]
    query: str = Field(min_length=1, max_length=3000)
    parts: list[str] = Field(min_length=1, max_length=10)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    clarification: str | None = None
    temporal_mode: Literal["auto", "current", "historical", "future", "snapshot"] = "auto"


class Verdict(Model):
    supported: list[StrictInt] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    checks: list["SemanticCheck"] = Field(default_factory=list)


class SemanticCheck(Model):
    index: StrictInt
    status: CheckStatus
    reasons: list[str] = Field(default_factory=list)
    limitations: list[Support] = Field(default_factory=list, max_length=12)


Verdict.model_rebuild()
