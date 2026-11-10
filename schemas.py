"""Общий контракт поиска и ответа v1. Координаты относятся к исходному тексту."""
from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


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

    @model_validator(mode="after")
    def date_has_meaning(self):
        if self.event_date and not self.date_kind:
            raise ValueError("Для даты события нужен date_kind")
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


AnswerStatus = Literal["answer", "partial", "clarify", "insufficient", "out_of_scope", "refuse", "error"]


class Answer(Model):
    status: AnswerStatus
    text: str = Field(min_length=1, max_length=8000)
    citations: list[Citation] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)

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


class Claim(Model):
    text: str = Field(min_length=1, max_length=1600)
    supports: list[Support] = Field(min_length=1, max_length=8)
    addresses: list[StrictInt] = Field(min_length=1)  # индексы частей вопроса из Intent.parts


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
    supported: list[StrictInt]
    missing_info: list[str] = Field(default_factory=list)
