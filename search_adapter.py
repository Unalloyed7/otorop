"""Подключение поиска №1 и явно выбираемый резервный поиск по настоящей базе.

Временный адаптер не разрешает автоматически условия допсоглашений. Он передаёт
договор и связанные документы целиком, чтобы движок сохранил условия и исключения.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import math
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET
import io

from llm_client import LLMClient, env_value
from schemas import Applicability, Coordinates, Evidence, SearchFilters, SearchResult


class SearchError(RuntimeError):
    pass


def tokens(text: str) -> list[str]:
    # Только простой лексический резерв. Морфология и гибридный индекс — зона №1.
    return [s[:7] for s in re.findall(r"[а-яёa-z0-9]+", text.lower().replace("ё", "е")) if len(s) > 1]


def norm(text: str) -> str:
    return re.sub(r"[^а-яa-z0-9]", "", text.lower().replace("ё", "е"))


def party_key(text: str) -> str:
    return norm(re.sub(r"^\s*(?:ООО|ПАО|АО)\s*", "", text, flags=re.I))


def _date(text):
    try:
        return datetime.strptime(text, "%d.%m.%Y").date()
    except ValueError:
        return None


def read_docx(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if z.getinfo("word/document.xml").file_size > 8_000_000:
            raise SearchError("Слишком большой DOCX")
        root = ET.fromstring(z.read("word/document.xml"))
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    return "\n".join("".join(x.text or "" for x in p.iter(ns + "t")) for p in root.iter(ns + "p"))


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    path: str
    title: str
    text: str
    contract_number: str | None
    counterparty: str | None
    applicability: Applicability


def document(path: str, text: str) -> SourceDocument:
    header = text[:1400]
    contract = re.search(r"(?:договор[^\n№]{0,50})№\s*([А-ЯA-ZЁ]+[-–]\d+[/-]\d{4})", header, re.I)
    party = re.search(r'ООО\s*[«"]([^»"\n]+)[»"]', header) if contract else None
    revision = re.search(r"Редакция от (\d{2}\.\d{2}\.\d{4})", header, re.I)
    start = re.search(r"Действует с (\d{2}\.\d{2}\.\d{4})", header, re.I)
    stop = re.search(r"УТРАТИЛА СИЛУ с (\d{2}\.\d{2}\.\d{4})", header, re.I)
    snapshot = re.search(r"Дата выгрузки:\s*(\d{2}\.\d{2}\.\d{4})", header)
    # Название только из оригинального заголовка или имени файла.
    title = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].replace("__", ": ").replace("_", " ")
    if snapshot:
        source = re.search(r"^Источник:\s*(.+)", text)
        if source:
            title = source.group(1).split(" (с изменениями")[0]
    edition = ("Редакция от " + revision[1]) if revision else ("Действующая редакция; выгрузка " + snapshot[1] if snapshot else "Предоставленный текст; условия действия см. в документе")
    app = Applicability(
        edition=edition, valid_from=_date(start[1]) if start else (_date(revision[1]) if revision else None),
        valid_to=_date(stop[1]) if stop else None, date_kind="event" if revision else None,
        current_only=bool(snapshot), snapshot_date=_date(snapshot[1]) if snapshot else None,
        conditions=["Применимость и изменения проверяются по полному тексту; автоматическое слияние редакций не выполнено"],
    )
    return SourceDocument(hashlib.sha256(path.encode()).hexdigest()[:16], path, title[:500], text,
                          contract[1] if contract else None, party[1] if party else None, app)


def chunks(doc: SourceDocument) -> list[Evidence]:
    # Короткие документы целиком: не теряем 7.5, особые случаи и условия допсоглашений.
    spans = []
    start = 0
    while start < len(doc.text):
        end = min(start + 6500, len(doc.text))
        if end < len(doc.text):
            boundary = doc.text.rfind("\n\n", start + 2500, end)
            if boundary > start:
                end = boundary + 2
        spans.append((start, end))
        if end == len(doc.text):
            break
        start = max(start + 1, end - 700)  # перекрытие сохраняет соседние условия
    result = []
    for start, end in spans:
        text = doc.text[start:end]
        headings = re.findall(r"(?m)^((?:Статья\s+\d+[.\d]*[^\n]*|\d+(?:\.\d+)*\.[^\n]{0,100}))", text)
        section = "; ".join(headings[:3])[:250] or "Текст документа"
        matches = list(re.finditer(r"(?m)^(\d+(?:\.\d+)*\.)\s+", text))
        section_spans = [{"start": m.start(), "end": matches[i + 1].start() if i + 1 < len(matches) else len(text),
                          "label": "п. " + m.group(1).rstrip(".")} for i, m in enumerate(matches)]
        result.append(Evidence(
            document_id=doc.document_id, chunk_id=f"{doc.document_id}:{start}:{end}",
            document_title=doc.title, section=section, text=text,
            coordinates=Coordinates(source_path=doc.path, char_start=start, char_end=end),
            applicability=doc.applicability.model_copy(deep=True),
            contract_number=doc.contract_number, counterparty=doc.counterparty,
            metadata={"backend": "bootstrap_lexical", "offset_unit": "unicode_codepoint", "section_spans": section_spans,
                      "source_format": "docx_extracted_text" if doc.path.endswith(".docx") else "utf8_text"},
        ))
    return result


class LexicalSearch:
    def __init__(self, documents: list[SourceDocument], client: LLMClient | None = None, top_k: int = 12):
        if not documents:
            raise SearchError("Нет документов для индекса")
        self.documents = documents
        self.evidence = [e for d in documents for e in chunks(d)]
        self.client = client
        self.top_k = top_k
        self._terms = [Counter(tokens(e.document_title + " " + e.text)) for e in self.evidence]
        self._df = Counter(t for terms in self._terms for t in terms)
        self._avg = sum(sum(x.values()) for x in self._terms) / len(self._terms)

    @classmethod
    def from_zip(cls, path: Path, client: LLMClient | None = None, top_k: int = 12):
        docs = []
        try:
            with zipfile.ZipFile(path) as z:
                total = 0
                for info in sorted(z.infolist(), key=lambda x: x.filename):
                    p = Path(info.filename)
                    if info.is_dir() or "__MACOSX" in p.parts:
                        continue
                    if p.suffix.lower() not in {".txt", ".md", ".docx"}:
                        raise SearchError("Неподдерживаемый формат документа; подключите индекс №1")
                    total += info.file_size
                    if info.file_size > 8_000_000 or total > 100_000_000 or len(docs) >= 5000:
                        raise SearchError("Превышен размер базы временного адаптера")
                    raw = z.read(info)
                    text = read_docx(raw) if p.suffix.lower() == ".docx" else raw.decode("utf-8-sig")
                    if text.strip():
                        docs.append(document(info.filename, text))
        except (OSError, ValueError, UnicodeError, zipfile.BadZipFile, ET.ParseError, KeyError):
            raise SearchError("Ошибка чтения базы знаний") from None
        return cls(docs, client, top_k)

    async def search(self, query: str, filters: SearchFilters) -> SearchResult:
        # Точное сравнение, без подмены Альфа-Трейд на Альфа-Трейдинг.
        if filters.counterparty:
            parties = sorted({d.counterparty for d in self.documents if d.counterparty})
            exact = [p for p in parties if party_key(p) == party_key(filters.counterparty)]
            near = [p for p in parties if party_key(filters.counterparty) in party_key(p)]
            if not exact and len(near) > 1 and not filters.contract_number:
                return SearchResult(clarification="Уточните контрагента: " + " или ".join(near) + "?", candidates=near)
            if not exact:
                return SearchResult(missing_info=["Договор с контрагентом " + filters.counterparty])
        if filters.contract_number and not any(d.contract_number and norm(d.contract_number) == norm(filters.contract_number) for d in self.documents):
            return SearchResult(missing_info=["Договор № " + filters.contract_number])
        query_terms = set(tokens(query))
        scored = []
        for e, terms in zip(self.evidence, self._terms):
            if filters.document_id and e.document_id != filters.document_id:
                continue
            # Глобальные регламенты и законы сохраняются при фильтре договора.
            if e.contract_number and filters.contract_number and norm(e.contract_number) != norm(filters.contract_number):
                continue
            if e.counterparty and filters.counterparty and party_key(e.counterparty) != party_key(filters.counterparty):
                continue
            size = sum(terms.values())
            score = sum(math.log(1 + (len(self.evidence) - self._df[t] + .5) / (self._df[t] + .5)) *
                        (terms[t] * 2.2) / (terms[t] + 1.2 * (.25 + .75 * size / self._avg))
                        for t in query_terms if terms[t])
            if filters.contract_number and e.contract_number:
                score += 15  # договор и все его допсоглашения не теряются
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda x: (-x[0], x[1].chunk_id))
        selected = [e.model_copy(update={"metadata": {**e.metadata, "score": score}}) for score, e in scored[:self.top_k]]
        if self.client and self.client.settings.reranker_model and selected:
            order = await self.client.rerank(query, [e.document_title + "\n" + e.text for e in selected])
            # Не отбрасываем связанные договоры после реранкинга.
            related = [e for e in selected if e.contract_number and filters.contract_number]
            selected = list({e.chunk_id: e for e in [*(selected[i] for i in order), *related]}.values())
        return SearchResult(evidence=selected)


async def build_search(client: LLMClient | None):
    factory_path = env_value("SEARCH_FACTORY", default="retrieval:create_search")
    data_dir = Path(env_value("DATA_DIR", default="data"))
    module, attribute = factory_path.split(":", 1)
    factory = getattr(importlib.import_module(module), attribute)
    backend = factory(data_dir=data_dir, client=client)
    return await backend if inspect.isawaitable(backend) else backend


def create_lexical_search(*, data_dir: Path, client: LLMClient | None):
    """Резерв включается явно: SEARCH_FACTORY=search_adapter:create_lexical_search."""
    for filename in ("knowledge-base.zip", "kb.zip", "knowledge_base.zip"):
        path = data_dir / filename
        if path.is_file():
            return LexicalSearch.from_zip(path, client, client.settings.top_k if client else 12)
    raise SearchError("Нет базы знаний в DATA_DIR")
