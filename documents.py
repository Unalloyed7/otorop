"""Load and structurally split the organizer-provided knowledge base."""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from datetime import date, timedelta
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree as ET

PROCESSOR_VERSION = "documents-v2"
SUPPORTED_SUFFIXES = {".txt", ".md", ".docx"}

_CONTRACT_RE = re.compile(
    r"(?<![А-ЯЁA-Z0-9])([А-ЯЁA-Z]{1,6}\s*[-–—]\s*\d{1,5}\s*/\s*\d{4})(?!\d)",
    re.IGNORECASE,
)
_ORG_RE = re.compile(
    r"\b(?:ООО|АО|ПАО|ИП)\s*[«\"“]([^»\"”\n,;]+)[»\"”]",
    re.IGNORECASE,
)
_NUMERIC_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\d)")
_WORD_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|"
    r"сентября|октября|ноября|декабря)\s+(\d{4})\b",
    re.IGNORECASE,
)
_NUMBERED_LINE_RE = re.compile(
    r"^\s*(?:(Статья)\s+)?(\d+(?:\.\d+){0,5})(?:[.)])?(?=\s|$)\s*(.*)$",
    re.IGNORECASE,
)
_INLINE_REFERENCE_RE = re.compile(
    r"\b(?:пункт(?:а|е|ом)?|п\.|стать(?:я|и|е)|ст\.)\s*(\d+(?:\.\d+){0,5})",
    re.IGNORECASE,
)
_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}


class DocumentLoadError(ValueError):
    """Raised with the source path when a knowledge-base document is unreadable."""


def _normalize_contract(value: str) -> str:
    return re.sub(r"\s+", "", value.upper().replace("Ё", "Е")).replace("–", "-").replace("—", "-")


def _iso_date(text: str) -> str | None:
    match = _NUMERIC_DATE_RE.search(text)
    if match:
        day, month, year = (int(part) for part in match.groups())
    else:
        word_match = _WORD_DATE_RE.search(text)
        if not word_match:
            return None
        day = int(word_match.group(1))
        month = _MONTHS[word_match.group(2).casefold()]
        year = int(word_match.group(3))
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _first_date(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return _iso_date(text[match.end():match.end() + 180])


def _first_date_with_source(pattern: str, text: str) -> tuple[str | None, str | None]:
    anchor = re.search(pattern, text, re.IGNORECASE)
    if not anchor:
        return None, None
    window_end = min(len(text), anchor.end() + 180)
    window = text[anchor.end():window_end]
    date_match = _NUMERIC_DATE_RE.search(window) or _WORD_DATE_RE.search(window)
    if not date_match:
        return None, None
    value = _iso_date(date_match.group(0))
    if not value:
        return None, None
    source = " ".join(text[anchor.start():anchor.end() + date_match.end()].split())
    return value, source


def _extract_metadata(source_file: str, text: str) -> dict[str, Any]:
    title_part = PurePosixPath(source_file).stem.split("__")[-1].replace("_", " ").strip()
    nonempty_lines = [line.strip().lstrip("\ufeff") for line in text.splitlines() if line.strip()]
    title_line = next(
        (
            line for line in nonempty_lines
            if re.search(
                r"\b(?:договор|соглашение|регламент|письмо|служебная записка|"
                r"федеральный закон|статья|кодекс)\b",
                line,
                re.IGNORECASE,
            )
        ),
        nonempty_lines[0] if nonempty_lines else "",
    )
    title = title_line if len(title_line) <= 180 else title_part
    searchable = f"{title_part}\n{text}"
    normalized_name = title_part.casefold()
    header = text[:1600].casefold()

    if (
        "законы__" in source_file.casefold()
        or "152фз" in normalized_name
        or "мрот" in normalized_name
        or re.search(r"(?:^|[_\s])(?:нк|тк|коап)[_\s-]*ст", normalized_name)
        or re.search(r"^\s*(?:статья\s+\d+|федеральный закон)", header)
    ):
        document_type = "law"
    elif re.search(r"\bдоп(?:олнительное)?[ -]соглашение\b", normalized_name, re.IGNORECASE):
        document_type = "amendment"
    elif "регламент" in normalized_name or "регламент согласования" in header:
        document_type = "internal_policy"
    elif "письмо" in normalized_name or "служебная записка" in normalized_name:
        document_type = "letter"
    elif "договор" in normalized_name or re.search(r"^\s*договор\b", header):
        document_type = "agreement"
    else:
        document_type = "other"

    contract_match = _CONTRACT_RE.search(searchable)
    contract_number = _normalize_contract(contract_match.group(1)) if contract_match else None
    source_contract_number = contract_number if document_type == "amendment" else None

    counterparties: list[str] = []
    seen_names: set[str] = set()
    for match in _ORG_RE.finditer(text):
        name = " ".join(match.group(1).split())
        key = name.casefold().replace("ё", "е")
        if key not in seen_names:
            counterparties.append(name)
            seen_names.add(key)
    external_parties = [name for name in counterparties if name.casefold().replace("ё", "е") != "меридиан"]
    counterparty = external_parties[0] if len(external_parties) == 1 else None

    edition = _first_date(r"редакци(?:я|и)\s+от", text)
    first_lines = "\n".join(text.splitlines()[:6])
    signed_date_source = None
    if document_type in {"agreement", "amendment"}:
        signed_date = None
        for line in first_lines.splitlines():
            if not re.search(r"\bг\.", line, re.IGNORECASE):
                continue
            separator = re.search(r"[—–]\s*", line)
            if separator:
                signed_date = _iso_date(line[separator.end():])
                if signed_date:
                    signed_date_source = " ".join(line.split())
                    break
    elif document_type == "letter":
        signed_date, signed_date_source = _first_date_with_source(r"\bот\s+", first_lines)
    else:
        signed_date = None
    approval_date, approval_date_source = _first_date_with_source(
        r"утвержд[её]н[ао]?\s+[^.\n]{0,100}?\bот\s+",
        text,
    )
    effective_date_source = None
    if signed_date and re.search(r"вступает в силу с (?:даты|момента) подписания", text, re.IGNORECASE):
        effective_date = signed_date
        effective_match = re.search(
            r"вступает в силу с (?:даты|момента) подписания[^.\n]*",
            text,
            re.IGNORECASE,
        )
        effective_date_source = (
            " ".join(effective_match.group(0).split()) if effective_match else signed_date_source
        )
    else:
        effective_date, effective_date_source = _first_date_with_source(
            r"(?:вступает в силу с|вступил[ао]? в силу с)",
            text,
        )
    valid_from, valid_from_source = _first_date_with_source(
        r"(?:действует с|действует начиная с)",
        text,
    )
    if valid_from is None:
        valid_from = effective_date
        valid_from_source = effective_date_source

    valid_to, valid_to_source = _first_date_with_source(
        r"(?:утрат(?:ил|ила|ило|или)\s+силу\s+с|действует\s+до|действовал[ао]?\s+до)",
        text,
    )
    if valid_to and re.search(r"(?:действует|действовал[ао]?)\s+до", text, re.IGNORECASE):
        end_date = date.fromisoformat(valid_to) + timedelta(days=1)
        valid_to = end_date.isoformat()
        valid_to_source = f"{valid_to_source} (исключительная граница: {valid_to})"
    snapshot_date, snapshot_date_source = _first_date_with_source(
        r"(?:дата\s+выгрузки|дата\s+снимка|дата\s+получения\s+редакции)\s*[:—-]?\s*",
        text[:2000],
    )
    current_only = document_type == "law" and (
        snapshot_date is not None
        or bool(re.search(r"(?:^|[_\s])действующ(?:ая|его|ей|ую)(?:[_\s.]|$)", normalized_name))
    )
    applicability_conditions: list[dict[str, str]] = []
    if document_type == "amendment":
        for line in text.splitlines():
            if (
                not re.search(r"(?:применя(?:ется|ются)|распространя(?:ется|ются))", line, re.IGNORECASE)
                or not re.search(r"спецификац", line, re.IGNORECASE)
            ):
                continue
            applicability_date = _iso_date(line)
            if not applicability_date:
                continue
            source_text = " ".join(line.split())
            numeric_date = _NUMERIC_DATE_RE.search(source_text)
            word_date = _WORD_DATE_RE.search(source_text)
            date_match = numeric_date or word_date
            assert date_match is not None
            lead = source_text[:date_match.start()].casefold()
            if "после" in lead:
                operator = "after"
            elif re.search(r"\bдо\b", lead):
                operator = "before"
            else:
                operator = "on_or_after"
            applicability_conditions.append({
                "kind": "specification_signed_at",
                "operator": operator,
                "date": applicability_date,
                "source_text": source_text,
            })

    uncertainties: list[str] = []
    if valid_to and not valid_from:
        uncertainties.append("Дата начала действия документа в тексте явно не указана.")
    if document_type == "law" and not valid_from and not valid_to:
        uncertainties.append("Исторический период действия этой редакции не указан.")
    if current_only and snapshot_date and not valid_from and not valid_to:
        uncertainties.append(
            "Дата выгрузки подтверждает только состояние источника на эту дату, но не период действия нормы."
        )
    if current_only and not snapshot_date:
        uncertainties.append("Дата выгрузки текущей редакции закона в тексте не указана.")
    date_sources = [
        {"kind": kind, "source_text": source}
        for kind, source in (
            ("signed_date", signed_date_source),
            ("approval_date", approval_date_source),
            ("effective_date", effective_date_source),
            ("valid_from", valid_from_source),
            ("valid_to", valid_to_source),
            ("snapshot_date", snapshot_date_source),
        )
        if source
    ]
    if document_type in {"agreement", "amendment"} and not contract_number:
        uncertainties.append("Номер договора не удалось извлечь из документа.")
    if document_type == "amendment" and not source_contract_number:
        uncertainties.append("Связь допсоглашения с исходным договором не определена.")

    return {
        "title": title or title_part,
        "document_type": document_type,
        "contract_number": contract_number,
        "source_contract_number": source_contract_number,
        "amendment_number": 1 if document_type == "amendment" and re.search(r"№\s*1\b|соглашение\s+1\b", searchable, re.IGNORECASE) else None,
        "counterparty": counterparty,
        "counterparties": counterparties,
        "edition": edition,
        "signed_date": signed_date,
        "signed_date_source": signed_date_source,
        "approval_date": approval_date,
        "approval_date_source": approval_date_source,
        "effective_date": effective_date,
        "effective_date_source": effective_date_source,
        "valid_from": valid_from,
        "valid_from_source": valid_from_source,
        "valid_to": valid_to,
        "valid_to_source": valid_to_source,
        "snapshot_date": snapshot_date,
        "snapshot_date_source": snapshot_date_source,
        "date_sources": date_sources,
        "current_only": current_only,
        "date_uncertainties": list(uncertainties),
        "applicability_conditions": applicability_conditions,
        "uncertainties": uncertainties,
    }


def _read_docx(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        xml = archive.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for paragraph in root.iter(ns + "p"):
        pieces: list[str] = []
        for node in paragraph.iter():
            if node.tag == ns + "t":
                pieces.append(node.text or "")
            elif node.tag == ns + "tab":
                pieces.append("\t")
            elif node.tag in {ns + "br", ns + "cr"}:
                pieces.append("\n")
        paragraphs.append("".join(pieces))
    return "\n".join(paragraphs)


def _read_source(name: str, raw: bytes) -> str:
    suffix = PurePosixPath(name).suffix.casefold()
    try:
        text = _read_docx(raw) if suffix == ".docx" else raw.decode("utf-8-sig")
    except (UnicodeDecodeError, KeyError, ValueError, zipfile.BadZipFile, ET.ParseError) as exc:
        raise DocumentLoadError(f"Не удалось прочитать документ {name!r}: {exc}") from exc
    if not text.strip():
        raise DocumentLoadError(f"Документ {name!r} пустой.")
    return text


def load_documents(kb_path: str | Path) -> list[dict[str, Any]]:
    """Read supported documents from a UTF-8 ZIP archive or an extracted directory."""
    path = Path(kb_path)
    if path.is_dir():
        sources = []
        for file_path in sorted(path.rglob("*")):
            relative = file_path.relative_to(path).as_posix()
            if (
                not file_path.is_file()
                or file_path.name.startswith(".")
                or relative.startswith("__MACOSX/")
                or file_path.suffix.casefold() not in SUPPORTED_SUFFIXES
            ):
                continue
            try:
                raw = file_path.read_bytes()
            except OSError as exc:
                raise DocumentLoadError(f"Не удалось прочитать документ {relative!r}: {exc}") from exc
            sources.append((relative, raw))
    elif path.is_file() and path.suffix.casefold() == ".zip":
        try:
            with zipfile.ZipFile(path) as archive:
                sources = [
                    (name, archive.read(name))
                    for name in sorted(archive.namelist())
                    if not name.endswith("/")
                    and not name.startswith("__MACOSX/")
                    and PurePosixPath(name).suffix.casefold() in SUPPORTED_SUFFIXES
                ]
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            raise DocumentLoadError(f"Не удалось прочитать архив базы знаний {str(path)!r}: {exc}") from exc
    else:
        raise FileNotFoundError(f"Ожидался каталог базы знаний или ZIP: {path}")

    if not sources:
        raise DocumentLoadError(f"В {str(path)!r} не найдены документы поддерживаемых форматов.")

    documents: list[dict[str, Any]] = []
    for source_file, raw in sources:
        text = _read_source(source_file, raw)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        document_id = hashlib.sha256(
            f"{PROCESSOR_VERSION}\0{source_file}\0{digest}".encode("utf-8")
        ).hexdigest()[:24]
        metadata = _extract_metadata(source_file, text)
        documents.append({
            "document_id": document_id,
            "source_file": source_file,
            "text": text,
            "content_sha256": digest,
            **metadata,
        })
    return documents


def _split_long_range(text: str, start: int, end: int, max_chars: int, overlap: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > max_chars:
        limit = cursor + max_chars
        lower = cursor + max_chars * 2 // 3
        boundary = max(text.rfind("\n", lower, limit), text.rfind(" ", lower, limit), text.rfind(". ", lower, limit))
        if boundary <= cursor:
            boundary = limit
        piece_end = boundary + (1 if text[boundary:boundary + 1] == "." else 0)
        ranges.append((cursor, piece_end))
        cursor = max(cursor + 1, piece_end - overlap)
        while cursor < end and text[cursor].isspace():
            cursor += 1
    if cursor < end:
        ranges.append((cursor, end))
    return ranges


def split_document(
    document: dict[str, Any],
    max_chars: int = 1800,
    overlap: int = 180,
) -> list[dict[str, Any]]:
    """Split text by numbered legal provisions and paragraphs without changing offsets."""
    if max_chars < 200 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Ожидаются max_chars >= 200 и 0 <= overlap < max_chars.")
    text = document["text"]
    lines: list[tuple[int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        line_end = offset + len(line)
        start = offset
        end = line_end
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            lines.append((start, end))
        offset = line_end
    if offset < len(text):
        start = offset
        end = len(text)
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            lines.append((start, end))

    chunks: list[dict[str, Any]] = []
    heading_path: list[str] = []
    current_locator: str | None = None
    for start, end in lines:
        line_text = text[start:end]
        numbered = _NUMBERED_LINE_RE.match(line_text)
        locator = current_locator
        referenced_section: str | None = None
        if numbered:
            prefix = "Статья " if numbered.group(1) else ""
            locator = f"{prefix}{numbered.group(2).rstrip('.')}"
            current_locator = locator
            depth = numbered.group(2).count(".") + 1
            if len(heading_path) >= depth:
                heading_path = heading_path[: depth - 1]
            rest = numbered.group(3).strip()
            if not rest or len(rest) < 90:
                heading_path.append(line_text)
        else:
            inline_reference = _INLINE_REFERENCE_RE.search(line_text)
            if inline_reference:
                referenced_section = inline_reference.group(1).rstrip(".")
                if (
                    document["document_type"] == "amendment"
                    and re.match(r"^\s*(?:изложить|дополнить|исключить|заменить)\b", line_text, re.IGNORECASE)
                ):
                    locator = referenced_section
                    current_locator = locator
        for piece_start, piece_end in _split_long_range(text, start, end, max_chars, overlap):
            piece_text = text[piece_start:piece_end]
            chunk_digest = hashlib.sha256(
                f"{document['document_id']}\0{piece_start}\0{piece_end}\0{piece_text}".encode("utf-8")
            ).hexdigest()[:24]
            chunks.append({
                **{key: value for key, value in document.items() if key != "text"},
                "chunk_id": chunk_digest,
                "section_reference": locator,
                "referenced_section": referenced_section,
                "heading_path": list(heading_path),
                "text": piece_text,
                "start_char": piece_start,
                "end_char": piece_end,
            })
    if not chunks:
        raise DocumentLoadError(f"Документ {document['source_file']!r} не содержит текста для индексации.")
    for position, chunk in enumerate(chunks):
        chunk["previous_chunk_id"] = chunks[position - 1]["chunk_id"] if position else None
        chunk["next_chunk_id"] = chunks[position + 1]["chunk_id"] if position + 1 < len(chunks) else None
    return chunks
