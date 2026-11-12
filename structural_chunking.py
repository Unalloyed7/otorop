"""Legal units with exact source ranges; physical line wraps are not chunk boundaries."""
from __future__ import annotations

import hashlib
import re
from typing import Any

ARTICLE = re.compile(r"^\s*Статья\s+(\d+(?:\.\d+)*)(?:\.(?=\s|$))?(?:\s+(.*)|$)", re.I)
POINT = re.compile(r"^\s*[«\"“]?\s*(\d+(?:\.\d+)*)([.)])?(?:\s+(.*)|$)")
CHAPTER = re.compile(r"^\s*(?:Глава|Раздел|Подраздел)\s+[\dIVXLCDMА-Я]+[.\s]", re.I)
ACTION = re.compile(r"^\s*(?:изложить|дополнить|исключить|заменить)\s+(?:пункт(?:а|е)?|п\.|статью|статьи)\s*(\d+(?:\.\d+)*)", re.I)
REFERENCE = re.compile(r"\b(?:пункт(?:а|е|ом)?|п\.|стать(?:я|и|е|ю)|ст\.)\s*(\d+(?:\.\d+)*)", re.I)
INSTRUCTION = re.compile(r"^(?:ВНИМАНИЕ\s+ДЛЯ\s+ИИ|system\s*:|ignore\s+(?:all|previous)\s+instructions)", re.I)
FOOTER = re.compile(r"^(?:поставщик|покупатель|арендодатель|арендатор)\s*:", re.I)
AMENDMENT_END = re.compile(r"^(?:Остальные условия|Настоящее дополнительное соглашение)", re.I)
ABBREVIATION = re.compile(r"(?:\b(?:г|гг|ст|п|подп|ч|руб|коп|им|ул|д|стр|ред)|т\.\s*д)\.$", re.I)


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _sentence_boundaries(text: str, start: int, end: int) -> list[int]:
    boundaries = []
    for match in re.finditer(r"[.!?;][»\"”)]*\s+", text[start:end]):
        stop = start + match.end()
        prefix = text[start:start + match.start() + 1]
        if not ABBREVIATION.search(prefix):
            boundaries.append(stop)
    return boundaries


def split_long_range(text: str, start: int, end: int, max_chars: int, overlap: int) -> list[tuple[int, int]]:
    """Prefer paragraphs, then sentences, then word boundaries; keep exact slices."""
    paragraphs = [start + m.end() for m in re.finditer(r"\r?\n\s*\r?\n", text[start:end])]
    sentences = _sentence_boundaries(text, start, end)
    ranges = []
    cursor = start
    while end - cursor > max_chars:
        limit = cursor + max_chars
        lower = cursor + max_chars // 2
        paragraph = [p for p in paragraphs if lower <= p <= limit]
        sentence = [p for p in sentences if lower <= p <= limit]
        if paragraph or sentence:
            boundary = (paragraph or sentence)[-1]
        else:
            spaces = list(re.finditer(r"\s+", text[cursor:limit]))
            boundary = cursor + spaces[-1].start() if spaces else limit
        piece_start, piece_end = _trim(text, cursor, boundary)
        if piece_end <= cursor:
            piece_end = limit
        ranges.append((piece_start, piece_end))
        target = max(cursor + 1, piece_end - overlap)
        # Overlap starts at a sentence/paragraph or at a whole word, never inside
        # a word unless a single unbroken token itself exceeds the size limit.
        candidates = [p for p in paragraphs + sentences if target <= p < piece_end]
        if candidates:
            next_cursor = min(candidates)
        elif target < len(text) and target > 0 and not text[target - 1].isspace():
            whitespace = re.search(r"\s+", text[target:piece_end])
            next_cursor = target + whitespace.end() if whitespace else boundary
        else:
            next_cursor = target
        cursor = max(cursor + 1, next_cursor)
        while cursor < end and text[cursor].isspace():
            cursor += 1
    tail = _trim(text, cursor, end)
    if tail[0] < tail[1]:
        ranges.append(tail)
    return ranges


def _is_title(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return bool(letters) and len(line) < 250 and not line.endswith((".", ";", ":", "!", "?")) and all(c.isupper() for c in letters)


def _blocks(document: dict[str, Any]) -> list[dict[str, Any]]:
    text = document["text"]
    amendment = document.get("document_type") == "amendment"
    conditions = document.get("applicability_conditions", [])
    blocks: list[dict[str, Any]] = []
    stack: list[tuple[int, int]] = []
    current: int | None = None
    active_amendment: str | None = None
    offset = 0
    blank = False
    for physical_line in text.splitlines(keepends=True):
        start, end = _trim(text, offset, offset + len(physical_line))
        offset += len(physical_line)
        if start == end:
            blank = True
            continue
        line = text[start:end]
        article = ARTICLE.match(line)
        point = POINT.match(line)
        # Dates, amounts and numbered lists (1), 2), ...) are not sibling clauses.
        if point and ((not point[2] and "." not in point[1]) or point[2] == ")"):
            point = None
        action = ACTION.match(line) if amendment else None
        condition = next((c for c in conditions if start <= c["start_char"] < end), None)
        kind, reference, level = "paragraph", None, None
        if action:
            kind, reference = "amendment_instruction", action[1].rstrip(".")
            active_amendment = reference
        elif article:
            kind, reference, level = "provision", "Статья " + article[1], 0
            stack = [s for s in stack if s[0] < 0]
        elif point:
            kind, reference = "provision", point[1].rstrip(".")
            level = reference.count(".") + 1
        elif amendment and condition:
            kind = "applicability_condition"
            reference = condition.get("target_reference") or active_amendment
        elif CHAPTER.match(line):
            kind, reference, level = "heading", line, -1
            stack = []
        elif _is_title(line):
            kind = "heading"
            stack = []
        elif INSTRUCTION.match(line):
            kind = "untrusted_instruction"
        elif FOOTER.match(line) or (amendment and AMENDMENT_END.match(line)):
            kind = "footer"
            active_amendment = None

        previous = blocks[current] if current is not None else None
        boundary = kind != "paragraph"
        if previous and kind == "paragraph":
            boundary = (
                previous["kind"] in {"heading", "untrusted_instruction", "footer", "amendment_instruction"}
                or (previous["kind"] == "paragraph" and blank)
            )
        if previous and not boundary:
            previous["end"] = end
        else:
            if level is not None:
                while stack and stack[-1][0] >= level:
                    stack.pop()
            ancestors = [index for _, index in stack]
            # Amendment replacement, directive and applicability are separate
            # structural units; their explicit relation is built by retrieval.
            if amendment and kind == "provision" and reference == active_amendment:
                ancestors = []
            block = {
                "start": start, "end": end, "reference": reference,
                "kind": kind, "ancestors": ancestors,
                "heading_path": [text[blocks[i]["start"]:blocks[i]["end"]] for i in ancestors
                                 if blocks[i]["kind"] == "heading" or len(text[blocks[i]["start"]:blocks[i]["end"]]) < 180],
            }
            blocks.append(block)
            current = len(blocks) - 1
            if level is not None:
                stack.append((level, current))
        blank = False
    return blocks


def split_structural_document(document: dict[str, Any], max_chars: int = 1800, overlap: int = 180) -> list[dict[str, Any]]:
    if max_chars < 200 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Ожидаются max_chars >= 200 и 0 <= overlap < max_chars.")
    text = document["text"]
    blocks = _blocks(document)
    chunks: list[dict[str, Any]] = []
    by_block: dict[int, list[dict[str, Any]]] = {}
    metadata = {key: value for key, value in document.items() if key != "text"}
    for index, block in enumerate(blocks):
        start, end = block["start"], block["end"]
        provision_id = hashlib.sha256(f"{document['document_id']}\0provision\0{start}\0{end}".encode()).hexdigest()[:24]
        ranges = split_long_range(text, start, end, max_chars, overlap)
        block_chunks = []
        for part, (piece_start, piece_end) in enumerate(ranges):
            piece_text = text[piece_start:piece_end]
            reference_text = piece_text
            if part == 0 and ARTICLE.match(reference_text):
                reference_text = reference_text[ARTICLE.match(reference_text).end():]
            cited = REFERENCE.search(reference_text)
            chunk_id = hashlib.sha256(f"{document['document_id']}\0{piece_start}\0{piece_end}\0{piece_text}".encode()).hexdigest()[:24]
            chunk = {
                **metadata, "chunk_id": chunk_id, "text": piece_text,
                "start_char": piece_start, "end_char": piece_end,
                "section_reference": block["reference"],
                "referenced_section": cited[1].rstrip(".") if cited else None,
                "heading_path": block["heading_path"],
                "chunk_role": block["kind"],
                "provision_id": provision_id,
                "provision_start_char": start, "provision_end_char": end,
                "part_index": part, "part_count": len(ranges),
            }
            block_chunks.append(chunk)
            chunks.append(chunk)
        by_block[index] = block_chunks
    for index, block_chunks in by_block.items():
        ancestors = [c["chunk_id"] for i in blocks[index]["ancestors"] for c in by_block[i]]
        siblings = [c["chunk_id"] for c in block_chunks]
        for chunk in block_chunks:
            chunk["ancestor_chunk_ids"] = ancestors
            chunk["provision_chunk_ids"] = siblings

    conditions = document.get("applicability_conditions", [])
    for chunk in chunks:
        chunk["applicability_conditions"] = [c for c in conditions if (
            chunk["start_char"] <= c["start_char"] and chunk["end_char"] >= c["end_char"]
        ) or (
            c.get("target_reference") and c["target_reference"] == chunk["section_reference"]
            and chunk["end_char"] <= c["start_char"]
        )]
    # Exact condition ranges remain first-class anchors for the provision graph.
    for condition in conditions:
        start, end = condition["start_char"], condition["end_char"]
        exact = next((c for c in chunks if c["start_char"] == start and c["end_char"] == end), None)
        if exact:
            exact["chunk_role"] = "applicability_condition"
            continue
        owner = next((c for c in chunks if c["provision_start_char"] <= start and c["provision_end_char"] >= end), None)
        piece = text[start:end]
        chunk_id = hashlib.sha256(f"{document['document_id']}\0{start}\0{end}\0{piece}".encode()).hexdigest()[:24]
        chunk = {
            **metadata, "chunk_id": chunk_id, "text": piece,
            "start_char": start, "end_char": end,
            "section_reference": condition.get("target_reference"), "referenced_section": None,
            "heading_path": owner["heading_path"] if owner else [],
            "chunk_role": "applicability_condition", "applicability_conditions": [condition],
            "provision_id": owner["provision_id"] if owner else chunk_id,
            "provision_start_char": owner["provision_start_char"] if owner else start,
            "provision_end_char": owner["provision_end_char"] if owner else end,
            "provision_chunk_ids": owner["provision_chunk_ids"] if owner else [chunk_id],
            "ancestor_chunk_ids": owner["ancestor_chunk_ids"] if owner else [],
        }
        chunks.append(chunk)
    chunks.sort(key=lambda item: (item["start_char"], item["end_char"], item["chunk_id"]))
    for index, chunk in enumerate(chunks):
        chunk["previous_chunk_id"] = chunks[index - 1]["chunk_id"] if index else None
        chunk["next_chunk_id"] = chunks[index + 1]["chunk_id"] if index + 1 < len(chunks) else None
    if not chunks:
        raise ValueError(f"Документ {document['source_file']!r} не содержит текста для индексации.")
    return chunks
