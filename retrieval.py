"""Offline BM25 retrieval for the organizer-provided knowledge base."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
import tracemalloc
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from documents import PROCESSOR_VERSION, load_documents, split_document
from llm_client import LLMClient, ModelAPIError, ModelProtocolError, Settings

RETRIEVAL_VERSION = "retrieval-v6"
EMBEDDING_CACHE_VERSION = 2
EMBEDDING_RETRY_BASE_SECONDS = 30.0
EMBEDDING_RETRY_MAX_SECONDS = 300.0
logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(
    r"[a-zа-яё]+\s*[-–—]\s*\d+(?:\s*/\s*\d+)?|"
    r"\d+(?:\s*[-–—/]\s*\d+)+|[a-zа-яё0-9]+",
    re.IGNORECASE,
)
_CONTRACT_RE = re.compile(
    r"(?<![А-ЯЁA-Z0-9])([А-ЯЁA-Z]{1,6}\s*[-–—]\s*\d{1,5}\s*/\s*\d{4})(?!\d)",
    re.IGNORECASE,
)
_REFERENCE_RE = re.compile(r"\b(?:пункт(?:е|а|ом)?|п\.|стать(?:я|е|и)|ст\.)\s*(\d+(?:\.\d+){0,5})", re.IGNORECASE)
_RU_SUFFIXES = sorted(
    {
        "иями", "ями", "ами", "иями", "ского", "скому", "скими", "ских", "ская",
        "ское", "ские", "ского", "ской", "скую", "шему", "шего", "щими", "ющему",
        "ющими", "ющего", "ющую", "ющее", "ющие", "аться", "яться", "иться",
        "аться", "енных", "енный", "ение", "ения", "ению", "ением", "аний",
        "ание", "ания", "анию", "анием", "остью", "остью", "остью", "иями",
        "ого", "ему", "ому", "ими", "ыми", "его", "ая", "яя", "ое", "ее",
        "ые", "ие", "ий", "ый", "ой", "ых", "ую", "юю", "ою", "ею", "ам", "ям",
        "ах", "ях", "ов", "ев", "ом", "ем", "ым", "им", "ей", "ой", "ия",
        "ии", "ию", "ать", "ить", "ется", "атся", "ится", "ю", "а", "я", "ы",
        "и", "о", "е", "у", "ь",
    },
    key=len,
    reverse=True,
)
_QUERY_SYNONYMS = {
    "пропуск": ("доступ", "проезд", "въезд"),
    "дефект": ("недостатк", "браков"),
    "недостатк": ("дефект", "браков"),
    "аванс": ("предоплат",),
    "отнест": ("переда", "сдат"),
    "юротдел": ("юридическ", "отдел"),
    "одобр": ("соглас", "утвержд"),
    "соглас": ("одобр", "утвержд"),
    "утвержд": ("одобр", "соглас"),
}
_QUERY_STOPWORDS = {
    "а", "и", "или", "по", "с", "в", "на", "к", "от", "до", "для", "что",
    "как", "где", "когда", "какой", "какая", "какие", "нас", "наш", "наши",
    "это", "ли", "бы", "же", "договор", "кто", "должен",
}


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().replace("ё", "е").replace("–", "-").replace("—", "-")).strip()


def _stem_token(token: str) -> str:
    if not re.fullmatch(r"[а-яё]+", token):
        return token
    if token.startswith("одобр"):
        return "одобр"
    if token.startswith("соглас"):
        return "соглас"
    if token.startswith(("утвержд", "утверд")):
        return "утвержд"
    if token.startswith("оплач"):
        return "оплат"
    if token.startswith(("гарант", "претенз")):
        return "гарант" if token.startswith("гарант") else "претенз"
    for suffix in _RU_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[:-len(suffix)]
    return token


def _tokens(text: str) -> list[str]:
    tokens = []
    for match in _TOKEN_RE.finditer(_normalize(text)):
        token = re.sub(r"\s+", "", match.group(0))
        if token:
            tokens.append(_stem_token(token))
    return tokens


def _contains_token_phrase(tokens: list[str], phrase: list[str]) -> bool:
    return bool(phrase) and any(tokens[start:start + len(phrase)] == phrase for start in range(len(tokens) - len(phrase) + 1))


def _normalize_contract(value: str) -> str:
    return re.sub(r"\s+", "", value.upper().replace("Ё", "Е")).replace("–", "-").replace("—", "-")


def _party_key(value: str) -> str:
    normalized = _normalize(value)
    normalized = re.sub(r"^(?:ооо|ао|пао|ип)\s*", "", normalized)
    return re.sub(r"[«»\"“”]", "", normalized).strip()


def _date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    result = str(value).strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", result):
        raise ValueError(f"Дата должна иметь формат ГГГГ-ММ-ДД: {value!r}")
    try:
        return date.fromisoformat(result).isoformat()
    except ValueError as exc:
        raise ValueError(f"Некорректная календарная дата: {value!r}") from exc


def _evaluate_condition(condition: dict[str, Any], specification_signed_at: str | None) -> str:
    if not specification_signed_at:
        return "unknown"
    if condition.get("operator") == "when_absent" or not condition.get("date"):
        return "unknown"
    date = condition["date"]
    operator = condition["operator"]
    if operator == "after":
        return "applicable" if specification_signed_at > date else "not_applicable"
    if operator == "before":
        return "applicable" if specification_signed_at < date else "not_applicable"
    if operator == "on_or_before":
        return "applicable" if specification_signed_at <= date else "not_applicable"
    if operator == "on_or_after":
        return "applicable" if specification_signed_at >= date else "not_applicable"
    return "unknown"


def evaluate_applicability(
    evidence: dict[str, Any],
    specification_signed_at: str | None = None,
    event_dates: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Evaluate explicit conditions without inferring missing dates or terms."""
    conditions = evidence.get("applicability_conditions") or []
    if not conditions:
        return {"status": "not_specified", "conditions": []}
    date = _date(specification_signed_at)
    normalized_event_dates = {
        kind: _date(value)
        for kind, value in (event_dates or {}).items()
        if value is not None
    }
    results = [
        {
            **condition,
            "status": _evaluate_condition(
                condition,
                date if condition.get("kind") == "specification_signed_at"
                else normalized_event_dates.get(condition.get("kind")),
            ),
        }
        for condition in conditions
    ]
    statuses = {result["status"] for result in results}
    if "not_applicable" in statuses:
        status = "not_applicable"
    elif statuses == {"applicable"}:
        status = "applicable"
    else:
        status = "unknown"
    return {"status": status, "conditions": results}


class SearchIndex:
    """In-memory index. Evidence dictionaries are JSON-serializable."""

    def __init__(self, documents: list[dict[str, Any]], max_chunk_chars: int = 1800) -> None:
        self.max_chunk_chars = max_chunk_chars
        self.documents: dict[str, dict[str, Any]] = {doc["document_id"]: doc for doc in documents}
        self.chunks = [
            chunk
            for document in documents
            for chunk in split_document(document, max_chars=self.max_chunk_chars)
        ]
        self._by_chunk_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self._position_by_chunk_id = {
            chunk["chunk_id"]: position
            for position, chunk in enumerate(self.chunks)
        }
        self._link_documents()
        self._link_provisions()
        self._build_bm25()
        manifest = [
            (doc["source_file"], doc["content_sha256"])
            for doc in sorted(documents, key=lambda item: item["source_file"])
        ]
        self.fingerprint = hashlib.sha256(
            json.dumps(
                [PROCESSOR_VERSION, RETRIEVAL_VERSION, self.max_chunk_chars, manifest],
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

    def _link_documents(self) -> None:
        ids = sorted(self.documents)
        for document in self.documents.values():
            contract = document.get("contract_number")
            related = [
                other_id
                for other_id in ids
                if other_id != document["document_id"]
                and contract
                and self.documents[other_id].get("contract_number") == contract
            ]
            explicit = set(related)
            text = _normalize(document["text"])
            mentions_policy_family = bool(re.search(
                r"\bрегламент[а-яё]*\s+согласован[а-яё]*\s+договор[а-яё]*",
                text,
            ))
            referenced_editions = {
                f"{int(match.group(1)):02d}.{int(match.group(2)):02d}.{match.group(3)}"
                for match in re.finditer(
                    r"(?:ред\.?|редакци[яи])\s*(?:от\s*)?(\d{1,2})[./](\d{1,2})[./](\d{4})",
                    text,
                )
            }
            for other_id in ids:
                if other_id == document["document_id"]:
                    continue
                other = self.documents[other_id]
                title = _normalize(other["title"])
                policy_reference = (
                    other["document_type"] == "internal_policy"
                    and mentions_policy_family
                    and (
                        document["document_type"] == "internal_policy"
                        or not referenced_editions
                        or (
                            other.get("edition")
                            and date.fromisoformat(other["edition"]).strftime("%d.%m.%Y") in referenced_editions
                        )
                    )
                )
                if policy_reference or (len(title) > 16 and title in text):
                    explicit.add(other_id)
            document["related_document_ids"] = sorted(related)
            document["linked_document_ids"] = sorted(explicit)
        reverse_links: dict[str, set[str]] = defaultdict(set)
        for document in self.documents.values():
            for linked_id in document["linked_document_ids"]:
                reverse_links[linked_id].add(document["document_id"])
        for document_id, linked_ids in reverse_links.items():
            document = self.documents[document_id]
            document["linked_document_ids"] = sorted(
                set(document["linked_document_ids"]) | linked_ids
            )
        for chunk in self.chunks:
            document = self.documents[chunk["document_id"]]
            chunk["related_document_ids"] = list(document["related_document_ids"])
            chunk["linked_document_ids"] = list(document["linked_document_ids"])

    def _link_provisions(self) -> None:
        action_pattern = re.compile(
            r"\b(изложить|дополнить|исключить|заменить)\s+"
            r"(?:пункт(?:а|е)?|п\.|статью|статьи)\s*(\d+(?:\.\d+){0,5})",
            re.IGNORECASE,
        )
        condition_pattern = re.compile(
            r"\b(?:за\s+исключением|исключая|кроме|если\s+иное|"
            r"при\s+условии|при\s+отсутствии|не\s+применяется)\b",
            re.IGNORECASE,
        )
        exception_pattern = re.compile(
            r"\b(?:за\s+исключением|исключая|кроме|не\s+применяется|исключение)\b",
            re.IGNORECASE,
        )

        def section_key(value: Any) -> str | None:
            if not value:
                return None
            return re.sub(r"^статья\s+", "", str(value), flags=re.IGNORECASE).rstrip(".")

        def provision_targets(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
            # Several pieces of one long clause are one reference target, not
            # competing provisions. Structural completion retrieves the rest.
            unique = {}
            for candidate in sorted(candidates, key=lambda c: c["start_char"]):
                key = (candidate["document_id"], candidate.get("provision_id", candidate["chunk_id"]))
                unique.setdefault(key, candidate)
            return list(unique.values())

        def add_relation(
            relation_type: str,
            from_chunk: dict[str, Any],
            to_chunk: dict[str, Any],
            quote_chunk: dict[str, Any],
            *,
            confidence: str,
            target_reference: str | None = None,
        ) -> dict[str, Any]:
            quote = quote_chunk["text"]
            identity = "\0".join((
                relation_type,
                from_chunk["chunk_id"],
                to_chunk["chunk_id"],
                quote_chunk["chunk_id"],
                confidence,
            ))
            relation = {
                "relation_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
                "type": relation_type,
                "from_chunk_id": from_chunk["chunk_id"],
                "to_chunk_id": to_chunk["chunk_id"],
                "from_document_id": from_chunk["document_id"],
                "to_document_id": to_chunk["document_id"],
                "target_reference": target_reference,
                "confidence": confidence,
                "quote": quote,
                "supporting_chunk_ids": [quote_chunk["chunk_id"]],
            }
            if not any(existing["relation_id"] == relation["relation_id"] for existing in self.provision_relations):
                self.provision_relations.append(relation)
                from_chunk["provision_relations"].append(relation)
                if to_chunk["chunk_id"] != from_chunk["chunk_id"]:
                    to_chunk["provision_relations"].append(relation)
                quote_chunk["provision_relations"].append(relation)
            return relation

        self.provision_relations: list[dict[str, Any]] = []
        self.provision_gaps: list[dict[str, Any]] = []
        for chunk in self.chunks:
            chunk["provision_relations"] = []
            chunk["provision_gaps"] = []
            chunk["local_conditions"] = []

        chunks_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in self.chunks:
            chunks_by_document[chunk["document_id"]].append(chunk)
        for entries in chunks_by_document.values():
            entries.sort(key=lambda item: (item["start_char"], item["end_char"], item["chunk_id"]))

        for document in self.documents.values():
            if document.get("document_type") != "amendment":
                continue
            source_chunks = chunks_by_document[document["document_id"]]
            contract_number = document.get("source_contract_number") or document.get("contract_number")
            for position, instruction in enumerate(source_chunks):
                action = action_pattern.search(instruction["text"])
                if not action:
                    continue
                verb = action.group(1).casefold()
                target_reference = action.group(2).rstrip(".")
                target_documents = [
                    candidate for candidate in self.documents.values()
                    if candidate.get("document_type") == "agreement"
                    and contract_number
                    and candidate.get("contract_number") == contract_number
                ]
                targets = [
                    candidate
                    for target_document in target_documents
                    for candidate in chunks_by_document[target_document["document_id"]]
                    if section_key(candidate.get("section_reference")) == target_reference
                ]
                if not targets:
                    gap = {
                        "kind": "referenced_provision_not_found",
                        "source_chunk_id": instruction["chunk_id"],
                        "target_reference": target_reference,
                        "contract_number": contract_number,
                        "source_text": instruction["text"],
                        "message": f"В базе не найден изменяемый пункт {target_reference} договора.",
                    }
                    self.provision_gaps.append(gap)
                    instruction["provision_gaps"].append(gap)
                    continue
                replacement = next(
                    (
                        candidate for candidate in source_chunks[position + 1:]
                        if section_key(candidate.get("section_reference")) == target_reference
                        and not action_pattern.search(candidate["text"])
                        and candidate.get("chunk_role") != "applicability_condition"
                        and not re.search(
                            r"^\s*(?:остальн\w+\s+услови|настояще[её]\s+дополнительное\s+соглашение|поставщик:|покупатель:)",
                            candidate["text"],
                            re.IGNORECASE,
                        )
                    ),
                    instruction,
                )
                relation_type = {
                    "изложить": "changes_provision",
                    "дополнить": "supplements_provision",
                    "исключить": "removes_provision",
                    "заменить": "replaces_provision",
                }[verb]
                for target in targets:
                    add_relation(
                        relation_type,
                        target,
                        replacement,
                        instruction,
                        confidence="explicit",
                        target_reference=target_reference,
                    )

        for document in self.documents.values():
            for source in chunks_by_document[document["document_id"]]:
                condition_records = source.get("applicability_conditions") or []
                if source.get("chunk_role") == "applicability_condition":
                    condition_records = [
                        condition for condition in condition_records
                        if condition.get("source_text") == source["text"]
                    ]
                for condition in condition_records:
                    target_reference = condition.get("target_reference")
                    if not target_reference:
                        continue
                    modified = [
                        relation for relation in self.provision_relations
                        if relation["type"] in {
                            "changes_provision", "supplements_provision", "replaces_provision"
                        }
                        and relation["to_document_id"] == document["document_id"]
                        and relation.get("target_reference") == target_reference
                    ]
                    condition_chunk = next(
                        (
                            candidate for candidate in chunks_by_document[document["document_id"]]
                            if condition.get("start_char") == candidate["start_char"]
                            and condition.get("end_char") == candidate["end_char"]
                        ),
                        source,
                    )
                    for change in modified:
                        changed_chunk = self._by_chunk_id[change["to_chunk_id"]]
                        add_relation(
                            "defines_applicability",
                            changed_chunk,
                            condition_chunk,
                            condition_chunk,
                            confidence="explicit",
                            target_reference=target_reference,
                        )

        policy_documents = [
            document for document in self.documents.values()
            if document.get("document_type") == "internal_policy"
        ]
        for older in policy_documents:
            for newer in policy_documents:
                if (
                    older["document_id"] == newer["document_id"]
                    or _normalize(older["title"]) != _normalize(newer["title"])
                    or not older.get("edition")
                    or not newer.get("edition")
                    or older["edition"] >= newer["edition"]
                ):
                    continue
                old_date = date.fromisoformat(older["edition"]).strftime("%d.%m.%Y")
                explicit_quote = next(
                    (
                        chunk for chunk in chunks_by_document[newer["document_id"]]
                        if re.search(
                            r"редакци[яи]\s+от\s+" + re.escape(old_date)
                            + r".{0,80}(?:утрат|утрач)\w*\s+силу",
                            chunk["text"],
                            re.IGNORECASE,
                        )
                    ),
                    None,
                )
                if explicit_quote is None:
                    continue
                old_sections: dict[str, dict[str, Any]] = {}
                new_sections: dict[str, dict[str, Any]] = {}
                for chunk in chunks_by_document[older["document_id"]]:
                    reference = section_key(chunk.get("section_reference"))
                    if reference:
                        old_sections.setdefault(reference, chunk)
                for chunk in chunks_by_document[newer["document_id"]]:
                    reference = section_key(chunk.get("section_reference"))
                    if reference:
                        new_sections.setdefault(reference, chunk)
                for reference in sorted(old_sections.keys() & new_sections.keys()):
                    add_relation(
                        "supersedes_edition",
                        old_sections[reference],
                        new_sections[reference],
                        explicit_quote,
                        confidence="explicit",
                        target_reference=reference,
                    )

        for document in self.documents.values():
            for source in chunks_by_document[document["document_id"]]:
                marker = condition_pattern.search(source["text"])
                if not marker:
                    continue
                is_exception = bool(exception_pattern.search(source["text"]))
                condition_kind = (
                    "exception" if is_exception
                    else "condition"
                )
                source["local_conditions"].append({
                    "kind": condition_kind,
                    "scope": "provision",
                    "applies_to_section": source.get("section_reference"),
                    "confidence": "explicit",
                    "source_chunk_id": source["chunk_id"],
                    "quote": source["text"],
                    "unknowns": [],
                })
                explicit_reference = source.get("referenced_section")
                if explicit_reference:
                    linked_documents = set(document.get("linked_document_ids", []))
                    linked_documents.add(document["document_id"])
                    targets = [
                        candidate
                        for linked_id in linked_documents
                        for candidate in chunks_by_document[linked_id]
                        if section_key(candidate.get("section_reference")) == section_key(explicit_reference)
                    ]
                    targets = provision_targets(targets)
                    if len(targets) == 1:
                        add_relation(
                            "establishes_exception" if condition_kind == "exception" else "references_provision",
                            targets[0],
                            source,
                            source,
                            confidence="explicit",
                            target_reference=str(explicit_reference),
                        )
                        continue
                current_section = section_key(source.get("section_reference"))
                if is_exception and current_section and "." in current_section:
                    parent = current_section.rsplit(".", 1)[0]
                    preceding = [
                        candidate for candidate in chunks_by_document[document["document_id"]]
                        if candidate["start_char"] < source["start_char"]
                        and (section_key(candidate.get("section_reference")) or "").startswith(parent + ".")
                        and section_key(candidate.get("section_reference")) != current_section
                    ]
                    if preceding:
                        nearest = preceding[-1]
                        add_relation(
                            "possible_exception_to",
                            nearest,
                            source,
                            source,
                            confidence="inferred",
                            target_reference=current_section,
                        )

        for document in self.documents.values():
            linked_document_ids = set(document.get("linked_document_ids", []))
            for source in chunks_by_document[document["document_id"]]:
                reference = source.get("referenced_section")
                if not reference or source.get("document_type") == "amendment":
                    continue
                if (
                    source.get("document_type") == "law"
                    and re.match(r"^\s*(?:источник|заголовок)\s*:", source["text"], re.IGNORECASE)
                ):
                    continue
                reference_text = source["text"]
                law_aliases = (
                    (r"\b(?:трудового\s+кодекса|тк\s*рф)\b", r"(?:тк|трудов)"),
                    (r"\b(?:налогового\s+кодекса|нк\s*рф)\b", r"(?:нк|налогов)"),
                    (r"\bкоап\b", r"коап"),
                    (r"\b(?:152[\s-]*фз|персональных\s+данных)\b", r"(?:152фз|152-фз)"),
                )
                target_document_ids = linked_document_ids | {document["document_id"]}
                for alias_pattern, filename_pattern in law_aliases:
                    if re.search(alias_pattern, reference_text, re.IGNORECASE):
                        target_document_ids.update(
                            target_id for target_id, target_document in self.documents.items()
                            if target_document.get("document_type") == "law"
                            and re.search(
                                filename_pattern,
                                target_document.get("source_file", ""),
                                re.IGNORECASE,
                            )
                        )
                referenced_targets = [
                    candidate
                    for linked_id in target_document_ids
                    for candidate in chunks_by_document[linked_id]
                    if section_key(candidate.get("section_reference")) == section_key(reference)
                ]
                referenced_targets = provision_targets(referenced_targets)
                if len(referenced_targets) == 1:
                    target = referenced_targets[0]
                    if target["chunk_id"] != source["chunk_id"]:
                        quote_type = (
                            "supplements_procedure"
                            if re.search(r"\bдополнени[ея]\b", source["text"], re.IGNORECASE)
                            else "references_provision"
                        )
                        add_relation(
                            quote_type,
                            source,
                            target,
                            source,
                            confidence="explicit",
                            target_reference=str(reference),
                        )
                elif len(referenced_targets) != 0:
                    gap = {
                        "kind": "referenced_provision_ambiguous",
                        "source_chunk_id": source["chunk_id"],
                        "target_reference": str(reference),
                        "candidate_chunk_ids": [candidate["chunk_id"] for candidate in referenced_targets],
                        "source_text": source["text"],
                        "message": f"Ссылка на пункт {reference} неоднозначна между положениями базы.",
                    }
                    source["provision_gaps"].append(gap)
                    self.provision_gaps.append(gap)
                else:
                    gap = {
                        "kind": "referenced_provision_not_found",
                        "source_chunk_id": source["chunk_id"],
                        "target_reference": str(reference),
                        "source_text": source["text"],
                        "message": f"Ссылка на пункт {reference} есть в источнике, но цель не найдена в связанной части базы.",
                    }
                    source["provision_gaps"].append(gap)
                    self.provision_gaps.append(gap)

    def _build_bm25(self) -> None:
        self._term_frequencies: list[Counter[str]] = []
        self._document_frequency: Counter[str] = Counter()
        lengths: list[int] = []
        for chunk in self.chunks:
            tokens = _tokens(chunk["text"])
            frequencies = Counter(tokens)
            self._term_frequencies.append(frequencies)
            self._document_frequency.update(frequencies.keys())
            lengths.append(len(tokens))
        self._average_length = sum(lengths) / max(len(lengths), 1)
        self._lengths = lengths

    def _bm25(self, query_tokens: list[str], position: int) -> float:
        frequencies = self._term_frequencies[position]
        length = self._lengths[position]
        score = 0.0
        for term in query_tokens:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = self._document_frequency[term]
            idf = math.log(1.0 + (len(self.chunks) - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.5 * (1.0 - 0.75 + 0.75 * length / max(self._average_length, 1))
            score += idf * frequency * 2.5 / denominator
        return score

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        *,
        ranking: str = "bm25",
        dense_scores: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Поисковый запрос не должен быть пустым.")
        if top_k < 1:
            raise ValueError("top_k должен быть не меньше 1.")
        if ranking not in {"bm25", "semantic", "hybrid"}:
            raise ValueError(f"Неизвестный режим поиска: {ranking!r}")
        if ranking != "bm25" and dense_scores is None:
            raise ValueError("Для семантического поиска нужны оценки dense retrieval.")
        filters = dict(filters or {})
        query_tokens = _tokens(query)
        query_contracts = {_normalize_contract(match.group(1)) for match in _CONTRACT_RE.finditer(query)}
        contract_tokens = {
            token
            for match in _CONTRACT_RE.finditer(query)
            for token in _tokens(match.group(1))
        }
        bm25_query_tokens = [
            token for token in query_tokens
            if token not in contract_tokens and token not in _QUERY_STOPWORDS
        ]
        query_aliases = sorted({
            synonym
            for token in query_tokens
            for synonym in _QUERY_SYNONYMS.get(token, ())
            if synonym not in query_tokens
        })
        alias_weight = (
            0.5
            if set(query_tokens).intersection({"одобр", "соглас", "утвержд"})
            else 3.0
        )
        normalized_query = _normalize(query)
        query_references = {match.group(1).rstrip(".") for match in _REFERENCE_RE.finditer(query)}
        as_of = _date(filters.get("as_of") or filters.get("valid_on") or filters.get("event_date"))
        spec_date = _date(filters.get("specification_signed_at"))
        signed_date = _date(filters.get("signed_at"))
        policy_date = as_of or signed_date or spec_date or datetime.now(timezone(timedelta(hours=3))).date().isoformat()
        snapshot_date = _date(filters.get("snapshot_date"))
        wanted_contract = filters.get("contract_number")
        wanted_contract = _normalize_contract(str(wanted_contract)) if wanted_contract else None
        requested_contract = wanted_contract or (
            next(iter(query_contracts)) if len(query_contracts) == 1 else None
        )
        wanted_party = _party_key(str(filters["counterparty"])) if filters.get("counterparty") else None
        wanted_parties = {
            _party_key(str(party))
            for party in filters.get("counterparty_candidates", [])
        }
        wanted_type = filters.get("document_type")
        wanted_document_id = filters.get("document_id")
        only_applicable = bool(filters.get("only_applicable", False))
        related_context_ids: set[str] = set()
        if requested_contract:
            for document in self.documents.values():
                if document.get("contract_number") == requested_contract:
                    related_context_ids.update(
                        linked_id
                        for linked_id in document.get("linked_document_ids", [])
                        if self.documents[linked_id].get("document_type") in {"internal_policy", "letter"}
                    )

        scored: list[tuple[float, int]] = []
        for position, chunk in enumerate(self.chunks):
            document = self.documents[chunk["document_id"]]
            if wanted_document_id and document["document_id"] != wanted_document_id:
                continue
            if requested_contract:
                document_contract = document.get("contract_number")
                if document_contract and document_contract != requested_contract:
                    continue
                document_type = document.get("document_type")
                globally_applicable = document_type in {"law", "internal_policy"}
                if (
                    not document_contract
                    and document["document_id"] not in related_context_ids
                    and not globally_applicable
                ):
                    continue
            if wanted_party:
                parties = {_party_key(name) for name in document.get("counterparties", [])}
                is_global_context = requested_contract and document.get("document_type") in {
                    "law", "internal_policy", "letter"
                }
                if (not parties or wanted_party not in parties) and not is_global_context:
                    continue
            if wanted_parties:
                parties = {_party_key(name) for name in document.get("counterparties", [])}
                is_global_context = requested_contract and document.get("document_type") in {
                    "law", "internal_policy", "letter"
                }
                if not parties.intersection(wanted_parties) and not is_global_context:
                    continue
            if wanted_type and document.get("document_type") != wanted_type:
                continue
            if document.get("document_type") == "internal_policy" and (
                (document.get("valid_from") and policy_date < document["valid_from"])
                or (document.get("valid_to") and policy_date >= document["valid_to"])
            ):
                continue
            if as_of and (
                (document.get("valid_from") and as_of < document["valid_from"])
                or (document.get("valid_to") and as_of >= document["valid_to"])
            ):
                continue
            if signed_date and document.get("signed_date") and document["signed_date"] != signed_date:
                continue
            if snapshot_date and document.get("snapshot_date") != snapshot_date:
                continue
            applicability = evaluate_applicability(
                chunk,
                spec_date,
                {
                    "contract_signed_at": signed_date,
                    "event": as_of,
                    "delivery": as_of,
                    "notice": as_of,
                },
            )
            if only_applicable and document.get("document_type") == "amendment":
                scoped_conditions = [
                    condition for condition in document.get("applicability_conditions", [])
                    if condition.get("target_reference")
                ]
                if scoped_conditions and not any(
                    condition.get("target_reference") == chunk.get("section_reference")
                    for condition in chunk.get("applicability_conditions", [])
                ):
                    continue
            if only_applicable and applicability["status"] == "not_applicable":
                continue

            score = self._bm25(bm25_query_tokens, position)
            if query_aliases:
                score += self._bm25(query_aliases, position) * alias_weight
            if (wanted_contract or query_contracts) and bm25_query_tokens:
                focus_tokens = set(bm25_query_tokens) | set(query_aliases)
                if not focus_tokens.intersection(self._term_frequencies[position]):
                    score *= 0.2
            chunk_text = _normalize(chunk["text"])
            if normalized_query and (
                normalized_query in chunk_text
                or normalized_query in _normalize(document["title"])
            ):
                score += 4.0
            if query_contracts and document.get("contract_number") in query_contracts:
                score += 2.5
            if any(
                _contains_token_phrase(query_tokens, _tokens(name))
                for name in document.get("counterparties") or []
            ):
                score += 2.0
            if query_references and chunk.get("section_reference"):
                locator = str(chunk["section_reference"]).casefold().replace("статья ", "").rstrip(".")
                if locator in query_references:
                    score += 4.0
            if spec_date and applicability["status"] == "applicable":
                score += 0.5
            dense_score = dense_scores.get(chunk["chunk_id"]) if dense_scores is not None else None
            if score > 0 or (ranking != "bm25" and dense_score is not None):
                scored.append((score, position))

        lexical_scores = {position: score for score, position in scored}
        if ranking == "semantic":
            scored = [
                (float(dense_scores[self.chunks[position]["chunk_id"]]), position)
                for _, position in scored
            ]
        elif ranking == "hybrid":
            lexical_order = sorted(
                lexical_scores,
                key=lambda position: (
                    -lexical_scores[position],
                    self.chunks[position]["source_file"],
                    self.chunks[position]["start_char"],
                ),
            )
            dense_order = sorted(
                (position for _, position in scored),
                key=lambda position: (
                    -float(dense_scores[self.chunks[position]["chunk_id"]]),
                    self.chunks[position]["source_file"],
                    self.chunks[position]["start_char"],
                ),
            )
            lexical_rank = {position: rank for rank, position in enumerate(lexical_order, 1)}
            dense_rank = {position: rank for rank, position in enumerate(dense_order, 1)}
            scored = [
                (
                    (1.0 / (60 + lexical_rank[position]) if position in lexical_rank else 0.0)
                    + (1.0 / (60 + dense_rank[position]) if position in dense_rank else 0.0),
                    position,
                )
                for position in dense_rank
            ]
        if requested_contract:
            best_contract_lexical = max(
                (
                    lexical_scores.get(position, 0.0)
                    for position, chunk in enumerate(self.chunks)
                    if self.documents[chunk["document_id"]].get("contract_number") == requested_contract
                ),
                default=0.0,
            )
            has_explicit_law_reference = bool(re.search(
                r"\b(?:закон|кодекс|стать[яеи]|ст\.)\b",
                normalized_query,
            )) or any(
                marker in normalized_query
                for marker in ("федеральный закон", "тк рф", "нк рф", "коап", "152-фз")
            )
            if best_contract_lexical > 0 and not has_explicit_law_reference:
                scored = [
                    (score, position)
                    for score, position in scored
                    if (
                        self.documents[self.chunks[position]["document_id"]].get("contract_number")
                        == requested_contract
                        or self.documents[self.chunks[position]["document_id"]].get("document_type")
                        not in {"law", "internal_policy"}
                        or lexical_scores.get(position, 0.0) >= best_contract_lexical * 0.8
                    )
                ]
        scored.sort(key=lambda item: (-item[0], self.chunks[item[1]]["source_file"], self.chunks[item[1]]["start_char"]))
        if requested_contract:
            best_per_document: dict[str, tuple[float, int]] = {}
            for score, position in scored:
                document_id = self.chunks[position]["document_id"]
                document = self.documents[document_id]
                if document.get("contract_number") != requested_contract:
                    continue
                best_per_document.setdefault(document_id, (score, position))
            pinned = sorted(
                best_per_document.values(),
                key=lambda item: (-item[0], self.chunks[item[1]]["source_file"], self.chunks[item[1]]["start_char"]),
            )
            pinned_positions = {position for _, position in pinned}
            target_references = {
                self.chunks[position]["section_reference"]
                for _, position in pinned
                if self.chunks[position].get("section_reference")
            }
            matching_reference_chunks: list[tuple[float, int]] = []
            if target_references:
                matching_by_document_and_reference: dict[
                    tuple[str, str], tuple[float, int]
                ] = {}
                for score, position in scored:
                    chunk = self.chunks[position]
                    document = self.documents[chunk["document_id"]]
                    reference = chunk.get("section_reference")
                    if (
                        document.get("contract_number") == requested_contract
                        and reference in target_references
                    ):
                        matching_by_document_and_reference.setdefault(
                            (chunk["document_id"], reference),
                            (score, position),
                        )
                for document_id, document in self.documents.items():
                    if document.get("contract_number") != requested_contract:
                        continue
                    for reference in target_references:
                        best_match = matching_by_document_and_reference.get(
                            (document_id, reference)
                        )
                        if best_match:
                            if best_match[1] not in pinned_positions:
                                matching_reference_chunks.append(best_match)
                                pinned_positions.add(best_match[1])
            pinned.extend(matching_reference_chunks)
            general_by_document: dict[str, tuple[float, int]] = {}
            for score, position in scored:
                document = self.documents[self.chunks[position]["document_id"]]
                if (
                    document.get("document_type") in {"law", "internal_policy"}
                    or document["document_id"] in related_context_ids
                ) and lexical_scores.get(position, 0.0) > 0:
                    general_by_document.setdefault(document["document_id"], (score, position))
            pinned.extend(
                item for item in general_by_document.values()
                if item[1] not in pinned_positions
            )
            pinned_positions.update(position for _, position in general_by_document.values())
            scored = pinned + [
                (score, position) for score, position in scored
                if position not in pinned_positions
            ]
        results: list[dict[str, Any]] = []
        for score, position in scored[:top_k]:
            evidence = dict(self.chunks[position])
            evidence["score"] = round(score, 6)
            evidence["applicability"] = evaluate_applicability(
                evidence,
                spec_date,
                {
                    "contract_signed_at": signed_date,
                    "event": as_of,
                    "delivery": as_of,
                    "notice": as_of,
                },
            )
            evidence["retrieval_reason"] = ranking
            results.append(evidence)
        if top_k > 1 and results:
            available_positions = {position: score for score, position in scored}
            result_positions = {position for _, position in scored[:top_k]}
            context: dict[int, float] = {}
            for anchor_score, anchor_position in scored[: min(3, top_k)]:
                anchor = self.chunks[anchor_position]
                reference = anchor.get("section_reference")
                if not reference or "." not in reference or reference.startswith("Статья "):
                    continue
                parent = reference.rsplit(".", 1)[0]
                for neighbor_id in (anchor.get("previous_chunk_id"), anchor.get("next_chunk_id")):
                    neighbor_position = self._position_by_chunk_id.get(neighbor_id)
                    if neighbor_position is None or neighbor_position not in available_positions:
                        continue
                    neighbor = self.chunks[neighbor_position]
                    neighbor_reference = neighbor.get("section_reference")
                    if (
                        neighbor["document_id"] == anchor["document_id"]
                        and neighbor_reference
                        and "." in neighbor_reference
                        and neighbor_reference.rsplit(".", 1)[0] == parent
                        and neighbor_position not in result_positions
                    ):
                        context[neighbor_position] = max(
                            context.get(neighbor_position, 0.0),
                            available_positions[neighbor_position],
                            anchor_score * 0.6,
                        )
            for position, score in context.items():
                evidence = dict(self.chunks[position])
                evidence["score"] = round(score, 6)
                evidence["applicability"] = evaluate_applicability(
                    evidence,
                    spec_date,
                    {
                        "contract_signed_at": signed_date,
                        "event": as_of,
                        "delivery": as_of,
                        "notice": as_of,
                    },
                )
                evidence["retrieval_reason"] = "sibling_context"
                results.append(evidence)
            results.sort(key=lambda item: (-item["score"], item["source_file"], item["start_char"]))
            results = results[:top_k]
        return results

    def get_document(self, document_id: str) -> dict[str, Any]:
        try:
            return dict(self.documents[document_id])
        except KeyError as exc:
            raise KeyError(f"Неизвестный document_id: {document_id}") from exc

    def get_chunk(self, chunk_id: str) -> dict[str, Any]:
        try:
            return dict(self._by_chunk_id[chunk_id])
        except KeyError as exc:
            raise KeyError(f"Неизвестный chunk_id: {chunk_id}") from exc

    def get_neighbors(self, chunk_id: str, window: int = 1) -> list[dict[str, Any]]:
        if window < 0:
            raise ValueError("window не должен быть отрицательным.")
        chunk = self.get_chunk(chunk_id)
        document_chunks = [
            item for item in self.chunks if item["document_id"] == chunk["document_id"]
        ]
        position = next(i for i, item in enumerate(document_chunks) if item["chunk_id"] == chunk_id)
        return [
            dict(item)
            for item in document_chunks[max(0, position - window): position + window + 1]
        ]

    def stats(self) -> dict[str, Any]:
        payload = [
            {key: value for key, value in chunk.items() if key not in {"previous_chunk_id", "next_chunk_id"}}
            for chunk in self.chunks
        ]
        index_bytes = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        return {
            "fingerprint": self.fingerprint,
            "document_count": len(self.documents),
            "chunk_count": len(self.chunks),
            "serialized_index_bytes": index_bytes,
        }


def build_index(kb_path: str | Path) -> SearchIndex:
    """Build an index from exactly the knowledge-base path supplied by the caller."""
    return SearchIndex(load_documents(kb_path))


class SchemaSearchBackend:
    """Adapt the local index to the shared async SearchBackend contract."""

    def __init__(self, index: SearchIndex, client: Any = None, top_k: int = 10) -> None:
        self.index = index
        self.client = client
        self.top_k = top_k
        self._embedding_lock = asyncio.Lock()
        self._document_vectors: dict[str, list[float]] | None = None
        self.diagnostics: list[str] = []
        self.embedding_status = "not_configured"
        self.reranker_status = "not_configured"
        self._embedding_failures = 0
        self._embedding_retry_after = 0.0
        self._embedding_cache_file: Path | None = None

    def _cache_path(self) -> Path:
        if self._embedding_cache_file is None:
            cache_root = os.environ.get("OTOROP_EMBEDDING_CACHE_DIR")
            if not cache_root:
                cache_root = str(
                    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
                    / "otorop"
                    / "embeddings"
                )
            cache_identity = json.dumps(
                {
                    "cache_version": EMBEDDING_CACHE_VERSION,
                    "fingerprint": self.index.fingerprint,
                    "model": self.client.settings.embedding_model,
                    "base_url": self.client.settings.base_url,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            cache_name = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()
            self._embedding_cache_file = Path(cache_root) / f"{cache_name}.json"
        return self._embedding_cache_file

    def _embedding_inputs(self) -> dict[str, str]:
        return {
            chunk["chunk_id"]: "\n".join(dict.fromkeys(
                value for value in (chunk["title"], *chunk.get("heading_path", []),
                                    chunk.get("section_reference"), chunk["text"]) if value
            ))
            for chunk in self.index.chunks
        }

    def _load_cached_vectors(self, inputs: dict[str, str]) -> dict[str, list[float]] | None:
        cache_path = self._cache_path()
        try:
            with cache_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
            if (
                not isinstance(payload, dict)
                or payload.get("cache_version") != EMBEDDING_CACHE_VERSION
                or payload.get("fingerprint") != self.index.fingerprint
                or payload.get("model") != self.client.settings.embedding_model
                or payload.get("base_url") != self.client.settings.base_url
            ):
                raise ValueError("метаданные кэша не совпадают с индексом или моделью")
            entries = payload.get("entries")
            if not isinstance(entries, dict):
                raise ValueError("в кэше нет таблицы векторов")
            vectors: dict[str, list[float]] = {}
            dimension: int | None = None
            invalid_entries = 0
            for chunk_id, entry in entries.items():
                if chunk_id not in inputs or not isinstance(entry, dict):
                    invalid_entries += 1
                    continue
                expected_hash = hashlib.sha256(inputs[chunk_id].encode("utf-8")).hexdigest()
                if entry.get("input_sha256") != expected_hash:
                    invalid_entries += 1
                    continue
                raw_vector = entry.get("vector")
                if (
                    not isinstance(raw_vector, list)
                    or not raw_vector
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in raw_vector)
                ):
                    invalid_entries += 1
                    continue
                if dimension is None:
                    dimension = len(raw_vector)
                elif len(raw_vector) != dimension:
                    invalid_entries += 1
                    continue
                vectors[chunk_id] = [float(value) for value in raw_vector]
            if invalid_entries:
                diagnostic = f"пропущено повреждённых или устаревших векторов в кэше: {invalid_entries}"
                self.diagnostics.append(diagnostic)
                logger.warning("retrieval: %s", diagnostic)
            return vectors
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            diagnostic = f"кэш embeddings проигнорирован: {exc}"
            self.diagnostics.append(diagnostic)
            logger.warning("retrieval: %s", diagnostic)
            return None

    def _save_cached_vectors(
        self,
        vectors: dict[str, list[float]],
        inputs: dict[str, str],
    ) -> None:
        cache_path = self._cache_path()
        temporary_path: Path | None = None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=cache_path.parent,
                prefix=".embedding-cache-",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                json.dump(
                    {
                        "cache_version": EMBEDDING_CACHE_VERSION,
                        "fingerprint": self.index.fingerprint,
                        "model": self.client.settings.embedding_model,
                        "base_url": self.client.settings.base_url,
                        "entries": {
                            chunk_id: {
                                "input_sha256": hashlib.sha256(
                                    inputs[chunk_id].encode("utf-8")
                                ).hexdigest(),
                                "vector": vector,
                            }
                            for chunk_id, vector in vectors.items()
                        },
                    },
                    stream,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, cache_path)
        except (OSError, TypeError, ValueError) as exc:
            diagnostic = f"кэш embeddings не сохранён; векторы останутся в памяти: {exc}"
            self.diagnostics.append(diagnostic)
            logger.warning("retrieval: %s", diagnostic)
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as cleanup_exc:
                    cleanup_diagnostic = f"временный файл кэша не удалён: {cleanup_exc}"
                    self.diagnostics.append(cleanup_diagnostic)
                    logger.warning("retrieval: %s", cleanup_diagnostic)

    def get_document(self, document_id: str) -> dict[str, Any]:
        return self.index.get_document(document_id)

    def get_chunk(self, chunk_id: str) -> dict[str, Any]:
        return self.index.get_chunk(chunk_id)

    def get_neighbors(self, chunk_id: str, window: int = 1) -> list[dict[str, Any]]:
        return self.index.get_neighbors(chunk_id, window=window)

    async def search_with_mode(
        self,
        query: str,
        filters: Any,
        mode: str,
        *,
        apply_reranker: bool = False,
        top_k: int | None = None,
    ) -> Any:
        if mode not in {"bm25", "semantic", "hybrid"}:
            raise ValueError(f"Неизвестный режим поиска: {mode!r}")
        from schemas import SearchResult

        filter_values = (
            filters.model_dump(mode="json", exclude_none=True)
            if hasattr(filters, "model_dump")
            else dict(filters or {})
        )
        index_filters: dict[str, Any] = {}
        for key in ("contract_number", "counterparty", "document_id"):
            if filter_values.get(key):
                index_filters[key] = filter_values[key]
        candidates: list[str] = []
        requested_party = index_filters.get("counterparty")
        if requested_party:
            parties = sorted({
                party
                for document in self.index.documents.values()
                for party in document.get("counterparties", [])
            })
            normalized_party = _party_key(requested_party)
            exact_parties = [party for party in parties if _party_key(party) == normalized_party]
            near_parties = [
                party for party in parties
                if normalized_party in _party_key(party)
            ]
            if exact_parties:
                index_filters["counterparty"] = exact_parties[0]
            elif len(near_parties) == 1:
                index_filters["counterparty"] = near_parties[0]
            elif len(near_parties) > 1:
                candidates = near_parties
                index_filters.pop("counterparty")
                index_filters["counterparty_candidates"] = near_parties
        event_date = filter_values.get("event_date")
        date_kind = filter_values.get("date_kind")
        if event_date:
            if date_kind in {"specification_signed", "specification_signed_at"}:
                index_filters["specification_signed_at"] = event_date
            elif date_kind in {"contract_signed", "document_signed"}:
                index_filters["signed_at"] = event_date
            elif date_kind in {"law_snapshot", "snapshot", "snapshot_date"}:
                index_filters["snapshot_date"] = event_date
            elif date_kind in {"event", "delivery", "delivered", "as_of", "document_validity"}:
                index_filters["as_of"] = event_date
            else:
                raise ValueError(f"Неизвестный тип даты события: {date_kind!r}")

        result_limit = self.top_k if top_k is None else top_k
        if result_limit < 1:
            raise ValueError("top_k должен быть не меньше 1.")
        dense_scores = await self._dense_scores(query) if mode in {"semantic", "hybrid"} else None
        if mode != "bm25" and dense_scores is None:
            raise ModelAPIError("embeddings: семантический режим недоступен")
        evidence_items = self.index.search(
            query,
            index_filters,
            top_k=result_limit,
            ranking=mode,
            dense_scores=dense_scores,
        )
        if (
            self.client is not None
            and self.client.settings.reranker_model
            and evidence_items
            and apply_reranker
        ):
            try:
                ranked_indices = await self.client.rerank(
                    query,
                    [item["title"] + "\n" + item["text"] for item in evidence_items],
                    top_n=min(self.client.settings.reranker_top_n, result_limit),
                )
                reranked = [evidence_items[position] for position in ranked_indices]
                query_contracts = {
                    _normalize_contract(match.group(1))
                    for match in _CONTRACT_RE.finditer(query)
                }
                contract_number = index_filters.get("contract_number") or (
                    next(iter(query_contracts)) if len(query_contracts) == 1 else None
                )
                related = [
                    item for item in evidence_items
                    if contract_number and item.get("contract_number") == contract_number
                ]
                evidence_items = list({
                    item["chunk_id"]: item for item in [*reranked, *related]
                }.values())
                self.reranker_status = "ready"
            except ModelAPIError as exc:
                self.reranker_status = "disabled_after_error"
                diagnostic = f"reranker недоступен; использован {mode}: {exc}"
                self.diagnostics.append(diagnostic)
                logger.warning("retrieval: %s", diagnostic)
        elif self.client is not None and self.client.settings.reranker_model and apply_reranker:
            self.reranker_status = "ready_not_needed"

        return SearchResult(evidence=self._as_evidence(evidence_items), candidates=candidates)

    def _as_evidence(self, evidence_items):
        from schemas import Applicability, Coordinates, Evidence

        converted = []
        for item in evidence_items:
            document = self.index.get_document(item["document_id"])
            snapshot = date.fromisoformat(item["snapshot_date"]) if item.get("snapshot_date") else None
            current_only = bool(item.get("current_only"))
            condition_records = item.get("applicability_conditions", [])
            conditions = [
                str(condition.get("source_text") or condition)
                for condition in condition_records
            ]
            for date_key in ("valid_from_source", "valid_to_source"):
                if item.get(date_key) and item[date_key] not in conditions:
                    conditions.append(item[date_key])
            if item.get("renewal"):
                conditions.append(item["renewal"]["source_text"])
            applicability_result = item.get("applicability") or {}
            source_date_kinds = {
                condition.get("kind")
                for condition in condition_records
                if condition.get("kind")
            }
            metadata = {
                key: item[key]
                for key in (
                    "document_type",
                    "source_contract_number",
                    "amendment_number",
                    "counterparties",
                    "signed_date",
                    "signed_date_source",
                    "approval_date",
                    "approval_date_source",
                    "effective_date",
                    "effective_date_source",
                    "valid_from_source",
                    "valid_to_source",
                    "snapshot_date",
                    "snapshot_date_source",
                    "current_only",
                    "related_document_ids",
                    "linked_document_ids",
                    "applicability_conditions",
                    "uncertainties",
                    "date_uncertainties",
                    "date_sources",
                    "provision_date_sources",
                    "renewal",
                    "heading_path",
                    "chunk_role",
                    "provision_id",
                    "provision_start_char",
                    "provision_end_char",
                    "provision_chunk_ids",
                    "ancestor_chunk_ids",
                    "part_index",
                    "part_count",
                    "local_conditions",
                    "provision_relations",
                    "provision_gaps",
                    "score",
                    "retrieval_reason",
                )
                if key in item
            }
            # resolved различает известность результата, но не его знак.
            metadata["applicability_status"] = applicability_result.get("status", "not_specified")
            converted.append(Evidence(
                document_id=item["document_id"],
                chunk_id=item["chunk_id"],
                document_title=item["title"],
                section=item.get("section_reference") or "Фрагмент без структурного номера",
                text=item["text"],
                coordinates=Coordinates(
                    source_path=item["source_file"],
                    char_start=item["start_char"],
                    char_end=item["end_char"],
                ),
                applicability=Applicability(
                    edition=item.get("edition") or (
                        f"Действующая редакция; выгрузка {snapshot.strftime('%d.%m.%Y')}"
                        if current_only and snapshot else "Редакция в источнике не указана"
                    ),
                    valid_from=item.get("valid_from"),
                    valid_to=item.get("valid_to"),
                    date_kind="event" if document.get("document_type") == "internal_policy" else (next(iter(source_date_kinds)) if len(source_date_kinds) == 1 else None),
                    conditions=conditions,
                    current_only=current_only,
                    snapshot_date=snapshot,
                    resolved=applicability_result.get("status") in {"applicable", "not_applicable"},
                ),
                contract_number=item.get("contract_number"),
                counterparty=item.get("counterparty"),
                metadata=metadata,
            ))
        return converted

    def bundle_result(self, bundle: dict[str, Any]) -> Any:
        """Adapt dependency-closed evidence without changing the public schema."""
        from schemas import SearchResult

        evidence = self._as_evidence(bundle["evidence"])
        groups = bundle["groups"]
        for item in evidence:
            item.metadata["bundle_parts"] = [group["part"] for group in groups
                                              if item.chunk_id in group["evidence_chunk_ids"]]
            item.metadata["bundle_truncated"] = bundle["truncated"]
        missing = list(bundle["missing_parts"])
        if bundle["truncated"]:
            missing.append("Часть связанных оснований не поместилась в бюджет контекста; полный комплект не проверен")
        clarifications = list(dict.fromkeys(g["clarification"] for g in groups if g.get("clarification")))
        candidates = list(dict.fromkeys(p for g in groups for p in g.get("candidates", [])))
        return SearchResult(evidence=evidence, missing_info=missing,
                            clarification="; ".join(clarifications) or None, candidates=candidates)

    async def search_bundle(
        self,
        parts: Sequence[str],
        filters: Any,
        *,
        top_k: int | None = None,
        max_context_chars: int = 12000,
        max_context_chunks: int | None = None,
        query_context: str | None = None,
        existing_chunk_ids: Sequence[str] = (),
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Search question parts and return dependency-closed, provenance-safe groups."""
        normalized_parts = [part.strip() for part in parts if isinstance(part, str) and part.strip()]
        if not normalized_parts:
            raise ValueError("Для комплектования оснований нужна хотя бы одна часть вопроса.")
        if max_context_chars < 1:
            raise ValueError("max_context_chars должен быть положительным.")
        if max_context_chunks is not None and max_context_chunks < 1:
            raise ValueError("max_context_chunks должен быть положительным.")
        result_limit = self.top_k if top_k is None else top_k
        if result_limit < 1:
            raise ValueError("top_k должен быть не меньше 1.")

        direct_by_group: list[list[dict[str, Any]]] = []
        group_sources: list[dict[str, Any]] = []
        direct_results: dict[str, Any] = {}
        scores_by_chunk: dict[str, float] = {}
        filter_values = (
            filters.model_dump(mode="json", exclude_none=True)
            if hasattr(filters, "model_dump")
            else dict(filters or {})
        )
        specification_date = filter_values.get("specification_signed_at")
        if filter_values.get("date_kind") in {"specification_signed", "specification_signed_at"}:
            specification_date = filter_values.get("event_date") or specification_date
        specification_date = _date(specification_date)
        bundle_event_dates: dict[str, str | None] = {
            "contract_signed_at": filter_values.get("signed_at"),
            "event": filter_values.get("as_of") or filter_values.get("event_date"),
            "delivery": filter_values.get("as_of") or filter_values.get("event_date"),
            "notice": filter_values.get("as_of") or filter_values.get("event_date"),
        }
        for part in normalized_parts:
            search_query = part if not query_context or part == query_context else part + "\n" + query_context
            if part not in direct_results:
                if mode is None:
                    direct_results[part] = await self.search(search_query, filters)
                else:
                    direct_results[part] = await self.search_with_mode(
                        search_query,
                        filters,
                        mode,
                        top_k=result_limit,
                    )
            result = direct_results[part]
            direct_items = []
            for evidence in result.evidence[:result_limit]:
                item = self.index.get_chunk(evidence.chunk_id)
                item["score"] = evidence.metadata.get("score", 0.0)
                scores_by_chunk[evidence.chunk_id] = float(item["score"] or 0.0)
                item["applicability"] = evaluate_applicability(
                    item,
                    specification_date,
                    bundle_event_dates,
                )
                direct_items.append(item)
            direct_by_group.append(direct_items)
            group_sources.append({
                "part": part,
                "query": search_query,
                "direct_chunk_ids": [item["chunk_id"] for item in direct_items],
                "missing": not bool(direct_items),
                "clarification": result.clarification,
                "candidates": list(result.candidates),
            })

        relations_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for relation in self.index.provision_relations:
            relations_by_chunk[relation["from_chunk_id"]].append(relation)
            if relation["to_chunk_id"] != relation["from_chunk_id"]:
                relations_by_chunk[relation["to_chunk_id"]].append(relation)

        def dependency_closure(seed_id: str) -> tuple[set[str], dict[str, dict[str, Any]]]:
            chunk_ids = {seed_id}
            found_relations: dict[str, dict[str, Any]] = {}
            frontier = [seed_id]
            for _ in range(3):
                next_frontier: list[str] = []
                for chunk_id in frontier:
                    chunk = self.index.get_chunk(chunk_id)
                    for related_id in (*chunk.get("provision_chunk_ids", []), *chunk.get("ancestor_chunk_ids", [])):
                        if related_id not in chunk_ids:
                            chunk_ids.add(related_id)
                            next_frontier.append(related_id)
                    for relation in relations_by_chunk.get(chunk_id, []):
                        found_relations[relation["relation_id"]] = relation
                        for related_id in (
                            relation["from_chunk_id"],
                            relation["to_chunk_id"],
                            *relation["supporting_chunk_ids"],
                        ):
                            if related_id not in chunk_ids:
                                chunk_ids.add(related_id)
                                next_frontier.append(related_id)
                frontier = next_frontier
                if not frontier:
                    break
            # The graph radius is bounded, but an included legal unit is never
            # partial: complete all of its pieces and ancestor introductions,
            # even when the graph reached it at the last hop.
            frontier = list(chunk_ids)
            while frontier:
                chunk = self.index.get_chunk(frontier.pop())
                for related_id in (*chunk.get("provision_chunk_ids", []), *chunk.get("ancestor_chunk_ids", [])):
                    if related_id not in chunk_ids:
                        chunk_ids.add(related_id)
                        frontier.append(related_id)
            return chunk_ids, found_relations

        admitted_ids: set[str] = set(existing_chunk_ids)
        for chunk_id in admitted_ids:
            self.index.get_chunk(chunk_id)
        admitted_relations: dict[str, dict[str, Any]] = {}
        omitted_ids: set[str] = set()
        group_records: list[dict[str, Any]] = []

        def evidence_cost(chunk_ids: set[str]) -> int:
            # The engine sends individual Evidence rows, including overlap.
            # Budget those rows, not just their merged unique source ranges.
            return sum(len(self.index.get_chunk(chunk_id)["text"]) for chunk_id in chunk_ids)

        if evidence_cost(admitted_ids) > max_context_chars or (
            max_context_chunks is not None and len(admitted_ids) > max_context_chunks
        ):
            raise ValueError("Уже переданный контекст превышает бюджет комплекта.")

        for group, direct_items in zip(group_sources, direct_by_group):
            ordered_hits = sorted(
                direct_items,
                key=lambda item: (
                    -float(item.get("score") or 0.0),
                    item["source_file"],
                    item["start_char"],
                ),
            )
            group_ids: set[str] = set()
            group_relations: dict[str, dict[str, Any]] = {}
            group_omitted: set[str] = set()
            for hit in ordered_hits:
                bundle_ids, bundle_relations = dependency_closure(hit["chunk_id"])
                candidate_ids = admitted_ids | bundle_ids
                if evidence_cost(candidate_ids) > max_context_chars or (
                    max_context_chunks is not None and len(candidate_ids) > max_context_chunks
                ):
                    group_omitted.update(bundle_ids - admitted_ids)
                    omitted_ids.update(bundle_ids - admitted_ids)
                    continue
                admitted_ids.update(bundle_ids)
                group_ids.update(bundle_ids)
                group_relations.update(bundle_relations)
                admitted_relations.update(bundle_relations)
            group_records.append({
                **group,
                "evidence_chunk_ids": sorted(group_ids),
                "relation_ids": sorted(group_relations),
                "truncated": bool(group_omitted),
                "omitted_chunk_ids": sorted(group_omitted),
            })

        def evidence_payload(chunk_id: str) -> dict[str, Any]:
            chunk = self.index.get_chunk(chunk_id)
            payload = {
                key: value
                for key, value in chunk.items()
                if key not in {"previous_chunk_id", "next_chunk_id"}
            }
            payload["applicability"] = evaluate_applicability(
                payload,
                specification_date,
                bundle_event_dates,
            )
            if chunk_id in scores_by_chunk:
                payload["score"] = scores_by_chunk[chunk_id]
            return payload

        ranked_ids = list(dict.fromkeys(item["chunk_id"] for group in direct_by_group for item in group))
        ranked_ids.extend(sorted(admitted_ids - set(ranked_ids), key=lambda chunk_id: (
            self.index.get_chunk(chunk_id)["source_file"], self.index.get_chunk(chunk_id)["start_char"])))
        evidence_payloads = [evidence_payload(chunk_id) for chunk_id in ranked_ids if chunk_id in admitted_ids]
        context_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk_id in admitted_ids:
            context_by_document[self.index.get_chunk(chunk_id)["document_id"]].append(
                self.index.get_chunk(chunk_id)
            )
        context_spans: list[dict[str, Any]] = []
        for document_id, chunks in context_by_document.items():
            document = self.index.get_document(document_id)
            intervals: list[dict[str, Any]] = []
            for chunk in sorted(chunks, key=lambda item: (item["start_char"], item["end_char"])):
                if intervals and chunk["start_char"] <= intervals[-1]["end_char"]:
                    interval = intervals[-1]
                    interval["end_char"] = max(interval["end_char"], chunk["end_char"])
                    interval["source_chunk_ids"].append(chunk["chunk_id"])
                else:
                    intervals.append({
                        "document_id": document_id,
                        "source_file": document["source_file"],
                        "start_char": chunk["start_char"],
                        "end_char": chunk["end_char"],
                        "source_chunk_ids": [chunk["chunk_id"]],
                    })
            for interval in intervals:
                interval["text"] = document["text"][interval["start_char"]:interval["end_char"]]
                context_spans.append(interval)

        relevant_gaps = [
            gap for gap in self.index.provision_gaps
            if gap.get("source_chunk_id") in admitted_ids
            or gap.get("source_chunk_id") in omitted_ids
        ]
        return {
            "groups": group_records,
            "evidence": evidence_payloads,
            "relations": list(admitted_relations.values()),
            "gaps": relevant_gaps,
            "missing_parts": [group["part"] for group in group_records if group["missing"]],
            "context_spans": context_spans,
            "context_chars": sum(len(span["text"]) for span in context_spans),
            "evidence_chars": sum(len(item["text"]) for item in evidence_payloads),
            "max_context_chars": max_context_chars,
            "max_context_chunks": max_context_chunks,
            "truncated": bool(omitted_ids),
            "omitted_chunk_ids": sorted(omitted_ids),
        }

    async def _ensure_document_vectors(self) -> dict[str, list[float]] | None:
        if self.client is None or not self.client.settings.embedding_model:
            self.embedding_status = "not_configured"
            return None
        if self._document_vectors is not None:
            self.embedding_status = "ready"
            return self._document_vectors
        async with self._embedding_lock:
            if self._document_vectors is not None:
                self.embedding_status = "ready"
                return self._document_vectors
            inputs = self._embedding_inputs()
            cached_vectors = self._load_cached_vectors(inputs) or {}
            if len(cached_vectors) == len(inputs):
                self._document_vectors = cached_vectors
                self._embedding_failures = 0
                self._embedding_retry_after = 0.0
                self.embedding_status = "ready"
                return cached_vectors
            vectors = dict(cached_vectors)
            dimension = len(next(iter(vectors.values()))) if vectors else None
            missing_chunks = [
                chunk for chunk in self.index.chunks
                if chunk["chunk_id"] not in vectors
            ]
            for start in range(0, len(missing_chunks), 32):
                batch_chunks = missing_chunks[start:start + 32]
                batch = [inputs[chunk["chunk_id"]] for chunk in batch_chunks]
                batch_vectors = await self.client.embeddings(batch)
                if len(batch_vectors) != len(batch):
                    raise ModelProtocolError("/embeddings: количество векторов не совпало с текстами")
                for chunk, vector in zip(batch_chunks, batch_vectors):
                    if dimension is None:
                        dimension = len(vector)
                    elif len(vector) != dimension:
                        raise ModelProtocolError("/embeddings: размерность векторов корпуса различается")
                    vectors[chunk["chunk_id"]] = vector
                self._save_cached_vectors(vectors, inputs)
            self._document_vectors = vectors
            self._embedding_failures = 0
            self._embedding_retry_after = 0.0
            self.embedding_status = "ready"
            return vectors

    async def _dense_scores(self, query: str) -> dict[str, float] | None:
        vectors = await self._ensure_document_vectors()
        if vectors is None:
            return None
        query_vectors = await self.client.embeddings([query])
        if len(query_vectors) != 1:
            raise ModelProtocolError("/embeddings: ожидался один вектор запроса")
        query_vector = query_vectors[0]
        if any(len(vector) != len(query_vector) for vector in vectors.values()):
            raise ModelProtocolError("/embeddings: размерность запроса и корпуса не совпадает")
        return {
            chunk_id: math.fsum(left * right for left, right in zip(vector, query_vector))
            for chunk_id, vector in vectors.items()
        }

    async def search(self, query: str, filters: Any) -> Any:
        configured_embeddings = (
            self.client is not None and bool(self.client.settings.embedding_model)
        )
        in_cooldown = configured_embeddings and time.monotonic() < self._embedding_retry_after
        mode = "hybrid" if configured_embeddings and not in_cooldown else "bm25"
        if in_cooldown:
            self.embedding_status = "cooldown"
        try:
            return await self.search_with_mode(
                query,
                filters,
                mode,
                apply_reranker=(
                    mode == "hybrid"
                    and bool(self.client and self.client.settings.reranker_model)
                ),
            )
        except ModelAPIError as exc:
            if not configured_embeddings:
                raise
            self._embedding_failures = min(self._embedding_failures + 1, 5)
            retry_delay = min(
                EMBEDDING_RETRY_BASE_SECONDS * (2 ** (self._embedding_failures - 1)),
                EMBEDDING_RETRY_MAX_SECONDS,
            )
            self._embedding_retry_after = time.monotonic() + retry_delay
            self.embedding_status = "disabled_after_error"
            diagnostic = (
                f"embeddings недоступны; повтор через {retry_delay:g} с; "
                f"текущий запрос продолжен через BM25: {exc}"
            )
            self.diagnostics.append(diagnostic)
            logger.warning("retrieval: %s", diagnostic)
            return await self.search_with_mode(
                query,
                filters,
                "bm25",
                apply_reranker=False,
            )


async def benchmark_modes(
    index: SearchIndex,
    cases: list[dict[str, Any]],
    client: Any = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Compare BM25, dense, and hybrid recall against manually labeled chunk IDs."""
    if not cases:
        raise ValueError("Для сравнения режимов нужен хотя бы один размеченный запрос.")
    if top_k < 1:
        raise ValueError("top_k должен быть не меньше 1.")
    backend = SchemaSearchBackend(index, client=client, top_k=top_k)
    modes = ("bm25", "semantic", "hybrid")
    mode_results: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    unavailable: dict[str, str] = {}
    for mode in modes:
        for case in cases:
            expected = set(case["relevant_chunk_ids"])
            if not expected:
                raise ValueError("Каждый benchmark case должен иметь relevant_chunk_ids.")
            unknown_ids = expected.difference(index._by_chunk_id)
            if unknown_ids:
                raise ValueError(
                    "Разметка содержит неизвестные chunk_id: "
                    + ", ".join(sorted(unknown_ids))
                )
            if mode != "bm25" and mode in unavailable:
                break
            try:
                result = await backend.search_with_mode(
                    case["query"],
                    case.get("filters", {}),
                    mode,
                )
            except ModelAPIError as exc:
                unavailable[mode] = str(exc)
                if mode == "semantic":
                    unavailable["hybrid"] = str(exc)
                break
            retrieved = [evidence.chunk_id for evidence in result.evidence]
            mode_results[mode].append({
                "query": case["query"],
                "relevant": len(expected),
                "found": len(expected.intersection(retrieved)),
                "recall_at_k": round(len(expected.intersection(retrieved)) / len(expected), 4),
                "retrieved_chunk_ids": retrieved,
            })
    summary: dict[str, Any] = {
        "top_k": top_k,
        "case_count": len(cases),
        "modes": {},
        "diagnostics": backend.diagnostics,
    }
    for mode, results in mode_results.items():
        if mode in unavailable:
            summary["modes"][mode] = {"status": "unavailable", "reason": unavailable[mode]}
        elif mode != "bm25" and not results:
            summary["modes"][mode] = {
                "status": "unavailable",
                "reason": "Модель embeddings не настроена.",
            }
        else:
            summary["modes"][mode] = {
                "status": "ready",
                "mean_recall_at_k": round(
                    sum(item["recall_at_k"] for item in results) / len(results), 4
                ) if results else 0.0,
                "queries": results,
            }
    return summary

def create_search(data_dir: str | Path, client: Any = None) -> SchemaSearchBackend:
    """Factory referenced by SEARCH_FACTORY=retrieval:create_search."""
    data_path = Path(data_dir)
    if data_path.is_file():
        kb_path = data_path
    else:
        candidates = ("knowledge-base.zip", "kb.zip", "knowledge_base.zip")
        kb_path = next(
            (data_path / name for name in candidates if (data_path / name).is_file()),
            data_path,
        )
    index = build_index(kb_path)
    configured_top_k = getattr(getattr(client, "settings", None), "top_k", 10)
    return SchemaSearchBackend(index, client=client, top_k=configured_top_k)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Построить и проверить полнотекстовый индекс базы знаний.")
    parser.add_argument("kb_path", type=Path, help="ZIP-архив базы знаний или распакованный каталог")
    parser.add_argument("--query", help="Выполнить один поисковый запрос")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--stats", action="store_true", help="Вывести замеры сборки и размер сериализуемого индекса")
    parser.add_argument(
        "--compare-csv",
        type=Path,
        help="Сравнить режимы по CSV с колонками query,relevant_chunk_ids,filters_json",
    )
    args = parser.parse_args()

    if args.stats:
        tracemalloc.start()
    start = time.perf_counter()
    index = build_index(args.kb_path)
    build_ms = (time.perf_counter() - start) * 1000
    stats = index.stats()
    if args.stats:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stats.update({"build_ms": round(build_ms, 3), "peak_python_bytes": peak})
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.query:
        search_start = time.perf_counter()
        evidence = index.search(args.query, top_k=args.top_k)
        search_ms = (time.perf_counter() - search_start) * 1000
        print(json.dumps({
            "query": args.query,
            "search_ms": round(search_ms, 3),
            "evidence": evidence,
        }, ensure_ascii=False, indent=2))
    if args.compare_csv:
        with args.compare_csv.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"query", "relevant_chunk_ids"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("--compare-csv требует query,relevant_chunk_ids.")
            cases = []
            for row in reader:
                try:
                    filters = json.loads(row.get("filters_json") or "{}")
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Некорректный filters_json в строке {reader.line_num}.") from exc
                cases.append({
                    "query": row["query"],
                    "relevant_chunk_ids": [
                        chunk_id.strip()
                        for chunk_id in row["relevant_chunk_ids"].split(";")
                        if chunk_id.strip()
                    ],
                    "filters": filters,
                })
        client = None
        if any(os.environ.get(name) for name in ("LLM_API_KEY", "OPENAI_API_KEY")):
            client = LLMClient(Settings.from_env())
        async def run_comparison() -> dict[str, Any]:
            try:
                return await benchmark_modes(
                    index,
                    cases,
                    client=client,
                    top_k=args.top_k,
                )
            finally:
                if client is not None:
                    await client.aclose()

        comparison = asyncio.run(run_comparison())
        print(json.dumps({"comparison": comparison}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
