"""Offline BM25 retrieval for the organizer-provided knowledge base."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from documents import PROCESSOR_VERSION, load_documents, split_document

RETRIEVAL_VERSION = "retrieval-v1"
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
        "ые", "ие", "ий", "ый", "ой", "ую", "юю", "ою", "ею", "ам", "ям",
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
}
_QUERY_STOPWORDS = {
    "а", "и", "или", "по", "с", "в", "на", "к", "от", "до", "для", "что",
    "как", "где", "когда", "какой", "какая", "какие", "нас", "наш", "наши",
    "это", "ли", "бы", "же", "договор",
}


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().replace("ё", "е").replace("–", "-").replace("—", "-")).strip()


def _stem_token(token: str) -> str:
    if not re.fullmatch(r"[а-яё]+", token):
        return token
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


def _evaluate_condition(condition: dict[str, str], specification_signed_at: str | None) -> str:
    if not specification_signed_at:
        return "unknown"
    date = condition["date"]
    operator = condition["operator"]
    if operator == "after":
        return "applicable" if specification_signed_at > date else "not_applicable"
    if operator == "before":
        return "applicable" if specification_signed_at < date else "not_applicable"
    if operator == "on_or_after":
        return "applicable" if specification_signed_at >= date else "not_applicable"
    return "unknown"


def evaluate_applicability(
    evidence: dict[str, Any],
    specification_signed_at: str | None = None,
) -> dict[str, Any]:
    """Evaluate explicit conditions without inferring missing dates or terms."""
    conditions = evidence.get("applicability_conditions") or []
    if not conditions:
        return {"status": "not_specified", "conditions": []}
    date = _date(specification_signed_at)
    results = [
        {**condition, "status": _evaluate_condition(condition, date)}
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
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Поисковый запрос не должен быть пустым.")
        if top_k < 1:
            raise ValueError("top_k должен быть не меньше 1.")
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
        normalized_query = _normalize(query)
        query_references = {match.group(1).rstrip(".") for match in _REFERENCE_RE.finditer(query)}
        as_of = _date(filters.get("as_of") or filters.get("valid_on") or filters.get("event_date"))
        spec_date = _date(filters.get("specification_signed_at"))
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
                if not document_contract and document["document_id"] not in related_context_ids:
                    continue
            if wanted_party:
                parties = {_party_key(name) for name in document.get("counterparties", [])}
                if not parties or wanted_party not in parties:
                    continue
            if wanted_parties:
                parties = {_party_key(name) for name in document.get("counterparties", [])}
                if not parties.intersection(wanted_parties):
                    continue
            if wanted_type and document.get("document_type") != wanted_type:
                continue
            if as_of and (
                (document.get("valid_from") and as_of < document["valid_from"])
                or (document.get("valid_to") and as_of >= document["valid_to"])
            ):
                continue
            applicability = evaluate_applicability(chunk, spec_date)
            if only_applicable and applicability["status"] == "not_applicable":
                continue

            score = self._bm25(bm25_query_tokens, position)
            if query_aliases:
                score += self._bm25(query_aliases, position) * 3.0
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
            if score > 0:
                scored.append((score, position))

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
                for document_id, document in self.documents.items():
                    if document.get("contract_number") != requested_contract:
                        continue
                    for reference in target_references:
                        matches = [
                            (score, position)
                            for score, position in scored
                            if self.chunks[position]["document_id"] == document_id
                            and self.chunks[position]["section_reference"] == reference
                        ]
                        if matches:
                            best_match = matches[0]
                            if best_match[1] not in pinned_positions:
                                matching_reference_chunks.append(best_match)
                                pinned_positions.add(best_match[1])
            pinned.extend(matching_reference_chunks)
            scored = pinned + [
                (score, position) for score, position in scored
                if position not in pinned_positions
            ]
        results: list[dict[str, Any]] = []
        for score, position in scored[:top_k]:
            evidence = dict(self.chunks[position])
            evidence["score"] = round(score, 6)
            evidence["applicability"] = evaluate_applicability(evidence, spec_date)
            evidence["retrieval_reason"] = "bm25"
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
                evidence["applicability"] = evaluate_applicability(evidence, spec_date)
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

    def get_document(self, document_id: str) -> dict[str, Any]:
        return self.index.get_document(document_id)

    def get_chunk(self, chunk_id: str) -> dict[str, Any]:
        return self.index.get_chunk(chunk_id)

    def get_neighbors(self, chunk_id: str, window: int = 1) -> list[dict[str, Any]]:
        return self.index.get_neighbors(chunk_id, window=window)

    async def search(self, query: str, filters: Any) -> Any:
        from schemas import Applicability, Coordinates, Evidence, SearchResult

        filter_values = filters.model_dump(mode="json", exclude_none=True) if filters else {}
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
            index_filters["as_of"] = event_date
            if date_kind in {"specification_signed", "specification_signed_at"}:
                index_filters["specification_signed_at"] = event_date

        evidence_items = self.index.search(query, index_filters, top_k=self.top_k)
        if (
            self.client is not None
            and self.client.settings.reranker_model
            and evidence_items
        ):
            ranked_indices = await self.client.rerank(
                query,
                [item["title"] + "\n" + item["text"] for item in evidence_items],
                top_n=self.client.settings.reranker_top_n,
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

        converted = []
        for item in evidence_items:
            condition_records = item.get("applicability_conditions", [])
            conditions = [
                str(condition.get("source_text") or condition)
                for condition in condition_records
            ]
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
                    "related_document_ids",
                    "linked_document_ids",
                    "applicability_conditions",
                    "uncertainties",
                    "score",
                    "retrieval_reason",
                )
                if key in item
            }
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
                    edition=item.get("edition") or "Редакция в источнике не указана",
                    valid_from=item.get("valid_from"),
                    valid_to=item.get("valid_to"),
                    date_kind=next(iter(source_date_kinds)) if len(source_date_kinds) == 1 else None,
                    conditions=conditions,
                    resolved=applicability_result.get("status") in {"applicable", "not_applicable"},
                ),
                contract_number=item.get("contract_number"),
                counterparty=item.get("counterparty"),
                metadata=metadata,
            ))
        return SearchResult(evidence=converted, candidates=candidates)


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
    return 0


if __name__ == "__main__":
    sys.exit(_main())
