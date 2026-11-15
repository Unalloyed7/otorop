"""Budgeted evidence assembly and recovery driven by the same coverage matrix."""
from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Callable, Sequence

from knowledge_pipeline import relation_can_cover, relation_is_rejected

CONDITIONS = re.compile(
    r"\b(?:когда|с какого|до какого|при каком|услови\w*|применя\w*|после|"
    r"начиная с|дата|редакци\w*|действу\w*|продл\w*|уведомлен\w*)\b", re.I,
)
EXCEPTIONS = re.compile(
    r"\b(?:исключени\w*|кроме|за исключением|не распространя\w*|не применя\w*|"
    r"льгот\w*|особый порядок)\b", re.I,
)


class _Assembly:
    def __init__(self, backend, filters, groups, chars, count, existing, evaluate, normalize_party):
        self.backend, self.index = backend, backend.index
        self.filters, self.groups = filters, groups
        self.max_chars, self.max_count = chars, count
        self.existing = set(existing)
        self.evaluate, self.normalize_party = evaluate, normalize_party
        self.bindings = defaultdict(set)
        self.edges = defaultdict(list)
        self.optional_edges = defaultdict(list)
        self.rejected = []
        for relation in self.index.provision_relations:
            if relation_is_rejected(relation):
                self.rejected.append(relation)
                continue
            mandatory = relation_can_cover(relation) and relation["type"] not in {
                "references", "references_provision", "supplements_procedure",
            }
            if mandatory:
                for chunk_id in {relation["from_chunk_id"], relation["to_chunk_id"]}:
                    self.edges[chunk_id].append(relation)
            else:
                # Ordinary references and unverified edge types provide optional
                # context. Walking incoming references recursively can pull an
                # entire tax code into a single answer and hide the direct rule.
                self.optional_edges[relation["from_chunk_id"]].append(relation)
        if not self.fits(self.existing):
            raise ValueError("Уже переданный контекст превышает бюджет комплекта.")

    def fits(self, ids):
        return ((self.max_count is None or len(ids) <= self.max_count)
                and sum(len(self.index.get_chunk(cid)["text"]) for cid in ids) <= self.max_chars)

    def closure(self, seed):
        ids, relations, frontier = {seed}, {}, [seed]
        for _ in range(3):
            next_frontier = []
            for cid in frontier:
                chunk = self.index.get_chunk(cid)
                neighbours = set(chunk.get("provision_chunk_ids", []))
                neighbours.update(chunk.get("ancestor_chunk_ids", []))
                neighbours.update(self.bindings[cid])
                for relation in self.edges[cid]:
                    relations[relation["relation_id"]] = relation
                    neighbours.update((relation["from_chunk_id"], relation["to_chunk_id"],
                                       *relation["supporting_chunk_ids"]))
                next_frontier.extend(neighbours - ids)
                ids.update(neighbours)
            frontier = next_frontier
            if not frontier:
                break
        # Complete structural units even at the final graph hop.
        frontier = list(ids)
        while frontier:
            chunk = self.index.get_chunk(frontier.pop())
            extra = set(chunk.get("provision_chunk_ids", []))
            extra.update(chunk.get("ancestor_chunk_ids", []))
            extra.update(self.bindings[chunk["chunk_id"]])
            frontier.extend(extra - ids)
            ids.update(extra)
        return ids, relations

    def resolve(self, group, gaps):
        for gap in gaps:
            if gap.get("kind") == "referenced_provision_ambiguous":
                candidates = [self.index.get_chunk(cid) for cid in gap.get("candidate_chunk_ids", [])
                              if cid in self.index._by_chunk_id]
            elif gap.get("kind") == "referenced_document_ambiguous":
                candidates = []
                for doc_id in gap.get("candidate_document_ids", []):
                    chunks = [c for c in self.index.chunks if c["document_id"] == doc_id]
                    if chunks:
                        candidates.append(min(chunks, key=lambda c: c["start_char"]))
            else:
                continue
            used = []
            for name in ("contract_number", "document_id", "counterparty"):
                wanted = self.filters.get(name)
                if not wanted:
                    continue
                used.append(name)
                if name == "counterparty":
                    candidates = [c for c in candidates if self.normalize_party(str(wanted)) in {
                        self.normalize_party(p) for p in c.get("counterparties", [])
                        or [c.get("counterparty") or ""]}]
                else:
                    candidates = [c for c in candidates if c.get(name) == wanted]
            unique = {(c["document_id"], c.get("provision_id", c["chunk_id"])): c for c in candidates}
            if not used or len(unique) != 1:
                continue
            chosen = next(iter(unique.values()))
            source_id, target_id = gap["source_chunk_id"], chosen["chunk_id"]
            if any(r["source_chunk_id"] == source_id and r["target_reference"] == gap.get("target_reference")
                   for r in group["gap_resolutions"]):
                continue
            self.bindings[source_id].add(target_id)
            group["gap_resolutions"].append({
                "gap_id": gap.get("candidate_id") or source_id,
                "source_chunk_id": source_id, "target_reference": gap.get("target_reference"),
                "kind": gap["kind"], "status": "resolved_by_explicit_filter",
                "chunk_id": target_id, "filter": ",".join(used),
            })

    def snapshot(self):
        admitted, closures = set(self.existing), {}
        ordered = []
        for gi, group in enumerate(self.groups):
            for rank, hit in enumerate(group["hits"]):
                ordered.append((hit["round"], rank, gi, hit["chunk_id"]))
        # Scores from different BM25/dense/reranker requests are not comparable.
        # Admit all original candidates before any recovery candidates, sharing
        # the budget across question parts. Never evict an admitted dependency.
        for _, _, _, cid in sorted(ordered):
            if cid not in closures:
                closures[cid] = self.closure(cid)
            candidate = admitted | closures[cid][0]
            if self.fits(candidate):
                admitted = candidate
        # Add one hop of optional references only after primary/recovery units.
        # They never evict those units and are not claimed to be mandatory proof.
        optional_by_group = []
        for group in self.groups:
            group_ids, group_edges = set(), {}
            for hit in group["hits"]:
                base_ids = closures[hit["chunk_id"]][0]
                if not base_ids <= admitted:
                    continue
                for cid in sorted(base_ids):
                    for edge in self.optional_edges[cid]:
                        target_ids, target_edges = self.closure(edge["to_chunk_id"])
                        for support_id in edge["supporting_chunk_ids"]:
                            support_ids, support_edges = self.closure(support_id)
                            target_ids.update(support_ids)
                            target_edges.update(support_edges)
                        proposed = admitted | target_ids
                        if self.fits(proposed):
                            admitted = proposed
                            group_ids.update(target_ids)
                            group_edges.update(target_edges)
                            group_edges[edge["relation_id"]] = edge
            optional_by_group.append((group_ids, group_edges))
        records, relations, omitted = [], {}, set()
        for gi, group in enumerate(self.groups):
            extra_ids, extra_edges = optional_by_group[gi]
            ids, edges, missing = set(extra_ids), dict(extra_edges), set()
            for hit in group["hits"]:
                closure_ids, closure_edges = closures[hit["chunk_id"]]
                if closure_ids <= admitted:
                    ids.update(closure_ids)
                    edges.update(closure_edges)
                else:
                    missing.update(closure_ids - admitted)
            relations.update(edges)
            omitted.update(missing)
            resolutions = [r for r in group["gap_resolutions"]
                           if {r["source_chunk_id"], r["chunk_id"]} <= ids]
            gaps = [g for g in self.index.provision_gaps if g.get("source_chunk_id") in ids
                    and not any(r["source_chunk_id"] == g.get("source_chunk_id")
                                and r["target_reference"] == g.get("target_reference") for r in resolutions)]
            basis = [h["chunk_id"] for h in group["hits"] if h["chunk_id"] in ids]
            basis = list(dict.fromkeys(basis + [r["chunk_id"] for r in resolutions]))
            condition_ids, exception_ids = set(), set()
            for cid in ids:
                chunk = self.index.get_chunk(cid)
                kinds = {c.get("kind") for c in chunk.get("local_conditions", [])}
                if (chunk.get("chunk_role") == "applicability_condition"
                        or chunk.get("applicability_conditions") or "condition" in kinds):
                    condition_ids.add(cid)
                if "exception" in kinds:
                    exception_ids.add(cid)
            for edge in edges.values():
                if not relation_can_cover(edge):
                    continue
                edge_ids = {edge["from_chunk_id"], edge["to_chunk_id"], *edge["supporting_chunk_ids"]} & ids
                if edge["type"] in {"defines_applicability", "condition_for", "applies_to"}:
                    condition_ids.update(edge_ids)
                if edge["type"] in {"establishes_exception", "exception_to"}:
                    exception_ids.update(edge_ids)
            def requirement(pattern, found):
                required = bool(pattern.search(group["part"]))
                return {"required": required, "status": "found" if found else
                        "missing_in_retrieved_bundle" if required else "not_explicitly_requested",
                        "chunk_ids": sorted(found)}
            conditions = requirement(CONDITIONS, condition_ids)
            exceptions = requirement(EXCEPTIONS, exception_ids)
            complete = (bool(basis) and not gaps and not missing
                        and conditions["status"] != "missing_in_retrieved_bundle"
                        and exceptions["status"] != "missing_in_retrieved_bundle")
            coverage = {
                "status": "covered" if complete else "partial" if basis else "uncovered",
                "basis": {"required": True, "status": "found" if basis else "missing", "chunk_ids": basis},
                "conditions": conditions, "exceptions": exceptions,
                "gaps": {"status": "present" if gaps else "none_detected_in_indexed_graph",
                         "items": gaps, "gap_resolutions": resolutions},
            }
            records.append({**{k: v for k, v in group.items() if k != "hits"},
                            "evidence_chunk_ids": sorted(ids), "relation_ids": sorted(edges),
                            "gap_resolutions": resolutions, "truncated": bool(missing),
                            "omitted_chunk_ids": sorted(missing), "missing": not complete, "coverage": coverage})
        ranked = list(dict.fromkeys(cid for _, _, _, cid in sorted(ordered) if cid in admitted))
        ranked.extend(sorted(admitted - set(ranked), key=lambda cid: (
            self.index.get_chunk(cid)["source_file"], self.index.get_chunk(cid)["start_char"])))
        evidence, scores = [], {h["chunk_id"]: h["score"] for g in self.groups for h in reversed(g["hits"])}
        for cid in ranked:
            item = dict(self.index.get_chunk(cid))
            item["applicability"] = self.evaluate(item)
            if cid in scores:
                item["score"] = scores[cid]
            evidence.append(item)
        ranges = defaultdict(list)
        for cid in admitted:
            c = self.index.get_chunk(cid)
            ranges[c["document_id"]].append(c)
        spans = []
        for doc_id in sorted(ranges):
            doc = self.index.get_document(doc_id)
            merged = []
            for chunk in sorted(ranges[doc_id], key=lambda c: (c["start_char"], c["end_char"])):
                if merged and chunk["start_char"] <= merged[-1]["end_char"]:
                    merged[-1]["end_char"] = max(merged[-1]["end_char"], chunk["end_char"])
                    merged[-1]["source_chunk_ids"].append(chunk["chunk_id"])
                else:
                    merged.append({"document_id": doc_id, "source_file": doc["source_file"],
                                   "start_char": chunk["start_char"], "end_char": chunk["end_char"],
                                   "source_chunk_ids": [chunk["chunk_id"]]})
            for span in merged:
                span["text"] = doc["text"][span["start_char"]:span["end_char"]]
                spans.append(span)
        return {"groups": records, "evidence": evidence, "relations": list(relations.values()),
                "rejected_relations": [r for r in self.rejected if {r["from_chunk_id"], r["to_chunk_id"]} & admitted],
                "gaps": [g for g in self.index.provision_gaps if g.get("source_chunk_id") in admitted | omitted],
                "missing_parts": [g["part"] for g in records if g["missing"]], "context_spans": spans,
                "context_chars": sum(len(s["text"]) for s in spans),
                "evidence_chars": sum(len(e["text"]) for e in evidence),
                "max_context_chars": self.max_chars, "max_context_chunks": self.max_count,
                "truncated": bool(omitted), "omitted_chunk_ids": sorted(omitted)}


async def build_bundle(backend: Any, parts: Sequence[str], filters: Any, *, top_k: int | None,
                       max_context_chars: int, max_context_chunks: int | None, query_context: str | None,
                       existing_chunk_ids: Sequence[str], mode: str | None, recovery_rounds: int,
                       recovery_query_budget: int, evaluate: Callable, normalize_date: Callable,
                       normalize_party: Callable) -> dict[str, Any]:
    parts = list(dict.fromkeys(p.strip() for p in parts if isinstance(p, str) and p.strip()))
    limit = backend.top_k if top_k is None else top_k
    if not parts or limit < 1 or max_context_chars < 1 or (max_context_chunks is not None and max_context_chunks < 1):
        raise ValueError("Нужны части вопроса и положительные лимиты контекста/выдачи.")
    if recovery_rounds < 0 or recovery_query_budget < 0:
        raise ValueError("Бюджет повторного поиска не должен быть отрицательным.")
    values = filters.model_dump(mode="json", exclude_none=True) if hasattr(filters, "model_dump") else dict(filters or {})
    specification = values.get("specification_signed_at")
    if values.get("date_kind") in {"specification_signed", "specification_signed_at"}:
        specification = values.get("event_date") or specification
    dates = {"contract_signed_at": values.get("signed_at"), **{
        k: values.get("as_of") or values.get("event_date") for k in ("event", "delivery", "notice")}}
    async def search(query, document_id=None):
        scoped_filters = {**values, "document_id": document_id} if document_id else filters
        return (await backend.search(query, scoped_filters) if mode is None else
                await backend.search_with_mode(query, scoped_filters, mode, top_k=limit))
    groups = []
    for part in parts:
        query = part if not query_context or part == query_context else part + "\n" + query_context
        result = await search(query)
        hits = [{"chunk_id": e.chunk_id, "score": e.metadata.get("score", 0), "round": 0}
                for e in result.evidence[:limit]]
        groups.append({"part": part, "query": query, "hits": hits,
                       "direct_chunk_ids": [h["chunk_id"] for h in hits], "recovered_chunk_ids": [],
                       "gap_resolutions": [], "clarification": result.clarification, "candidates": list(result.candidates)})
    assembly = _Assembly(backend, values, groups, max_context_chars, max_context_chunks, existing_chunk_ids,
                         lambda item: evaluate(item, normalize_date(specification), dates), normalize_party)
    queries, seen = [], set()
    stop, rounds = "not_needed", 0
    def inspect():
        state = assembly.snapshot()
        for group, record in zip(groups, state["groups"]):
            assembly.resolve(group, record["coverage"]["gaps"]["items"])
        return assembly.snapshot()
    for round_number in range(recovery_rounds + 1):
        state = inspect()
        if all(not g["missing"] for g in state["groups"]):
            stop = "coverage_satisfied"
            break
        if len(queries) >= recovery_query_budget:
            stop = "query_budget_exhausted"
            break
        if round_number == recovery_rounds:
            stop = "round_limit_reached"
            break
        work = []
        for gi, record in enumerate(state["groups"]):
            coverage = record["coverage"]
            if (coverage["basis"]["status"] == "missing"
                    and record["direct_chunk_ids"] and record["truncated"]):
                # A known legal unit does not fit. Do not replace it with a
                # small unrelated heading and misreport a successful recovery.
                continue
            needs = []
            if coverage["basis"]["status"] == "missing":
                needs.append(("basis", "подтверждающее положение порядок"))
            for kind, words in (("conditions", "условия применения дата вступления в силу"),
                                ("exceptions", "исключения особый порядок не применяется")):
                if coverage[kind]["status"] == "missing_in_retrieved_bundle":
                    needs.append((kind, words))
            for gap in coverage["gaps"]["items"]:
                if str(gap.get("kind", "")).startswith(("referenced_", "unresolved_")):
                    needs.append(("unresolved_" + gap["kind"], " ".join(str(v) for v in (
                        "пункт", gap.get("target_reference"), gap.get("target_document")) if v)))
            for reason, words in needs:
                query = " ".join(filter(None, (record["part"], query_context, words)))[:2400]
                scopes = [None]
                if reason in {"conditions", "exceptions"} and record["direct_chunk_ids"] and not values.get("document_id"):
                    scopes.append(backend.index.get_chunk(record["direct_chunk_ids"][0])["document_id"])
                for scope in scopes:
                    key = (gi, query, scope)
                    if key not in seen:
                        seen.add(key)
                        work.append((gi, query, reason, scope))
        if not work:
            stop = "context_budget_exhausted" if state["truncated"] else "unresolved_coverage"
            break
        rounds = round_number + 1
        added = 0
        for gi, query, reason, scope in work:
            if len(queries) >= recovery_query_budget:
                break
            current = inspect()["groups"][gi]["coverage"]
            if reason in {"conditions", "exceptions"} and current[reason]["status"] != "missing_in_retrieved_bundle":
                continue
            if reason == "basis" and current["basis"]["status"] == "found":
                continue
            result = await search(query, scope)
            group = groups[gi]
            known = {h["chunk_id"] for h in group["hits"]}
            new = []
            for e in result.evidence[:limit]:
                if e.chunk_id not in known:
                    group["hits"].append({"chunk_id": e.chunk_id, "score": e.metadata.get("score", 0), "round": rounds})
                    group["recovered_chunk_ids"].append(e.chunk_id)
                    known.add(e.chunk_id)
                    new.append(e.chunk_id)
            added += len(new)
            if result.clarification:
                group["clarification"] = result.clarification
            group["candidates"] = list(dict.fromkeys(group["candidates"] + list(result.candidates)))
            queries.append({"part": group["part"], "round": rounds, "reason": reason, "query": query,
                            "document_id": scope,
                            "new_chunk_count": len(new), "result_chunk_ids": [e.chunk_id for e in result.evidence[:limit]]})
        if not added:
            state = inspect()
            stop = "coverage_satisfied" if all(not g["missing"] for g in state["groups"]) else "no_new_evidence"
            break
    state = inspect()
    state["recovery"] = {"rounds": rounds, "queries_used": len(queries), "query_budget": recovery_query_budget,
                         "round_limit": recovery_rounds, "stop_reason": stop, "queries": queries}
    state["pipeline"] = backend.index.pipeline_diagnostics.copy()
    return state
