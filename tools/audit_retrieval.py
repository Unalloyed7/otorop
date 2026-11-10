"""Проверить наличие нужных положений и отсутствие неподходящих редакций без API."""
from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval import create_search
from llm_client import compact_evidence
from schemas import SearchFilters
from temporal import temporal_issue


def matches(chunk, target):
    return (target["source"] in chunk["source_file"]
            and (not target.get("section") or chunk["section_reference"] == target["section"])
            and (not target.get("contains") or target["contains"] in chunk["text"]))


async def audit(data_dir, annotations, *, top_k=12):
    backend = create_search(data_dir)
    backend.top_k = top_k
    reference = datetime.now(timezone(timedelta(hours=3))).date()
    question_path = Path(data_dir) / "demo_questions.csv"
    questions = {}
    if question_path.is_file():
        with question_path.open(encoding="utf-8-sig", newline="") as f:
            questions = {r["id"]: r["question"] for r in csv.DictReader(f)}
    rows = []
    for case in annotations["cases"]:
        modes = ["annotated"] + (["raw"] if case.get("question_id") in questions else [])
        for mode in modes:
            filters = SearchFilters.model_validate(case.get("filters", {}) if mode == "annotated" else {})
            result = await backend.search(case["query"] if mode == "annotated" else questions[case["question_id"]], filters)
            eligible = [e for e in result.evidence if temporal_issue(e, filters, mode="auto", reference=reference) is None]
            ids = {e.chunk_id for e in eligible}
            checks = []
            for target in case["required"]:
                gold = {c["chunk_id"] for c in backend.index.chunks if matches(c, target)}
                if not gold:
                    raise ValueError(f"Разметка не соответствует базе: {case['id']}, {target}")
                checks.append({**target, "found": bool(ids & gold)})
            unwanted = [e.chunk_id for e in eligible if any(matches(backend.index.get_chunk(e.chunk_id), t) for t in case.get("forbidden", []))]
            original = {"evidence": [e.model_dump(mode="json") for e in eligible]}
            compact, docs, chunks = compact_evidence(original)
            for passage, evidence in zip(compact["evidence"], original["evidence"]):
                restored = {**compact["documents"][passage["document_id"]], **passage}
                assert docs[restored["document_id"]] == evidence["document_id"]
                assert chunks[restored["chunk_id"]] == evidence["chunk_id"]
                for key in ("text", "section", "document_title", "applicability", "counterparty", "contract_number"):
                    assert restored[key] == evidence[key], key
                assert restored["metadata"] == {k: v for k, v in evidence["metadata"].items() if k not in {"score", "retrieval_reason"}}
            rows.append({"id": case["id"], "mode": mode, "checks": checks, "all_required": all(c["found"] for c in checks), "unwanted_ids": unwanted, "passed": all(c["found"] for c in checks) and not unwanted, "evidence_ids": [e.chunk_id for e in result.evidence], "eligible_ids": [e.chunk_id for e in eligible], "payload_before_chars": len(json.dumps(original, ensure_ascii=False)), "payload_after_chars": len(json.dumps(compact, ensure_ascii=False))})
    return {"reference_date": reference.isoformat(), "evaluation": "offline BM25 source coverage; annotated query is a manual oracle, not model intent; not answer accuracy", "top_k": top_k, "fingerprint": backend.index.fingerprint, "modes": {m: {"cases": sum(r["mode"] == m for r in rows), "all_required": sum(r["mode"] == m and r["all_required"] for r in rows), "passed": sum(r["mode"] == m and r["passed"] for r in rows)} for m in {r["mode"] for r in rows}}, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--cases", type=Path, default=ROOT / "evals/retrieval_cases.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(audit(args.data_dir, json.loads(args.cases.read_text())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["modes"], ensure_ascii=False))
    for row in result["cases"]:
        if not row["passed"]:
            print(row["id"], row["mode"], "missing:", [t for t in row["checks"] if not t["found"]], "unwanted:", len(row["unwanted_ids"]))


if __name__ == "__main__":
    main()
