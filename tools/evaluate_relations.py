"""Сравнить граф связей с ручным эталоном на ограниченных размеченных областях."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval import build_index


def _endpoint(index: Any, chunk_id: str) -> tuple[str, str]:
    chunk = index.get_chunk(chunk_id)
    document = index.get_document(chunk["document_id"])
    return document["source_file"], str(chunk.get("section_reference") or "")


def _key(edge: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(edge["type"]),
        str(edge["from_source_file"]),
        str(edge.get("from_section") or ""),
        str(edge["to_source_file"]),
        str(edge.get("to_section") or ""),
        str(edge.get("target_reference") or ""),
    )


def evaluate(index: Any, gold: dict[str, Any]) -> dict[str, Any]:
    rows = []
    total_tp = total_fp = total_fn = 0
    for scope in gold["scopes"]:
        expected = {_key(edge) for edge in scope["expected"]}
        predicted = set()
        semantic_statuses: dict[str, int] = {}
        for relation in index.provision_relations:
            support = relation.get("supporting_ranges", [])
            matched_scope = any(
                item.get("source_file") == scope["support_source_file"]
                and scope["support_text_contains"] in str(item.get("text") or "")
                for item in support
            )
            if not matched_scope:
                continue
            semantic_status = relation.get("semantic_validation", {}).get(
                "status", "not_checked"
            )
            semantic_statuses[semantic_status] = semantic_statuses.get(semantic_status, 0) + 1
            from_file, from_section = _endpoint(index, relation["from_chunk_id"])
            to_file, to_section = _endpoint(index, relation["to_chunk_id"])
            predicted.add(_key({
                "type": relation["type"],
                "from_source_file": from_file,
                "from_section": from_section,
                "to_source_file": to_file,
                "to_section": to_section,
                "target_reference": relation.get("target_reference"),
            }))
        tp = expected & predicted
        fp = predicted - expected
        fn = expected - predicted
        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)
        rows.append({
            "name": scope["name"],
            "expected": len(expected),
            "predicted": len(predicted),
            "true_positive": len(tp),
            "false_positive": len(fp),
            "false_negative": len(fn),
            "precision": round(len(tp) / len(predicted), 4) if predicted else 0.0,
            "recall": round(len(tp) / len(expected), 4) if expected else 0.0,
            "semantic_type_checks": semantic_statuses,
            "false_positive_edges": [list(edge) for edge in sorted(fp)],
            "missing_edges": [list(edge) for edge in sorted(fn)],
        })
    denominator = 2 * total_tp + total_fp + total_fn
    return {
        "evaluation": "curated_manual_gold_sample",
        "scope_count": len(rows),
        "micro": {
            "true_positive": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
            "precision": round(total_tp / (total_tp + total_fp), 4)
            if total_tp + total_fp else 0.0,
            "recall": round(total_tp / (total_tp + total_fn), 4)
            if total_tp + total_fn else 0.0,
            "f1": round(2 * total_tp / denominator, 4) if denominator else 0.0,
        },
        "scopes": rows,
        "limitations": [
            "Метрики относятся только к явно размеченным областям; это не оценка всего графа.",
            "Эталон следует проверять и расширять вручную до использования как независимого качества.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kb_path", type=Path)
    parser.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "evaluation" / "relation_gold.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    index = build_index(args.kb_path)
    report = evaluate(index, gold)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
