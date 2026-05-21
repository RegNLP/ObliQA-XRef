#!/usr/bin/env python3
"""Export ObliQA-XRef generated answers to the RePASs input JSON schema.

RePASs expects a JSON list whose records contain:
- QuestionID
- RetrievedPassages
- Answer

Extra metadata is retained for downstream grouping and audit, although the
current RePASs evaluator ignores it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_test_map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["item_id"]): row for row in read_jsonl(path)}


def load_passage_lookup(path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not path.exists():
        return lookup
    for obj in read_jsonl(path):
        pid = str(obj.get("passage_uid") or obj.get("passage_id") or obj.get("pid") or "")
        text = str(obj.get("passage") or obj.get("text") or "")
        if pid and text:
            lookup[pid] = text
    return lookup


def normalize_answers(raw: Any) -> dict[str, dict[str, Any]]:
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    if isinstance(raw, list):
        out: dict[str, dict[str, Any]] = {}
        for row in raw:
            if isinstance(row, dict) and row.get("item_id"):
                out[str(row["item_id"])] = row
        return out
    raise ValueError("Generated-answer JSON must be a dict keyed by item_id or a list of records")


def passage_texts_from_record(
    answer_row: dict[str, Any],
    passage_lookup: dict[str, str],
) -> list[str]:
    retrieved_passages = answer_row.get("retrieved_passages") or []
    texts: list[str] = []
    if isinstance(retrieved_passages, list):
        for passage in retrieved_passages:
            if isinstance(passage, str) and passage:
                texts.append(passage)
            elif isinstance(passage, dict):
                text = str(passage.get("text") or passage.get("passage") or "")
                if text:
                    texts.append(text)
    if texts:
        return texts

    docids = answer_row.get("retrieved_docids") or []
    if isinstance(docids, list):
        for docid in docids:
            text = passage_lookup.get(str(docid))
            if text:
                texts.append(text)
    return texts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--test_jsonl",
        default=(
            "ObliQA-XRef_Out_Datasets/final_large_merged/"
            "ObliQA-XRef-ADGM-ALL/test.jsonl"
        ),
    )
    ap.add_argument("--answers_json", required=True)
    ap.add_argument(
        "--passage_corpus",
        default="runs/adapter_adgm/processed/passage_corpus.jsonl",
    )
    ap.add_argument("--output_json", required=True)
    ap.add_argument(
        "--retrieval_method",
        default=None,
        help="Optional override when answer records do not include retrieval_method",
    )
    ap.add_argument(
        "--oracle_gold_pair",
        action="store_true",
        help="Diagnostic mode: use gold source_text and target_text as RetrievedPassages",
    )
    args = ap.parse_args()

    test_items = load_test_map(Path(args.test_jsonl))
    answers = normalize_answers(load_json(Path(args.answers_json)))
    passage_lookup = load_passage_lookup(Path(args.passage_corpus))

    out_rows: list[dict[str, Any]] = []
    for qid, answer_row in answers.items():
        item = test_items.get(qid, {})
        if args.oracle_gold_pair:
            retrieved = [
                text
                for text in [str(item.get("source_text") or ""), str(item.get("target_text") or "")]
                if text
            ]
        else:
            retrieved = passage_texts_from_record(answer_row, passage_lookup)

        generated_answer = str(
            answer_row.get("generated_answer") or answer_row.get("Answer") or answer_row.get("answer") or ""
        )
        row = {
            "QuestionID": qid,
            "RetrievedPassages": retrieved,
            "Answer": generated_answer,
            "question": item.get("question") or answer_row.get("question") or "",
            "gold_answer": item.get("gold_answer") or answer_row.get("gold_answer") or "",
            "source_text": item.get("source_text") or "",
            "target_text": item.get("target_text") or "",
            "source_passage_id": item.get("source_passage_id") or "",
            "target_passage_id": item.get("target_passage_id") or "",
            "generation_method": (
                item.get("generation_method")
                or item.get("method")
                or answer_row.get("generation_method")
                or answer_row.get("method")
                or ""
            ),
            "sampling_regime": item.get("sampling_regime") or answer_row.get("sampling_regime") or "",
            "retrieval_method": args.retrieval_method or answer_row.get("retrieval_method") or "",
        }
        out_rows.append(row)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=2)


if __name__ == "__main__":
    main()
