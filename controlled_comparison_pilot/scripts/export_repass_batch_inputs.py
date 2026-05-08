#!/usr/bin/env python3
"""Export batch GPT answers to RePASs input JSON files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PID_TAG_RE = re.compile(r"\s*\[PID:[^\]]+\]\s*")
CONDITIONS = ("general", "task_aware")
DATASETS = ("obliqa_bm25_gpt52", "obliqa_mp_bm25_gpt52", "xref_adgm_bm25_gpt52")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--expected_count", type=int, default=None)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def clean_answer(answer: str) -> str:
    cleaned = PID_TAG_RE.sub(" ", answer)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    return cleaned.strip()


def passage_text(block: Any) -> str:
    text = str(block or "").strip()
    marker = "\nText: "
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text[len("Text: ") :].strip() if text.startswith("Text: ") else text


def passage_texts(value: Any) -> list[str]:
    if isinstance(value, list):
        return [passage_text(item) for item in value]
    return [passage_text(value)] if value else []


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    tagged = str(record.get("generated_answer_with_grounding") or "")
    cleaned = clean_answer(tagged)
    return {
        "QuestionID": record.get("QuestionID"),
        "RetrievedPassages": passage_texts(record.get("RetrievedPassages")),
        "Answer": cleaned,
        "Question": record.get("Question"),
        "generated_answer_with_grounding": tagged,
        "clean_answer": cleaned,
        "used_passage_ids": record.get("used_passage_ids"),
        "RetrievedPassageIDs": record.get("RetrievedPassageIDs"),
        "dataset": record.get("dataset"),
        "prompt_condition": record.get("prompt_condition"),
        "prompt_version": record.get("prompt_version"),
        "retriever": record.get("retriever"),
        "k": record.get("k"),
        "json_parse_ok": record.get("json_parse_ok"),
        "actual_model_or_deployment": record.get("actual_model_or_deployment"),
        "provider": record.get("provider"),
        "api_version": record.get("api_version"),
    }


def validate(records: list[dict[str, Any]], path: Path, expected_count: int | None) -> list[str]:
    errors: list[str] = []
    if expected_count is not None and len(records) != expected_count:
        errors.append(f"{path}: expected {expected_count} records, found {len(records)}")
    for idx, row in enumerate(records, start=1):
        for field in ("QuestionID", "RetrievedPassages", "Answer"):
            if field not in row or row[field] in (None, ""):
                errors.append(f"{path}: record {idx} missing {field}")
        if PID_TAG_RE.search(str(row.get("Answer") or "")):
            errors.append(f"{path}: record {idx} Answer contains [PID:...] tag")
        if "generated_answer_with_grounding" not in row:
            errors.append(f"{path}: record {idx} missing generated_answer_with_grounding")
        if "used_passage_ids" not in row:
            errors.append(f"{path}: record {idx} missing used_passage_ids")
    return errors


def main() -> None:
    args = parse_args()
    all_errors: list[str] = []
    for condition in CONDITIONS:
        for dataset in DATASETS:
            in_path = args.answers_root / condition / f"{dataset}_answers.json"
            out_path = args.output_root / condition / f"{dataset}.json"
            rows = read_json(in_path)
            if not isinstance(rows, list):
                raise ValueError(f"{in_path} must contain a JSON list")
            out_rows = [convert_record(row) for row in rows]
            write_json(out_path, out_rows)
            errors = validate(out_rows, out_path, args.expected_count)
            all_errors.extend(errors)
            print(f"Wrote {out_path}: {len(out_rows)} records", flush=True)
            false_parse = [row.get("QuestionID") for row in out_rows if row.get("json_parse_ok") is False]
            if false_parse:
                print(f"  json_parse_ok false: {false_parse[:10]}", flush=True)
    if all_errors:
        for error in all_errors:
            print(f"Validation error: {error}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
