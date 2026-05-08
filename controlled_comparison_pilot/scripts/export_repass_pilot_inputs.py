#!/usr/bin/env python3
"""Export GPT-5.2 pilot answers to RePASs input JSON files.

This only prepares RePASs inputs. It does not run RePASs or call any model API.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PID_TAG_RE = re.compile(r"\s*\[PID:[^\]]+\]\s*")

CONDITIONS = ("general", "task_aware")
DATASETS = (
    "obliqa_bm25_gpt52",
    "obliqa_mp_bm25_gpt52",
    "xref_adgm_bm25_gpt52",
)
REQUIRED_OUTPUT_FIELDS = ("QuestionID", "RetrievedPassages", "Answer")
PRESERVED_METADATA_FIELDS = (
    "Question",
    "generated_answer_with_grounding",
    "clean_answer",
    "used_passage_ids",
    "RetrievedPassageIDs",
    "dataset",
    "prompt_condition",
    "prompt_version",
    "retriever",
    "k",
    "json_parse_ok",
    "actual_model_or_deployment",
    "provider",
    "api_version",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="controlled_comparison_pilot directory.",
    )
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


def passage_text_from_formatted_block(block: Any) -> str:
    text = str(block or "").strip()
    marker = "\nText: "
    if marker in text:
        return text.split(marker, 1)[1].strip()
    if text.startswith("Text: "):
        return text[len("Text: ") :].strip()
    return text


def passage_texts(retrieved_passages: Any) -> list[str]:
    if not isinstance(retrieved_passages, list):
        return [passage_text_from_formatted_block(retrieved_passages)] if retrieved_passages else []
    return [passage_text_from_formatted_block(block) for block in retrieved_passages]


def make_repass_record(record: dict[str, Any]) -> dict[str, Any]:
    tagged_answer = str(record.get("generated_answer_with_grounding") or "")
    cleaned = clean_answer(tagged_answer)
    out = {
        "QuestionID": record.get("QuestionID"),
        "RetrievedPassages": passage_texts(record.get("RetrievedPassages")),
        "Answer": cleaned,
    }
    metadata = {
        "Question": record.get("Question"),
        "generated_answer_with_grounding": tagged_answer,
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
    out.update(metadata)
    return out


def input_path(root: Path, condition: str, dataset: str) -> Path:
    return root / "answers" / condition / f"{dataset}_answers.json"


def output_path(root: Path, condition: str, dataset: str) -> Path:
    return root / "repass_inputs" / condition / f"{dataset}.json"


def validate_output_record(record: dict[str, Any], path: Path, idx: int) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_OUTPUT_FIELDS:
        if field not in record or record[field] in (None, ""):
            errors.append(f"{path}: record {idx} missing required field {field}")
    if PID_TAG_RE.search(str(record.get("Answer") or "")):
        errors.append(f"{path}: record {idx} Answer still contains a [PID:...] tag")
    if "generated_answer_with_grounding" not in record:
        errors.append(f"{path}: record {idx} missing generated_answer_with_grounding metadata")
    if "used_passage_ids" not in record:
        errors.append(f"{path}: record {idx} missing used_passage_ids metadata")
    return errors


def export_file(in_path: Path, out_path: Path) -> dict[str, Any]:
    records = read_json(in_path)
    if not isinstance(records, list):
        raise ValueError(f"{in_path} must contain a JSON list")

    output_records = [make_repass_record(record) for record in records]
    write_json(out_path, output_records)

    validation_errors: list[str] = []
    json_parse_false: list[str] = []
    for idx, record in enumerate(output_records, start=1):
        validation_errors.extend(validate_output_record(record, out_path, idx))
        if record.get("json_parse_ok") is False:
            json_parse_false.append(str(record.get("QuestionID") or f"record_{idx}"))

    return {
        "input": str(in_path.relative_to(ROOT)),
        "output": str(out_path.relative_to(ROOT)),
        "records": len(output_records),
        "validation_errors": validation_errors,
        "json_parse_false_question_ids": json_parse_false,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    summaries: list[dict[str, Any]] = []
    errors: list[str] = []

    for condition in CONDITIONS:
        for dataset in DATASETS:
            in_path = input_path(root, condition, dataset)
            out_path = output_path(root, condition, dataset)
            try:
                summaries.append(export_file(in_path, out_path))
            except Exception as exc:
                errors.append(f"{in_path}: {exc}")

    total_records = sum(item["records"] for item in summaries)
    print(f"Exported {total_records} records across {len(summaries)} files.")
    for item in summaries:
        print(f"{item['output']}: {item['records']} records")
        for qid in item["json_parse_false_question_ids"]:
            print(f"  json_parse_ok false: {qid}")
        for error in item["validation_errors"]:
            print(f"  validation error: {error}")
            errors.append(error)

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
