from __future__ import annotations

import csv
import json
from pathlib import Path

from obliqaxref.benchmark_metadata import OBLIQA_XREF_METADATA_FIELDS
from obliqaxref.eval.Analysis.benchmark_statistics import (
    build_difficulty_rows,
    build_statistics_rows,
    generate_benchmark_statistics,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_statistics_counts_by_corpus_method_persona_and_difficulty(tmp_path):
    dep = tmp_path / "final_dependency_valid.jsonl"
    _write_jsonl(
        dep,
        [
            {
                "item_id": "q1",
                "corpus": "adgm",
                "method": "DPEL",
                "persona": "professional",
                "ir_difficulty_label": "easy",
                "question": "What must the firm do?",
                "gold_answer": "It must file.",
                "source_text": "The firm must comply with Rule 1.",
                "target_text": "Rule 1 says the firm must file promptly.",
                "answer_validation_passed": True,
            },
            {
                "item_id": "q2",
                "corpus": "adgm",
                "method": "SCHEMA",
                "persona": "basic",
                "ir_difficulty_label": "hard",
                "question": "When must it file?",
                "gold_answer": "Within five days.",
                "source_text": "The firm follows the filing rule.",
                "target_text": "The filing rule says within five days.",
                "answer_validation_passed": False,
            },
        ],
    )

    paths = generate_benchmark_statistics(inputs=[dep], out_dir=tmp_path / "out")

    assert paths["statistics"].exists()
    assert paths["difficulty"].exists()
    assert paths["markdown"].exists()
    assert paths["latex"].exists()

    with paths["statistics"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert any(row["group_type"] == "method" and row["group_value"] == "DPEL" and row["count"] == "1" for row in rows)
    assert any(row["group_type"] == "persona" and row["group_value"] == "basic" and row["count"] == "1" for row in rows)
    assert all(field in rows[0] for field in OBLIQA_XREF_METADATA_FIELDS)

    difficulty = build_difficulty_rows([
        {**record, "cohort": "dependency_valid", "benchmark_family": "ObliQA", "benchmark_name": "ObliQA-XRef"}
        for record in json.loads("[" + dep.read_text(encoding="utf-8").strip().replace("\n", ",") + "]")
    ])
    assert {row["ir_difficulty_label"] for row in difficulty} == {"easy", "hard"}


def test_statistics_length_medians_and_missing_fields_do_not_crash(tmp_path):
    records = [
        {
            "item_id": "q1",
            "cohort": "answer_valid",
            "corpus": "ukfin",
            "method": "DPEL",
            "question": "one two three",
            "gold_answer": "one two",
            "source_text": "one two three four",
            "target_text": "five six",
        },
        {
            "item_id": "q2",
            "cohort": "answer_valid",
            "corpus": "ukfin",
            "method": "DPEL",
            "question": "one",
            "gold_answer": "one two three four",
        },
    ]

    rows = build_statistics_rows(records)
    cm = next(
        row for row in rows
        if row.get("stat_type") == "length" and row.get("group_value") == "ukfin / DPEL"
    )

    assert cm["question_len_median"] == 2.0
    assert cm["answer_len_median"] == 3.0
    assert cm["source_len_median"] == 4.0


def test_statistics_outputs_include_metadata_defaults_for_older_inputs(tmp_path):
    inp = tmp_path / "final_answer_valid.jsonl"
    _write_jsonl(
        inp,
        [{
            "item_id": "q1",
            "corpus": "adgm",
            "method": "DPEL",
            "ir_difficulty_label": "medium",
            "question": "Q",
            "gold_answer": "A",
        }],
    )

    paths = generate_benchmark_statistics(inputs=[inp], out_dir=tmp_path / "stats")

    with paths["statistics"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["benchmark_family"] == "ObliQA"
    assert rows[0]["benchmark_name"] == "ObliQA-XRef"
    assert "source_insufficiency_required" in rows[0]
