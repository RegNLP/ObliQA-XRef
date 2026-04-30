from __future__ import annotations

import csv
import json
from pathlib import Path

from obliqaxref.eval.Analysis.human_audit import (
    aggregate_human_audit,
    cohen_kappa,
    export_human_audit_sample,
    majority_vote,
    normalize_yes_no,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_human_audit_export_stratifies_and_records_shortfalls(tmp_path):
    input_path = tmp_path / "final_answer_valid.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "item_id": "q1",
                "corpus": "adgm",
                "method": "DPEL",
                "ir_difficulty_label": "hard",
                "question": "Q1",
                "gold_answer": "A1",
                "source_passage_id": "s1",
                "target_passage_id": "t1",
            },
            {
                "item_id": "q2",
                "corpus": "adgm",
                "method": "SCHEMA",
                "ir_difficulty_label": "easy",
                "question": "Q2",
                "gold_answer": "A2",
                "source_passage_id": "s2",
                "target_passage_id": "t2",
            },
        ],
    )

    paths = export_human_audit_sample(inputs=[input_path], out_dir=tmp_path / "audit", n=5, seed=13)

    assert paths["sample"].exists()
    assert paths["report"].exists()
    assert paths["instructions"].exists()
    with paths["sample"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["annotation_id"].startswith("HA-")
    for field in (
        "annotator_id",
        "question_understandable",
        "both_passages_necessary",
        "overall_valid",
        "annotator_comments",
    ):
        assert field in rows[0]
        assert rows[0][field] == ""
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["sampled_count"] == 2
    assert any(stratum["shortfall"] > 0 for stratum in report["strata"])


def test_yes_no_normalization_majority_vote_and_kappa():
    assert normalize_yes_no("yes") is True
    assert normalize_yes_no("Pass") is True
    assert normalize_yes_no("0") is False
    assert normalize_yes_no("invalid") is False
    assert normalize_yes_no("maybe") is None
    assert majority_vote([True, False, True]) is True
    assert majority_vote([True, False]) is None

    assert cohen_kappa([(True, True), (False, False)]) == 1.0
    assert round(cohen_kappa([(True, True), (True, False), (False, False)]), 3) == 0.4


def test_human_audit_aggregation_outputs_majority_and_agreement(tmp_path):
    ann_path = tmp_path / "annotations.csv"
    fields = [
        "item_id",
        "corpus",
        "method",
        "ir_difficulty_label",
        "annotator_id",
        "question_understandable",
        "source_relevant",
        "target_relevant",
        "both_passages_necessary",
        "source_explicitly_cites_target",
        "answer_supported",
        "overall_valid",
    ]
    with ann_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "item_id": "q1",
            "corpus": "adgm",
            "method": "DPEL",
            "ir_difficulty_label": "hard",
            "annotator_id": "a1",
            "question_understandable": "yes",
            "source_relevant": "yes",
            "target_relevant": "yes",
            "both_passages_necessary": "yes",
            "source_explicitly_cites_target": "yes",
            "answer_supported": "yes",
            "overall_valid": "yes",
        })
        writer.writerow({
            "item_id": "q1",
            "corpus": "adgm",
            "method": "DPEL",
            "ir_difficulty_label": "hard",
            "annotator_id": "a2",
            "question_understandable": "yes",
            "source_relevant": "yes",
            "target_relevant": "no",
            "both_passages_necessary": "yes",
            "source_explicitly_cites_target": "yes",
            "answer_supported": "yes",
            "overall_valid": "no",
        })

    paths = aggregate_human_audit(inputs=[ann_path], out_dir=tmp_path / "out", min_group_size=2)

    assert paths["aggregated_items"].exists()
    assert paths["summary"].exists()
    assert paths["markdown"].exists()
    assert paths["agreement"].exists()
    with paths["aggregated_items"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["question_understandable_majority"] == "yes"
    assert rows[0]["target_relevant_majority"] == ""
    assert rows[0]["overall_valid_majority"] == ""
    agreement = json.loads(paths["agreement"].read_text(encoding="utf-8"))
    assert agreement["annotator_count"] == 2
    assert agreement["percent_agreement"]["question_understandable"] == 1.0
    assert agreement["percent_agreement"]["target_relevant"] == 0.5
    assert "target_relevant" in agreement["cohen_kappa"]
