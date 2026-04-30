from __future__ import annotations

import csv
import json
from pathlib import Path

from obliqaxref.benchmark_metadata import OBLIQA_XREF_METADATA, OBLIQA_XREF_METADATA_FIELDS
from obliqaxref.curate.run import assemble_final_benchmark
from obliqaxref.eval.FinalizeDataset.finalize_dataset import finalize_dataset_main


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_final_cohort_jsonl_and_csv_contain_obliqa_metadata(tmp_path):
    curate_out = tmp_path / "curate"
    _write_jsonl(
        curate_out / "curate_judge" / "judge_responses_pass.jsonl",
        [{"item_id": "q1", "decision_qp_final": "PASS_QP"}],
    )
    _write_jsonl(
        curate_out / "curate_answer" / "answer_responses_pass.jsonl",
        [{"item_id": "q1", "decision_ans_final": "PASS_ANS"}],
    )
    _write_jsonl(curate_out / "curate_answer" / "answer_responses_drop.jsonl", [])
    _write_jsonl(
        curate_out / "curated_items.judge.jsonl",
        [{
            "item_id": "q1",
            "ir_difficulty_label": "easy",
            "source_vote_count": 1,
            "target_vote_count": 1,
            "both_vote_count": 1,
        }],
    )
    items_file = tmp_path / "items.jsonl"
    _write_jsonl(
        items_file,
        [{
            "item_id": "q1",
            "question": "Q?",
            "gold_answer": "A [#SRC:s1] [#TGT:t1]",
            "source_passage_id": "s1",
            "target_passage_id": "t1",
            "method": "DPEL",
            "pair_uid": "p1",
        }],
    )

    assemble_final_benchmark(curate_out, items_file)

    record = _read_jsonl(curate_out / "final_answer_valid.jsonl")[0]
    for field, expected in OBLIQA_XREF_METADATA.items():
        assert record[field] == expected

    with (curate_out / "final_answer_valid.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert set(OBLIQA_XREF_METADATA_FIELDS).issubset(set(reader.fieldnames or []))
        row = next(reader)
    assert row["benchmark_family"] == "ObliQA"
    assert row["benchmark_name"] == "ObliQA-XRef"


def test_finalize_injects_metadata_for_older_records_and_preserves_ids_and_splits(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    curate_out = tmp_path / "runs" / "curate_adgm" / "out"
    _write_jsonl(
        curate_out / "final_answer_valid.jsonl",
        [
            {
                "item_id": "q1",
                "question": "Q1",
                "gold_answer": "A1",
                "source_passage_id": "s1",
                "target_passage_id": "t1",
                "method": "DPEL",
                "split": "train",
            },
            {
                "item_id": "q2",
                "question": "Q2",
                "gold_answer": "A2",
                "source_passage_id": "s2",
                "target_passage_id": "t2",
                "method": "SCHEMA",
                "split": "test",
            },
        ],
    )

    finalize_dataset_main(out_dir="out", corpus="adgm", cohort="answer_valid", seed=123)

    all_records = _read_jsonl(tmp_path / "out" / "ObliQA-XRef-ADGM-ALL.jsonl")
    assert [record["item_id"] for record in all_records] == ["q1", "q2"]
    for record in all_records:
        assert record["benchmark_family"] == "ObliQA"
        assert record["benchmark_name"] == "ObliQA-XRef"

    train_ids = [record["item_id"] for record in _read_jsonl(tmp_path / "out" / "ObliQA-XRef-ADGM-ALL-train.jsonl")]
    test_ids = [record["item_id"] for record in _read_jsonl(tmp_path / "out" / "ObliQA-XRef-ADGM-ALL-test.jsonl")]
    assert train_ids == ["q1"]
    assert test_ids == ["q2"]
