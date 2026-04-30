from __future__ import annotations

import json
from pathlib import Path

from obliqaxref.eval.FinalizeDataset.finalize_dataset import finalize_dataset_main


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_finalize_prefers_explicit_answer_valid_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    curate_out = tmp_path / "runs" / "curate_adgm" / "out"

    _write_jsonl(
        curate_out / "final_answer_valid.jsonl",
        [{
            "item_id": "q_answer_valid",
            "question": "Q",
            "gold_answer": "A",
            "source_passage_id": "s",
            "target_passage_id": "t",
            "method": "dpel",
        }],
    )
    _write_jsonl(
        curate_out / "curated_items.keep.jsonl",
        [{"item_id": "q_stale_keep", "decision": "KEEP"}],
    )

    finalize_dataset_main(out_dir="out", corpus="adgm", cohort="answer_valid", seed=1)

    records = _read_jsonl(tmp_path / "out" / "ObliQA-XRef-ADGM-ALL.jsonl")
    assert [record["item_id"] for record in records] == ["q_answer_valid"]


def test_finalize_fallback_ignores_stale_keep_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    curate_out = tmp_path / "runs" / "curate_adgm" / "out"

    _write_jsonl(
        curate_out / "curated_items.judge.jsonl",
        [
            {
                "item_id": "q_pass",
                "question": "Q",
                "gold_answer": "A",
                "source_passage_id": "s",
                "target_passage_id": "t",
                "method": "dpel",
            },
            {
                "item_id": "q_stale_keep",
                "question": "Q",
                "gold_answer": "A",
                "source_passage_id": "s",
                "target_passage_id": "t",
                "method": "schema",
            },
        ],
    )
    _write_jsonl(
        curate_out / "curate_judge" / "judge_responses_pass.jsonl",
        [{"item_id": "q_pass", "decision_qp_final": "PASS_QP"}],
    )
    _write_jsonl(
        curate_out / "curate_answer" / "answer_responses_pass.jsonl",
        [{"item_id": "q_pass", "decision_ans_final": "PASS_ANS"}],
    )
    _write_jsonl(
        curate_out / "curated_items.keep.jsonl",
        [{"item_id": "q_stale_keep", "decision": "KEEP"}],
    )

    finalize_dataset_main(out_dir="out", corpus="adgm", cohort="answer_pass", seed=1)

    records = _read_jsonl(tmp_path / "out" / "ObliQA-XRef-ADGM-ALL.jsonl")
    assert [record["item_id"] for record in records] == ["q_pass"]
