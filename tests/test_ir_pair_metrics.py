from __future__ import annotations

import math
import csv
import json

from obliqaxref.eval.DownstreamEval.ir_eval import (
    aggregate_pair_metrics,
    compute_metrics,
    compute_pair_diagnostics,
    main,
)


def test_pair_diagnostics_both_retrieved_and_pair_mrr():
    run = {"q1": {"d0": 4.0, "src": 3.0, "d1": 2.0, "tgt": 1.0}}
    rows = compute_pair_diagnostics(
        run,
        {"q1": "src"},
        {"q1": "tgt"},
        qids={"q1"},
        ks=(5, 10, 20),
        retriever="bm25",
        corpus="adgm",
        method="DPEL",
        split="test",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["query_id"] == "q1"
    assert row["item_id"] == "q1"
    assert row["retriever"] == "bm25"
    assert row["source_rank"] == 2
    assert row["target_rank"] == 4
    assert row["pair_rank"] == 4
    assert row["pair_rr"] == 0.25
    assert row["retrieval_outcome_at_5"] == "both"
    assert row["retrieval_outcome_at_10"] == "both"
    assert row["retrieval_outcome_at_20"] == "both"

    metrics = aggregate_pair_metrics(rows, ks=(5, 10, 20))
    assert metrics["Both@5"] == 1.0
    assert metrics["PairMRR"] == 0.25


def test_pair_diagnostics_source_only_target_only_neither_are_exclusive():
    run = {
        "q_src": {"src": 3.0, "x": 2.0},
        "q_tgt": {"tgt": 3.0, "x": 2.0},
        "q_none": {"x": 3.0, "y": 2.0},
    }
    src_map = {"q_src": "src", "q_tgt": "src", "q_none": "src"}
    tgt_map = {"q_src": "tgt", "q_tgt": "tgt", "q_none": "tgt"}

    rows = compute_pair_diagnostics(run, src_map, tgt_map, qids=set(src_map), ks=(5, 10, 20))
    outcomes = {row["query_id"]: row["retrieval_outcome_at_5"] for row in rows}

    assert outcomes == {
        "q_src": "src_only",
        "q_tgt": "tgt_only",
        "q_none": "neither",
    }

    metrics = aggregate_pair_metrics(rows, ks=(5, 10, 20))
    for k in (5, 10, 20):
        total = (
            metrics[f"Both@{k}"]
            + metrics[f"SRC-only@{k}"]
            + metrics[f"TGT-only@{k}"]
            + metrics[f"Neither@{k}"]
        )
        assert math.isclose(total, 1.0)
        assert metrics[f"Both@{k}"] == 0.0
        assert metrics[f"SRC-only@{k}"] == 1 / 3
        assert metrics[f"TGT-only@{k}"] == 1 / 3
        assert metrics[f"Neither@{k}"] == 1 / 3
    assert metrics["PairMRR"] == 0.0


def test_pair_diagnostics_cutoff_behavior_for_5_10_20():
    run = {
        "q1": {
            "d1": 20.0,
            "d2": 19.0,
            "d3": 18.0,
            "d4": 17.0,
            "src": 16.0,
            "d6": 15.0,
            "d7": 14.0,
            "d8": 13.0,
            "d9": 12.0,
            "tgt": 11.0,
        }
    }
    rows = compute_pair_diagnostics(
        run,
        {"q1": "src"},
        {"q1": "tgt"},
        qids={"q1"},
        ks=(5, 10, 20),
    )

    row = rows[0]
    assert row["source_rank"] == 5
    assert row["target_rank"] == 10
    assert row["pair_rank"] == 10
    assert row["pair_rr"] == 0.1
    assert row["retrieval_outcome_at_5"] == "src_only"
    assert row["retrieval_outcome_at_10"] == "both"
    assert row["retrieval_outcome_at_20"] == "both"


def test_pair_diagnostics_missing_source_and_target_ranks():
    rows = compute_pair_diagnostics(
        {"q1": {"d1": 2.0, "d2": 1.0}},
        {"q1": "src"},
        {"q1": "tgt"},
        qids={"q1"},
        ks=(5, 10, 20),
    )

    row = rows[0]
    assert row["source_rank"] is None
    assert row["target_rank"] is None
    assert row["pair_rank"] is None
    assert row["pair_rr"] == 0.0
    assert row["retrieval_outcome_at_5"] == "neither"
    assert row["retrieval_outcome_at_10"] == "neither"
    assert row["retrieval_outcome_at_20"] == "neither"


def test_compute_metrics_preserves_standard_metrics_and_adds_pair_keys():
    run = {"q1": {"src": 2.0, "tgt": 1.0}}
    qrels = {"q1": {"src": 1, "tgt": 1}}
    metrics = compute_metrics(
        run,
        qrels,
        {"q1": "src"},
        {"q1": "tgt"},
        k=10,
        ks=(5, 10, 20),
        diag_samples=0,
    )

    for k in (5, 10, 20):
        assert f"Recall@{k}" in metrics
        assert f"MAP@{k}" in metrics
        assert f"nDCG@{k}" in metrics
        assert metrics[f"Both@{k}"] == 1.0
    assert metrics["PairMRR"] == 0.5


def test_main_writes_required_retrieval_csv_outputs(tmp_path):
    dataset_dir = tmp_path / "ObliQA-XRef-ADGM-ALL"
    dataset_dir.mkdir()
    item = {
        "item_id": "q1",
        "question": "Q",
        "source_passage_id": "src",
        "target_passage_id": "tgt",
        "method": "DPEL",
    }
    for path in (
        dataset_dir / "test.jsonl",
        tmp_path / "ObliQA-XRef-ADGM-DPEL-ALL-test.jsonl",
    ):
        path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    (dataset_dir / "bm25.trec").write_text(
        "q1 Q0 src 1 2.0 bm25\nq1 Q0 tgt 2 1.0 bm25\n",
        encoding="utf-8",
    )

    main(corpus="adgm", methods=["bm25"], root_dir=str(tmp_path), diag_samples=0)

    metrics_path = tmp_path / "retrieval_metrics_full.csv"
    diagnostics_path = tmp_path / "retrieval_diagnostics_per_query.csv"
    assert metrics_path.exists()
    assert diagnostics_path.exists()

    with metrics_path.open(newline="", encoding="utf-8") as f:
        metrics_rows = list(csv.DictReader(f))
    assert metrics_rows[0]["corpus"] == "adgm"
    assert metrics_rows[0]["method"] == "DPEL"
    assert metrics_rows[0]["retriever"] == "bm25"
    assert float(metrics_rows[0]["Both@5"]) == 1.0
    assert float(metrics_rows[0]["PairMRR"]) == 0.5

    with diagnostics_path.open(newline="", encoding="utf-8") as f:
        diagnostic_rows = list(csv.DictReader(f))
    assert diagnostic_rows[0]["query_id"] == "q1"
    assert diagnostic_rows[0]["retrieval_outcome_at_5"] == "both"
    assert diagnostic_rows[0]["source_rank"] == "1"
    assert diagnostic_rows[0]["target_rank"] == "2"
