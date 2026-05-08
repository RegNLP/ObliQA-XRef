#!/usr/bin/env python3
"""Compute retrieval metrics for the 5+5+5 controlled-comparison pilot."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytrec_eval


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from obliqaxref.eval.DownstreamEval.ir_eval import (  # noqa: E402
    aggregate_pair_metrics,
    compute_pair_diagnostics,
)


K = 10

DATASETS = {
    "ObliQA": {
        "split": "test",
        "sample_type": "pilot_seed42_test5",
        "sample_size": 5,
        "retriever": "bm25",
        "qrels": ROOT / "qrels/obliqa_test5.qrels",
        "run": ROOT / "runs/obliqa_bm25_top10.trec",
        "xref": False,
    },
    "ObliQA-MP": {
        "split": "test",
        "sample_type": "pilot_seed42_test5",
        "sample_size": 5,
        "retriever": "bm25",
        "qrels": ROOT / "qrels/obliqa_mp_test5.qrels",
        "run": ROOT / "runs/obliqa_mp_bm25_top10.trec",
        "xref": False,
    },
    "ObliQA-XRef-ADGM": {
        "split": "test",
        "sample_type": "pilot_seed42_test5_2dpel_3schema",
        "sample_size": 5,
        "retriever": "bm25",
        "qrels": ROOT / "qrels/xref_adgm_test5.qrels",
        "run": ROOT / "runs/xref_adgm_bm25_top10.trec",
        "sample_metadata": ROOT / "samples/xref_adgm_test5.json",
        "xref": True,
    },
}

FIELDNAMES = [
    "dataset",
    "split",
    "sample_type",
    "sample_size",
    "retriever",
    "k",
    "Recall@10",
    "MAP@10",
    "nDCG@10",
    "Hit@10",
    "RelCount@10",
    "Both@10",
    "SRC-only@10",
    "TGT-only@10",
    "Neither@10",
    "PairMRR",
    "notes",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Malformed qrels line {path}:{lineno}: {line}")
            qid, _zero, docid, rel = parts
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def read_run(path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 6:
                raise ValueError(f"Malformed TREC run line {path}:{lineno}: {line}")
            qid, _q0, docid, _rank, score, _tag = parts
            run.setdefault(qid, {})[docid] = float(score)
    return run


def topk_docids(scores: dict[str, float], k: int) -> list[str]:
    return [docid for docid, _score in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]


def compute_common_metrics(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    *,
    k: int,
) -> dict[str, float]:
    all_qids = set(qrels)
    run_eval = {qid: run.get(qid, {}) for qid in all_qids}
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {f"recall_{k}", f"map_cut_{k}", f"ndcg_cut_{k}"},
    )
    results = evaluator.evaluate(run_eval)
    denom = max(1, len(all_qids))

    hit_count = 0
    rel_count_total = 0
    for qid in all_qids:
        relevant = {docid for docid, rel in qrels[qid].items() if rel > 0}
        retrieved_relevant = relevant & set(topk_docids(run.get(qid, {}), k))
        if retrieved_relevant:
            hit_count += 1
        rel_count_total += len(retrieved_relevant)

    return {
        f"Recall@{k}": sum(results[qid].get(f"recall_{k}", 0.0) for qid in all_qids) / denom,
        f"MAP@{k}": sum(results[qid].get(f"map_cut_{k}", 0.0) for qid in all_qids) / denom,
        f"nDCG@{k}": sum(results[qid].get(f"ndcg_cut_{k}", 0.0) for qid in all_qids) / denom,
        f"Hit@{k}": hit_count / denom,
        f"RelCount@{k}": rel_count_total / denom,
    }


def load_xref_maps(sample_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    samples = read_json(sample_path)
    src_map: dict[str, str] = {}
    tgt_map: dict[str, str] = {}
    for row in samples:
        qid = str(row.get("item_id") or "")
        src = str(row.get("source_passage_id") or "")
        tgt = str(row.get("target_passage_id") or "")
        if qid and src and tgt:
            src_map[qid] = src
            tgt_map[qid] = tgt
    return src_map, tgt_map


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def compute_row(dataset: str, cfg: dict[str, Any]) -> dict[str, Any]:
    qrels = read_qrels(cfg["qrels"])
    run = read_run(cfg["run"])
    notes: list[str] = []

    missing_run_qids = sorted(set(qrels) - set(run))
    if missing_run_qids:
        notes.append(f"missing_run_qids={len(missing_run_qids)}")

    common = compute_common_metrics(qrels, run, k=K)
    row: dict[str, Any] = {
        "dataset": dataset,
        "split": cfg["split"],
        "sample_type": cfg["sample_type"],
        "sample_size": cfg["sample_size"],
        "retriever": cfg["retriever"],
        "k": K,
        "Recall@10": common["Recall@10"],
        "MAP@10": common["MAP@10"],
        "nDCG@10": common["nDCG@10"],
        "Hit@10": common["Hit@10"],
        "RelCount@10": common["RelCount@10"],
        "Both@10": None,
        "SRC-only@10": None,
        "TGT-only@10": None,
        "Neither@10": None,
        "PairMRR": None,
        "notes": "; ".join(notes),
    }

    if cfg.get("xref"):
        src_map, tgt_map = load_xref_maps(cfg["sample_metadata"])
        diagnostics = compute_pair_diagnostics(
            run,
            src_map,
            tgt_map,
            qids=set(qrels),
            ks=(K,),
            retriever=cfg["retriever"],
            corpus=dataset,
            method="pilot",
            split=cfg["split"],
        )
        pair_metrics = aggregate_pair_metrics(diagnostics, ks=(K,))
        for key in ["Both@10", "SRC-only@10", "TGT-only@10", "Neither@10", "PairMRR"]:
            row[key] = pair_metrics[key]

    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for field in [
                "Recall@10",
                "MAP@10",
                "nDCG@10",
                "Hit@10",
                "RelCount@10",
                "Both@10",
                "SRC-only@10",
                "TGT-only@10",
                "Neither@10",
                "PairMRR",
            ]:
                out[field] = format_float(out.get(field))
            writer.writerow(out)


def main() -> None:
    rows = [compute_row(dataset, cfg) for dataset, cfg in DATASETS.items()]
    json_path = ROOT / "reports/pilot_retrieval_metrics.json"
    csv_path = ROOT / "reports/pilot_retrieval_metrics.csv"
    write_json(json_path, rows)
    write_csv(csv_path, rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
