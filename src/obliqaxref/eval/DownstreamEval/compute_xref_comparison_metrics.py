#!/usr/bin/env python3
"""Compute compact retrieval metrics for the ObliQA family comparison table.

This script is intentionally separate from ir_eval.py so it does not overwrite
the full retrieval diagnostics already produced for the paper.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from obliqaxref.eval.DownstreamEval import ir_eval


def compute_official_metrics(
    *,
    corpus: str,
    root: Path,
    retriever: str,
    items: list[dict[str, Any]],
    k: int,
) -> dict[str, float]:
    """Use the same pytrec_eval-backed implementation as ir_eval.py."""
    qrels, src_map, tgt_map = ir_eval.build_qrels(items)
    run = ir_eval.load_trec_run(corpus, retriever, root)
    metrics = ir_eval.compute_metrics(
        run=run,
        qrels=qrels,
        src_map=src_map,
        tgt_map=tgt_map,
        k=k,
        diag_samples=0,
    )
    metrics["test_size"] = float(len(items))
    return metrics


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="ObliQA-XRef_Out_Datasets/final_large_merged")
    ap.add_argument("--corpus", default="adgm")
    ap.add_argument("--retrievers", nargs="+", default=["bm25", "bm25_fusion"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--out_csv",
        default="ObliQA-XRef_Out_Datasets/final_large_merged/retrieval_metrics_xref_comparison.csv",
    )
    args = ap.parse_args()

    root = Path(args.root)
    items_all = ir_eval.load_test_split(args.corpus, root)
    items_dpel = ir_eval.load_test_split_subset(args.corpus, "DPEL", root)
    items_schema = ir_eval.load_test_split_subset(args.corpus, "SCHEMA", root)
    corpus_label = args.corpus.upper()
    slices = [
        (f"{corpus_label}-ALL", items_all),
        (f"{corpus_label}-DPEL", items_dpel),
        (f"{corpus_label}-SCHEMA", items_schema),
    ]

    rows: list[dict[str, Any]] = []
    for retriever in args.retrievers:
        for dataset_slice, slice_items in slices:
            metrics = compute_official_metrics(
                corpus=args.corpus,
                root=root,
                retriever=retriever,
                items=slice_items,
                k=args.k,
            )
            rows.append(
                {
                    "dataset_slice": dataset_slice,
                    "sampling": "all",
                    "retriever": retriever,
                    "test_size": int(metrics["test_size"]),
                    f"Recall@{args.k}": f"{metrics[f'Recall@{args.k}']:.6f}",
                    f"MAP@{args.k}": f"{metrics[f'MAP@{args.k}']:.6f}",
                    f"Both@{args.k}": f"{metrics[f'Both@{args.k}']:.6f}",
                    "PairMRR": f"{metrics['PairMRR']:.6f}",
                }
            )

    fieldnames = [
        "dataset_slice",
        "sampling",
        "retriever",
        "test_size",
        f"Recall@{args.k}",
        f"MAP@{args.k}",
        f"Both@{args.k}",
        "PairMRR",
    ]
    write_csv(Path(args.out_csv), rows, fieldnames)


if __name__ == "__main__":
    main()
