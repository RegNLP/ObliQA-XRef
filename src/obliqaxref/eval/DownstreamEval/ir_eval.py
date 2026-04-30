#!/usr/bin/env python3

"""
ObliQA-XRef Downstream IR Evaluation (Test Split)

Loads:
- test split items (JSONL) -> builds qrels with exactly two relevant docs per query (S and T)
- TREC run files (method.trec) -> scores per query-doc

Computes (at cutoff k):
- Recall@k, MAP@k (map_cut_k), nDCG@k (ndcg_cut_k) via pytrec_eval, averaged over ALL test queries
- Citation-aware diagnostics over ALL test queries:
    Both@k, SRC-only@k, TGT-only@k, Neither@k
  for k={5,10,20}, plus PairMRR

Also prints a small sample of Neither@k cases for debugging.
Also writes retrieval_metrics_full.csv and retrieval_diagnostics_per_query.csv.

Expected TREC format per line:
qid Q0 docid rank score tag

Paths:
ObliQA-XRef_Out_Datasets/
  ObliQA-XRef-{CORPUS}-ALL/
    test.jsonl
    bm25.trec
    e5.trec
    rrf.trec
    ce_rerank_union200.trec
"""

import json
import logging
import random
import csv
from pathlib import Path
from typing import Any

import pytrec_eval

# ----------------------------
# IO helpers
# ----------------------------


def _normalize_docid(docid: str, strip_hyphens: bool = False) -> str:
    if not isinstance(docid, str):
        docid = str(docid)
    if strip_hyphens:
        return docid.replace("-", "")
    return docid


def load_test_split(corpus: str, root: Path) -> list[dict[str, Any]]:
    split_path = root / f"ObliQA-XRef-{corpus.upper()}-ALL" / "test.jsonl"
    if not split_path.exists():
        alt = root / f"ObliQA-XRef-{corpus.upper()}-ALL-test.jsonl"
        if alt.exists():
            split_path = alt
        else:
            raise FileNotFoundError(f"Missing test split: {split_path}")
    items: list[dict[str, Any]] = []
    with split_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_test_split_subset(corpus: str, subset: str, root: Path) -> list[dict[str, Any]]:
    """
    Load per-generation-method test split when available, e.g.:
      ObliQA-XRef_Out_Datasets/ObliQA-XRef-UKFIN-DPEL-ALL-test.jsonl
      ObliQA-XRef_Out_Datasets/ObliQA-XRef-UKFIN-SCHEMA-ALL-test.jsonl
    Fallback to filtering the combined test.jsonl by item['method'] == subset.
    """
    subset_up = subset.strip().upper()
    direct = root / f"ObliQA-XRef-{corpus.upper()}-{subset_up}-ALL-test.jsonl"
    items: list[dict[str, Any]] = []
    if direct.exists():
        with direct.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    # fallback to combined + filter
    combined = load_test_split(corpus, root)
    for it in combined:
        if (it.get("method") or "").strip().upper() == subset_up:
            items.append(it)
    return items


def load_trec_run(
    corpus: str, method: str, root: Path, normalize_docids: bool = False
) -> dict[str, dict[str, float]]:
    trec_path = root / f"ObliQA-XRef-{corpus.upper()}-ALL" / f"{method}.trec"
    if not trec_path.exists():
        raise FileNotFoundError(f"Missing TREC run: {trec_path}")

    run: dict[str, dict[str, float]] = {}
    with trec_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 6:
                # keep going but warn; malformed lines can drop queries silently
                logging.warning(
                    f"[{trec_path.name}:{lineno}] Malformed TREC line (expected 6 cols): {line}"
                )
                continue

            qid, _q0, docid, _rank, score, _tag = parts
            try:
                s = float(score)
            except ValueError:
                logging.warning(f"[{trec_path.name}:{lineno}] Non-float score: {score}")
                continue
            if normalize_docids:
                docid = _normalize_docid(docid, strip_hyphens=True)
            run.setdefault(qid, {})[docid] = s
    return run


# ----------------------------
# Qrels construction
# ----------------------------


def build_qrels(
    items: list[dict[str, Any]], normalize_docids: bool = False
) -> tuple[dict[str, dict[str, int]], dict[str, str], dict[str, str]]:
    """
    Returns:
      qrels[qid] = {src_pid: 1, tgt_pid: 1}
      src_map[qid] = src_pid
      tgt_map[qid] = tgt_pid
    """
    qrels: dict[str, dict[str, int]] = {}
    src_map: dict[str, str] = {}
    tgt_map: dict[str, str] = {}

    for item in items:
        qid = item["item_id"]
        src_pid = item["source_passage_id"]
        tgt_pid = item["target_passage_id"]
        if normalize_docids:
            src_pid = _normalize_docid(src_pid, strip_hyphens=True)
            tgt_pid = _normalize_docid(tgt_pid, strip_hyphens=True)

        if qid in qrels:
            logging.warning(f"Duplicate item_id in test split: {qid} (overwriting)")

        qrels[qid] = {src_pid: 1, tgt_pid: 1}
        src_map[qid] = src_pid
        tgt_map[qid] = tgt_pid

    return qrels, src_map, tgt_map


# ----------------------------
# Evaluation
# ----------------------------


PAIR_DIAGNOSTIC_KS = (5, 10, 20)


def _topk_docids(doc_scores: dict[str, float], k: int) -> list[str]:
    if not doc_scores:
        return []
    return [docid for docid, _ in sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))[:k]]


def _ranked_docids(doc_scores: dict[str, float]) -> list[str]:
    if not doc_scores:
        return []
    return [docid for docid, _ in sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))]


def _retrieval_outcome(found_src: bool, found_tgt: bool) -> str:
    if found_src and found_tgt:
        return "both"
    if found_src:
        return "src_only"
    if found_tgt:
        return "tgt_only"
    return "neither"


def compute_pair_diagnostics(
    run: dict[str, dict[str, float]],
    src_map: dict[str, str],
    tgt_map: dict[str, str],
    *,
    qids: set[str] | None = None,
    ks: tuple[int, ...] = PAIR_DIAGNOSTIC_KS,
    retriever: str = "",
    corpus: str = "",
    method: str = "",
    split: str = "",
) -> list[dict[str, Any]]:
    """Compute per-query citation-pair diagnostics for a retrieval run."""
    eval_qids = sorted(qids if qids is not None else set(src_map) | set(tgt_map))
    diagnostics: list[dict[str, Any]] = []

    for qid in eval_qids:
        source_id = src_map.get(qid)
        target_id = tgt_map.get(qid)
        ranked = _ranked_docids(run.get(qid, {}))
        rank_by_docid = {docid: idx for idx, docid in enumerate(ranked, start=1)}

        source_rank = rank_by_docid.get(source_id) if source_id else None
        target_rank = rank_by_docid.get(target_id) if target_id else None
        if source_rank is not None and target_rank is not None:
            pair_rank = max(source_rank, target_rank)
            pair_rr = 1.0 / pair_rank
        else:
            pair_rank = None
            pair_rr = 0.0

        row: dict[str, Any] = {
            "corpus": corpus,
            "dataset": corpus,
            "method": method,
            "retriever": retriever,
            "split": split,
            "query_id": qid,
            "item_id": qid,
            "source_id": source_id,
            "target_id": target_id,
            "source_rank": source_rank,
            "target_rank": target_rank,
            "pair_rank": pair_rank,
            "pair_rr": pair_rr,
        }

        for k in ks:
            topk_ids = set(ranked[:k])
            found_src = bool(source_id) and source_id in topk_ids
            found_tgt = bool(target_id) and target_id in topk_ids
            row[f"retrieval_outcome_at_{k}"] = _retrieval_outcome(found_src, found_tgt)

        diagnostics.append(row)

    return diagnostics


def aggregate_pair_metrics(
    diagnostics: list[dict[str, Any]],
    *,
    ks: tuple[int, ...] = PAIR_DIAGNOSTIC_KS,
) -> dict[str, float]:
    """Aggregate mutually exclusive pair retrieval outcomes and PairMRR."""
    denom = max(1, len(diagnostics))
    metrics: dict[str, float] = {}

    for k in ks:
        counts = {"both": 0, "src_only": 0, "tgt_only": 0, "neither": 0}
        for row in diagnostics:
            outcome = row.get(f"retrieval_outcome_at_{k}", "neither")
            counts[outcome] = counts.get(outcome, 0) + 1
        metrics[f"Both@{k}"] = counts["both"] / denom
        metrics[f"SRC-only@{k}"] = counts["src_only"] / denom
        metrics[f"TGT-only@{k}"] = counts["tgt_only"] / denom
        metrics[f"Neither@{k}"] = counts["neither"] / denom

    metrics["PairMRR"] = sum(float(row.get("pair_rr") or 0.0) for row in diagnostics) / denom
    return metrics


def compute_metrics(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    src_map: dict[str, str],
    tgt_map: dict[str, str],
    k: int = 10,
    ks: tuple[int, ...] = PAIR_DIAGNOSTIC_KS,
    diag_samples: int = 5,
    seed: int = 13,
) -> dict[str, float]:
    """
    Computes pytrec_eval metrics averaged over ALL qrels queries,
    plus citation-aware diagnostics over ALL qrels queries.

    Important: if a qid is missing from run, it is treated as an empty ranking (all metrics 0).
    """
    all_qids = set(qrels.keys())
    all_ks = tuple(sorted(set(ks) | {k}))

    # For pytrec_eval: ensure every qid exists in run_eval
    run_eval: dict[str, dict[str, float]] = {qid: run.get(qid, {}) for qid in all_qids}
    qrels_eval: dict[str, dict[str, int]] = {qid: qrels[qid] for qid in all_qids}

    metrics_set = set()
    for cutoff in all_ks:
        metrics_set.update({f"recall_{cutoff}", f"ndcg_cut_{cutoff}", f"map_cut_{cutoff}"})
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_eval, metrics_set)
    results = evaluator.evaluate(run_eval)

    # Average over ALL test queries
    denom = max(1, len(all_qids))
    metrics_out: dict[str, float] = {}
    for cutoff in all_ks:
        metrics_out[f"Recall@{cutoff}"] = (
            sum(results[qid].get(f"recall_{cutoff}", 0.0) for qid in all_qids) / denom
        )
        metrics_out[f"MAP@{cutoff}"] = (
            sum(results[qid].get(f"map_cut_{cutoff}", 0.0) for qid in all_qids) / denom
        )
        metrics_out[f"nDCG@{cutoff}"] = (
            sum(results[qid].get(f"ndcg_cut_{cutoff}", 0.0) for qid in all_qids) / denom
        )

    diagnostics = compute_pair_diagnostics(run, src_map, tgt_map, qids=all_qids, ks=all_ks)
    metrics_out.update(aggregate_pair_metrics(diagnostics, ks=all_ks))

    # Print a few Neither@k samples (from qrels population, not only run-overlap)
    neither_cases: list[dict[str, Any]] = []
    for row in diagnostics:
        if row.get(f"retrieval_outcome_at_{k}") == "neither":
            neither_cases.append(
                {
                    "qid": row["query_id"],
                    "src_id": row["source_id"],
                    "tgt_id": row["target_id"],
                    "topk_ids": _topk_docids(run.get(row["query_id"], {}), k),
                }
            )

    if diag_samples > 0 and neither_cases:
        rng = random.Random(seed)
        sample = rng.sample(neither_cases, min(diag_samples, len(neither_cases)))
        print(f"\n--- DIAGNOSTIC: {len(sample)} random Neither@{k} queries ---")
        for q in sample:
            print(f"Query ID: {q['qid']}")
            print(f"Top-{k} passage IDs: {q['topk_ids']}")
            print(f"Source ID: {q['src_id']}")
            print(f"Target ID: {q['tgt_id']}")
            print("---")

    metrics_out.update({
        "num_qrels": float(len(all_qids)),
        "num_run_qids": float(len(run.keys())),
        "num_overlap_qids": float(len(all_qids & set(run.keys()))),
    })
    return metrics_out


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _preserve_other_corpora_rows(
    path: Path,
    *,
    current_corpus: str,
    fieldnames: list[str],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {field: row.get(field) for field in fieldnames}
        for row in rows
        if (row.get("corpus") or "").lower() != current_corpus.lower()
    ]


# ----------------------------
# Main
# ----------------------------


def main(
    corpus: str,
    k: int = 10,
    methods: list[str] = None,
    root_dir: str = "ObliQA-XRef_Out_Datasets",
    diag_samples: int = 5,
    normalize_docids: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("obliqaxref.ir_eval")

    root = Path(root_dir)

    if methods is None:
        methods = ["bm25", "e5", "rrf", "ce_rerank_union200"]

    logger.info(f"Running IR evaluation for corpus={corpus} on test split @k={k}")
    items_all = load_test_split(corpus, root=root)
    # Prepare full set info for logging only
    qrels_all, src_all, tgt_all = build_qrels(items_all, normalize_docids=normalize_docids)
    logger.info(f"Loaded combined test items: {len(items_all)} (qrels queries: {len(qrels_all)})")
    # Note: per user request, we save 4 outputs (per corpus × per gen-method),
    # prioritizing per-method test files if present.
    metrics_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for subset in ["DPEL", "SCHEMA"]:
        sub_items = load_test_split_subset(corpus, subset, root)
        if not sub_items:
            continue
        logger.info(f"Evaluating subset={subset} with {len(sub_items)} items ...")
        qrels_sub, src_sub, tgt_sub = build_qrels(sub_items, normalize_docids=normalize_docids)
        results_sub: dict[str, dict[str, float]] = {}
        for method in methods:
            logger.info(f"  method={method} on subset={subset} ...")
            run = load_trec_run(corpus, method, root=root, normalize_docids=normalize_docids)
            metrics = compute_metrics(
                run=run,
                qrels=qrels_sub,
                src_map=src_sub,
                tgt_map=tgt_sub,
                k=k,
                ks=PAIR_DIAGNOSTIC_KS,
                diag_samples=diag_samples,
            )
            results_sub[method] = metrics
            metrics_rows.append(
                {
                    "corpus": corpus,
                    "dataset": corpus,
                    "method": subset,
                    "retriever": method,
                    "split": "test",
                    **metrics,
                }
            )
            diagnostic_rows.extend(
                compute_pair_diagnostics(
                    run,
                    src_sub,
                    tgt_sub,
                    qids=set(qrels_sub.keys()),
                    ks=PAIR_DIAGNOSTIC_KS,
                    retriever=method,
                    corpus=corpus,
                    method=subset,
                    split="test",
                )
            )
        out_path_sub = root / f"ir_eval_{corpus}_{subset.lower()}_test.json"
        with out_path_sub.open("w", encoding="utf-8") as f:
            json.dump(results_sub, f, indent=2)
        logger.info(f"Saved IR eval results to {out_path_sub}")

    if metrics_rows:
        metric_fieldnames = [
            "corpus",
            "dataset",
            "method",
            "retriever",
            "split",
            "num_qrels",
            "num_run_qids",
            "num_overlap_qids",
        ]
        for cutoff in PAIR_DIAGNOSTIC_KS:
            metric_fieldnames.extend(
                [
                    f"Recall@{cutoff}",
                    f"MAP@{cutoff}",
                    f"nDCG@{cutoff}",
                    f"Both@{cutoff}",
                    f"SRC-only@{cutoff}",
                    f"TGT-only@{cutoff}",
                    f"Neither@{cutoff}",
                ]
            )
        if k not in PAIR_DIAGNOSTIC_KS:
            metric_fieldnames.extend([f"Recall@{k}", f"MAP@{k}", f"nDCG@{k}"])
        metric_fieldnames.append("PairMRR")
        metrics_path = root / "retrieval_metrics_full.csv"
        preserved_rows = _preserve_other_corpora_rows(
            metrics_path,
            current_corpus=corpus,
            fieldnames=metric_fieldnames,
        )
        _write_csv(metrics_path, preserved_rows + metrics_rows, metric_fieldnames)
        logger.info("Saved aggregate retrieval metrics to %s", metrics_path)

    if diagnostic_rows:
        diagnostic_fieldnames = [
            "corpus",
            "dataset",
            "method",
            "retriever",
            "split",
            "query_id",
            "item_id",
            "source_id",
            "target_id",
            "source_rank",
            "target_rank",
            "pair_rank",
            "retrieval_outcome_at_5",
            "retrieval_outcome_at_10",
            "retrieval_outcome_at_20",
            "pair_rr",
        ]
        diagnostics_path = root / "retrieval_diagnostics_per_query.csv"
        preserved_rows = _preserve_other_corpora_rows(
            diagnostics_path,
            current_corpus=corpus,
            fieldnames=diagnostic_fieldnames,
        )
        _write_csv(diagnostics_path, preserved_rows + diagnostic_rows, diagnostic_fieldnames)
        logger.info("Saved per-query retrieval diagnostics to %s", diagnostics_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="adgm", help="Corpus key, e.g., adgm or ukfin")
    ap.add_argument("--k", type=int, default=10, help="Cutoff k for metrics")
    ap.add_argument("--root", default="ObliQA-XRef_Out_Datasets", help="Root output directory")
    ap.add_argument(
        "--methods", nargs="*", default=None, help="List of method names (without .trec)"
    )
    ap.add_argument(
        "--diag-samples", type=int, default=5, help="How many Neither@k samples to print"
    )
    ap.add_argument(
        "--normalize-docids",
        action="store_true",
        help="Normalize doc IDs (strip hyphens) in both qrels and runs for matching",
    )
    args = ap.parse_args()

    main(
        corpus=args.corpus,
        k=args.k,
        methods=args.methods,
        root_dir=args.root,
        diag_samples=args.diag_samples,
        normalize_docids=args.normalize_docids,
    )
