#!/usr/bin/env python3
"""Compute a combined controlled IR table from saved canonical BM25 outputs."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    import pytrec_eval
except Exception:  # pragma: no cover - optional runtime dependency
    pytrec_eval = None


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


K = 10
RETRIEVER = "bm25"
RETRIEVAL_SETTING = "controlled_shared_corpus_canonical_bm25"
RETRIEVAL_CORPUS = "shared_adgm_13015"
XREF_TEST_PATH = (
    REPO_ROOT / "ObliQA-XRef_Out_Datasets/final_large_merged/ObliQA-XRef-ADGM-ALL/test.jsonl"
)
REPORT_DIR = ROOT / "full_ir/reports"

CSV_OUT = REPORT_DIR / "combined_controlled_ir_table.csv"
JSON_OUT = REPORT_DIR / "combined_controlled_ir_table.json"
MD_OUT = REPORT_DIR / "combined_controlled_ir_table.md"
LATEX_OUT = REPORT_DIR / "combined_controlled_ir_table_latex_draft.tex"

OBLIQA_QRELS = ROOT / "full_ir/qrels/obliqa_test.qrels"
OBLIQA_MP_QRELS = ROOT / "full_ir/qrels/obliqa_mp_test.qrels"
XREF_QRELS = ROOT / "full_ir/qrels/xref_adgm_test.qrels"

OBLIQA_RUN = ROOT / "full_ir/runs/obliqa_bm25_top10.trec"
OBLIQA_MP_RUN = ROOT / "full_ir/runs/obliqa_mp_bm25_top10.trec"
XREF_RUN = ROOT / "full_ir/runs/xref_adgm_bm25_top10.trec"

EXPECTED_COUNTS = {
    "ObliQA-XRef-ADGM-ALL": 502,
    "ObliQA-XRef-ADGM-DPEL": 173,
    "ObliQA-XRef-ADGM-SCHEMA": 329,
    "ObliQA-XRef-ADGM-DPEL + mixed_difficulty": 99,
    "ObliQA-XRef-ADGM-DPEL + hard_enriched": 74,
    "ObliQA-XRef-ADGM-SCHEMA + mixed_difficulty": 188,
    "ObliQA-XRef-ADGM-SCHEMA + hard_enriched": 141,
}

FIELDNAMES = [
    "dataset_or_slice",
    "method_type",
    "sampling",
    "test_size",
    "retriever",
    "retrieval_setting",
    "retrieval_corpus",
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
    return [docid for docid, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]


def filter_mapping(mapping: dict[str, dict[str, Any]], qids: set[str]) -> dict[str, dict[str, Any]]:
    return {qid: mapping[qid] for qid in qids if qid in mapping}


def average_precision_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, docid in enumerate(ranked[:k], start=1):
        if docid in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant)


def dcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    dcg = 0.0
    for rank, docid in enumerate(ranked[:k], start=1):
        if docid in relevant:
            dcg += 1.0 / (1.0 if rank == 1 else math.log2(rank))
    return dcg


def ndcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    ideal_len = min(k, len(relevant))
    if ideal_len == 0:
        return 0.0
    ideal_dcg = sum(1.0 / (1.0 if rank == 1 else math.log2(rank)) for rank in range(1, ideal_len + 1))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(relevant, ranked, k) / ideal_dcg


def compute_common_metrics(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    *,
    k: int,
) -> dict[str, float]:
    qids = set(qrels)
    denom = max(1, len(qids))

    hit = 0
    rel_count_total = 0
    recall_total = 0.0
    map_total = 0.0
    ndcg_total = 0.0

    res: dict[str, dict[str, float]] = {}
    if pytrec_eval is not None:
        eval_run = {qid: run.get(qid, {}) for qid in qids}
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels,
            {f"recall_{k}", f"map_cut_{k}", f"ndcg_cut_{k}"},
        )
        res = evaluator.evaluate(eval_run)

    for qid in qids:
        relevant = {docid for docid, rel in qrels[qid].items() if rel > 0}
        ranked = topk_docids(run.get(qid, {}), len(run.get(qid, {})))
        rel_at_k = relevant & set(ranked[:k])
        if rel_at_k:
            hit += 1
        rel_count_total += len(rel_at_k)
        if pytrec_eval is None:
            recall_total += len(rel_at_k) / max(1, len(relevant))
            map_total += average_precision_at_k(relevant, ranked, k)
            ndcg_total += ndcg_at_k(relevant, ranked, k)

    if pytrec_eval is not None:
        recall_total = sum(res[qid].get(f"recall_{k}", 0.0) for qid in qids)
        map_total = sum(res[qid].get(f"map_cut_{k}", 0.0) for qid in qids)
        ndcg_total = sum(res[qid].get(f"ndcg_cut_{k}", 0.0) for qid in qids)

    return {
        f"Recall@{k}": recall_total / denom,
        f"MAP@{k}": map_total / denom,
        f"nDCG@{k}": ndcg_total / denom,
        f"Hit@{k}": hit / denom,
        f"RelCount@{k}": rel_count_total / denom,
    }


def compute_xref_pair_metrics(
    run: dict[str, dict[str, float]],
    src_map: dict[str, str],
    tgt_map: dict[str, str],
    *,
    k: int,
    qids: set[str],
) -> dict[str, float]:
    denom = max(1, len(qids))
    both = 0
    src_only = 0
    tgt_only = 0
    neither = 0
    pair_mrr_total = 0.0

    for qid in sorted(qids):
        src = src_map.get(qid)
        tgt = tgt_map.get(qid)
        ranked_all = topk_docids(run.get(qid, {}), len(run.get(qid, {})))
        ranked_k = ranked_all[:k]
        in_k = set(ranked_k)
        found_src = bool(src) and src in in_k
        found_tgt = bool(tgt) and tgt in in_k

        if found_src and found_tgt:
            both += 1
        elif found_src:
            src_only += 1
        elif found_tgt:
            tgt_only += 1
        else:
            neither += 1

        seen_src = False
        seen_tgt = False
        rr = 0.0
        for rank, docid in enumerate(ranked_all, start=1):
            if src and docid == src:
                seen_src = True
            if tgt and docid == tgt:
                seen_tgt = True
            if seen_src and seen_tgt:
                rr = 1.0 / rank
                break
        pair_mrr_total += rr

    return {
        f"Both@{k}": both / denom,
        f"SRC-only@{k}": src_only / denom,
        f"TGT-only@{k}": tgt_only / denom,
        f"Neither@{k}": neither / denom,
        "PairMRR": pair_mrr_total / denom,
    }


def format_float(value: float | None, *, digits: int = 3, dash_for_none: bool = False) -> str:
    if value is None:
        return "—" if dash_for_none else ""
    return f"{value:.{digits}f}"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_markdown(rows: list[dict[str, Any]], warnings: list[str]) -> str:
    headers = [
        "dataset_or_slice",
        "method_type",
        "sampling",
        "test_size",
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
    ]
    lines = [
        "# Combined Controlled IR Table",
        "",
        "All rows use the same retrieval setting (`controlled_shared_corpus_canonical_bm25`) and the same shared retrieval corpus (`shared_adgm_13015`). ObliQA and ObliQA-MP do not have pair-aware metrics, so those cells are shown as —.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if header in {"Recall@10", "MAP@10", "nDCG@10", "Hit@10", "RelCount@10", "Both@10", "SRC-only@10", "TGT-only@10", "Neither@10", "PairMRR"}:
                values.append(format_float(value, digits=3, dash_for_none=True))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def build_latex(rows: list[dict[str, Any]], warnings: list[str]) -> str:
    metric_headers = [
        "Dataset",
        "Method",
        "Sampling",
        "N",
        "R@10",
        "MAP@10",
        "nDCG@10",
        "Hit@10",
        "RelCt@10",
        "Both@10",
        "SRC-only",
        "TGT-only",
        "Neither",
        "PairMRR",
    ]
    lines = [
        "% Draft table only. Do not include directly without paper-specific styling review.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lllrrrrrrrrrrr}",
        "\\toprule",
        " & ".join(metric_headers) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        vals = [
            latex_escape(str(row["dataset_or_slice"])),
            latex_escape(str(row["method_type"])),
            latex_escape(str(row["sampling"])),
            str(row["test_size"]),
            format_float(row["Recall@10"], digits=3, dash_for_none=True),
            format_float(row["MAP@10"], digits=3, dash_for_none=True),
            format_float(row["nDCG@10"], digits=3, dash_for_none=True),
            format_float(row["Hit@10"], digits=3, dash_for_none=True),
            format_float(row["RelCount@10"], digits=3, dash_for_none=True),
            format_float(row["Both@10"], digits=3, dash_for_none=True),
            format_float(row["SRC-only@10"], digits=3, dash_for_none=True),
            format_float(row["TGT-only@10"], digits=3, dash_for_none=True),
            format_float(row["Neither@10"], digits=3, dash_for_none=True),
            format_float(row["PairMRR"], digits=3, dash_for_none=True),
        ]
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Combined controlled IR results over the shared ADGM passage corpus using canonical BM25. Pair-aware metrics apply only to ObliQA-XRef slices.}",
        "\\label{tab:combined-controlled-ir}",
        "\\end{table*}",
    ])
    if warnings:
        lines.extend(["", "% Warnings: "])
        for warning in warnings:
            lines.append("% " + warning)
    lines.append("")
    return "\n".join(lines)


def load_xref_items() -> list[dict[str, Any]]:
    return read_jsonl(XREF_TEST_PATH)


def build_xref_slices(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], str, str]:
    method_field = "generation_method"
    sampling_field = "sampling_regime"
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    def make_slice(name: str, *, method: str | None = None, sampling: str | None = None) -> dict[str, Any]:
        qids = set()
        for item in items:
            if method is not None and item.get(method_field) != method:
                continue
            if sampling is not None and item.get(sampling_field) != sampling:
                continue
            qids.add(str(item["item_id"]))
        expected = EXPECTED_COUNTS.get(name)
        notes = []
        if expected is not None and len(qids) != expected:
            warning = f"count mismatch for {name}: expected {expected}, observed {len(qids)}"
            warnings.append(warning)
            notes.append(warning)
        return {
            "dataset_or_slice": name,
            "method_type": method or "ALL",
            "sampling": sampling or "ALL",
            "qids": qids,
            "notes": notes,
        }

    rows.append(make_slice("ObliQA-XRef-ADGM-ALL"))
    rows.append(make_slice("ObliQA-XRef-ADGM-DPEL", method="DPEL"))
    rows.append(make_slice("ObliQA-XRef-ADGM-SCHEMA", method="SCHEMA"))
    rows.append(make_slice("ObliQA-XRef-ADGM-DPEL + mixed_difficulty", method="DPEL", sampling="mixed_difficulty"))
    rows.append(make_slice("ObliQA-XRef-ADGM-DPEL + hard_enriched", method="DPEL", sampling="hard_enriched"))
    rows.append(make_slice("ObliQA-XRef-ADGM-SCHEMA + mixed_difficulty", method="SCHEMA", sampling="mixed_difficulty"))
    rows.append(make_slice("ObliQA-XRef-ADGM-SCHEMA + hard_enriched", method="SCHEMA", sampling="hard_enriched"))
    return rows, warnings, method_field, sampling_field


def compute_row(
    *,
    dataset_or_slice: str,
    method_type: str,
    sampling: str,
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    notes: list[str],
    src_map: dict[str, str] | None = None,
    tgt_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    common = compute_common_metrics(qrels, run, k=K)
    row: dict[str, Any] = {
        "dataset_or_slice": dataset_or_slice,
        "method_type": method_type,
        "sampling": sampling,
        "test_size": len(qrels),
        "retriever": RETRIEVER,
        "retrieval_setting": RETRIEVAL_SETTING,
        "retrieval_corpus": RETRIEVAL_CORPUS,
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
    if src_map is not None and tgt_map is not None:
        pair = compute_xref_pair_metrics(run, src_map, tgt_map, k=K, qids=set(qrels))
        row["Both@10"] = pair["Both@10"]
        row["SRC-only@10"] = pair["SRC-only@10"]
        row["TGT-only@10"] = pair["TGT-only@10"]
        row["Neither@10"] = pair["Neither@10"]
        row["PairMRR"] = pair["PairMRR"]
    return row


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    obliqa_qrels = read_qrels(OBLIQA_QRELS)
    obliqa_mp_qrels = read_qrels(OBLIQA_MP_QRELS)
    xref_qrels = read_qrels(XREF_QRELS)

    obliqa_run = read_run(OBLIQA_RUN)
    obliqa_mp_run = read_run(OBLIQA_MP_RUN)
    xref_run = read_run(XREF_RUN)

    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    rows.append(
        compute_row(
            dataset_or_slice="ObliQA",
            method_type="",
            sampling="",
            qrels=obliqa_qrels,
            run=obliqa_run,
            notes=[],
        )
    )
    rows.append(
        compute_row(
            dataset_or_slice="ObliQA-MP",
            method_type="",
            sampling="",
            qrels=obliqa_mp_qrels,
            run=obliqa_mp_run,
            notes=[],
        )
    )

    xref_items = load_xref_items()
    xref_slices, xref_warnings, method_field, sampling_field = build_xref_slices(xref_items)
    warnings.extend(xref_warnings)

    src_map_all = {str(item["item_id"]): str(item["source_passage_id"]) for item in xref_items}
    tgt_map_all = {str(item["item_id"]): str(item["target_passage_id"]) for item in xref_items}

    for xref_slice in xref_slices:
        qids = xref_slice["qids"]
        qrels_slice = {qid: xref_qrels[qid] for qid in qids if qid in xref_qrels}
        run_slice = {qid: xref_run.get(qid, {}) for qid in qids}
        src_map = {qid: src_map_all[qid] for qid in qids}
        tgt_map = {qid: tgt_map_all[qid] for qid in qids}
        notes = list(xref_slice["notes"])
        missing_in_qrels = sorted(qids - set(qrels_slice))
        if missing_in_qrels:
            note = f"missing_qrels_qids={len(missing_in_qrels)}"
            notes.append(note)
            warnings.append(f"{xref_slice['dataset_or_slice']}: {note}")
        missing_in_run = sorted(qids - set(xref_run))
        if missing_in_run:
            note = f"missing_run_qids={len(missing_in_run)}"
            notes.append(note)
            warnings.append(f"{xref_slice['dataset_or_slice']}: {note}")
        rows.append(
            compute_row(
                dataset_or_slice=xref_slice["dataset_or_slice"],
                method_type=xref_slice["method_type"],
                sampling=xref_slice["sampling"],
                qrels=qrels_slice,
                run=run_slice,
                notes=notes,
                src_map=src_map,
                tgt_map=tgt_map,
            )
        )

    write_csv(CSV_OUT, rows)
    write_json(JSON_OUT, rows)
    MD_OUT.write_text(build_markdown(rows, warnings), encoding="utf-8")
    LATEX_OUT.write_text(build_latex(rows, warnings), encoding="utf-8")

    print(f"Wrote {CSV_OUT}")
    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {MD_OUT}")
    print(f"Wrote {LATEX_OUT}")
    print(f"XRef method field: {method_field}")
    print(f"XRef sampling field: {sampling_field}")
    if pytrec_eval is None:
        print("Warning: pytrec_eval unavailable; used pure-Python Recall/MAP/nDCG fallback.")
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()