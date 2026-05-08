#!/usr/bin/env python3
"""Analyze paired general-vs-task-aware RePASs deltas for sample_300."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_ROOT = ROOT / "batch_runs/sample_300"
METRICS = [
    "entailment_score",
    "contradiction_score",
    "obligation_coverage_score",
    "composite_score",
]
DATASETS = {
    "obliqa": {
        "general": "sample300_general_obliqa_bm25_gpt52/results.csv",
        "task_aware": "sample300_task_obliqa_bm25_gpt52/results.csv",
    },
    "obliqa_mp": {
        "general": "sample300_general_obliqa_mp_bm25_gpt52/results.csv",
        "task_aware": "sample300_task_obliqa_mp_bm25_gpt52/results.csv",
    },
    "xref_adgm": {
        "general": "sample300_general_xref_adgm_bm25_gpt52/results.csv",
        "task_aware": "sample300_task_xref_adgm_bm25_gpt52/results.csv",
    },
}

SUMMARY_FIELDS = [
    "dataset",
    "metric",
    "n_paired",
    "mean_general",
    "mean_task_aware",
    "mean_delta",
    "median_delta",
    "std_delta",
    "min_delta",
    "max_delta",
    "count_task_better",
    "count_general_better",
    "count_equal",
    "percent_task_better",
    "percent_general_better",
    "count_task_lower_contradiction",
    "count_general_lower_contradiction",
    "mean_contradiction_reduction",
    "paired_t_pvalue",
    "wilcoxon_pvalue",
    "test_note",
]

PER_ITEM_FIELDS = [
    "dataset",
    "QuestionID",
    "general_entailment_score",
    "task_aware_entailment_score",
    "delta_entailment_score",
    "general_contradiction_score",
    "task_aware_contradiction_score",
    "delta_contradiction_score",
    "contradiction_reduction",
    "general_obligation_coverage_score",
    "task_aware_obligation_coverage_score",
    "delta_obligation_coverage_score",
    "general_composite_score",
    "task_aware_composite_score",
    "delta_composite_score",
    "composite_winner",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_root", type=Path, default=DEFAULT_BATCH_ROOT)
    return parser.parse_args()


def read_results(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        qid = str(row["QuestionID"])
        if qid in out:
            raise ValueError(f"Duplicate QuestionID {qid} in {path}")
        out[qid] = {metric: float(row[metric]) for metric in METRICS}
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def std_sample(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def optional_tests(deltas: list[float]) -> tuple[float | None, float | None, str]:
    try:
        from scipy import stats  # type: ignore
    except Exception:
        return None, None, "scipy unavailable"
    if not deltas:
        return None, None, "no paired deltas"
    try:
        t_res = stats.ttest_1samp(deltas, 0.0)
        t_p = float(t_res.pvalue)
    except Exception as exc:
        t_p = None
        t_note = f"paired t-test failed: {exc}"
    else:
        t_note = ""
    try:
        if all(delta == 0 for delta in deltas):
            w_p = None
            w_note = "wilcoxon skipped: all deltas zero"
        else:
            w_res = stats.wilcoxon(deltas)
            w_p = float(w_res.pvalue)
            w_note = ""
    except Exception as exc:
        w_p = None
        w_note = f"wilcoxon failed: {exc}"
    note = "; ".join(note for note in [t_note, w_note] if note) or "ok"
    return t_p, w_p, note


def winner(metric: str, general: float, task: float) -> str:
    if task == general:
        return "equal"
    if metric == "contradiction_score":
        return "task_aware" if task < general else "general"
    return "task_aware" if task > general else "general"


def summarize_metric(dataset: str, metric: str, paired_rows: list[dict[str, Any]]) -> dict[str, Any]:
    general_vals = [row[f"general_{metric}"] for row in paired_rows]
    task_vals = [row[f"task_aware_{metric}"] for row in paired_rows]
    deltas = [row[f"delta_{metric}"] for row in paired_rows]
    n = len(deltas)
    task_better = sum(1 for row in paired_rows if winner(metric, row[f"general_{metric}"], row[f"task_aware_{metric}"]) == "task_aware")
    general_better = sum(1 for row in paired_rows if winner(metric, row[f"general_{metric}"], row[f"task_aware_{metric}"]) == "general")
    equal = n - task_better - general_better
    t_p, w_p, note = optional_tests(deltas)
    row: dict[str, Any] = {
        "dataset": dataset,
        "metric": metric,
        "n_paired": n,
        "mean_general": mean(general_vals),
        "mean_task_aware": mean(task_vals),
        "mean_delta": mean(deltas),
        "median_delta": statistics.median(deltas) if deltas else math.nan,
        "std_delta": std_sample(deltas),
        "min_delta": min(deltas) if deltas else math.nan,
        "max_delta": max(deltas) if deltas else math.nan,
        "count_task_better": task_better,
        "count_general_better": general_better,
        "count_equal": equal,
        "percent_task_better": task_better / n if n else math.nan,
        "percent_general_better": general_better / n if n else math.nan,
        "count_task_lower_contradiction": "",
        "count_general_lower_contradiction": "",
        "mean_contradiction_reduction": "",
        "paired_t_pvalue": t_p,
        "wilcoxon_pvalue": w_p,
        "test_note": note,
    }
    if metric == "contradiction_score":
        row["count_task_lower_contradiction"] = sum(
            1 for row_item in paired_rows if row_item["task_aware_contradiction_score"] < row_item["general_contradiction_score"]
        )
        row["count_general_lower_contradiction"] = sum(
            1 for row_item in paired_rows if row_item["general_contradiction_score"] < row_item["task_aware_contradiction_score"]
        )
        reductions = [row_item["general_contradiction_score"] - row_item["task_aware_contradiction_score"] for row_item in paired_rows]
        row["mean_contradiction_reduction"] = mean(reductions)
    return row


def build_pair_rows(dataset: str, general: dict[str, dict[str, float]], task: dict[str, dict[str, float]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    general_ids = set(general)
    task_ids = set(task)
    paired_ids = sorted(general_ids & task_ids)
    rows: list[dict[str, Any]] = []
    for qid in paired_ids:
        row: dict[str, Any] = {"dataset": dataset, "QuestionID": qid}
        for metric in METRICS:
            g = general[qid][metric]
            t = task[qid][metric]
            row[f"general_{metric}"] = g
            row[f"task_aware_{metric}"] = t
            row[f"delta_{metric}"] = t - g
        row["contradiction_reduction"] = row["general_contradiction_score"] - row["task_aware_contradiction_score"]
        row["composite_winner"] = winner("composite_score", row["general_composite_score"], row["task_aware_composite_score"])
        rows.append(row)
    mismatch = {
        "dataset": dataset,
        "general_count": len(general_ids),
        "task_aware_count": len(task_ids),
        "paired_count": len(paired_ids),
        "missing_from_task_aware": sorted(general_ids - task_ids),
        "missing_from_general": sorted(task_ids - general_ids),
    }
    return rows, mismatch


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def interpretation(summary_rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    for row in summary_rows:
        by_dataset.setdefault(row["dataset"], {})[row["metric"]] = row
    for dataset, metrics in by_dataset.items():
        composite = metrics["composite_score"]
        obligation = metrics["obligation_coverage_score"]
        contradiction = metrics["contradiction_score"]
        comp_winner = "task-aware" if composite["mean_task_aware"] > composite["mean_general"] else "general" if composite["mean_general"] > composite["mean_task_aware"] else "tie"
        obligation_direction = "improves" if obligation["mean_delta"] > 0 else "decreases" if obligation["mean_delta"] < 0 else "does not change"
        contradiction_direction = "reduces" if contradiction["mean_contradiction_reduction"] > 0 else "increases" if contradiction["mean_contradiction_reduction"] < 0 else "does not change"
        task_pct = composite["percent_task_better"] * 100
        general_pct = composite["percent_general_better"] * 100
        consistency = "consistent" if task_pct >= 60 or general_pct >= 60 else "mixed"
        lines.append(
            f"{dataset}: {comp_winner} has the higher mean composite score; "
            f"task-aware {obligation_direction} obligation coverage "
            f"(mean delta {obligation['mean_delta']:.6f}) and {contradiction_direction} contradiction "
            f"(mean reduction {contradiction['mean_contradiction_reduction']:.6f}). "
            f"Composite wins are {consistency}: task-aware {task_pct:.1f}%, general {general_pct:.1f}%."
        )
    return lines


def write_md(path: Path, summary_rows: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    lines = [
        "# RePASs Prompt Delta Analysis",
        "",
        "## Summary",
        "",
        "| dataset | metric | n | mean_general | mean_task_aware | mean_delta | median_delta | std_delta | task_better | general_better | equal | task_better_% | general_better_% |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["metric"],
                    str(row["n_paired"]),
                    fmt(row["mean_general"]),
                    fmt(row["mean_task_aware"]),
                    fmt(row["mean_delta"]),
                    fmt(row["median_delta"]),
                    fmt(row["std_delta"]),
                    str(row["count_task_better"]),
                    str(row["count_general_better"]),
                    str(row["count_equal"]),
                    fmt(row["percent_task_better"]),
                    fmt(row["percent_general_better"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {line}" for line in interpretation(summary_rows))
    lines.extend(["", "## Pairing Checks", ""])
    for mismatch in mismatches:
        issue_count = len(mismatch["missing_from_task_aware"]) + len(mismatch["missing_from_general"])
        lines.append(
            f"- {mismatch['dataset']}: paired={mismatch['paired_count']}, "
            f"general={mismatch['general_count']}, task_aware={mismatch['task_aware_count']}, mismatches={issue_count}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repass_root = args.batch_root / "repass_outputs"
    report_root = args.batch_root / "reports"
    all_pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    for dataset, paths in DATASETS.items():
        general = read_results(repass_root / paths["general"])
        task = read_results(repass_root / paths["task_aware"])
        pair_rows, mismatch = build_pair_rows(dataset, general, task)
        mismatches.append(mismatch)
        all_pair_rows.extend(pair_rows)
        for metric in METRICS:
            summary_rows.append(summarize_metric(dataset, metric, pair_rows))

    summary_csv = report_root / "repass_prompt_delta_summary.csv"
    summary_json = report_root / "repass_prompt_delta_summary.json"
    summary_md = report_root / "repass_prompt_delta_summary.md"
    per_item_csv = report_root / "repass_prompt_delta_per_item.csv"

    write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)
    write_json(summary_json, {"summary": summary_rows, "pairing": mismatches, "interpretation": interpretation(summary_rows)})
    write_md(summary_md, summary_rows, mismatches)
    write_csv(per_item_csv, all_pair_rows, PER_ITEM_FIELDS)

    for mismatch in mismatches:
        print(
            f"{mismatch['dataset']}: paired={mismatch['paired_count']} "
            f"general={mismatch['general_count']} task_aware={mismatch['task_aware_count']}"
        )
        if mismatch["missing_from_task_aware"] or mismatch["missing_from_general"]:
            print(f"  missing_from_task_aware={len(mismatch['missing_from_task_aware'])}")
            print(f"  missing_from_general={len(mismatch['missing_from_general'])}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")
    print(f"Wrote {per_item_csv}")


if __name__ == "__main__":
    main()
