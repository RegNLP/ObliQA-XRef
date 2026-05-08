#!/usr/bin/env python3
"""Run RePASs sequentially for the controlled batch inputs and summarize."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
JOBS = [
    ("obliqa", "general", "general/obliqa_bm25_gpt52.json", "general_obliqa_bm25_gpt52"),
    ("obliqa_mp", "general", "general/obliqa_mp_bm25_gpt52.json", "general_obliqa_mp_bm25_gpt52"),
    ("xref_adgm", "general", "general/xref_adgm_bm25_gpt52.json", "general_xref_adgm_bm25_gpt52"),
    ("obliqa", "task_aware", "task_aware/obliqa_bm25_gpt52.json", "task_obliqa_bm25_gpt52"),
    ("obliqa_mp", "task_aware", "task_aware/obliqa_mp_bm25_gpt52.json", "task_obliqa_mp_bm25_gpt52"),
    ("xref_adgm", "task_aware", "task_aware/xref_adgm_bm25_gpt52.json", "task_xref_adgm_bm25_gpt52"),
]
SUMMARY_FIELDS = [
    "dataset",
    "prompt_condition",
    "sample_size",
    "mean_entailment_score",
    "mean_contradiction_score",
    "mean_obligation_coverage_score",
    "mean_composite_score",
    "runtime_seconds",
    "skipped_or_failed_rows",
    "output_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repass_repo", type=Path, required=True)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--summary_md", type=Path, required=True)
    parser.add_argument("--python_bin", type=Path, default=Path("/opt/anaconda3/bin/python"))
    parser.add_argument("--sample_size", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(rows: list[dict[str, str]], field: str) -> float | None:
    vals = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return sum(vals) / len(vals) if vals else None


def write_summary(csv_path: Path, json_path: Path, md_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    lines = [
        "| " + " | ".join(SUMMARY_FIELDS) + " |",
        "| " + " | ".join(["---"] * len(SUMMARY_FIELDS)) + " |",
    ]
    for row in rows:
        vals = []
        for field in SUMMARY_FIELDS:
            val = row.get(field)
            vals.append(f"{val:.6f}" if isinstance(val, float) else "" if val is None else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_group(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if overwrite:
            shutil.rmtree(dst)
        else:
            return
    shutil.copytree(src, dst)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    job_log_root = args.output_root / "logs"
    job_log_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for dataset, condition, rel_input, suffix in JOBS:
        group = f"sample{args.sample_size}_{suffix}"
        input_path = args.input_root / rel_input
        output_path = args.output_root / group
        repass_data_path = args.repass_repo / "data" / group
        log_path = job_log_root / f"{group}.log"
        input_rows = read_json(input_path)
        expected = len(input_rows) if isinstance(input_rows, list) else args.sample_size

        print(f"Starting RePASs job {group}", flush=True)
        start = time.perf_counter()
        if output_path.exists() and not args.overwrite and not args.resume:
            print(f"Skipping existing output {output_path}", flush=True)
            runtime = 0.0
        else:
            if args.overwrite and repass_data_path.exists():
                shutil.rmtree(repass_data_path)
            cmd = [
                str(args.python_bin),
                "scripts/evaluate_model.py",
                "--input_file",
                str(input_path),
                "--group_method_name",
                group,
            ]
            with log_path.open("w", encoding="utf-8") as log:
                log.write(" ".join(cmd) + "\n")
                log.flush()
                proc = subprocess.run(
                    cmd,
                    cwd=args.repass_repo,
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            runtime = time.perf_counter() - start
            if proc.returncode != 0:
                failures.append(f"{group}: RePASs exited {proc.returncode}; see {log_path}")
                print(f"Failed RePASs job {group}; see {log_path}", flush=True)
            elif repass_data_path.exists():
                copy_group(repass_data_path, output_path, overwrite=True)
                print(f"Completed RePASs job {group} in {runtime:.1f}s", flush=True)
            else:
                failures.append(f"{group}: missing RePASs output directory {repass_data_path}")

        results_csv = output_path / "results.csv"
        if not results_csv.exists() and repass_data_path.exists():
            copy_group(repass_data_path, output_path, overwrite=True)
        result_rows = read_results(output_path / "results.csv")
        summary_rows.append(
            {
                "dataset": dataset,
                "prompt_condition": condition,
                "sample_size": len(result_rows),
                "mean_entailment_score": mean(result_rows, "entailment_score"),
                "mean_contradiction_score": mean(result_rows, "contradiction_score"),
                "mean_obligation_coverage_score": mean(result_rows, "obligation_coverage_score"),
                "mean_composite_score": mean(result_rows, "composite_score"),
                "runtime_seconds": runtime,
                "skipped_or_failed_rows": expected - len(result_rows),
                "output_path": str(output_path),
            }
        )

    write_summary(args.summary_csv, args.summary_json, args.summary_md, summary_rows)
    print(f"Wrote {args.summary_csv}", flush=True)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
