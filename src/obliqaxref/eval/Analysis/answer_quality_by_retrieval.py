from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METRIC_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rouge_l", ("rouge_l", "rougeL_f1", "rouge_l_f1")),
    ("token_overlap", ("token_overlap", "passage_overlap_frac", "f1")),
    ("relevance", ("relevance", "gpt_relevance")),
    ("faithfulness", ("faithfulness", "gpt_faithfulness")),
    ("entailment", ("entailment", "nli_entailment")),
    ("contradiction", ("contradiction", "nli_contradiction")),
    ("utility", ("utility",)),
    ("tag_plus", ("tag_plus", "Tag+", "has_id_tag")),
    ("evidence_tag_count", ("evidence_tag_count", "n_id_tags")),
    ("answer_length", ("answer_length", "len_words")),
)

GROUP_KEYS = ("retrieval_outcome", "corpus", "method", "retriever", "generator_model", "split")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _infer_answer_eval_files(root: Path, corpus: str | None = None) -> list[Path]:
    pattern = f"answer_eval_{corpus}_*_test.json" if corpus else "answer_eval_*_*_test.json"
    return sorted(root.glob(pattern))


def _parse_answer_eval_filename(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    prefix = "answer_eval_"
    suffix = "_test"
    core = stem[len(prefix) : -len(suffix)] if stem.startswith(prefix) and stem.endswith(suffix) else stem
    parts = core.split("_")
    corpus = parts[0] if parts else ""
    subset = parts[1] if len(parts) > 1 else ""
    method = "_".join(parts[2:]) if len(parts) > 2 else ""
    return corpus, subset.upper(), method


def load_answer_eval_rows(root: Path, corpus: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _infer_answer_eval_files(root, corpus):
        if path.name.endswith("_compact.csv"):
            continue
        file_corpus, subset, retriever = _parse_answer_eval_filename(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        results = data.get("results", []) if isinstance(data, dict) else data
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            row = dict(result)
            row.setdefault("corpus", file_corpus)
            row.setdefault("method", subset)
            row.setdefault("retriever", retriever)
            row.setdefault("split", "test")
            row.setdefault("generator_model", data.get("model", "") if isinstance(data, dict) else "")
            scores = row.get("nli_scores")
            if isinstance(scores, dict):
                row.setdefault("entailment", scores.get("entailment"))
                row.setdefault("contradiction", scores.get("contradiction"))
            rows.append(row)
    return rows


def _normalize_outcome(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "both": "both",
        "src_only": "source_only",
        "source_only": "source_only",
        "tgt_only": "target_only",
        "target_only": "target_only",
        "neither": "neither",
    }
    return aliases.get(text, text or "unknown")


def _diag_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("item_id") or row.get("query_id") or ""), str(row.get("retriever") or row.get("method") or ""))


def join_answer_quality_with_retrieval(
    retrieval_diagnostics: list[dict[str, Any]],
    answer_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics_by_key: dict[tuple[str, str], dict[str, Any]] = {
        _diag_key(row): row for row in retrieval_diagnostics
    }
    diagnostics_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in retrieval_diagnostics:
        diagnostics_by_item[str(row.get("item_id") or row.get("query_id") or "")].append(row)

    joined: list[dict[str, Any]] = []
    for ans in answer_rows:
        item_id = str(ans.get("item_id") or ans.get("query_id") or "")
        retriever = str(ans.get("retriever") or ans.get("method") or "")
        diag = diagnostics_by_key.get((item_id, retriever))
        if diag is None and item_id:
            candidates = diagnostics_by_item.get(item_id, [])
            diag = candidates[0] if len(candidates) == 1 else None
        if diag is None:
            continue

        row: dict[str, Any] = {
            "item_id": item_id,
            "query_id": diag.get("query_id", item_id),
            "corpus": ans.get("corpus") or diag.get("corpus") or diag.get("dataset") or "",
            "method": ans.get("method") or diag.get("method") or "",
            "retriever": retriever or diag.get("retriever") or "",
            "generator_model": ans.get("generator_model", ""),
            "split": ans.get("split") or diag.get("split") or "",
            "source_id": diag.get("source_id", ""),
            "target_id": diag.get("target_id", ""),
            "retrieval_outcome_at_5": _normalize_outcome(diag.get("retrieval_outcome_at_5")),
            "retrieval_outcome_at_10": _normalize_outcome(diag.get("retrieval_outcome_at_10")),
            "retrieval_outcome_at_20": _normalize_outcome(diag.get("retrieval_outcome_at_20")),
        }
        row["retrieval_outcome"] = row["retrieval_outcome_at_10"]

        for out_name, aliases in METRIC_SPECS:
            row[out_name] = _pick(ans, aliases)
        joined.append(row)
    return joined


def _metric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _as_float(row.get(metric))
        if value is not None and math.isfinite(value):
            values.append(value)
    return values


def summarize_by_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in GROUP_KEYS)].append(row)

    contrast_groups: dict[tuple[Any, ...], dict[str, float | None]] = {}
    for key, group_rows in groups.items():
        contrast_key = key[1:]
        contrast_groups.setdefault(contrast_key, {})

    broader: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        broader[tuple(row.get(key, "") for key in GROUP_KEYS[1:])].append(row)

    for contrast_key, group_rows in broader.items():
        both_rows = [row for row in group_rows if row.get("retrieval_outcome") == "both"]
        non_both_rows = [row for row in group_rows if row.get("retrieval_outcome") != "both"]
        for metric, _aliases in METRIC_SPECS:
            both_values = _metric_values(both_rows, metric)
            non_both_values = _metric_values(non_both_rows, metric)
            diff = None
            if both_values and non_both_values:
                diff = statistics.fmean(both_values) - statistics.fmean(non_both_values)
            contrast_groups.setdefault(contrast_key, {})[f"both_minus_non_both_{metric}"] = diff

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(groups):
        group_rows = groups[key]
        row = {field: value for field, value in zip(GROUP_KEYS, key, strict=False)}
        row["count"] = len(group_rows)
        for metric, _aliases in METRIC_SPECS:
            values = _metric_values(group_rows, metric)
            row[f"{metric}_count"] = len(values)
            row[f"{metric}_mean"] = statistics.fmean(values) if values else None
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
            row[f"{metric}_median"] = statistics.median(values) if values else None
        row.update(contrast_groups.get(key[1:], {}))
        summary_rows.append(row)
    return summary_rows


def write_markdown_summary(path: Path, summary_rows: list[dict[str, Any]], *, min_group_size: int = 5) -> None:
    warnings = [row for row in summary_rows if int(row.get("count") or 0) < min_group_size]
    lines = [
        "# Answer Quality by Retrieval Outcome",
        "",
        f"Grouped rows: {len(summary_rows)}",
        "",
        "## Outcome Counts",
        "",
        "| retrieval_outcome | count | rouge_l_mean | faithfulness_mean | answer_length_mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    by_outcome: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_outcome[str(row.get("retrieval_outcome", ""))].append(row)
    for outcome in sorted(by_outcome):
        rows = by_outcome[outcome]
        count = sum(int(row.get("count") or 0) for row in rows)
        flat = []
        for row in rows:
            flat.extend([row] * int(row.get("count") or 0))
        def weighted(metric: str) -> str:
            num = sum((float(row.get(f"{metric}_mean") or 0.0) * int(row.get("count") or 0)) for row in rows if row.get(f"{metric}_mean") not in (None, ""))
            den = sum(int(row.get("count") or 0) for row in rows if row.get(f"{metric}_mean") not in (None, ""))
            return f"{num / den:.4f}" if den else ""
        lines.append(f"| {outcome} | {count} | {weighted('rouge_l')} | {weighted('faithfulness')} | {weighted('answer_length')} |")

    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.append(f"- {len(warnings)} groups have fewer than {min_group_size} items; subgroup estimates may be unstable.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def answer_quality_by_retrieval_outcome(
    *,
    root_dir: str | Path = "ObliQA-XRef_Out_Datasets",
    diagnostics_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    corpus: str | None = None,
    min_group_size: int = 5,
) -> dict[str, Path]:
    root = Path(root_dir)
    diagnostics_file = Path(diagnostics_path) if diagnostics_path else root / "retrieval_diagnostics_per_query.csv"
    output_dir = Path(out_dir) if out_dir else root

    diagnostics = _read_csv(diagnostics_file)
    answer_rows = load_answer_eval_rows(root, corpus=corpus)
    joined = join_answer_quality_with_retrieval(diagnostics, answer_rows)
    summary = summarize_by_group(joined)

    detail_path = output_dir / "answer_quality_by_retrieval_outcome.csv"
    summary_path = output_dir / "answer_quality_by_retrieval_outcome_summary.csv"
    md_path = output_dir / "answer_quality_by_retrieval_outcome_summary.md"

    detail_fields = [
        "item_id",
        "query_id",
        "corpus",
        "method",
        "retriever",
        "generator_model",
        "split",
        "source_id",
        "target_id",
        "retrieval_outcome",
        "retrieval_outcome_at_5",
        "retrieval_outcome_at_10",
        "retrieval_outcome_at_20",
    ] + [name for name, _aliases in METRIC_SPECS]
    summary_fields = list(GROUP_KEYS) + ["count"]
    for metric, _aliases in METRIC_SPECS:
        summary_fields.extend([
            f"{metric}_count",
            f"{metric}_mean",
            f"{metric}_std",
            f"{metric}_median",
            f"both_minus_non_both_{metric}",
        ])

    _write_csv(detail_path, joined, detail_fields)
    _write_csv(summary_path, summary, summary_fields)
    write_markdown_summary(md_path, summary, min_group_size=min_group_size)
    return {"detail": detail_path, "summary": summary_path, "markdown": md_path}
