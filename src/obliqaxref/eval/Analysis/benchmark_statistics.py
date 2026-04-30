from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from obliqaxref.benchmark_metadata import (
    OBLIQA_XREF_METADATA_FIELDS,
    with_obliqa_xref_metadata,
)


COUNT_FIELDS = [
    "generated_count",
    "structurally_valid_count",
    "citation_leakage_count",
    "citation_dependency_passed_count",
    "answer_validation_passed_count",
    "answer_validation_failed_count",
    "final_dependency_valid_count",
    "final_answer_valid_count",
    "final_answer_failed_count",
]

GROUP_FIELDS = [
    "corpus",
    "method",
    "persona",
    "split",
    "ir_difficulty_label",
    "reference_type",
    "citation_leakage",
    "answer_validation_passed",
    "judge_schema_version",
]

TEXT_FIELDS = {
    "question": ("question",),
    "answer": ("gold_answer", "answer", "expected_answer"),
    "source": ("source_text", "source_passage_text"),
    "target": ("target_text", "target_passage_text"),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def discover_final_cohort_inputs() -> list[Path]:
    paths: list[Path] = []
    for name in (
        "final_dependency_valid.jsonl",
        "final_answer_valid.jsonl",
        "final_answer_failed.jsonl",
    ):
        paths.extend(sorted(Path("runs").glob(f"curate_*/out/{name}")))
    return sorted(paths)


def _infer_corpus(row: dict[str, Any], path: Path) -> str:
    corpus = str(row.get("corpus") or row.get("dataset") or "").lower()
    if corpus:
        return corpus
    text = str(path).lower()
    if "adgm" in text:
        return "adgm"
    if "ukfin" in text:
        return "ukfin"
    return ""


def _infer_cohort(path: Path) -> str:
    stem = path.stem
    if "final_dependency_valid" in stem:
        return "dependency_valid"
    if "final_answer_valid" in stem:
        return "answer_valid"
    if "final_answer_failed" in stem:
        return "answer_failed"
    return "input"


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        raw = _read_csv(path) if path.suffix.lower() == ".csv" else _read_jsonl(path)
        cohort = _infer_cohort(path)
        for row in raw:
            rec = with_obliqa_xref_metadata(row)
            rec.setdefault("corpus", _infer_corpus(rec, path))
            rec.setdefault("cohort", cohort)
            rec["_input_file"] = str(path)
            records.append(rec)
    return records


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"true", "1", "yes", "y", "pass", "valid"}:
        return True
    if text in {"false", "0", "no", "n", "fail", "invalid"}:
        return False
    return None


def _bool_key(value: Any) -> str:
    parsed = _parse_bool(value)
    if parsed is True:
        return "true"
    if parsed is False:
        return "false"
    return "missing"


def _word_count(text: Any) -> int | None:
    if text is None:
        return None
    text_s = str(text).strip()
    if not text_s:
        return None
    words = re.findall(r"\b\w+\b", text_s)
    return len(words) if words else None


def _pick(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if row.get(name) not in (None, ""):
            return row.get(name)
    return None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def _median_iqr(values: list[int | float | None]) -> dict[str, float | None]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return {"median": None, "iqr": None}
    q1 = _quantile(nums, 0.25)
    q3 = _quantile(nums, 0.75)
    return {
        "median": statistics.median(nums),
        "iqr": (q3 - q1) if q1 is not None and q3 is not None else None,
    }


def _length_features(row: dict[str, Any]) -> dict[str, int | None]:
    question = _word_count(_pick(row, TEXT_FIELDS["question"]))
    answer = _word_count(_pick(row, TEXT_FIELDS["answer"]))
    source = _word_count(_pick(row, TEXT_FIELDS["source"]))
    target = _word_count(_pick(row, TEXT_FIELDS["target"]))
    combined = None
    if source is not None and target is not None:
        combined = int(source) + int(target)
    return {
        "question_len": question,
        "answer_len": answer,
        "source_len": source,
        "target_len": target,
        "combined_source_target_len": combined,
    }


def _tokens(text: Any) -> set[str]:
    return {tok.lower() for tok in re.findall(r"\b\w+\b", str(text or ""))}


def _source_target_overlap(row: dict[str, Any]) -> float | None:
    for key in ("source_target_lexical_overlap", "source_target_overlap"):
        if row.get(key) not in (None, ""):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return None
    src = _tokens(_pick(row, TEXT_FIELDS["source"]))
    tgt = _tokens(_pick(row, TEXT_FIELDS["target"]))
    if not src or not tgt:
        return None
    return len(src & tgt) / len(src | tgt)


def _metadata_columns(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in OBLIQA_XREF_METADATA_FIELDS}


def _group_key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    values = []
    for field in fields:
        value = row.get(field, "")
        if field in {"citation_leakage", "answer_validation_passed"}:
            value = _bool_key(value)
        values.append(str(value or "missing"))
    return tuple(values)


def _length_row(rows: list[dict[str, Any]], *, group_type: str, group_value: str) -> dict[str, Any]:
    features = [_length_features(row) for row in rows]
    out: dict[str, Any] = {"group_type": group_type, "group_value": group_value, "count": len(rows)}
    for name in (
        "question_len",
        "answer_len",
        "source_len",
        "target_len",
        "combined_source_target_len",
    ):
        stats = _median_iqr([feature[name] for feature in features])
        out[f"{name}_median"] = stats["median"]
        out[f"{name}_iqr"] = stats["iqr"]
    return out


def build_statistics_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    count_summary = build_cohort_count_summary(records)
    for field in COUNT_FIELDS:
        rows.append({
            **(_metadata_columns(records[0]) if records else {}),
            "stat_type": "construction_count",
            "group_type": "construction",
            "group_value": field,
            "cohort": "all",
            "count": count_summary.get(field),
        })

    if records:
        rows.append({
            **_metadata_columns(records[0]),
            "stat_type": "cohort_count",
            "group_type": "overall",
            "group_value": "all",
            "cohort": "all",
            "count": len(records),
        })

    by_cohort = Counter(str(row.get("cohort") or "input") for row in records)
    for cohort, count in sorted(by_cohort.items()):
        rows.append({
            **(_metadata_columns(records[0]) if records else {}),
            "stat_type": "cohort_count",
            "group_type": "cohort",
            "group_value": cohort,
            "cohort": cohort,
            "count": count,
        })

    for field in GROUP_FIELDS:
        counts: Counter[tuple[str, str]] = Counter()
        for row in records:
            value = _group_key(row, (field,))[0]
            cohort = str(row.get("cohort") or "input")
            counts[(cohort, value)] += 1
        for (cohort, value), count in sorted(counts.items()):
            rows.append({
                **(_metadata_columns(records[0]) if records else {}),
                "stat_type": "group_count",
                "group_type": field,
                "group_value": value,
                "cohort": cohort,
                "count": count,
            })

    length_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        cohort = str(row.get("cohort") or "input")
        length_groups[(cohort, "overall", "all")].append(row)
        length_groups[(cohort, "corpus", str(row.get("corpus") or "missing"))].append(row)
        corpus_method = f"{row.get('corpus') or 'missing'} / {row.get('method') or 'missing'}"
        length_groups[(cohort, "corpus_method", corpus_method)].append(row)

    for (cohort, group_type, group_value), group_rows in sorted(length_groups.items()):
        rows.append({
            **(_metadata_columns(group_rows[0]) if group_rows else {}),
            "stat_type": "length",
            "cohort": cohort,
            **_length_row(group_rows, group_type=group_type, group_value=group_value),
        })
    return rows


def build_difficulty_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_cohort[str(row.get("cohort") or "input")].append(row)

    for cohort, cohort_rows in sorted(by_cohort.items()):
        total = len(cohort_rows)
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in cohort_rows:
            by_label[str(row.get("ir_difficulty_label") or "missing")].append(row)
        for label, label_rows in sorted(by_label.items()):
            lengths = [_length_features(row) for row in label_rows]
            overlaps = [_source_target_overlap(row) for row in label_rows]
            valid_overlaps = [value for value in overlaps if value is not None]
            answer_valid = [_parse_bool(row.get("answer_validation_passed")) for row in label_rows]
            answer_valid_known = [value for value in answer_valid if value is not None]
            rows.append({
                **_metadata_columns(label_rows[0]),
                "cohort": cohort,
                "ir_difficulty_label": label,
                "count": len(label_rows),
                "percentage": len(label_rows) / total if total else 0.0,
                "question_len_median": _median_iqr([x["question_len"] for x in lengths])["median"],
                "answer_len_median": _median_iqr([x["answer_len"] for x in lengths])["median"],
                "source_target_lexical_overlap_median": (
                    statistics.median(valid_overlaps) if valid_overlaps else None
                ),
                "answer_validation_pass_rate": (
                    sum(1 for value in answer_valid_known if value is True) / len(answer_valid_known)
                    if answer_valid_known else None
                ),
            })
    return rows


def build_cohort_count_summary(records: list[dict[str, Any]]) -> dict[str, int | None]:
    summary: dict[str, int | None] = {field: None for field in COUNT_FIELDS}
    summary["final_dependency_valid_count"] = sum(1 for r in records if r.get("cohort") == "dependency_valid")
    summary["final_answer_valid_count"] = sum(1 for r in records if r.get("cohort") == "answer_valid")
    summary["final_answer_failed_count"] = sum(1 for r in records if r.get("cohort") == "answer_failed")
    summary["citation_dependency_passed_count"] = (
        int(summary["final_dependency_valid_count"] or 0)
        or int(summary["final_answer_valid_count"] or 0) + int(summary["final_answer_failed_count"] or 0)
    )
    summary["answer_validation_passed_count"] = summary["final_answer_valid_count"]
    summary["answer_validation_failed_count"] = summary["final_answer_failed_count"]
    summary["citation_leakage_count"] = sum(1 for r in records if _parse_bool(r.get("citation_leakage")) is True)
    generated_values = []
    structural_values = []
    for row in records:
        for key in ("generated_count",):
            if row.get(key) not in (None, ""):
                try:
                    generated_values.append(int(row[key]))
                except (TypeError, ValueError):
                    pass
        for key in ("structurally_valid_count",):
            if row.get(key) not in (None, ""):
                try:
                    structural_values.append(int(row[key]))
                except (TypeError, ValueError):
                    pass
    summary["generated_count"] = max(generated_values) if generated_values else None
    summary["structurally_valid_count"] = max(structural_values) if structural_values else None
    return summary


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    text = str(value)
    return text.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def write_latex_tables(path: Path, stats_rows: list[dict[str, Any]], difficulty_rows: list[dict[str, Any]]) -> None:
    corpus_method_rows = [
        row for row in stats_rows
        if row.get("stat_type") == "length" and row.get("group_type") == "corpus_method"
    ]
    lines = [
        "% Auto-generated ObliQA-XRef benchmark statistics tables.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Cohort & Corpus/Method & N & Q Len & Ans Len & Evidence Len \\\\",
        "\\midrule",
    ]
    for row in corpus_method_rows:
        lines.append(
            f"{_fmt(row.get('cohort'))} & {_fmt(row.get('group_value'))} & "
            f"{_fmt(row.get('count'))} & {_fmt(row.get('question_len_median'))} & "
            f"{_fmt(row.get('answer_len_median'))} & "
            f"{_fmt(row.get('combined_source_target_len_median'))} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{ObliQA-XRef benchmark characteristics by corpus and generation method.}",
        "\\end{table}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Cohort & Difficulty & N & \\% & Answer Pass \\\\",
        "\\midrule",
    ])
    for row in difficulty_rows:
        lines.append(
            f"{_fmt(row.get('cohort'))} & {_fmt(row.get('ir_difficulty_label'))} & "
            f"{_fmt(row.get('count'))} & {_fmt((row.get('percentage') or 0) * 100)} & "
            f"{_fmt(row.get('answer_validation_pass_rate'))} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Difficulty distribution for ObliQA-XRef cohorts.}",
        "\\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_summary(
    path: Path,
    records: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
    difficulty_rows: list[dict[str, Any]],
    *,
    min_group_size: int = 5,
) -> None:
    counts = build_cohort_count_summary(records)
    warnings: list[str] = []
    for field in ("corpus", "method", "persona", "split", "reference_type", "ir_difficulty_label"):
        missing = sum(1 for row in records if not row.get(field))
        if missing:
            warnings.append(f"{missing} records are missing {field}.")
    small = [
        row for row in stats_rows
        if row.get("stat_type") == "group_count" and int(row.get("count") or 0) < min_group_size
    ]
    if small:
        warnings.append(f"{len(small)} count groups have fewer than {min_group_size} records.")

    lines = [
        "# ObliQA-XRef Benchmark Statistics",
        "",
        "## Cohorts",
        "",
        f"- Total dependency-valid items: {counts.get('final_dependency_valid_count') or 0}",
        f"- Total answer-valid items: {counts.get('final_answer_valid_count') or 0}",
        f"- Total answer-failed items: {counts.get('final_answer_failed_count') or 0}",
        "",
        "## Counts by Corpus/Method",
        "",
        "| cohort | corpus / method | count | question median | answer median | evidence median |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in stats_rows:
        if row.get("stat_type") == "length" and row.get("group_type") == "corpus_method":
            lines.append(
                f"| {row.get('cohort')} | {row.get('group_value')} | {row.get('count')} | "
                f"{_fmt(row.get('question_len_median'))} | {_fmt(row.get('answer_len_median'))} | "
                f"{_fmt(row.get('combined_source_target_len_median'))} |"
            )
    lines.extend([
        "",
        "## Difficulty Distribution",
        "",
        "| cohort | difficulty | count | percentage | answer pass rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for row in difficulty_rows:
        lines.append(
            f"| {row.get('cohort')} | {row.get('ir_difficulty_label')} | {row.get('count')} | "
            f"{_fmt((row.get('percentage') or 0) * 100)} | {_fmt(row.get('answer_validation_pass_rate'))} |"
        )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_benchmark_statistics(
    *,
    inputs: list[str | Path] | None = None,
    out_dir: str | Path = "ObliQA-XRef_Out_Datasets",
    min_group_size: int = 5,
) -> dict[str, Path]:
    input_paths = [Path(path) for path in inputs] if inputs else discover_final_cohort_inputs()
    records = _load_records(input_paths)
    stats_rows = build_statistics_rows(records)
    difficulty_rows = build_difficulty_rows(records)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "benchmark_statistics.csv"
    difficulty_path = output_dir / "benchmark_statistics_by_difficulty.csv"
    md_path = output_dir / "benchmark_statistics_summary.md"
    tex_path = output_dir / "benchmark_statistics_latex_tables.tex"

    stat_fields = list(OBLIQA_XREF_METADATA_FIELDS) + [
        "stat_type",
        "group_type",
        "group_value",
        "cohort",
        "count",
        "question_len_median",
        "question_len_iqr",
        "answer_len_median",
        "answer_len_iqr",
        "source_len_median",
        "source_len_iqr",
        "target_len_median",
        "target_len_iqr",
        "combined_source_target_len_median",
        "combined_source_target_len_iqr",
    ]
    difficulty_fields = list(OBLIQA_XREF_METADATA_FIELDS) + [
        "cohort",
        "ir_difficulty_label",
        "count",
        "percentage",
        "question_len_median",
        "answer_len_median",
        "source_target_lexical_overlap_median",
        "answer_validation_pass_rate",
    ]
    _write_csv(stats_path, stats_rows, stat_fields)
    _write_csv(difficulty_path, difficulty_rows, difficulty_fields)
    write_markdown_summary(
        md_path,
        records,
        stats_rows,
        difficulty_rows,
        min_group_size=min_group_size,
    )
    write_latex_tables(tex_path, stats_rows, difficulty_rows)
    return {
        "statistics": stats_path,
        "difficulty": difficulty_path,
        "markdown": md_path,
        "latex": tex_path,
    }
