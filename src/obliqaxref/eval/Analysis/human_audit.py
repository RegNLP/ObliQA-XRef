from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


AUDIT_FIELDS = [
    "annotation_id",
    "item_id",
    "corpus",
    "method",
    "persona",
    "split",
    "ir_difficulty_label",
    "question",
    "answer",
    "source_id",
    "source_text",
    "target_id",
    "target_text",
    "reference_text",
    "reference_type",
    "citation_leakage",
    "source_vote_count",
    "target_vote_count",
    "both_vote_count",
    "judge_schema_version",
    "citation_dependent",
    "answer_validation_passed",
]

ANNOTATION_FIELDS = [
    "annotator_id",
    "question_understandable",
    "source_relevant",
    "target_relevant",
    "both_passages_necessary",
    "source_explicitly_cites_target",
    "answer_supported",
    "overall_valid",
    "annotator_comments",
]

CRITERIA = [
    "question_understandable",
    "source_relevant",
    "target_relevant",
    "both_passages_necessary",
    "source_explicitly_cites_target",
    "answer_supported",
    "overall_valid",
]

YES_VALUES = {"yes", "y", "true", "1", "pass", "valid"}
NO_VALUES = {"no", "n", "false", "0", "fail", "invalid"}


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


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix.lower() == ".csv":
            records.extend(_read_csv(path))
        else:
            records.extend(_read_jsonl(path))
    return records


def discover_final_answer_valid_inputs() -> list[Path]:
    return sorted(Path("runs").glob("curate_*/out/final_answer_valid.jsonl"))


def _infer_corpus(record: dict[str, Any], path_hint: str = "") -> str:
    corpus = str(record.get("corpus") or record.get("dataset") or "").lower()
    if corpus:
        return corpus
    lowered = path_hint.lower()
    if "adgm" in lowered:
        return "adgm"
    if "ukfin" in lowered:
        return "ukfin"
    return ""


def _value(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return ""


def _format_jsonish(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _audit_row(record: dict[str, Any], *, annotation_id: str, path_hint: str = "") -> dict[str, Any]:
    row = {
        "annotation_id": annotation_id,
        "item_id": _value(record, "item_id", "query_id", "qa_id"),
        "corpus": _infer_corpus(record, path_hint),
        "method": str(_value(record, "method", "generation_method")).upper(),
        "persona": _value(record, "persona"),
        "split": _value(record, "split"),
        "ir_difficulty_label": _value(record, "ir_difficulty_label", "difficulty"),
        "question": _value(record, "question"),
        "answer": _value(record, "answer", "gold_answer", "expected_answer"),
        "source_id": _value(record, "source_id", "source_passage_id", "source_passage_uid"),
        "source_text": _value(record, "source_text", "source_passage_text"),
        "target_id": _value(record, "target_id", "target_passage_id", "target_passage_uid"),
        "target_text": _value(record, "target_text", "target_passage_text"),
        "reference_text": _value(record, "reference_text", "ReferenceText"),
        "reference_type": _value(record, "reference_type", "ReferenceType"),
        "citation_leakage": _format_jsonish(_value(record, "citation_leakage")),
        "source_vote_count": _value(record, "source_vote_count"),
        "target_vote_count": _value(record, "target_vote_count"),
        "both_vote_count": _value(record, "both_vote_count"),
        "judge_schema_version": _value(record, "judge_schema_version"),
        "citation_dependent": _value(record, "citation_dependent"),
        "answer_validation_passed": _value(record, "answer_validation_passed"),
    }
    for field in ANNOTATION_FIELDS:
        row[field] = ""
    return row


def _stratum_key(row: dict[str, Any], *, include_persona: bool) -> tuple[str, ...]:
    values = [
        str(row.get("corpus", "")),
        str(row.get("method", "")),
        str(row.get("ir_difficulty_label", "")),
    ]
    if include_persona:
        values.append(str(row.get("persona", "")))
    return tuple(values)


def _allocate_targets(strata: dict[tuple[str, ...], list[dict[str, Any]]], n: int) -> dict[tuple[str, ...], int]:
    total = sum(len(rows) for rows in strata.values())
    if total <= 0 or n <= 0:
        return {key: 0 for key in strata}
    raw = {key: (n * len(rows) / total) for key, rows in strata.items()}
    targets = {key: int(raw[key]) for key in strata}
    if n >= len(strata):
        for key in strata:
            targets[key] = max(1, targets[key])
    while sum(targets.values()) > n:
        key = min((k for k in targets if targets[k] > 0), key=lambda k: raw[k] - targets[k])
        targets[key] -= 1
    while sum(targets.values()) < n:
        key = max(strata, key=lambda k: (raw[k] - targets[k], len(strata[k]) - targets[k]))
        targets[key] += 1
    return targets


def _write_instructions(path: Path) -> None:
    text = """# Human Audit Instructions

Annotate each row using only the provided question, answer, source passage, and target passage. Do not use outside documents or legal knowledge.

Fields:
- annotator_id: your stable annotator identifier.
- question_understandable: yes if the question is clear enough to answer from the provided passages.
- source_relevant: yes if the source passage is relevant to the question.
- target_relevant: yes if the target passage is relevant to the question.
- both_passages_necessary: yes if a complete correct answer needs both passages.
- source_explicitly_cites_target: yes if the source passage explicitly cross-references the target passage.
- answer_supported: yes if the answer is fully supported by the provided source and target passages.
- overall_valid: yes if the item is suitable for the citation-dependent QA benchmark.
- annotator_comments: brief notes for unclear, invalid, or borderline cases.

Use yes/no labels where possible. Borderline citation-dependency cases should be marked no for both_passages_necessary and overall_valid.
"""
    path.write_text(text, encoding="utf-8")


def export_human_audit_sample(
    *,
    inputs: list[str | Path] | None = None,
    out_dir: str | Path = "ObliQA-XRef_Out_Datasets",
    n: int = 200,
    seed: int = 13,
    stratify_persona: bool = False,
) -> dict[str, Path]:
    input_paths = [Path(p) for p in inputs] if inputs else discover_final_answer_valid_inputs()
    loaded: list[tuple[dict[str, Any], str]] = []
    for path in input_paths:
        for record in _load_records([path]):
            loaded.append((record, str(path)))

    rows = [
        _audit_row(record, annotation_id="", path_hint=path_hint)
        for record, path_hint in loaded
    ]
    strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[_stratum_key(row, include_persona=stratify_persona)].append(row)

    rng = random.Random(seed)
    targets = _allocate_targets(strata, n)
    sampled: list[dict[str, Any]] = []
    report_strata: list[dict[str, Any]] = []
    for key in sorted(strata):
        candidates = list(strata[key])
        rng.shuffle(candidates)
        target = targets.get(key, 0)
        take = min(target, len(candidates))
        sampled.extend(candidates[:take])
        report_strata.append({
            "stratum": list(key),
            "available": len(candidates),
            "target": target,
            "sampled": take,
            "shortfall": max(0, target - take),
        })

    sampled.sort(key=lambda row: (row.get("corpus", ""), row.get("method", ""), row.get("item_id", "")))
    for idx, row in enumerate(sampled, start=1):
        row["annotation_id"] = f"HA-{idx:06d}"

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "human_audit_sample.csv"
    report_path = output_dir / "human_audit_sampling_report.json"
    instructions_path = output_dir / "human_audit_instructions.md"

    _write_csv(sample_path, sampled, AUDIT_FIELDS + ANNOTATION_FIELDS)
    report = {
        "input_files": [str(path) for path in input_paths],
        "available_count": len(rows),
        "requested_count": n,
        "sampled_count": len(sampled),
        "seed": seed,
        "stratify_persona": stratify_persona,
        "strata": report_strata,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_instructions(instructions_path)
    return {"sample": sample_path, "report": report_path, "instructions": instructions_path}


def normalize_yes_no(value: Any) -> bool | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in YES_VALUES:
        return True
    if text in NO_VALUES:
        return False
    return None


def majority_vote(values: list[bool | None]) -> bool | None:
    valid = [value for value in values if value is not None]
    yes = sum(1 for value in valid if value is True)
    no = sum(1 for value in valid if value is False)
    if yes > no:
        return True
    if no > yes:
        return False
    return None


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    a_yes = sum(1 for a, _b in pairs if a) / n
    b_yes = sum(1 for _a, b in pairs if b) / n
    expected = a_yes * b_yes + (1 - a_yes) * (1 - b_yes)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def _pct_yes(rows: list[dict[str, Any]], criterion: str) -> float | None:
    values = [normalize_yes_no(row.get(criterion)) for row in rows]
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(1 for value in valid if value is True) / len(valid)


def _summarize_group(rows: list[dict[str, Any]], group_type: str, group_value: str) -> dict[str, Any]:
    out: dict[str, Any] = {"group_type": group_type, "group_value": group_value, "n_items": len(rows)}
    for criterion in CRITERIA:
        value = _pct_yes(rows, f"{criterion}_majority") if rows and f"{criterion}_majority" in rows[0] else _pct_yes(rows, criterion)
        out[f"{criterion}_pct_yes"] = value
    return out


def aggregate_human_audit(
    *,
    inputs: list[str | Path],
    out_dir: str | Path = "ObliQA-XRef_Out_Datasets",
    min_group_size: int = 5,
) -> dict[str, Path]:
    input_paths = [Path(path) for path in inputs]
    annotations = _load_records(input_paths)
    warnings: list[str] = []
    invalid_labels: list[dict[str, str]] = []
    if any(not row.get("annotator_id") for row in annotations):
        warnings.append("Some rows are missing annotator_id.")

    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        item_id = str(row.get("item_id") or row.get("query_id") or row.get("annotation_id") or "")
        by_item[item_id].append(row)
        for criterion in CRITERIA:
            raw = row.get(criterion)
            if raw not in (None, "") and normalize_yes_no(raw) is None:
                invalid_labels.append({"item_id": item_id, "criterion": criterion, "value": str(raw)})

    aggregated: list[dict[str, Any]] = []
    agreement_by_criterion: dict[str, float | None] = {}
    kappa_by_criterion: dict[str, float | None] = {}

    for item_id, item_rows in sorted(by_item.items()):
        base = dict(item_rows[0])
        base["item_id"] = item_id
        annotators = sorted({str(row.get("annotator_id") or "") for row in item_rows if row.get("annotator_id")})
        base["annotator_count"] = len(annotators)
        for criterion in CRITERIA:
            values = [normalize_yes_no(row.get(criterion)) for row in item_rows]
            vote = majority_vote(values)
            valid = [value for value in values if value is not None]
            yes = sum(1 for value in valid if value is True)
            no = sum(1 for value in valid if value is False)
            base[f"{criterion}_majority"] = "" if vote is None else ("yes" if vote else "no")
            base[f"{criterion}_yes_count"] = yes
            base[f"{criterion}_no_count"] = no
            base[f"{criterion}_n"] = len(valid)
            if yes and no:
                warnings.append(f"Conflicting annotations for item {item_id}, criterion {criterion}.")
        aggregated.append(base)

    for criterion in CRITERIA:
        item_agreements: list[float] = []
        pairs: list[tuple[bool, bool]] = []
        for item_rows in by_item.values():
            labels = [normalize_yes_no(row.get(criterion)) for row in item_rows]
            valid = [label for label in labels if label is not None]
            if len(valid) >= 2:
                counts = Counter(valid)
                item_agreements.append(max(counts.values()) / len(valid))
            annotator_rows = [row for row in item_rows if row.get("annotator_id")]
            annotator_rows.sort(key=lambda row: str(row.get("annotator_id")))
            if len(annotator_rows) == 2:
                a = normalize_yes_no(annotator_rows[0].get(criterion))
                b = normalize_yes_no(annotator_rows[1].get(criterion))
                if a is not None and b is not None:
                    pairs.append((a, b))
        agreement_by_criterion[criterion] = (
            sum(item_agreements) / len(item_agreements) if item_agreements else None
        )
        kappa_by_criterion[criterion] = cohen_kappa(pairs)

    summary_rows: list[dict[str, Any]] = [_summarize_group(aggregated, "overall", "all")]
    for field in ("corpus", "method", "ir_difficulty_label"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in aggregated:
            grouped[str(row.get(field) or "")].append(row)
        for value, rows in sorted(grouped.items()):
            if len(rows) < min_group_size:
                warnings.append(f"Very small subgroup: {field}={value} has {len(rows)} items.")
            summary_rows.append(_summarize_group(rows, field, value))

    grouped_cm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregated:
        grouped_cm[f"{row.get('corpus', '')} x {row.get('method', '')}"].append(row)
    for value, rows in sorted(grouped_cm.items()):
        if len(rows) < min_group_size:
            warnings.append(f"Very small subgroup: corpus_method={value} has {len(rows)} items.")
        summary_rows.append(_summarize_group(rows, "corpus_method", value))

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_path = output_dir / "human_audit_aggregated_items.csv"
    summary_path = output_dir / "human_audit_summary.csv"
    md_path = output_dir / "human_audit_summary.md"
    agreement_path = output_dir / "human_audit_agreement.json"

    agg_fields = list(dict.fromkeys(list(aggregated[0].keys()) if aggregated else AUDIT_FIELDS + ANNOTATION_FIELDS))
    _write_csv(agg_path, aggregated, agg_fields)
    summary_fields = ["group_type", "group_value", "n_items"] + [f"{criterion}_pct_yes" for criterion in CRITERIA]
    _write_csv(summary_path, summary_rows, summary_fields)

    agreement = {
        "annotated_items": len(aggregated),
        "annotator_count": len({str(row.get("annotator_id")) for row in annotations if row.get("annotator_id")}),
        "percent_agreement": agreement_by_criterion,
        "cohen_kappa": kappa_by_criterion,
        "invalid_labels": invalid_labels,
        "warnings": sorted(set(warnings)),
    }
    agreement_path.write_text(json.dumps(agreement, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = summary_rows[0] if summary_rows else {}
    lines = [
        "# Human Audit Summary",
        "",
        f"Annotated items: {len(aggregated)}",
        f"Annotators: {agreement['annotator_count']}",
        "",
        "## Validity Percentages",
        "",
        "| criterion | percent yes | percent agreement | Cohen's kappa |",
        "| --- | ---: | ---: | ---: |",
    ]
    for criterion in CRITERIA:
        pct = overall.get(f"{criterion}_pct_yes")
        agree = agreement_by_criterion.get(criterion)
        kappa = kappa_by_criterion.get(criterion)
        lines.append(
            f"| {criterion} | {_fmt_pct(pct)} | {_fmt_pct(agree)} | {_fmt_num(kappa)} |"
        )
    lines.extend([
        "",
        "Interpretation: percentages reflect majority-voted item labels. Agreement statistics are computed over items with at least two valid annotations.",
    ])
    if warnings or invalid_labels:
        lines.extend(["", "## Warnings", ""])
        for warning in sorted(set(warnings)):
            lines.append(f"- {warning}")
        if invalid_labels:
            lines.append(f"- Invalid labels found: {len(invalid_labels)}.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "aggregated_items": agg_path,
        "summary": summary_path,
        "markdown": md_path,
        "agreement": agreement_path,
    }


def _fmt_pct(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value) * 100:.1f}%"


def _fmt_num(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.3f}"
