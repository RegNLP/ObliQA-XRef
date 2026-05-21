#!/usr/bin/env python3
"""Compute agreement tables for human/LLM annotation CSVs.

Drop annotator result CSVs into ``human_annotation_agreement/judger_results`` and
rerun this script. Each CSV must contain ``qa_id`` and the annotation columns
listed in ``ANNOTATION_FIELDS``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path


ANNOTATION_FIELDS = [
    ("question_validity_clarity", "Q1_question_understandability"),
    ("question_fluency_naturalness", "Q2_question_fluency_reuse"),
    ("source_target_evidence_dependency", "Q3_evidence_dependency"),
]


def annotator_name(path: Path) -> str:
    name = path.stem
    if name.endswith(".partial"):
        name = name[: -len(".partial")] + "_partial"
    return name


def read_annotations(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row.get("qa_id", ""): row for row in csv.DictReader(f) if row.get("qa_id")}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percent(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b)) / n
    cats = sorted(set(a) | set(b))
    dist_a = Counter(a)
    dist_b = Counter(b)
    expected = sum((dist_a[c] / n) * (dist_b[c] / n) for c in cats)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return round((observed - expected) / (1.0 - expected), 4)


def fleiss_kappa(votes: list[list[str]]) -> float:
    """Nominal Fleiss' kappa for complete item-by-annotator vote rows."""
    if not votes:
        return 0.0
    n_raters = len(votes[0])
    if n_raters < 2:
        return 0.0
    categories = sorted({vote for row in votes for vote in row})
    item_agreements = []
    category_totals = Counter()
    for row in votes:
        counts = Counter(row)
        category_totals.update(counts)
        item_agreements.append(
            (sum(count * count for count in counts.values()) - n_raters)
            / (n_raters * (n_raters - 1))
        )
    p_bar = sum(item_agreements) / len(item_agreements)
    total_votes = len(votes) * n_raters
    p_e = sum((category_totals[c] / total_votes) ** 2 for c in categories)
    if p_e == 1.0:
        return 1.0 if p_bar == 1.0 else 0.0
    return round((p_bar - p_e) / (1.0 - p_e), 4)


def krippendorff_alpha_nominal(votes: list[list[str]]) -> float:
    """Nominal Krippendorff's alpha for item-by-annotator vote rows.

    Empty labels are treated as missing. This matches the paper setting while
    remaining valid if a future annotator leaves an item blank.
    """
    observed_disagreements = 0
    observed_pairs = 0
    all_votes: list[str] = []
    for row in votes:
        labels = [vote for vote in row if vote]
        all_votes.extend(labels)
        for a, b in combinations(labels, 2):
            observed_pairs += 1
            if a != b:
                observed_disagreements += 1

    if observed_pairs == 0:
        return 0.0

    do = observed_disagreements / observed_pairs
    counts = Counter(all_votes)
    n_votes = sum(counts.values())
    if n_votes <= 1:
        return 1.0

    expected_agreement = sum(count * (count - 1) for count in counts.values()) / (
        n_votes * (n_votes - 1)
    )
    de = 1.0 - expected_agreement
    if de == 0.0:
        return 1.0 if do == 0.0 else 0.0
    return round(1.0 - (do / de), 4)


def complete_ids(data: dict[str, dict[str, dict[str, str]]], field: str) -> list[str]:
    common = set.intersection(*(set(rows) for rows in data.values()))
    return sorted(
        qid
        for qid in common
        if all(data[ann][qid].get(field, "") for ann in data)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("human_annotation_agreement/judger_results"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("human_annotation_agreement/tables"),
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Include files whose names contain '.partial'. Defaults to complete annotator files only.",
    )
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*.csv"))
    if not args.include_partial:
        paths = [p for p in paths if ".partial" not in p.name]
    if len(paths) < 2:
        raise SystemExit(f"Need at least two CSVs in {args.input_dir}")

    data = {annotator_name(path): read_annotations(path) for path in paths}
    annotators = list(data)

    inventory_rows = []
    for name, rows in data.items():
        inventory_rows.append(
            {
                "annotator": name,
                "file": str(args.input_dir / f"{name}.csv"),
                "rows": len(rows),
                **{
                    f"{label}_completed": sum(1 for row in rows.values() if row.get(field, ""))
                    for field, label in ANNOTATION_FIELDS
                },
            }
        )
    write_csv(
        args.output_dir / "annotator_inventory.csv",
        inventory_rows,
        ["annotator", "file", "rows"]
        + [f"{label}_completed" for _, label in ANNOTATION_FIELDS],
    )

    pairwise_rows = []
    for field, label in ANNOTATION_FIELDS:
        for a, b in combinations(annotators, 2):
            ids = sorted(set(data[a]) & set(data[b]))
            ids = [qid for qid in ids if data[a][qid].get(field, "") and data[b][qid].get(field, "")]
            a_vals = [data[a][qid][field] for qid in ids]
            b_vals = [data[b][qid][field] for qid in ids]
            same = sum(x == y for x, y in zip(a_vals, b_vals))
            pairwise_rows.append(
                {
                    "field": label,
                    "annotator_a": a,
                    "annotator_b": b,
                    "n_common": len(ids),
                    "same": same,
                    "different": len(ids) - same,
                    "percent_agreement": percent(same, len(ids)),
                    "cohen_kappa": cohen_kappa(a_vals, b_vals),
                }
            )
    write_csv(
        args.output_dir / "pairwise_agreement.csv",
        pairwise_rows,
        [
            "field",
            "annotator_a",
            "annotator_b",
            "n_common",
            "same",
            "different",
            "percent_agreement",
            "cohen_kappa",
        ],
    )

    multi_rows = []
    for field, label in ANNOTATION_FIELDS:
        ids = complete_ids(data, field)
        vote_rows = [[data[ann][qid][field] for ann in annotators] for qid in ids]
        unanimous = sum(len(set(votes)) == 1 for votes in vote_rows)
        majority = sum(max(Counter(votes).values()) >= 2 for votes in vote_rows)
        all_different = sum(len(set(votes)) == len(votes) for votes in vote_rows)
        majority_dist = Counter(Counter(votes).most_common(1)[0][0] for votes in vote_rows)
        row = {
            "field": label,
            "annotators": ";".join(annotators),
            "n_items_complete": len(ids),
            "n_annotators": len(annotators),
            "unanimous": unanimous,
            "unanimous_rate": percent(unanimous, len(ids)),
            "has_majority": majority,
            "majority_rate": percent(majority, len(ids)),
            "all_different": all_different,
            "all_different_rate": percent(all_different, len(ids)),
            "fleiss_kappa": fleiss_kappa(vote_rows),
            "krippendorff_alpha": krippendorff_alpha_nominal(vote_rows),
            "majority_label_distribution": json.dumps(dict(majority_dist), sort_keys=True),
        }
        multi_rows.append(row)
    write_csv(
        args.output_dir / "multi_annotator_agreement.csv",
        multi_rows,
        [
            "field",
            "annotators",
            "n_items_complete",
            "n_annotators",
            "unanimous",
            "unanimous_rate",
            "has_majority",
            "majority_rate",
            "all_different",
            "all_different_rate",
            "fleiss_kappa",
            "krippendorff_alpha",
            "majority_label_distribution",
        ],
    )

    item_rows = []
    common_all = sorted(set.intersection(*(set(rows) for rows in data.values())))
    for qid in common_all:
        row: dict[str, object] = {"qa_id": qid}
        question = next((data[ann][qid].get("question", "") for ann in annotators if qid in data[ann]), "")
        row["question"] = question
        for field, label in ANNOTATION_FIELDS:
            votes = [data[ann][qid].get(field, "") for ann in annotators]
            counts = Counter(v for v in votes if v)
            row[f"{label}_agreement_type"] = (
                "unanimous"
                if len(set(votes)) == 1 and votes[0]
                else "majority"
                if counts and counts.most_common(1)[0][1] >= 2
                else "no_majority"
            )
            row[f"{label}_majority_label"] = counts.most_common(1)[0][0] if counts else ""
            for ann, vote in zip(annotators, votes):
                row[f"{label}__{ann}"] = vote
        item_rows.append(row)
    item_fields = ["qa_id", "question"]
    for _, label in ANNOTATION_FIELDS:
        item_fields.extend([f"{label}_agreement_type", f"{label}_majority_label"])
        item_fields.extend(f"{label}__{ann}" for ann in annotators)
    write_csv(args.output_dir / "item_level_votes.csv", item_rows, item_fields)

    print(f"Loaded annotators: {', '.join(annotators)}")
    print(f"Wrote tables to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
