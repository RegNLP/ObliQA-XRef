"""
Export a 200-item stratified human-verification subset from the final merged dataset.

Stratification: 8 strata × 25 items = 200
  corpus:            ADGM, UKFIN
  generation_method: DPEL, SCHEMA
  sampling_regime:   mixed_difficulty, hard_enriched

Outputs (under ObliQA-XRef_Out_Datasets/final_large_merged/human_verification/):
  human_verification_subset_200.csv
  human_verification_subset_200.jsonl
  human_verification_subset_200_stats.json
  README_human_verification_subset.md

Usage:
  python src/obliqaxref/eval/export_human_verification_subset.py [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
ITEMS_PER_STRATUM = 25
CORPORA = ["adgm", "ukfin"]
METHODS = ["DPEL", "SCHEMA"]
REGIMES = ["mixed_difficulty", "hard_enriched"]

INPUT_TEMPLATE = (
    "ObliQA-XRef_Out_Datasets/final_large_merged/ObliQA-XRef-{CORPUS}-ALL/final.jsonl"
)
DEFAULT_OUT_DIR = Path(
    "ObliQA-XRef_Out_Datasets/final_large_merged/human_verification"
)
ANNOTATOR_HTML = "data/HumanAnnotation.html"
PASSAGE_CORPUS_TEMPLATE = "data/{corpus}/processed/passage_corpus.jsonl"

# CSV column order for the annotator
CSV_FIELDNAMES = [
    "qa_id",
    "question",
    "expected_answer",
    "source_text",
    "target_text",
    "method",
    "split",
    "persona",
    "corpus",
    "sampling_regime",
    "source_passage_pid",
    "target_passage_pid",
    "source_doc_id",
    "target_doc_id",
    "source_passage_label",
    "target_passage_label",
    "reference_text",
    "reference_type",
    "ReferenceText",
    "ReferenceType",
    "annotation_id",
    "original_item_id",
    "source_path",
    # Annotation columns (empty; annotator fills these)
    "q_understandable",
    "question_phrasing_reuse",
    "cross_reference_connection_type",
    "evidence_dependency",
    "comment",
]

# ---------------------------------------------------------------------------
# Field-name resolution helpers (handles variation in final.jsonl field names)
# ---------------------------------------------------------------------------

_QUESTION_KEYS = ["question", "Question"]
_ANSWER_KEYS = ["gold_answer", "answer", "reference_answer", "expected_answer", "Answer"]
_SOURCE_TEXT_KEYS = [
    "source_text",
    "source_passage",
    "source_passage_text",
    "SourcePassage",
    "source",
]
_TARGET_TEXT_KEYS = [
    "target_text",
    "target_passage",
    "target_passage_text",
    "TargetPassage",
    "target",
]
_SOURCE_ID_KEYS = [
    "source_passage_id",
    "source_id",
    "SOURCE_ID",
    "SourceID",
]
_TARGET_ID_KEYS = [
    "target_passage_id",
    "target_id",
    "TARGET_ID",
    "TargetID",
]
_METHOD_KEYS = ["generation_method", "method", "source_method"]
_CORPUS_KEYS = ["corpus"]
_REGIME_KEYS = ["sampling_regime"]


def _pick(rec: dict[str, Any], keys: list[str], default: str = "") -> str:
    for k in keys:
        v = rec.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


@lru_cache(maxsize=None)
def _passage_id_by_pid(corpus: str) -> dict[str, str]:
    """Map internal passage UID/PID to human-readable passage_id where available."""
    path = Path(PASSAGE_CORPUS_TEMPLATE.format(corpus=corpus.lower()))
    if not path.exists():
        return {}

    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            uid = rec.get("pid") or rec.get("passage_uid")
            passage_id = rec.get("passage_id")
            if uid and passage_id:
                mapping[str(uid).strip()] = str(passage_id).strip()
    return mapping


def _display_passage_id(corpus: str, passage_pid: str, existing_label: Any = "") -> str:
    """Prefer meaningful corpus passage_id labels; fall back to existing labels/PIDs."""
    if not passage_pid:
        return str(existing_label or "")
    resolved = _passage_id_by_pid(corpus).get(passage_pid)
    if resolved:
        return resolved
    if existing_label is not None and str(existing_label).strip():
        return str(existing_label).strip()
    return passage_pid


# ---------------------------------------------------------------------------
# Load & stratify
# ---------------------------------------------------------------------------

def load_and_stratify(
    corpus: str,
) -> tuple[dict[tuple[str, str, str], list[dict]], list[str]]:
    """Load final.jsonl for a corpus; bucket by (corpus_norm, method_upper, regime)."""
    path = Path(INPUT_TEMPLATE.format(CORPUS=corpus.upper()))
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    warnings: list[str] = []

    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"{path.name} line {lineno}: JSON error – {exc}")
                continue

            method_raw = _pick(rec, _METHOD_KEYS)
            method_norm = method_raw.upper() if method_raw else "UNKNOWN"
            regime = _pick(rec, _REGIME_KEYS)
            corpus_norm = _pick(rec, _CORPUS_KEYS, corpus).lower()

            if method_norm not in ("DPEL", "SCHEMA"):
                warnings.append(
                    f"{path.name} line {lineno}: unexpected method {method_raw!r}; skipping"
                )
                continue
            if regime not in REGIMES:
                warnings.append(
                    f"{path.name} line {lineno}: unexpected regime {regime!r}; skipping"
                )
                continue

            key = (corpus_norm, method_norm, regime)
            buckets[key].append(rec)

    logger.info(
        "  %s: loaded %d records into %d strata",
        corpus.upper(),
        sum(len(v) for v in buckets.values()),
        len(buckets),
    )
    return dict(buckets), warnings


# ---------------------------------------------------------------------------
# Build CSV row from a raw record
# ---------------------------------------------------------------------------

def build_row(
    rec: dict[str, Any],
    annotation_id: str,
    source_file: str,
) -> dict[str, str]:
    item_id = rec.get("item_id", "")
    source_pid = _pick(rec, _SOURCE_ID_KEYS)
    target_pid = _pick(rec, _TARGET_ID_KEYS)
    corpus_norm = _pick(rec, _CORPUS_KEYS).lower()

    # Stable qa_id
    if item_id:
        qa_id = item_id
    elif source_pid and target_pid:
        qa_id = f"{corpus_norm}__{source_pid}__{target_pid}"
    else:
        qa_id = annotation_id  # last resort

    ref_text = rec.get("reference_text") or rec.get("ReferenceText") or ""
    ref_type = rec.get("reference_type") or rec.get("ReferenceType") or ""

    return {
        "qa_id": qa_id,
        "question": _pick(rec, _QUESTION_KEYS),
        "expected_answer": _pick(rec, _ANSWER_KEYS),
        "source_text": _pick(rec, _SOURCE_TEXT_KEYS),
        "target_text": _pick(rec, _TARGET_TEXT_KEYS),
        "method": _pick(rec, _METHOD_KEYS, "").upper(),
        "split": rec.get("split", ""),
        "persona": rec.get("persona", ""),
        "corpus": corpus_norm.upper(),
        "sampling_regime": _pick(rec, _REGIME_KEYS),
        "source_passage_pid": source_pid,
        "target_passage_pid": target_pid,
        "source_doc_id": rec.get("source_doc_id", ""),
        "target_doc_id": rec.get("target_doc_id", ""),
        "source_passage_label": _display_passage_id(
            corpus_norm, source_pid, rec.get("source_passage_label", "")
        ),
        "target_passage_label": _display_passage_id(
            corpus_norm, target_pid, rec.get("target_passage_label", "")
        ),
        "reference_text": str(ref_text),
        "reference_type": str(ref_type),
        "ReferenceText": str(ref_text),
        "ReferenceType": str(ref_type),
        "annotation_id": annotation_id,
        "original_item_id": item_id,
        "source_path": source_file,
        # Annotation columns empty — annotator fills these
        "q_understandable": "",
        "question_phrasing_reuse": "",
        "cross_reference_connection_type": "",
        "evidence_dependency": "",
        "comment": "",
    }


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export(out_dir: Path = DEFAULT_OUT_DIR) -> None:
    rng = random.Random(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    all_recs: list[dict[str, Any]] = []
    warnings: list[str] = []
    counts_by_stratum: dict[str, int] = {}
    input_files: list[str] = []

    logger.info("=" * 60)
    logger.info("ObliQA-XRef Human Verification Subset Export")
    logger.info("Seed=%d, items_per_stratum=%d, total_target=%d",
                SEED, ITEMS_PER_STRATUM, ITEMS_PER_STRATUM * len(CORPORA) * len(METHODS) * len(REGIMES))
    logger.info("=" * 60)

    annotation_counter = 0

    for corpus in CORPORA:
        input_file = INPUT_TEMPLATE.format(CORPUS=corpus.upper())
        input_files.append(input_file)
        logger.info("\nLoading %s ...", input_file)
        buckets, load_warns = load_and_stratify(corpus)
        warnings.extend(load_warns)

        for method in METHODS:
            for regime in REGIMES:
                key = (corpus.lower(), method, regime)
                stratum_label = f"{corpus.upper()}_{method}_{regime}"
                pool = buckets.get(key, [])

                if len(pool) == 0:
                    msg = (
                        f"STRATUM {stratum_label}: NO ITEMS FOUND — "
                        "expected at least 1 item; sampling nothing for this stratum."
                    )
                    warnings.append(msg)
                    logger.warning(msg)
                    counts_by_stratum[stratum_label] = 0
                    continue

                if len(pool) < ITEMS_PER_STRATUM:
                    msg = (
                        f"STRATUM {stratum_label}: only {len(pool)} items available "
                        f"(requested {ITEMS_PER_STRATUM}); sampling all {len(pool)}."
                    )
                    warnings.append(msg)
                    logger.warning(msg)

                n = min(ITEMS_PER_STRATUM, len(pool))
                sampled = rng.sample(pool, n)
                counts_by_stratum[stratum_label] = n

                for rec in sampled:
                    annotation_counter += 1
                    annotation_id = f"HV{annotation_counter:04d}"
                    row = build_row(rec, annotation_id, input_file)
                    all_rows.append(row)
                    # Also keep the raw record augmented with annotation_id for JSONL
                    aug = dict(rec)
                    aug["annotation_id"] = annotation_id
                    aug["source_path_export"] = input_file
                    all_recs.append(aug)

                logger.info(
                    "  ✓ %s: sampled %d / %d items", stratum_label, n, len(pool)
                )

    total_sampled = len(all_rows)
    logger.info("\n  Total sampled: %d", total_sampled)

    # -------------------------------------------------------------------
    # Write CSV
    # -------------------------------------------------------------------
    csv_path = out_dir / "human_verification_subset_200.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info("\nWrote CSV: %s  (%d data rows)", csv_path, total_sampled)

    # -------------------------------------------------------------------
    # Write JSONL
    # -------------------------------------------------------------------
    jsonl_path = out_dir / "human_verification_subset_200.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in all_recs:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote JSONL: %s", jsonl_path)

    # -------------------------------------------------------------------
    # Compute stats
    # -------------------------------------------------------------------
    counts_by_corpus: dict[str, int] = defaultdict(int)
    counts_by_method: dict[str, int] = defaultdict(int)
    counts_by_regime: dict[str, int] = defaultdict(int)
    for row in all_rows:
        counts_by_corpus[row["corpus"]] += 1
        counts_by_method[row["method"]] += 1
        counts_by_regime[row["sampling_regime"]] += 1

    stats: dict[str, Any] = {
        "total_sampled": total_sampled,
        "seed": SEED,
        "items_per_stratum_requested": ITEMS_PER_STRATUM,
        "input_files": input_files,
        "output_csv": str(csv_path),
        "output_jsonl": str(jsonl_path),
        "annotator_html_path": ANNOTATOR_HTML,
        "counts_by_corpus": dict(counts_by_corpus),
        "counts_by_method": dict(counts_by_method),
        "counts_by_sampling_regime": dict(counts_by_regime),
        "counts_by_stratum": counts_by_stratum,
        "missing_field_warnings": warnings if warnings else [],
    }

    stats_path = out_dir / "human_verification_subset_200_stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote stats: %s", stats_path)

    # -------------------------------------------------------------------
    # Write README
    # -------------------------------------------------------------------
    readme_path = out_dir / "README_human_verification_subset.md"
    readme_text = _build_readme(
        total_sampled=total_sampled,
        counts_by_stratum=counts_by_stratum,
        csv_filename=csv_path.name,
        annotator_html=ANNOTATOR_HTML,
    )
    with readme_path.open("w", encoding="utf-8") as fh:
        fh.write(readme_text)
    logger.info("Wrote README: %s", readme_path)

    # -------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION")
    logger.info("=" * 60)

    _validate(csv_path, jsonl_path, stats_path, readme_path, all_rows, counts_by_stratum, warnings)

    # -------------------------------------------------------------------
    # First-3-rows preview
    # -------------------------------------------------------------------
    logger.info("\nFirst 3 sampled rows preview:")
    logger.info("%-10s %-20s %-8s %-8s %-20s %-50s %-40s %-40s",
                "ann_id", "qa_id", "corpus", "method", "sampling_regime",
                "question (50ch)", "source_text (40ch)", "target_text (40ch)")
    for row in all_rows[:3]:
        logger.info(
            "%-10s %-20s %-8s %-8s %-20s %-50s %-40s %-40s",
            row["annotation_id"],
            row["qa_id"][:20],
            row["corpus"],
            row["method"],
            row["sampling_regime"],
            (row["question"] or "")[:50],
            (row["source_text"] or "")[:40],
            (row["target_text"] or "")[:40],
        )

    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate(
    csv_path: Path,
    jsonl_path: Path,
    stats_path: Path,
    readme_path: Path,
    all_rows: list[dict],
    counts_by_stratum: dict[str, int],
    warnings: list[str],
) -> None:
    errors: list[str] = []

    for path, label in [
        (csv_path, "CSV"),
        (jsonl_path, "JSONL"),
        (stats_path, "stats JSON"),
        (readme_path, "README"),
    ]:
        if path.exists():
            logger.info("  ✓ %s exists: %s", label, path.name)
        else:
            errors.append(f"MISSING {label}: {path}")

    # Row count
    n = len(all_rows)
    if n == ITEMS_PER_STRATUM * len(CORPORA) * len(METHODS) * len(REGIMES):
        logger.info("  ✓ Total rows: %d (= 200)", n)
    else:
        msg = f"  ⚠ Total rows: {n} (expected 200)"
        logger.warning(msg)
        warnings.append(msg)

    # Per-stratum counts
    for key, cnt in sorted(counts_by_stratum.items()):
        if cnt == ITEMS_PER_STRATUM:
            logger.info("  ✓ %s: %d rows", key, cnt)
        else:
            msg = f"  ⚠ {key}: {cnt} rows (expected {ITEMS_PER_STRATUM})"
            logger.warning(msg)

    # Required UI columns
    required_ui = ["qa_id", "question", "source_text", "target_text"]
    for col in required_ui:
        present = all(col in row for row in all_rows)
        if present:
            logger.info("  ✓ Column '%s' present in all rows", col)
        else:
            errors.append(f"Column '{col}' missing in some rows")

    # No empty required fields
    for col in required_ui:
        empties = [
            row["annotation_id"]
            for row in all_rows
            if not (row.get(col) or "").strip()
        ]
        if not empties:
            logger.info("  ✓ No empty values in '%s'", col)
        else:
            msg = f"  ⚠ Empty '{col}' in {len(empties)} rows: {empties[:5]}"
            logger.warning(msg)
            warnings.append(msg)

    if errors:
        for e in errors:
            logger.error("  ✗ %s", e)
        raise RuntimeError(f"Validation failed: {errors}")

    if warnings:
        logger.info("\n  Warnings (%d total):", len(warnings))
        for w in warnings:
            logger.warning("    %s", w)
    else:
        logger.info("  No warnings.")


# ---------------------------------------------------------------------------
# README builder
# ---------------------------------------------------------------------------

def _build_readme(
    total_sampled: int,
    counts_by_stratum: dict[str, int],
    csv_filename: str,
    annotator_html: str,
) -> str:
    stratum_rows = "\n".join(
        f"| {k} | {v} |" for k, v in sorted(counts_by_stratum.items())
    )
    return f"""\
# ObliQA-XRef Human Verification Subset

This directory contains a **{total_sampled}-item stratified human-verification subset**
extracted from the final merged ObliQA-XRef dataset for manual audit.

## How to annotate

1. Open **`{annotator_html}`** in a web browser (no server required — it is a
   self-contained single-page HTML file).
2. Click **"Start assignment"** and upload **`{csv_filename}`** from this directory.
3. The tool shows each item's question, source passage, and target passage.
   Navigate with the Previous / Next buttons and fill in the four annotation
   questions for each item:

   | Column | Values | Meaning |
   |---|---|---|
   | `q_understandable` | `yes` / `no` / `partial_unclear` | Is the question understandable? |
   | `question_phrasing_reuse` | `low_reuse` / `moderate_reuse` / `high_reuse` / `unclear` | To what extent does the question reuse wording from the passages? |
   | `cross_reference_connection_type` | `strict_explicit` / `range_based_explicit` / `implicit_semantic` / `no_relation` | What is the nature of the cross-reference connection? |
   | `evidence_dependency` | `both_needed` / `source_sufficient` / `target_sufficient` / `insufficient_mismatch_unclear` | Which passage(s) appear to provide the evidence needed for the question? |
   | `comment` | free text | Optional short note. |

4. Click **"Save progress"** to download the updated CSV at any time.
   Click **"Continue"** on a later visit to resume from where you left off.

## Sampling methodology

- **Seed**: {SEED}
- **Items per stratum**: {ITEMS_PER_STRATUM}
- **Stratification axes**: corpus × generation_method × sampling_regime

| Stratum | Items sampled |
|---|---|
{stratum_rows}

Items were drawn uniformly at random (without replacement) from the
`final.jsonl` files under
`ObliQA-XRef_Out_Datasets/final_large_merged/ObliQA-XRef-{{CORPUS}}-ALL/`.

## Files in this directory

| File | Description |
|---|---|
| `{csv_filename}` | Annotation-ready CSV (upload to annotator) |
| `human_verification_subset_200.jsonl` | Same items in JSONL with full metadata |
| `human_verification_subset_200_stats.json` | Sampling statistics and field-warning log |
| `README_human_verification_subset.md` | This file |
"""


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export 200-item stratified human-verification subset."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()
    export(out_dir=args.out_dir)


if __name__ == "__main__":
    main()
