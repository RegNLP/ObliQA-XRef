# src/obliqaxref/generate/common/sampling.py
"""
Difficulty-aware cross-reference row sampler.

Features extracted per cross-reference row
------------------------------------------
source_length_words                      int   — whitespace-token count of SourcePassage
target_length_words                      int   — whitespace-token count of TargetPassage
source_target_lexical_overlap            float — Jaccard similarity of content-word sets
source_target_jaccard_similarity         float — alias of source_target_lexical_overlap
number_of_outgoing_refs_from_source      int   — rows sharing the same SourceID
number_of_incoming_refs_to_target        int   — rows sharing the same TargetID
source_contains_multiple_refs            bool  — number_of_outgoing_refs_from_source > 1
reference_text_present_in_source         bool  — ReferenceText substring in SourcePassage (case-insensitive)
target_contains_definition_like_text     bool  — definition-cue regex match in TargetPassage
target_contains_duration_or_deadline     bool  — temporal/deadline cue regex match
target_contains_threshold_or_numeric_value bool — numeric threshold cue regex match
target_contains_exception_or_condition   bool  — exception/condition cue regex match
low_source_target_overlap                bool  — jaccard < max_jaccard_for_low_overlap

Sampling modes
--------------
random                        — uniform random sample (default; backward-compatible)
low_overlap                   — prefer source-target pairs with lowest Jaccard similarity
multi_ref_source              — prefer source passages that reference multiple targets
target_definition_or_condition — prefer targets with definition/condition cues
mixed_difficulty              — stratified sample across all difficulty buckets
"""

from __future__ import annotations

import math
import random as _random_module
import re
import statistics
from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# Stopword set (no external dependencies)
# ---------------------------------------------------------------------------
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "to", "for", "on", "by",
    "with", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "that", "this", "these", "those", "at", "from",
    "as", "has", "have", "had", "not", "but", "if", "no", "so",
    "do", "does", "did", "will", "would", "may", "might", "shall", "should",
    "can", "could", "into", "out", "up", "than", "such", "any", "each", "all",
    "also", "which", "who", "whom", "when", "where", "how", "what", "more",
    "other", "their", "there", "they", "them", "then", "s", "re",
})

# ---------------------------------------------------------------------------
# Regex patterns for target-passage content cues
# ---------------------------------------------------------------------------
_DEFINITION_RE = re.compile(
    r"\b(means|meaning|defined\s+as|definition\s+of|refers\s+to|shall\s+mean|is\s+defined|definition)\b",
    re.IGNORECASE,
)

_DURATION_RE = re.compile(
    r"""(?:
        \d+\s*(?:calendar\s+|business\s+)?(?:day|days|month|months|year|years|week|weeks|hour|hours)
        | within\s+\d+
        | not\s+later\s+than
        | no\s+later\s+than
        | deadline
        | before\s+the\s+(?:end|date|expiry)
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_THRESHOLD_RE = re.compile(
    r"""(?:
        \d+(?:\.\d+)?\s*(?:%|percent|per\s+cent)
        | \b(?:at\s+least|at\s+most|no\s+more\s+than|no\s+less\s+than
              |minimum|maximum|threshold|exceeds?|below|above
              |greater\s+than|less\s+than)
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_EXCEPTION_RE = re.compile(
    r"\b(unless|except|provided\s+that|subject\s+to|notwithstanding|other\s+than"
    r"|conditional|condition|requirement)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
VALID_MODES: frozenset[str] = frozenset({
    "random",
    "low_overlap",
    "multi_ref_source",
    "target_definition_or_condition",
    "mixed_difficulty",
})

DEFAULT_BUCKET_WEIGHTS: dict[str, float] = {
    "low_overlap": 0.30,
    "multi_ref": 0.20,
    "definition_condition": 0.30,
    "random": 0.20,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> frozenset[str]:
    """Return lowercase alpha tokens from *text*, excluding stopwords."""
    if not text:
        return frozenset()
    tokens = re.findall(r"[a-z]+", text.lower())
    return frozenset(t for t in tokens if t not in _STOPWORDS and len(t) > 1)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|.  Returns 0.0 when both sets are empty."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _stats_dict(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": round(statistics.mean(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "median": round(statistics.median(values), 4),
    }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_xref_features(
    row: dict[str, Any],
    *,
    outgoing_counts: dict[str, int],
    incoming_counts: dict[str, int],
    max_jaccard_for_low_overlap: float = 0.15,
) -> dict[str, Any]:
    """Compute difficulty-aware features for a single cross-reference row.

    Parameters
    ----------
    row:
        A dict from ``crossref_resolved.cleaned.csv`` with at least the keys:
        ``SourceID``, ``TargetID``, ``ReferenceText``, ``SourcePassage``, ``TargetPassage``.
    outgoing_counts:
        Precomputed mapping ``SourceID → count``.
    incoming_counts:
        Precomputed mapping ``TargetID → count``.
    max_jaccard_for_low_overlap:
        Pairs whose Jaccard similarity falls below this threshold are flagged as
        ``low_source_target_overlap=True``.

    Returns
    -------
    dict
        All feature values as a flat dict (no nesting).
    """
    src_text = str(row.get("SourcePassage") or "")
    tgt_text = str(row.get("TargetPassage") or "")
    ref_text = str(row.get("ReferenceText") or "")
    src_id = str(row.get("SourceID") or "")
    tgt_id = str(row.get("TargetID") or "")

    src_tokens = _tokenize(src_text)
    tgt_tokens = _tokenize(tgt_text)
    jac = _jaccard(src_tokens, tgt_tokens)

    n_out = outgoing_counts.get(src_id, 1)
    n_in = incoming_counts.get(tgt_id, 1)

    return {
        "source_length_words": len(src_text.split()),
        "target_length_words": len(tgt_text.split()),
        "source_target_lexical_overlap": round(jac, 4),
        "source_target_jaccard_similarity": round(jac, 4),
        "number_of_outgoing_refs_from_source": n_out,
        "number_of_incoming_refs_to_target": n_in,
        "source_contains_multiple_refs": n_out > 1,
        "reference_text_present_in_source": bool(
            ref_text and ref_text.lower() in src_text.lower()
        ),
        "target_contains_definition_like_text": bool(_DEFINITION_RE.search(tgt_text)),
        "target_contains_duration_or_deadline": bool(_DURATION_RE.search(tgt_text)),
        "target_contains_threshold_or_numeric_value": bool(_THRESHOLD_RE.search(tgt_text)),
        "target_contains_exception_or_condition": bool(_EXCEPTION_RE.search(tgt_text)),
        "low_source_target_overlap": jac < max_jaccard_for_low_overlap,
    }


def build_feature_table(
    rows: list[dict[str, Any]],
    max_jaccard_for_low_overlap: float = 0.15,
) -> list[dict[str, Any]]:
    """Return new dicts that merge each original row with its computed features.

    Precomputes outgoing/incoming reference counts for O(n) total complexity.
    Original row dicts are not mutated.
    """
    outgoing_counts: dict[str, int] = Counter(
        str(r.get("SourceID") or "") for r in rows
    )
    incoming_counts: dict[str, int] = Counter(
        str(r.get("TargetID") or "") for r in rows
    )
    enriched: list[dict[str, Any]] = []
    for r in rows:
        features = extract_xref_features(
            r,
            outgoing_counts=outgoing_counts,
            incoming_counts=incoming_counts,
            max_jaccard_for_low_overlap=max_jaccard_for_low_overlap,
        )
        enriched.append({**r, **features})
    return enriched


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _compute_feature_summary(rows_with_features: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate statistics over a list of enriched rows."""
    if not rows_with_features:
        return {}
    src_lens = [r["source_length_words"] for r in rows_with_features]
    tgt_lens = [r["target_length_words"] for r in rows_with_features]
    jaccards = [r["source_target_jaccard_similarity"] for r in rows_with_features]
    return {
        "source_length_words": _stats_dict(src_lens),
        "target_length_words": _stats_dict(tgt_lens),
        "source_target_jaccard_similarity": _stats_dict(jaccards),
        "n_low_overlap": sum(1 for r in rows_with_features if r["low_source_target_overlap"]),
        "n_multi_ref_source": sum(1 for r in rows_with_features if r["source_contains_multiple_refs"]),
        "n_target_definition_like_text": sum(
            1 for r in rows_with_features if r["target_contains_definition_like_text"]
        ),
        "n_target_duration_or_deadline": sum(
            1 for r in rows_with_features if r["target_contains_duration_or_deadline"]
        ),
        "n_target_threshold_or_numeric": sum(
            1 for r in rows_with_features if r["target_contains_threshold_or_numeric_value"]
        ),
        "n_target_exception_or_condition": sum(
            1 for r in rows_with_features if r["target_contains_exception_or_condition"]
        ),
        "n_reference_text_present_in_source": sum(
            1 for r in rows_with_features if r["reference_text_present_in_source"]
        ),
    }


# ---------------------------------------------------------------------------
# Bucket assignment (for mixed_difficulty)
# ---------------------------------------------------------------------------

def _assign_bucket(row: dict[str, Any]) -> str:
    """Assign a row to its highest-priority difficulty bucket (priority order below)."""
    if row.get("low_source_target_overlap"):
        return "low_overlap"
    if row.get("source_contains_multiple_refs"):
        return "multi_ref"
    if row.get("target_contains_definition_like_text") or row.get("target_contains_exception_or_condition"):
        return "definition_condition"
    return "random"


# ---------------------------------------------------------------------------
# Sampling strategies (all operate on enriched row dicts)
# ---------------------------------------------------------------------------

def _sample_random(
    rows: list[dict[str, Any]], n: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = _random_module.Random(seed)
    selected = rng.sample(rows, n) if n < len(rows) else list(rows)
    return selected, {"random": len(selected)}


def _sample_low_overlap(
    rows: list[dict[str, Any]], n: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sort by Jaccard ascending; take the *n* pairs with lowest overlap."""
    sorted_rows = sorted(rows, key=lambda r: r.get("source_target_jaccard_similarity", 1.0))
    candidates = sorted_rows[:n] if n < len(sorted_rows) else list(sorted_rows)
    # Shuffle within the selection for diversity
    _random_module.Random(seed).shuffle(candidates)
    return candidates, {"low_overlap": len(candidates)}


def _sample_mode_filtered(
    rows: list[dict[str, Any]],
    n: int,
    seed: int,
    filter_keys: tuple[str, ...],
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sample from rows matching any of *filter_keys*; fill remainder randomly."""
    matching = [r for r in rows if any(r.get(k) for k in filter_keys)]
    non_matching = [r for r in rows if not any(r.get(k) for k in filter_keys)]
    rng = _random_module.Random(seed)
    if len(matching) >= n:
        selected = rng.sample(matching, n)
        return selected, {label: n}
    selected = list(matching)
    filler_n = min(n - len(selected), len(non_matching))
    filler = rng.sample(non_matching, filler_n) if filler_n > 0 else []
    selected.extend(filler)
    return selected, {label: len(matching), "random": len(filler)}


def _sample_mixed_difficulty(
    rows: list[dict[str, Any]],
    n: int,
    seed: int,
    bucket_weights: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Stratified sample across four difficulty buckets using priority assignment."""
    # Assign each row to exactly one bucket (priority: low_overlap > multi_ref > definition_condition > random)
    buckets: dict[str, list[dict[str, Any]]] = {
        "low_overlap": [],
        "multi_ref": [],
        "definition_condition": [],
        "random": [],
    }
    for r in rows:
        buckets[_assign_bucket(r)].append(r)

    # Normalize weights
    total_w = sum(bucket_weights.get(k, 0.0) for k in buckets)
    if total_w <= 0.0:
        total_w = 1.0
    norm_w = {k: bucket_weights.get(k, 0.0) / total_w for k in buckets}

    # Allocate target counts per bucket (floored to ints)
    allocations: dict[str, int] = {
        k: min(len(buckets[k]), math.floor(n * norm_w[k])) for k in buckets
    }

    # Distribute rounding remainder (highest-weight buckets first)
    sorted_keys = sorted(buckets, key=lambda k: -norm_w[k])
    for k in sorted_keys:
        extra = n - sum(allocations.values())
        if extra <= 0:
            break
        headroom = len(buckets[k]) - allocations[k]
        if headroom > 0:
            allocations[k] += min(extra, headroom)

    rng = _random_module.Random(seed)
    selected: list[dict[str, Any]] = []
    bucket_counts: dict[str, int] = {}
    for k, pool in buckets.items():
        k_n = allocations[k]
        if k_n <= 0:
            bucket_counts[k] = 0
            continue
        sampled = rng.sample(pool, k_n) if k_n < len(pool) else list(pool)
        selected.extend(sampled)
        bucket_counts[k] = len(sampled)

    # Fill any shortfall (all buckets exhausted before reaching n)
    if len(selected) < n:
        selected_ids = {(r.get("SourceID"), r.get("TargetID")) for r in selected}
        remaining = [r for r in rows if (r.get("SourceID"), r.get("TargetID")) not in selected_ids]
        extra_n = min(n - len(selected), len(remaining))
        if extra_n > 0:
            extra = rng.sample(remaining, extra_n)
            selected.extend(extra)
            bucket_counts["random"] = bucket_counts.get("random", 0) + extra_n

    return selected, bucket_counts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def sample_xref_rows(
    rows: list[dict[str, Any]],
    n: int,
    mode: str = "random",
    seed: int = 13,
    max_jaccard_for_low_overlap: float = 0.15,
    mixed_difficulty_bucket_weights: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample cross-reference rows using the specified difficulty-aware strategy.

    Parameters
    ----------
    rows:
        All candidate cross-reference rows from ``crossref_resolved.cleaned.csv``.
    n:
        Number of rows to sample.  ``n <= 0`` or ``n >= len(rows)`` returns all rows.
    mode:
        One of ``VALID_MODES``.
    seed:
        Random seed for reproducibility.
    max_jaccard_for_low_overlap:
        Jaccard threshold for the ``low_source_target_overlap`` boolean feature.
    mixed_difficulty_bucket_weights:
        Relative weights per bucket for ``mixed_difficulty`` mode.  Defaults to
        ``DEFAULT_BUCKET_WEIGHTS`` when ``None``.

    Returns
    -------
    (sampled_rows, report)
        ``sampled_rows`` is a list of row dicts **with features merged in** as
        additional keys.  Downstream code that only reads CSV-native keys is
        unaffected by the extra fields.
        ``report`` is a JSON-serialisable dict suitable for ``sampling_report.json``.
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"sampling_mode must be one of {sorted(VALID_MODES)!r}, got {mode!r}"
        )

    n_candidates = len(rows)

    # Enrich every row with features (needed for non-random modes and for the report)
    enriched = build_feature_table(rows, max_jaccard_for_low_overlap=max_jaccard_for_low_overlap)
    feature_summary = _compute_feature_summary(enriched)

    # No sampling required
    if n <= 0 or n >= n_candidates:
        report: dict[str, Any] = {
            "sampling_mode": mode,
            "n_candidates": n_candidates,
            "n_sampled": n_candidates,
            "n_requested": n,
            "seed": seed,
            "no_sampling_applied": True,
            "bucket_counts": {},
            "feature_summary": feature_summary,
        }
        return enriched, report

    # Dispatch to the appropriate strategy
    if mode == "random":
        sampled, bucket_counts = _sample_random(enriched, n, seed)
    elif mode == "low_overlap":
        sampled, bucket_counts = _sample_low_overlap(enriched, n, seed)
    elif mode == "multi_ref_source":
        sampled, bucket_counts = _sample_mode_filtered(
            enriched, n, seed,
            filter_keys=("source_contains_multiple_refs",),
            label="multi_ref_source",
        )
    elif mode == "target_definition_or_condition":
        sampled, bucket_counts = _sample_mode_filtered(
            enriched, n, seed,
            filter_keys=("target_contains_definition_like_text", "target_contains_exception_or_condition"),
            label="target_definition_or_condition",
        )
    else:  # mixed_difficulty
        weights = mixed_difficulty_bucket_weights or DEFAULT_BUCKET_WEIGHTS
        sampled, bucket_counts = _sample_mixed_difficulty(enriched, n, seed, weights)

    report = {
        "sampling_mode": mode,
        "n_candidates": n_candidates,
        "n_sampled": len(sampled),
        "n_requested": n,
        "seed": seed,
        "no_sampling_applied": False,
        "bucket_counts": bucket_counts,
        "feature_summary": feature_summary,
    }
    return sampled, report
