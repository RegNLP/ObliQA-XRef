from __future__ import annotations

from typing import Any


OBLIQA_XREF_METADATA: dict[str, Any] = {
    "benchmark_family": "ObliQA",
    "benchmark_name": "ObliQA-XRef",
    "benchmark_role": "citation-dependent regulatory QA",
    "evidence_structure": "source_to_target_cross_reference",
    "source_insufficiency_required": True,
    "target_necessity_required": True,
    "relation_to_obliqa": "complementary diagnostic extension",
    "relation_to_obliqa_mp": (
        "distinct from general multi-passage QA; focuses on explicit source-to-target "
        "cross-reference dependency"
    ),
}

OBLIQA_XREF_METADATA_FIELDS: tuple[str, ...] = tuple(OBLIQA_XREF_METADATA)


def with_obliqa_xref_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *record* with default ObliQA-XRef metadata populated."""
    out = dict(record)
    for key, value in OBLIQA_XREF_METADATA.items():
        out.setdefault(key, value)
    return out


def add_obliqa_xref_metadata_inplace(record: dict[str, Any]) -> dict[str, Any]:
    """Populate default ObliQA-XRef metadata on *record* and return it."""
    for key, value in OBLIQA_XREF_METADATA.items():
        record.setdefault(key, value)
    return record
