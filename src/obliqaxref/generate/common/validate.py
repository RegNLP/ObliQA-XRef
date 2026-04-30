# src/obliqaxref/generate/common/validate.py
"""
ObliQA-XRef Generator — validation utilities

This module provides:
- lightweight structural validators for SchemaItem and QAItem
- answer tag enforcement ([#SRC:...] and [#TGT:...])
- answer span validation (substring + offsets)
- optional strict constraints (length bounds, no-citation policy)
- structured citation leakage detection (detect_citation_leakage)

No model calls. Deterministic and fast.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from obliqaxref.generate.common.filters import looks_like_empty, norm_ws
from obliqaxref.generate.types import AnswerSpan, QAItem, SchemaItem, SpanType


# ---------------------------------------------------------------------
# Tag enforcement
# ---------------------------------------------------------------------
def has_required_tags(answer: str, source_uid: str, target_uid: str) -> bool:
    """
    Answer must contain both distinct passage tags, exactly as written.
    """
    if not answer:
        return False
    if source_uid == target_uid:
        return False
    return f"[#SRC:{source_uid}]" in answer and f"[#TGT:{target_uid}]" in answer


# ---------------------------------------------------------------------
# Citation-like token detection (for optional no-citations policy)
# ---------------------------------------------------------------------
_CITE_RE = re.compile(
    r"""(?ix)
    \b(?:rule|section|part|chapter|appendix|schedule)\b
    |
    \bFSMR\b
    |
    \b\d+(?:\.\d+)+(?:\([^)]+\))*\b
    """
)


def contains_citation_like_token(text: str) -> bool:
    if not text:
        return False
    return bool(_CITE_RE.search(text))


# ---------------------------------------------------------------------
# Structured citation leakage detection
# ---------------------------------------------------------------------

# Currency codes that share the ALL-CAPS format of regulatory corpus codes
# but must NOT be flagged as corpus references (e.g. "GBP 3.5").
_CURRENCY_CODES: frozenset[str] = frozenset(
    {
        "GBP", "USD", "EUR", "AED", "CHF", "JPY",
        "CAD", "AUD", "NZD", "SGD", "HKD", "SEK",
        "NOK", "DKK", "CNY", "INR", "BRL", "MXN",
    }
)

# Each tuple: (pattern_name, leakage_type, compiled_regex)
# Ordered from most specific to least specific.
# Design rules:
#   - Every pattern requires a leading keyword or ALL-CAPS acronym.
#   - Plain numbers, dates, percentages, and monetary values are never matched.
#   - part_ref / chapter_ref / schedule_ref require capital first letter to
#     avoid matching common lowercase prose ("part of...", "chapter heading").
_LEAKAGE_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    # Rule 3.2.1 / Rules 3.2.1 (requires dotted number with ≥2 levels)
    (
        "rule_ref",
        "rule_ref",
        re.compile(r"\bRules?\s+\d+(?:\.\d+)+(?:\s*\(\w+\))*", re.IGNORECASE),
    ),
    # Section 58 / Section 58(2) / section 3.2.1
    (
        "section_ref",
        "section_ref",
        re.compile(r"\bSections?\s+\d+(?:\.\d+)*(?:\s*\(\w+\))?", re.IGNORECASE),
    ),
    # Article 12 / Article 3(1)
    (
        "article_ref",
        "article_ref",
        re.compile(r"\bArticles?\s+\d+(?:\.\d+)*(?:\s*\(\w+\))?", re.IGNORECASE),
    ),
    # paragraph (3) / paragraph 3 / paragraphs 3.1
    (
        "paragraph_ref",
        "paragraph_ref",
        re.compile(
            r"\bparagraphs?\s*\(\d+\)|\bparagraphs?\s+\d+(?:\.\d+)*",
            re.IGNORECASE,
        ),
    ),
    # subsection 4 / subsection 3(a)
    (
        "subsection_ref",
        "subsection_ref",
        re.compile(r"\bsubsections?\s+\d+(?:\.\d+)*(?:\s*\(\w+\))?", re.IGNORECASE),
    ),
    # Part 2 / Part 3A — capital P only, avoids "part of something"
    (
        "part_ref",
        "part_ref",
        re.compile(r"\bParts?\s+\d+[A-Za-z]?\b"),
    ),
    # Chapter 5 — capital C
    (
        "chapter_ref",
        "chapter_ref",
        re.compile(r"\bChapters?\s+\d+(?:\.\d+)*\b"),
    ),
    # Schedule 1 — capital S
    (
        "schedule_ref",
        "schedule_ref",
        re.compile(r"\bSchedules?\s+\d+(?:\.\d+)*\b"),
    ),
    # Regulatory corpus codes: GEN 2.1.3 / COB 4.5.1 / AML 3.2.1
    # Requires ALL-CAPS acronym (2–6 letters) + dotted number (≥2 levels).
    # Currency codes (GBP, USD, …) are excluded via _CURRENCY_CODES.
    (
        "corpus_code",
        "corpus_code",
        re.compile(r"\b[A-Z]{2,6}\s+\d+\.\d+(?:\.\d+)*\b"),
    ),
]

_NORMALIZE_STRIP_RE = re.compile(r"[().,;:'\"/\\]")
_NORMALIZE_WS_RE = re.compile(r"[\s\-_]+")


def _normalize_for_ref_check(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for fuzzy reference matching."""
    text = _NORMALIZE_STRIP_RE.sub("", text.lower().strip())
    return _NORMALIZE_WS_RE.sub(" ", text).strip()


def detect_citation_leakage(
    question: str,
    reference_text: str | None = None,
) -> dict[str, Any]:
    """Detect explicit citation-like references in *question*.

    Looks for regulatory structural references (rule numbers, section IDs,
    article numbers, corpus codes such as ``GEN 2.1.3``) that would indicate
    the question leaks document-specific identifiers a retrieval system should
    discover, not assume.

    Conservative by design: requires a leading keyword or ALL-CAPS corpus
    acronym before any number. Plain numbers, dates (``28 April 2026``),
    percentages (``3.5%``), monetary values (``£12.50``), and quantity phrases
    (``2 years``, ``3 business days``) are **not** flagged.

    Args:
        question: The generated question text to analyse.
        reference_text: Optional raw reference string (e.g. a passage title or
            cross-reference ID). When provided, a normalised substring check is
            run to detect if the reference string itself was leaked verbatim
            into the question.

    Returns:
        A dict with four keys:

        * ``has_leakage`` (bool): ``True`` if any leakage was detected.
        * ``matched_patterns`` (list[str]): Internal pattern names that fired.
        * ``matched_spans`` (list[str]): Substrings of *question* that matched.
        * ``leakage_type`` (list[str]): Semantic category for each unique match
          type — one or more of: ``rule_ref``, ``section_ref``, ``article_ref``,
          ``paragraph_ref``, ``subsection_ref``, ``part_ref``, ``chapter_ref``,
          ``schedule_ref``, ``corpus_code``, ``reference_text``.
    """
    result: dict[str, Any] = {
        "has_leakage": False,
        "matched_patterns": [],
        "matched_spans": [],
        "leakage_type": [],
    }

    if not question or not question.strip():
        return result

    seen: set[tuple[str, str]] = set()

    for pattern_name, leak_type, regex in _LEAKAGE_PATTERNS:
        for m in regex.finditer(question):
            span = m.group(0)

            # False-positive guard: skip known currency codes that share the
            # ALL-CAPS format of corpus codes (e.g. "GBP 3.5").
            if pattern_name == "corpus_code":
                code = span.split()[0]
                if code in _CURRENCY_CODES:
                    continue

            key = (pattern_name, span)
            if key in seen:
                continue
            seen.add(key)

            result["matched_patterns"].append(pattern_name)
            result["matched_spans"].append(span)
            if leak_type not in result["leakage_type"]:
                result["leakage_type"].append(leak_type)

    # Optional: check whether reference_text (normalised) appears in question.
    if reference_text and reference_text.strip():
        norm_q = _normalize_for_ref_check(question)
        norm_ref = _normalize_for_ref_check(reference_text)
        if norm_ref and norm_ref in norm_q:
            result["matched_patterns"].append("reference_text_match")
            result["matched_spans"].append(reference_text[:120])
            if "reference_text" not in result["leakage_type"]:
                result["leakage_type"].append("reference_text")

    result["has_leakage"] = bool(result["matched_patterns"])
    return result


# ---------------------------------------------------------------------
# Answer span validation
# ---------------------------------------------------------------------
def span_valid(span: AnswerSpan, target_text: str) -> bool:
    """
    Validate that a span is an exact substring at [start,end) and type is allowed.
    """
    if span is None:
        return False
    if span.type not in SpanType:
        return False
    if not isinstance(span.start, int) or not isinstance(span.end, int):
        return False
    if span.start < 0 or span.end <= span.start or span.end > len(target_text or ""):
        return False
    return (target_text or "")[span.start : span.end] == (span.text or "")


# ---------------------------------------------------------------------
# QA validation
# ---------------------------------------------------------------------
@dataclass
class QAValidationResult:
    ok: bool
    errors: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors)}


def validate_qa_item(
    qa: QAItem,
    *,
    require_tags: bool = True,
    min_words: int = 50,
    max_words: int = 1000,
    no_citations: bool = False,
    reference_text: str | None = None,
) -> QAValidationResult:
    errs: list[str] = []

    if looks_like_empty(qa.question):
        errs.append("empty_question")
    if looks_like_empty(qa.expected_answer):
        errs.append("empty_expected_answer")

    # Ensure provenance pointers exist
    if not qa.pair_uid:
        errs.append("missing_pair_uid")
    if not qa.source_passage_uid or not qa.target_passage_uid:
        errs.append("missing_passage_uids")

    # Tags
    if require_tags:
        if not has_required_tags(
            qa.expected_answer or "", qa.source_passage_uid or "", qa.target_passage_uid or ""
        ):
            errs.append("missing_required_tags")

    # Word length bounds
    ans = norm_ws(qa.expected_answer or "")
    if ans:
        wc = len(ans.split())
        if wc < min_words:
            errs.append(f"answer_too_short:{wc}")
        if wc > max_words:
            errs.append(f"answer_too_long:{wc}")

    # No-citations policy
    if no_citations:
        if contains_citation_like_token(qa.question or ""):
            errs.append("citation_in_question")
        if contains_citation_like_token(qa.expected_answer or ""):
            errs.append("citation_in_answer")
        # Structured leakage check: detects specific citation patterns (rule IDs,
        # section numbers, corpus codes, etc.) in the question text.
        leakage = detect_citation_leakage(qa.question or "", reference_text=reference_text)
        if leakage["has_leakage"]:
            errs.append("question_contains_citation_leakage")

    return QAValidationResult(ok=(len(errs) == 0), errors=errs)


# ---------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------
@dataclass
class SchemaValidationResult:
    ok: bool
    errors: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors)}


def validate_schema_item(
    it: SchemaItem,
    *,
    require_hooks: bool = True,
    require_spans_if_not_title: bool = False,
    max_spans: int = 3,
) -> SchemaValidationResult:
    errs: list[str] = []

    if not it.pair_uid:
        errs.append("missing_pair_uid")
    if not it.source_passage_uid or not it.target_passage_uid:
        errs.append("missing_passage_uids")

    if looks_like_empty(it.source_text) or looks_like_empty(it.target_text):
        errs.append("empty_source_or_target_text")

    if require_hooks:
        if looks_like_empty(it.semantic_hook):
            errs.append("missing_semantic_hook")
        if looks_like_empty(it.citation_hook):
            # citation_hook may legitimately be empty for some corpora; keep as warning-level error
            errs.append("missing_citation_hook")

    # Spans
    spans = it.answer_spans or []
    if len(spans) > max_spans:
        errs.append(f"too_many_spans:{len(spans)}")

    if bool(it.target_is_title):
        if spans:
            errs.append("title_target_should_not_have_spans")
    else:
        if require_spans_if_not_title and not spans:
            errs.append("missing_spans_for_non_title_target")

    # Validate each span against target text
    for i, sp in enumerate(spans[:max_spans]):
        if not span_valid(sp, it.target_text or ""):
            errs.append(f"invalid_span:{i}")

    return SchemaValidationResult(ok=(len(errs) == 0), errors=errs)
