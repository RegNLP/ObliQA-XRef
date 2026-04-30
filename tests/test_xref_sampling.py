# tests/test_xref_sampling.py
"""
Unit tests for difficulty-aware cross-reference row sampling.

Covers:
- _tokenize(): stopword removal, lowercase, multi-char filter
- _jaccard(): edge cases and known values
- extract_xref_features(): all 13 boolean/numeric features
- build_feature_table(): bulk extraction, count lookups
- sample_xref_rows() for every mode
- Backward-compatibility of random mode
- SamplingConfig defaults and validation
- Report structure from sample_xref_rows()
- GenerateOverrides.sampling_mode field
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from obliqaxref.generate.common.sampling import (
    DEFAULT_BUCKET_WEIGHTS,
    VALID_MODES,
    _jaccard,
    _tokenize,
    build_feature_table,
    extract_xref_features,
    sample_xref_rows,
)


# =============================================================================
# Fixtures
# =============================================================================

def _make_row(
    source: str = "The firm must comply with rule 3.1 and section 4.2.",
    target: str = "Outsourcing means the arrangement under which a service is provided.",
    ref: str = "3.1",
    src_id: str = "src-A",
    tgt_id: str = "tgt-A",
) -> dict[str, Any]:
    return {
        "SourceID": src_id,
        "TargetID": tgt_id,
        "ReferenceText": ref,
        "ReferenceType": "External",
        "SourceDocumentID": "1",
        "TargetDocumentID": "2",
        "SourcePassageID": "1.1",
        "TargetPassageID": "2.1",
        "SourcePassage": source,
        "TargetPassage": target,
    }


def _make_rows(n: int = 20) -> list[dict[str, Any]]:
    """Make a small set of synthetic rows with varied characteristics."""
    rows = []
    for i in range(n):
        src = f"The authorised person must comply with the obligation set out in section {i}. "
        src += "This applies to all regulated activities and must be observed at all times."
        if i % 3 == 0:
            tgt = "Outsourcing means the arrangement whereby an external party provides services."
        elif i % 3 == 1:
            tgt = (
                f"Unless otherwise specified, the firm shall notify the regulator within 30 days. "
                f"At least 20% of transactions must meet the threshold. Subject to rule {i}."
            )
        else:
            tgt = (
                f"Capital requirement: the firm must hold at least 8 percent of risk-weighted assets. "
                f"The minimum requirement shall not fall below the threshold defined in section {i}."
            )
        rows.append(_make_row(source=src, target=tgt, src_id=f"src-{i}", tgt_id=f"tgt-{i}"))
    # Make first min(5, n) rows share the same SourceID (multi-ref source)
    for i in range(min(5, n)):
        rows[i]["SourceID"] = "src-MULTI"
    return rows


# =============================================================================
# _tokenize
# =============================================================================

class TestTokenize:
    def test_returns_frozenset(self):
        result = _tokenize("hello world")
        assert isinstance(result, frozenset)

    def test_lowercase(self):
        result = _tokenize("HELLO World")
        assert "hello" in result
        assert "world" in result

    def test_removes_stopwords(self):
        result = _tokenize("the firm and the authority")
        assert "the" not in result
        assert "and" not in result
        assert "firm" in result

    def test_removes_single_char(self):
        result = _tokenize("a b c firm")
        assert "a" not in result
        assert "b" not in result
        assert "firm" in result

    def test_empty_string(self):
        assert _tokenize("") == frozenset()

    def test_only_stopwords_and_punctuation(self):
        result = _tokenize("the and or of")
        assert len(result) == 0

    def test_non_alpha_excluded(self):
        result = _tokenize("firm-123 obligation")
        assert "firm" in result
        assert "obligation" in result
        assert "123" not in result


# =============================================================================
# _jaccard
# =============================================================================

class TestJaccard:
    def test_identical_sets(self):
        a = frozenset(["firm", "must", "comply"])
        assert _jaccard(a, a) == pytest.approx(1.0)

    def test_disjoint_sets(self):
        a = frozenset(["firm"])
        b = frozenset(["regulator"])
        assert _jaccard(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = frozenset(["firm", "comply", "obligation"])
        b = frozenset(["firm", "comply", "threshold"])
        # intersection=2, union=4
        assert _jaccard(a, b) == pytest.approx(2 / 4)

    def test_both_empty(self):
        assert _jaccard(frozenset(), frozenset()) == pytest.approx(0.0)

    def test_one_empty(self):
        assert _jaccard(frozenset(["firm"]), frozenset()) == pytest.approx(0.0)


# =============================================================================
# extract_xref_features
# =============================================================================

class TestExtractXrefFeatures:
    def _features(self, row: dict, outgoing: dict | None = None, incoming: dict | None = None):
        out = outgoing or {row["SourceID"]: 1}
        inc = incoming or {row["TargetID"]: 1}
        return extract_xref_features(row, outgoing_counts=out, incoming_counts=inc)

    def test_source_length_words(self):
        row = _make_row(source="one two three four five")
        feats = self._features(row)
        assert feats["source_length_words"] == 5

    def test_target_length_words(self):
        row = _make_row(target="word1 word2")
        feats = self._features(row)
        assert feats["target_length_words"] == 2

    def test_jaccard_aliases_are_equal(self):
        row = _make_row()
        feats = self._features(row)
        assert feats["source_target_lexical_overlap"] == feats["source_target_jaccard_similarity"]

    def test_high_overlap_when_same_text(self):
        text = "the firm must comply with the outsourcing obligation at all times"
        row = _make_row(source=text, target=text)
        feats = self._features(row)
        assert feats["source_target_jaccard_similarity"] > 0.8

    def test_low_overlap_when_different_text(self):
        row = _make_row(
            source="The authorised person must submit a capital return to the regulator.",
            target="Outsourcing means the arrangement under which an external party provides services.",
        )
        feats = extract_xref_features(
            row,
            outgoing_counts={row["SourceID"]: 1},
            incoming_counts={row["TargetID"]: 1},
            max_jaccard_for_low_overlap=0.15,
        )
        assert feats["source_target_jaccard_similarity"] < 0.15
        assert feats["low_source_target_overlap"] is True

    def test_high_overlap_not_flagged_as_low(self):
        text = "capital threshold requirement regulator firm obligation reporting"
        row = _make_row(source=text, target=text)
        feats = extract_xref_features(
            row,
            outgoing_counts={row["SourceID"]: 1},
            incoming_counts={row["TargetID"]: 1},
            max_jaccard_for_low_overlap=0.15,
        )
        assert feats["low_source_target_overlap"] is False

    def test_outgoing_count_from_precomputed(self):
        row = _make_row(src_id="src-X")
        feats = self._features(row, outgoing={"src-X": 7})
        assert feats["number_of_outgoing_refs_from_source"] == 7
        assert feats["source_contains_multiple_refs"] is True

    def test_single_outgoing_not_multi(self):
        row = _make_row(src_id="src-Y")
        feats = self._features(row, outgoing={"src-Y": 1})
        assert feats["source_contains_multiple_refs"] is False

    def test_incoming_count_from_precomputed(self):
        row = _make_row(tgt_id="tgt-Z")
        feats = self._features(row, incoming={"tgt-Z": 4})
        assert feats["number_of_incoming_refs_to_target"] == 4

    def test_reference_text_present_in_source(self):
        row = _make_row(source="See rule 3.1 for obligations.", ref="3.1")
        feats = self._features(row)
        assert feats["reference_text_present_in_source"] is True

    def test_reference_text_absent_from_source(self):
        row = _make_row(source="General obligations apply.", ref="99.9")
        feats = self._features(row)
        assert feats["reference_text_present_in_source"] is False

    def test_definition_cue_detected(self):
        row = _make_row(target="Outsourcing means the arrangement whereby a third party provides services.")
        feats = self._features(row)
        assert feats["target_contains_definition_like_text"] is True

    def test_definition_cue_absent(self):
        row = _make_row(target="The firm must hold adequate capital at all times.")
        feats = self._features(row)
        assert feats["target_contains_definition_like_text"] is False

    def test_duration_cue_detected(self):
        row = _make_row(target="The firm must notify the regulator within 30 days of the event.")
        feats = self._features(row)
        assert feats["target_contains_duration_or_deadline"] is True

    def test_threshold_cue_detected(self):
        row = _make_row(target="Capital ratio must be at least 8 percent of risk-weighted assets.")
        feats = self._features(row)
        assert feats["target_contains_threshold_or_numeric_value"] is True

    def test_exception_cue_detected(self):
        row = _make_row(target="Unless otherwise agreed, the provision applies to all transactions.")
        feats = self._features(row)
        assert feats["target_contains_exception_or_condition"] is True

    def test_condition_cue_detected(self):
        row = _make_row(target="Subject to the approval of the board, the firm may proceed.")
        feats = self._features(row)
        assert feats["target_contains_exception_or_condition"] is True

    def test_empty_passages_do_not_crash(self):
        row = _make_row(source="", target="")
        feats = self._features(row)
        assert feats["source_length_words"] == 0
        assert feats["target_length_words"] == 0
        assert feats["source_target_jaccard_similarity"] == 0.0


# =============================================================================
# build_feature_table
# =============================================================================

class TestBuildFeatureTable:
    def test_returns_same_length(self):
        rows = _make_rows(10)
        enriched = build_feature_table(rows)
        assert len(enriched) == 10

    def test_original_rows_not_mutated(self):
        rows = _make_rows(5)
        originals = [dict(r) for r in rows]
        build_feature_table(rows)
        for orig, row in zip(originals, rows):
            assert orig == row

    def test_features_present_in_enriched(self):
        rows = _make_rows(3)
        enriched = build_feature_table(rows)
        for r in enriched:
            assert "source_length_words" in r
            assert "low_source_target_overlap" in r
            assert "source_contains_multiple_refs" in r

    def test_csv_keys_preserved(self):
        rows = _make_rows(2)
        enriched = build_feature_table(rows)
        for r in enriched:
            assert "SourceID" in r
            assert "TargetPassage" in r

    def test_multi_ref_counts_correct(self):
        # src-MULTI appears in first 5 rows (from _make_rows fixture)
        rows = _make_rows(10)
        enriched = build_feature_table(rows)
        for r in enriched[:5]:
            assert r["number_of_outgoing_refs_from_source"] == 5
            assert r["source_contains_multiple_refs"] is True


# =============================================================================
# sample_xref_rows — basic properties
# =============================================================================

class TestSampleXrefRowsBasic:
    def test_invalid_mode_raises(self):
        rows = _make_rows(10)
        with pytest.raises(ValueError, match="sampling_mode"):
            sample_xref_rows(rows, n=5, mode="invalid_mode")

    def test_returns_two_values(self):
        rows = _make_rows(10)
        result = sample_xref_rows(rows, n=5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_features_in_sampled_rows(self):
        rows = _make_rows(10)
        sampled, _ = sample_xref_rows(rows, n=5)
        for r in sampled:
            assert "source_length_words" in r
            assert "low_source_target_overlap" in r

    def test_n_zero_returns_all(self):
        rows = _make_rows(10)
        sampled, report = sample_xref_rows(rows, n=0)
        assert len(sampled) == 10
        assert report["no_sampling_applied"] is True

    def test_n_exceeds_len_returns_all(self):
        rows = _make_rows(10)
        sampled, report = sample_xref_rows(rows, n=999)
        assert len(sampled) == 10
        assert report["no_sampling_applied"] is True

    def test_report_has_required_keys(self):
        rows = _make_rows(20)
        _, report = sample_xref_rows(rows, n=10)
        for key in ("sampling_mode", "n_candidates", "n_sampled", "n_requested", "seed",
                    "no_sampling_applied", "bucket_counts", "feature_summary"):
            assert key in report, f"Missing key: {key}"

    def test_report_feature_summary_has_stats(self):
        rows = _make_rows(20)
        _, report = sample_xref_rows(rows, n=10)
        fs = report["feature_summary"]
        assert "source_length_words" in fs
        assert "source_target_jaccard_similarity" in fs
        assert "n_low_overlap" in fs

    def test_valid_modes_accepted(self):
        rows = _make_rows(20)
        for mode in VALID_MODES:
            sampled, report = sample_xref_rows(rows, n=10, mode=mode)
            assert len(sampled) == 10
            assert report["sampling_mode"] == mode


# =============================================================================
# sample_xref_rows — random mode (backward-compatibility)
# =============================================================================

class TestSampleRandom:
    def test_returns_n_rows(self):
        rows = _make_rows(20)
        sampled, _ = sample_xref_rows(rows, n=10, mode="random", seed=13)
        assert len(sampled) == 10

    def test_same_seed_same_result(self):
        rows = _make_rows(20)
        s1, _ = sample_xref_rows(rows, n=10, mode="random", seed=42)
        s2, _ = sample_xref_rows(rows, n=10, mode="random", seed=42)
        ids1 = [(r["SourceID"], r["TargetID"]) for r in s1]
        ids2 = [(r["SourceID"], r["TargetID"]) for r in s2]
        assert ids1 == ids2

    def test_different_seeds_different_results(self):
        rows = _make_rows(20)
        s1, _ = sample_xref_rows(rows, n=10, mode="random", seed=1)
        s2, _ = sample_xref_rows(rows, n=10, mode="random", seed=2)
        ids1 = {(r["SourceID"], r["TargetID"]) for r in s1}
        ids2 = {(r["SourceID"], r["TargetID"]) for r in s2}
        assert ids1 != ids2

    def test_bucket_counts_key_is_random(self):
        rows = _make_rows(20)
        _, report = sample_xref_rows(rows, n=10, mode="random", seed=13)
        assert "random" in report["bucket_counts"]

    def test_no_duplicates_in_sample(self):
        rows = _make_rows(20)
        sampled, _ = sample_xref_rows(rows, n=15, mode="random", seed=7)
        ids = [(r["SourceID"], r["TargetID"]) for r in sampled]
        assert len(ids) == len(set(ids))


# =============================================================================
# sample_xref_rows — low_overlap mode
# =============================================================================

class TestSampleLowOverlap:
    def test_returns_n_rows(self):
        rows = _make_rows(20)
        sampled, _ = sample_xref_rows(rows, n=8, mode="low_overlap", seed=13)
        assert len(sampled) == 8

    def test_selected_rows_have_lower_overlap_than_population_mean(self):
        rows = _make_rows(20)
        enriched = build_feature_table(rows)
        pop_mean = sum(r["source_target_jaccard_similarity"] for r in enriched) / len(enriched)
        sampled, _ = sample_xref_rows(rows, n=5, mode="low_overlap", seed=13)
        sel_mean = sum(r["source_target_jaccard_similarity"] for r in sampled) / len(sampled)
        assert sel_mean <= pop_mean + 0.05  # sampled overlap should be <= population mean

    def test_bucket_counts_key_is_low_overlap(self):
        rows = _make_rows(20)
        _, report = sample_xref_rows(rows, n=8, mode="low_overlap")
        assert "low_overlap" in report["bucket_counts"]


# =============================================================================
# sample_xref_rows — multi_ref_source mode
# =============================================================================

class TestSampleMultiRefSource:
    def test_returns_n_rows(self):
        rows = _make_rows(20)
        sampled, _ = sample_xref_rows(rows, n=8, mode="multi_ref_source", seed=13)
        assert len(sampled) == 8

    def test_prefers_multi_ref_rows(self):
        rows = _make_rows(20)
        # First 5 rows share SourceID "src-MULTI" (set in _make_rows)
        sampled, report = sample_xref_rows(rows, n=5, mode="multi_ref_source", seed=13)
        # Since there are exactly 5 multi-ref rows and we request 5, all should be multi-ref
        assert report["bucket_counts"].get("multi_ref_source", 0) == 5

    def test_fills_with_random_when_insufficient(self):
        # Only 5 multi-ref rows, request 10 → 5 multi-ref + 5 random
        rows = _make_rows(20)
        sampled, report = sample_xref_rows(rows, n=10, mode="multi_ref_source", seed=13)
        assert len(sampled) == 10
        assert report["bucket_counts"].get("multi_ref_source", 0) == 5
        assert report["bucket_counts"].get("random", 0) == 5


# =============================================================================
# sample_xref_rows — target_definition_or_condition mode
# =============================================================================

class TestSampleTargetDefinitionOrCondition:
    def test_returns_n_rows(self):
        rows = _make_rows(20)
        sampled, _ = sample_xref_rows(rows, n=8, mode="target_definition_or_condition", seed=13)
        assert len(sampled) == 8

    def test_bucket_key_present(self):
        rows = _make_rows(20)
        _, report = sample_xref_rows(rows, n=8, mode="target_definition_or_condition")
        assert "target_definition_or_condition" in report["bucket_counts"]

    def test_all_matching_selected_when_abundant(self):
        """When more matching rows than n, all sampled should be from the matching set."""
        # Create rows where every target has a definition cue
        rows = []
        for i in range(10):
            rows.append(
                _make_row(
                    target=f"Obligation {i} means the duty to comply with the requirements.",
                    src_id=f"src-{i}", tgt_id=f"tgt-{i}",
                )
            )
        sampled, report = sample_xref_rows(rows, n=5, mode="target_definition_or_condition", seed=1)
        assert len(sampled) == 5
        assert report["bucket_counts"].get("target_definition_or_condition", 0) == 5


# =============================================================================
# sample_xref_rows — mixed_difficulty mode
# =============================================================================

class TestSampleMixedDifficulty:
    def test_returns_n_rows(self):
        rows = _make_rows(20)
        sampled, _ = sample_xref_rows(rows, n=12, mode="mixed_difficulty", seed=13)
        assert len(sampled) == 12

    def test_no_duplicate_rows(self):
        rows = _make_rows(20)
        sampled, _ = sample_xref_rows(rows, n=12, mode="mixed_difficulty", seed=13)
        ids = [(r["SourceID"], r["TargetID"]) for r in sampled]
        assert len(ids) == len(set(ids))

    def test_report_has_all_bucket_keys(self):
        rows = _make_rows(20)
        _, report = sample_xref_rows(rows, n=12, mode="mixed_difficulty", seed=13)
        bc = report["bucket_counts"]
        for key in ("low_overlap", "multi_ref", "definition_condition", "random"):
            assert key in bc

    def test_custom_bucket_weights(self):
        rows = _make_rows(20)
        weights = {"low_overlap": 0.5, "multi_ref": 0.5, "definition_condition": 0.0, "random": 0.0}
        sampled, _ = sample_xref_rows(rows, n=10, mode="mixed_difficulty", seed=1,
                                       mixed_difficulty_bucket_weights=weights)
        assert len(sampled) <= 10

    def test_total_sampled_matches_n(self):
        rows = _make_rows(30)
        sampled, report = sample_xref_rows(rows, n=20, mode="mixed_difficulty", seed=42)
        assert len(sampled) == report["n_sampled"]
        assert report["n_sampled"] == 20


# =============================================================================
# SamplingConfig
# =============================================================================

class TestSamplingConfig:
    def test_defaults(self):
        from obliqaxref.config import SamplingConfig
        sc = SamplingConfig()
        assert sc.sampling_mode == "random"
        assert sc.max_jaccard_for_low_overlap == pytest.approx(0.15)
        assert "low_overlap" in sc.mixed_difficulty_bucket_weights
        assert "multi_ref" in sc.mixed_difficulty_bucket_weights

    def test_invalid_sampling_mode_raises(self):
        from obliqaxref.config import SamplingConfig
        with pytest.raises(Exception):
            SamplingConfig(sampling_mode="bogus")

    def test_all_valid_modes_accepted(self):
        from obliqaxref.config import SamplingConfig
        for mode in VALID_MODES:
            sc = SamplingConfig(sampling_mode=mode)
            assert sc.sampling_mode == mode

    def test_run_config_has_sampling_field(self):
        from obliqaxref.config import RunConfig, SamplingConfig
        cfg = RunConfig.model_validate({
            "run_id": "test",
            "paths": {"input_dir": "/in", "work_dir": "/work", "output_dir": "/out"},
            "adapter": {"corpus": "ukfin"},
        })
        assert isinstance(cfg.sampling, SamplingConfig)
        assert cfg.sampling.sampling_mode == "random"

    def test_sampling_mode_propagated_from_yaml(self):
        from obliqaxref.config import RunConfig
        cfg = RunConfig.model_validate({
            "run_id": "test",
            "paths": {"input_dir": "/in", "work_dir": "/work", "output_dir": "/out"},
            "adapter": {"corpus": "ukfin"},
            "sampling": {"sampling_mode": "low_overlap"},
        })
        assert cfg.sampling.sampling_mode == "low_overlap"


# =============================================================================
# GenerateOverrides.sampling_mode
# =============================================================================

class TestGenerateOverridesSamplingMode:
    def test_default_is_none(self):
        from obliqaxref.generate.run import GenerateOverrides
        o = GenerateOverrides()
        assert o.sampling_mode is None

    def test_valid_mode_accepted(self):
        from obliqaxref.generate.run import GenerateOverrides
        for mode in VALID_MODES:
            o = GenerateOverrides(sampling_mode=mode)
            assert o.sampling_mode == mode

    def test_invalid_mode_raises(self):
        from obliqaxref.generate.run import GenerateOverrides
        with pytest.raises(ValueError, match="sampling_mode"):
            GenerateOverrides(sampling_mode="nonexistent")
