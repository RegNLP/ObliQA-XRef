# tests/test_citation_free_generation.py
"""
Tests for citation-free generation config feature:
- QAItem serialization includes citation leakage fields
- citation_leakage_action validation (keep/filter/separate)
- DPEL prompt contains no-citations-in-question instruction when enabled
- SCHEMA prompt contains no-citations-in-question instruction when enabled
- GenerateOverrides rejects invalid citation_leakage_action
- GenerationConfig rejects invalid citation_leakage_action
"""

from __future__ import annotations

import dataclasses

import pytest

from obliqaxref.generate.dpel.generate import DPELGenConfig, build_dpel_prompt
from obliqaxref.generate.schema.generate import SchemaGenConfig, build_schema_qa_prompt
from obliqaxref.generate.types import Method, Persona, QAItem, to_json, qa_item_from_json


# =============================================================================
# Helpers
# =============================================================================

def _make_qa(**overrides) -> QAItem:
    defaults = dict(
        qa_uid="abc123",
        method=Method.DPEL,
        persona=Persona.PROFESSIONAL,
        question="What obligation applies?",
        expected_answer="The obligation is X [#SRC:p1] and Y [#TGT:p2].",
        pair_uid="pair1",
        source_passage_uid="p1",
        target_passage_uid="p2",
        gen_model="gpt-4o",
        gen_ts=1714000000,
        run_seed=13,
    )
    defaults.update(overrides)
    return QAItem(**defaults)


# =============================================================================
# QAItem: new fields present with correct defaults
# =============================================================================

class TestQAItemNewFields:
    def test_citation_leakage_defaults_false(self):
        qa = _make_qa()
        assert qa.citation_leakage is False

    def test_citation_leakage_matches_defaults_empty(self):
        qa = _make_qa()
        assert qa.citation_leakage_matches == []

    def test_citation_leakage_types_defaults_empty(self):
        qa = _make_qa()
        assert qa.citation_leakage_types == []

    def test_can_set_leakage_true(self):
        qa = _make_qa(citation_leakage=True, citation_leakage_matches=["Rule 3.4"], citation_leakage_types=["rule_ref"])
        assert qa.citation_leakage is True
        assert qa.citation_leakage_matches == ["Rule 3.4"]
        assert qa.citation_leakage_types == ["rule_ref"]


# =============================================================================
# QAItem: serialization round-trip includes leakage fields
# =============================================================================

class TestQAItemSerialization:
    def test_to_json_includes_citation_leakage(self):
        qa = _make_qa()
        d = to_json(qa)
        assert "citation_leakage" in d
        assert d["citation_leakage"] is False

    def test_to_json_includes_citation_leakage_matches(self):
        qa = _make_qa()
        d = to_json(qa)
        assert "citation_leakage_matches" in d
        assert d["citation_leakage_matches"] == []

    def test_to_json_includes_citation_leakage_types(self):
        qa = _make_qa()
        d = to_json(qa)
        assert "citation_leakage_types" in d
        assert d["citation_leakage_types"] == []

    def test_to_json_leakage_true_serialized(self):
        qa = _make_qa(
            citation_leakage=True,
            citation_leakage_matches=["Section 42"],
            citation_leakage_types=["section_ref"],
        )
        d = to_json(qa)
        assert d["citation_leakage"] is True
        assert d["citation_leakage_matches"] == ["Section 42"]
        assert d["citation_leakage_types"] == ["section_ref"]

    def test_from_json_backward_compat_missing_leakage_fields(self):
        """Old records without leakage fields should deserialise with defaults."""
        old_record = {
            "qa_uid": "abc123",
            "method": "DPEL",
            "persona": "professional",
            "question": "What applies?",
            "expected_answer": "X [#SRC:p1] [#TGT:p2].",
            "pair_uid": "pair1",
            "source_passage_uid": "p1",
            "target_passage_uid": "p2",
            "gen_model": "gpt-4o",
            "gen_ts": 1714000000,
            "run_seed": 13,
        }
        qa = qa_item_from_json(old_record)
        assert qa.citation_leakage is False
        assert qa.citation_leakage_matches == []
        assert qa.citation_leakage_types == []

    def test_from_json_with_leakage_fields(self):
        record = {
            "qa_uid": "abc123",
            "method": "DPEL",
            "persona": "professional",
            "question": "What is Rule 3.4?",
            "expected_answer": "It requires X [#SRC:p1] [#TGT:p2].",
            "pair_uid": "pair1",
            "source_passage_uid": "p1",
            "target_passage_uid": "p2",
            "gen_model": "gpt-4o",
            "gen_ts": 1714000000,
            "run_seed": 13,
            "citation_leakage": True,
            "citation_leakage_matches": ["Rule 3.4"],
            "citation_leakage_types": ["rule_ref"],
        }
        qa = qa_item_from_json(record)
        assert qa.citation_leakage is True
        assert qa.citation_leakage_matches == ["Rule 3.4"]
        assert qa.citation_leakage_types == ["rule_ref"]

    def test_round_trip_preserves_leakage(self):
        qa = _make_qa(
            citation_leakage=True,
            citation_leakage_matches=["Article 12"],
            citation_leakage_types=["article_ref"],
        )
        qa2 = qa_item_from_json(to_json(qa))
        assert qa2.citation_leakage == qa.citation_leakage
        assert qa2.citation_leakage_matches == qa.citation_leakage_matches
        assert qa2.citation_leakage_types == qa.citation_leakage_types


# =============================================================================
# GenerateOverrides: citation_leakage_action validation
# =============================================================================

class TestGenerateOverridesValidation:
    def test_default_citation_leakage_action_is_keep(self):
        from obliqaxref.generate.run import GenerateOverrides
        o = GenerateOverrides()
        assert o.citation_leakage_action == "keep"

    def test_default_no_citations_in_question_is_true(self):
        from obliqaxref.generate.run import GenerateOverrides
        o = GenerateOverrides()
        assert o.no_citations_in_question is True

    def test_valid_action_filter(self):
        from obliqaxref.generate.run import GenerateOverrides
        o = GenerateOverrides(citation_leakage_action="filter")
        assert o.citation_leakage_action == "filter"

    def test_valid_action_separate(self):
        from obliqaxref.generate.run import GenerateOverrides
        o = GenerateOverrides(citation_leakage_action="separate")
        assert o.citation_leakage_action == "separate"

    def test_invalid_action_raises(self):
        from obliqaxref.generate.run import GenerateOverrides
        with pytest.raises(ValueError, match="citation_leakage_action"):
            GenerateOverrides(citation_leakage_action="drop")


# =============================================================================
# GenerationConfig: citation_leakage_action validation
# =============================================================================

class TestGenerationConfigValidation:
    def test_default_no_citations_in_question_is_true(self):
        from obliqaxref.config import GenerationConfig
        cfg = GenerationConfig()
        assert cfg.no_citations_in_question is True

    def test_default_citation_leakage_action_is_keep(self):
        from obliqaxref.config import GenerationConfig
        cfg = GenerationConfig()
        assert cfg.citation_leakage_action == "keep"

    def test_valid_actions(self):
        from obliqaxref.config import GenerationConfig
        for action in ("keep", "filter", "separate"):
            cfg = GenerationConfig(citation_leakage_action=action)
            assert cfg.citation_leakage_action == action

    def test_invalid_action_raises(self):
        from obliqaxref.config import GenerationConfig
        with pytest.raises(ValueError, match="citation_leakage_action"):
            GenerationConfig(citation_leakage_action="purge")


# =============================================================================
# DPEL prompt: no_citations_in_question clause injected when enabled
# =============================================================================

_NO_CITE_Q_MARKER = "NO-CITATIONS-IN-QUESTION POLICY"
_NO_CITE_FULL_MARKER = "NO-CITATIONS POLICY"


class TestDPELPromptNoCitationsInQuestion:
    def _prompt(self, **kwargs) -> str:
        defaults = dict(
            source_text="Source passage text.",
            target_text="Target passage text.",
            source_uid="src1",
            target_uid="tgt1",
            max_per_persona=2,
            sample_n=3,
        )
        defaults.update(kwargs)
        return build_dpel_prompt(**defaults)

    def test_no_clause_by_default(self):
        p = self._prompt()
        assert _NO_CITE_Q_MARKER not in p
        assert _NO_CITE_FULL_MARKER not in p

    def test_question_only_clause_when_no_citations_in_question_true(self):
        p = self._prompt(no_citations_in_question=True)
        assert _NO_CITE_Q_MARKER in p

    def test_question_only_clause_not_present_when_false(self):
        p = self._prompt(no_citations_in_question=False)
        assert _NO_CITE_Q_MARKER not in p

    def test_full_clause_when_no_citations_true(self):
        p = self._prompt(no_citations=True)
        assert _NO_CITE_FULL_MARKER in p
        assert _NO_CITE_Q_MARKER not in p  # full clause takes priority

    def test_no_citations_takes_priority_over_question_only(self):
        """When both flags are True, no_citations (full) takes priority."""
        p = self._prompt(no_citations=True, no_citations_in_question=True)
        assert _NO_CITE_FULL_MARKER in p
        assert _NO_CITE_Q_MARKER not in p

    def test_question_only_clause_mentions_answer_still_required(self):
        p = self._prompt(no_citations_in_question=True)
        assert "ANSWER evidence tags" in p or "ANSWER" in p

    def test_question_only_clause_mentions_question_restriction(self):
        p = self._prompt(no_citations_in_question=True)
        assert "QUESTION" in p

    def test_question_only_clause_mentions_rule_numbers(self):
        p = self._prompt(no_citations_in_question=True)
        assert "rule numbers" in p.lower() or "rule" in p.lower()


# =============================================================================
# SCHEMA prompt: no_citations_in_question clause injected when enabled
# =============================================================================

class TestSCHEMAPromptNoCitationsInQuestion:
    def _prompt(self, **kwargs) -> str:
        from obliqaxref.generate.schema.extract import AnswerSpan
        defaults = dict(
            source_text="Source passage text.",
            target_text="Target passage text.",
            source_uid="src1",
            target_uid="tgt1",
            semantic_hook="compliance obligation",
            citation_hook="Rule 3.4",
            source_item_type="Obligation",
            target_item_type="Procedure",
            answer_spans=[],
            max_per_persona=2,
            sample_n=3,
        )
        defaults.update(kwargs)
        return build_schema_qa_prompt(**defaults)

    def test_no_clause_by_default(self):
        # With both flags explicitly off, neither citation-policy clause should appear
        p = self._prompt(no_citations=False, no_citations_in_question=False)
        assert _NO_CITE_Q_MARKER not in p
        assert _NO_CITE_FULL_MARKER not in p

    def test_full_clause_present_by_default(self):
        # no_citations=True is the new default → STRICT clause must be baked in
        p = self._prompt()
        assert _NO_CITE_FULL_MARKER in p

    def test_question_only_clause_when_no_citations_in_question_true(self):
        # Isolate the Q-only path by disabling the full no_citations flag
        p = self._prompt(no_citations=False, no_citations_in_question=True)
        assert _NO_CITE_Q_MARKER in p

    def test_question_only_clause_not_present_when_false(self):
        p = self._prompt(no_citations_in_question=False)
        assert _NO_CITE_Q_MARKER not in p

    def test_full_clause_when_no_citations_true(self):
        p = self._prompt(no_citations=True)
        assert _NO_CITE_FULL_MARKER in p
        assert _NO_CITE_Q_MARKER not in p

    def test_full_takes_priority(self):
        p = self._prompt(no_citations=True, no_citations_in_question=True)
        assert _NO_CITE_FULL_MARKER in p
        assert _NO_CITE_Q_MARKER not in p

    def test_question_only_clause_question_restriction_mentioned(self):
        p = self._prompt(no_citations_in_question=True)
        assert "QUESTION" in p

    def test_question_only_clause_answer_tags_still_required(self):
        p = self._prompt(no_citations_in_question=True)
        assert "ANSWER" in p


# =============================================================================
# DPELGenConfig: no_citations_in_question field
# =============================================================================

class TestDPELGenConfigNoCitationsInQuestion:
    def test_default_is_false(self):
        cfg = DPELGenConfig(model="gpt-4o")
        assert cfg.no_citations_in_question is False

    def test_can_set_true(self):
        cfg = DPELGenConfig(model="gpt-4o", no_citations_in_question=True)
        assert cfg.no_citations_in_question is True


# =============================================================================
# SchemaGenConfig: no_citations_in_question field
# =============================================================================

class TestSchemaGenConfigNoCitationsInQuestion:
    def test_default_is_false(self):
        cfg = SchemaGenConfig(model="gpt-4o")
        assert cfg.no_citations_in_question is False

    def test_can_set_true(self):
        cfg = SchemaGenConfig(model="gpt-4o", no_citations_in_question=True)
        assert cfg.no_citations_in_question is True

    def test_no_citations_default_is_true(self):
        cfg = SchemaGenConfig(model="gpt-4o")
        assert cfg.no_citations is True

    def test_dual_anchors_mode_default_is_always(self):
        cfg = SchemaGenConfig(model="gpt-4o")
        assert cfg.dual_anchors_mode == "always"

    def test_dual_anchors_mode_can_be_changed(self):
        cfg = SchemaGenConfig(model="gpt-4o", dual_anchors_mode="off")
        assert cfg.dual_anchors_mode == "off"


# =============================================================================
# SCHEMA prompt: dual_anchors_mode rule injected correctly
# =============================================================================

class TestSCHEMAPromptDualAnchorsMode:
    def _prompt(self, **kwargs) -> str:
        from obliqaxref.generate.schema.extract import AnswerSpan
        defaults = dict(
            source_text="Source passage text.",
            target_text="Target passage text.",
            source_uid="src1",
            target_uid="tgt1",
            semantic_hook="compliance obligation",
            citation_hook="Rule 3.4",
            source_item_type="Obligation",
            target_item_type="Procedure",
            answer_spans=[],
            max_per_persona=2,
            sample_n=3,
        )
        defaults.update(kwargs)
        return build_schema_qa_prompt(**defaults)

    def test_dual_anchor_hard_rule_present_when_always(self):
        p = self._prompt(dual_anchors_mode="always")
        assert "DUAL-ANCHOR REQUIREMENT" in p

    def test_dual_anchor_hard_rule_absent_when_off(self):
        p = self._prompt(dual_anchors_mode="off")
        assert "DUAL-ANCHOR REQUIREMENT" not in p

    def test_dual_anchor_default_is_always(self):
        p = self._prompt()
        assert "DUAL-ANCHOR REQUIREMENT" in p

    def test_dual_anchor_always_mentions_source_and_target(self):
        p = self._prompt(dual_anchors_mode="always")
        assert "SOURCE" in p and "TARGET" in p

    def test_dual_anchor_always_mentions_abort(self):
        p = self._prompt(dual_anchors_mode="always")
        assert "empty array" in p


# =============================================================================
# dataclasses.replace compatibility (frozen QAItem)
# =============================================================================

class TestQAItemReplaceForAnnotation:
    def test_replace_sets_leakage_fields(self):
        qa = _make_qa()
        annotated = dataclasses.replace(
            qa,
            citation_leakage=True,
            citation_leakage_matches=["Rule 3"],
            citation_leakage_types=["rule_ref"],
        )
        assert annotated.citation_leakage is True
        assert annotated.citation_leakage_matches == ["Rule 3"]
        assert annotated.citation_leakage_types == ["rule_ref"]
        # Original unchanged
        assert qa.citation_leakage is False

    def test_replace_preserves_other_fields(self):
        qa = _make_qa()
        annotated = dataclasses.replace(qa, citation_leakage=True)
        assert annotated.qa_uid == qa.qa_uid
        assert annotated.question == qa.question
        assert annotated.expected_answer == qa.expected_answer
