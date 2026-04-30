# tests/test_ir_difficulty_label.py
"""
Unit tests for IR difficulty labelling and detailed vote computation.

Covers:
- assign_ir_difficulty_label(): all 6 label paths
- compute_detailed_votes(): vote detail aggregation
- Edge cases: empty runs, single retriever, ties
"""

from __future__ import annotations

import pytest

from obliqaxref.curate.run import assign_ir_difficulty_label, compute_detailed_votes


# =============================================================================
# Helpers
# =============================================================================

def _make_item(item_id: str = "q1", src: str = "src1", tgt: str = "tgt1") -> dict:
    return {
        "item_id": item_id,
        "source_passage_id": src,
        "target_passage_id": tgt,
    }


def _make_runs(*retriever_configs: tuple[str, list[str]]) -> dict:
    """
    Build a runs dict from (run_name, retrieved_ids) tuples.
    Each retrieved_ids list is stored under item_id "q1".
    """
    return {name: {"q1": pids} for name, pids in retriever_configs}


# =============================================================================
# assign_ir_difficulty_label
# =============================================================================

class TestAssignIrDifficultyLabel:

    # --- easy (majority got both) ---

    def test_easy_all_retrievers_got_both(self):
        assert assign_ir_difficulty_label(
            source_vote_count=3,
            target_vote_count=3,
            both_vote_count=3,
            num_retrievers=3,
        ) == "easy"

    def test_easy_majority_got_both(self):
        # 3 out of 4 > 50%
        assert assign_ir_difficulty_label(
            source_vote_count=3,
            target_vote_count=3,
            both_vote_count=3,
            num_retrievers=4,
        ) == "easy"

    def test_easy_exact_majority_threshold_is_strict(self):
        # 2 out of 4 = 50%, not > 50% → medium
        assert assign_ir_difficulty_label(
            source_vote_count=2,
            target_vote_count=2,
            both_vote_count=2,
            num_retrievers=4,
        ) == "medium"

    # --- medium (at least one got both, not majority) ---

    def test_medium_one_of_three_got_both(self):
        assert assign_ir_difficulty_label(
            source_vote_count=2,
            target_vote_count=1,
            both_vote_count=1,
            num_retrievers=3,
        ) == "medium"

    def test_medium_single_retriever_got_both(self):
        assert assign_ir_difficulty_label(
            source_vote_count=1,
            target_vote_count=1,
            both_vote_count=1,
            num_retrievers=5,
        ) == "medium"

    # --- hard (no retriever got both, but each got one) ---

    def test_hard_no_both_but_each_retrieved_individually(self):
        assert assign_ir_difficulty_label(
            source_vote_count=2,
            target_vote_count=3,
            both_vote_count=0,
            num_retrievers=5,
        ) == "hard"

    def test_hard_source_and_target_each_retrieved_by_one(self):
        assert assign_ir_difficulty_label(
            source_vote_count=1,
            target_vote_count=1,
            both_vote_count=0,
            num_retrievers=3,
        ) == "hard"

    # --- source_only ---

    def test_source_only_target_never_retrieved(self):
        assert assign_ir_difficulty_label(
            source_vote_count=4,
            target_vote_count=0,
            both_vote_count=0,
            num_retrievers=4,
        ) == "source_only"

    def test_source_only_single_retriever(self):
        assert assign_ir_difficulty_label(
            source_vote_count=1,
            target_vote_count=0,
            both_vote_count=0,
            num_retrievers=1,
        ) == "source_only"

    # --- target_only ---

    def test_target_only_source_never_retrieved(self):
        assert assign_ir_difficulty_label(
            source_vote_count=0,
            target_vote_count=2,
            both_vote_count=0,
            num_retrievers=3,
        ) == "target_only"

    # --- neither ---

    def test_neither_nothing_retrieved(self):
        assert assign_ir_difficulty_label(
            source_vote_count=0,
            target_vote_count=0,
            both_vote_count=0,
            num_retrievers=5,
        ) == "neither"

    def test_neither_no_retrievers(self):
        # Zero runs — all zeros → neither
        assert assign_ir_difficulty_label(
            source_vote_count=0,
            target_vote_count=0,
            both_vote_count=0,
            num_retrievers=0,
        ) == "neither"

    # --- boundary: single retriever ---

    def test_single_retriever_easy_when_both_found(self):
        # 1 out of 1 > 50% → easy
        assert assign_ir_difficulty_label(
            source_vote_count=1,
            target_vote_count=1,
            both_vote_count=1,
            num_retrievers=1,
        ) == "easy"


# =============================================================================
# compute_detailed_votes
# =============================================================================

class TestComputeDetailedVotes:

    def test_all_retrievers_got_both(self):
        item = _make_item()
        runs = _make_runs(
            ("bm25", ["src1", "tgt1", "other"]),
            ("dense", ["src1", "tgt1"]),
        )
        result = compute_detailed_votes(item, runs)

        assert result["source_vote_count"] == 2
        assert result["target_vote_count"] == 2
        assert result["both_vote_count"] == 2
        assert set(result["retrievers_recovering_source"]) == {"bm25", "dense"}
        assert set(result["retrievers_recovering_target"]) == {"bm25", "dense"}
        assert set(result["retrievers_recovering_both"]) == {"bm25", "dense"}

    def test_source_only_retrieval(self):
        item = _make_item()
        runs = _make_runs(
            ("bm25", ["src1", "other_a", "other_b"]),
            ("dense", ["other_c"]),
        )
        result = compute_detailed_votes(item, runs)

        assert result["source_vote_count"] == 1
        assert result["target_vote_count"] == 0
        assert result["both_vote_count"] == 0
        assert result["retrievers_recovering_source"] == ["bm25"]
        assert result["retrievers_recovering_target"] == []
        assert result["retrievers_recovering_both"] == []

    def test_target_only_retrieval(self):
        item = _make_item()
        runs = _make_runs(
            ("bm25", ["other_a"]),
            ("e5", ["tgt1"]),
        )
        result = compute_detailed_votes(item, runs)

        assert result["source_vote_count"] == 0
        assert result["target_vote_count"] == 1
        assert result["both_vote_count"] == 0
        assert result["retrievers_recovering_target"] == ["e5"]

    def test_no_retrieval(self):
        item = _make_item()
        runs = _make_runs(
            ("bm25", ["other1", "other2"]),
            ("dense", ["other3"]),
        )
        result = compute_detailed_votes(item, runs)

        assert result["source_vote_count"] == 0
        assert result["target_vote_count"] == 0
        assert result["both_vote_count"] == 0
        assert result["retrievers_recovering_both"] == []

    def test_mixed_partial_retrieval(self):
        # bm25 gets source only, e5 gets both, dense gets target only
        item = _make_item()
        runs = _make_runs(
            ("bm25", ["src1"]),
            ("e5", ["src1", "tgt1"]),
            ("dense", ["tgt1"]),
        )
        result = compute_detailed_votes(item, runs)

        assert result["source_vote_count"] == 2    # bm25 + e5
        assert result["target_vote_count"] == 2    # e5 + dense
        assert result["both_vote_count"] == 1      # e5 only
        assert set(result["retrievers_recovering_source"]) == {"bm25", "e5"}
        assert set(result["retrievers_recovering_target"]) == {"e5", "dense"}
        assert result["retrievers_recovering_both"] == ["e5"]

    def test_empty_runs(self):
        item = _make_item()
        runs = {}
        result = compute_detailed_votes(item, runs)

        assert result["source_vote_count"] == 0
        assert result["target_vote_count"] == 0
        assert result["both_vote_count"] == 0
        assert result["retrievers_recovering_source"] == []
        assert result["retrievers_recovering_target"] == []
        assert result["retrievers_recovering_both"] == []

    def test_item_not_in_run(self):
        # item_id not present in any run results
        item = _make_item(item_id="missing_q")
        runs = {"bm25": {"q1": ["src1", "tgt1"]}}
        result = compute_detailed_votes(item, runs)

        assert result["source_vote_count"] == 0
        assert result["target_vote_count"] == 0
        assert result["both_vote_count"] == 0

    def test_source_equals_target_edge_case(self):
        # Degenerate: source and target are the same passage
        item = _make_item(src="p1", tgt="p1")
        runs = _make_runs(("bm25", ["p1"]))
        result = compute_detailed_votes(item, runs)

        # Both source and target = p1, so bm25 got both
        assert result["source_vote_count"] == 1
        assert result["target_vote_count"] == 1
        assert result["both_vote_count"] == 1

    def test_multiple_items_only_target_item_counted(self):
        # The function is per-item; other item_ids in the run don't affect result
        item = _make_item(item_id="q1", src="src1", tgt="tgt1")
        runs = {
            "bm25": {
                "q1": ["src1"],
                "q2": ["src1", "tgt1"],  # another item
            }
        }
        result = compute_detailed_votes(item, runs)
        assert result["source_vote_count"] == 1
        assert result["target_vote_count"] == 0


# =============================================================================
# Round-trip: compute_detailed_votes → assign_ir_difficulty_label
# =============================================================================

class TestVoteDetailToLabel:

    def _label(self, runs, item=None) -> str:
        if item is None:
            item = _make_item()
        detail = compute_detailed_votes(item, runs)
        return assign_ir_difficulty_label(
            detail["source_vote_count"],
            detail["target_vote_count"],
            detail["both_vote_count"],
            num_retrievers=len(runs),
        )

    def test_easy_majority_retrievers_got_both(self):
        runs = _make_runs(
            ("r1", ["src1", "tgt1"]),
            ("r2", ["src1", "tgt1"]),
            ("r3", ["src1", "tgt1"]),
        )
        assert self._label(runs) == "easy"

    def test_medium_one_of_four_got_both(self):
        runs = _make_runs(
            ("r1", ["src1", "tgt1"]),
            ("r2", ["src1"]),
            ("r3", ["tgt1"]),
            ("r4", []),
        )
        assert self._label(runs) == "medium"

    def test_hard_each_retrieved_by_different_retrievers(self):
        runs = _make_runs(
            ("r1", ["src1"]),
            ("r2", ["tgt1"]),
        )
        assert self._label(runs) == "hard"

    def test_source_only_label(self):
        runs = _make_runs(
            ("r1", ["src1"]),
            ("r2", ["src1"]),
        )
        assert self._label(runs) == "source_only"

    def test_target_only_label(self):
        runs = _make_runs(
            ("r1", ["tgt1"]),
        )
        assert self._label(runs) == "target_only"

    def test_neither_label(self):
        runs = _make_runs(
            ("r1", ["other1"]),
            ("r2", ["other2"]),
        )
        assert self._label(runs) == "neither"
