# tests/test_assemble_final_benchmark.py
"""
Unit tests for final-benchmark assembly.

Covers:
- assign_difficulty_tier(): 2-way tier collapse
- assemble_final_benchmark(): JSONL / CSV output, hard-cases split, stats
- Edge cases: empty inputs, missing files, all-challenging, all-retrievable
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from obliqaxref.curate.run import (
    _BENCHMARK_CSV_FIELDS,
    _CHALLENGING_LABELS,
    assemble_final_benchmark,
    assign_difficulty_tier,
)


# =============================================================================
# assign_difficulty_tier
# =============================================================================


class TestAssignDifficultyTier:
    def test_easy_is_retrievable(self):
        assert assign_difficulty_tier("easy") == "retrievable"

    def test_medium_is_retrievable(self):
        assert assign_difficulty_tier("medium") == "retrievable"

    def test_hard_is_challenging(self):
        assert assign_difficulty_tier("hard") == "challenging"

    def test_source_only_is_challenging(self):
        assert assign_difficulty_tier("source_only") == "challenging"

    def test_target_only_is_challenging(self):
        assert assign_difficulty_tier("target_only") == "challenging"

    def test_neither_is_challenging(self):
        assert assign_difficulty_tier("neither") == "challenging"

    def test_unknown_label_is_retrievable(self):
        # Unknown labels are not in _CHALLENGING_LABELS so map to retrievable
        assert assign_difficulty_tier("unknown") == "retrievable"

    def test_challenging_labels_constant_is_correct(self):
        assert _CHALLENGING_LABELS == {"hard", "source_only", "target_only", "neither"}


# =============================================================================
# Helpers
# =============================================================================


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _setup_curate_out(
    tmp_path: Path,
    *,
    judge_pass: list[str],
    answer_pass: list[str],
    answer_drop: list[str] | None = None,
    ir_items: list[dict],
    gen_items: list[dict],
    keep_items: list[dict] | None = None,
    judge_metadata: list[dict] | None = None,
    answer_metadata: list[dict] | None = None,
) -> tuple[Path, Path]:
    """Write fixture files and return (curate_out, items_file)."""
    curate_out = tmp_path / "curate"
    judge_dir = curate_out / "curate_judge"
    answer_dir = curate_out / "curate_answer"
    judge_dir.mkdir(parents=True)
    answer_dir.mkdir(parents=True)

    # judge_responses_pass.jsonl
    _write_jsonl(
        judge_dir / "judge_responses_pass.jsonl",
        [{"item_id": iid, "decision_qp_final": "PASS_QP"} for iid in judge_pass],
    )

    # answer_responses_pass.jsonl
    _write_jsonl(
        answer_dir / "answer_responses_pass.jsonl",
        [{"item_id": iid, "decision_ans_final": "PASS_ANS"} for iid in answer_pass],
    )
    _write_jsonl(
        answer_dir / "answer_responses_drop.jsonl",
        [{"item_id": iid, "decision_ans_final": "DROP_ANS"} for iid in (answer_drop or [])],
    )
    if judge_metadata is not None:
        _write_jsonl(judge_dir / "judge_responses_aggregated.jsonl", judge_metadata)
    if answer_metadata is not None:
        _write_jsonl(answer_dir / "answer_responses_aggregated.jsonl", answer_metadata)
    if keep_items is not None:
        _write_jsonl(curate_out / "curated_items.keep.jsonl", keep_items)

    # curated_items.judge.jsonl
    _write_jsonl(curate_out / "curated_items.judge.jsonl", ir_items)

    # generator items
    items_file = tmp_path / "items.jsonl"
    _write_jsonl(items_file, gen_items)

    return curate_out, items_file


# =============================================================================
# assemble_final_benchmark
# =============================================================================


class TestAssembleFinalBenchmark:

    # --- basic happy path ---

    def test_returns_stats_dict(self, tmp_path):
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=["q1"],
            answer_pass=["q1"],
            ir_items=[{
                "item_id": "q1",
                "ir_difficulty_label": "easy",
                "source_passage_id": "s1",
                "target_passage_id": "t1",
                "source_vote_count": 3,
                "target_vote_count": 3,
                "both_vote_count": 3,
                "retrievers_recovering_source": ["bm25"],
                "retrievers_recovering_target": ["bm25"],
                "retrievers_recovering_both": ["bm25"],
            }],
            gen_items=[{
                "item_id": "q1",
                "question": "Q?",
                "gold_answer": "A.",
                "source_passage_id": "s1",
                "target_passage_id": "t1",
                "method": "dpel",
                "pair_uid": "p1",
            }],
        )
        stats = assemble_final_benchmark(curate_out, items_file)
        assert stats["total_final"] == 1
        assert stats["total_hard"] == 0

    def test_explicit_and_compatibility_output_files_created(self, tmp_path):
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=["q1"],
            answer_pass=["q1"],
            ir_items=[{"item_id": "q1", "ir_difficulty_label": "medium",
                        "source_vote_count": 1, "target_vote_count": 1,
                        "both_vote_count": 1,
                        "retrievers_recovering_source": [],
                        "retrievers_recovering_target": [],
                        "retrievers_recovering_both": []}],
            gen_items=[{"item_id": "q1", "question": "Q", "gold_answer": "A",
                        "source_passage_id": "s1", "target_passage_id": "t1",
                        "method": "schema", "pair_uid": "p1"}],
        )
        assemble_final_benchmark(curate_out, items_file)

        for stem in (
            "final_dependency_valid",
            "final_answer_valid",
            "final_answer_failed",
            "final_benchmark",
            "final_hard",
        ):
            assert (curate_out / f"{stem}.jsonl").exists()
            assert (curate_out / f"{stem}.csv").exists()
        assert (curate_out / "final_benchmark_stats.json").exists()

    # --- intersection logic ---

    def test_intersection_only_keeps_both_pass(self, tmp_path):
        # q1 passes both; q2 only judge; q3 only answer
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=["q1", "q2"],
            answer_pass=["q1", "q3"],
            ir_items=[
                {"item_id": "q1", "ir_difficulty_label": "easy",
                 "source_vote_count": 1, "target_vote_count": 1, "both_vote_count": 1,
                 "retrievers_recovering_source": [], "retrievers_recovering_target": [],
                 "retrievers_recovering_both": []},
            ],
            gen_items=[
                {"item_id": "q1", "question": "Q1", "gold_answer": "A1",
                 "source_passage_id": "s1", "target_passage_id": "t1",
                 "method": "dpel", "pair_uid": "p1"},
            ],
        )
        stats = assemble_final_benchmark(curate_out, items_file)
        assert stats["total_final"] == 1

        items = _read_jsonl(curate_out / "final_benchmark.jsonl")
        ids = {item["item_id"] for item in items}
        assert ids == {"q1"}

    def test_judge_pass_answer_pass_goes_to_final_answer_valid(self, tmp_path):
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=["q1"],
            answer_pass=["q1"],
            ir_items=[{"item_id": "q1", "ir_difficulty_label": "easy"}],
            gen_items=[{
                "item_id": "q1",
                "question": "Q1",
                "gold_answer": "A [#SRC:s1] [#TGT:t1]",
                "source_passage_id": "s1",
                "target_passage_id": "t1",
            }],
        )
        assemble_final_benchmark(curate_out, items_file)

        answer_valid_ids = {it["item_id"] for it in _read_jsonl(curate_out / "final_answer_valid.jsonl")}
        assert answer_valid_ids == {"q1"}

    def test_judge_pass_answer_fail_goes_to_final_answer_failed(self, tmp_path):
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=["q1"],
            answer_pass=[],
            answer_drop=["q1"],
            ir_items=[{"item_id": "q1", "ir_difficulty_label": "medium"}],
            gen_items=[{
                "item_id": "q1",
                "question": "Q1",
                "gold_answer": "A [#SRC:s1] [#TGT:t1]",
                "source_passage_id": "s1",
                "target_passage_id": "t1",
            }],
        )
        assemble_final_benchmark(curate_out, items_file)

        answer_valid = _read_jsonl(curate_out / "final_answer_valid.jsonl")
        answer_failed_ids = {
            it["item_id"] for it in _read_jsonl(curate_out / "final_answer_failed.jsonl")
        }
        assert answer_valid == []
        assert answer_failed_ids == {"q1"}

    def test_judge_drop_excluded_from_dependency_and_answer_valid(self, tmp_path):
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=["q_pass"],
            answer_pass=["q_pass", "q_drop"],
            ir_items=[
                {"item_id": "q_pass", "ir_difficulty_label": "easy"},
                {"item_id": "q_drop", "ir_difficulty_label": "hard"},
            ],
            gen_items=[
                {"item_id": "q_pass", "question": "Q", "gold_answer": "A",
                 "source_passage_id": "s", "target_passage_id": "t"},
                {"item_id": "q_drop", "question": "Q", "gold_answer": "A",
                 "source_passage_id": "s", "target_passage_id": "t"},
            ],
        )
        assemble_final_benchmark(curate_out, items_file)

        dependency_ids = {
            it["item_id"] for it in _read_jsonl(curate_out / "final_dependency_valid.jsonl")
        }
        answer_ids = {
            it["item_id"] for it in _read_jsonl(curate_out / "final_answer_valid.jsonl")
        }
        assert dependency_ids == {"q_pass"}
        assert answer_ids == {"q_pass"}

    def test_hard_valid_cases_are_retained_in_answer_valid(self, tmp_path):
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=["q_hard"],
            answer_pass=["q_hard"],
            ir_items=[{
                "item_id": "q_hard",
                "ir_difficulty_label": "hard",
                "source_vote_count": 1,
                "target_vote_count": 1,
                "both_vote_count": 0,
            }],
            gen_items=[{
                "item_id": "q_hard",
                "question": "Q",
                "gold_answer": "A",
                "source_passage_id": "s",
                "target_passage_id": "t",
            }],
        )
        assemble_final_benchmark(curate_out, items_file)

        item = _read_jsonl(curate_out / "final_answer_valid.jsonl")[0]
        assert item["item_id"] == "q_hard"
        assert item["difficulty_tier"] == "challenging"

    def test_stale_keep_file_does_not_control_final_export(self, tmp_path):
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=["q_pass"],
            answer_pass=["q_pass"],
            keep_items=[{"item_id": "q_keep_only", "decision": "KEEP"}],
            ir_items=[
                {"item_id": "q_pass", "ir_difficulty_label": "easy"},
                {"item_id": "q_keep_only", "ir_difficulty_label": "easy"},
            ],
            gen_items=[
                {"item_id": "q_pass", "question": "Q", "gold_answer": "A",
                 "source_passage_id": "s", "target_passage_id": "t"},
                {"item_id": "q_keep_only", "question": "Q", "gold_answer": "A",
                 "source_passage_id": "s", "target_passage_id": "t"},
            ],
        )
        assemble_final_benchmark(curate_out, items_file)

        dependency_ids = {
            it["item_id"] for it in _read_jsonl(curate_out / "final_dependency_valid.jsonl")
        }
        answer_ids = {
            it["item_id"] for it in _read_jsonl(curate_out / "final_answer_valid.jsonl")
        }
        assert dependency_ids == {"q_pass"}
        assert answer_ids == {"q_pass"}

    def test_answer_and_judge_metadata_are_exported(self, tmp_path):
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=["q1"],
            answer_pass=["q1"],
            ir_items=[{"item_id": "q1", "ir_difficulty_label": "easy"}],
            gen_items=[{
                "item_id": "q1",
                "question": "Q",
                "gold_answer": "Answer with [#SRC:s1] [#TGT:t1]",
                "source_passage_id": "s1",
                "target_passage_id": "t1",
            }],
            judge_metadata=[{
                "item_id": "q1",
                "judge_schema_version": "v2",
                "source_alone_sufficient": False,
                "target_alone_sufficient": False,
                "target_adds_essential_information": True,
                "citation_dependent": True,
                "answer_supported_by_judge": True,
            }],
            answer_metadata=[{
                "item_id": "q1",
                "decision_ans_final": "PASS_ANS",
                "confidence_mean": 0.91,
                "answer_responsive": True,
                "answer_supported": True,
                "runs": [{"notes": "grounded"}],
            }],
        )
        assemble_final_benchmark(curate_out, items_file)

        item = _read_jsonl(curate_out / "final_answer_valid.jsonl")[0]
        assert item["answer_validation_passed"] is True
        assert item["answer_validation_score"] == 0.91
        assert item["answer_validation_reasons"] == ["grounded"]
        assert item["missing_source_tag"] is False
        assert item["missing_target_tag"] is False
        assert item["answer_responsive"] is True
        assert item["answer_supported"] is True
        assert item["judge_schema_version"] == "v2"
        assert item["source_alone_sufficient"] is False
        assert item["target_alone_sufficient"] is False
        assert item["target_adds_essential_information"] is True
        assert item["citation_dependent"] is True
        assert item["answer_supported_by_judge"] is True

    def test_no_overlap_returns_empty_benchmark(self, tmp_path):
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=["q1"],
            answer_pass=["q2"],
            ir_items=[],
            gen_items=[],
        )
        stats = assemble_final_benchmark(curate_out, items_file)
        assert stats["total_final"] == 0
        assert stats["total_hard"] == 0

    # --- hard / challenging split ---

    def test_hard_items_in_final_hard_only(self, tmp_path):
        ir_items = [
            {"item_id": "q1", "ir_difficulty_label": "easy",
             "source_vote_count": 2, "target_vote_count": 2, "both_vote_count": 2,
             "retrievers_recovering_source": [], "retrievers_recovering_target": [],
             "retrievers_recovering_both": []},
            {"item_id": "q2", "ir_difficulty_label": "hard",
             "source_vote_count": 1, "target_vote_count": 1, "both_vote_count": 0,
             "retrievers_recovering_source": [], "retrievers_recovering_target": [],
             "retrievers_recovering_both": []},
            {"item_id": "q3", "ir_difficulty_label": "neither",
             "source_vote_count": 0, "target_vote_count": 0, "both_vote_count": 0,
             "retrievers_recovering_source": [], "retrievers_recovering_target": [],
             "retrievers_recovering_both": []},
        ]
        gen_items = [
            {"item_id": "q1", "question": "Q1", "gold_answer": "A1",
             "source_passage_id": "s1", "target_passage_id": "t1",
             "method": "dpel", "pair_uid": "p1"},
            {"item_id": "q2", "question": "Q2", "gold_answer": "A2",
             "source_passage_id": "s2", "target_passage_id": "t2",
             "method": "dpel", "pair_uid": "p2"},
            {"item_id": "q3", "question": "Q3", "gold_answer": "A3",
             "source_passage_id": "s3", "target_passage_id": "t3",
             "method": "schema", "pair_uid": "p3"},
        ]
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=["q1", "q2", "q3"],
            answer_pass=["q1", "q2", "q3"],
            ir_items=ir_items,
            gen_items=gen_items,
        )
        stats = assemble_final_benchmark(curate_out, items_file)

        assert stats["total_final"] == 3
        assert stats["total_hard"] == 2  # q2 (hard) + q3 (neither)

        hard = _read_jsonl(curate_out / "final_hard.jsonl")
        hard_ids = {item["item_id"] for item in hard}
        assert hard_ids == {"q2", "q3"}

        # q1 should NOT be in final_hard
        assert "q1" not in hard_ids

    def test_all_challenging_fills_final_hard(self, tmp_path):
        ir_items = [
            {"item_id": f"q{i}", "ir_difficulty_label": lbl,
             "source_vote_count": 0, "target_vote_count": 0, "both_vote_count": 0,
             "retrievers_recovering_source": [], "retrievers_recovering_target": [],
             "retrievers_recovering_both": []}
            for i, lbl in enumerate(["hard", "source_only", "target_only", "neither"])
        ]
        gen_items = [
            {"item_id": f"q{i}", "question": f"Q{i}", "gold_answer": f"A{i}",
             "source_passage_id": f"s{i}", "target_passage_id": f"t{i}",
             "method": "dpel", "pair_uid": f"p{i}"}
            for i in range(4)
        ]
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=[f"q{i}" for i in range(4)],
            answer_pass=[f"q{i}" for i in range(4)],
            ir_items=ir_items,
            gen_items=gen_items,
        )
        stats = assemble_final_benchmark(curate_out, items_file)
        assert stats["total_final"] == 4
        assert stats["total_hard"] == 4
        assert stats["difficulty_tier_counts"]["challenging"] == 4
        assert stats["difficulty_tier_counts"]["retrievable"] == 0

    def test_all_retrievable_empty_final_hard(self, tmp_path):
        ir_items = [
            {"item_id": f"q{i}", "ir_difficulty_label": lbl,
             "source_vote_count": 2, "target_vote_count": 2, "both_vote_count": 2,
             "retrievers_recovering_source": [], "retrievers_recovering_target": [],
             "retrievers_recovering_both": []}
            for i, lbl in enumerate(["easy", "medium"])
        ]
        gen_items = [
            {"item_id": f"q{i}", "question": f"Q{i}", "gold_answer": f"A{i}",
             "source_passage_id": f"s{i}", "target_passage_id": f"t{i}",
             "method": "dpel", "pair_uid": f"p{i}"}
            for i in range(2)
        ]
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=["q0", "q1"],
            answer_pass=["q0", "q1"],
            ir_items=ir_items,
            gen_items=gen_items,
        )
        stats = assemble_final_benchmark(curate_out, items_file)
        assert stats["total_hard"] == 0
        assert stats["difficulty_tier_counts"]["retrievable"] == 2

        hard = _read_jsonl(curate_out / "final_hard.jsonl")
        assert hard == []

    # --- output fields ---

    def test_benchmark_record_has_required_fields(self, tmp_path):
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=["q1"],
            answer_pass=["q1"],
            ir_items=[{
                "item_id": "q1",
                "ir_difficulty_label": "hard",
                "source_passage_id": "s1",
                "target_passage_id": "t1",
                "source_vote_count": 1,
                "target_vote_count": 1,
                "both_vote_count": 0,
                "retrievers_recovering_source": ["bm25"],
                "retrievers_recovering_target": ["e5"],
                "retrievers_recovering_both": [],
            }],
            gen_items=[{
                "item_id": "q1",
                "question": "What cross-references section 3?",
                "gold_answer": "Section 5 cross-references section 3.",
                "source_passage_id": "s1",
                "target_passage_id": "t1",
                "method": "dpel",
                "pair_uid": "pair_001",
                "persona": "compliance officer",
            }],
        )
        assemble_final_benchmark(curate_out, items_file)
        items = _read_jsonl(curate_out / "final_benchmark.jsonl")
        assert len(items) == 1
        item = items[0]

        assert item["item_id"] == "q1"
        assert item["question"] == "What cross-references section 3?"
        assert item["gold_answer"] == "Section 5 cross-references section 3."
        assert item["ir_difficulty_label"] == "hard"
        assert item["difficulty_tier"] == "challenging"
        assert item["both_vote_count"] == 0
        assert item["retrievers_recovering_source"] == ["bm25"]
        assert item["retrievers_recovering_target"] == ["e5"]
        assert item["retrievers_recovering_both"] == []
        assert item["method"] == "dpel"
        assert item["pair_uid"] == "pair_001"
        assert item["persona"] == "compliance officer"

    def test_csv_has_correct_header(self, tmp_path):
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=["q1"],
            answer_pass=["q1"],
            ir_items=[{"item_id": "q1", "ir_difficulty_label": "easy",
                        "source_vote_count": 1, "target_vote_count": 1,
                        "both_vote_count": 1,
                        "retrievers_recovering_source": [],
                        "retrievers_recovering_target": [],
                        "retrievers_recovering_both": []}],
            gen_items=[{"item_id": "q1", "question": "Q", "gold_answer": "A",
                        "source_passage_id": "s1", "target_passage_id": "t1",
                        "method": "dpel", "pair_uid": "p1"}],
        )
        assemble_final_benchmark(curate_out, items_file)
        rows = _read_csv(curate_out / "final_benchmark.csv")
        assert rows, "CSV should have at least one data row"
        assert set(rows[0].keys()) == set(_BENCHMARK_CSV_FIELDS)

    def test_difficulty_tier_in_csv_row(self, tmp_path):
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=["q1"],
            answer_pass=["q1"],
            ir_items=[{"item_id": "q1", "ir_difficulty_label": "source_only",
                        "source_vote_count": 1, "target_vote_count": 0,
                        "both_vote_count": 0,
                        "retrievers_recovering_source": [],
                        "retrievers_recovering_target": [],
                        "retrievers_recovering_both": []}],
            gen_items=[{"item_id": "q1", "question": "Q", "gold_answer": "A",
                        "source_passage_id": "s1", "target_passage_id": "t1",
                        "method": "schema", "pair_uid": "p1"}],
        )
        assemble_final_benchmark(curate_out, items_file)
        rows = _read_csv(curate_out / "final_benchmark.csv")
        assert rows[0]["difficulty_tier"] == "challenging"
        assert rows[0]["ir_difficulty_label"] == "source_only"

    # --- missing upstream files ---

    def test_missing_judge_pass_file_returns_empty(self, tmp_path):
        curate_out = tmp_path / "curate"
        answer_dir = curate_out / "curate_answer"
        answer_dir.mkdir(parents=True)
        _write_jsonl(
            answer_dir / "answer_responses_pass.jsonl",
            [{"item_id": "q1", "decision_ans_final": "PASS_ANS"}],
        )
        items_file = tmp_path / "items.jsonl"
        items_file.write_text("")

        result = assemble_final_benchmark(curate_out, items_file)
        assert result == {}

    def test_missing_answer_pass_file_still_writes_dependency_valid(self, tmp_path):
        curate_out = tmp_path / "curate"
        judge_dir = curate_out / "curate_judge"
        judge_dir.mkdir(parents=True)
        _write_jsonl(
            judge_dir / "judge_responses_pass.jsonl",
            [{"item_id": "q1", "decision_qp_final": "PASS_QP"}],
        )
        items_file = tmp_path / "items.jsonl"
        items_file.write_text("")

        result = assemble_final_benchmark(curate_out, items_file)
        assert result["final_dependency_valid_count"] == 1
        assert result["final_answer_valid_count"] == 0
        assert _read_jsonl(curate_out / "final_dependency_valid.jsonl")[0]["item_id"] == "q1"
        assert _read_jsonl(curate_out / "final_answer_valid.jsonl") == []

    # --- stats structure ---

    def test_stats_keys_present(self, tmp_path):
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=["q1"],
            answer_pass=["q1"],
            ir_items=[{"item_id": "q1", "ir_difficulty_label": "easy",
                        "source_vote_count": 2, "target_vote_count": 2,
                        "both_vote_count": 2,
                        "retrievers_recovering_source": [],
                        "retrievers_recovering_target": [],
                        "retrievers_recovering_both": []}],
            gen_items=[{"item_id": "q1", "question": "Q", "gold_answer": "A",
                        "source_passage_id": "s1", "target_passage_id": "t1",
                        "method": "dpel", "pair_uid": "p1"}],
        )
        stats = assemble_final_benchmark(curate_out, items_file)
        for key in (
            "generated_count",
            "citation_dependency_passed_count",
            "answer_validation_passed_count",
            "answer_validation_failed_count",
            "final_dependency_valid_count",
            "final_answer_valid_count",
            "total_final",
            "total_hard",
            "difficulty_tier_counts",
            "ir_difficulty_label_counts",
        ):
            assert key in stats, f"Missing key: {key}"

    def test_stats_json_persisted_correctly(self, tmp_path):
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=["q1"],
            answer_pass=["q1"],
            ir_items=[{"item_id": "q1", "ir_difficulty_label": "medium",
                        "source_vote_count": 1, "target_vote_count": 1,
                        "both_vote_count": 1,
                        "retrievers_recovering_source": [],
                        "retrievers_recovering_target": [],
                        "retrievers_recovering_both": []}],
            gen_items=[{"item_id": "q1", "question": "Q", "gold_answer": "A",
                        "source_passage_id": "s1", "target_passage_id": "t1",
                        "method": "schema", "pair_uid": "p1"}],
        )
        returned = assemble_final_benchmark(curate_out, items_file)

        with open(curate_out / "final_benchmark_stats.json") as f:
            persisted = json.load(f)

        assert persisted == returned

    def test_ir_difficulty_label_counts_correct(self, tmp_path):
        ir_items = [
            {"item_id": "q0", "ir_difficulty_label": "easy",
             "source_vote_count": 3, "target_vote_count": 3, "both_vote_count": 3,
             "retrievers_recovering_source": [], "retrievers_recovering_target": [],
             "retrievers_recovering_both": []},
            {"item_id": "q1", "ir_difficulty_label": "hard",
             "source_vote_count": 1, "target_vote_count": 1, "both_vote_count": 0,
             "retrievers_recovering_source": [], "retrievers_recovering_target": [],
             "retrievers_recovering_both": []},
            {"item_id": "q2", "ir_difficulty_label": "hard",
             "source_vote_count": 0, "target_vote_count": 1, "both_vote_count": 0,
             "retrievers_recovering_source": [], "retrievers_recovering_target": [],
             "retrievers_recovering_both": []},
        ]
        gen_items = [
            {"item_id": f"q{i}", "question": "Q", "gold_answer": "A",
             "source_passage_id": "s", "target_passage_id": "t",
             "method": "dpel", "pair_uid": "p"}
            for i in range(3)
        ]
        curate_out, items_file = _setup_curate_out(
            tmp_path,
            judge_pass=["q0", "q1", "q2"],
            answer_pass=["q0", "q1", "q2"],
            ir_items=ir_items,
            gen_items=gen_items,
        )
        stats = assemble_final_benchmark(curate_out, items_file)
        assert stats["ir_difficulty_label_counts"]["easy"] == 1
        assert stats["ir_difficulty_label_counts"]["hard"] == 2
        assert stats["difficulty_tier_counts"]["retrievable"] == 1
        assert stats["difficulty_tier_counts"]["challenging"] == 2
