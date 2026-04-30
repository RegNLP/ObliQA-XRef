from __future__ import annotations

import csv
import json

from obliqaxref.eval.Analysis.answer_quality_by_retrieval import (
    answer_quality_by_retrieval_outcome,
    join_answer_quality_with_retrieval,
    summarize_by_group,
)


def test_answer_quality_join_uses_item_and_retriever():
    diagnostics = [
        {
            "item_id": "q1",
            "query_id": "q1",
            "corpus": "adgm",
            "method": "DPEL",
            "retriever": "bm25",
            "source_id": "s1",
            "target_id": "t1",
            "retrieval_outcome_at_5": "src_only",
            "retrieval_outcome_at_10": "both",
            "retrieval_outcome_at_20": "both",
        }
    ]
    answers = [
        {
            "item_id": "q1",
            "corpus": "adgm",
            "method": "DPEL",
            "retriever": "bm25",
            "rougeL_f1": 0.75,
            "passage_overlap_frac": 0.5,
            "gpt_relevance": 0.8,
            "gpt_faithfulness": 0.9,
            "nli_scores": {"entailment": 2.0, "contradiction": -1.0},
            "has_id_tag": True,
            "n_id_tags": 2,
            "len_words": 40,
        }
    ]

    joined = join_answer_quality_with_retrieval(diagnostics, answers)

    assert len(joined) == 1
    assert joined[0]["retrieval_outcome"] == "both"
    assert joined[0]["retrieval_outcome_at_5"] == "source_only"
    assert joined[0]["rouge_l"] == 0.75
    assert joined[0]["faithfulness"] == 0.9
    assert joined[0]["tag_plus"] is True


def test_answer_quality_summary_computes_outcome_groups():
    rows = [
        {
            "retrieval_outcome": "both",
            "corpus": "adgm",
            "method": "DPEL",
            "retriever": "bm25",
            "generator_model": "",
            "split": "test",
            "rouge_l": 0.8,
        },
        {
            "retrieval_outcome": "source_only",
            "corpus": "adgm",
            "method": "DPEL",
            "retriever": "bm25",
            "generator_model": "",
            "split": "test",
            "rouge_l": 0.5,
        },
        {
            "retrieval_outcome": "target_only",
            "corpus": "adgm",
            "method": "DPEL",
            "retriever": "bm25",
            "generator_model": "",
            "split": "test",
            "rouge_l": 0.4,
        },
        {
            "retrieval_outcome": "neither",
            "corpus": "adgm",
            "method": "DPEL",
            "retriever": "bm25",
            "generator_model": "",
            "split": "test",
            "rouge_l": 0.2,
        },
    ]

    summary = summarize_by_group(rows)
    outcomes = {row["retrieval_outcome"] for row in summary}

    assert outcomes == {"both", "source_only", "target_only", "neither"}
    both_row = next(row for row in summary if row["retrieval_outcome"] == "both")
    assert both_row["rouge_l_mean"] == 0.8
    assert round(both_row["both_minus_non_both_rouge_l"], 4) == round(0.8 - ((0.5 + 0.4 + 0.2) / 3), 4)


def test_answer_quality_command_function_creates_expected_files(tmp_path):
    diag_path = tmp_path / "retrieval_diagnostics_per_query.csv"
    with diag_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "corpus",
                "method",
                "retriever",
                "split",
                "query_id",
                "item_id",
                "source_id",
                "target_id",
                "retrieval_outcome_at_5",
                "retrieval_outcome_at_10",
                "retrieval_outcome_at_20",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "corpus": "adgm",
            "method": "DPEL",
            "retriever": "bm25",
            "split": "test",
            "query_id": "q1",
            "item_id": "q1",
            "source_id": "s1",
            "target_id": "t1",
            "retrieval_outcome_at_5": "src_only",
            "retrieval_outcome_at_10": "both",
            "retrieval_outcome_at_20": "both",
        })
    (tmp_path / "answer_eval_adgm_dpel_bm25_test.json").write_text(
        json.dumps({"results": [{"item_id": "q1", "rougeL_f1": 0.7, "len_words": 10}]}),
        encoding="utf-8",
    )

    paths = answer_quality_by_retrieval_outcome(root_dir=tmp_path)

    assert paths["detail"].exists()
    assert paths["summary"].exists()
    assert paths["markdown"].exists()
    with paths["detail"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["item_id"] == "q1"
    assert rows[0]["retrieval_outcome"] == "both"
