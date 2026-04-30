from __future__ import annotations

import json

from obliqaxref.curate.ir.rerank import CrossEncoderReranker
from obliqaxref.curate.ir.types import RetrievalRun, SearchResult


class FakeCrossEncoder:
    def __init__(self, scores: dict[str, float] | None = None):
        self.scores = scores or {}
        self.pairs_seen: list[tuple[str, str]] = []

    def predict(self, pairs, show_progress_bar=False):
        self.pairs_seen.extend(pairs)
        return [self.scores.get(text, 0.0) for _query, text in pairs]


def _run(name: str, qid: str, passage_ids: list[str]) -> RetrievalRun:
    return RetrievalRun(
        run_name=name,
        results={
            qid: [
                SearchResult(passage_id=pid, score=float(len(passage_ids) - i), rank=i + 1)
                for i, pid in enumerate(passage_ids)
            ]
        },
        k=len(passage_ids),
    )


def test_rerank_union_uses_question_text_not_query_id():
    model = FakeCrossEncoder({"source passage text": 1.0, "target passage text": 0.5})
    reranker = CrossEncoderReranker(model=model)
    qid = "item_123"

    reranker.rerank_union(
        runs=[_run("bm25", qid, ["src", "tgt"])],
        passage_index={
            "src": {"passage_id": "src", "text": "source passage text"},
            "tgt": {"passage_id": "tgt", "text": "target passage text"},
        },
        queries={qid: "What must the firm disclose?"},
        final_k=2,
    )

    assert model.pairs_seen
    assert all(query == "What must the firm disclose?" for query, _text in model.pairs_seen)
    assert all(query != qid for query, _text in model.pairs_seen)


def test_rerank_sorts_descending_when_higher_score_is_better():
    model = FakeCrossEncoder({"low": -1.0, "high": 3.0, "middle": 1.0})
    reranker = CrossEncoderReranker(model=model)

    results = reranker.rerank(
        query="actual question",
        passages=[
            {"passage_id": "a", "text": "low"},
            {"passage_id": "b", "text": "high"},
            {"passage_id": "c", "text": "middle"},
        ],
        k=3,
    )

    assert [r.passage_id for r in results] == ["b", "c", "a"]
    assert [r.score for r in results] == [3.0, 1.0, -1.0]
    assert [r.rank for r in results] == [1, 2, 3]


def test_debug_reports_source_target_candidate_presence(tmp_path):
    model = FakeCrossEncoder(
        {
            "source text": 0.2,
            "target text": 0.9,
            "distractor text": 0.1,
        }
    )
    reranker = CrossEncoderReranker(model=model)
    qid = "q_debug"
    debug_path = tmp_path / "ce_debug.jsonl"

    reranker.rerank_union(
        runs=[
            _run("bm25", qid, ["src", "other"]),
            _run("e5", qid, ["other"]),
        ],
        passage_index={
            "src": {"passage_id": "src", "text": "source text"},
            "tgt": {"passage_id": "tgt", "text": "target text"},
            "other": {"passage_id": "other", "text": "distractor text"},
        },
        queries={qid: "Which obligation applies?"},
        gold_pairs={qid: ("src", "tgt")},
        final_k=3,
        debug_sample_size=1,
        debug_output_path=debug_path,
    )

    record = json.loads(debug_path.read_text(encoding="utf-8").strip())
    assert record["query_id"] == qid
    assert record["question"] == "Which obligation applies?"
    assert record["gold_source_id"] == "src"
    assert record["gold_target_id"] == "tgt"
    assert record["source_in_union"] is True
    assert record["target_in_union"] is False
    assert record["ce_top10_passage_ids"] == ["src", "other"]
    assert record["source_rank_after_ce"] == 1
    assert record["target_rank_after_ce"] is None
