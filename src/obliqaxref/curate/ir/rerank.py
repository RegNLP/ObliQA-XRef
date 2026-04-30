"""
Cross-encoder reranking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from obliqaxref.curate.ir.types import RetrievalRun, SearchResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranking.

    Reranks top-K results from multiple retrievers using a cross-encoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = None,
        model: Any | None = None,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model
            device: Device (cuda/cpu), auto-detected if None
        """
        self.model_name = model_name
        if model is not None:
            self.model = model
            return

        from sentence_transformers import CrossEncoder

        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name, device=device)
        logger.info("Cross-encoder loaded on %s", getattr(self.model, "device", "unknown device"))

    def rerank(
        self,
        query: str,
        passages: list[dict[str, str]],
        k: int = 100,
    ) -> list[SearchResult]:
        """
        Rerank passages for a single query.

        Args:
            query: Query text
            passages: List of {"passage_id": ..., "text": ...}
            k: Number of results to return

        Returns:
            Reranked SearchResult list
        """
        if not passages:
            return []

        # Prepare pairs for cross-encoder
        pairs = [(query, p["text"]) for p in passages]

        # Score with cross-encoder
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Sort by score (descending)
        passage_scores = list(zip(passages, scores))
        passage_scores.sort(key=lambda x: x[1], reverse=True)

        # Convert to SearchResult
        results = []
        for rank, (passage, score) in enumerate(passage_scores[:k], start=1):
            results.append(
                SearchResult(
                    passage_id=passage["passage_id"],
                    score=float(score),
                    rank=rank,
                )
            )

        return results

    def rerank_union(
        self,
        runs: list[RetrievalRun],
        passage_index: dict[str, dict[str, str]],
        queries: dict[str, str] | None = None,
        union_k: int = 200,
        final_k: int = 100,
        run_name: str = "ce_rerank_union200",
        gold_pairs: dict[str, tuple[str | None, str | None]] | None = None,
        debug_sample_size: int = 0,
        debug_output_path: str | Path | None = None,
    ) -> RetrievalRun:
        """
        Rerank union of top-K from multiple runs.

        Args:
            runs: List of RetrievalRun objects
            passage_index: Dict {passage_id: {"passage_id": ..., "text": ...}}
            queries: Optional dict {query_id: question_text}. If omitted, falls back
                to query_id for backward compatibility.
            union_k: Number of passages to take from each run before reranking
            final_k: Final number of results after reranking
            run_name: Name for the reranked run
            gold_pairs: Optional dict {query_id: (source_id, target_id)} for debug output
            debug_sample_size: Number of query-level debug records to write
            debug_output_path: JSONL path for CE debug records

        Returns:
            Reranked RetrievalRun
        """
        if not runs:
            raise ValueError("Need at least one run to rerank")

        # Get queries (intersection of all runs)
        query_ids = set(runs[0].results.keys())
        for run in runs[1:]:
            query_ids &= set(run.results.keys())

        logger.info(
            f"Reranking union of {len(runs)} runs for {len(query_ids)} queries "
            f"(union_k={union_k}, final_k={final_k})"
        )

        reranked_results = {}
        debug_records: list[dict[str, Any]] = []
        warned_missing_query_text = False

        for i, query_id in enumerate(sorted(query_ids), 1):
            if i % 10 == 0:
                logger.info(f"Reranked {i}/{len(query_ids)} queries")

            # Collect union of top-K passages from all runs
            union_passage_ids: list[str] = []
            seen_passage_ids: set[str] = set()
            for run in runs:
                results = run.results[query_id][:union_k]
                for result in results:
                    if result.passage_id not in seen_passage_ids:
                        seen_passage_ids.add(result.passage_id)
                        union_passage_ids.append(result.passage_id)

            # Get passage objects
            passages = []
            for passage_id in union_passage_ids:
                if passage_id in passage_index:
                    passages.append(passage_index[passage_id])
                else:
                    logger.warning(f"Passage {passage_id} not in index")

            # Rerank with cross-encoder
            query_text = (queries or {}).get(query_id)
            if not query_text:
                query_text = query_id
                if not warned_missing_query_text:
                    logger.warning(
                        "No question text mapping supplied to CrossEncoderReranker.rerank_union(); "
                        "falling back to query_id for backward compatibility."
                    )
                    warned_missing_query_text = True

            reranked = self.rerank(query_text, passages, k=final_k)
            reranked_results[query_id] = reranked

            if debug_output_path and len(debug_records) < max(0, debug_sample_size):
                source_id, target_id = (None, None)
                if gold_pairs and query_id in gold_pairs:
                    source_id, target_id = gold_pairs[query_id]

                top10 = reranked[:10]
                ranked_ids = [r.passage_id for r in reranked]
                debug_records.append(
                    {
                        "query_id": query_id,
                        "question": query_text,
                        "gold_source_id": source_id,
                        "gold_target_id": target_id,
                        "source_in_union": bool(source_id and source_id in seen_passage_ids),
                        "target_in_union": bool(target_id and target_id in seen_passage_ids),
                        "union_candidate_count": len(seen_passage_ids),
                        "union_candidate_ids_sample": union_passage_ids[:10],
                        "ce_top10_passage_ids": [r.passage_id for r in top10],
                        "ce_top10_scores": [r.score for r in top10],
                        "source_rank_after_ce": (
                            ranked_ids.index(source_id) + 1
                            if source_id and source_id in ranked_ids
                            else None
                        ),
                        "target_rank_after_ce": (
                            ranked_ids.index(target_id) + 1
                            if target_id and target_id in ranked_ids
                            else None
                        ),
                    }
                )

        logger.info(f"Cross-encoder reranking complete: {run_name}")

        if debug_output_path and debug_records:
            debug_path = Path(debug_output_path)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            with debug_path.open("w", encoding="utf-8") as f:
                for record in debug_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("Wrote cross-encoder debug sample: %s", debug_path)

        return RetrievalRun(
            run_name=run_name,
            results=reranked_results,
            k=final_k,
        )
