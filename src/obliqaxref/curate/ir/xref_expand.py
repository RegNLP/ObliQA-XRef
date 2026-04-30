"""
Cross-reference graph expansion for retrieval runs.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from obliqaxref.curate.ir.types import RetrievalRun, SearchResult


@dataclass(frozen=True)
class XRefGraph:
    outgoing: dict[str, list[str]]
    incoming: dict[str, list[str]]


def _unique_nonempty(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        value = (value or "").strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def load_xref_graph(crossref_file: Path, valid_passage_ids: set[str]) -> XRefGraph:
    """
    Load a directed cross-reference graph from adapter crossref CSV.

    The retrieval corpus currently indexes passage UIDs as TREC doc IDs. The
    adapter CSV also carries display passage IDs for some corpora, so this loader
    accepts either form but only keeps edges where both endpoints are present in
    the active retrieval passage index.
    """
    outgoing: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, list[str]] = defaultdict(list)
    edge_seen: set[tuple[str, str]] = set()

    with crossref_file.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_candidates = _unique_nonempty(
                [row.get("SourceID"), row.get("SourcePassageID")]
            )
            target_candidates = _unique_nonempty(
                [row.get("TargetID"), row.get("TargetPassageID")]
            )
            sources = [pid for pid in source_candidates if pid in valid_passage_ids]
            targets = [pid for pid in target_candidates if pid in valid_passage_ids]
            for source_id in sources:
                for target_id in targets:
                    if source_id == target_id:
                        continue
                    edge = (source_id, target_id)
                    if edge in edge_seen:
                        continue
                    edge_seen.add(edge)
                    outgoing[source_id].append(target_id)
                    incoming[target_id].append(source_id)

    return XRefGraph(outgoing=dict(outgoing), incoming=dict(incoming))


def _neighbours_for_seed(
    graph: XRefGraph,
    seed_id: str,
    direction: str,
    max_expanded_per_seed: int | None,
) -> list[str]:
    neighbours: list[str] = []
    if direction in {"outgoing", "both"}:
        neighbours.extend(graph.outgoing.get(seed_id, []))
    if direction in {"incoming", "both"}:
        neighbours.extend(graph.incoming.get(seed_id, []))

    deduped = _unique_nonempty(neighbours)
    if max_expanded_per_seed is not None:
        return deduped[:max_expanded_per_seed]
    return deduped


def expand_retrieval_run(
    base_run: RetrievalRun,
    graph: XRefGraph,
    *,
    run_name: str,
    seed_k: int = 20,
    final_k: int = 20,
    expansion_direction: str = "both",
    expansion_weight: float = 0.8,
    neighbour_score_mode: str = "max",
    max_expanded_per_seed: int | None = None,
    gold_pairs: dict[str, tuple[str | None, str | None]] | None = None,
) -> tuple[RetrievalRun, list[dict[str, Any]]]:
    """
    Add cross-reference neighbours to a base retrieval run and re-rank by score.

    Gold pairs are accepted only for optional diagnostics; they are not used for
    expansion or scoring.
    """
    if expansion_direction not in {"outgoing", "incoming", "both"}:
        raise ValueError("expansion_direction must be outgoing, incoming, or both")
    if neighbour_score_mode not in {"max", "sum"}:
        raise ValueError("neighbour_score_mode must be max or sum")
    if seed_k <= 0 or final_k <= 0:
        raise ValueError("seed_k and final_k must be positive")

    expanded_results: dict[str, list[SearchResult]] = {}
    diagnostics: list[dict[str, Any]] = []

    for query_id, results in base_run.results.items():
        seed_results = results[:seed_k]
        base_scores: dict[str, float] = {}
        neighbour_scores: dict[str, float] = {}
        added_by_expansion: set[str] = set()

        for result in seed_results:
            base_scores[result.passage_id] = max(
                base_scores.get(result.passage_id, float("-inf")),
                result.score,
            )
            neighbours = _neighbours_for_seed(
                graph,
                result.passage_id,
                expansion_direction,
                max_expanded_per_seed,
            )
            for neighbour_id in neighbours:
                candidate_score = result.score * expansion_weight
                if neighbour_score_mode == "sum":
                    neighbour_scores[neighbour_id] = (
                        neighbour_scores.get(neighbour_id, 0.0) + candidate_score
                    )
                else:
                    neighbour_scores[neighbour_id] = max(
                        neighbour_scores.get(neighbour_id, float("-inf")),
                        candidate_score,
                    )
                if neighbour_id not in base_scores:
                    added_by_expansion.add(neighbour_id)

        combined_scores = dict(base_scores)
        for passage_id, score in neighbour_scores.items():
            if neighbour_score_mode == "sum" and passage_id in combined_scores:
                combined_scores[passage_id] += score
            else:
                combined_scores[passage_id] = max(
                    combined_scores.get(passage_id, float("-inf")),
                    score,
                )

        sorted_items = sorted(combined_scores.items(), key=lambda x: (-x[1], x[0]))[:final_k]
        expanded_results[query_id] = [
            SearchResult(passage_id=passage_id, score=float(score), rank=rank)
            for rank, (passage_id, score) in enumerate(sorted_items, start=1)
        ]

        source_id, target_id = (None, None)
        if gold_pairs and query_id in gold_pairs:
            source_id, target_id = gold_pairs[query_id]
        final_ids = [r.passage_id for r in expanded_results[query_id]]
        diagnostics.append(
            {
                "query_id": query_id,
                "base_run": base_run.run_name,
                "expanded_run": run_name,
                "seed_k": seed_k,
                "final_k": final_k,
                "expansion_direction": expansion_direction,
                "expansion_weight": expansion_weight,
                "neighbour_score_mode": neighbour_score_mode,
                "seed_count": len(seed_results),
                "expanded_candidate_count": len(combined_scores),
                "added_by_expansion_count": len(added_by_expansion),
                "gold_source_id": source_id,
                "gold_target_id": target_id,
                "source_added_by_expansion": bool(source_id and source_id in added_by_expansion),
                "target_added_by_expansion": bool(target_id and target_id in added_by_expansion),
                "source_in_final": bool(source_id and source_id in final_ids),
                "target_in_final": bool(target_id and target_id in final_ids),
            }
        )

    return (
        RetrievalRun(run_name=run_name, results=expanded_results, k=final_k),
        diagnostics,
    )
