from __future__ import annotations

import csv
from pathlib import Path

from obliqaxref.curate.ir.types import RetrievalRun, SearchResult
from obliqaxref.curate.ir.xref_expand import expand_retrieval_run, load_xref_graph


def _write_xrefs(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["SourceID", "TargetID", "SourcePassageID", "TargetPassageID"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_xref_graph_keeps_only_active_passage_ids(tmp_path):
    xref_file = tmp_path / "crossref_resolved.cleaned.csv"
    _write_xrefs(
        xref_file,
        [
            {"SourceID": "s1", "TargetID": "t1"},
            {"SourceID": "s1", "TargetID": "missing"},
            {"SourcePassageID": "display-src", "TargetPassageID": "display-tgt"},
        ],
    )

    graph = load_xref_graph(xref_file, {"s1", "t1", "display-src", "display-tgt"})

    assert graph.outgoing["s1"] == ["t1"]
    assert graph.incoming["t1"] == ["s1"]
    assert graph.outgoing["display-src"] == ["display-tgt"]
    assert "missing" not in graph.incoming


def test_expand_retrieval_run_adds_outgoing_neighbour_and_reranks():
    base = RetrievalRun(
        run_name="bm25",
        k=2,
        results={
            "q1": [
                SearchResult(passage_id="seed", score=10.0, rank=1),
                SearchResult(passage_id="other", score=2.0, rank=2),
            ]
        },
    )
    graph = load_xref_graph_from_edges({"seed": ["target"]})

    expanded, diagnostics = expand_retrieval_run(
        base,
        graph,
        run_name="bm25_xref_expand",
        seed_k=1,
        final_k=3,
        expansion_direction="outgoing",
        expansion_weight=0.8,
        neighbour_score_mode="max",
        gold_pairs={"q1": ("source", "target")},
    )

    results = expanded.results["q1"]
    assert [r.passage_id for r in results] == ["seed", "target"]
    assert results[1].score == 8.0
    assert diagnostics[0]["target_added_by_expansion"] is True
    assert diagnostics[0]["target_in_final"] is True


def test_expand_retrieval_run_sum_mode_accumulates_multiple_parent_scores():
    base = RetrievalRun(
        run_name="rrf_bm25_e5",
        k=2,
        results={
            "q1": [
                SearchResult(passage_id="p1", score=5.0, rank=1),
                SearchResult(passage_id="p2", score=4.0, rank=2),
            ]
        },
    )
    graph = load_xref_graph_from_edges({"p1": ["shared"], "p2": ["shared"]})

    expanded, _ = expand_retrieval_run(
        base,
        graph,
        run_name="rrf_xref_expand",
        seed_k=2,
        final_k=3,
        expansion_direction="outgoing",
        expansion_weight=0.5,
        neighbour_score_mode="sum",
    )

    scores = {r.passage_id: r.score for r in expanded.results["q1"]}
    assert scores["shared"] == 4.5


def load_xref_graph_from_edges(edges: dict[str, list[str]]):
    outgoing = {source: list(targets) for source, targets in edges.items()}
    incoming: dict[str, list[str]] = {}
    for source, targets in edges.items():
        for target in targets:
            incoming.setdefault(target, []).append(source)
    from obliqaxref.curate.ir.xref_expand import XRefGraph

    return XRefGraph(outgoing=outgoing, incoming=incoming)
