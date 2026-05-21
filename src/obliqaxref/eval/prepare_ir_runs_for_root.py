from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from obliqaxref.curate.ir import (
    BM25Retriever,
    DenseRetriever,
    RRFFusion,
    CrossEncoderReranker,
)
from obliqaxref.curate.ir.xref_expand import load_xref_graph, expand_retrieval_run

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_test_items(root: Path, corpus: str) -> list[dict[str, Any]]:
    path = root / f"ObliQA-XRef-{corpus.upper()}-ALL" / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing test split: {path}")
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                import json
                items.append(json.loads(line))
    return items


def _load_corpus(corpus: str) -> list[dict[str, str]]:
    # Use adapter processed corpus
    path = Path(f"data/{corpus}/processed/passage_corpus.jsonl")
    if not path.exists():
        # Fallback to runs/adapter if available
        path = Path(f"runs/adapter_{corpus}/processed/passage_corpus.jsonl")
    passages = []
    if not path.exists():
        raise FileNotFoundError(f"Missing passage corpus for {corpus}: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            import json
            row = json.loads(line)
            pid = row.get("pid") or row.get("passage_uid")
            text = row.get("text") or row.get("passage")
            if pid and text:
                passages.append({"passage_id": str(pid), "text": str(text).strip()})
    return passages


def _prepare_queries(items: list[dict[str, Any]]) -> dict[str, str]:
    q = {}
    for it in items:
        iid = it.get("item_id")
        question = (it.get("question") or "").strip()
        if iid and question:
            q[iid] = question
    return q


def generate_ir_runs_for_root(corpus: str, root_dir: Path, k: int = 20, methods: list[str] | None = None) -> None:
    items = _load_test_items(root_dir, corpus)
    passages = _load_corpus(corpus)
    queries = _prepare_queries(items)
    dst_dir = root_dir / f"ObliQA-XRef-{corpus.upper()}-ALL"
    dst_dir.mkdir(parents=True, exist_ok=True)

    requested = set(methods or [
        "bm25",
        "ft_e5",
        "rrf_bm25_e5",
        "ce_rerank_union200",
        "bm25_xref_expand",
        "e5_xref_expand",
        "rrf_xref_expand",
    ])

    need_bm25 = any(m in requested for m in ["bm25", "rrf_bm25_e5", "ce_rerank_union200", "bm25_xref_expand", "rrf_xref_expand"]) \
        or ("rrf_bm25_e5" in requested) or ("ce_rerank_union200" in requested)
    need_e5 = any(m in requested for m in ["ft_e5", "rrf_bm25_e5", "ce_rerank_union200", "e5_xref_expand", "rrf_xref_expand"]) \
        or ("rrf_bm25_e5" in requested) or ("ce_rerank_union200" in requested)
    need_rrf = "rrf_bm25_e5" in requested or "rrf_xref_expand" in requested
    need_ce = "ce_rerank_union200" in requested
    need_xref = any(m in requested for m in ["bm25_xref_expand", "e5_xref_expand", "rrf_xref_expand"]) \
        or "rrf_xref_expand" in requested

    bm25_run = None
    e5_run = None
    rrf_run = None
    ce_run = None

    # Build only what is needed
    if need_bm25:
        logger.info("Building BM25 for %s over %d passages", corpus, len(passages))
        bm25 = BM25Retriever(passages)
        logger.info("Running BM25 (k=%d) for %s", k, corpus)
        bm25_run = bm25.batch_search(queries, k=k)

    if need_e5:
        logger.info("Building E5 for %s over %d passages", corpus, len(passages))
        e5 = DenseRetriever(passages, model_name="intfloat/e5-base-v2")
        logger.info("Running E5 (k=%d) for %s", k, corpus)
        e5_run = e5.batch_search(queries, k=k)

    if need_rrf:
        if bm25_run is None or e5_run is None:
            logger.info("RRF requires BM25 and E5; computing missing dependencies in-memory")
            if bm25_run is None:
                bm25 = BM25Retriever(passages)
                bm25_run = bm25.batch_search(queries, k=k)
            if e5_run is None:
                e5 = DenseRetriever(passages, model_name="intfloat/e5-base-v2")
                e5_run = e5.batch_search(queries, k=k)
        rrf = RRFFusion(k=60)
        rrf_run = rrf.fuse([bm25_run, e5_run], run_name="rrf_bm25_e5")

    if need_ce:
        if bm25_run is None or e5_run is None:
            logger.info("CE reranker requires BM25 and E5; computing missing dependencies in-memory")
            if bm25_run is None:
                bm25 = BM25Retriever(passages)
                bm25_run = bm25.batch_search(queries, k=k)
            if e5_run is None:
                e5 = DenseRetriever(passages, model_name="intfloat/e5-base-v2")
                e5_run = e5.batch_search(queries, k=k)
        ce = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        passage_index = {p["passage_id"]: p for p in passages}
        ce_run = ce.rerank_union([bm25_run, e5_run], passage_index, queries=queries, union_k=200, final_k=k, run_name="ce_rerank_union200")

    expanded_runs = []
    if need_xref:
        # Build passage_index lazily if needed
        try:
            passage_index
        except NameError:
            passage_index = {p["passage_id"]: p for p in passages}
        xref_csv = Path(f"data/{corpus}/processed/crossref_resolved.cleaned.csv")
        if xref_csv.exists():
            try:
                graph = load_xref_graph(xref_csv, set(passage_index.keys()))
                # Expand only requested base runs
                if "bm25_xref_expand" in requested:
                    if bm25_run is None:
                        bm25 = BM25Retriever(passages)
                        bm25_run = bm25.batch_search(queries, k=k)
                    erun, _ = expand_retrieval_run(bm25_run, graph, run_name="bm25_xref_expand", seed_k=k, final_k=k)
                    expanded_runs.append(erun)
                if "e5_xref_expand" in requested:
                    if e5_run is None:
                        e5 = DenseRetriever(passages, model_name="intfloat/e5-base-v2")
                        e5_run = e5.batch_search(queries, k=k)
                    erun, _ = expand_retrieval_run(e5_run, graph, run_name="e5_xref_expand", seed_k=k, final_k=k)
                    expanded_runs.append(erun)
                if "rrf_xref_expand" in requested:
                    if rrf_run is None:
                        if bm25_run is None:
                            bm25 = BM25Retriever(passages)
                            bm25_run = bm25.batch_search(queries, k=k)
                        if e5_run is None:
                            e5 = DenseRetriever(passages, model_name="intfloat/e5-base-v2")
                            e5_run = e5.batch_search(queries, k=k)
                        rrf = RRFFusion(k=60)
                        rrf_run = rrf.fuse([bm25_run, e5_run], run_name="rrf_bm25_e5")
                    erun, _ = expand_retrieval_run(rrf_run, graph, run_name="rrf_xref_expand", seed_k=k, final_k=k)
                    expanded_runs.append(erun)
            except Exception as e:
                logger.warning("XRef expansion failed for %s: %s", corpus, e)
        else:
            logger.warning("Crossref graph not found for %s: %s", corpus, xref_csv)

    # Collect runs to write based on requested methods
    name_to_run = {}
    if bm25_run is not None:
        name_to_run["bm25"] = bm25_run
    if e5_run is not None:
        name_to_run["ft_e5"] = e5_run
    if rrf_run is not None:
        name_to_run["rrf_bm25_e5"] = rrf_run
    if ce_run is not None:
        name_to_run["ce_rerank_union200"] = ce_run
    for erun in expanded_runs:
        name_to_run[erun.run_name] = erun

    for method_name, run in name_to_run.items():
        if method_name not in requested:
            logger.info("Computed dependency run %s in-memory (not requested); not writing.", method_name)
            continue
        trec = dst_dir / f"{method_name}.trec"
        with trec.open("w", encoding="utf-8") as f:
            for qid, results in run.results.items():
                for r in results:
                    f.write(f"{qid} Q0 {r.passage_id} {r.rank} {r.score} {method_name}\n")
        logger.info("Wrote %s", trec)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate fresh IR runs for a finalized dataset root (per corpus)")
    ap.add_argument("--root", required=True, help="Finalized dataset root (contains ObliQA-XRef-<CORPUS>-ALL)")
    ap.add_argument("--corpus", default="both", choices=["adgm", "ukfin", "both"], help="Corpus to prepare runs for")
    ap.add_argument("--k", type=int, default=20, help="Top-k cutoff for runs (default: 20)")
    ap.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=(
            "Subset of methods to generate: bm25 ft_e5 rrf_bm25_e5 ce_rerank_union200 "
            "bm25_xref_expand e5_xref_expand rrf_xref_expand (default: all)"
        ),
    )
    args = ap.parse_args(argv)

    root = Path(args.root)
    corpora = ["ukfin", "adgm"] if args.corpus == "both" else [args.corpus]
    for c in corpora:
        generate_ir_runs_for_root(c, root, k=args.k, methods=args.methods)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
