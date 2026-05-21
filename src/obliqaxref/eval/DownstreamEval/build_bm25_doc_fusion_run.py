#!/usr/bin/env python3
"""Build historical document-aware BM25 fusion runs for ObliQA-XRef.

This reconstructs the BM25(fusion) setting used for the ObliQA family
comparison: passage BM25 scores are fused with document BM25 scores after
per-query score normalization.

The implementation uses rank_bm25 when installed and falls back to a small
local BM25 implementation. It does not use dense retrieval, RRF, or
ObliQA-XRef's rrf_bm25_e5 runs.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from rank_bm25 import BM25Okapi as RankBM25Okapi
except Exception:  # pragma: no cover - optional dependency
    RankBM25Okapi = None


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "")]


class LocalBM25:
    """Small BM25Okapi-compatible fallback for lightweight run construction."""

    def __init__(self, corpus: list[list[str]], *, k1: float = 0.9, b: float = 0.4) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.term_freqs = [Counter(doc) for doc in corpus]
        df: Counter[str] = Counter()
        for doc in corpus:
            df.update(set(doc))
        n_docs = max(1, len(corpus))
        self.idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        q_terms = set(query_tokens)
        for tf, dl in zip(self.term_freqs, self.doc_len, strict=False):
            score = 0.0
            denom_norm = self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            for term in q_terms:
                freq = tf.get(term, 0)
                if freq <= 0:
                    continue
                score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1.0)) / (
                    freq + denom_norm
                )
            scores.append(score)
        return scores


def build_bm25(corpus_tokens: list[list[str]], *, k1: float, b: float) -> Any:
    if RankBM25Okapi is not None:
        return RankBM25Okapi(corpus_tokens, k1=k1, b=b)
    return LocalBM25(corpus_tokens, k1=k1, b=b)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_queries(path: Path) -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    for obj in read_jsonl(path):
        qid = str(obj.get("item_id") or "")
        question = str(obj.get("question") or "")
        if qid and question:
            queries.append((qid, question))
    return queries


def load_passages(path: Path) -> list[dict[str, str]]:
    passages: list[dict[str, str]] = []
    for obj in read_jsonl(path):
        pid = str(obj.get("passage_uid") or "")
        text = str(obj.get("passage") or "")
        doc_id = str(obj.get("doc_id") or "")
        if pid and text and doc_id:
            passages.append({"passage_uid": pid, "passage": text, "doc_id": doc_id})
    if not passages:
        raise ValueError(f"No usable passages found in {path}")
    return passages


def build_documents(passages: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    doc_texts: defaultdict[str, list[str]] = defaultdict(list)
    for row in passages:
        doc_texts[row["doc_id"]].append(row["passage"])
    doc_ids = sorted(doc_texts)
    docs = ["\n".join(doc_texts[doc_id]) for doc_id in doc_ids]
    return doc_ids, docs


def normalize(scores: Iterable[float], method: str) -> list[float]:
    vals = [float(s) for s in scores]
    if not vals or method == "none":
        return vals
    if method == "minmax":
        lo = min(vals)
        hi = max(vals)
        if hi <= lo:
            return [0.0 for _ in vals]
        return [(v - lo) / (hi - lo) for v in vals]
    if method == "zscore":
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var)
        if std <= 0:
            return [0.0 for _ in vals]
        return [(v - mean) / std for v in vals]
    raise ValueError(f"Unsupported normalization: {method}")


def write_trec(
    *,
    output_trec: Path,
    queries: list[tuple[str, str]],
    passages: list[dict[str, str]],
    passage_bm25: Any,
    doc_bm25: Any,
    doc_ids: list[str],
    doc_weight: float,
    top_k: int,
    normalization: str,
) -> None:
    doc_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    output_trec.parent.mkdir(parents=True, exist_ok=True)

    with output_trec.open("w", encoding="utf-8") as out:
        for qid, question in queries:
            q_tokens = tokenize(question)
            passage_scores = [float(s) for s in passage_bm25.get_scores(q_tokens)]
            doc_scores = [float(s) for s in doc_bm25.get_scores(q_tokens)]

            passage_norm = normalize(passage_scores, normalization)
            doc_norm = normalize(doc_scores, normalization)

            ranked: list[tuple[float, str]] = []
            for idx, passage in enumerate(passages):
                doc_id = passage["doc_id"]
                doc_score = doc_norm[doc_index[doc_id]] if doc_id in doc_index else 0.0
                fused_score = passage_norm[idx] + doc_weight * doc_score
                ranked.append((fused_score, passage["passage_uid"]))

            top = heapq.nlargest(top_k, ranked, key=lambda row: (row[0], row[1]))
            for rank, (score, passage_uid) in enumerate(top, start=1):
                out.write(f"{qid} Q0 {passage_uid} {rank} {score:.12f} bm25_doc_fusion\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--test_jsonl",
        default=(
            "ObliQA-XRef_Out_Datasets/final_large_merged/"
            "ObliQA-XRef-ADGM-ALL/test.jsonl"
        ),
        help="ObliQA-XRef test JSONL containing item_id and question",
    )
    ap.add_argument(
        "--passage_corpus",
        default="runs/adapter_adgm/processed/passage_corpus.jsonl",
        help="Passage corpus JSONL containing passage_uid, passage, and doc_id",
    )
    ap.add_argument(
        "--output_trec",
        default=(
            "ObliQA-XRef_Out_Datasets/final_large_merged/"
            "ObliQA-XRef-ADGM-ALL/bm25_fusion.trec"
        ),
        help="Output passage-level TREC run path",
    )
    ap.add_argument("--doc_weight", type=float, default=0.1)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--normalization", choices=["minmax", "zscore", "none"], default="minmax")
    ap.add_argument("--bm25_k1", type=float, default=0.9)
    ap.add_argument("--bm25_b", type=float, default=0.4)
    args = ap.parse_args()

    queries = load_queries(Path(args.test_jsonl))
    passages = load_passages(Path(args.passage_corpus))
    doc_ids, doc_texts = build_documents(passages)

    passage_bm25 = build_bm25(
        [tokenize(row["passage"]) for row in passages],
        k1=args.bm25_k1,
        b=args.bm25_b,
    )
    doc_bm25 = build_bm25([tokenize(text) for text in doc_texts], k1=args.bm25_k1, b=args.bm25_b)

    write_trec(
        output_trec=Path(args.output_trec),
        queries=queries,
        passages=passages,
        passage_bm25=passage_bm25,
        doc_bm25=doc_bm25,
        doc_ids=doc_ids,
        doc_weight=args.doc_weight,
        top_k=args.top_k,
        normalization=args.normalization,
    )


if __name__ == "__main__":
    main()
