#!/usr/bin/env python3
"""Run BM25(passage) top-10 retrieval for the controlled-comparison pilot."""

from __future__ import annotations

import heapq
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from rank_bm25 import BM25Okapi as RankBM25Okapi
except Exception:  # pragma: no cover - optional dependency
    RankBM25Okapi = None


ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
TOP_K = 10


DATASETS = {
    "obliqa": {
        "samples": ROOT / "samples/obliqa_test5.json",
        "corpus": ROOT / "corpora/obliqa_passages.jsonl",
        "run": ROOT / "runs/obliqa_bm25_top10.trec",
        "qid": "QuestionID",
        "question": "Question",
    },
    "obliqa_mp": {
        "samples": ROOT / "samples/obliqa_mp_test5.json",
        "corpus": ROOT / "corpora/obliqa_mp_passages.jsonl",
        "run": ROOT / "runs/obliqa_mp_bm25_top10.trec",
        "qid": "QuestionID",
        "question": "Question",
    },
    "xref_adgm": {
        "samples": ROOT / "samples/xref_adgm_test5.json",
        "corpus": ROOT / "corpora/xref_adgm_passages.jsonl",
        "run": ROOT / "runs/xref_adgm_bm25_top10.trec",
        "qid": "item_id",
        "question": "question",
    },
}


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "")]


class LocalBM25:
    def __init__(self, corpus: list[list[str]], *, k1: float = 0.9, b: float = 0.4) -> None:
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
        for tf, dl in zip(self.term_freqs, self.doc_len):
            norm = self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            score = 0.0
            for term in q_terms:
                freq = tf.get(term, 0)
                if freq <= 0:
                    continue
                score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1.0)) / (freq + norm)
            scores.append(score)
        return scores


def build_bm25(tokens: list[list[str]]) -> Any:
    if RankBM25Okapi is not None:
        return RankBM25Okapi(tokens, k1=0.9, b=0.4)
    return LocalBM25(tokens, k1=0.9, b=0.4)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    if not path.exists():
        return qrels
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                qrels.setdefault(parts[0], set()).add(parts[2])
    return qrels


def run_one(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    samples = read_json(cfg["samples"])
    corpus = read_jsonl(cfg["corpus"])
    if not corpus:
        raise ValueError(f"No corpus rows for {name}")

    doc_ids = [str(row["id"]) for row in corpus]
    texts = [str(row.get("contents") or "") for row in corpus]
    bm25 = build_bm25([tokenize(text) for text in texts])

    cfg["run"].parent.mkdir(parents=True, exist_ok=True)
    run_rows = 0
    with cfg["run"].open("w", encoding="utf-8") as f:
        for sample in samples:
            qid = str(sample.get(cfg["qid"]) or "")
            question = str(sample.get(cfg["question"]) or "")
            scores = bm25.get_scores(tokenize(question))
            top = heapq.nlargest(TOP_K, enumerate(scores), key=lambda x: (float(x[1]), -x[0]))
            for rank, (idx, score) in enumerate(top, start=1):
                f.write(f"{qid} Q0 {doc_ids[idx]} {rank} {float(score):.6f} bm25\n")
                run_rows += 1

    qrels_path = ROOT / f"qrels/{'obliqa_mp' if name == 'obliqa_mp' else name}_test5.qrels"
    if name == "xref_adgm":
        qrels_path = ROOT / "qrels/xref_adgm_test5.qrels"
    qrels = read_qrels(qrels_path)
    runs_by_qid: dict[str, list[str]] = {}
    with cfg["run"].open("r", encoding="utf-8") as f:
        for line in f:
            qid, _, pid, *_ = line.split()
            runs_by_qid.setdefault(qid, []).append(pid)

    any_hit = 0
    all_hit = 0
    missing_queries = 0
    for sample in samples:
        qid = str(sample.get(cfg["qid"]) or "")
        relevant = qrels.get(qid, set())
        retrieved = set(runs_by_qid.get(qid, []))
        if not retrieved:
            missing_queries += 1
        if relevant and relevant & retrieved:
            any_hit += 1
        if relevant and relevant <= retrieved:
            all_hit += 1

    return {
        "run_path": str(cfg["run"].relative_to(ROOT)),
        "run_rows": run_rows,
        "queries": len(samples),
        "top10_coverage": {
            "queries_with_any_relevant_at10": any_hit,
            "queries_with_all_relevant_at10": all_hit,
            "queries_with_no_run_rows": missing_queries,
        },
    }


def update_report(results: dict[str, Any]) -> None:
    report_path = ROOT / "reports/pilot_build_report.json"
    report = read_json(report_path) if report_path.exists() else {"datasets": {}}
    for name, result in results.items():
        report.setdefault("datasets", {}).setdefault(name, {})
        report["datasets"][name]["bm25_run_row_count"] = result["run_rows"]
        report["datasets"][name]["top10_coverage"] = result["top10_coverage"]
        report["datasets"][name]["bm25_run_path"] = result["run_path"]
    write_json(report_path, report)


def main() -> None:
    results = {}
    for name, cfg in DATASETS.items():
        print(f"Running BM25 pilot for {name}...")
        results[name] = run_one(name, cfg)
    update_report(results)
    print(f"Wrote BM25 runs under {ROOT / 'runs'}")


if __name__ == "__main__":
    main()
