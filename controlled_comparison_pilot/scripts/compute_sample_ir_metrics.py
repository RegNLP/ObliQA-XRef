#!/usr/bin/env python3
"""Compute IR metrics for the selected controlled batch sample.

Uses selected QuestionIDs from the batch answer inputs and saved full BM25
TREC/qrels files. It does not run retrieval.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    import pytrec_eval
except Exception as exc:  # pragma: no cover - environment check
    raise SystemExit(
        "pytrec_eval is required for sample IR metrics. "
        "Use an environment that has pytrec_eval installed, for example /opt/anaconda3/bin/python."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
K = 10
RETRIEVER = "bm25"
RETRIEVAL_SETTING = "controlled_shared_corpus_canonical_bm25"
RETRIEVAL_CORPUS = "shared_adgm_13015"

XREF_TEST_PATH = REPO_ROOT / "ObliQA-XRef_Out_Datasets/final_large_merged/ObliQA-XRef-ADGM-ALL/test.jsonl"

DATASETS = {
    "obliqa": {
        "label": "ObliQA",
        "input": "obliqa_bm25_gpt52_input.json",
        "qrels": ROOT / "full_ir/qrels/obliqa_test.qrels",
        "run": ROOT / "full_ir/runs/obliqa_bm25_top10.trec",
        "xref": False,
    },
    "obliqa_mp": {
        "label": "ObliQA-MP",
        "input": "obliqa_mp_bm25_gpt52_input.json",
        "qrels": ROOT / "full_ir/qrels/obliqa_mp_test.qrels",
        "run": ROOT / "full_ir/runs/obliqa_mp_bm25_top10.trec",
        "xref": False,
    },
    "xref_adgm": {
        "label": "ObliQA-XRef-ADGM",
        "input": "xref_adgm_bm25_gpt52_input.json",
        "qrels": ROOT / "full_ir/qrels/xref_adgm_test.qrels",
        "run": ROOT / "full_ir/runs/xref_adgm_bm25_top10.trec",
        "xref": True,
    },
}

FIELDNAMES = [
    "dataset_or_slice",
    "method_type",
    "sampling",
    "sample_size",
    "retriever",
    "retrieval_setting",
    "retrieval_corpus",
    "k",
    "Recall@10",
    "MAP@10",
    "nDCG@10",
    "Hit@10",
    "RelCount@10",
    "Both@10",
    "SRC-only@10",
    "TGT-only@10",
    "Neither@10",
    "PairMRR",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 4:
                raise ValueError(f"Malformed qrels line {path}:{lineno}: {line}")
            qid, _zero, docid, rel = parts
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def read_run(path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 6:
                raise ValueError(f"Malformed TREC run line {path}:{lineno}: {line}")
            qid, _q0, docid, _rank, score, _tag = parts
            run.setdefault(qid, {})[docid] = float(score)
    return run


def topk_docids(scores: dict[str, float], k: int) -> list[str]:
    return [docid for docid, _score in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]


def selected_qids(input_root: Path, dataset_key: str) -> list[str]:
    path = input_root / "general" / DATASETS[dataset_key]["input"]
    rows = read_json(path)
    return [str(row["QuestionID"]) for row in rows]


def filter_qrels(qrels: dict[str, dict[str, int]], qids: set[str]) -> dict[str, dict[str, int]]:
    missing = sorted(qids - set(qrels))
    if missing:
        raise ValueError(f"Selected qids missing from qrels: {missing[:5]} ({len(missing)} total)")
    return {qid: qrels[qid] for qid in sorted(qids)}


def compute_common_metrics(qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]]) -> dict[str, float]:
    qids = set(qrels)
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {f"recall_{K}", f"map_cut_{K}", f"ndcg_cut_{K}"},
    )
    results = evaluator.evaluate({qid: run.get(qid, {}) for qid in qids})
    denom = max(1, len(qids))
    hit_count = 0
    rel_count = 0
    for qid in qids:
        relevant = {docid for docid, rel in qrels[qid].items() if rel > 0}
        rel_at_k = relevant & set(topk_docids(run.get(qid, {}), K))
        hit_count += 1 if rel_at_k else 0
        rel_count += len(rel_at_k)
    return {
        "Recall@10": sum(results[qid].get(f"recall_{K}", 0.0) for qid in qids) / denom,
        "MAP@10": sum(results[qid].get(f"map_cut_{K}", 0.0) for qid in qids) / denom,
        "nDCG@10": sum(results[qid].get(f"ndcg_cut_{K}", 0.0) for qid in qids) / denom,
        "Hit@10": hit_count / denom,
        "RelCount@10": rel_count / denom,
    }


def xref_metadata() -> dict[str, dict[str, Any]]:
    return {str(row["item_id"]): row for row in read_jsonl(XREF_TEST_PATH)}


def compute_pair_metrics(
    run: dict[str, dict[str, float]],
    meta: dict[str, dict[str, Any]],
    qids: set[str],
) -> dict[str, float]:
    denom = max(1, len(qids))
    both = src_only = tgt_only = neither = 0
    pair_mrr = 0.0
    for qid in qids:
        row = meta[qid]
        src = str(row.get("source_passage_id") or "")
        tgt = str(row.get("target_passage_id") or "")
        ranked_all = topk_docids(run.get(qid, {}), len(run.get(qid, {})))
        ranked_k = ranked_all[:K]
        found_src = src in ranked_k
        found_tgt = tgt in ranked_k
        if found_src and found_tgt:
            both += 1
        elif found_src:
            src_only += 1
        elif found_tgt:
            tgt_only += 1
        else:
            neither += 1
        seen_src = seen_tgt = False
        for rank, docid in enumerate(ranked_all, start=1):
            seen_src = seen_src or docid == src
            seen_tgt = seen_tgt or docid == tgt
            if seen_src and seen_tgt:
                pair_mrr += 1.0 / rank
                break
    return {
        "Both@10": both / denom,
        "SRC-only@10": src_only / denom,
        "TGT-only@10": tgt_only / denom,
        "Neither@10": neither / denom,
        "PairMRR": pair_mrr / denom,
    }


def make_row(
    *,
    name: str,
    sampling: str,
    qids: set[str],
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    xref_meta: dict[str, dict[str, Any]] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    filtered_qrels = filter_qrels(qrels, qids)
    common = compute_common_metrics(filtered_qrels, run)
    row: dict[str, Any] = {
        "dataset_or_slice": name,
        "method_type": "controlled_answer_batch",
        "sampling": sampling,
        "sample_size": len(qids),
        "retriever": RETRIEVER,
        "retrieval_setting": RETRIEVAL_SETTING,
        "retrieval_corpus": RETRIEVAL_CORPUS,
        "k": K,
        **common,
        "Both@10": None,
        "SRC-only@10": None,
        "TGT-only@10": None,
        "Neither@10": None,
        "PairMRR": None,
        "notes": notes,
    }
    if xref_meta is not None:
        row.update(compute_pair_metrics(run, xref_meta, qids))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["| " + " | ".join(FIELDNAMES) + " |", "| " + " | ".join(["---"] * len(FIELDNAMES)) + " |"]
    for row in rows:
        vals = []
        for field in FIELDNAMES:
            val = row.get(field)
            vals.append(f"{val:.6f}" if isinstance(val, float) else "" if val is None else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    output_root = args.output_root
    rows: list[dict[str, Any]] = []

    xmeta = xref_metadata()
    for dataset_key, cfg in DATASETS.items():
        qids = set(selected_qids(input_root, dataset_key))
        qrels = read_qrels(cfg["qrels"])
        run = read_run(cfg["run"])
        rows.append(
            make_row(
                name=cfg["label"],
                sampling="selected_sample",
                qids=qids,
                qrels=qrels,
                run=run,
                xref_meta=xmeta if cfg["xref"] else None,
            )
        )
        if cfg["xref"]:
            breakdowns = [
                ("ObliQA-XRef-ADGM-ALL selected", lambda r: True),
                ("DPEL selected", lambda r: r.get("generation_method") == "DPEL"),
                ("SCHEMA selected", lambda r: r.get("generation_method") == "SCHEMA"),
                ("DPEL + mixed_difficulty selected", lambda r: r.get("generation_method") == "DPEL" and r.get("sampling_regime") == "mixed_difficulty"),
                ("DPEL + hard_enriched selected", lambda r: r.get("generation_method") == "DPEL" and r.get("sampling_regime") == "hard_enriched"),
                ("SCHEMA + mixed_difficulty selected", lambda r: r.get("generation_method") == "SCHEMA" and r.get("sampling_regime") == "mixed_difficulty"),
                ("SCHEMA + hard_enriched selected", lambda r: r.get("generation_method") == "SCHEMA" and r.get("sampling_regime") == "hard_enriched"),
            ]
            for name, predicate in breakdowns:
                slice_qids = {qid for qid in qids if predicate(xmeta[qid])}
                if slice_qids:
                    rows.append(
                        make_row(
                            name=name,
                            sampling="xref_selected_breakdown",
                            qids=slice_qids,
                            qrels=qrels,
                            run=run,
                            xref_meta=xmeta,
                        )
                    )

    write_json(output_root / "sample_ir_metrics.json", rows)
    write_csv(output_root / "sample_ir_metrics.csv", rows)
    write_md(output_root / "sample_ir_metrics.md", rows)
    print(f"Wrote {output_root / 'sample_ir_metrics.csv'}")


if __name__ == "__main__":
    main()
