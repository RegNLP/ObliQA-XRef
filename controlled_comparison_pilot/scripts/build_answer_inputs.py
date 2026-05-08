#!/usr/bin/env python3
"""Build answer-generation input JSON for the 5+5+5 pilot.

This script only exports inputs for later GPT-5.2 calls. It does not call any
LLM API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


DATASETS = {
    "obliqa": {
        "samples": ROOT / "samples/obliqa_test5.json",
        "corpus": ROOT / "corpora/obliqa_passages.jsonl",
        "run": ROOT / "runs/obliqa_bm25_top10.trec",
        "out": ROOT / "answer_inputs/obliqa_bm25_gpt52_input.json",
        "qid": "QuestionID",
        "question": "Question",
        "dataset": "ObliQA",
    },
    "obliqa_mp": {
        "samples": ROOT / "samples/obliqa_mp_test5.json",
        "corpus": ROOT / "corpora/obliqa_mp_passages.jsonl",
        "run": ROOT / "runs/obliqa_mp_bm25_top10.trec",
        "out": ROOT / "answer_inputs/obliqa_mp_bm25_gpt52_input.json",
        "qid": "QuestionID",
        "question": "Question",
        "dataset": "ObliQA-MP",
    },
    "xref_adgm": {
        "samples": ROOT / "samples/xref_adgm_test5.json",
        "corpus": ROOT / "corpora/xref_adgm_passages.jsonl",
        "run": ROOT / "runs/xref_adgm_bm25_top10.trec",
        "out": ROOT / "answer_inputs/xref_adgm_bm25_gpt52_input.json",
        "qid": "item_id",
        "question": "question",
        "dataset": "ObliQA-XRef-ADGM",
    },
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_run(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, pid, rank, score, run_name = parts[:6]
            out.setdefault(qid, []).append(pid)
    return out


def build_one(name: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    samples = read_json(cfg["samples"])
    corpus = {str(row["id"]): row for row in read_jsonl(cfg["corpus"])}
    run = read_run(cfg["run"])

    records: list[dict[str, Any]] = []
    for sample in samples:
        qid = str(sample.get(cfg["qid"]) or "")
        retrieved_ids = run.get(qid, [])
        passages = [str(corpus[pid].get("contents") or "") for pid in retrieved_ids if pid in corpus]
        record = {
            "QuestionID": qid,
            "Question": str(sample.get(cfg["question"]) or ""),
            "RetrievedPassages": passages,
            "RetrievedPassageIDs": retrieved_ids,
            "dataset": cfg["dataset"],
            "split": "test",
            "sample_seed": 42,
            "retriever": "bm25",
            "k": 10,
            "relevant_passage_ids": sample.get("relevant_passage_ids") or [],
        }
        if sample.get("gold_answer") is not None:
            record["gold_answer"] = sample.get("gold_answer")
        records.append(record)

    write_json(cfg["out"], records)
    return records


def update_report(sizes: dict[str, int]) -> None:
    report_path = ROOT / "reports/pilot_build_report.json"
    report = read_json(report_path) if report_path.exists() else {"datasets": {}}
    output_names = {
        "obliqa": "answer_inputs/obliqa_bm25_gpt52_input.json",
        "obliqa_mp": "answer_inputs/obliqa_mp_bm25_gpt52_input.json",
        "xref_adgm": "answer_inputs/xref_adgm_bm25_gpt52_input.json",
    }
    for name, size in sizes.items():
        report.setdefault("datasets", {}).setdefault(name, {})
        report["datasets"][name]["answer_input_count"] = size
        report["datasets"][name]["answer_input_path"] = output_names[name]
    write_json(report_path, report)


def main() -> None:
    sizes = {}
    for name, cfg in DATASETS.items():
        print(f"Building answer input for {name}...")
        sizes[name] = len(build_one(name, cfg))
    update_report(sizes)
    print(f"Wrote answer inputs under {ROOT / 'answer_inputs'}")


if __name__ == "__main__":
    main()
