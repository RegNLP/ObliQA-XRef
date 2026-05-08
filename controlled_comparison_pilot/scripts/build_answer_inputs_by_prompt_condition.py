#!/usr/bin/env python3
"""Build 5+5+5 answer-generation inputs for general and task-aware prompts.

This script only prepares JSON inputs. It does not call any LLM API.
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
        "qid": "QuestionID",
        "question": "Question",
        "dataset_name": "ObliQA",
        "task_aware_prompt_file": "controlled_comparison_pilot/prompts/task_aware_obliqa_prompt.txt",
        "task_aware_prompt_version": "controlled_bm25_gpt52_task_aware_obliqa_v1",
    },
    "obliqa_mp": {
        "samples": ROOT / "samples/obliqa_mp_test5.json",
        "corpus": ROOT / "corpora/obliqa_mp_passages.jsonl",
        "run": ROOT / "runs/obliqa_mp_bm25_top10.trec",
        "qid": "QuestionID",
        "question": "Question",
        "dataset_name": "ObliQA-MP",
        "task_aware_prompt_file": "controlled_comparison_pilot/prompts/task_aware_obliqa_mp_prompt.txt",
        "task_aware_prompt_version": "controlled_bm25_gpt52_task_aware_obliqa_mp_v1",
    },
    "xref_adgm": {
        "samples": ROOT / "samples/xref_adgm_test5.json",
        "corpus": ROOT / "corpora/xref_adgm_passages.jsonl",
        "run": ROOT / "runs/xref_adgm_bm25_top10.trec",
        "qid": "item_id",
        "question": "question",
        "dataset_name": "ObliQA-XRef-ADGM",
        "task_aware_prompt_file": "controlled_comparison_pilot/prompts/task_aware_xref_prompt.txt",
        "task_aware_prompt_version": "controlled_bm25_gpt52_task_aware_xref_v1",
    },
}


GENERAL_PROMPT_FILE = "controlled_comparison_pilot/prompts/general_grounded_prompt.txt"
GENERAL_PROMPT_VERSION = "controlled_bm25_gpt52_general_grounded_v1"


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


def read_trec_run(path: Path) -> dict[str, list[str]]:
    by_qid: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, pid = parts[0], parts[1], parts[2]
            by_qid.setdefault(qid, []).append(pid)
    return by_qid


def format_retrieved_passage(rank: int, passage_id: str, passage_text: str) -> str:
    return f"Passage {rank}\nPassage ID: {passage_id}\nText: {passage_text}"


def build_dataset_records(
    cfg: dict[str, Any],
    prompt_condition: str,
    prompt_file: str,
    prompt_version: str,
) -> list[dict[str, Any]]:
    samples = read_json(cfg["samples"])
    corpus = {str(row["id"]): row for row in read_jsonl(cfg["corpus"])}
    run = read_trec_run(cfg["run"])

    records: list[dict[str, Any]] = []
    for sample in samples:
        qid = str(sample.get(cfg["qid"]) or "")
        question = str(sample.get(cfg["question"]) or "")
        top_ids = run.get(qid, [])[:10]

        retrieved_passage_ids: list[str] = []
        retrieved_passages: list[str] = []
        for idx, pid in enumerate(top_ids, start=1):
            row = corpus.get(pid)
            if row is None:
                continue
            text = str(row.get("contents") or "")
            retrieved_passage_ids.append(pid)
            retrieved_passages.append(format_retrieved_passage(idx, pid, text))

        record: dict[str, Any] = {
            "QuestionID": qid,
            "Question": question,
            "RetrievedPassages": retrieved_passages,
            "RetrievedPassageIDs": retrieved_passage_ids,
            "dataset": cfg["dataset_name"],
            "split": "test",
            "sample_seed": 42,
            "retriever": "bm25",
            "k": 10,
            "prompt_condition": prompt_condition,
            "prompt_version": prompt_version,
            "prompt_file": prompt_file,
            "relevant_passage_ids": sample.get("relevant_passage_ids") or [],
        }
        if sample.get("gold_answer") is not None:
            record["gold_answer"] = sample.get("gold_answer")
        records.append(record)
    return records


def output_path(condition: str, dataset_key: str) -> Path:
    return ROOT / "answer_inputs" / condition / f"{dataset_key}_bm25_gpt52_input.json"


def main() -> None:
    for dataset_key, cfg in DATASETS.items():
        general_records = build_dataset_records(
            cfg=cfg,
            prompt_condition="general",
            prompt_file=GENERAL_PROMPT_FILE,
            prompt_version=GENERAL_PROMPT_VERSION,
        )
        write_json(output_path("general", dataset_key), general_records)

        task_records = build_dataset_records(
            cfg=cfg,
            prompt_condition="task_aware",
            prompt_file=cfg["task_aware_prompt_file"],
            prompt_version=cfg["task_aware_prompt_version"],
        )
        write_json(output_path("task_aware", dataset_key), task_records)

        print(
            f"{dataset_key}: general={len(general_records)} "
            f"task_aware={len(task_records)}"
        )


if __name__ == "__main__":
    main()
