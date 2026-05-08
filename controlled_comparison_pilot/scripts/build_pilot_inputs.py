#!/usr/bin/env python3
"""Build 5+5+5 controlled-comparison pilot inputs.

This script creates only local pilot artifacts: sampled examples, passage
corpora, qrels, and an initial build report. It does not call any LLM API and
does not run RePASs.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

XREF_TEST = REPO_ROOT / "ObliQA-XRef_Out_Datasets/final_large_merged/ObliQA-XRef-ADGM-ALL/test.jsonl"
XREF_CORPUS = REPO_ROOT / "runs/adapter_adgm/processed/passage_corpus.jsonl"
ID_SAFE_RE = re.compile(r"\s+")


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def text_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:n]


def safe_id(value: str) -> str:
    value = ID_SAFE_RE.sub("_", str(value or "").strip())
    return value.replace("/", "_")


def passage_text(passage: dict[str, Any]) -> str:
    return str(passage.get("Passage") or passage.get("passage") or passage.get("text") or "")


def doc_field(passage: dict[str, Any]) -> str:
    return str(passage.get("DocumentID") or passage.get("DocumentId") or passage.get("doc_id") or "")


def passage_id_field(passage: dict[str, Any]) -> str:
    return str(passage.get("PassageID") or passage.get("passage_id") or passage.get("ID") or passage.get("id") or "")


class IdRegistry:
    def __init__(self) -> None:
        self.base_to_text_hash: dict[str, str] = {}
        self.collision_warnings: list[dict[str, str]] = []

    def assign(self, base: str, text: str) -> str:
        base = safe_id(base) or f"passage::{text_hash(text)}"
        h = text_hash(text)
        existing = self.base_to_text_hash.get(base)
        if existing is None:
            self.base_to_text_hash[base] = h
            return base
        if existing == h:
            return base
        pid = f"{base}::{h}"
        self.collision_warnings.append({"base_id": base, "assigned_id": pid})
        return pid


def normalize_obliqa_passage_id(passage: dict[str, Any], registry: IdRegistry) -> str:
    doc_id = safe_id(doc_field(passage))
    passage_id = safe_id(str(passage.get("PassageID") or ""))
    base = f"{doc_id}::{passage_id}" if doc_id or passage_id else ""
    return registry.assign(base, passage_text(passage))


def normalize_obliqamp_passage_id(passage: dict[str, Any], registry: IdRegistry) -> str:
    primary = safe_id(str(passage.get("ID") or ""))
    if primary:
        base = primary
    else:
        doc_id = safe_id(doc_field(passage))
        pid = safe_id(str(passage.get("PassageID") or passage.get("passage_id") or ""))
        base = f"{doc_id}::{pid}" if doc_id or pid else ""
    return registry.assign(base, passage_text(passage))


def dataset_splits(ds: DatasetDict | Any) -> dict[str, Any]:
    if isinstance(ds, DatasetDict):
        return dict(ds)
    return {"train": ds}


def build_hf_corpus(
    ds: DatasetDict,
    *,
    source_dataset: str,
    id_fn,
) -> tuple[list[dict[str, str]], IdRegistry]:
    registry = IdRegistry()
    by_id: dict[str, dict[str, str]] = {}
    for split_name, split in dataset_splits(ds).items():
        for row in split:
            for passage in row.get("Passages") or []:
                text = passage_text(passage)
                if not text:
                    continue
                pid = id_fn(passage, registry)
                by_id.setdefault(
                    pid,
                    {
                        "id": pid,
                        "contents": text,
                        "doc_id": doc_field(passage),
                        "source_dataset": source_dataset,
                    },
                )
    return list(by_id.values()), registry


def enrich_hf_sample(rows: list[dict[str, Any]], *, dataset_name: str, id_fn, registry: IdRegistry) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        relevant_ids = []
        for passage in item.get("Passages") or []:
            relevant_ids.append(id_fn(passage, registry))
        item["dataset"] = dataset_name
        item["split"] = "test"
        item["sample_seed"] = SEED
        item["relevant_passage_ids"] = sorted(set(relevant_ids))
        enriched.append(item)
    return enriched


def sample_fixed(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    idxs = sorted(rng.sample(range(len(rows)), n))
    return [dict(rows[i]) for i in idxs]


def sample_xref(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    dpel = [r for r in rows if str(r.get("method") or r.get("generation_method") or "").upper() == "DPEL"]
    schema = [r for r in rows if str(r.get("method") or r.get("generation_method") or "").upper() == "SCHEMA"]
    selected: list[dict[str, Any]] = []
    if len(dpel) >= 2 and len(schema) >= 3:
        selected = rng.sample(dpel, 2) + rng.sample(schema, 3)
    else:
        selected = rng.sample(rows, 5)
    selected = sorted(selected, key=lambda r: str(r.get("item_id") or ""))
    enriched = []
    for row in selected:
        item = dict(row)
        item["dataset"] = "ObliQA-XRef-ADGM"
        item["split"] = "test"
        item["sample_seed"] = SEED
        item["relevant_passage_ids"] = [
            str(item.get("source_passage_id") or ""),
            str(item.get("target_passage_id") or ""),
        ]
        enriched.append(item)
    return enriched


def write_qrels(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid, pid in rows:
            if qid and pid:
                f.write(f"{qid} 0 {pid} 1\n")


def qrels_from_samples(samples: list[dict[str, Any]], qid_key: str) -> list[tuple[str, str]]:
    rows = []
    for sample in samples:
        qid = str(sample.get(qid_key) or "")
        for pid in sample.get("relevant_passage_ids") or []:
            rows.append((qid, str(pid)))
    return rows


def convert_xref_corpus(path: Path) -> list[dict[str, str]]:
    out = []
    for row in read_jsonl(path):
        pid = str(row.get("passage_uid") or "")
        text = str(row.get("passage") or "")
        if not pid or not text:
            continue
        out.append(
            {
                "id": pid,
                "contents": text,
                "doc_id": str(row.get("doc_id") or ""),
                "source_dataset": "ObliQA-XRef-ADGM",
            }
        )
    return out


def count_qrels(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def missing_relevant(corpus: list[dict[str, str]], samples: list[dict[str, Any]]) -> list[str]:
    ids = {row["id"] for row in corpus}
    missing = []
    for sample in samples:
        for pid in sample.get("relevant_passage_ids") or []:
            if pid and pid not in ids:
                missing.append(str(pid))
    return sorted(set(missing))


def main() -> None:
    print("Loading Hugging Face datasets...")
    obliqa = load_dataset("RegNLP/ObliQA")
    obliqamp = load_dataset("RegNLP/ObliQA-MP")
    xref_rows = read_jsonl(XREF_TEST)

    obliqa_test = list(obliqa["test"])
    obliqamp_test = list(obliqamp["test"])

    obliqa_corpus, obliqa_registry = build_hf_corpus(
        obliqa, source_dataset="ObliQA", id_fn=normalize_obliqa_passage_id
    )
    obliqamp_corpus, obliqamp_registry = build_hf_corpus(
        obliqamp, source_dataset="ObliQA-MP", id_fn=normalize_obliqamp_passage_id
    )
    xref_corpus = convert_xref_corpus(XREF_CORPUS)

    obliqa_samples = enrich_hf_sample(
        sample_fixed(obliqa_test, 5),
        dataset_name="ObliQA",
        id_fn=normalize_obliqa_passage_id,
        registry=obliqa_registry,
    )
    obliqamp_samples = enrich_hf_sample(
        sample_fixed(obliqamp_test, 5),
        dataset_name="ObliQA-MP",
        id_fn=normalize_obliqamp_passage_id,
        registry=obliqamp_registry,
    )
    xref_samples = sample_xref(xref_rows)

    write_json(ROOT / "samples/obliqa_test5.json", obliqa_samples)
    write_json(ROOT / "samples/obliqa_mp_test5.json", obliqamp_samples)
    write_json(ROOT / "samples/xref_adgm_test5.json", xref_samples)

    write_jsonl(ROOT / "corpora/obliqa_passages.jsonl", obliqa_corpus)
    write_jsonl(ROOT / "corpora/obliqa_mp_passages.jsonl", obliqamp_corpus)
    write_jsonl(ROOT / "corpora/xref_adgm_passages.jsonl", xref_corpus)

    qrels_paths = {
        "obliqa": ROOT / "qrels/obliqa_test5.qrels",
        "obliqa_mp": ROOT / "qrels/obliqa_mp_test5.qrels",
        "xref_adgm": ROOT / "qrels/xref_adgm_test5.qrels",
    }
    write_qrels(qrels_paths["obliqa"], qrels_from_samples(obliqa_samples, "QuestionID"))
    write_qrels(qrels_paths["obliqa_mp"], qrels_from_samples(obliqamp_samples, "QuestionID"))
    write_qrels(qrels_paths["xref_adgm"], qrels_from_samples(xref_samples, "item_id"))

    report = {
        "pilot": {
            "purpose": "5+5+5 controlled-comparison pipeline smoke pilot; not paper results",
            "sample_seed": SEED,
            "top_k": 10,
            "retriever": "bm25",
        },
        "datasets": {
            "obliqa": {
                "name": "RegNLP/ObliQA",
                "test_size": len(obliqa_test),
                "selected_sample_ids": [str(r.get("QuestionID")) for r in obliqa_samples],
                "corpus_passages": len(obliqa_corpus),
                "qrels_count": count_qrels(qrels_paths["obliqa"]),
                "missing_passage_ids": missing_relevant(obliqa_corpus, obliqa_samples),
                "id_collision_warnings": obliqa_registry.collision_warnings,
            },
            "obliqa_mp": {
                "name": "RegNLP/ObliQA-MP",
                "test_size": len(obliqamp_test),
                "selected_sample_ids": [str(r.get("QuestionID")) for r in obliqamp_samples],
                "corpus_passages": len(obliqamp_corpus),
                "qrels_count": count_qrels(qrels_paths["obliqa_mp"]),
                "missing_passage_ids": missing_relevant(obliqamp_corpus, obliqamp_samples),
                "id_collision_warnings": obliqamp_registry.collision_warnings,
            },
            "xref_adgm": {
                "name": "ObliQA-XRef-ADGM-ALL local test",
                "test_size": len(xref_rows),
                "selected_sample_ids": [str(r.get("item_id")) for r in xref_samples],
                "selected_methods": {},
                "corpus_passages": len(xref_corpus),
                "qrels_count": count_qrels(qrels_paths["xref_adgm"]),
                "missing_passage_ids": missing_relevant(xref_corpus, xref_samples),
                "id_collision_warnings": [],
            },
        },
        "issues": [],
    }
    # Make selected_methods readable and deterministic.
    counts: defaultdict[str, int] = defaultdict(int)
    for row in xref_samples:
        counts[str(row.get("method") or row.get("generation_method") or "")] += 1
    report["datasets"]["xref_adgm"]["selected_methods"] = dict(sorted(counts.items()))

    write_json(ROOT / "reports/pilot_build_report.json", report)
    print(f"Wrote pilot artifacts under {ROOT}")


if __name__ == "__main__":
    main()
