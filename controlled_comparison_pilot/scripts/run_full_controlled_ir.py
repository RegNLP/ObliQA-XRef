#!/usr/bin/env python3
"""Run full-test BM25(passage) controlled IR evaluation across 3 datasets.

This script prepares and evaluates:
1) RegNLP/ObliQA (HF)
2) RegNLP/ObliQA-MP (HF)
3) ObliQA-XRef-ADGM local full test split

Outputs under controlled_comparison_pilot/full_ir:
- corpora/*.jsonl
- qrels/*.qrels
- runs/*.trec
- reports/controlled_ir_metrics_full.{csv,json}
- reports/full_ir_run_report.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from obliqaxref.curate.ir.bm25 import BM25Retriever


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(msg, flush=True)


def stable_short_hash(text: str, n: int = 10) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:n]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_qrels(path: Path, qrels: dict[str, dict[str, int]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for qid in sorted(qrels):
            for docid, rel in sorted(qrels[qid].items()):
                f.write(f"{qid} 0 {docid} {int(rel)}\n")
                count += 1
    return count


@dataclass
class SharedCorpusIndex:
    source_path: str
    corpus_rows: list[dict[str, str]]
    uid_set: set[str]
    uid_to_docid: dict[str, str]
    pid_to_uids: dict[str, list[str]]
    pid_text_to_uids: dict[tuple[str, str], list[str]]


def normalize_target_doc_id(doc_id: str) -> str:
    doc = (doc_id or "").strip()
    if not doc:
        return ""
    if doc.startswith("adgm_doc_"):
        return doc
    return f"adgm_doc_{doc}"


def load_shared_adgm_corpus_index(repo_root: Path) -> SharedCorpusIndex:
    source_path = repo_root / "runs/adapter_adgm/processed/passage_corpus.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing shared ADGM passage corpus: {source_path}")

    corpus_src = read_jsonl(source_path)
    corpus_rows: list[dict[str, str]] = []
    uid_set: set[str] = set()
    uid_to_docid: dict[str, str] = {}
    pid_to_uids: dict[str, set[str]] = defaultdict(set)
    pid_text_to_uids: dict[tuple[str, str], set[str]] = defaultdict(set)

    for row in corpus_src:
        uid = str(row.get("passage_uid") or "").strip()
        text = str(row.get("passage") or "")
        doc_id = str(row.get("doc_id") or "")
        pid = str(row.get("passage_id") or "").strip()
        if not uid or not text:
            continue
        if uid in uid_set:
            continue
        uid_set.add(uid)
        uid_to_docid[uid] = doc_id
        corpus_rows.append(
            {
                "id": uid,
                "contents": text,
                "doc_id": doc_id,
                "source_dataset": "ADGM_SHARED",
            }
        )
        if pid:
            pid_to_uids[pid].add(uid)
            pid_text_to_uids[(pid, normalize_text(text))].add(uid)

    return SharedCorpusIndex(
        source_path=str(source_path),
        corpus_rows=sorted(corpus_rows, key=lambda x: x["id"]),
        uid_set=uid_set,
        uid_to_docid=uid_to_docid,
        pid_to_uids={k: sorted(v) for k, v in pid_to_uids.items()},
        pid_text_to_uids={k: sorted(v) for k, v in pid_text_to_uids.items()},
    )


def resolve_root(repo_root: Path, root_arg: str) -> Path:
    root = Path(root_arg)
    if root.is_absolute():
        return root
    return (repo_root / root).resolve()


def load_hf_dataset_splits(dataset_name: str) -> dict[str, list[dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Missing dependency 'datasets'. Install with: pip install datasets"
        ) from exc

    try:
        ds = load_dataset(dataset_name)
    except Exception as exc:  # pragma: no cover - runtime fetch
        raise RuntimeError(
            f"Failed to load Hugging Face dataset '{dataset_name}'. "
            "Check internet access, HF auth (if needed), and dataset name."
        ) from exc

    splits: dict[str, list[dict[str, Any]]] = {}
    split_names = list(ds.keys()) if hasattr(ds, "keys") else []
    if not split_names:
        raise RuntimeError(f"No splits found for Hugging Face dataset '{dataset_name}'.")

    for split_name in split_names:
        splits[split_name] = [dict(row) for row in ds[split_name]]
    return splits


def pick_first_nonempty(d: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        val = d.get(key)
        if val is not None and str(val).strip() != "":
            return str(val)
    return ""


def iter_passages(item: dict[str, Any]) -> list[dict[str, Any]]:
    passages = item.get("Passages")
    if not isinstance(passages, list):
        return []
    out: list[dict[str, Any]] = []
    for p in passages:
        if isinstance(p, dict):
            out.append(p)
    return out


def map_obliqa_passage_to_shared(
    passage: dict[str, Any],
    shared: SharedCorpusIndex,
) -> tuple[str | None, str]:
    for key in ["passage_uid", "PassageUID", "UID", "uid", "ID", "id"]:
        v = str(passage.get(key) or "").strip()
        if v and v in shared.uid_set:
            return v, f"direct:{key}"

    pid = str(passage.get("PassageID") or "").strip()
    if pid:
        by_pid = shared.pid_to_uids.get(pid, [])
        if len(by_pid) == 1:
            return by_pid[0], "passageid_unique"

        ptxt_norm = normalize_text(str(passage.get("Passage") or ""))
        by_pid_text = shared.pid_text_to_uids.get((pid, ptxt_norm), [])
        if len(by_pid_text) == 1:
            return by_pid_text[0], "passageid_text"
        if len(by_pid_text) > 1:
            doc_id = normalize_target_doc_id(str(passage.get("DocumentID") or ""))
            if doc_id:
                by_doc = [uid for uid in by_pid_text if shared.uid_to_docid.get(uid) == doc_id]
                if len(by_doc) == 1:
                    return by_doc[0], "passageid_text_docid_tiebreak"

    return None, "unmapped"


def map_obliqa_mp_passage_to_shared(
    passage: dict[str, Any],
    shared: SharedCorpusIndex,
) -> tuple[str | None, str]:
    primary_id = str(passage.get("ID") or "").strip()
    if primary_id and primary_id in shared.uid_set:
        return primary_id, "id_direct"

    pid = str(passage.get("PassageID") or "").strip()
    if pid:
        by_pid = shared.pid_to_uids.get(pid, [])
        if len(by_pid) == 1:
            return by_pid[0], "passageid_unique"

        ptxt_norm = normalize_text(str(passage.get("Passage") or ""))
        by_pid_text = shared.pid_text_to_uids.get((pid, ptxt_norm), [])
        if len(by_pid_text) == 1:
            return by_pid_text[0], "passageid_text"

        tie_candidates = by_pid_text if by_pid_text else by_pid
        if len(tie_candidates) > 1:
            doc_id = normalize_target_doc_id(str(passage.get("DocumentID") or ""))
            if doc_id:
                by_doc = [uid for uid in tie_candidates if shared.uid_to_docid.get(uid) == doc_id]
                if len(by_doc) == 1:
                    return by_doc[0], "docid_tiebreak"

    return None, "unmapped"


def canonical_obliqa_pid(doc_id: str, passage_id: str, passage_text: str) -> str:
    left = doc_id if doc_id else "DOC"
    right = passage_id if passage_id else f"PID_{stable_short_hash(passage_text)}"
    return f"{left}::{right}"


def canonical_obliqa_mp_pid(passage_obj: dict[str, Any], passage_text: str) -> str:
    preferred = pick_first_nonempty(passage_obj, ["id", "ID", "Id"])
    if preferred:
        return preferred
    fallback = pick_first_nonempty(passage_obj, ["PassageID", "passage_id", "passageId"])
    doc_id = pick_first_nonempty(passage_obj, ["DocumentID", "document_id", "doc_id", "DocID"])
    if fallback:
        if doc_id:
            return f"{doc_id}::{fallback}"
        return fallback
    return f"SYN::{stable_short_hash((doc_id or '') + '||' + passage_text)}"


def dedupe_with_collision_suffix(
    desired_id: str,
    text: str,
    id_to_text: dict[str, str],
    *,
    collision_counter: Counter[str],
) -> str:
    existing = id_to_text.get(desired_id)
    if existing is None:
        id_to_text[desired_id] = text
        return desired_id
    if existing == text:
        return desired_id

    collision_counter["id_text_collisions"] += 1
    hashed = f"{desired_id}::{stable_short_hash(text)}"
    existing_hashed = id_to_text.get(hashed)
    if existing_hashed is None or existing_hashed == text:
        id_to_text[hashed] = text
        return hashed

    i = 2
    while True:
        candidate = f"{hashed}_{i}"
        cur = id_to_text.get(candidate)
        if cur is None or cur == text:
            id_to_text[candidate] = text
            return candidate
        i += 1


@dataclass
class DatasetBuild:
    dataset_name: str
    split: str
    sample_type: str
    test_size: int
    corpus_rows: list[dict[str, str]]
    qrels: dict[str, dict[str, int]]
    queries: list[tuple[str, str]]
    pair_src: dict[str, str] | None
    pair_tgt: dict[str, str] | None
    warnings: list[str]
    collision_counts: dict[str, int]
    qrels_mapping_rate: float
    qrels_unmapped_count: int
    qrels_total_count: int


def build_obliqa(name: str, hf_name: str, shared: SharedCorpusIndex) -> DatasetBuild:
    log(f"[{name}] dataset loading start: {hf_name}")
    splits = load_hf_dataset_splits(hf_name)
    if "test" not in splits:
        raise RuntimeError(f"[{name}] missing required 'test' split in '{hf_name}'.")

    warnings: list[str] = []

    test_rows = splits["test"]
    queries: list[tuple[str, str]] = []
    qrels: dict[str, dict[str, int]] = {}
    missing_qid = 0
    missing_question = 0
    qrels_total_count = 0
    qrels_mapped_count = 0
    qrels_unmapped_count = 0
    map_method_counter: Counter[str] = Counter()
    log(f"[{name}] qrels mapping start")

    for item in test_rows:
        qid = str(item.get("QuestionID") or "").strip()
        question = str(item.get("Question") or "").strip()
        if not qid:
            missing_qid += 1
            continue
        if not question:
            missing_question += 1
        queries.append((qid, question))

        qrels.setdefault(qid, {})
        for p in iter_passages(item):
            qrels_total_count += 1
            mapped_uid, strategy = map_obliqa_passage_to_shared(p, shared)
            if mapped_uid is not None:
                qrels_mapped_count += 1
                map_method_counter[strategy] += 1
                qrels[qid][mapped_uid] = 1
            else:
                qrels_unmapped_count += 1
                if qrels_unmapped_count <= 20:
                    warnings.append(
                        "unmapped_qrel_obliqa="
                        f"qid:{qid} doc:{p.get('DocumentID')} pid:{p.get('PassageID')}"
                    )

    if missing_qid:
        warnings.append(f"missing_question_id_rows={missing_qid}")
    if missing_question:
        warnings.append(f"missing_question_text_rows={missing_question}")

    qrels_mapping_rate = qrels_mapped_count / max(1, qrels_total_count)
    log(f"[{name}] dataset loading end")
    log(f"[{name}] test_size={len(queries)}")
    log(f"[{name}] qrels mapping end")
    log(f"[{name}] qrels_count={qrels_total_count}")
    log(f"[{name}] qrels_mapping_rate={qrels_mapping_rate:.6f}")
    log(f"[{name}] qrels_unmapped_count={qrels_unmapped_count}")
    if map_method_counter:
        warnings.append(f"qrels_map_methods={dict(map_method_counter)}")

    return DatasetBuild(
        dataset_name=name,
        split="test",
        sample_type="full_test",
        test_size=len(queries),
        corpus_rows=shared.corpus_rows,
        qrels=qrels,
        queries=queries,
        pair_src=None,
        pair_tgt=None,
        warnings=warnings,
        collision_counts={"id_text_collisions": 0},
        qrels_mapping_rate=qrels_mapping_rate,
        qrels_unmapped_count=qrels_unmapped_count,
        qrels_total_count=qrels_total_count,
    )


def build_obliqa_mp(name: str, hf_name: str, shared: SharedCorpusIndex) -> DatasetBuild:
    log(f"[{name}] dataset loading start: {hf_name}")
    splits = load_hf_dataset_splits(hf_name)
    if "test" not in splits:
        raise RuntimeError(f"[{name}] missing required 'test' split in '{hf_name}'.")

    test_rows = splits["test"]
    warnings: list[str] = []
    queries: list[tuple[str, str]] = []
    qrels: dict[str, dict[str, int]] = {}
    missing_qid = 0
    missing_question = 0
    qrels_total_count = 0
    qrels_mapped_count = 0
    qrels_unmapped_count = 0
    map_method_counter: Counter[str] = Counter()
    log(f"[{name}] qrels mapping start")

    for item in test_rows:
        qid = str(item.get("QuestionID") or "").strip()
        question = str(item.get("Question") or "").strip()
        if not qid:
            missing_qid += 1
            continue
        if not question:
            missing_question += 1
        queries.append((qid, question))

        qrels.setdefault(qid, {})
        for p in iter_passages(item):
            qrels_total_count += 1
            mapped_uid, strategy = map_obliqa_mp_passage_to_shared(p, shared)
            if mapped_uid is not None:
                qrels_mapped_count += 1
                map_method_counter[strategy] += 1
                qrels[qid][mapped_uid] = 1
            else:
                qrels_unmapped_count += 1
                if qrels_unmapped_count <= 20:
                    warnings.append(
                        "unmapped_qrel_obliqa_mp="
                        f"qid:{qid} doc:{p.get('DocumentID')} pid:{p.get('PassageID')} id:{p.get('ID')}"
                    )

    if missing_qid:
        warnings.append(f"missing_question_id_rows={missing_qid}")
    if missing_question:
        warnings.append(f"missing_question_text_rows={missing_question}")

    qrels_mapping_rate = qrels_mapped_count / max(1, qrels_total_count)
    log(f"[{name}] dataset loading end")
    log(f"[{name}] test_size={len(queries)}")
    log(f"[{name}] qrels mapping end")
    log(f"[{name}] qrels_count={qrels_total_count}")
    log(f"[{name}] qrels_mapping_rate={qrels_mapping_rate:.6f}")
    log(f"[{name}] qrels_unmapped_count={qrels_unmapped_count}")
    if map_method_counter:
        warnings.append(f"qrels_map_methods={dict(map_method_counter)}")

    return DatasetBuild(
        dataset_name=name,
        split="test",
        sample_type="full_test",
        test_size=len(queries),
        corpus_rows=shared.corpus_rows,
        qrels=qrels,
        queries=queries,
        pair_src=None,
        pair_tgt=None,
        warnings=warnings,
        collision_counts={"id_text_collisions": 0},
        qrels_mapping_rate=qrels_mapping_rate,
        qrels_unmapped_count=qrels_unmapped_count,
        qrels_total_count=qrels_total_count,
    )


def build_xref_adgm(repo_root: Path, shared: SharedCorpusIndex) -> DatasetBuild:
    name = "ObliQA-XRef-ADGM"
    split_path = (
        repo_root
        / "ObliQA-XRef_Out_Datasets/final_large_merged/ObliQA-XRef-ADGM-ALL/test.jsonl"
    )
    log(f"[{name}] dataset loading start: {split_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing local test split: {split_path}")

    test_rows = read_jsonl(split_path)
    warnings: list[str] = []

    queries: list[tuple[str, str]] = []
    qrels: dict[str, dict[str, int]] = {}
    src_map: dict[str, str] = {}
    tgt_map: dict[str, str] = {}
    missing = 0
    qrels_total_count = 0
    qrels_mapped_count = 0
    qrels_unmapped_count = 0
    log(f"[{name}] qrels mapping start")
    for item in test_rows:
        qid = str(item.get("item_id") or "").strip()
        question = str(item.get("question") or "").strip()
        src = str(item.get("source_passage_id") or "").strip()
        tgt = str(item.get("target_passage_id") or "").strip()
        if not qid:
            missing += 1
            continue
        queries.append((qid, question))
        qrels[qid] = {}
        if src:
            qrels_total_count += 1
            qrels[qid][src] = 1
            src_map[qid] = src
            if src in shared.uid_set:
                qrels_mapped_count += 1
            else:
                qrels_unmapped_count += 1
                if qrels_unmapped_count <= 20:
                    warnings.append(f"unmapped_qrel_xref=qid:{qid} field:source id:{src}")
        if tgt:
            qrels_total_count += 1
            qrels[qid][tgt] = 1
            tgt_map[qid] = tgt
            if tgt in shared.uid_set:
                qrels_mapped_count += 1
            else:
                qrels_unmapped_count += 1
                if qrels_unmapped_count <= 20:
                    warnings.append(f"unmapped_qrel_xref=qid:{qid} field:target id:{tgt}")

    if missing:
        warnings.append(f"missing_item_id_rows={missing}")
    if len(queries) != 502:
        warnings.append(f"expected_test_size_502_observed={len(queries)}")

    qrels_mapping_rate = qrels_mapped_count / max(1, qrels_total_count)
    log(f"[{name}] dataset loading end")
    log(f"[{name}] test_size={len(queries)}")
    log(f"[{name}] qrels mapping end")
    log(f"[{name}] qrels_count={qrels_total_count}")
    log(f"[{name}] qrels_mapping_rate={qrels_mapping_rate:.6f}")
    log(f"[{name}] qrels_unmapped_count={qrels_unmapped_count}")

    return DatasetBuild(
        dataset_name=name,
        split="test",
        sample_type="full_test",
        test_size=len(queries),
        corpus_rows=shared.corpus_rows,
        qrels=qrels,
        queries=queries,
        pair_src=src_map,
        pair_tgt=tgt_map,
        warnings=warnings,
        collision_counts={"id_text_collisions": 0},
        qrels_mapping_rate=qrels_mapping_rate,
        qrels_unmapped_count=qrels_unmapped_count,
        qrels_total_count=qrels_total_count,
    )


def retrieve_bm25_topk(
    queries: list[tuple[str, str]],
    corpus_rows: list[dict[str, str]],
    run_path: Path,
    *,
    k: int,
    progress_every: int = 100,
) -> tuple[dict[str, dict[str, float]], int]:
    if not corpus_rows:
        raise ValueError("Corpus is empty; cannot run retrieval.")

    passages = [
        {
            "passage_id": str(r["id"]),
            "text": str(r.get("contents") or ""),
        }
        for r in corpus_rows
    ]

    log(f"BM25 index/build start: passages={len(passages)}")
    bm25 = BM25Retriever(passages)
    log("BM25 index/build end")

    run: dict[str, dict[str, float]] = {}
    run_rows = 0
    run_path.parent.mkdir(parents=True, exist_ok=True)
    with run_path.open("w", encoding="utf-8") as f:
        total = len(queries)
        for idx, (qid, question) in enumerate(queries, start=1):
            results = bm25.search(question, k=k)
            run[qid] = {}
            for r in results:
                docid = r.passage_id
                scoref = float(r.score)
                run[qid][docid] = scoref
                f.write(f"{qid} Q0 {docid} {r.rank} {scoref:.6f} bm25\n")
                run_rows += 1
            if idx % progress_every == 0 or idx == total:
                log(f"BM25 retrieval progress: {idx}/{total} queries")
    return run, run_rows


def rank_docids(scores: dict[str, float], k: int | None = None) -> list[str]:
    ranked = [docid for docid, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]
    if k is None:
        return ranked
    return ranked[:k]


def require_pytrec_eval() -> Any:
    try:
        import pytrec_eval  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Missing dependency 'pytrec_eval'. Install it before running this script, "
            "for example: pip install pytrec-eval"
        ) from exc
    return pytrec_eval


def compute_common_metrics(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    *,
    k: int,
    pytrec_eval_mod: Any,
) -> dict[str, float]:
    qids = set(qrels)
    eval_run = {qid: run.get(qid, {}) for qid in qids}
    evaluator = pytrec_eval_mod.RelevanceEvaluator(
        qrels,
        {f"recall_{k}", f"map_cut_{k}", f"ndcg_cut_{k}"},
    )
    res = evaluator.evaluate(eval_run)
    denom = max(1, len(qids))

    hit = 0
    rel_count_total = 0
    for qid in qids:
        rel_docs = {docid for docid, rel in qrels.get(qid, {}).items() if rel > 0}
        topk = set(rank_docids(run.get(qid, {}), k))
        rel_at_k = rel_docs & topk
        if rel_at_k:
            hit += 1
        rel_count_total += len(rel_at_k)

    return {
        f"Recall@{k}": sum(res[qid].get(f"recall_{k}", 0.0) for qid in qids) / denom,
        f"MAP@{k}": sum(res[qid].get(f"map_cut_{k}", 0.0) for qid in qids) / denom,
        f"nDCG@{k}": sum(res[qid].get(f"ndcg_cut_{k}", 0.0) for qid in qids) / denom,
        f"Hit@{k}": hit / denom,
        f"RelCount@{k}": rel_count_total / denom,
    }


def compute_xref_pair_metrics(
    run: dict[str, dict[str, float]],
    src_map: dict[str, str],
    tgt_map: dict[str, str],
    *,
    k: int,
    qids: set[str],
) -> dict[str, float]:
    denom = max(1, len(qids))
    both = 0
    src_only = 0
    tgt_only = 0
    neither = 0
    pair_mrr_total = 0.0

    for qid in sorted(qids):
        src = src_map.get(qid)
        tgt = tgt_map.get(qid)
        ranked_all = rank_docids(run.get(qid, {}), None)
        ranked_k = ranked_all[:k]
        in_k = set(ranked_k)
        found_src = bool(src) and src in in_k
        found_tgt = bool(tgt) and tgt in in_k

        if found_src and found_tgt:
            both += 1
        elif found_src:
            src_only += 1
        elif found_tgt:
            tgt_only += 1
        else:
            neither += 1

        seen_src = False
        seen_tgt = False
        rr = 0.0
        for rank, docid in enumerate(ranked_all, start=1):
            if src and docid == src:
                seen_src = True
            if tgt and docid == tgt:
                seen_tgt = True
            if seen_src and seen_tgt:
                rr = 1.0 / rank
                break
        pair_mrr_total += rr

    return {
        f"Both@{k}": both / denom,
        f"SRC-only@{k}": src_only / denom,
        f"TGT-only@{k}": tgt_only / denom,
        f"Neither@{k}": neither / denom,
        "PairMRR": pair_mrr_total / denom,
    }


def format_float(val: Any) -> str:
    if val is None or val == "":
        return ""
    if isinstance(val, (int, float)):
        return f"{float(val):.6f}"
    return str(val)


def print_metrics_table(rows: list[dict[str, Any]], k: int) -> None:
    log("\nMetrics summary:")
    header = (
        f"{'dataset':20} {'Recall@'+str(k):>10} {'MAP@'+str(k):>10} "
        f"{'nDCG@'+str(k):>10} {'Hit@'+str(k):>10} {'RelCount@'+str(k):>12} {'PairMRR':>10}"
    )
    log(header)
    log("-" * len(header))
    for row in rows:
        log(
            f"{row['dataset'][:20]:20} "
            f"{float(row['Recall@10']):10.6f} "
            f"{float(row['MAP@10']):10.6f} "
            f"{float(row['nDCG@10']):10.6f} "
            f"{float(row['Hit@10']):10.6f} "
            f"{float(row['RelCount@10']):12.6f} "
            f"{format_float(row.get('PairMRR') if row.get('PairMRR') is not None else ''):>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full controlled BM25 IR across all datasets.")
    parser.add_argument(
        "--root",
        default="controlled_comparison_pilot/full_ir",
        help="Output root directory.",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k cutoff.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)
    if args.k <= 0:
        raise ValueError("--k must be a positive integer")

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    out_root = resolve_root(repo_root, args.root)
    corpora_dir = out_root / "corpora"
    qrels_dir = out_root / "qrels"
    runs_dir = out_root / "runs"
    reports_dir = out_root / "reports"
    for d in [corpora_dir, qrels_dir, runs_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    start_ts = time.time()
    start_time = utc_now_iso()
    log(f"start_time={start_time}")
    log(f"output_root={out_root}")

    pytrec_eval_mod = require_pytrec_eval()

    log("shared corpus loading start")
    shared = load_shared_adgm_corpus_index(repo_root)
    log("shared corpus loading end")
    log(f"shared corpus size={len(shared.corpus_rows)}")
    shared_corpus_out = corpora_dir / "adgm_shared_passages.jsonl"
    log(f"shared corpus export path={shared_corpus_out}")
    write_jsonl(shared_corpus_out, shared.corpus_rows)
    log(f"shared corpus export end -> {shared_corpus_out}")

    all_warnings: list[str] = []
    id_collision_counts: dict[str, dict[str, int]] = {}

    builds: list[tuple[DatasetBuild, Path, Path]] = []
    obliqa = build_obliqa("ObliQA", "RegNLP/ObliQA", shared)
    builds.append(
        (
            obliqa,
            qrels_dir / "obliqa_test.qrels",
            runs_dir / "obliqa_bm25_top10.trec",
        )
    )
    obliqa_mp = build_obliqa_mp("ObliQA-MP", "RegNLP/ObliQA-MP", shared)
    builds.append(
        (
            obliqa_mp,
            qrels_dir / "obliqa_mp_test.qrels",
            runs_dir / "obliqa_mp_bm25_top10.trec",
        )
    )
    xref = build_xref_adgm(repo_root, shared)
    builds.append(
        (
            xref,
            qrels_dir / "xref_adgm_test.qrels",
            runs_dir / "xref_adgm_bm25_top10.trec",
        )
    )

    metrics_rows: list[dict[str, Any]] = []
    dataset_report: dict[str, Any] = {}

    for build, qrels_out, run_out in builds:
        log(f"\n[{build.dataset_name}] retrieval_corpus=shared_adgm_13015")
        log(f"[{build.dataset_name}] shared_corpus_path={shared.source_path}")

        log(f"[{build.dataset_name}] writing qrels -> {qrels_out}")
        qrels_count = write_qrels(qrels_out, build.qrels)
        log(f"[{build.dataset_name}] qrels_count={qrels_count}")

        log(f"[{build.dataset_name}] BM25 retrieval start (canonical BM25Retriever, k={args.k})")
        run, run_rows = retrieve_bm25_topk(build.queries, shared.corpus_rows, run_out, k=args.k)
        log(f"[{build.dataset_name}] BM25 retrieval end")
        log(f"[{build.dataset_name}] trec_run_path={run_out}")
        log(f"[{build.dataset_name}] run_row_count={run_rows}")

        log(f"[{build.dataset_name}] metrics computation start")
        common = compute_common_metrics(build.qrels, run, k=args.k, pytrec_eval_mod=pytrec_eval_mod)
        log(f"[{build.dataset_name}] metrics computation end")

        missing_run_qids = sorted(set(build.qrels) - set(run))
        notes: list[str] = []
        if missing_run_qids:
            msg = f"missing_run_qids={len(missing_run_qids)}"
            notes.append(msg)
            all_warnings.append(f"[{build.dataset_name}] {msg}")

        row: dict[str, Any] = {
            "dataset": build.dataset_name,
            "split": build.split,
            "sample_type": build.sample_type,
            "sample_size": build.test_size,
            "retriever": "bm25",
            "k": args.k,
            "Recall@10": common[f"Recall@{args.k}"],
            "MAP@10": common[f"MAP@{args.k}"],
            "nDCG@10": common[f"nDCG@{args.k}"],
            "Hit@10": common[f"Hit@{args.k}"],
            "RelCount@10": common[f"RelCount@{args.k}"],
            "Both@10": None,
            "SRC-only@10": None,
            "TGT-only@10": None,
            "Neither@10": None,
            "PairMRR": None,
            "corpus_size": len(shared.corpus_rows),
            "qrels_count": qrels_count,
            "run_rows": run_rows,
            "notes": "; ".join(
                notes
                + [
                    "retrieval_setting=controlled_shared_corpus_canonical_bm25",
                    "retrieval_corpus=shared_adgm_13015",
                    f"shared_corpus_path={shared.source_path}",
                    "bm25_backend=obliqaxref.curate.ir.BM25Retriever",
                    "bm25_parameters=rank_bm25.BM25Okapi defaults",
                    "tokenization=lowercase whitespace split",
                    f"cutoff_k={args.k}",
                    "historical_run_reproduction=false",
                    "canonical_repo_bm25=true",
                    f"qrels_mapping_rate={build.qrels_mapping_rate:.6f}",
                    f"qrels_unmapped_count={build.qrels_unmapped_count}",
                ]
                + build.warnings
            ),
        }

        if build.pair_src is not None and build.pair_tgt is not None:
            pair = compute_xref_pair_metrics(
                run,
                build.pair_src,
                build.pair_tgt,
                k=args.k,
                qids=set(build.qrels),
            )
            row["Both@10"] = pair[f"Both@{args.k}"]
            row["SRC-only@10"] = pair[f"SRC-only@{args.k}"]
            row["TGT-only@10"] = pair[f"TGT-only@{args.k}"]
            row["Neither@10"] = pair[f"Neither@{args.k}"]
            row["PairMRR"] = pair["PairMRR"]

        metrics_rows.append(row)

        all_warnings.extend(f"[{build.dataset_name}] {w}" for w in build.warnings)
        id_collision_counts[build.dataset_name] = dict(build.collision_counts)
        dataset_report[build.dataset_name] = {
            "test_size": build.test_size,
            "corpus_size": len(shared.corpus_rows),
            "qrels_count": qrels_count,
            "run_rows": run_rows,
            "corpus_path": str(shared_corpus_out),
            "qrels_path": str(qrels_out),
            "run_path": str(run_out),
            "warnings": build.warnings,
            "retrieval_corpus": "shared_adgm_13015",
            "shared_corpus_path": shared.source_path,
            "retrieval_setting": "controlled_shared_corpus_canonical_bm25",
            "bm25_backend": "obliqaxref.curate.ir.BM25Retriever",
            "bm25_parameters": "rank_bm25.BM25Okapi defaults",
            "tokenization": "lowercase whitespace split",
            "cutoff_k": args.k,
            "historical_run_reproduction": False,
            "canonical_repo_bm25": True,
            "qrels_mapping_rate": build.qrels_mapping_rate,
            "qrels_unmapped_count": build.qrels_unmapped_count,
        }

    fieldnames = [
        "dataset",
        "split",
        "sample_type",
        "sample_size",
        "retriever",
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
        "corpus_size",
        "qrels_count",
        "run_rows",
        "notes",
    ]

    metrics_json_path = reports_dir / "controlled_ir_metrics_full.json"
    metrics_csv_path = reports_dir / "controlled_ir_metrics_full.csv"
    report_path = reports_dir / "full_ir_run_report.json"

    write_json(metrics_json_path, metrics_rows)

    with metrics_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            out = dict(row)
            for metric_key in [
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
            ]:
                out[metric_key] = format_float(out.get(metric_key))
            writer.writerow(out)

    end_ts = time.time()
    end_time = utc_now_iso()
    runtime_seconds = end_ts - start_ts
    run_report = {
        "start_time": start_time,
        "end_time": end_time,
        "runtime_seconds": runtime_seconds,
        "retrieval_setting": "controlled_shared_corpus_canonical_bm25",
        "retrieval_corpus": "shared_adgm_13015",
        "bm25_backend": "obliqaxref.curate.ir.BM25Retriever",
        "bm25_parameters": "rank_bm25.BM25Okapi defaults",
        "tokenization": "lowercase whitespace split",
        "cutoff_k": args.k,
        "historical_run_reproduction": False,
        "canonical_repo_bm25": True,
        "xref_official_reference_run": "ObliQA-XRef_Out_Datasets/final_large_merged/ObliQA-XRef-ADGM-ALL/bm25.trec",
        "dataset_sizes": {k: v["test_size"] for k, v in dataset_report.items()},
        "corpus_sizes": {k: v["corpus_size"] for k, v in dataset_report.items()},
        "qrels_counts": {k: v["qrels_count"] for k, v in dataset_report.items()},
        "run_row_counts": {k: v["run_rows"] for k, v in dataset_report.items()},
        "output_paths": {
            "metrics_csv": str(metrics_csv_path),
            "metrics_json": str(metrics_json_path),
            "run_report_json": str(report_path),
            "shared_corpus": str(shared_corpus_out),
            "corpora": {k: v["corpus_path"] for k, v in dataset_report.items()},
            "qrels": {k: v["qrels_path"] for k, v in dataset_report.items()},
            "runs": {k: v["run_path"] for k, v in dataset_report.items()},
        },
        "shared_corpus_path": shared.source_path,
        "qrels_mapping_rates": {k: v["qrels_mapping_rate"] for k, v in dataset_report.items()},
        "qrels_unmapped_counts": {k: v["qrels_unmapped_count"] for k, v in dataset_report.items()},
        "warnings": all_warnings,
        "id_collision_counts": id_collision_counts,
    }
    write_json(report_path, run_report)

    print_metrics_table(metrics_rows, args.k)
    log("\nOutput paths:")
    log(f"- metrics_csv: {metrics_csv_path}")
    log(f"- metrics_json: {metrics_json_path}")
    log(f"- run_report_json: {report_path}")
    log(f"total_runtime_seconds={runtime_seconds:.3f}")


if __name__ == "__main__":
    main()
