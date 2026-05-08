#!/usr/bin/env python3
"""Build controlled batch answer-generation inputs from saved BM25 runs.

This script samples test questions, attaches top-10 passages from the existing
canonical BM25 TREC runs, and writes prompt-condition-specific input JSON files.
It does not run retrieval or call any model API.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_SAMPLE_SIZE = 300
DEFAULT_SEED = 42
K = 10
RETRIEVAL_SETTING = "controlled_shared_corpus_canonical_bm25"
RETRIEVAL_CORPUS = "shared_adgm_13015"

GENERAL_PROMPT_FILE = "controlled_comparison_pilot/prompts/general_grounded_prompt.txt"
PROMPT_FILES = {
    "general": {
        "obliqa": GENERAL_PROMPT_FILE,
        "obliqa_mp": GENERAL_PROMPT_FILE,
        "xref_adgm": GENERAL_PROMPT_FILE,
    },
    "task_aware": {
        "obliqa": "controlled_comparison_pilot/prompts/task_aware_obliqa_prompt.txt",
        "obliqa_mp": "controlled_comparison_pilot/prompts/task_aware_obliqa_mp_prompt.txt",
        "xref_adgm": "controlled_comparison_pilot/prompts/task_aware_xref_prompt.txt",
    },
}

DATASETS = {
    "obliqa": {
        "dataset_name": "ObliQA",
        "source": Path("/Users/tuba.gokhan/Desktop/Organize/ObliQA_test.json"),
        "source_format": "json",
        "qid": "QuestionID",
        "question": "Question",
        "strata": ["Group"],
        "run": ROOT / "full_ir/runs/obliqa_bm25_top10.trec",
        "qrels": ROOT / "full_ir/qrels/obliqa_test.qrels",
        "output_stem": "obliqa_bm25_gpt52",
    },
    "obliqa_mp": {
        "dataset_name": "ObliQA-MP",
        "source": Path("/Users/tuba.gokhan/Desktop/RegNLP_external/ObliQA-ML/ObliQA_MultiPassage_test.json"),
        "source_format": "json",
        "qid": "QuestionID",
        "question": "Question",
        "strata": ["passage_count_bucket"],
        "run": ROOT / "full_ir/runs/obliqa_mp_bm25_top10.trec",
        "qrels": ROOT / "full_ir/qrels/obliqa_mp_test.qrels",
        "output_stem": "obliqa_mp_bm25_gpt52",
    },
    "xref_adgm": {
        "dataset_name": "ObliQA-XRef-ADGM",
        "source": REPO_ROOT / "ObliQA-XRef_Out_Datasets/final_large_merged/ObliQA-XRef-ADGM-ALL/test.jsonl",
        "source_format": "jsonl",
        "qid": "item_id",
        "question": "question",
        "strata": ["generation_method", "sampling_regime"],
        "run": ROOT / "full_ir/runs/xref_adgm_bm25_top10.trec",
        "qrels": ROOT / "full_ir/qrels/xref_adgm_test.qrels",
        "output_stem": "xref_adgm_bm25_gpt52",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Batch output root. Defaults to controlled_comparison_pilot/batch_runs/sample_${sample_size}.",
    )
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                qrels.setdefault(parts[0], {})[parts[2]] = int(parts[3])
    return qrels


def read_trec_run(path: Path) -> dict[str, list[str]]:
    run: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 6:
                qid, _q0, docid, rank, score, _tag = parts
                run[qid].append((int(rank), docid, float(score)))
    return {
        qid: [docid for _rank, docid, _score in sorted(rows, key=lambda x: (x[0], -x[2]))[:K]]
        for qid, rows in run.items()
    }


def load_corpus() -> dict[str, str]:
    corpus: dict[str, str] = {}
    for row in read_jsonl(ROOT / "full_ir/corpora/adgm_shared_passages.jsonl"):
        pid = str(row.get("id") or row.get("passage_uid") or "")
        text = str(row.get("contents") or row.get("passage") or "")
        if pid:
            corpus[pid] = text
    return corpus


def prompt_version(prompt_file: str) -> str:
    path = REPO_ROOT / prompt_file
    first = path.read_text(encoding="utf-8").splitlines()[0].strip()
    return first.split(":", 1)[1].strip() if first.startswith("Prompt-Version:") else ""


def passage_count_bucket(row: dict[str, Any]) -> str:
    count = len(row.get("Passages") or [])
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 4:
        return "3-4"
    return "5+"


def stratum_key(row: dict[str, Any], fields: list[str]) -> str:
    parts: list[str] = []
    for field in fields:
        value = passage_count_bucket(row) if field == "passage_count_bucket" else row.get(field)
        parts.append(str(value if value not in (None, "") else "MISSING"))
    return " × ".join(parts)


def stratified_sample(
    rows: list[dict[str, Any]],
    *,
    qid_field: str,
    strata_fields: list[str],
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    if sample_size > len(rows):
        raise ValueError(f"Requested sample_size={sample_size}, but only {len(rows)} rows are available")

    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[stratum_key(row, strata_fields)].append(row)

    for group in by_stratum.values():
        group.sort(key=lambda r: str(r.get(qid_field) or ""))
        rng.shuffle(group)

    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    total = len(rows)
    for key, group in by_stratum.items():
        exact = sample_size * len(group) / total
        quotas[key] = min(len(group), int(exact))
        remainders.append((exact - int(exact), key))

    assigned = sum(quotas.values())
    for _rem, key in sorted(remainders, key=lambda x: (-x[0], x[1])):
        if assigned >= sample_size:
            break
        if quotas[key] < len(by_stratum[key]):
            quotas[key] += 1
            assigned += 1

    # If some small strata saturated, fill from remaining capacity.
    while assigned < sample_size:
        candidates = [k for k, g in by_stratum.items() if quotas[k] < len(g)]
        if not candidates:
            break
        key = sorted(candidates, key=lambda k: (quotas[k] / len(by_stratum[k]), k))[0]
        quotas[key] += 1
        assigned += 1

    selected: list[dict[str, Any]] = []
    for key in sorted(by_stratum):
        selected.extend(by_stratum[key][: quotas[key]])
    selected.sort(key=lambda r: str(r.get(qid_field) or ""))

    return selected, {
        "fields": strata_fields,
        "population_counts": dict(sorted(Counter(stratum_key(r, strata_fields) for r in rows).items())),
        "selected_counts": dict(sorted(Counter(stratum_key(r, strata_fields) for r in selected).items())),
    }


def relevant_ids_for_row(row: dict[str, Any], qrels: dict[str, dict[str, int]], qid: str) -> list[str]:
    if qrels.get(qid):
        return [pid for pid, rel in qrels[qid].items() if rel > 0]
    ids: list[str] = []
    for passage in row.get("Passages") or []:
        pid = passage.get("ID") or passage.get("PassageID")
        if pid:
            ids.append(str(pid))
    for field in ("source_passage_id", "target_passage_id"):
        if row.get(field):
            ids.append(str(row[field]))
    return ids


def format_retrieved_passage(rank: int, passage_id: str, passage_text: str) -> str:
    return f"Passage {rank}\nPassage ID: {passage_id}\nText: {passage_text}"


def make_records(
    selected: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    condition: str,
    prompt_file: str,
    prompt_ver: str,
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    corpus: dict[str, str],
    seed: int,
    sample_size: int,
    warnings: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in selected:
        qid = str(row.get(cfg["qid"]) or "")
        top_ids = run.get(qid, [])[:K]
        passages: list[str] = []
        kept_ids: list[str] = []
        for idx, pid in enumerate(top_ids, start=1):
            text = corpus.get(pid)
            if text is None:
                warnings.append(f"{cfg['dataset_name']} {qid}: missing corpus text for {pid}")
                continue
            kept_ids.append(pid)
            passages.append(format_retrieved_passage(idx, pid, text))

        record: dict[str, Any] = {
            "QuestionID": qid,
            "Question": str(row.get(cfg["question"]) or ""),
            "RetrievedPassages": passages,
            "RetrievedPassageIDs": kept_ids,
            "dataset": cfg["dataset_name"],
            "split": "test",
            "sample_seed": seed,
            "sample_size": sample_size,
            "retriever": "bm25",
            "retrieval_setting": RETRIEVAL_SETTING,
            "retrieval_corpus": RETRIEVAL_CORPUS,
            "k": K,
            "prompt_condition": condition,
            "prompt_version": prompt_ver,
            "prompt_file": prompt_file,
            "relevant_passage_ids": relevant_ids_for_row(row, qrels, qid),
        }
        if row.get("gold_answer") is not None:
            record["gold_answer"] = row.get("gold_answer")
        records.append(record)
    return records


def load_source(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_jsonl(cfg["source"]) if cfg["source_format"] == "jsonl" else read_json(cfg["source"])
    if not isinstance(rows, list):
        raise ValueError(f"{cfg['source']} did not contain a JSON list")
    return rows


def write_report(output_root: Path, report: dict[str, Any]) -> None:
    write_json(output_root / "reports/sample_selection_report.json", report)
    lines = [
        "# Sample Selection Report",
        "",
        f"- sample_size: {report['sample_size']}",
        f"- seed: {report['seed']}",
        "",
    ]
    for dataset, info in report["datasets"].items():
        lines.extend(
            [
                f"## {dataset}",
                f"- source_rows: {info['source_rows']}",
                f"- eligible_rows: {info['eligible_rows']}",
                f"- selected_rows: {info['selected_rows']}",
                f"- stratification_fields: {', '.join(info['stratification']['fields'])}",
                f"- every_selected_query_has_10_retrieved_passages: {info['every_selected_query_has_10_retrieved_passages']}",
                f"- selected_ids: {', '.join(info['selected_sample_ids'][:20])}"
                + (" ..." if len(info["selected_sample_ids"]) > 20 else ""),
                "",
                "| stratum | selected | population |",
                "| --- | ---: | ---: |",
            ]
        )
        selected_counts = info["stratification"]["selected_counts"]
        population_counts = info["stratification"]["population_counts"]
        for key in sorted(population_counts):
            lines.append(f"| {key} | {selected_counts.get(key, 0)} | {population_counts[key]} |")
        lines.append("")
    if report["warnings"]:
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("## Warnings")
        lines.append("- None")
    write_text(output_root / "reports/sample_selection_report.md", "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_root = args.output_root or ROOT / "batch_runs" / f"sample_{args.sample_size}"
    for subdir in ["answer_inputs", "answers", "repass_inputs", "repass_outputs", "reports", "logs"]:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    corpus = load_corpus()
    warnings: list[str] = []
    report: dict[str, Any] = {
        "sample_size": args.sample_size,
        "seed": args.seed,
        "output_root": str(output_root),
        "datasets": {},
        "warnings": warnings,
    }

    for dataset_key, cfg in DATASETS.items():
        source_rows = load_source(cfg)
        qrels = read_qrels(cfg["qrels"])
        run = read_trec_run(cfg["run"])
        eligible = [
            row
            for row in source_rows
            if str(row.get(cfg["qid"]) or "") in qrels and str(row.get(cfg["qid"]) or "") in run
        ]
        selected, strat_report = stratified_sample(
            eligible,
            qid_field=cfg["qid"],
            strata_fields=cfg["strata"],
            sample_size=args.sample_size,
            seed=args.seed,
        )
        selected_ids = [str(row.get(cfg["qid"]) or "") for row in selected]
        every_has_10 = all(len(run.get(qid, [])[:K]) == K for qid in selected_ids)
        if not every_has_10:
            warnings.append(f"{cfg['dataset_name']}: at least one selected query lacks {K} run entries")

        for condition in ("general", "task_aware"):
            prompt_file = PROMPT_FILES[condition][dataset_key]
            records = make_records(
                selected,
                cfg,
                condition=condition,
                prompt_file=prompt_file,
                prompt_ver=prompt_version(prompt_file),
                run=run,
                qrels=qrels,
                corpus=corpus,
                seed=args.seed,
                sample_size=args.sample_size,
                warnings=warnings,
            )
            out = output_root / "answer_inputs" / condition / f"{cfg['output_stem']}_input.json"
            write_json(out, records)
            print(f"Wrote {out}: {len(records)} records")

        report["datasets"][dataset_key] = {
            "dataset_name": cfg["dataset_name"],
            "source_path": str(cfg["source"]),
            "source_rows": len(source_rows),
            "eligible_rows": len(eligible),
            "selected_rows": len(selected),
            "selected_sample_ids": selected_ids,
            "stratification": strat_report,
            "every_selected_query_has_10_retrieved_passages": every_has_10,
        }

    write_report(output_root, report)
    print(f"Wrote {output_root / 'reports/sample_selection_report.json'}")
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for warning in warnings[:20]:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
