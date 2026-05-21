from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# -----------------------------
# Helpers
# -----------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def norm_question(q: str | None) -> str:
    s = (q or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s?]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def detect_corpus_from_path(p: Path) -> str:
    text = str(p).lower()
    if "/curate_adgm/" in text or "adgm" in text:
        return "adgm"
    if "/curate_ukfin/" in text or "ukfin" in text:
        return "ukfin"
    return "unknown"


def infer_sampling_regime_from_suffix(suffix: str) -> str:
    s = suffix.lower()
    if "hardenriched" in s:
        return "hard_enriched"
    return "mixed_difficulty"


def infer_run_group_from_suffix(suffix: str) -> str:
    s = suffix.lower()
    if "big2000" in s:
        return "big2000"
    if "big1000" in s:
        return "big1000"
    if "topup_q2" in s:
        return "topup_q2"
    if "xl" in s:
        return "xl"
    return "unknown"


def extract_run_suffix_from_path(p: Path) -> str:
    # Expect .../runs/curate_<corpus>/out_<suffix>/final_dependency_valid.jsonl
    parts = [comp for comp in p.parts]
    try:
        out_dir = next(seg for seg in parts if seg.startswith("out_"))
        return out_dir[len("out_"):]
    except StopIteration:
        return ""


def normalize_method(m: str | None) -> str:
    s = (m or "").strip().upper()
    if s in {"DPEL", "SCHEMA"}:
        return s
    if s.lower() in {"dpel", "schema"}:
        return s.upper()
    return s or "UNKNOWN"


@dataclass
class MergeItem:
    raw: dict[str, Any]
    corpus: str
    method: str
    sampling_regime: str
    run_group: str
    run_suffix: str
    source_path: str

    @property
    def src_id(self) -> str:
        return str(self.raw.get("source_passage_id") or self.raw.get("source_passage_uid") or "").strip()

    @property
    def tgt_id(self) -> str:
        return str(self.raw.get("target_passage_id") or self.raw.get("target_passage_uid") or "").strip()

    @property
    def question(self) -> str:
        return str(self.raw.get("question") or "")


def augment_item(obj: dict[str, Any], *, meta: dict[str, str]) -> dict[str, Any]:
    out = dict(obj)
    out.update({
        "corpus": meta["corpus"],
        "sampling_regime": meta["sampling_regime"],
        "run_group": meta["run_group"],
        "run_suffix": meta["run_suffix"],
        "source_path": meta["source_path"],
        "generation_method": normalize_method(obj.get("method") or obj.get("generation_method")),
        "final_basis": "dependency_valid",
    })
    return out


def merge_inputs(paths: list[Path]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for p in paths:
        if not p.exists():
            print(f"[WARN] Missing input: {p}", file=sys.stderr)
            continue
        corpus = detect_corpus_from_path(p)
        suffix = extract_run_suffix_from_path(p)
        sampling_regime = infer_sampling_regime_from_suffix(suffix)
        run_group = infer_run_group_from_suffix(suffix)
        rows = read_jsonl(p)
        meta = {
            "corpus": corpus,
            "sampling_regime": sampling_regime,
            "run_group": run_group,
            "run_suffix": suffix,
            "source_path": str(p),
        }
        for r in rows:
            merged.append(augment_item(r, meta=meta))
    return merged


def dedup_items(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Primary dedup by (corpus, src_id, tgt_id, norm_q)
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    # Diagnostics: question-only duplicates per corpus
    q_only_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    def _src(x):
        return str(x.get("source_passage_id") or x.get("source_passage_uid") or "").strip()

    def _tgt(x):
        return str(x.get("target_passage_id") or x.get("target_passage_uid") or "").strip()

    for r in rows:
        c = str(r.get("corpus") or "").strip().lower()
        nq = norm_question(r.get("question"))
        k = (c, _src(r), _tgt(r), nq)
        if k not in by_key:
            by_key[k] = r
        q_only_map[(c, nq)].append(r)

    deduped = list(by_key.values())

    # Build diagnostics for question-only duplicates
    q_dups = {
        f"{c}:{q}": [
            {
                "item_id": it.get("item_id"),
                "source_passage_id": it.get("source_passage_id"),
                "target_passage_id": it.get("target_passage_id"),
                "run_suffix": it.get("run_suffix"),
                "sampling_regime": it.get("sampling_regime"),
                "generation_method": it.get("generation_method"),
            }
            for it in lst
        ]
        for (c, q), lst in q_only_map.items() if len(lst) > 1
    }

    diag = {
        "raw_count": len(rows),
        "dedup_count": len(deduped),
        "question_only_duplicates": {
            "groups": len(q_dups),
            "examples": dict(list(q_dups.items())[:50])  # limit preview
        },
    }
    return deduped, diag


def _stratified_split(items: list[dict[str, Any]], seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    # Stratify by generation_method + sampling_regime
    import random

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        buckets[(it.get("generation_method", "UNKNOWN"), it.get("sampling_regime", ""))].append(it)

    splits = {"train": [], "dev": [], "test": []}
    random.seed(seed)
    for key, group in buckets.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(0.8 * n)
        n_dev = int(0.1 * n)
        train = group[:n_train]
        dev = group[n_train:n_train + n_dev]
        test = group[n_train + n_dev:]
        splits["train"].extend(train)
        splits["dev"].extend(dev)
        splits["test"].extend(test)
    return splits


def _counts_by(items: list[dict[str, Any]], *fields: str) -> dict[tuple, int]:
    ctr: Counter = Counter()
    for it in items:
        key = tuple((it.get(f) or "") for f in fields)
        ctr[key] += 1
    return dict(ctr)


def _nest_from_tuple_keys(d: dict[tuple, Any]) -> dict[str, Any]:
    """Convert a dict with tuple keys into a nested dict of str -> ... for JSON safety.

    Example: {('SCHEMA','hard_enriched'): 12} -> {'SCHEMA': {'hard_enriched': 12}}
    """
    out: dict[str, Any] = {}
    for key, value in d.items():
        if not isinstance(key, tuple):
            # Fallback: coerce to str key
            out[str(key)] = value
            continue
        cursor = out
        # build all but last as nested dicts
        for part in key[:-1]:
            part_s = str(part)
            if part_s not in cursor or not isinstance(cursor.get(part_s), dict):
                cursor[part_s] = {}
            cursor = cursor[part_s]
        cursor[str(key[-1])] = value
    return out


def write_qrels_for_test(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            iid = it.get("item_id")
            sp = it.get("source_passage_id") or it.get("source_passage_uid")
            tp = it.get("target_passage_id") or it.get("target_passage_uid")
            if not (iid and sp and tp):
                continue
            f.write(f"{iid} Q0 {sp} 1\n")
            f.write(f"{iid} Q0 {tp} 1\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Merge/dedup/finalize large ObliQA-XRef waves (dependency_valid basis)")
    ap.add_argument("--out-root", default="ObliQA-XRef_Out_Datasets/final_large_merged", help="Output root directory")
    ap.add_argument("--seed", type=int, default=42, help="Split seed (default: 42)")
    ap.add_argument("--inputs", nargs="*", default=[], help="Explicit input file paths (final_dependency_valid.jsonl)")
    args = ap.parse_args(argv)

    default_inputs = [
        "runs/curate_adgm/out_pilot_mixed_big2000/final_dependency_valid.jsonl",
        "runs/curate_ukfin/out_pilot_mixed_big2000/final_dependency_valid.jsonl",
        "runs/curate_adgm/out_pilot_hardenriched_big1000/final_dependency_valid.jsonl",
        "runs/curate_ukfin/out_pilot_hardenriched_big1000/final_dependency_valid.jsonl",
        "runs/curate_ukfin/out_pilot_ukfin_mixed_xl/final_dependency_valid.jsonl",
        "runs/curate_ukfin/out_pilot_ukfin_hardenriched_xl/final_dependency_valid.jsonl",
        "runs/curate_adgm/out_pilot_adgm_mixed_xl/final_dependency_valid.jsonl",
        "runs/curate_adgm/out_pilot_adgm_hardenriched_xl/final_dependency_valid.jsonl",
        "runs/curate_ukfin/out_pilot_ukfin_mixed_topup_q2/final_dependency_valid.jsonl",
        "runs/curate_ukfin/out_pilot_ukfin_hardenriched_topup_q2/final_dependency_valid.jsonl",
    ]

    input_paths = [Path(p) for p in (args.inputs or default_inputs)]
    out_root = Path(args.out_root)

    # Merge and augment
    merged_raw = merge_inputs(input_paths)

    # Validation checkpoints from user notes (use warnings only)
    expected_raw = 17569
    if len(merged_raw) != expected_raw:
        print(f"[WARN] Raw count {len(merged_raw)} != expected {expected_raw}")

    # Write merged raw
    write_jsonl(out_root / "merged_all_raw.jsonl", merged_raw)

    # Dedup
    merged_dedup, diag = dedup_items(merged_raw)
    write_jsonl(out_root / "merged_all_dedup.jsonl", merged_dedup)
    write_json(out_root / "stats" / "duplicates_report.json", diag)

    # Per-corpus splits
    by_corpus: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in merged_dedup:
        c = (it.get("corpus") or "").lower()
        by_corpus[c].append(it)

    split_stats_rows: list[dict[str, Any]] = []
    final_stats = {
        "raw_count": diag.get("raw_count"),
        "dedup_count": diag.get("dedup_count"),
        "by_corpus_counts": {k: len(v) for k, v in by_corpus.items()},
    }

    for corpus, items in by_corpus.items():
        # Write corpus-level final
        corpus_dir = out_root / f"ObliQA-XRef-{corpus.upper()}-ALL"
        write_jsonl(corpus_dir / "final.jsonl", items)

        # Split 80/10/10 stratified by method+sampling
        splits = _stratified_split(items, seed=args.seed)
        for name, subset in splits.items():
            write_jsonl(corpus_dir / f"{name}.jsonl", subset)

        # Write qrels for test split (convenience for some tools)
        write_qrels_for_test(splits.get("test", []), corpus_dir / "qrels.txt")

        # Stats per corpus
        counts_c = {
            "total": len(items),
            # Single-field keys are already strings
            "by_method": {str(k[0]): v for k, v in _counts_by(items, "generation_method").items()},
            "by_sampling_regime": {str(k[0]): v for k, v in _counts_by(items, "sampling_regime").items()},
            # Convert tuple-key dict to nested for JSON serialization
            "by_method_sampling": _nest_from_tuple_keys(
                _counts_by(items, "generation_method", "sampling_regime")
            ),
            "by_split": {k: len(v) for k, v in splits.items()},
        }
        final_stats[f"{corpus}_stats"] = counts_c

        # CSV rows
        for m, sub in counts_c["by_method_sampling"].items():
            if not isinstance(sub, dict):
                # Defensive: if serialization produced flat mapping, coerce
                sub = {"": sub}
            for sreg, cnt in sub.items():
                split_stats_rows.append({
                    "corpus": corpus,
                    "generation_method": m,
                    "sampling_regime": sreg,
                    "count": cnt,
                    "train": len([x for x in splits["train"] if x.get("generation_method") == m and x.get("sampling_regime") == sreg]),
                    "dev": len([x for x in splits["dev"] if x.get("generation_method") == m and x.get("sampling_regime") == sreg]),
                    "test": len([x for x in splits["test"] if x.get("generation_method") == m and x.get("sampling_regime") == sreg]),
                })

    write_json(out_root / "stats" / "final_merge_stats.json", final_stats)
    write_csv(out_root / "stats" / "final_merge_stats.csv", split_stats_rows,
              ["corpus", "generation_method", "sampling_regime", "count", "train", "dev", "test"])

    # Optional assertions (warn-only) using provided expected figures
    exp_adgm = 4992
    exp_ukfin = 12402
    got_adgm = len(by_corpus.get("adgm", []))
    got_ukfin = len(by_corpus.get("ukfin", []))
    if got_adgm != exp_adgm:
        print(f"[WARN] ADGM dedup count {got_adgm} != expected {exp_adgm}")
    if got_ukfin != exp_ukfin:
        print(f"[WARN] UKFIN dedup count {got_ukfin} != expected {exp_ukfin}")

    print("\n✓ Final merge/dedup complete")
    print(f"  Raw:   {diag.get('raw_count')}  → Dedup: {diag.get('dedup_count')}")
    for c in ("adgm", "ukfin"):
        print(f"  {c.upper()}: {len(by_corpus.get(c, []))}")
    print(f"  Output root: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
