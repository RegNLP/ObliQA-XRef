#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "Rerun IR-only for big ObliQA-XRef runs"
echo "Started: $(date)"
echo "============================================================"

run_ir_only() {
  local corpus="$1"
  local config="$2"
  local suffix="$3"

  echo ""
  echo "============================================================"
  echo "IR-ONLY CURATION: corpus=${corpus}, suffix=${suffix}"
  echo "Judge skipped; answer validation skipped"
  echo "Started: $(date)"
  echo "============================================================"

  python -m obliqaxref curate \
    -c "${config}" \
    --preset smoke \
    --pilot \
    --pilot-suffix "${suffix}" \
    --skip-judge \
    --skip-answer

  echo "Completed: corpus=${corpus}, suffix=${suffix} at $(date)"
}

run_ir_only "adgm"  "configs/adgm_dev.yaml"  "pilot_mixed_big2000"
run_ir_only "ukfin" "configs/ukfin_dev.yaml" "pilot_mixed_big2000"
run_ir_only "adgm"  "configs/adgm_dev.yaml"  "pilot_hardenriched_big1000"
run_ir_only "ukfin" "configs/ukfin_dev.yaml" "pilot_hardenriched_big1000"

echo ""
echo "============================================================"
echo "Sanity check after IR-only rerun"
echo "============================================================"

python - <<'PY'
from pathlib import Path
import json
from collections import Counter

runs = [
    ("ADGM mixed", "adgm", "pilot_mixed_big2000"),
    ("UKFIN mixed", "ukfin", "pilot_mixed_big2000"),
    ("ADGM hard", "adgm", "pilot_hardenriched_big1000"),
    ("UKFIN hard", "ukfin", "pilot_hardenriched_big1000"),
]

for name, corpus, suffix in runs:
    print("\n" + "="*80)
    print(name)
    gen = Path(f"runs/generate_{corpus}/out_{suffix}")
    cur = Path(f"runs/curate_{corpus}/out_{suffix}")

    items = gen / "generator/items.jsonl"
    n_items = sum(1 for _ in items.open()) if items.exists() else None
    print("items.jsonl:", n_items if n_items is not None else "MISSING")

    print("\nTREC qid counts:")
    for trec_name in [
        "bm25.trec",
        "ft_e5.trec",
        "rrf_bm25_e5.trec",
        "ce_rerank_union200.trec",
        "bm25_xref_expand.trec",
        "e5_xref_expand.trec",
        "rrf_xref_expand.trec",
    ]:
        p = gen / trec_name
        if not p.exists():
            print(f"  {trec_name}: MISSING")
            continue
        qids = set()
        with p.open() as f:
            for line in f:
                parts = line.split()
                if parts:
                    qids.add(parts[0])
        status = "OK" if n_items is not None and len(qids) == n_items else "CHECK"
        print(f"  {trec_name}: {len(qids)} qids {status}")

    decisions = cur / "decisions.jsonl"
    if decisions.exists():
        labels = Counter()
        for line in decisions.open():
            row = json.loads(line)
            labels[row.get("ir_difficulty_label", "MISSING")] += 1
        print("\ndecisions ir_difficulty_label:", dict(labels))
    else:
        print("\ndecisions.jsonl: MISSING")

    for fname in ["final_benchmark.jsonl", "final_dependency_valid.jsonl", "final_hard.jsonl"]:
        p = cur / fname
        if p.exists():
            print(f"{fname}: {sum(1 for _ in p.open())}")
        else:
            print(f"{fname}: MISSING")
PY

echo ""
echo "DONE"
