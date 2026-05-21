#!/usr/bin/env bash
set -euo pipefail

TEMP="0.0"

# -----------------------------
# XL unique-pair expansion
# -----------------------------
UKFIN_MIXED_SUFFIX="pilot_ukfin_mixed_xl"
UKFIN_MIXED_PILOT_N=5000
UKFIN_MIXED_MAX_PAIRS=3500
UKFIN_MIXED_MAX_Q=1

UKFIN_HARD_SUFFIX="pilot_ukfin_hardenriched_xl"
UKFIN_HARD_PILOT_N=3000
UKFIN_HARD_MAX_PAIRS=2500
UKFIN_HARD_MAX_Q=1

ADGM_MIXED_SUFFIX="pilot_adgm_mixed_xl"
ADGM_MIXED_PILOT_N=5000
ADGM_MIXED_MAX_PAIRS=1600
ADGM_MIXED_MAX_Q=1

ADGM_HARD_SUFFIX="pilot_adgm_hardenriched_xl"
ADGM_HARD_PILOT_N=3000
ADGM_HARD_MAX_PAIRS=1200
ADGM_HARD_MAX_Q=1

# -----------------------------
# UKFIN q2 top-up
# -----------------------------
UKFIN_MIXED_Q2_SUFFIX="pilot_ukfin_mixed_topup_q2"
UKFIN_MIXED_Q2_PILOT_N=5000
UKFIN_MIXED_Q2_MAX_PAIRS=2500
UKFIN_MIXED_Q2_MAX_Q=2

UKFIN_HARD_Q2_SUFFIX="pilot_ukfin_hardenriched_topup_q2"
UKFIN_HARD_Q2_PILOT_N=3000
UKFIN_HARD_Q2_MAX_PAIRS=1800
UKFIN_HARD_Q2_MAX_Q=2

echo "============================================================"
echo "ObliQA-XRef XL 48h generation batch"
echo "Started: $(date)"
echo "Policy: dependency_valid final benchmark; answer validation skipped"
echo "No new features; expansion only"
echo "============================================================"

echo ""
echo "=== Environment check ==="
python - <<'PY'
import os
keys = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT_GPT52",
]
for k in keys:
    v = os.getenv(k)
    if k == "AZURE_OPENAI_API_KEY":
        print(k, "SET" if v else "MISSING", f"(length {len(v)})" if v else "")
    else:
        print(k, v if v else "MISSING")
missing = [k for k in keys if not os.getenv(k)]
if missing:
    raise SystemExit(f"Missing environment variables: {missing}")
PY

count_file() {
  local label="$1"
  local path="$2"
  if [ -f "$path" ]; then
    printf "%-90s %s\n" "$label" "$(wc -l < "$path" | tr -d ' ')"
  else
    printf "%-90s %s\n" "$label" "MISSING"
  fi
}

run_generate_curate() {
  local corpus="$1"
  local config="$2"
  local suffix="$3"
  local pilot_n="$4"
  local max_pairs="$5"
  local max_q_per_pair="$6"
  local sampling_mode="$7"

  echo ""
  echo "============================================================"
  echo "GENERATE: corpus=${corpus}, suffix=${suffix}"
  echo "sampling=${sampling_mode}, pilot_n=${pilot_n}, max_pairs=${max_pairs}, max_q_per_pair=${max_q_per_pair}"
  echo "Started: $(date)"
  echo "============================================================"

  python -m obliqaxref generate \
    -c "${config}" \
    --preset smoke \
    --pilot \
    --pilot-n "${pilot_n}" \
    --pilot-suffix "${suffix}" \
    --method both \
    --max-pairs "${max_pairs}" \
    --max-q-per-pair "${max_q_per_pair}" \
    --sampling-mode "${sampling_mode}" \
    --temperature "${TEMP}"

  echo ""
  echo "============================================================"
  echo "CURATE: corpus=${corpus}, suffix=${suffix}"
  echo "IR + judge enabled; answer validation skipped"
  echo "Started: $(date)"
  echo "============================================================"

  python -m obliqaxref curate \
    -c "${config}" \
    --preset smoke \
    --pilot \
    --pilot-suffix "${suffix}" \
    --skip-answer

  echo ""
  echo "============================================================"
  echo "COMPLETED: corpus=${corpus}, suffix=${suffix}"
  echo "Finished: $(date)"
  echo "============================================================"
}

# ============================================================
# UKFIN FIRST: highest risk for dataset size
# ============================================================

run_generate_curate "ukfin" "configs/ukfin_dev.yaml" "${UKFIN_MIXED_SUFFIX}" \
  "${UKFIN_MIXED_PILOT_N}" "${UKFIN_MIXED_MAX_PAIRS}" "${UKFIN_MIXED_MAX_Q}" "mixed_difficulty"

run_generate_curate "ukfin" "configs/ukfin_dev.yaml" "${UKFIN_HARD_SUFFIX}" \
  "${UKFIN_HARD_PILOT_N}" "${UKFIN_HARD_MAX_PAIRS}" "${UKFIN_HARD_MAX_Q}" "hard_enriched"

# ============================================================
# ADGM XL expansion
# ============================================================

run_generate_curate "adgm" "configs/adgm_dev.yaml" "${ADGM_MIXED_SUFFIX}" \
  "${ADGM_MIXED_PILOT_N}" "${ADGM_MIXED_MAX_PAIRS}" "${ADGM_MIXED_MAX_Q}" "mixed_difficulty"

run_generate_curate "adgm" "configs/adgm_dev.yaml" "${ADGM_HARD_SUFFIX}" \
  "${ADGM_HARD_PILOT_N}" "${ADGM_HARD_MAX_PAIRS}" "${ADGM_HARD_MAX_Q}" "hard_enriched"

# ============================================================
# UKFIN q2 top-ups: only to increase size if unique-pair expansion is insufficient
# ============================================================

run_generate_curate "ukfin" "configs/ukfin_dev.yaml" "${UKFIN_MIXED_Q2_SUFFIX}" \
  "${UKFIN_MIXED_Q2_PILOT_N}" "${UKFIN_MIXED_Q2_MAX_PAIRS}" "${UKFIN_MIXED_Q2_MAX_Q}" "mixed_difficulty"

run_generate_curate "ukfin" "configs/ukfin_dev.yaml" "${UKFIN_HARD_Q2_SUFFIX}" \
  "${UKFIN_HARD_Q2_PILOT_N}" "${UKFIN_HARD_Q2_MAX_PAIRS}" "${UKFIN_HARD_Q2_MAX_Q}" "hard_enriched"

echo ""
echo "============================================================"
echo "XL RUN SUMMARY COUNTS"
echo "Finished all runs: $(date)"
echo "============================================================"

echo ""
echo "--- UKFIN MIXED XL ---"
count_file "UKFIN mixed XL DPEL QAs" "runs/generate_ukfin/out_${UKFIN_MIXED_SUFFIX}/dpel/dpel.qa.jsonl"
count_file "UKFIN mixed XL SCHEMA QAs" "runs/generate_ukfin/out_${UKFIN_MIXED_SUFFIX}/schema/schema.qa.jsonl"
count_file "UKFIN mixed XL judge PASS" "runs/curate_ukfin/out_${UKFIN_MIXED_SUFFIX}/curate_judge/judge_responses_pass.jsonl"
count_file "UKFIN mixed XL judge DROP" "runs/curate_ukfin/out_${UKFIN_MIXED_SUFFIX}/curate_judge/judge_responses_drop.jsonl"
count_file "UKFIN mixed XL final_dependency_valid" "runs/curate_ukfin/out_${UKFIN_MIXED_SUFFIX}/final_dependency_valid.jsonl"

echo ""
echo "--- UKFIN HARD XL ---"
count_file "UKFIN hard XL DPEL QAs" "runs/generate_ukfin/out_${UKFIN_HARD_SUFFIX}/dpel/dpel.qa.jsonl"
count_file "UKFIN hard XL SCHEMA QAs" "runs/generate_ukfin/out_${UKFIN_HARD_SUFFIX}/schema/schema.qa.jsonl"
count_file "UKFIN hard XL judge PASS" "runs/curate_ukfin/out_${UKFIN_HARD_SUFFIX}/curate_judge/judge_responses_pass.jsonl"
count_file "UKFIN hard XL judge DROP" "runs/curate_ukfin/out_${UKFIN_HARD_SUFFIX}/curate_judge/judge_responses_drop.jsonl"
count_file "UKFIN hard XL final_dependency_valid" "runs/curate_ukfin/out_${UKFIN_HARD_SUFFIX}/final_dependency_valid.jsonl"

echo ""
echo "--- ADGM MIXED XL ---"
count_file "ADGM mixed XL DPEL QAs" "runs/generate_adgm/out_${ADGM_MIXED_SUFFIX}/dpel/dpel.qa.jsonl"
count_file "ADGM mixed XL SCHEMA QAs" "runs/generate_adgm/out_${ADGM_MIXED_SUFFIX}/schema/schema.qa.jsonl"
count_file "ADGM mixed XL judge PASS" "runs/curate_adgm/out_${ADGM_MIXED_SUFFIX}/curate_judge/judge_responses_pass.jsonl"
count_file "ADGM mixed XL judge DROP" "runs/curate_adgm/out_${ADGM_MIXED_SUFFIX}/curate_judge/judge_responses_drop.jsonl"
count_file "ADGM mixed XL final_dependency_valid" "runs/curate_adgm/out_${ADGM_MIXED_SUFFIX}/final_dependency_valid.jsonl"

echo ""
echo "--- ADGM HARD XL ---"
count_file "ADGM hard XL DPEL QAs" "runs/generate_adgm/out_${ADGM_HARD_SUFFIX}/dpel/dpel.qa.jsonl"
count_file "ADGM hard XL SCHEMA QAs" "runs/generate_adgm/out_${ADGM_HARD_SUFFIX}/schema/schema.qa.jsonl"
count_file "ADGM hard XL judge PASS" "runs/curate_adgm/out_${ADGM_HARD_SUFFIX}/curate_judge/judge_responses_pass.jsonl"
count_file "ADGM hard XL judge DROP" "runs/curate_adgm/out_${ADGM_HARD_SUFFIX}/curate_judge/judge_responses_drop.jsonl"
count_file "ADGM hard XL final_dependency_valid" "runs/curate_adgm/out_${ADGM_HARD_SUFFIX}/final_dependency_valid.jsonl"

echo ""
echo "--- UKFIN MIXED Q2 TOP-UP ---"
count_file "UKFIN mixed q2 DPEL QAs" "runs/generate_ukfin/out_${UKFIN_MIXED_Q2_SUFFIX}/dpel/dpel.qa.jsonl"
count_file "UKFIN mixed q2 SCHEMA QAs" "runs/generate_ukfin/out_${UKFIN_MIXED_Q2_SUFFIX}/schema/schema.qa.jsonl"
count_file "UKFIN mixed q2 judge PASS" "runs/curate_ukfin/out_${UKFIN_MIXED_Q2_SUFFIX}/curate_judge/judge_responses_pass.jsonl"
count_file "UKFIN mixed q2 judge DROP" "runs/curate_ukfin/out_${UKFIN_MIXED_Q2_SUFFIX}/curate_judge/judge_responses_drop.jsonl"
count_file "UKFIN mixed q2 final_dependency_valid" "runs/curate_ukfin/out_${UKFIN_MIXED_Q2_SUFFIX}/final_dependency_valid.jsonl"

echo ""
echo "--- UKFIN HARD Q2 TOP-UP ---"
count_file "UKFIN hard q2 DPEL QAs" "runs/generate_ukfin/out_${UKFIN_HARD_Q2_SUFFIX}/dpel/dpel.qa.jsonl"
count_file "UKFIN hard q2 SCHEMA QAs" "runs/generate_ukfin/out_${UKFIN_HARD_Q2_SUFFIX}/schema/schema.qa.jsonl"
count_file "UKFIN hard q2 judge PASS" "runs/curate_ukfin/out_${UKFIN_HARD_Q2_SUFFIX}/curate_judge/judge_responses_pass.jsonl"
count_file "UKFIN hard q2 judge DROP" "runs/curate_ukfin/out_${UKFIN_HARD_Q2_SUFFIX}/curate_judge/judge_responses_drop.jsonl"
count_file "UKFIN hard q2 final_dependency_valid" "runs/curate_ukfin/out_${UKFIN_HARD_Q2_SUFFIX}/final_dependency_valid.jsonl"

echo ""
echo "============================================================"
echo "Difficulty summaries"
echo "============================================================"

python - <<'PY'
import json
from pathlib import Path

runs = [
    ("UKFIN mixed XL", "ukfin", "pilot_ukfin_mixed_xl"),
    ("UKFIN hard XL", "ukfin", "pilot_ukfin_hardenriched_xl"),
    ("ADGM mixed XL", "adgm", "pilot_adgm_mixed_xl"),
    ("ADGM hard XL", "adgm", "pilot_adgm_hardenriched_xl"),
    ("UKFIN mixed q2", "ukfin", "pilot_ukfin_mixed_topup_q2"),
    ("UKFIN hard q2", "ukfin", "pilot_ukfin_hardenriched_topup_q2"),
]

for name, corpus, suffix in runs:
    path = Path(f"runs/curate_{corpus}/out_{suffix}/stats.json")
    print(f"\n--- {name} ---")
    if not path.exists():
        print(f"MISSING: {path}")
        continue
    data = json.loads(path.read_text())
    total = data.get("total_items", 0) or 0
    counts = data.get("ir_difficulty_label_counts", {})
    print("total_items:", total)
    print("final_dependency_valid:", sum(1 for _ in Path(f"runs/curate_{corpus}/out_{suffix}/final_dependency_valid.jsonl").open()) if Path(f"runs/curate_{corpus}/out_{suffix}/final_dependency_valid.jsonl").exists() else "MISSING")
    for k in ["easy", "medium", "hard", "source_only", "target_only", "neither"]:
        v = counts.get(k, 0)
        pct = (100.0 * v / total) if total else 0.0
        print(f"{k:12s} {v:5d} ({pct:5.1f}%)")
PY

echo ""
echo "============================================================"
echo "Judge summaries"
echo "============================================================"

python - <<'PY'
import json
from pathlib import Path

runs = [
    ("UKFIN mixed XL", "ukfin", "pilot_ukfin_mixed_xl"),
    ("UKFIN hard XL", "ukfin", "pilot_ukfin_hardenriched_xl"),
    ("ADGM mixed XL", "adgm", "pilot_adgm_mixed_xl"),
    ("ADGM hard XL", "adgm", "pilot_adgm_hardenriched_xl"),
    ("UKFIN mixed q2", "ukfin", "pilot_ukfin_mixed_topup_q2"),
    ("UKFIN hard q2", "ukfin", "pilot_ukfin_hardenriched_topup_q2"),
]

for name, corpus, suffix in runs:
    path = Path(f"runs/curate_{corpus}/out_{suffix}/curate_judge/judge_stats.json")
    print(f"\n--- {name} ---")
    if not path.exists():
        print(f"MISSING: {path}")
        continue
    data = json.loads(path.read_text())
    total = data.get("total_items", 0) or 0
    passed = data.get("number_passed", 0) or 0
    dropped = data.get("drop_qp_count", 0) or 0
    pass_rate = (100.0 * passed / total) if total else 0.0
    print("total_items:", total)
    print("passed:", passed)
    print("dropped:", dropped)
    print(f"pass_rate: {pass_rate:.1f}%")
    print("reason_code_breakdown:", data.get("reason_code_breakdown", {}))
PY

echo ""
echo "DONE"
