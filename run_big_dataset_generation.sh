#!/usr/bin/env bash
set -euo pipefail

MIXED_SUFFIX="pilot_mixed_big2000"
MIXED_PILOT_N=2000
MIXED_MAX_PAIRS=800

HARD_SUFFIX="pilot_hardenriched_big1000"
HARD_PILOT_N=1000
HARD_MAX_PAIRS=400

MAX_Q_PER_PAIR=1
TEMP="0.0"

echo "============================================================"
echo "ObliQA-XRef BIG dataset generation batch"
echo "Started: $(date)"
echo "Mixed suffix: ${MIXED_SUFFIX}"
echo "Hard suffix:  ${HARD_SUFFIX}"
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
    printf "%-85s %s\n" "$label" "$(wc -l < "$path" | tr -d ' ')"
  else
    printf "%-85s %s\n" "$label" "MISSING"
  fi
}

run_generate_curate() {
  local corpus="$1"
  local config="$2"
  local suffix="$3"
  local pilot_n="$4"
  local max_pairs="$5"
  local sampling_mode="$6"

  echo ""
  echo "============================================================"
  echo "GENERATE: corpus=${corpus}, suffix=${suffix}, sampling=${sampling_mode}"
  echo "pilot_n=${pilot_n}, max_pairs=${max_pairs}, max_q_per_pair=${MAX_Q_PER_PAIR}"
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
    --max-q-per-pair "${MAX_Q_PER_PAIR}" \
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

# ------------------------------------------------------------------
# 1) Mixed difficulty: coverage / natural distribution / dataset size
# ------------------------------------------------------------------

run_generate_curate "adgm"  "configs/adgm_dev.yaml"  "${MIXED_SUFFIX}" "${MIXED_PILOT_N}" "${MIXED_MAX_PAIRS}" "mixed_difficulty"
run_generate_curate "ukfin" "configs/ukfin_dev.yaml" "${MIXED_SUFFIX}" "${MIXED_PILOT_N}" "${MIXED_MAX_PAIRS}" "mixed_difficulty"

# ------------------------------------------------------------------
# 2) Hard-enriched: challenge-oriented subset
# ------------------------------------------------------------------

run_generate_curate "adgm"  "configs/adgm_dev.yaml"  "${HARD_SUFFIX}" "${HARD_PILOT_N}" "${HARD_MAX_PAIRS}" "hard_enriched"
run_generate_curate "ukfin" "configs/ukfin_dev.yaml" "${HARD_SUFFIX}" "${HARD_PILOT_N}" "${HARD_MAX_PAIRS}" "hard_enriched"

echo ""
echo "============================================================"
echo "BIG RUN SUMMARY COUNTS"
echo "Finished all runs: $(date)"
echo "============================================================"

echo ""
echo "--- MIXED BIG2000: ADGM ---"
count_file "ADGM mixed DPEL QAs" "runs/generate_adgm/out_${MIXED_SUFFIX}/dpel/dpel.qa.jsonl"
count_file "ADGM mixed SCHEMA QAs" "runs/generate_adgm/out_${MIXED_SUFFIX}/schema/schema.qa.jsonl"
count_file "ADGM mixed judge PASS" "runs/curate_adgm/out_${MIXED_SUFFIX}/curate_judge/judge_responses_pass.jsonl"
count_file "ADGM mixed judge DROP" "runs/curate_adgm/out_${MIXED_SUFFIX}/curate_judge/judge_responses_drop.jsonl"
count_file "ADGM mixed final_dependency_valid" "runs/curate_adgm/out_${MIXED_SUFFIX}/final_dependency_valid.jsonl"
count_file "ADGM mixed final_benchmark" "runs/curate_adgm/out_${MIXED_SUFFIX}/final_benchmark.jsonl"
count_file "ADGM mixed final_hard" "runs/curate_adgm/out_${MIXED_SUFFIX}/final_hard.jsonl"

echo ""
echo "--- MIXED BIG2000: UKFIN ---"
count_file "UKFIN mixed DPEL QAs" "runs/generate_ukfin/out_${MIXED_SUFFIX}/dpel/dpel.qa.jsonl"
count_file "UKFIN mixed SCHEMA QAs" "runs/generate_ukfin/out_${MIXED_SUFFIX}/schema/schema.qa.jsonl"
count_file "UKFIN mixed judge PASS" "runs/curate_ukfin/out_${MIXED_SUFFIX}/curate_judge/judge_responses_pass.jsonl"
count_file "UKFIN mixed judge DROP" "runs/curate_ukfin/out_${MIXED_SUFFIX}/curate_judge/judge_responses_drop.jsonl"
count_file "UKFIN mixed final_dependency_valid" "runs/curate_ukfin/out_${MIXED_SUFFIX}/final_dependency_valid.jsonl"
count_file "UKFIN mixed final_benchmark" "runs/curate_ukfin/out_${MIXED_SUFFIX}/final_benchmark.jsonl"
count_file "UKFIN mixed final_hard" "runs/curate_ukfin/out_${MIXED_SUFFIX}/final_hard.jsonl"

echo ""
echo "--- HARD-ENRICHED BIG1000: ADGM ---"
count_file "ADGM hard DPEL QAs" "runs/generate_adgm/out_${HARD_SUFFIX}/dpel/dpel.qa.jsonl"
count_file "ADGM hard SCHEMA QAs" "runs/generate_adgm/out_${HARD_SUFFIX}/schema/schema.qa.jsonl"
count_file "ADGM hard judge PASS" "runs/curate_adgm/out_${HARD_SUFFIX}/curate_judge/judge_responses_pass.jsonl"
count_file "ADGM hard judge DROP" "runs/curate_adgm/out_${HARD_SUFFIX}/curate_judge/judge_responses_drop.jsonl"
count_file "ADGM hard final_dependency_valid" "runs/curate_adgm/out_${HARD_SUFFIX}/final_dependency_valid.jsonl"
count_file "ADGM hard final_benchmark" "runs/curate_adgm/out_${HARD_SUFFIX}/final_benchmark.jsonl"
count_file "ADGM hard final_hard" "runs/curate_adgm/out_${HARD_SUFFIX}/final_hard.jsonl"

echo ""
echo "--- HARD-ENRICHED BIG1000: UKFIN ---"
count_file "UKFIN hard DPEL QAs" "runs/generate_ukfin/out_${HARD_SUFFIX}/dpel/dpel.qa.jsonl"
count_file "UKFIN hard SCHEMA QAs" "runs/generate_ukfin/out_${HARD_SUFFIX}/schema/schema.qa.jsonl"
count_file "UKFIN hard judge PASS" "runs/curate_ukfin/out_${HARD_SUFFIX}/curate_judge/judge_responses_pass.jsonl"
count_file "UKFIN hard judge DROP" "runs/curate_ukfin/out_${HARD_SUFFIX}/curate_judge/judge_responses_drop.jsonl"
count_file "UKFIN hard final_dependency_valid" "runs/curate_ukfin/out_${HARD_SUFFIX}/final_dependency_valid.jsonl"
count_file "UKFIN hard final_benchmark" "runs/curate_ukfin/out_${HARD_SUFFIX}/final_benchmark.jsonl"
count_file "UKFIN hard final_hard" "runs/curate_ukfin/out_${HARD_SUFFIX}/final_hard.jsonl"

echo ""
echo "============================================================"
echo "Difficulty summaries"
echo "============================================================"

python - <<'PY'
import json
from pathlib import Path

runs = [
    ("ADGM mixed_big2000", Path("runs/curate_adgm/out_pilot_mixed_big2000/stats.json")),
    ("UKFIN mixed_big2000", Path("runs/curate_ukfin/out_pilot_mixed_big2000/stats.json")),
    ("ADGM hardenriched_big1000", Path("runs/curate_adgm/out_pilot_hardenriched_big1000/stats.json")),
    ("UKFIN hardenriched_big1000", Path("runs/curate_ukfin/out_pilot_hardenriched_big1000/stats.json")),
]

for name, path in runs:
    print(f"\n--- {name} ---")
    if not path.exists():
        print(f"MISSING: {path}")
        continue
    data = json.loads(path.read_text())
    total = data.get("total_items", 0) or 0
    final_total = data.get("final_benchmark_count", data.get("final_dependency_valid_count", "NA"))
    counts = data.get("ir_difficulty_label_counts", {})
    print("total_items:", total)
    print("final_benchmark_count:", final_total)
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
    ("ADGM mixed_big2000", Path("runs/curate_adgm/out_pilot_mixed_big2000/curate_judge/judge_stats.json")),
    ("UKFIN mixed_big2000", Path("runs/curate_ukfin/out_pilot_mixed_big2000/curate_judge/judge_stats.json")),
    ("ADGM hardenriched_big1000", Path("runs/curate_adgm/out_pilot_hardenriched_big1000/curate_judge/judge_stats.json")),
    ("UKFIN hardenriched_big1000", Path("runs/curate_ukfin/out_pilot_hardenriched_big1000/curate_judge/judge_stats.json")),
]

for name, path in runs:
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
