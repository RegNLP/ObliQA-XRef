#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PILOT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PILOT_ROOT}/.." && pwd)"

SAMPLE_SIZE="${SAMPLE_SIZE:-300}"
SEED="${SEED:-42}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"
RESUME="${RESUME:-1}"
OVERWRITE="${OVERWRITE:-0}"
REPASS_REPO="${REPASS_REPO:-/Users/tuba.gokhan/Desktop/RegNLP_external/RePASs}"
REPASS_PYTHON="${REPASS_PYTHON:-/opt/anaconda3/bin/python}"
IR_PYTHON="${IR_PYTHON:-/opt/anaconda3/bin/python}"

RUN_ROOT="${PILOT_ROOT}/batch_runs/sample_${SAMPLE_SIZE}"
LOG_DIR="${RUN_ROOT}/logs"
EXPECTED_GPT_CALLS=$((SAMPLE_SIZE * 3 * 2))

if [[ "${OSTYPE:-}" == darwin* && "${DRY_RUN_ONLY}" != "1" && -z "${UNDER_CAFFEINATE:-}" ]]; then
  if command -v caffeinate >/dev/null 2>&1; then
    export UNDER_CAFFEINATE=1
    exec caffeinate -s "$0" "$@"
  else
    echo "WARNING: caffeinate not found; prevent sleep manually for the overnight run." >&2
  fi
fi

mkdir -p \
  "${RUN_ROOT}/answer_inputs" \
  "${RUN_ROOT}/answers" \
  "${RUN_ROOT}/repass_inputs" \
  "${RUN_ROOT}/repass_outputs" \
  "${RUN_ROOT}/reports" \
  "${LOG_DIR}"

run_stage() {
  local label="$1"
  local log_path="$2"
  shift 2
  echo
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${label}"
  echo "Log: ${log_path}"
  "$@" 2>&1 | tee "${log_path}"
}

GPT_FLAGS=(--prompt_condition both --input_root "${RUN_ROOT}/answer_inputs" --output_root "${RUN_ROOT}/answers" --report_path "${RUN_ROOT}/reports/gpt52_batch_run_report.json" --log_progress_every 10)
REPASS_FLAGS=(--repass_repo "${REPASS_REPO}" --input_root "${RUN_ROOT}/repass_inputs" --output_root "${RUN_ROOT}/repass_outputs" --summary_csv "${RUN_ROOT}/reports/repass_batch_summary.csv" --summary_json "${RUN_ROOT}/reports/repass_batch_summary.json" --summary_md "${RUN_ROOT}/reports/repass_batch_summary.md" --python_bin "${REPASS_PYTHON}" --sample_size "${SAMPLE_SIZE}")

if [[ "${RESUME}" == "1" ]]; then
  GPT_FLAGS+=(--resume)
  REPASS_FLAGS+=(--resume)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  GPT_FLAGS+=(--overwrite)
  REPASS_FLAGS+=(--overwrite)
fi

{
  echo "Start time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Sample size: ${SAMPLE_SIZE}"
  echo "Seed: ${SEED}"
  echo "Expected GPT calls: ${EXPECTED_GPT_CALLS}"
  echo "Run root: ${RUN_ROOT}"
  echo "DRY_RUN_ONLY: ${DRY_RUN_ONLY}"
  echo "RESUME: ${RESUME}"
  echo "OVERWRITE: ${OVERWRITE}"
  echo "Rough runtime estimate: GPT generation depends on Azure latency; RePASs may take many hours for ${EXPECTED_GPT_CALLS} answer rows."
} | tee "${LOG_DIR}/00_start.log"

run_stage "1. Build answer inputs" "${LOG_DIR}/01_build_inputs.log" \
  python3 "${SCRIPT_DIR}/build_batch_answer_inputs.py" \
    --sample_size "${SAMPLE_SIZE}" \
    --seed "${SEED}" \
    --output_root "${RUN_ROOT}"

run_stage "2. Compute sample-level IR metrics" "${LOG_DIR}/02_sample_ir_metrics.log" \
  "${IR_PYTHON}" "${SCRIPT_DIR}/compute_sample_ir_metrics.py" \
    --input_root "${RUN_ROOT}/answer_inputs" \
    --output_root "${RUN_ROOT}/reports"

run_stage "3. GPT dry-run validation" "${LOG_DIR}/03_gpt_dry_run.log" \
  python3 "${SCRIPT_DIR}/run_gpt52_pilot_answers.py" \
    "${GPT_FLAGS[@]}" \
    --dry_run

if [[ "${DRY_RUN_ONLY}" == "1" ]]; then
  {
    echo "DRY_RUN_ONLY=1: stopping before GPT real run, RePASs export, and RePASs execution."
    echo "Final dry-run outputs:"
    echo "- ${RUN_ROOT}/answer_inputs"
    echo "- ${RUN_ROOT}/reports/sample_selection_report.json"
    echo "- ${RUN_ROOT}/reports/sample_ir_metrics.csv"
    echo "- ${RUN_ROOT}/reports/gpt52_batch_run_report.json"
  } | tee "${LOG_DIR}/07_summary.log"
  exit 0
fi

run_stage "4. GPT real run" "${LOG_DIR}/04_gpt_run.log" \
  python3 "${SCRIPT_DIR}/run_gpt52_pilot_answers.py" \
    "${GPT_FLAGS[@]}" \
    --run

run_stage "5. Export RePASs inputs" "${LOG_DIR}/05_export_repass_inputs.log" \
  python3 "${SCRIPT_DIR}/export_repass_batch_inputs.py" \
    --answers_root "${RUN_ROOT}/answers" \
    --output_root "${RUN_ROOT}/repass_inputs" \
    --expected_count "${SAMPLE_SIZE}"

run_stage "6. Run RePASs" "${LOG_DIR}/06_repass_run.log" \
  python3 "${SCRIPT_DIR}/run_repass_batch.py" \
    "${REPASS_FLAGS[@]}"

{
  echo "Completed time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Final output paths:"
  echo "- Answers: ${RUN_ROOT}/answers"
  echo "- RePASs inputs: ${RUN_ROOT}/repass_inputs"
  echo "- RePASs outputs: ${RUN_ROOT}/repass_outputs"
  echo "- Reports: ${RUN_ROOT}/reports"
  echo "- Logs: ${LOG_DIR}"
  echo
  echo "RePASs summary:"
  cat "${RUN_ROOT}/reports/repass_batch_summary.md"
} | tee "${LOG_DIR}/07_summary.log"
