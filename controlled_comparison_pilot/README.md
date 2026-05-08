# Controlled Comparison Pilot

This directory contains the controlled retrieval, answer-generation, and RePASs evaluation workflow used for the ObliQA / ObliQA-MP / ObliQA-XRef-ADGM comparison.

The purpose is to evaluate all three datasets under a shared ADGM retrieval corpus and a controlled BM25 setup, then compare GPT-5.2 grounded answer quality under two prompt conditions:

- `general`
- `task_aware`

## Datasets

- **ObliQA**: obligation-focused regulatory QA.
- **ObliQA-MP**: multi-passage regulatory QA.
- **ObliQA-XRef-ADGM**: citation-dependent regulatory QA built from source-to-target cross-reference dependencies.

All controlled retrieval experiments use the same shared retrieval corpus:

- `controlled_comparison_pilot/full_ir/corpora/adgm_shared_passages.jsonl`
- Corpus size: `13,015` ADGM passages

## Controlled IR Setup

The canonical controlled retrieval setting is:

- `retrieval_setting = controlled_shared_corpus_canonical_bm25`
- `retrieval_corpus = shared_adgm_13015`
- BM25 backend: `obliqaxref.curate.ir.BM25Retriever`
- Tokenization: lowercase whitespace split
- Retrieval depth for answer generation: top-10

Full controlled IR outputs are under:

- `controlled_comparison_pilot/full_ir/`
- `controlled_comparison_pilot/full_ir/runs/`
- `controlled_comparison_pilot/full_ir/qrels/`
- `controlled_comparison_pilot/full_ir/reports/`

Main full-IR table outputs:

- `full_ir/reports/combined_controlled_ir_table.csv`
- `full_ir/reports/combined_controlled_ir_table.json`
- `full_ir/reports/combined_controlled_ir_table.md`
- `full_ir/reports/combined_controlled_ir_table_latex_draft.tex`

## Prompt Conditions

Prompt files:

- `prompts/general_grounded_prompt.txt`
- `prompts/task_aware_obliqa_prompt.txt`
- `prompts/task_aware_obliqa_mp_prompt.txt`
- `prompts/task_aware_xref_prompt.txt`

The general prompt is dataset-neutral. The task-aware prompts add only task-specific guidance:

- ObliQA: obligation-focused details such as actors, duties, conditions, exceptions, thresholds, and deadlines.
- ObliQA-MP: multi-passage synthesis guidance.
- ObliQA-XRef-ADGM: cross-reference-aware guidance without revealing gold source/target roles.

All answer prompts require grounded JSON output:

```json
{
  "answer": "...",
  "used_passage_ids": ["..."]
}
```

Answers include passage-level grounding tags in the answer text:

```text
[PID:PASSAGE_ID]
```

The `used_passage_ids` field records the passage IDs used by the generated answer.

Rendered prompt audit outputs are under:

- `reports/rendered_prompt_audit/`
- `reports/prompt_audit_report.json`
- `reports/prompt_audit_report.md`

## Answer Generation And RePASs

GPT answer files preserve grounded answers with `[PID:...]` tags. RePASs inputs use a clean `Answer` field with those tags stripped, while preserving the grounded answer as metadata:

- `generated_answer_with_grounding`
- `clean_answer`
- `used_passage_ids`

The RePASs evaluator is expected outside this repository:

- RePASs repo: `/Users/tuba.gokhan/Desktop/RegNLP_external/RePASs/`
- Python used: `/opt/anaconda3/bin/python`
- Obligation classifier checkpoint: `/Users/tuba.gokhan/Desktop/RegNLP_external/RePASs/models/obligation-classifier-legalbert`

RePASs is slow and may take many hours for the full sample-300 batch.

## Main Scripts

- `scripts/build_batch_answer_inputs.py`: samples test questions and attaches saved BM25 top-10 passages.
- `scripts/compute_sample_ir_metrics.py`: computes IR metrics for exactly the selected sample IDs.
- `scripts/run_gpt52_pilot_answers.py`: safe GPT-5.2 answer generation runner with dry-run, resume, custom roots, and progress logging.
- `scripts/export_repass_batch_inputs.py`: exports GPT answers to RePASs input JSON, stripping `[PID:...]` tags from `Answer`.
- `scripts/run_repass_batch.py`: runs six RePASs jobs sequentially and summarizes metrics.
- `scripts/run_batch_answer_repass.sh`: end-to-end overnight batch runner.
- `scripts/repair_failed_gpt_answers.py`: retries only failed GPT JSON outputs.
- `scripts/analyze_repass_prompt_deltas.py`: paired general-vs-task-aware RePASs delta analysis by `QuestionID`.
- `scripts/compute_combined_controlled_ir_table.py`: combined full controlled IR report.
- `scripts/audit_rendered_prompts.py`: local rendered prompt safety audit.

## Commands

Build sample inputs:

```bash
python controlled_comparison_pilot/scripts/build_batch_answer_inputs.py \
  --sample_size 300 \
  --seed 42 \
  --output_root controlled_comparison_pilot/batch_runs/sample_300
```

Run sample-level IR metrics:

```bash
/opt/anaconda3/bin/python controlled_comparison_pilot/scripts/compute_sample_ir_metrics.py \
  --input_root controlled_comparison_pilot/batch_runs/sample_300/answer_inputs \
  --output_root controlled_comparison_pilot/batch_runs/sample_300/reports
```

Run the full batch pipeline:

```bash
SAMPLE_SIZE=300 SEED=42 RESUME=1 OVERWRITE=0 DRY_RUN_ONLY=0 \
  controlled_comparison_pilot/scripts/run_batch_answer_repass.sh
```

Resume an interrupted batch:

```bash
SAMPLE_SIZE=300 SEED=42 RESUME=1 OVERWRITE=0 DRY_RUN_ONLY=0 \
  controlled_comparison_pilot/scripts/run_batch_answer_repass.sh
```

Tail logs:

```bash
tail -f controlled_comparison_pilot/batch_runs/sample_300/logs/04_gpt_run.log
tail -f controlled_comparison_pilot/batch_runs/sample_300/logs/06_repass_run.log
```

Repair failed GPT JSON outputs:

```bash
python -u controlled_comparison_pilot/scripts/repair_failed_gpt_answers.py \
  --run \
  --max_retries_per_record 2
```

Analyze paired prompt deltas:

```bash
python controlled_comparison_pilot/scripts/analyze_repass_prompt_deltas.py
```

## Completed `sample_300` Run

The completed batch is stored under:

- `controlled_comparison_pilot/batch_runs/sample_300/`

Subfolders:

- `answer_inputs/`
- `answers/`
- `repass_inputs/`
- `repass_outputs/`
- `reports/`
- `logs/`

Run summary:

- Sample size: `300` per dataset.
- Prompt conditions: `general` and `task_aware`.
- Total GPT answers: `1,800`.
- GPT repair: `78` JSON parse / failed records repaired successfully.
- Final bad GPT records: `0`.
- RePASs rows: `1,800`.
- RePASs skipped/failed rows: `0`.

Important reports:

- `batch_runs/sample_300/reports/sample_selection_report.json`
- `batch_runs/sample_300/reports/sample_selection_report.md`
- `batch_runs/sample_300/reports/sample_ir_metrics.csv`
- `batch_runs/sample_300/reports/sample_ir_metrics.json`
- `batch_runs/sample_300/reports/sample_ir_metrics.md`
- `batch_runs/sample_300/reports/gpt52_batch_run_report.json`
- `batch_runs/sample_300/reports/gpt52_repair_report.json`
- `batch_runs/sample_300/reports/gpt52_repair_report.md`
- `batch_runs/sample_300/reports/repass_batch_summary.csv`
- `batch_runs/sample_300/reports/repass_batch_summary.json`
- `batch_runs/sample_300/reports/repass_batch_summary.md`
- `batch_runs/sample_300/reports/repass_prompt_delta_summary.csv`
- `batch_runs/sample_300/reports/repass_prompt_delta_summary.json`
- `batch_runs/sample_300/reports/repass_prompt_delta_summary.md`
- `batch_runs/sample_300/reports/repass_prompt_delta_per_item.csv`

The sample-level IR metrics are in `sample_ir_metrics.csv`. The RePASs aggregate summary is in `repass_batch_summary.csv`. The paired general-vs-task-aware analysis is in `repass_prompt_delta_summary.csv`, with per-item deltas in `repass_prompt_delta_per_item.csv`.

The paired prompt comparison shows mixed effects. Task-aware prompts improve obligation coverage on all three datasets, but mean composite effects vary by dataset: task-aware is slightly higher for ObliQA-MP, while general is higher for ObliQA and ObliQA-XRef-ADGM in the sample-300 RePASs results.

## Reproducibility Notes

- GPT/API calls are expensive and should not be rerun casually.
- Always run with `DRY_RUN_ONLY=1` before real batch runs.
- Use `RESUME=1` to avoid repeating completed calls.
- Do not use `OVERWRITE=1` unless intentionally regenerating outputs.
- RePASs is slow and may take many hours.
- The RePASs repository and model checkpoint are external dependencies and are not stored in this repo.
- Do not commit API keys, `.env` files, local credentials, or environment-specific secrets.
- Repair backup files named like `*.bak_YYYYMMDD_HHMMSS` are useful locally but are not required for reproducing the final reported results.

