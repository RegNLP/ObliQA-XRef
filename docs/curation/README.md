# ObliQA-XRef Curation (Strict Citation-Dependent)

Authoritative guide for the curation stage: inputs, steps, outputs, config, guardrails, and strict citation-dependency rules. Curation filters generator outputs with the LLM citation-dependency judge and answer validation. Any item whose source passage alone answers the question must DROP with `QP_NOT_CIT_DEP`.

## Overview

## Pipeline Overview

1) **IR retrieval (optional from curation CLI)** — Build TREC runs per retriever (BM25, dense, RRF, cross-encoder rerank).
2) **IR annotation** — Compute per-item vote counts across all retrievers. Assign a diagnostic `ir_difficulty_label` to every item. **IR outcome does NOT filter items** — all items proceed to the citation-dependency judge regardless of retriever agreement.
3) **Citation-dependency judge (all items)** — Multi-pass Azure/OpenAI LLM enforces strict citation-dependency. PASS requires `source_alone_insufficient=true`; ties DROP.
4) **Answer validation (optional)** — Secondary filter over citation-dependency PASS items; can be skipped.
5) **Final benchmark assembly** — Intersects judge PASS and answer PASS sets; enriches with difficulty metadata; writes `final_benchmark.jsonl/csv` and `final_hard.jsonl/csv`.

> **Key design principle (Tasks 5 & 6):** IR agreement is a *diagnostic label only*. Hard-to-retrieve items that pass citation-dependency and answer validation are **retained** in the final benchmark with `difficulty_tier = "challenging"` rather than being dropped.

## Directory Structure & Inputs

```
runs/curate_<corpus>_<tag>/
  inputs/
    generator/
      items.jsonl              # Generated items to curate (item_id, source_passage_id, target_passage_id, question, ...)
      passage_corpus.jsonl     # Canonical passages (passage_id, text, metadata)
    ir_runs/                   # IR membership runs (top-k passage IDs per item)
      runlist.json             # {k, runs, metadata}
      <run_name>.jsonl         # one JSON per item_id: {item_id, topk_passage_ids: [...]} OR
  output_dir/ (cfg.paths.output_dir)  # IR outputs, curated outputs, judge outputs (may coincide with curate_output_dir)
```

Notes:
- If `inputs/generator/items.jsonl` is missing, curation will merge method-specific files (DPEL + SCHEMA) using `merge_qa_items` into `output_dir/generator/items.jsonl`.
- IR retrieval writes TREC files into `output_dir`. Voting consumes those TREC files.

## Outputs

```
<curate_output_dir>/
  curated_items.judge.jsonl        # All items with IR annotation (all proceed to judge)
  decisions.jsonl                  # IR annotation per item (difficulty label + vote counts)
  stats.json                       # IR difficulty label counts, run metadata
  curate_judge/
    judge_queue.jsonl              # All items sent to citation-dependency judge
    judge_responses_aggregated.jsonl
    judge_responses_pass.jsonl     # Citation-dependency PASS (PASS_QP)
    judge_responses_drop.jsonl     # Citation-dependency DROP (DROP_QP)
    judge_stats.json
  curate_answer/                   # (If answer validation runs)
    answer_responses_aggregated.jsonl
    answer_responses_pass.jsonl    # Answer validation PASS (PASS_ANS)
    answer_responses_drop.jsonl
    answer_stats.json
  final_benchmark.jsonl            # All validated items (judge PASS ∩ answer PASS)
  final_benchmark.csv              # Flat CSV of key fields for all validated items
  final_hard.jsonl                 # Validated items with difficulty_tier = "challenging"
  final_hard.csv                   # Flat CSV of challenging items
  final_benchmark_stats.json       # Counts by difficulty_tier and ir_difficulty_label
```

`decisions.jsonl` audit schema per item: `item_id`, `source_passage_id`, `target_passage_id`, `ir_difficulty_label`, `source_vote_count`, `target_vote_count`, `both_vote_count`.

## IR Annotation and Difficulty Labels

IR vote counts are computed for every item and stored as metadata. They do **not** gate progression through the pipeline.

### Vote computation

For each retriever run and each item, the pipeline checks whether the source passage ID and target passage ID appear in the top-k results:

- `source_vote_count` — number of retrievers that recovered the source passage
- `target_vote_count` — number of retrievers that recovered the target passage
- `both_vote_count` — number of retrievers that recovered **both**

### `ir_difficulty_label` (6 values)

| Label | Condition |
|---|---|
| `easy` | `both_vote_count > num_retrievers / 2` (majority co-retrieved both) |
| `medium` | `both_vote_count ≥ 1` (at least one retriever co-retrieved both) |
| `hard` | `both_vote_count == 0`; source retrieved by ≥1 retriever, target by ≥1 different retriever |
| `source_only` | `both == 0`; only source retrieved |
| `target_only` | `both == 0`; only target retrieved |
| `neither` | No retriever recovered either passage |

### `difficulty_tier` (2 values, final benchmark)

The 6-way label is collapsed to a 2-way tier in the final benchmark records:

| Tier | IR labels |
|---|---|
| `retrievable` | `easy`, `medium` |
| `challenging` | `hard`, `source_only`, `target_only`, `neither` |

Items from **all** tiers are included in the final benchmark. `challenging` items are additionally exported as `final_hard.jsonl/csv`.

### Per-item IR fields (in `curated_items.judge.jsonl` and `final_benchmark.jsonl`)

```json
{
  "ir_difficulty_label": "hard",
  "difficulty_tier": "challenging",
  "source_vote_count": 1,
  "target_vote_count": 1,
  "both_vote_count": 0,
  "retrievers_recovering_source": ["bm25"],
  "retrievers_recovering_target": ["e5"],
  "retrievers_recovering_both": []
}
```

## Judge (Strict Citation-Dependency)

- All items (regardless of IR outcome) are sent to the citation-dependency judge.
- Backend: Azure OpenAI (recommended) or OpenAI, configured via YAML and environment.
- Queue (`curate_judge/judge_queue.jsonl`): question, source/target IDs, texts, IR metadata.
- Response: `decision_qp` ∈ {`PASS_QP`, `DROP_QP`}, `confidence`, `source_alone_insufficient`, `reason_code_qp`, optional rationale/meta.
- Aggregation: multi-pass, confidence-weighted; ties → DROP. PASS requires `source_alone_insufficient=true`; otherwise coerced to DROP with `QP_NOT_CIT_DEP`.
- Outputs: `judge_responses_aggregated.jsonl`, `judge_responses_pass.jsonl`, `judge_responses_drop.jsonl`, `judge_stats.json`.
- Environment (Azure example): `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, deployment name set in config.

### Judge reason codes (strict gate)

- `QP_NOT_CIT_DEP` — Source alone answers; fails strict citation dependency (default when `source_alone_insufficient=false`).
- `QP_WRONG_TARGET` — Target passage is off-topic or not needed to answer.
- `QP_SCOPE_MISMATCH` — Question scope/actor/condition conflicts with passages.
- `QP_UNDER_SPEC` — Question is ambiguous or missing key constraints.
- `QP_TOO_BROAD` — Question is overly general or multi-part.
- `QP_ILL_FORMED` — Judge parse/error or incoherent question; used as conservative fallback on LLM errors.

## Answer Validation (Optional)

- Runs after judge on citation-dependency PASS items (all judge PASS, not just a KEEP tier).
- Writes under `<curate_output_dir>/curate_answer/`.
- Failures do not stop pipeline; failures are logged and items remain unless filtered by the answer module.

## Final Benchmark Assembly (Phase 5)

After answer validation, the pipeline assembles the final benchmark by intersecting judge PASS and answer PASS item sets:

```
final_benchmark = { items where decision_qp_final = PASS_QP } ∩ { items where decision_ans_final = PASS_ANS }
```

Each final item carries:
- All original generator fields (`question`, `gold_answer`, `source_passage_id`, `target_passage_id`, `method`, `pair_uid`, `persona`)
- `ir_difficulty_label` — 6-way diagnostic label
- `difficulty_tier` — `"retrievable"` or `"challenging"`
- Vote counts and retriever lists

**Outputs**:

| File | Contents |
|---|---|
| `final_benchmark.jsonl` | All validated items (both tiers) |
| `final_benchmark.csv` | Key fields as flat CSV |
| `final_hard.jsonl` | Only `difficulty_tier = "challenging"` items |
| `final_hard.csv` | Flat CSV of challenging items |
| `final_benchmark_stats.json` | `total_final`, `total_hard`, `difficulty_tier_counts`, `ir_difficulty_label_counts` |

## How to Run

### Typical (IR → vote → judge → answer)

```bash
python -m obliqaxref.curate.cli ir \
  --config configs/project.yaml

python -m obliqaxref.curate.cli vote \
  --config configs/project.yaml
```

See [configs/project.yaml](../../configs/project.yaml) for all available configuration options.

### Voting only (IR already present)

```bash
python -m obliqaxref.curate.cli vote --config configs/project.yaml
```

### Full fleet helper (all corpora/methods)

```bash
python src/obliqaxref/curate/run_full_pipeline.py
```

### Useful overrides (CLI via CurateOverrides)

- `--skip-ir` — assume IR runs already exist in `output_dir`.
- `--skip-judge` — stop after voting (no LLM cost).
- `--skip-answer` — skip answer validation.
- `--keep-th` / `--judge-th` / `--ir-top-k` — override thresholds or k for quick experiments.

## Configuration Map (YAML Keys)

- `paths.input_dir` — where generator artifacts and passages live.
- `paths.output_dir` — where IR runs and curated outputs are written; also used for merge fallback.
- `paths.curate_output_dir` — optional override for curated outputs; defaults to `paths.output_dir`.
- `curation.ir_agreement.top_k` — retrieval depth; must match IR run generation.
- `curation.ir_agreement.keep_threshold` / `judge_threshold` — pair-level vote thresholds.
- `judge.enabled` — toggle LLM judge for borderline items.
- `judge.num_passes`, `judge.temperature`, `judge.llm_backend`, `judge.azure.deployment`, `judge.openai.model` — LLM settings.
- `answer.enabled` or CLI `skip_answer` — control answer validation phase.

## Restrictions and Guardrails

- **Strict citation-dependency:** Any item where the source alone answers must DROP with `QP_NOT_CIT_DEP`. PASS requires `source_alone_insufficient=true` from the judge response.
- **Conservative defaults:** Judge aggregation ties DROP; low-consensus flagged in `judge_stats.json`.
- **IR is diagnostic only:** IR agreement (or lack of it) does not filter items. Challenging items pass if they satisfy citation-dependency and answer validity.
- **Azure-only in code path:** Judge orchestration prefers Azure OpenAI; ensure credentials are set. Non-Azure may require config changes.
- **No IR threshold gating:** `keep_threshold` / `judge_threshold` YAML keys are preserved for backward compatibility but have no effect on item selection in the current pipeline.

## Tips and Troubleshooting

- If `items.jsonl` is missing, let the pipeline merge DPEL + SCHEMA outputs automatically, or pre-run merge yourself.
- All items go to the judge — LLM cost scales with the total item count, not a JUDGE tier.
- When experimenting, set `--skip-answer` to shorten runtime; re-enable for final benchmarks.
- Inspect `stats.json` for IR difficulty label distribution and `judge_stats.json` for reason-code breakdown (especially `QP_NOT_CIT_DEP`).
- Errors during judge passes fall back to DROP with reason `QP_ILL_FORMED`; see logs for details.
- `final_benchmark_stats.json` reports `difficulty_tier_counts` — use this to understand how many challenging items made it into the final benchmark.
