# ObliQA-XRef Curation

Curation turns generated DPEL/SCHEMA candidates into explicit benchmark cohorts. The current policy is:

- IR is **diagnostic only**.
- `dependency-valid` = judge PASS.
- `answer-valid` = judge PASS ∩ answer PASS.
- Final benchmark defaults to `dependency-valid` (judge PASS).
- Dependency-valid but answer-failed items are exported separately for diagnosis.

## Pipeline

1. **IR retrieval and voting**
   - Runs BM25, E5, RRF, CE reranking, and optional XRefExpand baselines.
   - Computes source/target/both retrieval votes.
   - Assigns `ir_difficulty_label`: `easy`, `medium`, `hard`, `source_only`, `target_only`, `neither`.
   - Does not select or drop final items.

2. **Judge v2 citation-dependency filtering**
   - Checks source relevance, target relevance, source insufficiency, target necessity, answer support, and true citation dependency.
   - Borderline cases fail unless citation dependency is clear.
   - Writes PASS/DROP judge records.

3. **Answer validation (optional; diagnostic only)**
   - Runs on judge PASS items when enabled.
   - Writes diagnostic cohorts `final_answer_valid` and `final_answer_failed`.
   - Does not control membership in the default final benchmark.

4. **Final cohort assembly**
   - Writes explicit final cohort files and compatibility aliases.
   - Adds IR difficulty metadata and ObliQA-family metadata.

## Main Outputs

Under `runs/curate_<corpus>/out/`:

```text
curated_items.judge.jsonl
decisions.jsonl
stats.json
curate_judge/
  judge_queue.jsonl
  judge_responses_aggregated.jsonl
  judge_responses_pass.jsonl
  judge_responses_drop.jsonl
  judge_stats.json
curate_answer/
  answer_responses_aggregated.jsonl
  answer_responses_pass.jsonl
  answer_responses_drop.jsonl
  answer_stats.json
final_dependency_valid.jsonl
final_dependency_valid.csv
final_answer_valid.jsonl
final_answer_valid.csv
final_answer_failed.jsonl
final_answer_failed.csv
final_benchmark.jsonl
final_benchmark.csv
final_hard.jsonl
final_hard.csv
final_benchmark_stats.json
```

`final_benchmark.*` is a compatibility alias controlled by `curation.final_export_basis`; the default downstream basis is `dependency_valid` (judge PASS). `final_answer_valid`/`final_answer_failed` are optional diagnostic subsets when answer validation is run. `final_hard.*` is derived from diagnostic IR labels and should be interpreted cautiously (IR is not a selection signal).

## IR Difficulty Metadata

For each generated item, the curation stage records:

- `source_vote_count`
- `target_vote_count`
- `both_vote_count`
- `retrievers_recovering_source`
- `retrievers_recovering_target`
- `retrievers_recovering_both`
- `ir_difficulty_label`
- `difficulty_tier`: `retrievable` or `challenging`

Difficulty labels are retained for analysis and sampling. They do not determine final validity.

## Judge v2 Metadata

Final records preserve judge v2 metadata when available:

- `judge_schema_version`
- `source_alone_sufficient`
- `target_alone_sufficient`
- `target_adds_essential_information`
- `citation_dependent`
- `answer_supported_by_judge`

Passing requires source and target relevance, source-alone insufficiency, target necessity, supported answer, citation dependency, and a score above the configured threshold.

## Answer Validation Metadata

Final records include answer-validation fields when available:

- `answer_validation_passed`
- `answer_validation_score`
- `answer_validation_reasons`
- `unsupported_claims`
- `missing_source_tag`
- `missing_target_tag`
- `answer_responsive`
- `answer_supported`

## Commands

```bash
# Full curation from config
python -m obliqaxref curate --config configs/project.yaml

# Corpus-specific dev curation
python -m obliqaxref curate --config configs/adgm_dev.yaml

# Retrieval/debug subcommand
python -m obliqaxref.curate.cli ir -c configs/adgm_dev.yaml

# Skip answer validation (paper-default diagnostic workflow)
python -m obliqaxref curate --config configs/project.yaml --skip-answer
```

## Configuration Notes

Relevant keys in `configs/project.yaml`:

- `curation.final_export_basis`: `answer_valid` or `dependency_valid`
- `curation.ir_agreement.top_k`: depth for diagnostic IR voting
- `curation.ir_agreement.xref_expansion`: XRefExpand baseline parameters
- `curation.judge`: judge model, pass threshold, passes, and temperature
- `curation.answer`: answer-validation model and pass threshold

Legacy IR threshold fields may still exist for compatibility, but current final cohort selection is not KEEP/JUDGE/DROP-driven.

## Troubleshooting

- If final answer-valid is empty, inspect `curate_judge/judge_stats.json` and `curate_answer/answer_stats.json`.
- If hard cases disappear, check answer validation. Hard judge PASS + answer PASS items should remain in `final_answer_valid`.
- If `final_benchmark` differs from expected, check `curation.final_export_basis`.
- Use the explicit `final_*` cohort files as finalization inputs in the current policy.
 - If judge drops everything due to missing passage texts, verify `source_text`/`target_text` are present in `curated_items.judge.jsonl`. Curation attaches texts using IDs via `pid`/`passage_uid`/`passage_id`/`id` and text via `text`/`passage`/`content`. If `generator/passages_index.jsonl` is empty, the judge will fall back to the adapter corpus automatically.
