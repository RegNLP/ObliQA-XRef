# `obliqaxref.eval`

Evaluation and analysis utilities for ObliQA-XRef.

Primary entrypoint:

```bash
python -m obliqaxref.eval.cli --help
```

## Commands

```bash
# Finalize answer-valid benchmark splits
python -m obliqaxref.eval.cli finalize --corpus both --cohort answer_valid

# Retrieval evaluation with pair-level diagnostics
python -m obliqaxref.eval.cli ir --corpus both --k 10

# Answer generation and evaluation
python -m obliqaxref.eval.cli answer --corpus both --methods bm25 e5 rrf ce_rerank_union200
python -m obliqaxref.eval.cli answer-eval --corpus both

# Analysis utilities
python -m obliqaxref.eval.cli answer-quality-by-retrieval --root ObliQA-XRef_Out_Datasets
python -m obliqaxref.eval.cli human-audit-export --out ObliQA-XRef_Out_Datasets --n 200 --seed 13
python -m obliqaxref.eval.cli human-audit-aggregate --inputs ann1.csv ann2.csv --out ObliQA-XRef_Out_Datasets
python -m obliqaxref.eval.cli benchmark-statistics --out ObliQA-XRef_Out_Datasets
```

## Retrieval Metrics

The IR evaluator preserves standard metrics and adds citation-pair diagnostics:

- Recall@5/10/20, MAP@5/10/20, nDCG@5/10/20
- Both@5/10/20
- SRC-only@5/10/20
- TGT-only@5/10/20
- Neither@5/10/20
- PairMRR

Outputs:

- `retrieval_metrics_full.csv`
- `retrieval_diagnostics_per_query.csv`

## Analysis Outputs

- `answer_quality_by_retrieval_outcome*.csv/md`
- `human_audit_sample.csv`
- `human_audit_sampling_report.json`
- `human_audit_instructions.md`
- `human_audit_aggregated_items.csv`
- `human_audit_summary.csv/md`
- `human_audit_agreement.json`
- `benchmark_statistics.csv`
- `benchmark_statistics_by_difficulty.csv`
- `benchmark_statistics_summary.md`
- `benchmark_statistics_latex_tables.tex`

Finalized records include ObliQA-family metadata fields. Older records are upgraded with defaults during export or statistics loading.
