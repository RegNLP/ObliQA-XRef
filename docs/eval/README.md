# ObliQA-XRef Evaluation And Analysis

The evaluation module finalizes curated cohorts, stages retrieval runs, evaluates retrieval/answer quality, and writes analysis exports.

Unified CLI:

```bash
python -m obliqaxref.eval.cli --help
```

## Finalization

The default experimental cohort is `answer_valid`.

```bash
python -m obliqaxref.eval.cli finalize --corpus both --cohort answer_valid
```

This reads `runs/curate_<corpus>/out/final_answer_valid.jsonl` and writes:

```text
ObliQA-XRef_Out_Datasets/
  ObliQA-XRef-ADGM-ALL.jsonl
  ObliQA-XRef-ADGM-ALL-train.jsonl
  ObliQA-XRef-ADGM-ALL-dev.jsonl
  ObliQA-XRef-ADGM-ALL-test.jsonl
  ObliQA-XRef-ADGM-ALL/
    test.jsonl
    bm25.trec
    e5.trec
    rrf.trec
    bm25_xref_expand.trec
    e5_xref_expand.trec
    rrf_xref_expand.trec
    ce_rerank_union200.trec
```

Older cohort aliases are accepted, but current policy distinguishes `dependency_valid`, `answer_valid`, and `answer_failed`.

## Retrieval Evaluation

Supported runs:

- BM25
- E5
- RRF
- CE reranker over union candidates
- BM25+XRefExpand
- E5+XRefExpand
- RRF+XRefExpand, if available

The CE reranker scores `(question text, passage text)` pairs, not `(query_id, passage text)`.

```bash
python -m obliqaxref.eval.cli ir --corpus both --k 10
```

Outputs:

- `retrieval_metrics_full.csv`
- `retrieval_diagnostics_per_query.csv`

Metrics:

- Recall@5/10/20
- MAP@5/10/20
- nDCG@5/10/20
- Both@5/10/20
- SRC-only@5/10/20
- TGT-only@5/10/20
- Neither@5/10/20
- PairMRR

## Answer Generation And Evaluation

```bash
python -m obliqaxref.eval.cli answer \
  --corpus both \
  --methods bm25 e5 rrf ce_rerank_union200

python -m obliqaxref.eval.cli answer-eval --corpus both
```

Answer generation uses retrieved passages only, without revealing SOURCE/TARGET roles. Answer evaluation writes per-item JSON plus compact summaries:

- `answer_gen_{corpus}_{subset}_{method}_test.json`
- `answer_eval_{corpus}_{subset}_{method}_test.json`
- `answer_eval_{corpus}_compact.csv`

## Analysis Utilities

```bash
# Join retrieval outcome with answer quality
python -m obliqaxref.eval.cli answer-quality-by-retrieval \
  --root ObliQA-XRef_Out_Datasets

# Export human audit sample
python -m obliqaxref.eval.cli human-audit-export \
  --input runs/curate_adgm/out/final_answer_valid.jsonl runs/curate_ukfin/out/final_answer_valid.jsonl \
  --out ObliQA-XRef_Out_Datasets \
  --n 200 --seed 13

# Aggregate completed annotations
python -m obliqaxref.eval.cli human-audit-aggregate \
  --inputs ann1.csv ann2.csv \
  --out ObliQA-XRef_Out_Datasets

# Benchmark statistics and paper-ready tables
python -m obliqaxref.eval.cli benchmark-statistics \
  --out ObliQA-XRef_Out_Datasets
```

Analysis outputs:

- `answer_quality_by_retrieval_outcome.csv`
- `answer_quality_by_retrieval_outcome_summary.csv`
- `answer_quality_by_retrieval_outcome_summary.md`
- `human_audit_sample.csv`
- `human_audit_sampling_report.json`
- `human_audit_instructions.md`
- `human_audit_aggregated_items.csv`
- `human_audit_summary.csv`
- `human_audit_summary.md`
- `human_audit_agreement.json`
- `benchmark_statistics.csv`
- `benchmark_statistics_by_difficulty.csv`
- `benchmark_statistics_summary.md`
- `benchmark_statistics_latex_tables.tex`

## Resource Statistics

Legacy resource-stat utilities remain available for corpus and cross-reference summaries:

```bash
python src/obliqaxref/eval/ResourceStats/cli.py compute --corpus adgm
python src/obliqaxref/eval/ResourceStats/cli.py compute --corpus ukfin
```

For current benchmark cohort reporting, prefer:

```bash
python -m obliqaxref.eval.cli benchmark-statistics
```
