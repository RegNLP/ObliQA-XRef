# ResourceStats

`ResourceStats` is the legacy intrinsic statistics utility for corpus and cross-reference summaries.

Use it for:

- number of documents and passages
- passage length distributions
- resolved cross-reference counts
- source/target passage coverage
- reference type breakdowns

```bash
python src/obliqaxref/eval/ResourceStats/cli.py compute --corpus adgm
python src/obliqaxref/eval/ResourceStats/cli.py compute --corpus ukfin
```

Outputs are written under:

```text
runs/stats/eval/resourcestats/<corpus>/resource_stats.json
```

For current benchmark cohort statistics, difficulty distributions, and paper-ready tables, prefer:

```bash
python -m obliqaxref.eval.cli benchmark-statistics \
  --out ObliQA-XRef_Out_Datasets
```

That command reads final cohort files:

- `final_dependency_valid.jsonl`
- `final_answer_valid.jsonl`
- `final_answer_failed.jsonl`

and writes:

- `benchmark_statistics.csv`
- `benchmark_statistics_by_difficulty.csv`
- `benchmark_statistics_summary.md`
- `benchmark_statistics_latex_tables.tex`
