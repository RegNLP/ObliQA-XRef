# ObliQA-XRef

ObliQA-XRef is a regulatory NLP benchmark-construction pipeline for **citation-dependent regulatory QA**. It is part of the ObliQA benchmark family:

- **ObliQA**: obligation-grounded regulatory QA
- **ObliQA-MP**: general multi-passage regulatory QA
- **ObliQA-XRef**: cross-reference-aware QA where the answer requires following an explicit source→target citation

The benchmark focuses on items where the source passage is relevant but insufficient, the cited target passage adds essential information, and the final answer is supported by both.

## Pipeline

1. **Adapter**: normalize corpus passages and resolve source→target cross-references.
2. **Generate**: create DPEL and SCHEMA QA candidates from cross-reference pairs.
3. **Curate**: run diagnostic retrieval, strict judge v2 citation-dependency filtering (final selector), and optional answer validation (diagnostic only).
4. **Finalize/Evaluate**: export dependency-valid (judge PASS) benchmark splits by default, run retrieval/answer diagnostics, and produce analysis tables.

Current curation policy:

- IR agreement is **diagnostic only**.
- `dependency-valid` = judge PASS.
- `answer-valid` = judge PASS ∩ answer PASS.
- Final benchmark defaults to `dependency-valid` (judge PASS).
- Dependency-valid but answer-failed items are retained separately for diagnosis.

## Key Outputs

Curation writes explicit final cohorts under `runs/curate_<corpus>/out/`:

- `final_dependency_valid.jsonl` / `.csv`
- `final_answer_valid.jsonl` / `.csv`
- `final_answer_failed.jsonl` / `.csv`
- `final_benchmark.jsonl` / `.csv` compatibility alias, based on `curation.final_export_basis`
- `final_hard.jsonl` / `.csv` challenging subset of the compatibility alias
- `final_benchmark_stats.json`

Finalized downstream datasets are written under `ObliQA-XRef_Out_Datasets/`.

All final exports include ObliQA-family metadata fields such as `benchmark_family`, `benchmark_name`, `benchmark_role`, `evidence_structure`, and relation fields for ObliQA / ObliQA-MP positioning.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m obliqaxref --help
```

Run the main stages:

```bash
python -m obliqaxref adapter --config configs/project.yaml
python -m obliqaxref generate --config configs/project.yaml
python -m obliqaxref curate --config configs/project.yaml
python -m obliqaxref.eval.cli finalize --corpus both --cohort dependency_valid
```

Corpus-specific dev configs are available in `configs/adgm_dev.yaml` and `configs/ukfin_dev.yaml`.

## Generation

ObliQA-XRef supports two generation methods:

- **DPEL**: direct passage-driven QA generation from source→target pairs.
- **SCHEMA**: schema-guided generation using extracted hooks, item types, and target spans.

Current generation safeguards:

- Citation leakage detector for rule/section identifiers in generated prose.
- `no_citations_in_question` config to prevent retrieval shortcuts in questions.
- DPEL prompt rules that avoid citation leakage.
- SCHEMA defaults with no citations in questions and stronger dual-anchor behavior.
- Difficulty-aware cross-reference sampling.
- Pilot-run mode for small, isolated end-to-end runs.

Examples:

```bash
# Standard generation
python -m obliqaxref generate --config configs/project.yaml

# Fast smoke run
python -m obliqaxref generate --config configs/project.yaml --preset smoke

# Sampling-limited generation
python -m obliqaxref generate --config configs/project.yaml --max-pairs 50

# Difficulty-aware sampling is configured in YAML, for example:
# sampling.sampling_mode: mixed_difficulty
python -m obliqaxref generate --config configs/project.yaml

# Pilot mode is configured in YAML:
# pilot.pilot_mode: true
# pilot.pilot_n_xrefs_per_corpus: 50
python -m obliqaxref generate --config configs/adgm_dev.yaml
```

See [docs/generator/README.md](docs/generator/README.md).

## Curation

The current curation stage:

- Runs BM25, E5, RRF, CE reranking, and XRefExpand baselines where configured.
- Assigns `ir_difficulty_label` as metadata only.
- Sends all eligible generated items to the judge.
- Uses judge schema v2 fields for source relevance, target relevance, source insufficiency, target necessity, answer support, and citation dependency.
- Answer validation is optional and produces diagnostic subsets `final_answer_valid` and `final_answer_failed` (it does not control the default final benchmark).

Examples:

```bash
# Full curation
python -m obliqaxref curate --config configs/project.yaml

# Curation IR subcommand used during retrieval/debug work
python -m obliqaxref.curate.cli ir -c configs/adgm_dev.yaml

# Skip expensive stages for debugging
python -m obliqaxref curate --config configs/project.yaml --skip-answer
```

See [docs/curation/README.md](docs/curation/README.md).

## Retrieval And Evaluation

Supported retrieval/evaluation methods include:

- BM25
- E5 dense retrieval
- RRF fusion
- Cross-encoder reranking over union candidates
- BM25+XRefExpand, E5+XRefExpand, and RRF+XRefExpand

The CE reranker uses the actual question text for scoring, not `query_id`.

Pair-level retrieval metrics:

- `Both@5/10/20`
- `SRC-only@5/10/20`
- `TGT-only@5/10/20`
- `Neither@5/10/20`
- `PairMRR`

Standard metrics such as Recall@k, MAP@k, and nDCG@k are preserved.
For citation-dependent QA, retrieving only one of the two evidence passages is insufficient; prefer pair-aware diagnostics (Both@K, SRC-only@K, TGT-only@K, Neither@K, PairMRR) as the primary retrieval indicators.

Examples:

```bash
# Finalize dependency-valid cohort (judge PASS) for experiments
python -m obliqaxref.eval.cli finalize --corpus both --cohort dependency_valid

# IR evaluation
python -m obliqaxref.eval.cli ir --corpus both --k 10

# Answer generation and evaluation
python -m obliqaxref.eval.cli answer --corpus both --methods bm25 e5 rrf ce_rerank_union200
python -m obliqaxref.eval.cli answer-eval --corpus both
```

See [docs/eval/README.md](docs/eval/README.md).

## Sampling Regimes

Two pre-generation sampling regimes are supported:

- `mixed_difficulty`: coverage-oriented; approximates a natural distribution and maximizes dataset size while retaining a mix of easier and harder pairs.
- `hard_enriched`: challenge-oriented heuristic that prefers lower source–target lexical overlap, document diversity, and moderately longer passages. It increases retrieval difficulty partially (especially for DPEL) but does not uniformly make all items hard; SCHEMA often remains easier for dense/RRF.

The generator writes `stats/sampling_report.json` with `sampling_mode`, and `stats/sampled_xrefs_with_features.jsonl` containing per-row features; for `hard_enriched`, a `hard_enriched_score` is included for transparency.

## Pilot Findings (Examples)

- Mixed overnight500
  - ADGM: generated=300, final benchmark=261 (dependency-valid). Retrieval tends to be easy, especially SCHEMA.
  - UKFIN: generated=183, final benchmark=116.
  - Combined final=377.
- Hard-enriched100
  - ADGM: generated=67, final benchmark=56.
  - UKFIN: generated=53, final benchmark=35.
  - Combined final=91.
  - Improvements are clearest for DPEL (e.g., ADGM DPEL BM25 Both@10=0.000; UKFIN DPEL CE Both@10=0.000). SCHEMA remains relatively easier for dense/RRF.

These pilots motivate presenting ObliQA-XRef as a benchmark with a natural/coverage subset (mixed_difficulty) and a challenge-oriented subset (hard_enriched), not as uniformly hard.

## Recommended Workflow

1) Generate (mixed_difficulty) and curate:

```bash
python -m obliqaxref generate -c configs/project.yaml --preset dev --sampling-mode mixed_difficulty
python -m obliqaxref curate   -c configs/project.yaml --preset dev --skip-answer
```

2) Generate (hard_enriched) and curate as a challenge subset:

```bash
python -m obliqaxref generate -c configs/project.yaml --preset dev --sampling-mode hard_enriched \
  --pilot --pilot-suffix pilot_hardenriched100
python -m obliqaxref curate   -c configs/project.yaml --preset dev --skip-answer \
  --pilot --pilot-suffix pilot_hardenriched100
```

3) Finalize a clean root for downstream diagnostics (no restaging):

```bash
rm -rf ObliQA-XRef_Out_Datasets/pilot_hardenriched100_clean
python -m obliqaxref.eval.cli finalize --corpus both --cohort dependency_valid \
  --curate-suffix pilot_hardenriched100 \
  --output-dir ObliQA-XRef_Out_Datasets/pilot_hardenriched100_clean
```

4) Sanity-check staged `.trec` qid counts match the finalized test split.

5) Run IR diagnostics without restaging (default behavior preserves existing `.trec` files):

```bash
python -m obliqaxref.eval.cli ir --corpus both \
  --root ObliQA-XRef_Out_Datasets/pilot_hardenriched100_clean \
  --methods bm25 ft_e5 rrf_bm25_e5 ce_rerank_union200 bm25_xref_expand e5_xref_expand rrf_xref_expand
```

Warning: do not interpret IR diagnostics if staged `.trec` qids do not match finalized test qids.

## Analysis Utilities

```bash
# Answer quality grouped by retrieval outcome
python -m obliqaxref.eval.cli answer-quality-by-retrieval \
  --root ObliQA-XRef_Out_Datasets

# Human audit sample export (you may also sample from final_dependency_valid.jsonl)
python -m obliqaxref.eval.cli human-audit-export \
  --input runs/curate_adgm/out/final_dependency_valid.jsonl runs/curate_ukfin/out/final_dependency_valid.jsonl \
  --out ObliQA-XRef_Out_Datasets \
  --n 200 --seed 13

# Human audit aggregation
python -m obliqaxref.eval.cli human-audit-aggregate \
  --inputs annotations_1.csv annotations_2.csv \
  --out ObliQA-XRef_Out_Datasets

# Benchmark statistics and paper-ready tables (defaults to dependency_valid when inputs omitted)
python -m obliqaxref.eval.cli benchmark-statistics --out ObliQA-XRef_Out_Datasets
```

## Project Structure

```text
configs/                 YAML configs
data/                    local corpus inputs
docs/                    stage documentation
scripts/                 utility scripts
src/obliqaxref/adapter/  corpus adapters
src/obliqaxref/generate/ DPEL and SCHEMA generation
src/obliqaxref/curate/   IR diagnostics, judge, answer validation, final cohorts
src/obliqaxref/eval/     finalize, retrieval eval, answer eval, analysis utilities
tests/                   regression tests
runs/                    local pipeline outputs
```

## Troubleshooting

| Issue | Check |
| --- | --- |
| Missing final items | Confirm judge PASS exists under `curate_judge/`. Answer PASS is optional and diagnostic. |
| Unexpected low retrieval agreement | IR is diagnostic; inspect `ir_difficulty_label` and pair metrics rather than dropping items. |
| CE reranker errors | Confirm `sentence_transformers` dependencies and model availability. |
| Missing XRefExpand runs | Confirm `crossref_resolved.cleaned.csv` exists and base TREC runs were written. |
| Judge dropped all items due to missing text | Check `curated_items.judge.jsonl` for `source_text`/`target_text`. Curation attaches texts using passage IDs via `pid`/`passage_uid`/`passage_id`/`id` and text via `text`/`passage`/`content`. If `generator/passages_index.jsonl` is empty, the judge falls back to the adapter corpus automatically. |
| Missing answer-quality joins | Confirm `retrieval_diagnostics_per_query.csv` and `answer_eval_*_test.json` are under the same root. |

## License

MIT. See [LICENSE](LICENSE).
