# `obliqaxref.generate`

Generator module for ObliQA-XRef citation-dependent QA candidates.

Inputs:

- `runs/adapter_<corpus>/processed/passage_corpus.jsonl`
- `runs/adapter_<corpus>/processed/crossref_resolved.cleaned.csv`

Methods:

- **DPEL**: direct passage-driven QA generation from source→target pairs.
- **SCHEMA**: schema extraction followed by schema-guided QA generation.

Current safeguards:

- citation leakage detector
- `no_citations_in_question`
- DPEL no-citation prompt rules
- SCHEMA dual-anchor defaults
- difficulty-aware cross-reference sampling
- pilot-run mode

Typical commands:

```bash
python -m obliqaxref generate --config configs/project.yaml
python -m obliqaxref generate --config configs/project.yaml --preset smoke
python -m obliqaxref generate --config configs/project.yaml --max-pairs 50
```

Primary outputs under `runs/generate_<corpus>/out/`:

- `generator/items.jsonl`
- `dpel/dpel.qa.jsonl`
- `schema/schema.extraction.jsonl`
- `schema/schema.qa.jsonl`
- `stats/generate_report.json`
