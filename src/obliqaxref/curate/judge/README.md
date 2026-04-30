# Curation Judge

The judge module implements the strict citation-dependency filter used by ObliQA-XRef curation.

Current policy:

- All eligible generated items are judged; IR agreement is diagnostic metadata only.
- Judge PASS items form the `dependency-valid` cohort.
- Answer validation is separate and controls the `answer-valid` cohort.
- Borderline cases should fail unless citation dependency is clear.

## Inputs

Each queue item contains:

- `item_id`
- `question`
- `source_passage_id`, `source_text`
- `target_passage_id`, `target_text`
- optional IR difficulty metadata

The judge is answer-aware only where the current prompt/schema requires answer support checks. Answer-validation remains a separate downstream gate for final answer-valid export.

## Judge v2 Rubric

The judge verifies:

1. SOURCE is relevant to the question.
2. TARGET is relevant to the question.
3. SOURCE alone is insufficient for a complete and correct answer.
4. TARGET adds essential missing information.
5. TARGET alone is insufficient if it lacks source-side regulatory context.
6. The answer is supported by SOURCE and TARGET.
7. The item genuinely requires following the source→target cross-reference.

Expected v2 fields include:

```json
{
  "passed": true,
  "final_score": 5,
  "subscores": {
    "realism": 5,
    "source_relevance": 5,
    "target_relevance": 5,
    "source_insufficiency": 5,
    "target_necessity": 5,
    "answer_support": 5
  },
  "binary_checks": {
    "source_relevant": true,
    "target_relevant": true,
    "source_alone_sufficient": false,
    "target_alone_sufficient": false,
    "target_adds_essential_information": true,
    "answer_supported": true,
    "citation_dependent": true
  },
  "reasons": []
}
```

Passing requires the configured score threshold and all required binary checks.

## Outputs

Under `runs/curate_<corpus>/out/curate_judge/`:

- `judge_queue.jsonl`
- `judge_responses_aggregated.jsonl`
- `judge_responses_pass.jsonl`
- `judge_responses_drop.jsonl`
- `judge_stats.json`

Final curated items preserve judge metadata when available:

- `judge_schema_version`
- `source_alone_sufficient`
- `target_alone_sufficient`
- `target_adds_essential_information`
- `citation_dependent`
- `answer_supported_by_judge`

## Usage

The normal entrypoint is the curation pipeline:

```bash
python -m obliqaxref curate --config configs/project.yaml
```

For dev configs:

```bash
python -m obliqaxref curate --config configs/adgm_dev.yaml
python -m obliqaxref curate --config configs/ukfin_dev.yaml
```

## Notes

- Failed LLM calls and invalid JSON are handled conservatively.
- Judge output is not the final experimental cohort by itself; final answer-valid export requires judge PASS ∩ answer PASS.
- IR labels such as `easy`, `hard`, or `neither` do not determine judge eligibility.
