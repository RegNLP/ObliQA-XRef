# Combined Controlled IR Table

All rows use the same retrieval setting (`controlled_shared_corpus_canonical_bm25`) and the same shared retrieval corpus (`shared_adgm_13015`). ObliQA and ObliQA-MP do not have pair-aware metrics, so those cells are shown as —.

| dataset_or_slice | method_type | sampling | test_size | Recall@10 | MAP@10 | nDCG@10 | Hit@10 | RelCount@10 | Both@10 | SRC-only@10 | TGT-only@10 | Neither@10 | PairMRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ObliQA |  |  | 2786 | 0.668 | 0.506 | 0.559 | 0.746 | 0.791 | — | — | — | — | — |
| ObliQA-MP |  |  | 447 | 0.513 | 0.387 | 0.503 | 0.919 | 1.157 | — | — | — | — | — |
| ObliQA-XRef-ADGM-ALL | ALL | ALL | 502 | 0.802 | 0.625 | 0.724 | 0.980 | 1.604 | 0.624 | 0.167 | 0.189 | 0.020 | 0.205 |
| ObliQA-XRef-ADGM-DPEL | DPEL | ALL | 173 | 0.798 | 0.609 | 0.714 | 0.988 | 1.595 | 0.607 | 0.110 | 0.272 | 0.012 | 0.189 |
| ObliQA-XRef-ADGM-SCHEMA | SCHEMA | ALL | 329 | 0.804 | 0.634 | 0.730 | 0.976 | 1.608 | 0.632 | 0.198 | 0.146 | 0.024 | 0.213 |
| ObliQA-XRef-ADGM-DPEL + mixed_difficulty | DPEL | mixed_difficulty | 99 | 0.778 | 0.598 | 0.703 | 0.990 | 1.556 | 0.566 | 0.091 | 0.333 | 0.010 | 0.178 |
| ObliQA-XRef-ADGM-DPEL + hard_enriched | DPEL | hard_enriched | 74 | 0.824 | 0.623 | 0.728 | 0.986 | 1.649 | 0.662 | 0.135 | 0.189 | 0.014 | 0.205 |
| ObliQA-XRef-ADGM-SCHEMA + mixed_difficulty | SCHEMA | mixed_difficulty | 188 | 0.827 | 0.632 | 0.731 | 0.984 | 1.654 | 0.670 | 0.176 | 0.138 | 0.016 | 0.221 |
| ObliQA-XRef-ADGM-SCHEMA + hard_enriched | SCHEMA | hard_enriched | 141 | 0.773 | 0.638 | 0.728 | 0.965 | 1.546 | 0.582 | 0.227 | 0.156 | 0.035 | 0.202 |
