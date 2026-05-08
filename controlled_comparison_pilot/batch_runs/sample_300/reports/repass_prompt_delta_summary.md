# RePASs Prompt Delta Analysis

## Summary

| dataset | metric | n | mean_general | mean_task_aware | mean_delta | median_delta | std_delta | task_better | general_better | equal | task_better_% | general_better_% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| obliqa | entailment_score | 300 | 0.506954 | 0.488182 | -0.018772 | 0.000000 | 0.250897 | 133 | 139 | 28 | 0.443333 | 0.463333 |
| obliqa | contradiction_score | 300 | 0.453694 | 0.477158 | 0.023464 | 0.000000 | 0.241861 | 128 | 144 | 28 | 0.426667 | 0.480000 |
| obliqa | obligation_coverage_score | 300 | 0.078187 | 0.081379 | 0.003192 | 0.000000 | 0.064124 | 74 | 62 | 164 | 0.246667 | 0.206667 |
| obliqa | composite_score | 300 | 0.377149 | 0.364134 | -0.013015 | 0.000000 | 0.122450 | 131 | 141 | 28 | 0.436667 | 0.470000 |
| obliqa_mp | entailment_score | 300 | 0.580596 | 0.574388 | -0.006208 | 0.000000 | 0.243830 | 148 | 146 | 6 | 0.493333 | 0.486667 |
| obliqa_mp | contradiction_score | 300 | 0.311521 | 0.299440 | -0.012081 | 0.000000 | 0.207570 | 149 | 145 | 6 | 0.496667 | 0.483333 |
| obliqa_mp | obligation_coverage_score | 300 | 0.113993 | 0.122316 | 0.008323 | 0.000000 | 0.071035 | 92 | 66 | 142 | 0.306667 | 0.220000 |
| obliqa_mp | composite_score | 300 | 0.461023 | 0.465755 | 0.004732 | 0.000000 | 0.116948 | 145 | 149 | 6 | 0.483333 | 0.496667 |
| xref_adgm | entailment_score | 300 | 0.612947 | 0.575134 | -0.037813 | -0.000495 | 0.226750 | 134 | 154 | 12 | 0.446667 | 0.513333 |
| xref_adgm | contradiction_score | 300 | 0.488387 | 0.482055 | -0.006332 | -0.000135 | 0.206237 | 151 | 137 | 12 | 0.503333 | 0.456667 |
| xref_adgm | obligation_coverage_score | 300 | 0.108725 | 0.116858 | 0.008132 | 0.000000 | 0.070075 | 85 | 48 | 167 | 0.283333 | 0.160000 |
| xref_adgm | composite_score | 300 | 0.411095 | 0.403312 | -0.007783 | -0.000025 | 0.106528 | 138 | 150 | 12 | 0.460000 | 0.500000 |

## Interpretation

- obliqa: general has the higher mean composite score; task-aware improves obligation coverage (mean delta 0.003192) and increases contradiction (mean reduction -0.023464). Composite wins are mixed: task-aware 43.7%, general 47.0%.
- obliqa_mp: task-aware has the higher mean composite score; task-aware improves obligation coverage (mean delta 0.008323) and reduces contradiction (mean reduction 0.012081). Composite wins are mixed: task-aware 48.3%, general 49.7%.
- xref_adgm: general has the higher mean composite score; task-aware improves obligation coverage (mean delta 0.008132) and reduces contradiction (mean reduction 0.006332). Composite wins are mixed: task-aware 46.0%, general 50.0%.

## Pairing Checks

- obliqa: paired=300, general=300, task_aware=300, mismatches=0
- obliqa_mp: paired=300, general=300, task_aware=300, mismatches=0
- xref_adgm: paired=300, general=300, task_aware=300, mismatches=0
