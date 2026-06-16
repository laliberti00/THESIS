# bangkok — fairness metric coherence (G2)

Definitions:
- **LT-sink**: from `fairness/verdict.json::inequity_sinks` (KL ratio ≥ ~2× global)
- **KL-sink**: KL ≥ 2 × global_KL_mean (0.137)
- **Gini-sink**: per-situation item-exposure Gini ≥ mean + 1·std (thresh = 0.969)

| situation | n_requests | LT | KL | Gini | LT-sink | KL-sink | Gini-sink |
|---|---:|---:|---:|---:|:---:|:---:|:---:|
| s0 | 6413 | 0.0198 | 0.0868 | 0.9715 |  |  | ✓ |
| s1 | 7659 | 0.0447 | 0.0716 | 0.9630 |  |  |  |
| s2 | 2254 | 0.0475 | 0.1042 | 0.9616 | ✓ |  |  |
| s3 | 27230 | 0.0300 | 0.0107 | 0.9606 |  |  |  |

**Coherence verdict**: DISAGREEMENT
- sinks_LT = [2]
- sinks_KL = []
- sinks_Gini = [0]
- **intersect (agree on)**: []
- **disagreement**: situations in some but not all metric: [0, 2]