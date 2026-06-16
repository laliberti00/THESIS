# tokyo_tist — fairness metric coherence (G2)

Definitions:
- **LT-sink**: from `fairness/verdict.json::inequity_sinks` (KL ratio ≥ ~2× global)
- **KL-sink**: KL ≥ 2 × global_KL_mean (0.067)
- **Gini-sink**: per-situation item-exposure Gini ≥ mean + 1·std (thresh = 0.969)

| situation | n_requests | LT | KL | Gini | LT-sink | KL-sink | Gini-sink |
|---|---:|---:|---:|---:|:---:|:---:|:---:|
| s0 | 22277 | 0.0358 | 0.0376 | 0.9537 |  |  |  |
| s1 | 8685 | 0.0179 | 0.0475 | 0.9657 |  |  |  |
| s2 | 21016 | 0.0162 | 0.0216 | 0.9653 |  |  |  |
| s3 | 7583 | 0.0125 | 0.0273 | 0.9687 |  |  |  |

**Coherence verdict**: COHERENT
- sinks_LT = []
- sinks_KL = []
- sinks_Gini = []