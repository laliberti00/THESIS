# saopaulo — fairness metric coherence (G2)

Definitions:
- **LT-sink**: from `fairness/verdict.json::inequity_sinks` (KL ratio ≥ ~2× global)
- **KL-sink**: KL ≥ 2 × global_KL_mean (0.462)
- **Gini-sink**: per-situation item-exposure Gini ≥ mean + 1·std (thresh = 0.956)

| situation | n_requests | LT | KL | Gini | LT-sink | KL-sink | Gini-sink |
|---|---:|---:|---:|---:|:---:|:---:|:---:|
| s0 | 2193 | 0.0563 | 0.0804 | 0.9346 |  |  |  |
| s1 | 1853 | 0.0292 | 0.3370 | 0.9557 |  |  |  |
| s2 | 8297 | 0.0785 | 0.0971 | 0.9156 |  |  |  |
| s3 | 1295 | 0.0456 | 0.2977 | 0.9483 |  |  |  |
| s4 | 3205 | 0.0706 | 0.0662 | 0.9217 |  |  |  |
| s5 | 328 | 0.0326 | 0.8059 | 0.9732 | ✓ | ✓ | ✓ |
| s6 | 2460 | 0.0658 | 0.0800 | 0.9204 |  |  |  |
| s7 | 4190 | 0.0639 | 0.0840 | 0.9279 |  |  |  |

**Coherence verdict**: COHERENT
- sinks_LT = [5]
- sinks_KL = [5]
- sinks_Gini = [5]