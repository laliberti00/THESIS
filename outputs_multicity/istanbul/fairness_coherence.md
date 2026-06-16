# istanbul — fairness metric coherence (G2)

Definitions:
- **LT-sink**: from `fairness/verdict.json::inequity_sinks` (KL ratio ≥ ~2× global)
- **KL-sink**: KL ≥ 2 × global_KL_mean (0.084)
- **Gini-sink**: per-situation item-exposure Gini ≥ mean + 1·std (thresh = 0.977)

| situation | n_requests | LT | KL | Gini | LT-sink | KL-sink | Gini-sink |
|---|---:|---:|---:|---:|:---:|:---:|:---:|
| s0 | 23015 | 0.0072 | 0.0719 | 0.9768 | ✓ |  | ✓ |
| s1 | 10582 | 0.0085 | 0.0625 | 0.9759 |  |  |  |
| s2 | 25713 | 0.0113 | 0.0202 | 0.9738 |  |  |  |
| s3 | 20083 | 0.0088 | 0.0129 | 0.9746 |  |  |  |
| s4 | 15596 | 0.0099 | 0.0652 | 0.9730 | ✓ |  |  |
| s5 | 12628 | 0.0059 | 0.0482 | 0.9778 |  |  | ✓ |
| s6 | 13694 | 0.0087 | 0.0259 | 0.9744 |  |  |  |
| s7 | 14909 | 0.0085 | 0.0285 | 0.9739 |  |  |  |

**Coherence verdict**: DISAGREEMENT
- sinks_LT = [0, 4]
- sinks_KL = []
- sinks_Gini = [0, 5]
- **intersect (agree on)**: []
- **disagreement**: situations in some but not all metric: [0, 4, 5]