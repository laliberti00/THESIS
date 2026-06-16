# nyc_tist — fairness metric coherence (G2)

Definitions:
- **LT-sink**: from `fairness/verdict.json::inequity_sinks` (KL ratio ≥ ~2× global)
- **KL-sink**: KL ≥ 2 × global_KL_mean (0.370)
- **Gini-sink**: per-situation item-exposure Gini ≥ mean + 1·std (thresh = 0.957)

| situation | n_requests | LT | KL | Gini | LT-sink | KL-sink | Gini-sink |
|---|---:|---:|---:|---:|:---:|:---:|:---:|
| s0 | 2034 | 0.0838 | 0.1065 | 0.9235 |  |  |  |
| s1 | 3146 | 0.0836 | 0.1057 | 0.9319 |  |  |  |
| s2 | 2116 | 0.0596 | 0.2403 | 0.9540 |  |  |  |
| s3 | 6656 | 0.0789 | 0.0755 | 0.9237 |  |  |  |
| s4 | 2214 | 0.0493 | 0.1330 | 0.9519 |  |  |  |
| s5 | 1536 | 0.0890 | 0.4497 | 0.9627 | ✓ | ✓ | ✓ |

**Coherence verdict**: COHERENT
- sinks_LT = [5]
- sinks_KL = [5]
- sinks_Gini = [5]