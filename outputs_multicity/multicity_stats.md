# G3 — Multi-city statistical hardening

Per-city paired bootstrap CI95 + Holm step-down over the 5 cities.

## ΔR@20 = B_full − B_blind (paired bootstrap on per-USER mean, B=10 000)

| city | n_users | B_blind R@20 | B_full R@20 | **ΔR@20** | **CI95** | Wilcoxon p | **Holm p** | reject H₀? | crosses 0? |
|---|---:|---:|---:|---:|---|---:|---:|:---:|:---:|
| istanbul | 22,631 | 0.0846 | 0.0780 | -0.00663 | [-0.00858, -0.00472] | 8.44e-20 | 4.22e-19 | ✅ | no |
| bangkok | 6,316 | 0.0923 | 0.0875 | -0.00483 | [-0.00816, -0.00152] | 4.03e-04 | 8.06e-04 | ✅ | no |
| nyc_tist | 4,113 | 0.0909 | 0.1080 | +0.01712 | [+0.01120, +0.02305] | 7.41e-09 | 2.96e-08 | ✅ | no |
| saopaulo | 4,395 | 0.0894 | 0.0889 | -0.00050 | [-0.00560, +0.00461] | 1.81e-01 | 1.81e-01 | ❌ | YES |
| tokyo_tist | 7,160 | 0.0747 | 0.0693 | -0.00537 | [-0.00860, -0.00207] | 2.83e-07 | 8.49e-07 | ✅ | no |

## ΔF1 = macro-F1(T-based) − macro-F1(time-only)

Per-city McNemar (already in Stage C); Holm across 5 cities.

| city | ΔF1 | n10 (T-only) | n01 (time-only) | McNemar p | **Holm p** | reject H₀? |
|---|---:|---:|---:|---:|---:|:---:|
| istanbul | +0.1004 | 21,137 | 20,307 | 4.66e-05 | 4.66e-05 | ✅ |
| bangkok | -0.0874 | 5,707 | 10,114 | 0.00e+00 | 0.00e+00 | ✅ |
| nyc_tist | +0.0655 | 3,334 | 4,144 | 0.00e+00 | 0.00e+00 | ✅ |
| saopaulo | +0.0612 | 3,769 | 4,883 | 0.00e+00 | 0.00e+00 | ✅ |
| tokyo_tist | -0.0826 | 11,866 | 17,416 | 0.00e+00 | 0.00e+00 | ✅ |