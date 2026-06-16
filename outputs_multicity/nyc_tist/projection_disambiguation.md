# nyc_tist — projection as disambiguator (G4)

K=6  n_test=17,702  n_boundary=3,839 (21.7%)

## Boundary-case recognition F1 (BEFORE vs AFTER feedback)

| | macro-F1 | accuracy |
|---|---:|---:|
| BEFORE feedback (r argmax) | 0.3792 | 0.4306 |
| AFTER feedback (r̃ = r ⊙ T[z_prev,:]) | 0.5269 | 0.5965 |
| **Δ** | **+0.1476** | **+0.1659** |

## Do-no-harm (macro-F1 over ALL test rows)

| | macro-F1 |
|---|---:|
| BEFORE | 0.8646 |
| AFTER | 0.8971 |
| **Δ** | **+0.0325** |

## Honest future-prediction read (from Stage C)

- macro-F1 (T-based vs time-only): **ΔF1 = +0.0655**
  → favors T-based
- McNemar paired: T-only-correct = 3,334, time-only-correct = 4,144, p = 0.00e+00
  → favors time-only
- **Honest read**: macro-F1 and McNemar DISAGREE