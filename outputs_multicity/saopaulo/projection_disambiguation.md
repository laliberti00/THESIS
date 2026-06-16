# saopaulo — projection as disambiguator (G4)

K=8  n_test=23,821  n_boundary=3,727 (15.6%)

## Boundary-case recognition F1 (BEFORE vs AFTER feedback)

| | macro-F1 | accuracy |
|---|---:|---:|
| BEFORE feedback (r argmax) | 0.4239 | 0.4800 |
| AFTER feedback (r̃ = r ⊙ T[z_prev,:]) | 0.5627 | 0.6246 |
| **Δ** | **+0.1388** | **+0.1446** |

## Do-no-harm (macro-F1 over ALL test rows)

| | macro-F1 |
|---|---:|
| BEFORE | 0.9218 |
| AFTER | 0.9427 |
| **Δ** | **+0.0209** |

## Honest future-prediction read (from Stage C)

- macro-F1 (T-based vs time-only): **ΔF1 = +0.0612**
  → favors T-based
- McNemar paired: T-only-correct = 3,769, time-only-correct = 4,883, p = 0.00e+00
  → favors time-only
- **Honest read**: macro-F1 and McNemar DISAGREE