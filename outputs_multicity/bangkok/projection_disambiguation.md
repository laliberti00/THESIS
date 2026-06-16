# bangkok — projection as disambiguator (G4)

K=4  n_test=43,556  n_boundary=10,093 (23.2%)

## Boundary-case recognition F1 (BEFORE vs AFTER feedback)

| | macro-F1 | accuracy |
|---|---:|---:|
| BEFORE feedback (r argmax) | 0.2626 | 0.3836 |
| AFTER feedback (r̃ = r ⊙ T[z_prev,:]) | 0.6885 | 0.8065 |
| **Δ** | **+0.4259** | **+0.4229** |

## Do-no-harm (macro-F1 over ALL test rows)

| | macro-F1 |
|---|---:|
| BEFORE | 0.8678 |
| AFTER | 0.9506 |
| **Δ** | **+0.0828** |

## Honest future-prediction read (from Stage C)

- macro-F1 (T-based vs time-only): **ΔF1 = -0.0874**
  → favors time-only
- McNemar paired: T-only-correct = 5,707, time-only-correct = 10,114, p = 0.00e+00
  → favors time-only
- **Honest read**: macro-F1 and McNemar AGREE