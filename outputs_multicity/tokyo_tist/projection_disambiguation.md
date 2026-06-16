# tokyo_tist — projection as disambiguator (G4)

K=4  n_test=59,561  n_boundary=8,816 (14.8%)

## Boundary-case recognition F1 (BEFORE vs AFTER feedback)

| | macro-F1 | accuracy |
|---|---:|---:|
| BEFORE feedback (r argmax) | 0.5316 | 0.6969 |
| AFTER feedback (r̃ = r ⊙ T[z_prev,:]) | 0.6484 | 0.7933 |
| **Δ** | **+0.1167** | **+0.0964** |

## Do-no-harm (macro-F1 over ALL test rows)

| | macro-F1 |
|---|---:|
| BEFORE | 0.9340 |
| AFTER | 0.9579 |
| **Δ** | **+0.0239** |

## Honest future-prediction read (from Stage C)

- macro-F1 (T-based vs time-only): **ΔF1 = -0.0826**
  → favors time-only
- McNemar paired: T-only-correct = 11,866, time-only-correct = 17,416, p = 0.00e+00
  → favors time-only
- **Honest read**: macro-F1 and McNemar AGREE