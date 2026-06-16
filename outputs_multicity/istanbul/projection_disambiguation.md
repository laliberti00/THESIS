# istanbul — projection as disambiguator (G4)

K=8  n_test=136,220  n_boundary=29,081 (21.3%)

## Boundary-case recognition F1 (BEFORE vs AFTER feedback)

| | macro-F1 | accuracy |
|---|---:|---:|
| BEFORE feedback (r argmax) | 0.3687 | 0.4556 |
| AFTER feedback (r̃ = r ⊙ T[z_prev,:]) | 0.5617 | 0.5681 |
| **Δ** | **+0.1929** | **+0.1125** |

## Do-no-harm (macro-F1 over ALL test rows)

| | macro-F1 |
|---|---:|
| BEFORE | 0.8770 |
| AFTER | 0.9052 |
| **Δ** | **+0.0281** |

## Honest future-prediction read (from Stage C)

- macro-F1 (T-based vs time-only): **ΔF1 = +0.1004**
  → favors T-based
- McNemar paired: T-only-correct = 21,137, time-only-correct = 20,307, p = 4.66e-05
  → favors T-based
- **Honest read**: macro-F1 and McNemar AGREE