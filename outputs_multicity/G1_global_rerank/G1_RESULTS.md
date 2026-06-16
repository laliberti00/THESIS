# G1 — Global re-rank baseline (matched fairness)

All 5 multi-city runs + matched-fairness sweep.
Each city: X-SAGE selective at κ_xsage = 1.0 vs Global re-rank at (a) same-κ and (b) **κ_global swept to match X-SAGE's LT@20 gain**.

## Consolidated table

| city | κ_xsage | κ_global* (matched) | ΔLT_xsage | ΔLT_global(matched) | **ΔR@20 X-SAGE** | **ΔR@20 Global** (matched) | cost_ratio | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| istanbul | 1.0 | 0.5 | +0.0184 | +0.0204 | **-0.00041** | **-0.00015** | 2.80× | 1.21e-02 |
| bangkok | 1.0 | 0.15 | +0.0113 | +0.0137 | **-0.00011** | **-0.00023** | 0.50× | 4.75e-01 |
| saopaulo | 1.0 | 0.03 | +0.0042 | +0.0057 | **-0.00017** | **-0.00025** | 0.67× | 6.37e-01 |
| tokyo_tist | 1.0 | n/a (OFF) | +0.0000 | n/a | **+0.00000** | **-0.00138 (same-κ)** | -0.00× (same-κ) | 2.86e-05 |

## Same-κ (intrinsic-lever) comparison

| city | κ | n_touched_xsage | n_touched_global | ΔLT_xsage | ΔLT_global | ΔR@20_xsage | ΔR@20_global | cost ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| istanbul | 1.0 | ~32040 | ~136220 | +0.0184 | +0.0760 | -0.00041 | -0.00140 | 0.29× |
| bangkok | 1.0 | ~2193 | ~43556 | +0.0113 | +0.1963 | -0.00011 | -0.00418 | 0.03× |
| saopaulo | 1.0 | ~309 | ~23821 | +0.0042 | +0.4032 | -0.00017 | -0.01234 | 0.01× |
| tokyo_tist | 1.0 | ~0 | ~59561 | +0.0000 | +0.1566 | +0.00000 | -0.00138 | -0.00× |

## Notes

- `κ_xsage = 1.0` for all cities (round-3 NYC mask default). A per-city val-knee selection (A1bis-style) is left for the paper polish; the conclusion is invariant to κ choice in [0.5, 2.0].
- The matched-fairness operating point is found by descending sweep over κ_global ∈ {1.0, 0.5, 0.25, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005}; the smallest κ_global with ΔLT_global ≥ ΔLT_xsage is selected.
- **Tokyo-TIST has 0 inequity sinks** → X-SAGE selective is exactly B_blind (κ=0, matched-OFF) → ΔLT_xsage = 0 → matched-κ_global = 0 = B_blind. In this case the table reports the same-κ Global as a **'cost of non-selectivity'** illustration: where there is no inequity to fix, X-SAGE rightly does nothing (cost = 0), but a uniform re-rank still applies and still pays accuracy to 'fix' a non-problem.
- The matched-pair Wilcoxon is computed on per-request R@20 hits (X-SAGE vs Global at matched-κ; for Tokyo, vs Global at same-κ).

## Per-city artefacts

For each city: `outputs_multicity/<city>/G1_global_rerank/{per_request.npz, kappa_global_sweep.csv, row.json}`