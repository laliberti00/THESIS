# R8 pre-registered predictions — round-4 multi-city generalisation

> **Commit timestamp precedes all Stage runs on the new cities.** The
> predictions below derive entirely from each city's TIST2015 raw
> TT_share and macro entropy (Step-1 size check) plus the round-3
> per-target-macro law (C5.0) and the C6 causal validation. NO peeking
> at any Stage output is allowed before these are committed.
>
> All confidences (HIGH/MED/LOW) are stated explicitly so a reviewer
> can score the hit rate after the runs.

## Round-3 evidence base (the law's anchors)

| dataset / city | TT_share | macro entropy | Stage-C ΔF1 | B6 lens verdict | B_full vs B_blind sign |
|---|---|---|---|---|---|
| TSMC NYC | ~0.25 | ~0.85 | **+0.171** (T wins, p≪0.001) | **GREEN** | **B_full +0.0150 (p_Holm=8e-5)** |
| TSMC TKY | ~0.71 | ~0.49 | **−0.041** (time-only wins, p≈0) | **RED**   | **B_full −0.0056 (p_Holm=8e-9)** |
| TSMC TKY* (C6, 25 % T&T) | counterfactual | — | −0.012 (closer to 0) | GREEN | **per-macro law confirmed** (T&T −, non-T&T +) |

The **central law (C5.0/C6)**: per-target-macro signs of (B_full − B_blind)
are invariant — negative on T&T-target requests, positive on non-T&T —
and the aggregate sign equals their share-weighted sum.

## Cities to be tested

(Sizes confirmed by the Step-1 size check; see `outputs_multicity/selection/final_cities.md`.)

| city_key | TIST name | raw TT_share | role |
|---|---|---|---|
| `istanbul` | Istanbul (TR) | ~0.060 | very-low T&T (deepest end of the spectrum) |
| `bangkok` | Bangkok (TH) | ~0.103 | low T&T (regional / SE Asia control) |
| `nyc_tist` | New York (US) | ~0.141 | low-mid T&T + **round-3 provenance bridge** to TSMC NYC (TT 0.25) |
| `saopaulo` | São Paulo (BR) | ~0.150 | low-mid T&T (Western-Hemisphere control) |
| `tokyo_tist` | Tokyo (JP) | ~0.450 | mid-high T&T + **round-3 provenance bridge** to TSMC TKY (TT 0.71) |

## Predictions, per city

### 1. Attractors (Stage A, K_opt and # of attractors)

| city | TT_share | predicted # attractors | confidence | basis |
|---|---|---|---|---|
| `istanbul`   | 0.060 | **≥ 4** (rich attractor structure) | HIGH | NYC pattern (5 attractors at TT 0.25) extrapolated — Istanbul is below NYC, so attractor count should be at least as rich. |
| `bangkok`    | 0.103 | **≥ 4** | HIGH | Same reasoning. |
| `nyc_tist`   | 0.141 | **≥ 4** (provenance probe: should also match NYC-TSMC's 5) | HIGH | If the structural attractor pattern is robust to collection pipeline, NYC-TIST recovers ~5 attractors. |
| `saopaulo`   | 0.150 | **≥ 4** | HIGH | Same reasoning. |
| `tokyo_tist` | 0.450 | **2 or 3** (intermediate collapse) | MED | Between NYC's 5 and TSMC TKY's 2. TIST TT 0.45 is below TSMC TKY's 0.71; expect partial collapse toward transit but with leisure attractors surviving. |

### 2. Stage-C projection: macro-F1 ΔF1 (T-based vs time-only)

| city | TT_share | predicted ΔF1 sign | confidence | basis |
|---|---|---|---|---|
| `istanbul`   | 0.060 | **strongly +** (≥ +0.10) | HIGH | Very low T&T → attractor structure clean → T-based prior wins. |
| `bangkok`    | 0.103 | **+** (≥ +0.05) | HIGH | Same regime. |
| `nyc_tist`   | 0.141 | **+** (≥ +0.05) | HIGH | NYC-TIST's TT is lower than TSMC NYC's, projection should be positive even if attenuated. |
| `saopaulo`   | 0.150 | **+** (≥ +0.05) | HIGH | Same regime. |
| `tokyo_tist` | 0.450 | **near zero or marginally negative**, plausibly in [−0.05, +0.05] | MED | Between TSMC TKY (−0.041 at TT 0.71) and TSMC TKY* (−0.012 at counterfactual TT 0.47). |

**Threshold prediction.** ΔF1 crosses zero somewhere between TT_share ≈ 0.40
and 0.55. None of our TIST cities should sit deeply negative.

### 3. Per-target-macro law (B_full vs B_blind)

The structural prediction (highest-confidence claim of the entire round):

> **For every city tested**, the per-request paired Δ(B_full − B_blind) has
> SAME signs as round-3: **negative on T&T-target requests, non-negative
> (and likely positive) on non-T&T-target requests**, with the aggregate
> sign predictable by share-weighted sum.

Per-city aggregate predictions (closed-form from C5.0 law):

| city | TT_share | predicted Δ T&T | predicted Δ non-T&T | predicted aggregate Δ | confidence |
|---|---|---|---|---|---|
| `istanbul`   | 0.060 | ~ −0.010 | ~ +0.025 | **+0.023** (TT-weighted) | HIGH |
| `bangkok`    | 0.103 | ~ −0.010 | ~ +0.025 | **+0.022** | HIGH |
| `nyc_tist`   | 0.141 | ~ −0.013 | ~ +0.026 | **+0.020** | HIGH |
| `saopaulo`   | 0.150 | ~ −0.013 | ~ +0.025 | **+0.020** | HIGH |
| `tokyo_tist` | 0.450 | ~ −0.011 | ~ +0.020 | **+0.006** (near zero, sign uncertain) | MED |

Per-macro magnitudes use the round-3 NYC pair (T&T −0.013, non-T&T +0.026)
as the reference; Tokyo-TIST is interpolated between NYC and TSMC TKY (−0.008, +0.0003).
**Aggregate sign**:
- HIGH-confidence positive on the 4 low-TT cities.
- MED-confidence near-zero on Tokyo-TIST — could flip negative if the
  TT_share=0.45 regime is enough to cross the boundary.

### 4. Fairness sinks (Stage B)

| city | predicted | confidence |
|---|---|---|
| every city | **≥ 1 inequity sink** (situation with KL_KL_ratio ≥ ~2× global) | HIGH (B6 backbone-agnostic finding generalises across data) |
| every city | sinks are **situation-localised** (not random; localised to specific time-band × intent combinations) | HIGH |
| no city | (no prediction on which `situation_id` is the sink) | — |

### 5. Provenance robustness — `nyc_tist` vs TSMC NYC, `tokyo_tist` vs TSMC TKY

**This is a natural causal test complementary to C6.** Two scenarios:

* **A (robust):** each city's behaviour follows its OWN TT_share.
  NYC-TIST (TT 0.14) ≈ TSMC NYC behaviour (TT 0.25) → both GREEN /
  both ΔF1 positive / both B_full beats B_blind.
  Tokyo-TIST (TT 0.45) ≠ TSMC TKY behaviour (TT 0.71) → Tokyo-TIST
  sits near the crossover (Stage-C ΔF1 ≈ 0; B_full aggregate sign
  ambiguous), TSMC TKY remains negative.
  **This is the predicted outcome: the law fires off TT_share, not off
  city identity**, so different collection pipelines on the same real
  city produce predictable, different behaviours.

* **B (fragile):** city identity matters more than TT_share. NYC-TIST
  and TSMC NYC behave similarly; Tokyo-TIST and TSMC TKY behave
  similarly — regardless of TT_share differences. **This would
  falsify the per-target-macro law as a TT_share-driven phenomenon.**

**We pre-register the prediction: Scenario A holds.** Confidence: HIGH on NYC pair, MED on Tokyo pair (its TT_share lands near the
crossover where the law is most sensitive to noise).

### Summary outcome bands (will be marked MATCH/MISS after runs)

| # | claim | bands | confidence |
|---|---|---|---|
| P1 | Istanbul / Bangkok / SP / NYC-TIST: ≥ 4 attractors | each | HIGH |
| P2 | Istanbul / Bangkok / SP / NYC-TIST: Stage-C ΔF1 ≥ +0.05 | each | HIGH |
| P3 | Tokyo-TIST: ΔF1 ∈ [−0.05, +0.05] | one | MED |
| P4 | All cities: per-macro signs preserved (T&T −, non-T&T +) — Holm-controlled | each | HIGH |
| P5 | Istanbul / Bangkok / SP / NYC-TIST: aggregate Δ B_full−B_blind > 0 | each | HIGH |
| P6 | Tokyo-TIST: aggregate Δ near zero (∈ [−0.01, +0.01]) | one | MED |
| P7 | All cities: ≥ 1 inequity sink at Stage B | each | HIGH |
| P8 | Provenance robustness: NYC-TIST behaviour predicted by its OWN TT_share (0.14), not TSMC's | one | HIGH |
| P9 | Provenance robustness: Tokyo-TIST behaviour predicted by its OWN TT_share (0.45), not TSMC's | one | MED |

These 9 predictions form the **falsifiable backbone** of the multi-city
generalisation claim. After PART-3 runs they will be scored. Anything
that misses must be explained, not silently dropped.

## What is NOT pre-registered

* The specific `situation_id` that hosts the sink (architecture-internal).
* Stage-A K_opt exact value (only the # of attractors, which is more
  robust).
* Absolute R@20 magnitudes (these scale with dataset density; only
  signs and relative bands are claimed).

## Files

* `outputs_multicity/PREDICTIONS_MULTICITY.md` (this file — R8 commit).
* `outputs_multicity/selection/final_cities.md` (post-k-core sizes,
  produced by `experiments/multicity/multicity_size_check.py`).
* `MULTICITY_REPORT.md` (PART-1 + PART-2 narrative at repo root).
