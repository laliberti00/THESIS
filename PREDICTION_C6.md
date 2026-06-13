# C6 — TKY* balanced counterfactual: pre-registered predictions (R8)

> **R8 commitment**: this file is committed BEFORE any C6 model or stage
> run. Predictions are recorded from the C1 diagnosis, the C5.0
> per-target-macro law, and the C2 transit-aware findings. Outcomes will
> be recorded next to the predictions after the runs complete.

## Construction

From raw TKY data:
1. Per-user uniform random downsampling of Travel & Transport check-ins
   (preserving chronological order of retained events) until global T&T
   share ≈ 25 % (the NYC level).
2. Re-derive all sequence-derived features (dist_prev, gaps, sessions,
   `intent_last_cat`) on the reduced sequences.
3. Run step01 from the reduced raw with the same k-core=10 / 80-10-10
   temporal split protocol.
4. Guard: if k-core 10 drops TKY* below ~800 users, relax the target to
   40 % T&T and record the change before proceeding.

## Predictions

These follow from three earlier results:
* C1: TKY's failures correlate with the 73 % T&T mass; if the
  "city = transit log" diagnosis is causal, reducing T&T should restore
  attractor structure.
* C5.0: B_full hurts T&T targets and helps non-T&T targets, on BOTH
  cities. The aggregate sign of B_full − B_blind is decided by the
  test-pool composition.
* C2: transit-aware mask intent recovers TKY projection by stripping
  T&T from the situation graph. Reducing T&T in the data should achieve
  the same effect WITHOUT any code-level mask.

### P-i (situation machinery, structure)

Under **keep-mode intent** (no mask), the macro graph of TKY* will have
**≥ 4 attractors** (indeg ≥ mean). Rationale: at ~25 % T&T, the macro
mass redistributes across Shop, Food, Outdoors, etc.; the in-degree
mean drops, more macros clear it. Confidence: HIGH.

### P-ii (projection, behaviour)

Under **keep-mode intent** (no mask), Stage-C projection on TKY* will
beat the time-only prior with **macro-F1 ΔF1 ≥ 0** (and probably
≥ +0.05 — between NYC's +0.171 and TKY's −0.041 we predict TKY*
projection sits in NYC-like territory because attractor structure is
restored). Confidence: HIGH if P-i passes; MEDIUM otherwise.

### P-iii (pool law, aggregate sign)

Under round-2 1.1 tuning recipe, **B_full minus B_blind aggregate
ΔR@20 ≥ 0** on TKY* test set. Rationale: at TKY*'s expected ~25 % T&T
test composition (= NYC's), the per-macro signs (B_full hurts T&T,
helps non-T&T) aggregate to positive — same as NYC. Confidence: HIGH.

### P-iv (pool law, per-macro invariance)

Under round-2 1.1 tuning recipe, the per-target-macro signs on TKY*
will be the SAME as on NYC and TKY:

  * B_full minus B_blind on T&T-target requests: **negative**
  * B_full minus B_blind on non-T&T-target requests: **positive**

This is the structural test: the per-macro effect is a property of the
features, not of the city. Confidence: HIGH.

## What the predictions are NOT predicting

* No prediction on absolute R@20 levels. The smaller-than-TKY user
  count after downsampling will move the absolute floor.
* No prediction on Stage-B lens sinks. We will run it as a curiosity
  check: do TKY*'s sinks look like NYC's, or are they situation-id
  arbitrary?
* No prediction on cluster K — let the auto-tune decide.

## Outcomes (to be filled AFTER the runs)

| prediction | outcome | matches? |
|---|---|---|
| P-i: ≥ 4 attractors under keep-mode | TBD | TBD |
| P-ii: Stage-C ΔF1 ≥ 0 | TBD | TBD |
| P-iii: B_full − B_blind aggregate ≥ 0 | TBD | TBD |
| P-iv: T&T sign negative AND non-T&T sign positive | TBD | TBD |

Curiosity reads:
* Stage-A K_opt: TBD
* Stage-B sinks: TBD
* Per-macro deltas in detail: TBD
