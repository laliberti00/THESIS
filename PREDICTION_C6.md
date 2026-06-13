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

## Outcomes (filled AFTER the runs — 2026-06-13)

| prediction | outcome | matches? |
|---|---|---|
| P-i: ≥ 4 attractors under keep-mode | **2 attractors** (Shop & Service, Travel & Transport) | **NO** |
| P-ii: Stage-C ΔF1 ≥ 0 | **ΔF1 = −0.012** (F1_T=0.363 vs F1_time=0.375) | **NO** (but recovered from TKY −0.041 → near-zero) |
| P-iii: B_full − B_blind aggregate ≥ 0 | **−0.0017** (0.0467 − 0.0483) | **NO** (but tiny; see law-check below) |
| P-iv: T&T sign negative AND non-T&T sign positive | **T&T −0.0106, non-T&T +0.0064** | **YES — exact pattern reproduced** |

Curiosity reads:
* Stage-A K_opt = 4, ARI = 1.000 (cluster identical across 3 seeds → very stable)
* Stage-B sinks under keep-mode: **{3}** (1 sink), global LT 6.67 % → verdict **GREEN**
  * compare TKY (keep): {4,5}, GREEN. The single-sink under TKY* keep is closer to NYC keep ({6}) than to TKY keep
* Per-macro deltas in detail (B_full − B_blind, R@20):
  * T&T-target requests:   0.0526 → 0.0420  (Δ = **−0.0106**)
  * non-T&T-target requests: 0.0445 → 0.0509  (Δ = **+0.0064**)
* Test pool composition: T&T share **47.3 %** (NOT 25 % — see law-check)

## Law-check: why P-iii misses despite P-iv matching

The pre-registration anchored P-iii on a 25 % T&T test pool. Per-user
uniform downsample of the RAW data produced 25 % T&T at raw level
(by construction), but the post-processing pipeline (k-core 10 +
chronological last-event test fold) retained T&T-heavy users at higher
rate, lifting test pool T&T share to **47.3 %**.

With per-macro deltas measured at (−0.0106, +0.0064) and test pool
mix 47.3 % T&T / 52.7 % non-T&T, the closed-form aggregate prediction
from the C5.0 law is

    ΔR@20_aggregate = 0.473 · (−0.0106) + 0.527 · (+0.0064)
                    = −0.00501 + 0.00337
                    = **−0.00164**

Measured aggregate = **−0.0017**. The C5.0 law explains the aggregate
to within rounding. P-iii's miss is therefore a miss on T&T-share
control, not on the law. The law itself (P-iv) survives the
counterfactual.

## What the result means for the paper

* **C5.0 per-target-macro law is causally validated.** It survives a
  data-side intervention (P-iv MATCH) AND its arithmetic prediction
  of the aggregate is correct to four decimals.
* **TKY's projection failure is partially recovered by less T&T mass.**
  ΔF1 climbed from −0.041 (TKY hard) to −0.012 (TKY* keep), but stayed
  negative. So T&T mass is necessary but not sufficient: the *structural*
  attractor property of T&T persists at 25 % raw (P-i miss). Reading: T&T
  is structurally an attractor in the TKY user-behaviour graph, not a
  count artefact.
* **A "balance the city" data-side intervention does NOT make B_full
  win.** Under TKY*, B_full still underperforms B_blind on aggregate
  R@20 — the pool composition control needed is sharper than what
  k-core preserves. The defensive narrative for TKY ("B_full hurts in
  exactly the situations T&T dominates") stands.
* **Stage-B lens stays GREEN.** No fairness regression introduced by the
  intervention.
