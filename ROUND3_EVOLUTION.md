# X-SAGE — Round-3 evolution (sessions 1 → 4)

> Snapshot of the round-3 work on `step02b-round3`, from `4f69b52`
> (round-2 close-out) to `e120d0d` (B8b). Companion to `ROUND3_REPORT.md`
> (full per-task detail). This document is the **executive evolution
> log** — what changed about the paper claims, in what order, why.

## TL;DR — the round-3 narrative arc

X-SAGE entered round 3 with two unresolved tensions:
1. **TKY underperformance.** On TKY, B_full < B_blind; the situational
   re-ranking carried a small accuracy cost; transit-mass (T&T 73 %)
   was suspect.
2. **Capability claims partially documented.** Audit/correction/
   projection had numbers; explainability had an argument but no
   empirical verification.

Round 3 closed both:

- **Pre-registered causal validation of the per-target-macro law.** C5.0
  (B_full hurts T&T-target requests, helps non-T&T) is no longer a
  correlation; under the C6 TKY* counterfactual (per-user T&T
  downsample), the law's signs survive AND its arithmetic prediction of
  the aggregate ΔR@20 matches the measured number to four decimals.
- **The fairness lens is backbone-agnostic.** Across 8 floor
  recommenders (B6), 7–8 flag the same sinks per city — inequity is
  structural to the city × situation pair, not a model artefact.
- **The exposure gain is position-effective AND honest about local
  cost.** B7 ratio (discounted-LT / flat-LT) = 1.02 NYC / 0.96 TKY;
  LT@k curves separate at k=3; provider universe widens by 5 %–23 %.
  B7b restates the accuracy claim: global ~0 cost, local small-and-
  bounded cost (ΔR@20 −0.0034 NYC / −0.0121 TKY on touched lists,
  identical to the global-rerank's per-list cost — so X-SAGE's
  contribution is *selectivity*, not a cheaper boost).
- **Explainability is intrinsically faithful, verified.** B8 verifies
  the algebraic identity `score_on − score_blind = κ·G1` on 76 934
  long-tail entries across both cities (max abs err 2.4 × 10⁻⁷ =
  float32 ε). B8b makes the auto-generated names robust by
  construction (intent stability 92 % NYC / 100 % TKY across 3 alt
  seeds), and reframes TKY's repetitive labels as the structural-
  poverty signature, not a naming defect.

User-side fairness gain (B4) and the dataset-scouting / R8 pre-
registration scaffolding (D1, PREDICTION_C6) complete the round.

## Chronological evolution by session

### Session 1 — `4f69b52` … `f51b6c0` (round-3 launch)

| commit | task | result |
|---|---|---|
| `4f69b52` | round-2 close-out (reference) | base of round 3 |
| early | A1 / A1bis | NYC mask sinks `{6,7}`: val-flagged exactly match test-flagged → not data-snooping |
| A3-recheck on TKY mask | situation lens unchanged | mask doesn't change the sink set, only re-arranges items |
| A4 | **statistical package** | TOST equivalence (δ=0.005, δ=0.0025), bootstrap CI95 on B_full−B_blind, A1 op-point adjudication, Wilcoxon + Holm step-down on the three-way comparison |

The shape of the round shifted at A4: matched-OFF identity proved
**TOST-equivalent at both δ thresholds on both cities** (the matched-
OFF is empirically X-SAGE with κ=0 → it must coincide with B_blind,
and TOST confirms; this is a regression guarantee, not a discovery).
The interesting TOST is X-SAGE-κ=0.1 vs B_blind: TOST-equivalent at
the primary δ=0.005, NOT equivalent at the sensitivity δ=0.0025 on
NYC (so the "near-zero cost" claim holds at 5 mille-recall, not at
2.5 mille-recall).

### Session 2 — `64538e7` … `509d88d` (TKY narrative)

| commit | task | result |
|---|---|---|
| `b1c70e6` | B3bis | descriptor degeneracy reconciliation — Mahalanobis vs ε_eff |
| `0747ffc` | B1 | eq.18 boundary disambiguation, ARI vs ε* curve |
| `64538e7` | C5.0 | **stratified accuracy read** — discovered the per-target-macro law |
| `6dc9c5e` | C2 | transit-aware intent (`mask`) recovers TKY projection ΔF1 +0.131, lens cost +0.02 LT |
| `509d88d` | A3-recheck on TKY mask | mask sinks differ from keep sinks |

C5.0 was the structural surprise: **on both cities, B_full hurts T&T-
target requests and helps non-T&T requests**. TKY's aggregate loss is
the per-macro × pool-composition law firing — not a tuning miss.

### Session 3 — `8ee0aa9` … `4bf3428` (appreciation + causal test)

| commit | task | result |
|---|---|---|
| `8ee0aa9` | A1bis NYC mask | val ⊇ test sinks {6, 7} exactly → no leakage |
| `314875d` | B5 anatomy | NYC 6.7 % touched (identity PASS); TKY 15.1 % touched (58 sub-threshold) |
| `985e331` | B6 lens backbone-agnostic | sinks identical across 7–8 of 8 floor recommenders per city |
| `e817cdd` | C5 B_full ablation | no single-axis ablation recovers B_blind on TKY |
| (inline) | B4 profiling + user-side fairness | TKY Gini −0.092, NYC unchanged |
| `e641087` | D1 dataset memo | TIST2015 primary, Last.fm-1K future; Frappe rejected with cause |
| `b29cb1b` | **C6 PREDICTION (R8)** | 4 predictions pre-registered BEFORE measured runs |
| `4bf3428` | **C6 OUTCOMES** | P-iv MATCH; aggregate predicted to 4 decimals from per-macro × pool |

**B6 is a paper headline**: a fairness lens that flags the same sinks
across 8 backbones is structural, not model-specific. **C6 is the
other paper headline**: C5.0 graduates from correlation to causation
via a per-user T&T downsample counterfactual; the law's prediction
`Δagg = 0.473·(−0.0106) + 0.527·(+0.0064) = −0.00164` matches measured
−0.0017 (4-decimal agreement).

### Session 4 — `7414842` … `e120d0d` (exposure + explainability)

| commit | task | result |
|---|---|---|
| `7414842` | B7 position-discounted exposure | ratio disc/flat = 1.02 NYC / 0.96 TKY; sep_k=3; provider universe +5 %/+23 % |
| `50d6f16` | B7b local accuracy in touched | ΔR@20 touched −0.0034 NYC / −0.0121 TKY; identical to global-rerank's per-list cost |
| `6a128ef` | B8 explainability + faithfulness | 100 % faithfulness on 76 934 LT entries; auto-named situations; time-band 88–100 % stable |
| `e120d0d` | B8b robust intent token | intent stability NYC 38 → 92 %; TKY 25 → 100 %; TKY's perfect stability *is* the structural-poverty evidence |

This session closed the two reviewer attack surfaces:
- **B7 / B7b**: "Your LT@20 is rank-blind and your `cost ≈ 0` is a
  dilution artefact" → answered with discounted-LT, exposure-at-k=3,
  and touched-only bootstrap CIs.
- **B8 / B8b**: "Your explanation is plausible but not faithful and
  the names are interpretive" → answered with algebraic-identity
  verification (100 %, 0 violations) and a deterministic naming rule
  whose intent token is invariant to argmax flips by construction.

## What the paper headlines look like now

1. **Backbone-agnostic fairness lens.** Same sinks across 7–8 of 8
   floor recommenders → inequity is structural to city × situation,
   not a model property.
2. **C5.0 per-target-macro law is causal.** Survives a data-side
   counterfactual (P-iv MATCH) AND its arithmetic prediction matches
   measured to 4 decimals (P-iii law-check).
3. **Sink re-ranking delivers effective exposure** (discounted ratio
   1.02 NYC / 0.96 TKY; separation k=3) **with bounded local cost**
   (touched-only ΔR@20 −0.0034/−0.0121, identical to global-rerank's
   per-list cost → X-SAGE's contribution is *selectivity*).
4. **User-side fairness gain on TKY**: long-tail-received Gini −0.092
   at the A4-quantified accuracy cost.
5. **Empirically-verified faithful explainability for the situational
   component.** Algebraic-identity verification, 100 % match on
   76 934 LT entries; deterministic naming, intent token 92 % / 100 %
   stable across seeds.

## Open items for session 5+

- **B2**: CST attribute weights (re-tune Stage A with weighted distance;
  report weight stability across L1 tree seeds). Ordering constraint
  (C2 decision) is satisfied.
- **D2**: TIST2015 third-city dry run with R8 pre-registration
  (predictions before measured runs).
- **Stat hardening** (see `STATISTICAL_VALIDATION_AUDIT.md`).
