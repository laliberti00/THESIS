# Statistical-validation audit — what we have, what's missing

> Independent review of every quantitative claim in `ROUND3_REPORT.md`
> against the statistical evidence backing it. Companion to
> `pipeline/step04_statistical_validation/STATISTICAL_PROTOCOL.md`
> (the protocol document carried over from phase 2).

## 1. What we already have (with file pointers)

### 1.1 Phase-2 protocol (carried into round 3)

`pipeline/step04_statistical_validation/STATISTICAL_PROTOCOL.md` and
`statistical_validation.py` define:
- Per-user paired Wilcoxon (`scipy.stats.wilcoxon`, `pratt` zero-method,
  two-sided);
- Multi-seed combination via Harmonic Mean P-value (HMP, Wilson 2019);
- Holm step-down family-wise control at α = 0.05;
- Bootstrap percentile CI95 (B = 10 000).

This package is reused by `experiments/round3_a4_statistics.py`.

### 1.2 Per-task statistical evidence

| task | test(s) used | what is significant | output file |
|---|---|---|---|
| **A4** Round-3 stats package | TOST equivalence (δ=0.005 primary, δ=0.0025 sensitivity); paired Wilcoxon + Holm on the three-way comparison; bootstrap CI95 on B_full − B_blind paired diff; A1 op-point bootstrap CI95 | matched-OFF identity equivalent (regression guarantee); X-SAGE κ=0.1 equivalent at δ=0.005, **not** at δ=0.0025 on NYC; B_full−B_blind on NYC: paired CI95 [+0.0039, +0.0259] (excludes 0); on TKY: Wilcoxon p ≈ 3.7e-6 (Holm p=1.1e-5, holm_reject = False under Holm-on-non-equivalent family but the raw effect is significant) | `outputs/{NYC,TKY}/xsage/round3/A4/{tost.json, verdict.json, bootstrap_ci.json, step04_threeway/primary.tsv}` |
| **B7b** local accuracy | paired bootstrap CI95 on per-request ΔR@20 and ΔNDCG@10 in touched subset (B = 10 000) | NYC ΔR@20 CI [−0.0102, +0.0000] (covers 0; marginal); TKY ΔR@20 CI [−0.0166, −0.0076] (excludes 0); both ΔNDCG@10 CIs exclude 0 | `outputs/{NYC,TKY}/<sit_root>/round3/B7b/verdict.json` |
| **B8.3** faithfulness | algebraic-identity test (deterministic, not stochastic) | 100 % match (0 violations on 4 410+33 881 requests, 76 934 LT entries with exact κ-lift, max abs err 2.4e-7 = float32 ε) | `outputs/round3/B8/faithfulness.json` |
| **B8b** name stability | Hungarian-matched cluster pairing across 3 alt seeds {43,44,45}; tiered match rate (strict / intent / time-band) | intent base-token: NYC 0.92, TKY 1.00 | `outputs/round3/B8b/<city>/name_stability_robust.json` |
| **Stage A** ARI cross-seed | adjusted Rand index across 2 seeds | NYC ARI 0.82 (mask); TKY ARI 0.66 | `outputs/{NYC,TKY}/<sit_root>/situations/summary.json` |
| **C6 R8 pre-registration** | predictions committed BEFORE measured runs (causal-inference best practice) | P-iv MATCH; closed-form aggregate predicted to 4 decimals | `PREDICTION_C6.md`, `outputs/round3/C6/OUTCOMES.json` |
| **B6 backbone-agnostic lens** | pairwise Jaccard sink-set overlap (descriptive) | NYC 7/8 models agree on `{6}`; TKY 6/8 agree on `{4,5}` | `outputs/round3/B6/README.md` |
| **Stage D / round-2 X10** matched-pair TOST | TOST + matched-pair on round-2 winner kernels | retained as round-2 close-out evidence | `outputs/{NYC,TKY}/xsage/three_way/` |

### 1.3 Summary of "have we got p-values?"

**Yes**, but they live in three places:
- **A4** — the canonical phase-2-style table: Wilcoxon raw p + Holm
  adjusted p, family of 4 comparisons (X-SAGE vs B_blind, X-SAGE vs
  B_full, R@20, NDCG@20). Already on disk per city.
- **TOST** — `tost.json` reports `p_lower` and `p_upper` for the two
  one-sided tests; "equivalent" iff both < 0.05. Round-3 A4 ran them
  at δ=0.005 (primary) and δ=0.0025 (sensitivity).
- **Bootstrap CI95** — A4, B7b. A 95 % CI that excludes 0 is the
  two-sided p < 0.05 equivalent for symmetric distributions; this is
  the operational p-value test for the per-request paired deltas.

## 2. Gaps — claims currently asserted without a formal test

These are claims in `ROUND3_REPORT.md` that would benefit from
*explicit* significance numbers. Each is a < 30-line code add; nothing
requires retraining.

> **Status update (2026-06-14, B9).** Items 2.1 (C5 ablation) and 2.2
> (C6 per-macro + propagated CI) — both **HIGH** — are **CLOSED** by
> B9 (`experiments/round3_b9_stat_hardening.py`,
> `outputs/round3/B9/…`). See `ROUND3_REPORT.md` § B9.
>
> **Status update (2026-06-14, B10).** Items 2.3 (B7 ratio CI), 2.4
> (B6 permutation), 2.5 (B8b name-stability Wilson), 2.7 (C2 / Stage-C
> McNemar) — **all CLOSED** by B10
> (`experiments/round3_b10_stat_closure.py`,
> `outputs/round3/B10/…`, paper-ready
> `STATISTICAL_VALIDATION_SUMMARY.md`). Item 2.6 (B7 global ΔR@5 CI)
> remains DEFERRED — low priority and the global delta is already a
> known ~0 with negligible practical impact. **The audit is complete
> apart from this LOW deferral.**

### 2.1 (HIGH priority) [**CLOSED — B9.2**] C5 ablation — paired tests on per-request deltas

`experiments/round3_c5_bfull_ablation.py` reports point-estimate R@20
for each ablated variant on TKY (M_full = 0.0309, M_minus_geo = 0.0322,
…, B_blind = 0.0500). Decision "no variant ≥ B_blind" rests on point
estimates only — no CI, no paired Wilcoxon.

**Fix.** For each variant *V*:
```
diff = perr_recall@20_V − perr_recall@20_Bblind   # n=33 881
mean, ci95_lo, ci95_hi = percentile_bootstrap_ci(diff)
w_stat, w_p = paired_wilcoxon(perr_V, perr_Bblind)
```
Apply Holm to the 5 raw p-values (M_full, M_minus_time, _geo, _fine,
_intent). Report which variants reject H₀ vs B_blind. **Expected
outcome:** all 5 differences are negative and Wilcoxon-significant,
strengthening "no recovery under any feature ablation" from a point-
estimate claim to a Holm-controlled rejection.

### 2.2 (HIGH priority) [**CLOSED — B9.1**] C6 per-target-macro deltas — paired tests

C6's P-iv MATCH ("T&T sign negative AND non-T&T sign positive on
TKY_BAL") is reported as a point estimate (T&T −0.0106, non-TT +0.0064)
without a CI or test. The "law" claim is structural; we should show
each per-macro Δ is significantly different from zero.

**Fix.** Two paired Wilcoxon tests (one per stratum) + bootstrap CI95
of the mean per-request delta within each stratum. Holm over the
2 tests. Also fit the closed-form arithmetic prediction with a small
parametric CI: with shares `(s_TT, s_non_TT)` taken as fixed and per-
macro paired SEs from bootstrap, propagate variance to predict the
aggregate's CI; compare to the measured aggregate CI. Same for NYC
and TKY (already in C5.0).

### 2.3 (MED priority) B7 exposure metrics — paired CI95

The flat LT@20 and discounted LT@20 deltas in B7 are point estimates
(+0.6508 / +0.6664 NYC touched, +0.7126 / +0.6868 TKY touched). No CI.

**Fix.** Per-request paired bootstrap on:
- ΔLT@20 (touched)
- ΔdiscLT@20 (touched)
- their ratio (Δdisc / Δflat) — bootstrap CI on the ratio is the
  honest reading of B7's "position-robust" claim.

### 2.4 (MED priority) B6 backbone-agnostic lens — significance of sink agreement

B6 reports pairwise Jaccard between sink sets across 8 recommenders.
Random-baseline reference: under uniform sink assignment (each model
flags ~1.4 sinks out of K=8 ≈ 17.5 %), the expected pairwise Jaccard
is ~0.04 — vs observed ~0.85. The claim "this is structural" would
land harder with the explicit comparison.

**Fix.** Add a permutation test (n=10 000): shuffle sink labels within
each model, recompute pairwise Jaccard mean, p = fraction ≥ observed.
Expected p ≪ 0.001.

### 2.5 (MED priority) Stability of names — formal cross-seed CI

B8b reports point estimates of name-stability rates (NYC intent 0.92,
TKY 1.00) over 3 alt seeds. A binomial CI is straightforward.

**Fix.** Wilson-score CI95 on each rate. With K=8, 3 seeds → n=24
trials per city; NYC intent 22/24 → Wilson CI95 [0.751, 0.985].
This formalises "92 %" as "92 % point, 95 % CI [0.75, 0.99]".

### 2.6 (LOW priority) Head metrics R@5 / NDCG@5 — paired CI

B7 reports global ΔR@5 (−0.0005 NYC / −0.0007 TKY) without CI. To
defend "head-safe globally", a paired bootstrap CI95 confirms the
delta covers 0 at large n_test.

### 2.7 (LOW priority) C2 transit-aware Δ vs `keep` baseline — paired test

`C2_REPORT.md` reports ΔF1 = +0.131 (TKY mask vs keep) for Stage C.
Stage C is per-request macro-F1; a paired McNemar (next-macro
classifier correct vs wrong) gives a hard test.

## 3. Multiple-comparison hygiene status

Currently in use:
- **Holm step-down** inside the A4 three-way family (4 comparisons).
- **Bootstrap CI95** treated as the 2-tailed p < 0.05 surrogate
  elsewhere.

Cross-family corrections are NOT applied. If the paper aggregates
several families (A4, B7b, C5, C6) into a single "is X-SAGE
significant" headline, a Bonferroni-Holm at the family level would be
defensible. For each *separate* claim (e.g. "X-SAGE recovers TKY
provider fairness"), the local Holm is sufficient.

**Recommendation.** Add a §"Statistical-validation summary" to the
paper that lists every per-task family, the chosen α, the correction
used, and the final reject / non-reject verdict. Keep
`STATISTICAL_PROTOCOL.md` as the technical appendix.

## 4. Recommended minimum stat-validation closure for the paper

Order by ROI (smallest code add → largest paper hardening):

1. **C5 ablation Wilcoxon + Holm (5 comparisons)** — closes the TKY
   defensive paragraph with a hard test.
2. **C6 P-iv per-macro Wilcoxon + arithmetic-CI on aggregate** — turns
   the 4-decimal match into a properly-tested law.
3. **B7 LT/discLT ratio CI** — promotes "position-robust" from point
   estimate to interval claim.
4. **B6 Jaccard permutation test** — turns "lens is backbone-agnostic"
   into a permutation-p ≪ 0.001 statement.
5. **B8b name-stability Wilson CI** — formalises the 92 %/100 % point
   estimates.

These 5 additions (each ≤ ~50 LOC, all on existing per-request arrays)
would close every "claim without test" in the round-3 narrative.

## 5. What's deliberately NOT statistically tested (and why)

- **Cluster identity per situation.** ARI is a descriptive
  resampling-style statistic; it isn't given a p-value because the
  null (two random partitions) is uninteresting at the data sizes
  involved (n_train ≥ 32 014).
- **Identity / algebraic checks** (A1bis val-flag exact match, B5
  identity, B8.3 faithfulness). These are deterministic equalities;
  testing them is verification, not inference.
- **C6 pre-registered predictions.** R8 commits the predictions
  BEFORE the runs; the "match / miss" outcome is the test, no
  separate p-value is needed.

## 6. Tooling notes

- `statistical_validation.py` already exposes `paired_wilcoxon`,
  `percentile_bootstrap_ci`, `harmonic_mean_p`, `holm_step_down`.
- All gap fixes can call these directly; no new dependencies.
- Per-request arrays are already saved as `.npz` for X-SAGE / B_blind
  / B_full — just need a small driver per task.
