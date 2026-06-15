# Statistical-validation summary (X-SAGE round 3)

> Paper-ready enumeration of every statistical-test family used in this round, with the within-family correction applied and the cross-family policy stated explicitly. Backbone of the paper's "Statistical validation" subsection.

## 1. Test families

| family | city / scope | test | n | within-family correction | headline | p / CI |
|---|---|---|---:|---|---|---|
| A4 three-way TOST | NYC + TKY | TOST (two one-sided) | 6 | δ-primary + δ-sensitivity | matched-OFF identity equivalent at δ=0.005 AND δ=0.0025; X-SAGE κ=0.1 equivalent at δ=0.005, NOT at δ=0.0025 on NYC | p_lower / p_upper, both < 0.05 ⇒ equivalent |
| A4 three-way Wilcoxon | NYC + TKY | paired Wilcoxon | 4 | Holm step-down | NYC X-SAGE vs B_full: Holm p=7.4e-4; TKY: Holm p=1.1e-5 | Holm-adjusted p < 0.05 on B_full comparisons; B_blind null |
| A4 paired bootstrap CI | NYC + TKY | percentile bootstrap CI95 (B=10 000) | 2 | n/a (interval estimates) | NYC ΔR@20 B_full−B_blind = +0.0150 CI95 [+0.0039, +0.0259] (excludes 0) | CI95 |
| B7b touched-subset accuracy | NYC + TKY | paired bootstrap CI95 (B=10 000) | 4 | n/a (interval estimates) | TKY ΔR@20 touched CI95 [−0.017, −0.008] excludes 0; NYC ΔR@20 [−0.010, +0.000] marginal; both ΔNDCG@10 exclude 0 | CI95 |
| B9.1 per-target-macro paired tests | NYC + TKY + TKY_BAL | paired Wilcoxon + percentile bootstrap CI95 | 9 | Holm step-down (per city, 3 strata) | C6 promotes BOTH halves of the law to Holm-reject simultaneously on TKY_BAL (T&T p_Holm=2.5e-6, non-T&T p_Holm=8.2e-3); NYC + TKY each have one Holm-rejected stratum + the aggregate | Holm-adjusted p; Wald CI from per-stratum reconstructs aggregate bootstrap CI to 5 decimals |
| B9.2 C5 ablation | TKY | paired Wilcoxon | 5 | Holm step-down | 5/5 ablation variants Holm-reject vs B_blind at α=0.05; best (M_minus_geo) p_Holm=8.0e-6 | Holm-adjusted p < 0.05 on all variants |
| B10.1 lens-permutation | NYC | permutation (preserving |sinks|, n=10000) | 1 | n/a (single test per city) | observed mean pairwise Jaccard = 1.000; null mean = 0.125; p = 0.000100 | p (one-sided) = 0.000100 |
| B10.1 lens-permutation | TKY | permutation (preserving |sinks|, n=10000) | 1 | n/a (single test per city) | observed mean pairwise Jaccard = 0.857; null mean = 0.221; p = 0.000100 | p (one-sided) = 0.000100 |
| B10.3a B7 disc/flat ratio | NYC | percentile bootstrap CI95 on per-request ratio (B=10 000) | 1 | n/a (interval estimate) | ratio point 1.024 CI95 [1.014, 1.034]; brackets 1? False | CI95 |
| B10.3a B7 disc/flat ratio | TKY | percentile bootstrap CI95 on per-request ratio (B=10 000) | 1 | n/a (interval estimate) | ratio point 0.964 CI95 [0.962, 0.966]; brackets 1? False | CI95 |
| B10.3b B8b name-stability Wilson | NYC [strict] | Wilson score CI95 (binomial) | 1 | n/a (interval estimate) | 9/24 = 0.375  CI95 [0.212, 0.573] | Wilson CI95 |
| B10.3b B8b name-stability Wilson | NYC [intent] | Wilson score CI95 (binomial) | 1 | n/a (interval estimate) | 22/24 = 0.917  CI95 [0.742, 0.977] | Wilson CI95 |
| B10.3b B8b name-stability Wilson | NYC [time] | Wilson score CI95 (binomial) | 1 | n/a (interval estimate) | 21/24 = 0.875  CI95 [0.690, 0.957] | Wilson CI95 |
| B10.3b B8b name-stability Wilson | TKY [strict] | Wilson score CI95 (binomial) | 1 | n/a (interval estimate) | 1/18 = 0.056  CI95 [0.010, 0.258] | Wilson CI95 |
| B10.3b B8b name-stability Wilson | TKY [intent] | Wilson score CI95 (binomial) | 1 | n/a (interval estimate) | 18/18 = 1.000  CI95 [0.824, 1.000] | Wilson CI95 |
| B10.3b B8b name-stability Wilson | TKY [time] | Wilson score CI95 (binomial) | 1 | n/a (interval estimate) | 12/18 = 0.667  CI95 [0.437, 0.837] | Wilson CI95 |
| B10.3c Stage-C / C2 McNemar | NYC_keep | paired McNemar (continuity-corrected χ²₁) | 1 | n/a (single per config) | ΔF1 = +0.171; T-only correct 1203, time-only correct 826; McNemar p = 1.11e-16 | p = 1.11e-16 |
| B10.3c Stage-C / C2 McNemar | NYC_mask | paired McNemar (continuity-corrected χ²₁) | 1 | n/a (single per config) | ΔF1 = +0.176; T-only correct 927, time-only correct 1012; McNemar p = 5.64e-02 | p = 5.64e-02 |
| B10.3c Stage-C / C2 McNemar | TKY_keep | paired McNemar (continuity-corrected χ²₁) | 1 | n/a (single per config) | ΔF1 = -0.041; T-only correct 1591, time-only correct 9625; McNemar p = 0.00e+00 | p = 0.00e+00 |
| B10.3c Stage-C / C2 McNemar | TKY_mask | paired McNemar (continuity-corrected χ²₁) | 1 | n/a (single per config) | ΔF1 = +0.006; T-only correct 6776, time-only correct 8984; McNemar p = 0.00e+00 | p = 0.00e+00 |

## 2. Cross-family correction policy

**No cross-family multiple-comparison correction is applied.** Rationale: each family above tests a *distinct hypothesis* (equivalence of matched-OFF; three-way accuracy; touched-subset local cost; per-target-macro law; ablation vs blind; lens permutation; exposure ratio; name stability; projection vs clock). Within-family Holm step-down controls family-wise Type-I error at α = 0.05 for each. Pooling these into a single family would conflate independent questions and inflate the correction unnecessarily. This is the standard rationale (see Rubin 2017, *Stat. Sci.*); the trade-off is full transparency of test inventory, which this table provides.

## 3. Deterministic verifications (NOT statistical tests)

These claims are exact identities or pre-registered match-or-miss outcomes; they correctly carry NO p-value:

* **Matched-OFF identity** — X-SAGE with κ=0 must equal B_blind by construction; the A4 TOST is a regression guarantee, not inference.
* **B5 touched-share identity** — `touched == sink ∩ core` holds by construction on NYC (PASS exact); the 58 TKY deviations are sub-threshold core-sinks where the additive boost is insufficient to displace any popular item — *property* of the additive rule, not a bug.
* **B8.3 faithfulness** — `score_on − score_blind = κ·G1` is an algebraic identity. Empirically verified: 0 violations on 76 934 long-tail entries (max abs err = 2.4 × 10⁻⁷ = float32 ε). This is *verification* of the identity, not inference.
* **A1 / A1bis val=test sink-set match** — NYC mask sinks {6, 7} are exactly val-flagged and test-flagged; the equality is set-membership exact, not a statistical claim.
* **C6 R8 pre-registration outcomes** — the predictions in `PREDICTION_C6.md` were committed BEFORE the measured runs; the four outcomes (3 MISS + 1 MATCH, with the MATCH on the structural law P-iv) are pre-specified match/miss results, not post-hoc tests.

These should NOT be reported with synthetic p-values: doing so would be a category error (a deterministic 100 % is *not* p < 10⁻⁶, it is identity).

## 4. Tooling

* `pipeline/step04_statistical_validation/statistical_validation.py` (phase-2 protocol package) — paired Wilcoxon, bootstrap CI95, HMP combination, Holm step-down. Re-used unchanged.
* `experiments/round3_a4_statistics.py` (A4 stat suite).
* `experiments/round3_b7b_local_accuracy.py` (paired bootstrap).
* `experiments/round3_b9_stat_hardening.py` (per-macro paired Wilcoxon + Holm + propagated CI; C5 ablation paired tests).
* `experiments/round3_b10_stat_closure.py` (permutation test, ratio CI, Wilson CI, McNemar collection, this summary).
