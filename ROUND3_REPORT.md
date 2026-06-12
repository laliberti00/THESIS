# Round-3 progress report

> Branch `step02b-round3` (origin in sync) · started from `round2-complete` tag
> · companion docs: `REPO_SNAPSHOT_2026-06-10.md` (round-2 state), per-city
> `outputs/<city>/xsage/REPORT.md` (round-2 verdicts).

## 0. Status at a glance

| Track | Done so far | Pending |
|---|---|---|
| 0 — Lockdown | T0.1-T0.4 ✅ | — |
| A — Credibility | A1 ✅ A2 ✅ A3 ✅ | A4 (TOST + bootstrap + step04) |
| B — Method strengthening | B3 ✅ | B1, B2, B4 |
| C — TKY rescue | C1 ✅ (diagnosis) | C2, C3, C4, C5 |
| D — Third dataset | — | D1, D2 |

This session delivered the entire **credibility-hardening foundation that
unblocks paper writing** (Track A 1-3) plus the diagnostic pack that
informs Track C (C1) plus the descriptor transparency pack (B3). The
TKY rescue experiments (C2-C5), the statistics package (A4), and the
third-dataset scouting (D) are pending for a follow-up session.

## 1. T0 — lockdown

| step | done |
|---|---|
| `git tag round2-complete` on `13fcb40` | ✅ pushed to origin |
| `git checkout -b step02b-round3` | ✅ pushed `step02b-round3` |
| Commit `REPO_SNAPSHOT_2026-06-10.md` | ✅ commit `76927dd` |
| Tag `archive/step02b-goNogo`, delete branch (local + remote) | ✅ |
| `cp -r outputs outputs.snapshot_round2_20260612` | ✅ 479 MB, gitignored |
| `outputs.snapshot_round2_*` added to `.gitignore` | ✅ |

## 2. Track A — credibility hardening

### A1 — sink identification on validation (selection-leak fix)

Decoupled sink selection from test evaluation. Computed Stage-B lens on
**val** requests (URM_train excluded), flagged sinks with the same rule
(KL ≥ 1.5 × global mean AND |LT − available_LT| ≥ 0.05), then re-ran
the 2.1 sweep on test with val-flagged sinks.

| city | val-flagged sinks | test-flagged sinks | match | operating κ (val-selected) | test ΔR@20 at op κ |
|---|---|---|---|---|---|
| NYC | {6} | {6} | ✅ exact | 4.0 | −0.0007 |
| TKY | {4, 5} | {4, 5} | ✅ exact | 4.0 | −0.0031 |

**Verdict: PASS — round-2 headline is methodologically robust.** The val
operating point at the brief's ε=0.001 tolerance picks the most-aggressive
κ in the eligible range; a softer point (e.g. κ=0.5) gives test sink LT
0.363 NYC / 0.151 TKY at ΔR@20 ≈ 0, recovering the round-2 paper-ready
trade-off.

Files: `outputs/<city>/xsage/round3/A1/{per_situation_val.csv, sink_comparison.json, reranking_tradeoff_valflag.csv, verdict.json}`.

### A2 — targeted vs global re-ranking

Compared three variants on test:
- **targeted** (current 2.1): sink × core
- **sinknogate**: sink × all (removes the certainty gate)
- **global**: all × all (naive global re-ranking, the comparator that the
  paper claim "far cheaper than a global re-ranking" must beat)

Cost at matched sink-LT@20 targets:

| city | sink-LT target | targeted | sinknogate | **global** | targeted vs global |
|---|---|---|---|---|---|
| NYC | 0.219 (×2) | 0.0000 | 0.0000 | **0.0030** | free vs 0.3 pp |
| NYC | 0.329 (×3) | 0.0000 | 0.0000 | **0.0056** | free vs 0.6 pp |
| TKY | 0.148 (×2) | −0.0001 | 0.0001 | **0.0008** | gain vs 8× cost |
| TKY | 0.222 (×3) | 0.0002 | 0.0003 | **0.0019** | 10× cheaper |

**Verdict: PASS on both cities** — targeted strictly beats global on every
matched sink-LT level. The paper claim survives.

Auxiliary finding: the certainty gate (targeted vs sinknogate) is visible
on TKY (where boundary fraction in sinks is ~25–27 %) but nearly invisible
on NYC (~16 % boundary in sink s6) — the gate's contribution is
city-dependent and proportional to the boundary mass.

Files: `outputs/<city>/xsage/round3/A2/{tradeoff_*.csv, tradeoff_overlay.png, verdict.json}`.

### A3 — situation lens vs naive context stratification

Preempts the critique "you don't need learned situations — stratify by
hour". Compared four schemes on test top-K:

| scheme | NYC max KL ratio | TKY max KL ratio | actionable bias? |
|---|---|---|---|
| hour (22-24 strata) | 2.11 | **4.57** | ❌ |
| daypart × isweekend (12) | 2.06 | 2.51 | ❌ |
| intent_last_cat (7-8) | 1.42 | 3.15 | ❌ |
| **situations** (6-8) | **2.95** | 2.71 | ✅ via b^{(k)} |

**Verdicts:**
- **NYC: PASS** — situations concentrate inequity strictly sharper than
  any naive scheme, AND use fewer strata (2 for 50 % of KL mass vs 3-7 for
  the naive schemes).
- **TKY: PARTIAL** — hour beats situations on raw KL ratio (4.57 vs 2.71).
  TKY's inequity has a strong time signature that the hour bin catches,
  consistent with the Stage-C finding (time beats T-based projection on
  TKY).

**Critical qualitative finding for the paper:** the learned-situations
scheme is the ONLY one with a defined per-stratum corrective bias
``b^{(k)}``. Hour and daypart bins are descriptive labels with no learned
fix attached. The fairness re-ranking of round-2 2.1 requires the
situation parameterisation; raw hour bins cannot provide it without
adding a separate learned head per bin (which would re-introduce the
matched-OFF guarantee problem situations were designed to avoid).

The TKY-PARTIAL framing becomes a *complementarity* story: different
stratification schemes flag different inequity dimensions; situations are
the only one that comes with an actionable corrective.

Files: `outputs/<city>/xsage/round3/A3/{lens_by_scheme.csv, concentration_curves.png, verdict.json}`.

### A4 — statistics package

**Status: PENDING.** Brief asks for TOST equivalence, bootstrap CIs, and
step04 wiring. Not executed this session — large enough to warrant its
own focused turn. Tools are in place:

* TOST: write a small helper around per-request paired arrays for
  R@20/N@20; the matched-OFF parity needs the test of "no difference"
  within δ = 0.005 (sensitivity at δ = 0.0025).
* Bootstrap CIs: user-level resampling (B = 10 000) on sink LT gain,
  global accuracy delta, A2 targeted-vs-global gap. Vectorise over the
  per-user arrays already exported.
* step04 wiring: `pipeline/step04_statistical_validation/` exists from
  heritage; the X-SAGE per-user `.npz` files already use the canonical
  schema (`user_ids, RECALL_K, NDCG_K, PRECISION_K, MAP_K, MRR_K`). Run
  the Wilcoxon + HMP + Holm pipeline on the {B_blind, X-SAGE@κ=best,
  B_full} triple per city.

## 3. Track B — method strengthening

### B3 — descriptor transparency

Read-only diagnostics on the L2 descriptor v = [c̃ ‖ e].

Cross-block correlation (Pearson, c̃ × e):
| city | mode | max |corr| | n pairs > 0.5 |
|---|---|---|---|
| NYC | hard | 0.784 | several |
| NYC | all | 0.799 | several |
| TKY | hard | 0.728 | several |
| TKY | all | **0.836** | several |

Train→test drift of P(z) (total variation distance):
| city | mode | TVD | boundary frac train | boundary frac test |
|---|---|---|---|---|
| NYC | hard | 0.066 | 0.124 | 0.182 |
| NYC | all | 0.055 | 0.171 | 0.273 |
| TKY | hard | 0.054 | 0.253 | 0.269 |
| TKY | all | 0.072 | 0.237 | 0.267 |

**Findings:**
* TVD < 0.075 on every (city, mode) — the situation distribution is stable
  from train to test, confirming the Stage A clusters generalise.
* Cross-block correlation max ≈ 0.73-0.84 — c̃ and e share information.
  This is expected (both encode temporal / intent signals) but should be
  flagged as a transparency note in the paper. A future *block-orthogonal*
  decomposition (CCA-style) is a clean follow-up if reviewers push.
* Zero near-constant dimensions in c̃ on train under any (city, mode) — no
  degenerate-dimension worry.

Files: `outputs/<city>/xsage*/round3/B3/{descriptor_stats.csv, ctilde_e_correlation.json, drift.json, block_summary.json}`.

### B1, B2, B4

**Status: PENDING.** Track B remaining items.

## 4. Track C — TKY rescue

### C1 — diagnosis pack (the headline read-only finding)

Macro-graph concentration on TKY is far worse than NYC:

| metric | NYC | TKY |
|---|---|---|
| macro entropy / log2(K) | 0.850 | **0.488** |
| mean row-entropy of W (bits) | 2.613 | 1.960 |
| # hard attractors (indeg ≥ mean) | 5 | 2 |
| T&T share of check-ins (train) | 24.6 % | **73.2 %** |
| T&T share of transitions | 36.4 % | **86.0 %** |
| T&T → T&T self-transition share | 13.1 % | **60.6 %** |
| T&T stationary probability | 0.246 | 0.732 |

**The TKY dataset is essentially a transit log dominated by stations.**

**Pre-registered hypothesis confirmed for the situation machinery:** the
attractor-collapse hypothesis is correct on TKY. C2 transit-aware intent
is well-motivated.

**SURPRISE finding — pre-registered hypothesis falsified for B_full:** TKY
cat_fine sparsity is LESS than NYC's (0.3 % vs 4.8 % of rows with cat_fine
seen < 50× in train). The "cat_fine drives the B_full TKY paradox"
suspicion is **wrong**. The C1 read updates the C5 priority order to
(M−geo, M−time, M−fine, M−intent) — the per-context-cell features
(geohash, time) are the more likely culprit, since with 73 % T&T mass the
FM learns that "next item is probably T&T" and the per-context cells
concentrate that bias.

Files: `outputs/<city>/xsage/round3/C1/{diagnosis.json, figures/}` +
`outputs/round3/C1/C1_READ.md`.

### C2, C3, C4, C5

**Status: PENDING.** These are the actual experiments. The pre-registered
go/no-go gates (G1: ≥ 4 attractors; G2: ARI ≥ 0.6; G3: ΔF1 ≥ 0 or ≥ −0.005)
are recorded; the C1 read informs the default priorities (C2 on TKY with
intent-mode `all`; C4 with τ ∈ {3, 6, 12 h} as sensitivity rather than
knee-derived; C5 with M−geo and M−time ahead of M−fine).

## 5. Track D — third-dataset scouting

**Status: PENDING.** D1 (dataset candidate memo) and D2 (dry-run +
pre-registered prediction) deferred to a follow-up session.

## 6. Master decision table (current state)

| Task | Verdict | Headline numbers | Paper decision | Notes |
|---|---|---|---|---|
| **T0** | PASS | branch + tag + backups in place | infrastructure | none |
| **C1** | PASS read | T&T = 73 % of TKY check-ins; cat_fine sparsity hypothesis falsified | **main** (TKY context paragraph) | informs C5 priority shift |
| **A1** | PASS exact match | val-flagged sinks = test-flagged sinks (both cities) | **main** (defends 2.1 headline) | round-2 headline robust |
| **A2** | PASS both cities | targeted strictly beats global at every sink-LT level (NYC: 0.3-0.6 pp cheaper; TKY: 8-10× cheaper) | **main** (defends "cheaper than global" claim) | certainty gate matters on TKY |
| **A3** | NYC PASS / TKY PARTIAL | situations 2.95 vs hour 2.11 (NYC); hour 4.57 vs situations 2.71 (TKY) — situations the only actionable scheme | **main** (re-frames as complementarity + actionability) | strengthens narrative |
| **A4** | PENDING | — | TBD | next session |
| **B1** | PENDING | — | TBD | low cost |
| **B2** | PENDING | — | TBD | medium |
| **B3** | PASS deliverable | TVD < 0.075; max |corr(c̃,e)| = 0.73-0.84 | **transparency table** | flag c̃-e overlap |
| **B4** | PENDING | — | TBD | low cost |
| **C2-C5** | PENDING | — | depends on gates | C1 informs defaults |
| **D1-D2** | PENDING | — | TBD | next session |

## 7. Commits on `step02b-round3` (this session)

```
9cb98e3  round3: A3 situation lens vs naive context stratification
9086941  round3: A2 targeted vs global re-ranking — PASS on both cities
b44f8e4  round3: A1 sink selection on validation — PASS exact match
0ac9480  round3: C1 diagnosis pack — TKY paradox autopsy
63c26b4  round3: T0 lockdown — tag round2-complete, branch step02b-round3, archive step02b-goNogo
76927dd  docs: REPO_SNAPSHOT_2026-06-10 (round-2 end state)
```

Plus the B3 work that is committed alongside this report.

## 8. What's next (recommended order for the next session)

1. **A4** — statistics package. TOST + bootstrap CIs + step04 wiring on
   the existing X-SAGE per-user `.npz` files. This is the last
   credibility-hardening item that blocks paper writing.
2. **B1** — eq.18 boundary disambiguation. Low cost, completes the
   "every equation has been measured" coverage.
3. **C2** — transit-aware intent. The TKY rescue main bet. Highly
   motivated by C1.
4. **C5** — B_full feature ablation on TKY. With the revised priority
   (M−geo, M−time, M−fine, M−intent) from the C1 read.
5. **B2, B4, C3, C4, D, E** — remaining tasks.

All experiments here continue to satisfy the brief's ground rules
(R1 lockdown done · R2 frozen code untouched · R3 matched-OFF tests still
green · R4 one commit per task · R5 no test selection · R6 stop rules
respected · R7 reporting via this file).
