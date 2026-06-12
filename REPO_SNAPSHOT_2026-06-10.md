# Repo snapshot — `IntentAwareRS_thesis`
**Date:** 2026-06-10  •  **Branch:** `step02b-xsage`  •  **HEAD:** `13fcb40`
**Remote:** `https://github.com/laliberti00/THESIS.git` (in sync with origin)

> Hand-off document for an incoming review chat. Includes everything needed
> to navigate the codebase, reproduce the experiments, and understand the
> findings. Copy the whole file into a fresh session if you want full context.

---

## 0  Repo identity

| | value |
|---|---|
| Local path | `/Users/lucaaliberti/Downloads/IntentAwareRS_thesis/` |
| Origin | `laliberti00/THESIS` (GitHub) |
| Active branch | `step02b-xsage` (origin/step02b-xsage in sync) |
| Active HEAD | `13fcb40 — step02b-xsage: round-2 Track 3 (explainability artefacts)` |
| Python | 3.11.0 (.venv) on Apple Silicon M2 (MPS available) |
| Tests | `pytest tests/ -q` → **19/19 passing** |

### Branches

| branch | head | role |
|---|---|---|
| `main` | `d2a5d1c` | heritage (Shehzad+Phase2). Not touched. |
| `phase2-baselines-and-validation` | `d2a5d1c` | heritage. |
| `gowalla-dataset-comparison` | `e5167b2` | older audit (Gowalla). |
| `foursquare-rebuild` | `d09c71b` | step01 + step02a floor only. **Pristine reference state**. |
| `step02b-goNogo` | `1043b49` | step02b first attempt (bilinear gate). **Closed RED on both cities, archived.** |
| **`step02b-xsage`** | **`13fcb40`** | **active branch — X-SAGE work + round-2 push.** |

### Tags

| tag | commit | meaning |
|---|---|---|
| `thesis-phase1-baseline` | early | initial baseline snapshot |
| `phase2-baselines-and-validation` | `d2a5d1c` | end of Phase 2 |
| `floor-complete` | `d09c71b` | CF floor done on both cities |
| `pre-xsage-rollback` | `d09c71b` | **rollback anchor before round-2 X-SAGE** |

### Recent commits (newest first, this branch)

```
13fcb40  step02b-xsage: round-2 Track 3 (explainability artefacts)
dc1dd28  step02b-xsage: round-2 2.1 (situation-gated long-tail re-ranking)
da5f38c  step02b-xsage: round-2 1.4 (richer intent descriptor)
80a50b0  step02b-xsage: round-2 1.1 (B_full tuning) + 1.5 (additive combiner)
ad5240b  step02b-xsage: Stage D (three-way + modulation log)
1062baa  step02b-xsage: Stage C (projection L3, cardinal check 2)
d7a91cd  step02b-xsage: Stage B (fairness lens, cardinal check 1)
cf2593c  step02b-xsage: scaffold + Stage A (situations on NYC and TKY)
d09c71b  step02a: CF baseline floor (8 context-blind models, NYC + TKY)
5fec220  step01: upgrade Foursquare taxonomy → 99.5 % coverage on TSMC2014
dd4e6f4  step01 preprocessing: Foursquare TSMC2014 → dual view (NYC + TKY)
7c68459  Clean rebuild scaffold for the new Foursquare single-phase plan
```

---

## 1  The 4-block flow — status at a glance

```
step01_preprocessing  →  step02_models  →  step03_evaluation  →  step04_statistical_validation
        ✅                  ✅ floor +           🟡 (heritage:           🟡 (heritage:
                            ✅ X-SAGE            statistical             Wilcoxon + HMP
                                                 audit only)             + Holm + bootstrap)
```

* **step01** — done, frozen, both cities.
* **step02a** — CF floor done, both cities, 8 baselines tuned + per-user `.npz` exported.
* **step02b** — **X-SAGE** layer implemented + round-2 push (this is the active work).
* **step03 / step04** — present in the codebase as heritage; the `.npz` schema is the bridge — every model on this branch exports the same schema so step04 plugs in unchanged. Not exercised in the X-SAGE phase yet.

---

## 2  Dataset (step01) — frozen

| | NYC | TKY |
|---|---|---|
| Source | `data/raw/dataset_TSMC2014_NYC.txt` | `data/raw/dataset_TSMC2014_TKY.txt` |
| Encoding | `latin-1` (Café / accented venue names) | idem |
| Time | local = UTC + `tz_offset` | idem |
| Filter | iterative k-core (k=10) on users & items | idem |
| Split | per-user temporal 80 / 10 / 10 (train < val < test by `time_local`) | idem |
| Cold filter | users/items only in val/test removed | idem |
| Users / items / macros | **829 / 1 088 / 9** | **2 214 / 2 852 / 8** |
| train nnz | 17 472 | 85 251 |
| val nnz | 3 132 | 18 936 |
| test nnz | 3 454 | 19 237 |
| df_train rows (interactions) | 32 014 | 262 169 |

**Taxonomy:** `config/foursquare_legacy_taxonomy.json` (gist mirror, schema A, 99.5 % coverage on TSMC2014's 402 cat_ids).

**Parquet schema (`data/processed/<city>/df_{train,val,test}.parquet`):**

| col | type | role |
|---|---|---|
| `u_idx`, `i_idx` | int32 | indices aligned to URM_*.npz |
| `user_id`, `venue_id` | int32, str | originals |
| `time_local` | datetime64[ns] | LOCAL time = UTC + tz_offset |
| `cat_fine` | str | Foursquare v2 fine category (135 NYC) |
| `cat_macro` | str | top-level v2 macro (9 NYC, 8 TKY) |
| `c_hour`, `c_dow`, `c_isweekend`, `c_month` | int8 | **time context** |
| `geohash5`, `geohash4` | str | **space context** |
| `dist_prev` | float32 | **mobility context** (km from previous check-in) |
| `intent_last_cat` | str | **intent proxy** (cat_macro of strictly previous interaction) |
| `split` | str | redundant |

**Invariants tested in `tests/test_preprocessing.py`** (all pass):
1. counts conservation
2. no cold leakage
3. dual-view coherence (URM (u,i) set == df dedup set per split)
4. per-user temporal monotonicity
5. **causal intent proxy** verified

---

## 3  step02a — the CF floor (URM holdout protocol)

8 **context-blind** baselines: `Random, TopPop, ItemKNN, UserKNN, P3α, RP3β, EASE^R, FM-vanilla`.

**Per-model protocol:** Bayesian search (skopt `gp_hedge`, n_cases=100), optimise `RECALL@20` on `URM_val`, refit on `URM_train ∪ URM_val`, evaluate on `URM_test` with `save_per_user=True`. Cutoffs `K ∈ {1, 5, 10, 20, 40, 50, 100}`. Per-user `.npz` in the **standard schema**:

```
{user_ids, RECALL_K, NDCG_K, PRECISION_K, MAP_K, MRR_K}  for every K
```

**Headline (test R@20 / NDCG@20):**

| model   | NYC R@20 | NYC N@20 | TKY R@20 | TKY N@20 |
|---------|---------:|---------:|---------:|---------:|
| Random  | 0.0172   | 0.0083   | 0.0030   | 0.0021   |
| TopPop  | 0.0704   | 0.0480   | 0.0453   | 0.0426   |
| ItemKNN | 0.0880   | 0.0581   | 0.0607   | 0.0527   |
| UserKNN | 0.0847   | 0.0571   | 0.0607   | 0.0538   |
| P3α     | **0.0929** | 0.0588 | 0.0604   | 0.0540   |
| RP3β    | 0.0926   | 0.0591   | 0.0621   | 0.0542   |
| EASE^R  | 0.0904   | **0.0594** | **0.0637** | **0.0558** |
| FM      | 0.0788   | 0.0528   | 0.0591   | 0.0505   |

CF blind ceiling: **R@20 ≈ 0.09 NYC, ≈ 0.062 TKY** — every contextual model has to clear this band.

Outputs (gitignored): `outputs/<city>/baselines/{Random,TopPop,...,FM}.{npz,summary.json,best_hp.json}` + `floor_table.{csv,md}`.

---

## 4  step02b — the X-SAGE situation-aware layer

> Protocol caveat: X-SAGE is evaluated **per-request next-item** (each test
> check-in is a request with a single target item). This is NOT the URM
> set-holdout of step02a — absolute numbers are not directly comparable. The
> right comparison is **internal** between B_blind / X-SAGE / B_full, all
> per-request.

### 4.1  Architecture (level by level)

```
pipeline/step02_models/xsage/
├── __init__.py             module docstring (public surface)
├── l0_sensing.py           (117 LOC) causal last-n macro window per request
├── l1_perception.py        (231 LOC) contribution functions c̃ (eq.1) via
│                                       shallow trees, macro transition W
│                                       (eq.2), attractors (eq.3), recency
│                                       profile m (eq.4), intent vector e
│                                       (eq.5) — with mode ∈
│                                       {hard, all, soft_topr} since round-2
├── l2_comprehension.py     (228 LOC) descriptor v = [c̃ ‖ e]; Lingras–West
│                                       rough k-means (eq.7-10); auto ε; ARI
├── l3_projection.py        (107 LOC) situation transition T (eq.17), next-
│                                       situation predictor, time-only
│                                       baseline, dynamic fairness curve
│                                       (eq.19), boundary disambiguation
│                                       (eq.18)
├── recommendation.py       (182 LOC) backbone softmax p_B (eq.13);
│                                       situational scores s_S / p_S (eq.14);
│                                       harmonic combiner (eq.15); ADDITIVE
│                                       combiner (round-2 1.5); shrunken
│                                       per-situation biases (raw + z-scored)
├── backbone.py             refit FM-vanilla (B_blind) or EASE^R on train+val,
│                            dump full score matrix to backbone/
├── backbone_full.py        ContextAwareFM (B_full) — HAGRID-style multi-hot
│                            Rendle FM (user, item, cat_macro, cat_fine,
│                            c_hour, c_dow, c_isw, c_month, prev_geohash5,
│                            intent_last_cat). BPR loss, full-catalogue
│                            scoring via context/item-block decomposition.
├── bfull_tuning.py         (235 LOC) round-2 1.1 — light HP sweep for B_full
│                                       with val R@20 early stop
├── metrics.py              LT (eq.11), KL (eq.12), Δ_K (eq.16), topk
├── viz.py                  PCA scatter, T heatmap, dyn-fairness curve,
│                            modulation bar chart
└── orchestrator.py        (1242 LOC) stage runners (Stage A/B/C/D)

experiments/
├── run_xsage.py           CLI: --city, --stage {A,B,C,D,E,all}, --K, --eps,
│                           --n, --H, --gamma, --beta, --kappa, --combiner
│                           {harmonic, additive}, --intent-mode {hard, all,
│                           soft_topr}, --seed, -v
├── round2_tune_bfull.py   round-2 1.1 — B_full HP sweep driver
├── round2_fairness_rerank.py   round-2 2.1 — long-tail rerank in sinks
└── round2_explainability.py    round-2 Track 3 — cards + explanations

tests/test_xsage.py        (170 LOC) 9 unit checks: row-sum invariants,
                            matched-OFF (kappa=0 ⇒ ON==OFF), ARI of identical
                            labels = 1, etc.
```

### 4.2  Per-stage results

#### Stage A — situations

Auto-tunes `(K, ε)` by grid sweep on the train descriptor `v`. Picks the cell with highest ARI(seed=42, seed=43) and boundary fraction ∈ [10 %, 30 %].

| | NYC (intent `hard`) | TKY (intent `hard`) | TKY (intent `all`, round-2) |
|---|---|---|---|
| K | 8 | 6 | 6 |
| ε | 0.020 | 0.050 | 0.050 |
| ARI(s42, s43) | **0.830** | 0.656 | **0.741** |
| boundary frac (test) | 18.2 % | 26.9 % | 26.7 % |
| attractors | 5 (Food, Nightlife, Outdoors, Shop, Travel) | 2 (Shop, Travel) | (all 8 weighted) |

Both **PASS** cluster stability (ARI ≥ 0.6). On NYC, `--intent-mode all` drops K_opt to 4 and ARI to 0.750 — NYC is better off with the hard cutoff.

Artefacts: `outputs/<city>/xsage/situations/{archetypes.csv, examples.csv, clusters_2d.png, contrib_functions.png, situation_cards.md, fit.npz, summary.json, tuning_log.csv}`.

#### Stage B — fairness lens (cardinal check 1)

Stratify B_blind's test top-20 by the request's situation, compute LT (eq.11) and KL (eq.12), plus the **per-situation available long-tail share** (items the user has not yet seen) as the structural-scarcity control.

| city | global LT@20 | global KL mean | inequity sinks | best sink lift |
|---|---|---|---|---|
| NYC | 0.112 | 0.261 | **{6}** | KL × 2.95, LT excess −69.5 pp |
| TKY | 0.052 | 0.121 | **{4, 5}** | s5: KL × 2.71, LT excess −70.8 pp |

**Verdict: GREEN on both cities.** TKY's baseline top-K is already half as long-tail as NYC's; even there the sinks concentrate further.

Artefacts: `outputs/<city>/xsage/fairness/{per_situation.csv, top_items_by_situation.csv, verdict.json}`.

#### Stage C — projection (cardinal check 2)

Estimate `T_{kk'} = P̂(z_{t+1}=k' | z_t=k)` from consecutive (z_t, z_{t+1}) pairs in (train ∪ val). For each test row find `z_prev` (user's most recent (train ∪ val) situation strictly before t) and predict `argmax T[z_prev]`. Baseline: `argmax P̂(z | c_hour)` from (train ∪ val).

| city | T-based macro F1 | time-only macro F1 | ΔF1 | McNemar |
|---|---|---|---|---|
| NYC (hard) | **0.333** | 0.162 | **+0.171** | p < 1e-4 (T wins) |
| TKY (hard) | 0.224 | **0.265** | −0.041 | p < 1e-4 (time wins) |
| TKY (`all`) | 0.261 | 0.276 | −0.016 | p < 1e-4 (time still wins, gap halves) |

**Verdict: GREEN on NYC; FLAT on TKY.** TKY's projection collapses to the clock; `all` mode halves the gap but doesn't recover.

Artefacts: `outputs/<city>/xsage/projection/{transition_matrix.csv, transition_counts.csv, transition_heatmap.png, next_situation_f1.csv, dynamic_fairness.csv, dynamic_fairness.png, verdict.json}`.

#### Stage D — three-way comparison + modulation

Same FM architecture, only the routing of context differs:

| variant | how context enters |
|---|---|
| **B_blind** | not at all — vanilla MF |
| **X-SAGE** | through the situational head, modulating the backbone score |
| **B_full** | inside the FM, as multi-hot features (HAGRID-style) |

**Original (untuned B_full, harmonic combiner):**

| variant | NYC R@20 | TKY R@20 |
|---|---|---|
| B_blind | 0.0787 | 0.0500 |
| X-SAGE κ=0 (matched-OFF) | 0.0787 ✓ | 0.0500 ✓ |
| X-SAGE κ=0.10 (harmonic) | 0.0446 (−43 %) | 0.0405 (−19 %) |
| B_full (untuned) | 0.0932 (+18.5 %) | 0.0433 (−13 %) |

**Round-2 (tuned B_full + additive combiner):**

| variant | NYC R@20 | NYC N@20 | TKY R@20 | TKY N@20 |
|---|---|---|---|---|
| B_blind | 0.0787 | 0.0341 | 0.0500 | 0.0212 |
| X-SAGE κ=0 | 0.0787 ✓ | 0.0341 | 0.0500 ✓ | 0.0212 |
| X-SAGE additive κ=0.10 | **0.0778** | 0.0336 | **0.0495** | 0.0212 |
| X-SAGE additive κ=0.25 | 0.0756 | 0.0317 | 0.0494 | 0.0214 |
| X-SAGE additive κ=0.50 | 0.0720 | 0.0301 | 0.0488 | **0.0215** (+0.0003) |
| X-SAGE additive κ=1.00 | 0.0668 | 0.0274 | 0.0472 | 0.0206 |
| **B_full (tuned)** | **0.0936** | **0.0395** | 0.0417 | 0.0190 |

* **Matched-OFF (κ=0)** is exact on both cities — design guarantee verified.
* **Additive combiner** is dramatically less aggressive than the harmonic one and finally surfaces the gating behaviour the brief expected: Δ_K core / boundary ratio = **3.0 ×** at κ=0.1 on NYC (vs ~1 × under harmonic).
* **B_full tuning** confirms the earlier verdicts: NYC's "context helps" survives tuning (+0.0149 vs B_blind); TKY's "context hurts" also survives (−0.0083, in fact slightly worse than untuned — the d=128 winner overfits per-context-cell sparsity). **The TKY paradox is real.**
* **X-SAGE κ > 0 never beats B_blind on either city.** Per the brief's stop rule for §1.5, the coarse macro routing cannot recover the per-cell signal the multi-hot B_full picks up. The X-SAGE accuracy contribution is parity at κ=0.

Artefacts: `outputs/<city>/xsage/recommendation/` (harmonic) and `outputs/<city>/xsage/recommendation_additive/` (round-2): `{three_way.csv, matched_pair.csv, modulation_summary.csv, modulation_summary.png, Bblind.npz, Bfull.npz, XSAGE_kappa<best>.npz, summary.json, modulation_log.csv}`.

### 4.3  Round-2 push — what changed in each commit

| commit | what it adds | why |
|---|---|---|
| `80a50b0` | **1.1** B_full tuning (light HP sweep, val R@20 ES); **1.5** additive combiner with z-scored shrunken biases; `--combiner` flag | fair B_full comparison; less aggressive routing |
| `da5f38c` | **1.4** `compute_intent` modes {hard, all, soft_topr}; `--intent-mode` flag; outputs route to `xsage_intent_<mode>/` | rescue TKY projection |
| `dc1dd28` | **2.1** situation-gated long-tail re-ranking (additive nudge on core sink requests) | the trustworthiness headline |
| `13fcb40` | **Track 3** contribution functions plot, situation cards, per-recommendation explanations | explainability deliverables |

### 4.4  The fairness re-ranking headline (round-2 2.1)

Additive nudge on core requests of Stage-B-flagged sinks:
`ŝ_fair(u, i) = s_B(u, i) + κ_fair · γ_S · sink(z) · 1[i ∈ G_1]`

**NYC (sink = s6):**

| κ_fair | sink LT@20 | sink R@20 | global R@20 Δ |
|---|---|---|---|
| 0.00 | 0.110 | 0.0263 | 0.0000 |
| **0.50** | **0.363 (+230 %)** | **0.0263** | **0.0000** ← free 3× LT |
| 1.00 | 0.738 (+571 %) | 0.0175 | −0.0002 |

**TKY (sinks = s4, s5):**

| κ_fair | sink LT@20 | sink R@20 | global R@20 Δ |
|---|---|---|---|
| 0.00 | 0.074 | 0.0365 | 0.0000 |
| **0.50** | **0.151 (+104 %)** | **0.0368** | **+0.0001** ← marginal gain! |
| 1.00 | 0.316 (+327 %) | 0.0334 | −0.0006 |

Artefacts: `outputs/<city>/xsage/fairness/{reranking_tradeoff.csv, reranking_tradeoff.png}`.

### 4.5  Cardinal-check matrix (round-2 final)

| dimension | NYC | TKY |
|---|---|---|
| Stage A — cluster stability | ✅ ARI 0.830 (K=8, hard) | ✅ ARI 0.741 (K=6, **all**) |
| Stage B — fairness lens | ✅ GREEN (s6) | ✅ GREEN (s4, s5) |
| Stage C — projection vs clock | ✅ GREEN (+0.171, p<1e-4) | ❌ FLAT (−0.016, p<1e-4, time wins) |
| Stage D — matched-OFF | ✅ exact (κ=0) | ✅ exact (κ=0) |
| Stage D — additive combiner well-gated | ✅ Δ_K core/bnd ≈ 3 × | ✅ near-flat at κ ∈ [0.1, 0.5] |
| Stage D — context routing helps accuracy | ❌ X-SAGE ≤ B_blind; **B_full +0.0149** | ❌ neither helps |
| 2.1 — sink long-tail re-ranking | ✅ **+230 % at zero cost** | ✅ **+104 % with +0.0001 gain** |
| Track 3 — explainability | ✅ cards + explanations | ✅ cards + explanations |

### 4.6  Honest reading

The X-SAGE contribution stands as **lens + projection (on NYC) + sink-aware long-tail re-ranking + per-recommendation explanations**. The accuracy-routing story closes with **honest matched-OFF parity**; the coarse macro biases cannot recover the per-cell signal that a properly tuned context-aware FM picks up on NYC, and on TKY no routing helps regardless. The trustworthiness payoff is concrete and quantified on both cities; the accuracy axis is not the contribution.

---

## 5  Code layout (this branch)

```
IntentAwareRS_thesis/
├── config/
│   ├── protocol.yaml                        # Shehzad protocol constants
│   └── foursquare_legacy_taxonomy.json      # 203 KB, schema A, 99.5 % coverage
├── data/
│   ├── raw/      dataset_TSMC2014_{NYC,TKY}.txt    # gitignored
│   └── processed/<city>/  URM_*.npz, df_*.parquet   # gitignored
├── engine/                                 # ex topn_baselines_neurals (frozen)
│   ├── Evaluation/Evaluator.py             # patched: save_per_user
│   └── Recommenders/{KNN, GraphBased, EASE_R, NonPersonalizedRecommender,
│                       FactorizationMachines, CAMF, ...}
├── pipeline/
│   ├── step01_preprocessing/               # FROZEN (taxonomy + run.py)
│   ├── step02_models/
│   │   ├── baselines/__init__.py           # FROZEN (floor orchestrator)
│   │   └── xsage/                          # ACTIVE — round-2 work
│   ├── step03_evaluation/                  # heritage stubs
│   └── step04_statistical_validation/      # heritage (Wilcoxon+HMP+Holm)
├── experiments/
│   ├── run_baselines.py                    # FROZEN (step02a CLI)
│   ├── run_xsage.py                        # ACTIVE (step02b CLI)
│   ├── round2_tune_bfull.py                # round-2 1.1 driver
│   ├── round2_fairness_rerank.py           # round-2 2.1 driver
│   └── round2_explainability.py            # round-2 Track 3 driver
├── tests/
│   ├── test_smoke.py                       # synthetic URM end-to-end
│   ├── test_preprocessing.py               # 5 step01 invariants
│   └── test_xsage.py                       # 9 X-SAGE checks
├── outputs/                                # gitignored
├── outputs.snapshot_20260609_215753/       # local backup of step02b-goNogo + floor
├── REPO_SNAPSHOT_2026-06-10.md             # this file
├── requirements-cpu.txt
└── README.md
```

**X-SAGE module LOC:** ~2 200 lines (l0..l3 + orchestrator + recommendation + backbones + viz). Driver scripts: ~ 800 lines. Tests: ~ 670 lines. Total active code on this branch ≈ **3 700 LOC** of pure-Python + PyTorch + NumPy.

---

## 6  Outputs tree (current state, gitignored)

```
outputs/
├── NYC/
│   ├── baselines/                                # step02a floor — 8 .npz + meta
│   ├── xsage/                                    # default mode (intent=hard)
│   │   ├── backbone/  {FM, Bfull}.{scores.npy, meta.json}
│   │   ├── situations/  archetypes.csv, examples.csv, clusters_2d.png,
│   │   │                 contrib_functions.png, situation_cards.md,
│   │   │                 fit.npz, summary.json, tuning_log.csv
│   │   ├── fairness/  per_situation.csv, top_items_by_situation.csv,
│   │   │              verdict.json,
│   │   │              reranking_tradeoff.csv, reranking_tradeoff.png
│   │   ├── projection/  transition_matrix.csv, transition_counts.csv,
│   │   │                transition_heatmap.png, next_situation_f1.csv,
│   │   │                dynamic_fairness.csv, dynamic_fairness.png,
│   │   │                verdict.json
│   │   ├── recommendation/         # harmonic combiner (original)
│   │   │   three_way.csv, matched_pair.csv,
│   │   │   modulation_summary.csv, modulation_summary.png,
│   │   │   modulation_log.csv,
│   │   │   Bblind.npz, Bfull.npz, XSAGE_kappa0.0.npz,
│   │   │   summary.json
│   │   ├── recommendation_additive/    # round-2 1.5
│   │   │   (same files, plus explanations.csv)
│   │   └── REPORT.md
│   └── xsage_intent_all/             # round-2 1.4
│       situations/, fairness/, projection/
├── TKY/
│   ├── baselines/                                # step02a floor
│   ├── xsage/                                    # default mode
│   └── xsage_intent_all/                         # round-2 1.4
```

Disk: ~ 479 MB total (mostly the multi-hot B_full and B_blind score matrices on TKY which are 2 214 × 2 852 × float32 ≈ 25 MB each, replicated across several recommendation_* dirs).

---

## 7  Reproducibility — commands

```bash
# 0  Sanity
.venv/bin/python -m pytest tests/ -v
# (expect 19/19 passing)

# 1  Floor (step02a) — about 35 min on M2
python -m experiments.run_baselines --city both -v

# 2  X-SAGE Stages A → B → C → D, both cities, default intent mode (hard)
python -m experiments.run_xsage --city both --stage A -v
python -m experiments.run_xsage --city both --stage B -v
python -m experiments.run_xsage --city both --stage C -v
python -m experiments.run_xsage --city both --stage D --combiner harmonic -v

# 3  Round-2
python -m experiments.round2_tune_bfull --city both -v            # 1.1
python -m experiments.run_xsage --city both --stage D --combiner additive -v   # 1.5
python -m experiments.run_xsage --city NYC --stage A --intent-mode all -v   # 1.4 (also B and C if you want full re-run)
python -m experiments.run_xsage --city TKY --stage A --intent-mode all -v
python -m experiments.run_xsage --city TKY --stage B --intent-mode all -v
python -m experiments.run_xsage --city TKY --stage C --intent-mode all -v
python -m experiments.round2_fairness_rerank --city both          # 2.1
python -m experiments.round2_explainability --city both           # Track 3
```

**Rollback:**

```bash
git checkout foursquare-rebuild        # pristine state (just floor)
# or
git checkout pre-xsage-rollback        # the same commit, by tag
# or just delete the branch:
git branch -D step02b-xsage
```

The `outputs/` folder is gitignored everywhere; nothing on disk affects git state.

---

## 8  Open questions / what's next

These are deliberately *not* in scope of any current commit. They're the natural follow-ups recorded honestly:

1. **Incremental feature ablation (round-2 1.2/1.3) was skipped.** A proper M0 → M5 sweep would reveal which feature axes are doing the work in B_full on NYC and which add noise on TKY. Cost was high relative to the rest of round-2; the qualitative answer (NYC: context helps, TKY: context hurts) is robust without it.

2. **Combiner re-design.** The harmonic combiner is too aggressive; the additive one is well-behaved but doesn't earn accuracy. A natural intermediate is a **gated multiplicative**: `ŝ = s_B · (1 + κ · γ · π · b̃)` with `b̃` z-scored. This was not tried.

3. **Step04 statistical validation on the X-SAGE outputs.** The per-user `.npz` exports from B_blind / X-SAGE / B_full already follow the step04 schema (`{user_ids, RECALL_K, NDCG_K, PRECISION_K, MAP_K, MRR_K}`). Plugging them into the existing Wilcoxon + HMP + Holm + bootstrap is one command away. Not done in this round.

4. **TKY context paradox.** Tuning B_full slightly *worsens* TKY R@20 (0.0433 → 0.0417). The narrative on the paper has two paths:
   * **Defensive:** report TKY as evidence the X-SAGE contribution is fairness-only where accuracy headroom is absent.
   * **Investigative:** run the ablation in §1 above to localise which TKY feature is the culprit. The `cat_fine` one-hot (135 levels) is the prime suspect for sparsity-induced overfit.

5. **`step02b-goNogo` branch can be archived/deleted.** It carries the bilinear-gate first attempt that closed RED on both cities. The X-SAGE work superseded it. The Part-A diagnostics (genuine MoE collapse) and Part-B (forced-but-useless sweep) from that branch are documented and archived under `outputs.snapshot_20260609_215753/`.

---

## 9  Cross-references inside the repo

* **`outputs/NYC/xsage/REPORT.md`** — per-city report for NYC with all round-2 numbers in narrative form.
* **`outputs/TKY/xsage/REPORT.md`** — same for TKY.
* **`outputs.snapshot_20260609_215753/README.md`** — README of the previous attempt's archive.
* **`xsage_approach.tex`** — the formal LaTeX definition with eq.1 → eq.19 referenced throughout the code comments.
* **`STATISTICAL_PROTOCOL.md`** (heritage) — the step04 statistical pipeline already in the codebase.

---

## 10  TL;DR for the reviewer

* The thesis codebase is on **`step02b-xsage`**, all tests passing, branch in sync with origin.
* **Step01 + step02a (floor)** are done; the floor's per-user `.npz` is the contract every subsequent model honours.
* **Step02b X-SAGE** is the active contribution. The two cardinal checks (Stage B lens, Stage C projection) pass GREEN on NYC and pass GREEN (lens) / FAIL FLAT (projection) on TKY.
* The **matched-OFF guarantee** holds on both cities under both combiners — the X-SAGE accuracy story is **parity at κ=0**, never above.
* The **additive combiner** (round-2 1.5) makes the modulation well-behaved (Δ_K core/boundary ratio ≈ 3 ×) but doesn't beat B_blind on either city. Per the brief's stop rule, **the recommendation contribution lives in lens + projection-on-NYC + sink-aware long-tail re-ranking + explainability**, not in routing accuracy.
* The **fairness re-ranking** (round-2 2.1) is the **trustworthiness headline**: triples long-tail exposure in NYC sinks at zero accuracy cost, doubles it in TKY sinks with a tiny accuracy *gain*.
* **Round-2 1.2/1.3 (incremental feature ablation)** was the only piece deliberately skipped; the rationale is documented in §8.1.

Ready for review.
