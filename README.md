# Situation-Aware Recommendation — Thesis Codebase

This repository hosts the code for a PhD thesis on **situation-aware
recommendation**. The paper is methodological (survey-grounded framework
operationalised on top of Factorization Machines), evaluated on **Foursquare
TSMC2014 (NYC + Tokyo)** in a **single-phase**, **CPU-only**, **top-N full
ranking** setting. Beyond-accuracy is a first-class concern: interpretability
(situational archetypes) and fairness (Ge et al. 2022) sit next to standard
accuracy metrics.

> Branch convention: this branch (`foursquare-rebuild`) is the active
> rebuild. The previous branches `main`, `phase2-baselines-and-validation`
> and `gowalla-dataset-comparison` remain as backup of the earlier Gowalla
> plan — nothing is lost.

---

## The 4-block flow

The repo is laid out so that the directory tree reads like the pipeline:

```
            ┌────────────────────────────────────────────────────────┐
            │  Block 1     Block 2       Block 3        Block 4      │
data/  ───► │ preprocess ─► models ───► evaluation ──► statistical   │
            │                                            validation  │
            └────────────────────────────────────────────────────────┘
```

| Block | What it does | Code lives in |
|---|---|---|
| **1. Preprocessing** | Foursquare TSMC2014 → dual view (sparse URM + per-interaction DF). Same user/item indexing, same train/val/test split. | `pipeline/step01_preprocessing/` |
| **2. Models** | CARS (FM-context, CAMF-context, TFM) and the situational FM proposal. Each model extends `engine/Recommenders/`. | `pipeline/step02_models/{cars,proposed}/` |
| **3. Evaluation** | Beyond-accuracy: fairness (KL + Long-tail Rate, Ge 2022, global *and* stratified-per-situation) + interpretability (archetype labelling). | `pipeline/step03_evaluation/` |
| **4. Statistical validation** | Paired Wilcoxon + Pratt + Harmonic Mean P-value + Holm on the 3-comparison primary family. Bootstrap CI on descriptive table. | `pipeline/step04_statistical_validation/` |

Each block has its own entry point in `experiments/`:

```bash
python -m experiments.run_preprocessing            # Block 1
python -m experiments.run_baselines                # Block 2, CF pavement
python -m experiments.run_cars                     # Block 2, CARS
python -m experiments.run_proposed                 # Block 2, proposal
python -m experiments.run_evaluation               # Block 3
python -m experiments.run_statistical_validation   # Block 4
```

---

## Repository layout (snapshot)

```
IntentAwareRS_thesis/
├── README.md                          this file
│
├── engine/                            Ferrari Dacrema framework, reused INTACT
│   ├── Data_manager/                  ↪ dataset readers + splitters
│   ├── Evaluation/
│   │   ├── Evaluator.py               ↪ EvaluatorHoldout (THESIS PATCH: save_per_user)
│   │   ├── metrics.py
│   │   └── EVALUATOR_PATCH.md         ↪ doc of the per-user patch
│   ├── HyperparameterTuning/          ↪ Bayesian search (scikit-optimize)
│   └── Recommenders/
│       ├── NonPersonalizedRecommender.py   (Random, TopPop)
│       ├── KNN/                       ↪ ItemKNN, UserKNN
│       ├── GraphBased/                ↪ P3α, RP3β
│       ├── EASE_R/                    ↪ EASE^R
│       ├── FactorizationMachines/     ↪ _BPRMFBase + FM-vanilla (backbone)
│       ├── CAMF/                      ↪ CAMF-vanilla (backbone)
│       └── Similarity/                ↪ shared similarity utilities
│
├── pipeline/                          PROJECT code (the new stuff)
│   ├── step01_preprocessing/          ↪ TSMC2014 → dual-view (not yet implemented)
│   ├── step02_models/
│   │   ├── cars/                      ↪ FM-context, CAMF-context, TFM (not yet)
│   │   └── proposed/                  ↪ situational FM (not yet, Sec.4 in paper)
│   ├── step03_evaluation/
│   │   ├── fairness.py                ↪ KL + Long-tail + per-situation (stub)
│   │   └── interpretability.py        ↪ archetype labelling (stub)
│   └── step04_statistical_validation/
│       ├── statistical_validation.py  ↪ Wilcoxon+Pratt + HMP + Holm + bootstrap
│       └── STATISTICAL_PROTOCOL.md    ↪ methodological doc
│
├── experiments/                       thin runners orchestrating the pipeline
│   ├── run_preprocessing.py
│   ├── run_baselines.py               ↪ CF pavement on a URM (synthetic-driven for now)
│   ├── run_cars.py
│   ├── run_proposed.py
│   ├── run_evaluation.py
│   └── run_statistical_validation.py
│
├── config/
│   └── protocol.yaml                  ↪ inherited Shehzad constants (cutoffs,
│                                        n_trials=100, optimize=Recall@20, seeds,
│                                        statistical-validation policy)
│
├── data/                              GITIGNORED
│   ├── raw/                           ↪ TSMC2014 TSVs land here (empty stub)
│   └── processed/                     ↪ pipeline.step01 output (empty stub)
│
├── outputs/                           GITIGNORED
│                                       per-user .npz, per-model TSV, tables,
│                                       figures — recomputable
│
├── tests/
│   └── test_smoke.py                  ↪ end-to-end smoke on a synthetic URM,
│                                        always passes on a clean checkout
│
├── requirements-cpu.txt               numpy<2, scipy<1.14, pandas<2.2, sklearn,
│                                       scikit-optimize, torch>=2.0, tables, …
├── setup_env.sh                       ./setup_env.sh creates .venv/ ready
└── .gitignore
```

---

## Quick start

```bash
cd ~/Downloads/IntentAwareRS_thesis
./setup_env.sh                   # creates .venv/ and installs requirements-cpu.txt
source .venv/bin/activate
python -m tests.test_smoke       # synthetic-URM smoke test, must print PASSED
```

Tested on: macOS 26.2 (arm64, Apple M2, 16 GB RAM), Python 3.11.0.

---

## What's in `engine/` (and what's intentionally NOT)

The `engine/` is the Ferrari Dacrema fork (formerly `topn_baselines_neurals/`)
used by Shehzad et al. (SIGIR 2025). It is reused **intact** — we extend it
from `pipeline/`, we don't modify it. The one exception is the additive patch
on `Evaluation/Evaluator.py` (`save_per_user` flag) documented in
[`engine/Evaluation/EVALUATOR_PATCH.md`](engine/Evaluation/EVALUATOR_PATCH.md).

What we removed from the engine in this branch (relative to `main`):

- The DCCF and BIGCF Recommenders — out-of-scope, citation-only in Related Work.
- Dataset readers for Gowalla, Amazon-Book, Tmall, Yelp 2018, IDS4NR
  (MovieLens, Beauty, Music), KGIN (AlibabaFashion, LastFM, AmazonBook),
  and the upstream MovieLens reader.
- The optional Recommender families that we don't use:
  `SLIM/`, `MatrixFactorization/`, `Neural/`, `FeatureWeighting/`, plus the
  monolithic `Recommender_import_list.py`.

> **Reviewer escape hatch.** If a reviewer asks "where is SLIM, where is
> IALS?" — they are canonical baselines in the Ferrari Dacrema tradition we
> cite — they are one command away:
> ```
> git checkout main -- topn_baselines_neurals/Recommenders/SLIM
> git checkout main -- topn_baselines_neurals/Recommenders/MatrixFactorization/IALSRecommender.py
> ```
> Recoverable from `main`, not burned bridges.

---

## What's in `pipeline/` (and what's intentionally a stub)

Everything in `pipeline/` is **project code** that will be filled in next
briefs. The current state is the minimal scaffold so that runners can be
shaped and the pipeline can be reasoned about end-to-end:

- `step01_preprocessing/` — empty, ships with a README of what the loader
  will produce.
- `step02_models/cars/` and `step02_models/proposed/` — empty (just READMEs)
  until the architecture is locked.
- `step03_evaluation/fairness.py` and `interpretability.py` — `NotImplementedError`
  stubs with documented planned signatures.
- `step04_statistical_validation/statistical_validation.py` — **fully working**,
  unchanged from Phase 2. Same Wilcoxon+Pratt + HMP + Holm + bootstrap. Read
  the co-located `STATISTICAL_PROTOCOL.md` for methodology.

---

## What's in `config/`

[`config/protocol.yaml`](config/protocol.yaml) is the canonical home for the
**evaluation protocol constants** inherited from Shehzad. Before the cleanup,
these constants lived hard-coded inside the three `run_experiments_*` scripts
of the upstream repo. Those scripts are now gone; the constants are not:

- `cutoff_list = [1, 5, 10, 20, 40, 50, 100]`
- `metric_to_optimize = RECALL`, `cutoff_to_optimize = 20`
- `n_cases = 100`, `n_random_starts = 5` (Bayesian search)
- `validation_portion = 0.1` (per-user, deterministic — see engine)
- Deep-model reference seeds (DCCF=2022, BIGCF=2023) kept for the record
  even though those models are not run here.
- The 5-seed set for stochastic models we DO train: `[2022, 2023, 42, 0, 1]`.
- Statistical-validation policy: Wilcoxon + Pratt + HMP + Holm + percentile
  bootstrap, `α = 0.05` familywise.

The legacy best HP values for Shehzad's tuned baselines on Gowalla/AmazonBook/Tmall
are also kept under `legacy_shehzad_best_hp:` as a frozen historical reference.
They are NOT used in this branch — Foursquare will get its own Bayesian
search.

---

## The protocol pieces that survived the rebuild

Two methodological documents survived because they describe pipeline assets
that travel with the project, not Gowalla-specific results:

- [`engine/Evaluation/EVALUATOR_PATCH.md`](engine/Evaluation/EVALUATOR_PATCH.md)
  — what the `save_per_user` patch changed in the Evaluator and what stays
  bit-identical; the per-`user_id` pairing contract for consumers.
- [`pipeline/step04_statistical_validation/STATISTICAL_PROTOCOL.md`](pipeline/step04_statistical_validation/STATISTICAL_PROTOCOL.md)
  — why Wilcoxon + Pratt; why HMP (Wilson 2019) and NOT mean-per-user across
  seeds; why Holm only on the primary family; percentile bootstrap; the
  primary-comparison family layout.

Everything else from the previous Gowalla-anchored phase (CODE_ANALYSIS,
DATA_INVENTORY, REPRODUCIBILITY_CHECK, SHEHZAD_PROTOCOL, PHASE1_SUMMARY,
GOWALLA_DATASETS_COMPARISON) is gone from this branch; it is preserved in
the history of `main` and `gowalla-dataset-comparison` and accessible via
`git show <commit>:<path>` if ever needed.
