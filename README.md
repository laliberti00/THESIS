# IntentAwareRS — Thesis Baseline

PhD thesis baseline. Forked and cleaned from the reproducibility package of
[Shehzad, Ferrari Dacrema & Jannach — *"A Worrying Reproducibility Study of Intent-Aware
Recommendation Models"*, ACM SIGIR 2025](https://github.com/RecSysEvaluation/IntentAwareRS).

This repository is the starting point for a thesis on **situation-aware Factorization
Machines** for top-N recommendation. In Phase 1 (this state) we have **(a)** trimmed
the upstream framework to the perimeter actually used by the thesis, **(b)** verified
that we can reproduce the non-neural baseline numbers of Shehzad on at least one
dataset, and **(c)** documented what the data and the pipeline really contain.

> ⚠️ Phase 1 status — **see [PHASE1_SUMMARY.md](PHASE1_SUMMARY.md) first.** It lists
> the open decisions (Plan A/B/C for the FM model) that need to be made before
> Phase 3 starts.

---

## Perimeter of this repo (what is kept vs. dropped)

**Kept (in scope for the thesis):**
- **Two deep intent-aware models**: DCCF (SIGIR 2023), BIGCF (SIGIR 2024).
- **Five non-neural baselines tuned by Shehzad**: ItemKNN, UserKNN, P3α, RP3β, EASE^R.
- **Three datasets**: Gowalla, Amazon-Book, Tmall (DCCF splits, see [DATA_INVENTORY.md](DATA_INVENTORY.md)).
- The `topn_baselines_neurals/` framework (a fork of `RecSys2019_DeepLearning_Evaluation`).

**Dropped (out of thesis scope):**
- DGCF (SIGIR 2020), KGIN (WWW 2021), IDS4NR (TKDE 2022) and their datasets
  (Yelp 2018, AlibabaFashion, lastFm, MovieLens, Beauty, Music).
- All of their entry-point scripts and HP-tuning scripts.
- The Python 3.6 + TensorFlow 1.14 environment that DGCF required.

For the rationale, see [PHASE1_SUMMARY.md](PHASE1_SUMMARY.md). For the original
Shehzad repo with all five models intact, see the upstream link above or check out
the tag `original-shehzad-v1` on the *other* local repo at
`~/Downloads/IntentAwareRS_original/`.

---

## Repo layout

```
IntentAwareRS_thesis/
├── README.md                                  this file
├── PHASE1_SUMMARY.md                          read this first
├── CODE_ANALYSIS.md                           how the framework is wired
├── DATA_INVENTORY.md                          what is really in the .pkl files
├── REPRODUCIBILITY_CHECK.md                   verified numbers vs. Shehzad
│
├── requirements-cpu.txt                       env for non-neural baselines
├── requirements-gpu.txt                       env for DCCF / BIGCF (CUDA)
├── setup_env.sh                               one-shot CPU env bootstrap
├── .gitignore
│
├── data/DCCF/{gowalla,amazonBook,tmall}/      train.pkl, test.pkl (+valid for AmazonBook)
├── results/{DCCF,BIGCF}/                      Shehzad's reference numbers (ground truth)
├── log/{gowalla,amazonbook,tmall}.log         Shehzad's original training logs
├── docs/tables_window/                        Shehzad's HTML result tables
│
├── topn_baselines_neurals/                    the framework
│   ├── Data_manager/                          dataset loaders + splitters
│   ├── Evaluation/                            EvaluatorHoldout + metrics
│   ├── HyperparameterTuning/                  scikit-optimize Bayesian search
│   └── Recommenders/                          baseline + DCCF + BIGCF + LightFM
│
├── inspect_pkls.py                            utility: dump .pkl shape/contents
├── repro_check_baseline.py                    runs the baselines on a dataset
│
└── run_experiments_for_DCCF_original_baselines.py     full DCCF + baseline eval
    run_experiments_for_BIGCF_original.py              BIGCF training/eval
    run_hyperparameter_search_baseline_DCCF_for_datasets.py  Bayesian HP search
```

---

## Environment setup

The framework needs two distinct environments. Pick the one matching your
hardware. **The CPU env covers everything we need for Phase 1 and Phase 2.**

### CPU env — non-neural baselines (works on macOS / Linux, no GPU)

This is the env to use locally on a laptop. It runs Random, TopPop, ItemKNN,
UserKNN, P3α, RP3β, and (RAM permitting) EASE^R.

```bash
cd ~/Downloads/IntentAwareRS_thesis
./setup_env.sh                        # creates .venv/ and installs requirements-cpu.txt
source .venv/bin/activate
```

Sanity check after activation:

```bash
python -c "import numpy, scipy, sklearn; print(numpy.__version__, scipy.__version__, sklearn.__version__)"
# Expected: numpy 1.26.x  scipy 1.13.x  sklearn 1.x
```

**Tested on:** macOS 26.2 (arm64, Apple M2, 16 GB RAM), Python 3.11.0.

### GPU env — DCCF and BIGCF (CUDA-capable machine required)

DCCF and BIGCF are heavy PyTorch models. On the original paper they were trained
with NVIDIA CUDA. We have not run them yet locally — they will be run on the
university workstation in Phase 2.

The recommended path is to mirror Shehzad's setup:

```bash
# On the GPU machine, with conda available:
conda create -n IntentAwareRS_thesis_gpu python=3.8
conda activate IntentAwareRS_thesis_gpu
pip install -r requirements-gpu.txt
```

Alternative without conda (Linux x86_64 + CUDA 11.8 NVIDIA):

```bash
python3.8 -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt
```

> Compatibility caveats — see [CODE_ANALYSIS.md §8.3](CODE_ANALYSIS.md):
> the framework was written for NumPy 1.23. We patched two NumPy 2.x
> incompatibilities (`np.int → int`) on the non-neural code path; the rest of
> the code still expects pre-2.0 NumPy aliases. With the pinned `numpy==1.23.5`
> in `requirements-gpu.txt` this is a non-issue. With the looser CPU env
> (`numpy<2`) you may hit additional `np.float`, `np.bool` errors if you try
> to run code paths beyond the non-neural baselines — patch them the same way.

---

## How to run experiments

All commands assume you are in the repo root with the right env active.

### 1. Inspect the datasets (quick, ~5 s)

```bash
python inspect_pkls.py
```

Prints shape / nnz / stats for the 14 `.pkl` files. The same output sources
[DATA_INVENTORY.md](DATA_INVENTORY.md).

### 2. Reproduce Shehzad's baselines (CPU)

```bash
# all baselines on one dataset (Random, TopPop, ItemKNN, UserKNN, P3α, RP3β)
python repro_check_baseline.py --dataset gowalla --model all

# one specific model
python repro_check_baseline.py --dataset gowalla --model RP3beta
```

Results are written to `repro_check_results/<dataset>_<model>.txt`. Compare
manually against `results/DCCF/<dataset>_<Model>Recommender.txt` (Shehzad's
ground truth). For a curated comparison see [REPRODUCIBILITY_CHECK.md](REPRODUCIBILITY_CHECK.md).

> EASE^R is **not** in `repro_check_baseline.py`'s default list: it requires
> a dense `n_items × n_items` Gram matrix, which is ≥14 GB for our smallest
> dataset (Tmall) and >26 GB for Gowalla. Run it only on a high-RAM machine.

### 3. Run the full DCCF baseline pipeline (GPU recommended, original Shehzad script)

This is Shehzad's own entry point. It trains DCCF *and* the baselines and saves
everything to `results/DCCF/`.

```bash
python run_experiments_for_DCCF_original_baselines.py --dataset gowalla
python run_experiments_for_DCCF_original_baselines.py --dataset amazonBook
python run_experiments_for_DCCF_original_baselines.py --dataset tmall
```

DCCF needs CUDA. The baseline part will still run on CPU if you skip the DCCF
section, but the script is monolithic — for CPU-only use `repro_check_baseline.py`
instead.

### 4. Run BIGCF (GPU recommended)

```bash
python run_experiments_for_BIGCF_original.py --dataset gowalla
python run_experiments_for_BIGCF_original.py --dataset amazonBook
python run_experiments_for_BIGCF_original.py --dataset tmall
```

BIGCF reuses the DCCF train/test split (see Shehzad's README for context). No
need to retune baselines for BIGCF.

### 5. Re-tune baseline hyperparameters (optional, slow)

Only needed if you want to redo the Bayesian search from scratch (Shehzad's
"best HP" values are already hardcoded in
`run_experiments_for_DCCF_original_baselines.py:84-100` and in
`repro_check_baseline.py`).

```bash
python run_hyperparameter_search_baseline_DCCF_for_datasets.py --dataset gowalla
```

100 trials × 5 models, multiprocessing. Hours on a laptop.

---

## What was verified in Phase 1

- **Reproducibility (Gowalla)**: ItemKNN, UserKNN, P3α, RP3β match Shehzad's
  reference numbers to ≤ 0.06% relative deviation on Recall@20 (and ≤ 0.015%
  on NDCG@20). RP3β and TopPop are bit-identical. See
  [REPRODUCIBILITY_CHECK.md](REPRODUCIBILITY_CHECK.md) for the table.
- **Data inventory**: the three datasets contain only `(user_index, item_index)`
  binary interactions — **no timestamps, no location, no category, no session,
  no mapping back to raw IDs**. See [DATA_INVENTORY.md](DATA_INVENTORY.md).
- **Framework**: full map of entry points, loaders, splitters, evaluator,
  hyperparameter search — see [CODE_ANALYSIS.md](CODE_ANALYSIS.md).

## What is open / not done in Phase 1

- AmazonBook and Tmall reproducibility checks not run yet (will run on the GPU
  machine alongside DCCF/BIGCF in Phase 2).
- DCCF and BIGCF training not executed locally (no NVIDIA GPU on the laptop).
- The choice between **Plan A** (no context, FM degenerates to user×item),
  **Plan B** (rebuild dataset from raw with contextual fields), **Plan C**
  (Plan B + re-tune all baselines) is **left open for discussion before Phase 3**.

See [PHASE1_SUMMARY.md](PHASE1_SUMMARY.md) for the full open-decisions list.

---

## Citing the upstream paper

```bibtex
@inproceedings{shehzad2025worrying,
  title={A Worrying Reproducibility Study of Intent-Aware Recommendation Models},
  author={Shehzad, Faisal and Ferrari Dacrema, Maurizio and Jannach, Dietmar},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on
             Research and Development in Information Retrieval},
  year={2025}
}
```
