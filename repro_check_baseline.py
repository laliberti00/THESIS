"""Reproducibility check for non-neural baselines on Shehzad's three DCCF datasets.

Mirrors what `run_experiments_for_DCCF_original_baselines.py` does for the
baseline branch (no DCCF training), but cleanly logs each model's results to
`repro_check_results/<dataset>_<model>.txt`.

Usage:
    python repro_check_baseline.py --dataset gowalla --model RP3beta
    python repro_check_baseline.py --dataset gowalla --model all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from topn_baselines_neurals.Data_manager.Gowalla_AmazonBook_Tmall_DCCF import (
    Gowalla_AmazonBook_Tmall_DCCF,
)
from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout
from topn_baselines_neurals.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from topn_baselines_neurals.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from topn_baselines_neurals.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from topn_baselines_neurals.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from topn_baselines_neurals.Recommenders.NonPersonalizedRecommender import Random, TopPop

# THESIS PHASE-2 ADD-ON: import the new BPR-MF based vanilla recommenders only
# on demand (their torch dependency is optional for the non-neural baselines).
def _import_FM():
    from topn_baselines_neurals.Recommenders.FactorizationMachines.FMRecommender import FMRecommender
    return FMRecommender

def _import_CAMF():
    from topn_baselines_neurals.Recommenders.CAMF.CAMFRecommender import CAMFRecommender
    return CAMFRecommender

# Best HP values as hard-coded in run_experiments_for_DCCF_original_baselines.py
# (do NOT trust docs/tables_window/tables_window_DCCF.html — see DATA_INVENTORY.md).
# FM and CAMF use modest defaults documented in their _BPRMFBase.HP_SEARCH_SPACE.
# These are *not* the result of a Bayesian search — they will be tuned in Phase 3.
BEST_HP = {
    "gowalla": {
        "ItemKNN": {"topK": 508, "similarity": "cosine"},
        "UserKNN": {"topK": 146, "similarity": "cosine"},
        "P3alpha": {
            "topK": 777,
            "alpha": 1.087096950563704,
            "normalize_similarity": False,
        },
        "RP3beta": {
            "topK": 777,
            "alpha": 0.5663562161452378,
            "beta": 0.001085447926739258,
            "normalize_similarity": True,
        },
        "FM":   {"n_components": 64, "learning_rate": 0.01, "user_alpha": 1e-5,
                 "item_alpha": 1e-5, "n_epochs": 10, "batch_size": 4096,
                 "negative_sampling_seed": 2026},
        "CAMF": {"n_components": 64, "learning_rate": 0.01, "user_alpha": 1e-5,
                 "item_alpha": 1e-5, "n_epochs": 10, "batch_size": 4096,
                 "negative_sampling_seed": 2026},
    },
    "amazonBook": {
        "ItemKNN": {"topK": 125, "similarity": "cosine"},
        "UserKNN": {"topK": 454, "similarity": "cosine"},
        "P3alpha": {
            "topK": 496,
            "alpha": 0.41477903655656115,
            "normalize_similarity": False,
        },
        "RP3beta": {
            "topK": 496,
            "alpha": 0.44477903655656115,
            "beta": 0.5968193614337285,
            "normalize_similarity": True,
        },
    },
    "tmall": {
        "ItemKNN": {"topK": 516, "similarity": "cosine"},
        "UserKNN": {"topK": 454, "similarity": "cosine"},
        "P3alpha": {"topK": 100, "alpha": 1, "normalize_similarity": False},
        "RP3beta": {
            "topK": 350,
            "alpha": 0.7681732734954694,
            "beta": 0.4181395996963926,
            "normalize_similarity": True,
        },
    },
}

# Models to run by default ("all"). EASE^R intentionally skipped:
# the dense (n_items × n_items) Gram matrix needs >16 GB RAM for our three datasets.
# FM and CAMF are NOT in "all": they require torch, run slower, and are exercised
# separately via --model FM / --model CAMF. Phase 3 will tune their HPs.
DEFAULT_MODELS = ["Random", "TopPop", "ItemKNN", "UserKNN", "P3alpha", "RP3beta"]

CLASS_MAP = {
    "Random": Random,
    "TopPop": TopPop,
    "ItemKNN": ItemKNNCFRecommender,
    "UserKNN": UserKNNCFRecommender,
    "P3alpha": P3alphaRecommender,
    "RP3beta": RP3betaRecommender,
    # lazily resolved (avoid mandatory torch import for non-FM users)
    "FM":      None,
    "CAMF":    None,
}


def _resolve_class(name: str):
    cls = CLASS_MAP[name]
    if cls is None:
        if name == "FM":
            cls = _import_FM()
        elif name == "CAMF":
            cls = _import_CAMF()
        CLASS_MAP[name] = cls
    return cls


def load(dataset: str):
    data_path = (Path(__file__).resolve().parent / "data" / "DCCF" / dataset).resolve()
    URM_train, URM_test = Gowalla_AmazonBook_Tmall_DCCF()._load_data_from_give_files(
        data_path, validation=False
    )
    return URM_train, URM_test


def run_one(dataset: str, model_name: str, URM_train, URM_test, out_dir: Path,
            save_per_user: bool = False) -> dict:
    print(f"\n>>> {dataset} / {model_name}{' (+per-user)' if save_per_user else ''}")
    rec_class = _resolve_class(model_name)
    rec = rec_class(URM_train)
    fit_params = BEST_HP.get(dataset, {}).get(model_name, {})

    t0 = time.time()
    rec.fit(**fit_params)
    train_time = time.time() - t0

    evaluator = EvaluatorHoldout(
        URM_test, [1, 5, 10, 20, 40, 50, 100],
        exclude_seen=True, save_per_user=save_per_user,
    )
    t0 = time.time()
    results_df, results_str = evaluator.evaluateRecommender(rec)
    eval_time = time.time() - t0

    results_df["TrainingTime(s)"] = [train_time] + [0] * (results_df.shape[0] - 1)
    results_df["EvalTime(s)"] = [eval_time] + [0] * (results_df.shape[0] - 1)

    out_path = out_dir / f"{dataset}_{model_name}.txt"
    results_df.to_csv(out_path, sep="\t", index=True, index_label="cutoff")
    print(f"    Train {train_time:.1f}s | Eval {eval_time:.1f}s | -> {out_path.name}")
    print("    " + results_str.replace("\n", "\n    "))

    # === Phase 2 add-on: dump per-user metrics to .npz ====================
    if save_per_user:
        per_user_dir = out_dir / "per_user"
        per_user_dir.mkdir(parents=True, exist_ok=True)
        # Each .npz holds one array per (cutoff, metric) keyed as
        # "<metric>_<cutoff>" plus a "user_ids" array of the same length.
        npz_payload = {"user_ids": evaluator.per_user_user_ids}
        for cutoff in [1, 5, 10, 20, 40, 50, 100]:
            for metric in evaluator.PER_USER_METRICS:
                npz_payload[f"{metric}_{cutoff}"] = evaluator.per_user_metrics[cutoff][metric]
        npz_path = per_user_dir / f"{dataset}_{model_name}.npz"
        np.savez_compressed(npz_path, **npz_payload)
        print(f"    per-user → {npz_path.relative_to(out_dir.parent)}  "
              f"({len(evaluator.per_user_user_ids)} users × "
              f"{len(evaluator.PER_USER_METRICS)} metrics × {len([1,5,10,20,40,50,100])} cutoffs)")
    # ======================================================================

    return {
        "dataset": dataset,
        "model": model_name,
        "train_time_s": train_time,
        "eval_time_s": eval_time,
        "recall_at_20": float(results_df.loc[20, "RECALL"]),
        "ndcg_at_20": float(results_df.loc[20, "NDCG"]),
        "fit_params": fit_params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["gowalla", "amazonBook", "tmall"])
    parser.add_argument("--model", default="all", help="Model name, or 'all'")
    parser.add_argument("--out-dir", default="repro_check_results")
    parser.add_argument("--save-per-user", action="store_true",
                        help="Also dump per-user metrics to <out-dir>/per_user/*.npz")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset}...")
    t0 = time.time()
    URM_train, URM_test = load(args.dataset)
    print(f"Loaded in {time.time()-t0:.1f}s. URM_train nnz={URM_train.nnz}, URM_test nnz={URM_test.nnz}")

    if args.model == "all":
        models = DEFAULT_MODELS
    else:
        models = [args.model]

    summary = []
    for m in models:
        try:
            summary.append(run_one(args.dataset, m, URM_train, URM_test, out_dir,
                                   save_per_user=args.save_per_user))
        except Exception as e:
            print(f"!!! {args.dataset}/{m} failed: {type(e).__name__}: {e}")
            summary.append({"dataset": args.dataset, "model": m, "error": str(e)})

    print("\n=== SUMMARY ===")
    for row in summary:
        if "error" in row:
            print(f"  {row['dataset']:>10s} / {row['model']:<10s}  ERROR  {row['error']}")
        else:
            print(
                f"  {row['dataset']:>10s} / {row['model']:<10s}  "
                f"R@20={row['recall_at_20']:.6f}  N@20={row['ndcg_at_20']:.6f}  "
                f"train={row['train_time_s']:.1f}s eval={row['eval_time_s']:.1f}s"
            )


if __name__ == "__main__":
    main()
