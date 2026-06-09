"""experiments/run_baselines.py — runs the engine's CF baselines on a URM.

Skeleton runner: model-agnostic on URM input. Plugs in the CF baselines that
live in `engine/Recommenders/` (TopPop, ItemKNN, UserKNN, P3α, RP3β, EASE^R).

This script does NOT yet load Foursquare data — the data loader will be
provided by `pipeline.step01_preprocessing` in the next brief. For now the
script accepts a pre-built URM via the function `run_baselines(URM_train,
URM_test, ...)` so that the smoke test can drive it end-to-end with a
synthetic URM.

Once preprocessing is in place, the CLI will load Foursquare URMs from
data/processed/<city>/ and forward them to run_baselines().
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Engine imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.Evaluation.Evaluator import EvaluatorHoldout
from engine.Recommenders.NonPersonalizedRecommender import Random, TopPop
from engine.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from engine.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from engine.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from engine.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from engine.Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

CLASS_MAP: dict[str, type] = {
    "Random": Random,
    "TopPop": TopPop,
    "ItemKNN": ItemKNNCFRecommender,
    "UserKNN": UserKNNCFRecommender,
    "P3alpha": P3alphaRecommender,
    "RP3beta": RP3betaRecommender,
    "EASE_R": EASE_R_Recommender,
}

# Cutoff list inherited from Shehzad protocol (see config/protocol.yaml)
DEFAULT_CUTOFFS = [1, 5, 10, 20, 40, 50, 100]


def run_baselines(URM_train,
                  URM_test,
                  models: list[str] | None = None,
                  fit_params_by_model: dict[str, dict] | None = None,
                  cutoffs: list[int] | None = None,
                  save_per_user: bool = False,
                  out_dir: Path | None = None,
                  exclude_seen: bool = True,
                  verbose: bool = True) -> list[dict]:
    """Run a list of CF baselines on the given URM.

    :param URM_train: scipy sparse user×item training matrix.
    :param URM_test: scipy sparse user×item test matrix (same shape).
    :param models: list of model names from ``CLASS_MAP``. Default: all.
    :param fit_params_by_model: dict model_name → kwargs passed to ``.fit()``.
    :param cutoffs: list of top-K cutoffs. Default: Shehzad set.
    :param save_per_user: forward the flag to EvaluatorHoldout. If True and
        ``out_dir`` is set, dumps per-user .npz next to the TSV.
    :param out_dir: where to write per-model TSV and (optionally) per_user .npz.
    :param exclude_seen: standard for top-N implicit-feedback eval.
    :param verbose: print per-model lines.
    """
    if models is None:
        models = list(CLASS_MAP.keys())
    if fit_params_by_model is None:
        fit_params_by_model = {}
    if cutoffs is None:
        cutoffs = DEFAULT_CUTOFFS

    summary: list[dict] = []
    for name in models:
        if name not in CLASS_MAP:
            raise ValueError(f"Unknown model: {name!r}. Choices: {list(CLASS_MAP)}")
        cls = CLASS_MAP[name]
        if verbose:
            print(f"\n>>> {name}")
        rec = cls(URM_train, verbose=verbose) if "verbose" in cls.__init__.__code__.co_varnames \
            else cls(URM_train)
        fit_params = fit_params_by_model.get(name, {})

        t0 = time.time()
        rec.fit(**fit_params)
        train_time = time.time() - t0

        evaluator = EvaluatorHoldout(URM_test, cutoffs,
                                     exclude_seen=exclude_seen,
                                     save_per_user=save_per_user,
                                     verbose=verbose)
        t0 = time.time()
        results_df, results_str = evaluator.evaluateRecommender(rec)
        eval_time = time.time() - t0

        if verbose:
            print(f"    Train {train_time:.2f}s | Eval {eval_time:.2f}s")

        # Optional persistence
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(out_dir / f"{name}.tsv", sep="\t", index=True,
                              index_label="cutoff")
            if save_per_user:
                per_user_dir = out_dir / "per_user"
                per_user_dir.mkdir(parents=True, exist_ok=True)
                payload: dict[str, np.ndarray] = {
                    "user_ids": evaluator.per_user_user_ids
                }
                for cutoff in cutoffs:
                    for metric in evaluator.PER_USER_METRICS:
                        payload[f"{metric}_{cutoff}"] = \
                            evaluator.per_user_metrics[cutoff][metric]
                np.savez_compressed(per_user_dir / f"{name}.npz", **payload)

        summary.append({
            "model": name,
            "train_time_s": train_time,
            "eval_time_s": eval_time,
            "recall_at_20": float(results_df.loc[20, "RECALL"]),
            "ndcg_at_20": float(results_df.loc[20, "NDCG"]),
        })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default=None,
                        help="Foursquare city. Once preprocessing is wired, "
                             "URM_train/URM_test will be loaded from "
                             "data/processed/<city>/.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/baselines"))
    args = parser.parse_args()

    if args.city is None:
        print("experiments/run_baselines.py — STUB CLI.")
        print("  No --city provided. Pre-built URM input via direct function "
              "call is the supported entry point until preprocessing lands.")
        return 0

    raise NotImplementedError(
        f"Foursquare loader not implemented yet (--city {args.city!r}). "
        "Use pipeline.step01_preprocessing once it is added.")


if __name__ == "__main__":
    sys.exit(main())
