"""experiments/run_baselines.py — CF baseline pavement for the new Foursquare
single-phase plan.

Loads the URMs produced by step01 (data/processed/<city>/), runs the 8
context-blind baselines (Random, TopPop, ItemKNN, UserKNN, P3α, RP3β, EASE^R,
FM-vanilla), with Bayesian search on Recall@20 over the val split for those
that have hyper-parameters, refits on (train ∪ val), evaluates on test with
``save_per_user=True``, exports per-user .npz in the standard schema, and
writes the per-city floor_table.{csv,md}.

The orchestration lives in ``pipeline.step02_models.baselines``; this CLI is
a thin wrapper.

Examples:
    python -m experiments.run_baselines --city NYC -v
    python -m experiments.run_baselines --city TKY -v
    python -m experiments.run_baselines --city both

The legacy synthetic-URM entry point used by the smoke test is also kept
(see ``run_baselines(URM_train, URM_test, ...)``) so ``tests/test_smoke.py``
remains green.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.Evaluation.Evaluator import EvaluatorHoldout
from engine.Recommenders.NonPersonalizedRecommender import Random, TopPop
from engine.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from engine.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from engine.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from engine.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from engine.Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

# Cutoff list inherited from Shehzad protocol (see config/protocol.yaml).
DEFAULT_CUTOFFS = [1, 5, 10, 20, 40, 50, 100]

# Used by the smoke test (synthetic URM) and any caller wanting to skip tuning.
CLASS_MAP: dict[str, type] = {
    "Random": Random,
    "TopPop": TopPop,
    "ItemKNN": ItemKNNCFRecommender,
    "UserKNN": UserKNNCFRecommender,
    "P3alpha": P3alphaRecommender,
    "RP3beta": RP3betaRecommender,
    "EASE_R": EASE_R_Recommender,
}


# ---------------------------------------------------------------------------
# Legacy entry: synthetic / pre-built URM driver used by the smoke test.
# ---------------------------------------------------------------------------

def run_baselines(URM_train,
                  URM_test,
                  models: list[str] | None = None,
                  fit_params_by_model: dict[str, dict] | None = None,
                  cutoffs: list[int] | None = None,
                  save_per_user: bool = False,
                  out_dir: Path | None = None,
                  exclude_seen: bool = True,
                  verbose: bool = True) -> list[dict]:
    """Pre-built-URM driver. NOT used in the new floor pipeline (which goes
    via ``pipeline.step02_models.baselines.run_floor_for_city``), kept only
    so the smoke test can exercise engine + evaluator + .npz dump end to end
    without any disk I/O.
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
        rec = cls(URM_train, verbose=verbose) \
            if "verbose" in cls.__init__.__code__.co_varnames else cls(URM_train)
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


# ---------------------------------------------------------------------------
# CLI entry point (the new Foursquare floor)
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", default="both",
                        help="Built-in: NYC, TKY, both. Arbitrary strings ok "
                             "if data/processed/<city>/ exists (e.g. TKY_BAL).")
    parser.add_argument("--models", default=None,
                        help="Comma-separated subset of models. Default: all 8.")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Use class defaults (debug only).")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for Bayesian search and Random/FM. "
                             "Defaults to protocol.yaml multi_seed_set[0].")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    from pipeline.step02_models.baselines import (
        run_floor_for_city, DEFAULT_MODEL_ORDER,
    )

    selected_models = (args.models.split(",")
                       if args.models else DEFAULT_MODEL_ORDER)
    cities = ["NYC", "TKY"] if args.city == "both" else [args.city]

    overall: dict[str, dict[str, dict]] = {}
    for city in cities:
        print(f"\n>>> Floor for {city}")
        t0 = time.time()
        summaries = run_floor_for_city(
            city, models=selected_models,
            skip_tuning=args.skip_tuning, seed=args.seed,
        )
        elapsed = time.time() - t0
        overall[city] = summaries
        out_dir = REPO_ROOT / "outputs" / city / "baselines"
        print(f"    {city} floor done in {elapsed:.1f}s")
        print(f"    table: {out_dir / 'floor_table.md'}")
        # quick console table
        print(f"\n    {'model':12s}  {'R@20':>8s}  {'NDCG@20':>8s}  {'tot_s':>7s}")
        for m, s in summaries.items():
            r20 = s["aggregates"]["20"]["RECALL"]
            n20 = s["aggregates"]["20"]["NDCG"]
            print(f"    {m:12s}  {r20:>8.4f}  {n20:>8.4f}  "
                  f"{s['wallclock_total_s']:>7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
