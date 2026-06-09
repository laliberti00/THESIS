"""step02 / baselines — CF baseline pavement on the Foursquare URM views.

This module is the orchestrator behind ``experiments/run_baselines.py``. It
loads the dual-view URMs produced by step01, runs a Bayesian hyper-parameter
search on the val URM (skopt.gp_minimize, n_cases=100, n_random_starts=5,
optimise Recall@20), refits the recommender on (train + val) with the best
hyper-parameters, evaluates on the test URM with save_per_user=True so we get
the per-user metric vectors, and exports them to disk in the standard
.npz schema the statistical-validation pipeline consumes.

Models covered (the 8 context-blind baselines from the brief):
    Random, TopPop, ItemKNN, UserKNN, P3α, RP3β, EASE^R, FM-vanilla.

Random and TopPop have nothing to tune. The others go through Bayesian search.

Outputs per city (e.g. outputs/NYC/baselines/):
    <Model>.npz                per-user metric arrays + user_ids
    <Model>.best_hp.json       chosen hyper-parameters
    <Model>.summary.json       aggregates, train/eval times, search trace
    floor_table.csv / .md      pivoted table across all 8 models
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sps
import yaml
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real

from engine.Evaluation.Evaluator import EvaluatorHoldout
from engine.Recommenders.NonPersonalizedRecommender import Random, TopPop
from engine.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from engine.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from engine.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from engine.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from engine.Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from engine.Recommenders.FactorizationMachines.FMRecommender import FMRecommender

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED = REPO_ROOT / "data" / "processed"
PROTOCOL_YAML = REPO_ROOT / "config" / "protocol.yaml"


# ---------------------------------------------------------------------------
# Search-space definitions (skopt). Random and TopPop are absent on purpose.
# KNN spaces kept narrower than full Shehzad (cosine only) — on Foursquare
# with <3k items the extra knobs cost wallclock for little benefit.
# ---------------------------------------------------------------------------

SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "ItemKNN": {
        "topK":       Integer(5, 1000),
        "shrink":     Integer(0, 1000),
        "similarity": Categorical(["cosine"]),
        "normalize":  Categorical([True, False]),
    },
    "UserKNN": {
        "topK":       Integer(5, 1000),
        "shrink":     Integer(0, 1000),
        "similarity": Categorical(["cosine"]),
        "normalize":  Categorical([True, False]),
    },
    "P3alpha": {
        "topK":                  Integer(5, 1000),
        "alpha":                 Real(0.0, 2.0, prior="uniform"),
        "normalize_similarity":  Categorical([True, False]),
    },
    "RP3beta": {
        "topK":                  Integer(5, 1000),
        "alpha":                 Real(0.0, 2.0, prior="uniform"),
        "beta":                  Real(0.0, 2.0, prior="uniform"),
        "normalize_similarity":  Categorical([True, False]),
    },
    "EASE_R": {
        "l2_norm":               Real(1.0, 1e7, prior="log-uniform"),
    },
    "FM": {
        "n_components":          Categorical([32, 64, 128]),
        "learning_rate":         Real(1e-4, 1e-1, prior="log-uniform"),
        "user_alpha":            Real(1e-6, 1e-2, prior="log-uniform"),
        "item_alpha":            Real(1e-6, 1e-2, prior="log-uniform"),
        "n_epochs":              Integer(5, 20),
    },
}

CLASS_MAP: dict[str, type] = {
    "Random":   Random,
    "TopPop":   TopPop,
    "ItemKNN":  ItemKNNCFRecommender,
    "UserKNN":  UserKNNCFRecommender,
    "P3alpha":  P3alphaRecommender,
    "RP3beta":  RP3betaRecommender,
    "EASE_R":   EASE_R_Recommender,
    "FM":       FMRecommender,
}

DEFAULT_MODEL_ORDER = [
    "Random", "TopPop", "ItemKNN", "UserKNN",
    "P3alpha", "RP3beta", "EASE_R", "FM",
]


# ---------------------------------------------------------------------------
# Protocol loader
# ---------------------------------------------------------------------------

@dataclass
class Protocol:
    cutoffs: list[int]
    cutoff_to_optimize: int
    metric_to_optimize: str
    n_cases: int
    n_random_starts: int
    exclude_seen: bool
    multi_seed_set: list[int]

    @classmethod
    def from_yaml(cls, path: Path) -> "Protocol":
        cfg = yaml.safe_load(open(path))
        return cls(
            cutoffs=cfg["evaluation"]["cutoff_list"],
            cutoff_to_optimize=cfg["evaluation"]["cutoff_to_optimize"],
            metric_to_optimize=cfg["evaluation"]["metric_to_optimize"],
            n_cases=cfg["hyperparameter_search"]["n_cases"],
            n_random_starts=cfg["hyperparameter_search"]["n_random_starts"],
            exclude_seen=cfg["evaluation"]["exclude_seen"],
            multi_seed_set=cfg["seeds"]["multi_seed_set"],
        )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_urm_views(city: str) -> tuple[sps.csr_matrix, sps.csr_matrix, sps.csr_matrix]:
    """Load the 3 URMs produced by step01 for the given city."""
    base = PROCESSED / city
    if not base.exists():
        raise FileNotFoundError(
            f"{base} not found. Run step01 first: "
            f"python -m experiments.run_preprocessing --city {city}")
    train = sps.load_npz(base / "URM_train.npz").tocsr()
    val = sps.load_npz(base / "URM_val.npz").tocsr()
    test = sps.load_npz(base / "URM_test.npz").tocsr()
    return train, val, test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce(kwargs: dict) -> dict:
    """Convert skopt return types (numpy scalars / bool) to plain Python."""
    out = {}
    for k, v in kwargs.items():
        if isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        elif isinstance(v, np.bool_):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


def _build_recommender(model_name: str, urm_train) -> object:
    cls = CLASS_MAP[model_name]
    if "verbose" in cls.__init__.__code__.co_varnames:
        return cls(urm_train, verbose=False)
    return cls(urm_train)


def _evaluate_quick(rec_class, fit_kwargs: dict, urm_train, urm_val,
                    cutoff_to_optimize: int,
                    metric_to_optimize: str,
                    exclude_seen: bool) -> float:
    """Fit on urm_train, evaluate on urm_val, return target metric. 0.0 on
    any failure (degenerate HP combination)."""
    try:
        rec = rec_class(urm_train, verbose=False) \
            if "verbose" in rec_class.__init__.__code__.co_varnames \
            else rec_class(urm_train)
        rec.fit(**fit_kwargs)
        evaluator = EvaluatorHoldout(urm_val, [cutoff_to_optimize],
                                      exclude_seen=exclude_seen,
                                      verbose=False, save_per_user=False)
        df, _ = evaluator.evaluateRecommender(rec)
        val = float(df.loc[cutoff_to_optimize, metric_to_optimize])
        return val if np.isfinite(val) else 0.0
    except Exception as e:
        logger.debug("  HP candidate failed: %s", e)
        return 0.0


# ---------------------------------------------------------------------------
# Bayesian tuning
# ---------------------------------------------------------------------------

def bayesian_tune(model_name: str,
                  urm_train: sps.csr_matrix,
                  urm_val: sps.csr_matrix,
                  protocol: Protocol,
                  seed: int = 42,
                  ) -> tuple[dict, list[float]]:
    """Bayesian search over the search-space of model_name. Returns
    (best_hp_kwargs, score_trace)."""
    if model_name not in SEARCH_SPACES:
        return {}, []
    space_dict = SEARCH_SPACES[model_name]
    dim_names = list(space_dict.keys())
    dimensions = [space_dict[n] for n in dim_names]
    rec_class = CLASS_MAP[model_name]
    trace: list[float] = []

    def objective(x):
        kwargs = _coerce(dict(zip(dim_names, x)))
        if model_name == "FM":
            kwargs["negative_sampling_seed"] = seed
        score = _evaluate_quick(
            rec_class, kwargs, urm_train, urm_val,
            protocol.cutoff_to_optimize,
            protocol.metric_to_optimize,
            protocol.exclude_seen,
        )
        trace.append(score)
        return -score

    t0 = time.time()
    logger.info("[tune %s] n_cases=%d n_random=%d ...",
                model_name, protocol.n_cases, protocol.n_random_starts)
    result = gp_minimize(
        objective, dimensions,
        n_calls=protocol.n_cases,
        n_initial_points=protocol.n_random_starts,
        random_state=seed,
        acq_func="gp_hedge",
        verbose=False,
    )
    best_kwargs = _coerce(dict(zip(dim_names, result.x)))
    if model_name == "FM":
        best_kwargs["negative_sampling_seed"] = seed
    elapsed = time.time() - t0
    logger.info("[tune %s] best %s@%d=%.5f  hp=%s  (%.1fs)",
                model_name, protocol.metric_to_optimize,
                protocol.cutoff_to_optimize, -result.fun,
                best_kwargs, elapsed)
    return best_kwargs, trace


# ---------------------------------------------------------------------------
# Final refit + test + export
# ---------------------------------------------------------------------------

def train_eval_export(model_name: str,
                      best_hp: dict,
                      urm_train_final: sps.csr_matrix,
                      urm_test: sps.csr_matrix,
                      protocol: Protocol,
                      out_dir: Path,
                      random_seed: int) -> dict:
    """Refit on urm_train_final (= train ∪ val) with best_hp, evaluate on
    urm_test with save_per_user, dump <Model>.npz + .best_hp.json + .summary.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = _build_recommender(model_name, urm_train_final)

    fit_kwargs = dict(best_hp)
    if model_name == "Random":
        fit_kwargs = {"random_seed": random_seed}

    t0 = time.time()
    if fit_kwargs:
        rec.fit(**fit_kwargs)
    else:
        rec.fit()
    train_time = time.time() - t0

    evaluator = EvaluatorHoldout(urm_test, protocol.cutoffs,
                                  exclude_seen=protocol.exclude_seen,
                                  verbose=False, save_per_user=True)
    t1 = time.time()
    results_df, _ = evaluator.evaluateRecommender(rec)
    eval_time = time.time() - t1

    payload: dict[str, np.ndarray] = {"user_ids": evaluator.per_user_user_ids}
    for cutoff in protocol.cutoffs:
        for metric in evaluator.PER_USER_METRICS:
            payload[f"{metric}_{cutoff}"] = evaluator.per_user_metrics[cutoff][metric]
    np.savez_compressed(out_dir / f"{model_name}.npz", **payload)

    (out_dir / f"{model_name}.best_hp.json").write_text(
        json.dumps(best_hp, indent=2, default=str)
    )

    summary = {
        "model": model_name,
        "train_time_s": train_time,
        "eval_time_s": eval_time,
        "n_users_evaluated": int(len(evaluator.per_user_user_ids)),
        "aggregates": {str(c): {m: float(results_df.loc[c, m])
                                  for m in results_df.columns
                                  if m in {"PRECISION", "RECALL", "MAP", "MRR",
                                            "NDCG", "F1", "NOVELTY", "COVERAGE_ITEM"}}
                       for c in protocol.cutoffs},
    }
    (out_dir / f"{model_name}.summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


# ---------------------------------------------------------------------------
# Floor table builder
# ---------------------------------------------------------------------------

def build_floor_table(summaries: dict[str, dict],
                      protocol: Protocol,
                      out_dir: Path) -> pd.DataFrame:
    """Build the per-city floor table (row per model) and write CSV + MD."""
    rows = []
    for model in DEFAULT_MODEL_ORDER:
        if model not in summaries:
            continue
        s = summaries[model]
        row = {"model": model}
        for c in protocol.cutoffs:
            agg = s["aggregates"][str(c)]
            row[f"R@{c}"] = agg["RECALL"]
            row[f"NDCG@{c}"] = agg["NDCG"]
        row["train_s"] = s["train_time_s"]
        row["eval_s"] = s["eval_time_s"]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(out_dir / "floor_table.csv", float_format="%.5f")
    try:
        df.to_markdown(out_dir / "floor_table.md", floatfmt=".4f")
    except ImportError:
        # tabulate not installed; fall back to simple markdown
        with open(out_dir / "floor_table.md", "w") as f:
            f.write(df.round(4).to_markdown(floatfmt=".4f") if False
                    else "model | " + " | ".join(df.columns) + "\n")
            f.write("--- | " + " | ".join("---" for _ in df.columns) + "\n")
            for idx, r in df.iterrows():
                f.write(f"{idx} | " + " | ".join(f"{v:.4f}" for v in r) + "\n")
    return df


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_floor_for_city(city: str,
                       models: list[str] | None = None,
                       skip_tuning: bool = False,
                       seed: int | None = None) -> dict[str, dict]:
    """End-to-end floor for one city."""
    if models is None:
        models = DEFAULT_MODEL_ORDER

    protocol = Protocol.from_yaml(PROTOCOL_YAML)
    if seed is None:
        seed = protocol.multi_seed_set[0]
    out_dir = REPO_ROOT / "outputs" / city / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    urm_train, urm_val, urm_test = load_urm_views(city)
    logger.info("[%s] URM_train %s nnz=%d  URM_val nnz=%d  URM_test nnz=%d",
                city, urm_train.shape, urm_train.nnz, urm_val.nnz, urm_test.nnz)

    # union of train+val, binarised (>0 → 1.0)
    urm_sum = (urm_train + urm_val).tocsr()
    urm_sum.data[:] = 1.0
    urm_train_final = urm_sum.astype(np.float32)
    logger.info("[%s] URM_train_final (train ∪ val) nnz=%d",
                city, urm_train_final.nnz)

    summaries: dict[str, dict] = {}
    for model in models:
        t_total = time.time()
        if model in ("Random", "TopPop") or skip_tuning:
            best_hp: dict = {}
            trace: list[float] = []
        else:
            best_hp, trace = bayesian_tune(
                model, urm_train, urm_val, protocol, seed=seed
            )
        summary = train_eval_export(
            model, best_hp, urm_train_final, urm_test, protocol,
            out_dir, random_seed=seed
        )
        summary["best_hp"] = best_hp
        summary["tune_trace_len"] = len(trace)
        summary["tune_best_val_metric"] = max(trace) if trace else None
        summary["wallclock_total_s"] = time.time() - t_total
        summaries[model] = summary
        r20 = summary["aggregates"][str(protocol.cutoff_to_optimize)]["RECALL"]
        n20 = summary["aggregates"][str(protocol.cutoff_to_optimize)]["NDCG"]
        logger.info("[%s] %s  R@20=%.4f  N@20=%.4f  total=%.1fs",
                    city, model, r20, n20, summary["wallclock_total_s"])
    build_floor_table(summaries, protocol, out_dir)
    return summaries
