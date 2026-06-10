"""Backbone scoring helpers for X-SAGE.

The brief defines:
    * ``B_blind`` — the SARE-faithful, context-BLIND backbone the situation
      modulates. We use the floor's ``FMRecommender`` (vanilla MF) refit on
      ``train ∪ val`` with the HPs saved by the orchestrator's Bayesian search.
    * ``EASE^R`` — secondary robustness check, same data, different
      architecture.

Both come from ``engine.Recommenders.*`` and use the URM only — perfect for
the SARE-faithful design (any context must enter through the X-SAGE head).

This module exposes:
    refit_backbone(city, model_name, ...)  →  scores (n_users, n_items)
    load_or_refit(city, model_name, ...)    →  cached on disk under
        ``outputs/<city>/xsage/backbone/<model>.scores.npy`` (+ .meta.json)

Reading the score matrix at inference is cheap: per-request top-K is just
``np.argpartition`` on the user's row after masking the train∪val history.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR_TEMPLATE = "outputs/{city}/baselines"


def _load_best_hp(city: str, model_name: str) -> dict:
    p = REPO_ROOT / BASELINE_DIR_TEMPLATE.format(city=city) / f"{model_name}.best_hp.json"
    if not p.exists():
        raise FileNotFoundError(
            f"No best_hp.json found at {p}. Re-run the floor on {city} first:\n"
            f"    python -m experiments.run_baselines --city {city}")
    return json.loads(p.read_text(encoding="utf-8"))


def refit_backbone(city: str, model_name: str = "FM",
                     out_dir: Path | None = None,
                     verbose: bool = False) -> dict:
    """Refit the requested floor model on ``URM_train ∪ URM_val`` with the
    Bayesian-search HPs, and dump the full per-user / per-item score matrix.

    Args:
        city:          ``"NYC"`` or ``"TKY"``.
        model_name:    ``"FM"`` (B_blind) or ``"EASE_R"`` (robustness).
        out_dir:       defaults to ``outputs/<city>/xsage/backbone/``.

    Returns:
        ``{"scores": (n_users, n_items) float32, "model": <recommender>,
          "n_users", "n_items", "wallclock_s"}``
    """
    p = REPO_ROOT / "data" / "processed" / city
    urm_train = sps.load_npz(p / "URM_train.npz").tocsr()
    urm_val = sps.load_npz(p / "URM_val.npz").tocsr()
    n_users, n_items = urm_train.shape
    urm = (urm_train + urm_val).tocsr(); urm.data[:] = 1.0
    urm = urm.astype(np.float32)

    hp = _load_best_hp(city, model_name)
    if verbose:
        print(f"    refitting {model_name} on {city} train+val with HP={hp}")

    t0 = time.time()
    if model_name == "FM":
        from engine.Recommenders.FactorizationMachines.FMRecommender import FMRecommender
        # _BPRMFBase swallows the extras (n_components etc.) — pass them through.
        rec = FMRecommender(urm)
        rec.fit(**hp)
    elif model_name == "EASE_R":
        from engine.Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
        rec = EASE_R_Recommender(urm)
        rec.fit(**hp)
    else:
        raise ValueError(f"Unknown backbone model_name {model_name!r}")
    fit_s = time.time() - t0

    # Full per-user score matrix.
    t0 = time.time()
    scores = rec._compute_item_score(np.arange(n_users)).astype(np.float32)
    if scores.ndim == 1:
        # Some recommenders return (n_items,) when a single user is requested
        # — but np.arange(n_users) means batch mode. Just in case:
        scores = scores.reshape(n_users, n_items)
    score_s = time.time() - t0

    if verbose:
        print(f"    refit: {fit_s:.1f}s  | full-score dump: {score_s:.1f}s  | "
              f"matrix shape={scores.shape}")

    if out_dir is None:
        out_dir = REPO_ROOT / "outputs" / city / "xsage" / "backbone"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{model_name}.scores.npy", scores)
    (out_dir / f"{model_name}.meta.json").write_text(json.dumps({
        "model": model_name,
        "city": city,
        "hp": hp,
        "n_users": int(n_users),
        "n_items": int(n_items),
        "fit_seconds": float(fit_s),
        "score_dump_seconds": float(score_s),
    }, indent=2), encoding="utf-8")

    return {
        "scores": scores, "model": rec,
        "n_users": int(n_users), "n_items": int(n_items),
        "wallclock_s": fit_s + score_s,
        "hp": hp,
    }


def load_or_refit(city: str, model_name: str = "FM",
                    refresh: bool = False, verbose: bool = False) -> np.ndarray:
    """Load cached score matrix; refit on cache miss (or if ``refresh=True``)."""
    cache_path = REPO_ROOT / "outputs" / city / "xsage" / "backbone" \
                 / f"{model_name}.scores.npy"
    if cache_path.exists() and not refresh:
        if verbose:
            print(f"    using cached scores: {cache_path}")
        return np.load(cache_path).astype(np.float32)
    return refit_backbone(city, model_name=model_name, verbose=verbose)["scores"]


def excluded_mask(city: str, n_items: int) -> sps.csr_matrix:
    """``URM_train + URM_val`` binarised — items to exclude from ranking per
    user (per the floor's protocol)."""
    p = REPO_ROOT / "data" / "processed" / city
    mask = (sps.load_npz(p / "URM_train.npz")
              + sps.load_npz(p / "URM_val.npz")).tocsr()
    mask.data[:] = 1.0
    return mask
