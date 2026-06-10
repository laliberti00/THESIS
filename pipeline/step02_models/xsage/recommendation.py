"""Recommendation layer — backbone score ``p_B`` + situational head ``p_S`` →
confidence-weighted harmonic combiner ``p̂`` (eqs.13-15), then top-K ranking
and list displacement ``Δ_K`` (eq.16).

We deliberately keep this layer thin and operate on per-request arrays so the
caller can mix it with any backbone (FM-vanilla, EASE^R, or the future B_full
context-aware FM) without touching this module.
"""
from __future__ import annotations

import numpy as np

from .metrics import topk_from_scores


# ---------------------------------------------------------------------------
# Probability distributions over items
# ---------------------------------------------------------------------------

def softmax_scores(scores_per_item: np.ndarray, tau: float = 1.0,
                     eps: float = 1e-12) -> np.ndarray:
    """``p ∝ exp(scores / τ)``. Stabilised by subtracting the per-row max.

    Accepts 1D (single request) or 2D (B, I). Returns same shape, rows sum
    to 1.
    """
    s = np.asarray(scores_per_item, dtype=np.float64) / max(tau, 1e-9)
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return (e / np.maximum(e.sum(axis=-1, keepdims=True), eps)).astype(np.float32)


# ---------------------------------------------------------------------------
# Shrunken per-situation category biases  b^{(k)} ∈ R^{n_macros}
# ---------------------------------------------------------------------------

def fit_situation_biases(z_train: np.ndarray, cat_macro_train: np.ndarray,
                           K: int, n_macros: int,
                           lam: float = 50.0) -> np.ndarray:
    """``b^{(k)}_c = log( (n_kc + lam · p_c) / (n_k. + lam) )`` − ``log p_c``.

    Shrinkage toward the global category distribution ``p_c`` controlled by
    ``lam`` (the effective pseudo-count). For ``lam → ∞`` we recover the
    backbone (all biases ≈ 0).
    """
    z_train = z_train.astype(np.int64)
    cat_macro_train = cat_macro_train.astype(np.int64)
    # global cat distribution
    counts_global = np.bincount(cat_macro_train, minlength=n_macros).astype(np.float64)
    p_global = counts_global / counts_global.sum()
    # per-situation counts
    counts_k = np.zeros((K, n_macros), dtype=np.float64)
    np.add.at(counts_k, (z_train, cat_macro_train), 1)
    # Bayesian shrinkage
    smoothed = (counts_k + lam * p_global[None, :])
    smoothed = smoothed / smoothed.sum(axis=1, keepdims=True)
    b = np.log(smoothed) - np.log(p_global + 1e-12)
    return b.astype(np.float32)


# ---------------------------------------------------------------------------
# Situational item score s_S(u, i) = Σ_k r_k · b^{(k)}_{cat(i)}
# ---------------------------------------------------------------------------

def situational_item_scores(membership_per_request: np.ndarray,
                              biases_per_situation: np.ndarray,
                              item_cat_macro: np.ndarray) -> np.ndarray:
    """For each request, return scores for every item.

    Args:
        membership_per_request:  (B, K)
        biases_per_situation:    (K, n_macros)
        item_cat_macro:          (I,) int — each item's cat_macro index.

    Returns:
        (B, I) — ``s_S(u, i) = Σ_k r_k · b^{(k)}_{cat(i)}``.
    """
    # (B, K) @ (K, n_macros) → (B, n_macros), then index per-item along last axis.
    s_per_request_macro = membership_per_request @ biases_per_situation     # (B, n_macros)
    return s_per_request_macro[:, item_cat_macro].astype(np.float32)        # (B, I)


# ---------------------------------------------------------------------------
# Confidence-weighted harmonic combiner (eq.15)
# ---------------------------------------------------------------------------

def harmonic_combine(p_B: np.ndarray, p_S: np.ndarray,
                       c_B: np.ndarray, c_S: np.ndarray,
                       eps: float = 1e-9) -> np.ndarray:
    """``p̂(i) = (c_B + c_S) / (c_B / p_B(i) + c_S / p_S(i))``.

    Handles ``c_S = 0`` (collapses to backbone) and ``c_B = 0`` (collapses to
    situational) by safe division.

    Shapes: ``p_B, p_S`` are (B, I); ``c_B, c_S`` are (B,) or (B, 1). Output
    is not normalised — it's a per-row positive score; we re-normalise to a
    probability only if the caller wants it.
    """
    c_B = np.asarray(c_B, dtype=np.float32)
    c_S = np.asarray(c_S, dtype=np.float32)
    if c_B.ndim == 1: c_B = c_B[:, None]
    if c_S.ndim == 1: c_S = c_S[:, None]
    num = c_B + c_S
    den = c_B / np.maximum(p_B, eps) + c_S / np.maximum(p_S, eps)
    # When c_S == 0 everywhere on a row, den ≈ c_B / p_B → result = p_B (as expected).
    return (num / np.maximum(den, eps)).astype(np.float32)


def backbone_confidence(p_B: np.ndarray) -> np.ndarray:
    """``c_B = 1 − 1 / (max_i p_B + 1)`` per request. (Bounded in (0, 1).)"""
    pmax = p_B.max(axis=-1)
    return 1.0 - 1.0 / (pmax + 1.0)


def situation_confidence(kappa: float, gamma_per_request: np.ndarray) -> np.ndarray:
    """``c_S = κ_S · γ_S`` where ``γ_S = 1`` on core, ``1 / |T|`` on boundary.

    γ is provided by the caller (computed from the rough-k-means result).
    """
    return kappa * gamma_per_request.astype(np.float32)
