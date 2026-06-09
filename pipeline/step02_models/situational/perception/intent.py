"""L1.b — intent perception.

For each request ``(u, t)`` we look at the user's history strictly before ``t``
(within a recency window of at most ``n_window`` rows) and summarise it as

    μ ∈ Δ^{n_macros}   time-decayed soft-histogram over cat_macro,
    Δt  (min)          minutes since the last interaction,
    ℓ   (rows)         window length,
    H                  Shannon entropy of μ (in [0, log K]),
    ρ   (km)           Σ dist_prev within the window,

then project ``[μ ‖ Δt ‖ ℓ ‖ H ‖ ρ]`` linearly to the intent vector
``e ∈ R^{d_e}``. ``Δt, ℓ, ρ`` are z-scored using *train statistics only*.

Empty window (no prior interaction) → a learned *no-intent* vector.

This module is **NumPy-side** (pre-computation) plus a small ``nn.Module`` for
the linear projection. We precompute the (B, n_macros + 3) feature matrix per
row once and look it up at training/eval time — far cheaper than walking the
history on every batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

EPS = 1e-12


@dataclass
class IntentStats:
    """Running standardisation stats for (Δt, ℓ, ρ). Estimated on *train* only.

    Empty rows (window length == 0) are excluded so their learned no-intent
    vector isn't dragged towards 0.
    """
    dt_mean: float
    dt_std: float
    len_mean: float
    len_std: float
    rho_mean: float
    rho_std: float

    @classmethod
    def from_features(cls, raw: np.ndarray, n_macros: int) -> "IntentStats":
        """Estimate stats from non-empty train rows of `raw` (B, n_macros+3)."""
        non_empty = raw[:, n_macros + 1] > 0  # ℓ > 0
        if non_empty.sum() == 0:
            return cls(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        sub = raw[non_empty]
        dt = sub[:, n_macros]
        ll = sub[:, n_macros + 1]
        rh = sub[:, n_macros + 2]
        return cls(
            dt_mean=float(dt.mean()), dt_std=float(dt.std()) or 1.0,
            len_mean=float(ll.mean()), len_std=float(ll.std()) or 1.0,
            rho_mean=float(rh.mean()), rho_std=float(rh.std()) or 1.0,
        )

    def standardise(self, raw: np.ndarray, n_macros: int) -> np.ndarray:
        """Return a copy of `raw` with (Δt, ℓ, ρ) z-scored. μ untouched.

        Empty rows (ℓ==0) get all three set to 0 (matches the no-intent path).
        """
        out = raw.copy()
        out[:, n_macros] = (out[:, n_macros] - self.dt_mean) / self.dt_std
        out[:, n_macros + 1] = (out[:, n_macros + 1] - self.len_mean) / self.len_std
        out[:, n_macros + 2] = (out[:, n_macros + 2] - self.rho_mean) / self.rho_std
        empty = raw[:, n_macros + 1] == 0
        out[empty, n_macros:n_macros + 3] = 0.0
        return out


def precompute_intent_features(rows: pd.DataFrame,
                                 history: pd.DataFrame,
                                 macro_to_idx: dict[str, int],
                                 n_window: int = 10,
                                 tau_minutes: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute the (n_macros + 3)-wide intent feature matrix for every row in
    ``rows`` using only rows in ``history`` that are *strictly earlier* (same
    user, ``time_local`` < request time).

    Args:
        rows: target dataframe (e.g. train, val or test rows). Must contain
              ``user_id``, ``time_local``.
        history: rows the model is *allowed* to remember (e.g. train; for the
                 refit step, train+val). Must contain ``user_id``, ``time_local``,
                 ``cat_macro``, ``dist_prev``.
        macro_to_idx: stable mapping cat_macro → 0..n_macros-1.
        n_window: last-N strictly-before window size.
        tau_minutes: decay constant of the soft histogram.

    Returns:
        raw_features: (B, n_macros + 3) float32 with columns
                      [μ_0, …, μ_{K-1}, Δt(min), ℓ, ρ(km)].
        empty_mask:   (B,) bool — True where the window had 0 rows.
    """
    n_macros = len(macro_to_idx)
    history_sorted = (history[["user_id", "time_local", "cat_macro", "dist_prev"]]
                       .sort_values(["user_id", "time_local"])
                       .reset_index(drop=True))
    # Group histories per user once.
    by_user: dict[int, dict[str, np.ndarray]] = {}
    for u, g in history_sorted.groupby("user_id", sort=False):
        by_user[int(u)] = {
            "t": g["time_local"].values.astype("datetime64[ns]"),
            "m": np.array([macro_to_idx.get(x, -1) for x in g["cat_macro"].values],
                          dtype=np.int32),
            "d": g["dist_prev"].values.astype(np.float32),
        }

    B = len(rows)
    out = np.zeros((B, n_macros + 3), dtype=np.float32)
    empty = np.zeros(B, dtype=bool)
    users = rows["user_id"].values.astype(np.int64)
    times = rows["time_local"].values.astype("datetime64[ns]")

    for b in range(B):
        u = int(users[b])
        t = times[b]
        h = by_user.get(u)
        if h is None:
            empty[b] = True
            continue
        # strictly before
        cut = np.searchsorted(h["t"], t, side="left")
        if cut == 0:
            empty[b] = True
            continue
        # last n_window strictly before t
        start = max(0, cut - n_window)
        t_win = h["t"][start:cut]
        m_win = h["m"][start:cut]
        d_win = h["d"][start:cut]
        # Δt minutes between (t - last window time)
        dt_min = (t - t_win[-1]).astype("timedelta64[s]").astype(np.float64) / 60.0
        # time-decay weights w.r.t. *t*
        elapsed = (t - t_win).astype("timedelta64[s]").astype(np.float64) / 60.0
        w = np.exp(-elapsed / tau_minutes)
        # soft histogram
        mu = np.zeros(n_macros, dtype=np.float64)
        valid = m_win >= 0
        if valid.any():
            np.add.at(mu, m_win[valid], w[valid])
        s = mu.sum()
        if s > 0:
            mu /= s
        out[b, :n_macros] = mu.astype(np.float32)
        out[b, n_macros] = float(dt_min)
        out[b, n_macros + 1] = float(len(t_win))
        out[b, n_macros + 2] = float(d_win.sum())
    return out, empty


class IntentEncoder(nn.Module):
    """Linear projection of the precomputed intent features → ``e``.

    The "no-intent" case (empty window) is replaced by a learned vector
    instead of mapping the all-zero feature row.

    Inputs:
        feat: (B, n_macros + 3) standardised features
        empty: (B,) bool — True ⇒ use the learned no-intent vector

    Output:
        e: (B, d_e)
    """

    def __init__(self, n_macros: int, d_e: int = 8) -> None:
        super().__init__()
        in_dim = n_macros + 3
        self.linear = nn.Linear(in_dim, d_e)
        # learned no-intent vector
        self.no_intent = nn.Parameter(torch.zeros(d_e))
        self.n_macros = n_macros
        self.d_e = d_e

    def forward(self, feat: torch.Tensor, empty: torch.Tensor) -> torch.Tensor:
        e = self.linear(feat)
        no_int_row = self.no_intent.unsqueeze(0).expand_as(e)
        return torch.where(empty.unsqueeze(-1), no_int_row, e)


def shannon_entropy(mu: np.ndarray) -> np.ndarray:
    """Vectorised row-wise entropy. ``mu`` is (B, K) with rows summing to ≤ 1.

    For an empty row (all-zero) returns 0.
    """
    safe = np.clip(mu, EPS, 1.0)
    H = -(mu * np.log(safe)).sum(axis=-1)
    return H.astype(np.float32)
