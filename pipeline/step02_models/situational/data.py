"""Dataset assembly for the situation-aware FM.

Reads ``data/processed/<city>/`` (parquet + URM), builds stable vocabularies for
``cat_macro`` (over all splits) and ``geohash5`` (over all splits + a reserved
sentinel for the "no previous location" case), precomputes intent features
for every row of every split, and packs everything into NumPy arrays plus a
sparse mask the trainer / ranker can consume directly.

History rules:
    train rows → history is the train rows that precede them.
    val   rows → history is the train rows (only).
    test  rows → history is train ∪ val rows that precede them (refit step).

For each row we also store the *previous* check-in's ``geohash5`` (mapped to
the geohash vocab; 0 = no previous location for the user). This is the only
spatial signal the gate is allowed to look at — never the current row's
``geohash5``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sps

from .perception.intent import (IntentStats, precompute_intent_features,
                                  shannon_entropy)


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

def build_macro_vocab(parquets: list[pd.DataFrame]) -> dict[str, int]:
    """Stable cat_macro → index, sorted alphabetically.

    Includes every macro observed across the supplied splits (so that
    e.g. NYC's 9 macros and TKY's 8 macros are handled correctly).
    """
    macros = sorted({m for df in parquets for m in df["cat_macro"].unique()})
    return {m: i for i, m in enumerate(macros)}


def build_geohash_vocab(parquets: list[pd.DataFrame]) -> dict[str, int]:
    """Stable geohash5 → index ≥ 1. Index 0 is reserved for ``no previous``."""
    cells = sorted({g for df in parquets for g in df["geohash5"].unique()})
    return {g: i + 1 for i, g in enumerate(cells)}


# ---------------------------------------------------------------------------
# Row-level features
# ---------------------------------------------------------------------------

def map_row_macros(df: pd.DataFrame,
                    macro_to_idx: dict[str, int]) -> np.ndarray:
    """Map each row's ``cat_macro`` → its index; unknowns → -1."""
    return np.array([macro_to_idx.get(m, -1) for m in df["cat_macro"].values],
                    dtype=np.int32)


def map_prev_geohash(rows: pd.DataFrame,
                      history: pd.DataFrame,
                      geo_to_idx: dict[str, int]) -> np.ndarray:
    """For every row in ``rows`` look up the most recent strictly-prior row in
    ``history`` (same user) and return its geohash index. 0 means no prior.

    ``rows`` and ``history`` must share ``user_id``, ``time_local``, ``geohash5``.
    """
    hsorted = (history[["user_id", "time_local", "geohash5"]]
               .sort_values(["user_id", "time_local"]).reset_index(drop=True))
    by_user: dict[int, dict[str, np.ndarray]] = {}
    for u, g in hsorted.groupby("user_id", sort=False):
        by_user[int(u)] = {
            "t": g["time_local"].values.astype("datetime64[ns]"),
            "gh": g["geohash5"].values,
        }
    n = len(rows)
    out = np.zeros(n, dtype=np.int32)
    users = rows["user_id"].values.astype(np.int64)
    times = rows["time_local"].values.astype("datetime64[ns]")
    for b in range(n):
        u = int(users[b]); t = times[b]
        h = by_user.get(u)
        if h is None:
            continue
        cut = np.searchsorted(h["t"], t, side="left")
        if cut == 0:
            continue
        out[b] = geo_to_idx.get(h["gh"][cut - 1], 0)
    return out


# ---------------------------------------------------------------------------
# Bundle returned by `build_city_dataset`
# ---------------------------------------------------------------------------

@dataclass
class SplitArrays:
    """All per-row tensors needed by trainer + ranker for one split."""
    u: np.ndarray               # (B,) user idx (= u_idx)
    i: np.ndarray               # (B,) target item idx (= i_idx)
    m: np.ndarray               # (B,) cat_macro idx of the target
    c_hour: np.ndarray          # (B,) int8
    c_dow: np.ndarray           # (B,) int8
    c_month: np.ndarray         # (B,) int8 — note: 0..11 form
    c_isweekend: np.ndarray     # (B,) int8
    prev_geo: np.ndarray        # (B,) int32 — 0 if no previous
    intent_feat: np.ndarray     # (B, n_macros + 3) standardised
    intent_empty: np.ndarray    # (B,) bool


@dataclass
class CityDataset:
    """Everything the trainer/ranker need to consume one city."""
    train: SplitArrays
    val: SplitArrays
    test: SplitArrays
    n_users: int
    n_items: int
    n_macros: int
    n_geo: int                     # excluding the sentinel; embedding size = n_geo + 1
    macro_to_idx: dict[str, int]
    geo_to_idx: dict[str, int]
    item_cat_macro: np.ndarray     # (n_items,) — catalogue level
    urm_train: sps.csr_matrix      # mask for negative sampling / ranking exclusion
    urm_val: sps.csr_matrix
    urm_test: sps.csr_matrix
    stats: IntentStats             # standardisation parameters (train only)


# ---------------------------------------------------------------------------
# Catalogue-level item → cat_macro
# ---------------------------------------------------------------------------

def build_item_cat_macro(parquets: list[pd.DataFrame],
                          n_items: int,
                          macro_to_idx: dict[str, int]) -> np.ndarray:
    """For each ``i_idx`` in [0, n_items), look up the most frequent
    ``cat_macro`` across the supplied splits (concatenate train+val+test for
    full coverage).
    """
    big = pd.concat(parquets, ignore_index=True)
    counts = big.groupby(["i_idx", "cat_macro"]).size().reset_index(name="n")
    best = counts.sort_values(["i_idx", "n"], ascending=[True, False]) \
                 .drop_duplicates("i_idx", keep="first")
    out = np.full(n_items, -1, dtype=np.int32)
    for _, row in best.iterrows():
        out[int(row["i_idx"])] = macro_to_idx.get(row["cat_macro"], -1)
    if (out < 0).any():
        missing = int((out < 0).sum())
        raise ValueError(f"{missing} items have no observed cat_macro — "
                         "catalogue cannot be built")
    return out


# ---------------------------------------------------------------------------
# Main entry — build a city dataset
# ---------------------------------------------------------------------------

def build_city_dataset(processed_dir: Path,
                        n_window: int = 10,
                        tau_minutes: float = 60.0,
                        verbose: bool = False) -> CityDataset:
    p = Path(processed_dir)
    df_train = pd.read_parquet(p / "df_train.parquet")
    df_val = pd.read_parquet(p / "df_val.parquet")
    df_test = pd.read_parquet(p / "df_test.parquet")
    urm_train = sps.load_npz(p / "URM_train.npz").tocsr()
    urm_val = sps.load_npz(p / "URM_val.npz").tocsr()
    urm_test = sps.load_npz(p / "URM_test.npz").tocsr()

    n_users, n_items = urm_train.shape
    macro_to_idx = build_macro_vocab([df_train, df_val, df_test])
    geo_to_idx = build_geohash_vocab([df_train, df_val, df_test])
    n_macros = len(macro_to_idx)
    n_geo = len(geo_to_idx)

    if verbose:
        print(f"  vocab: {n_macros} macros, {n_geo} geohash5 cells "
              f"(+ 1 sentinel)")

    item_cat_macro = build_item_cat_macro([df_train, df_val, df_test],
                                            n_items, macro_to_idx)

    # ---- intent feature precomputation ----------------------------------
    # History rules (per the brief, §5):
    #   train rows → history = train rows strictly before (self)
    #   val   rows → history = train (full, no time cut beyond per-row)
    #   test  rows → history = train ∪ val (refit step)
    raw_train, empty_train = precompute_intent_features(
        df_train, df_train, macro_to_idx, n_window=n_window,
        tau_minutes=tau_minutes,
    )
    stats = IntentStats.from_features(raw_train, n_macros)
    intent_train = stats.standardise(raw_train, n_macros)

    raw_val, empty_val = precompute_intent_features(
        df_val, df_train, macro_to_idx, n_window=n_window,
        tau_minutes=tau_minutes,
    )
    intent_val = stats.standardise(raw_val, n_macros)

    raw_test, empty_test = precompute_intent_features(
        df_test, pd.concat([df_train, df_val], ignore_index=True),
        macro_to_idx, n_window=n_window, tau_minutes=tau_minutes,
    )
    intent_test = stats.standardise(raw_test, n_macros)

    # ---- previous-location geohash (per request) ------------------------
    prev_geo_train = map_prev_geohash(df_train, df_train, geo_to_idx)
    prev_geo_val = map_prev_geohash(df_val, df_train, geo_to_idx)
    prev_geo_test = map_prev_geohash(
        df_test, pd.concat([df_train, df_val], ignore_index=True), geo_to_idx)

    def pack(df: pd.DataFrame, intent: np.ndarray, empty: np.ndarray,
             prev_geo: np.ndarray) -> SplitArrays:
        return SplitArrays(
            u=df["u_idx"].values.astype(np.int32),
            i=df["i_idx"].values.astype(np.int32),
            m=map_row_macros(df, macro_to_idx),
            c_hour=df["c_hour"].values.astype(np.int8),
            c_dow=df["c_dow"].values.astype(np.int8),
            # Foursquare months are 1..12 — turn into 0..11.
            c_month=(df["c_month"].values - df["c_month"].values.min()).astype(np.int8),
            c_isweekend=df["c_isweekend"].values.astype(np.int8),
            prev_geo=prev_geo.astype(np.int32),
            intent_feat=intent.astype(np.float32),
            intent_empty=empty.astype(bool),
        )

    train = pack(df_train, intent_train, empty_train, prev_geo_train)
    val = pack(df_val, intent_val, empty_val, prev_geo_val)
    test = pack(df_test, intent_test, empty_test, prev_geo_test)

    if verbose:
        for name, sa in (("train", train), ("val", val), ("test", test)):
            print(f"  {name}: B={len(sa.u):7d}  empty intent={int(sa.intent_empty.sum())}")

    return CityDataset(
        train=train, val=val, test=test,
        n_users=n_users, n_items=n_items,
        n_macros=n_macros, n_geo=n_geo,
        macro_to_idx=macro_to_idx, geo_to_idx=geo_to_idx,
        item_cat_macro=item_cat_macro,
        urm_train=urm_train, urm_val=urm_val, urm_test=urm_test,
        stats=stats,
    )
