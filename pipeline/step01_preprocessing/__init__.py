"""step01 — Foursquare TSMC2014 preprocessing pipeline.

Single-call public API:

    from pipeline.step01_preprocessing import preprocess_tsmc2014
    result = preprocess_tsmc2014(
        raw_tsv="data/raw/dataset_TSMC2014_NYC.txt",
        out_dir="data/processed/NYC",
        taxonomy_path="config/foursquare_legacy_taxonomy.json",
        k_core=10,
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
    )

End-to-end this performs (see brief STEP 1 → 8):

  1.  Parse the 8-column TSV (latin-1).
  2.  Build local time = UTC + tz_offset.
  3.  Iterative user-item k-core (k=10 by default).
  4.  Per-user temporal split 80/10/10 on local time.
  5.  Cold filter: drop val/test rows whose user_id or item_id is absent
      from train.
  6.  Contextual features:
        - temporal: c_hour, c_dow, c_isweekend, c_month     (from time_local)
        - spatial : geohash5, geohash4, dist_prev           (from lat/lon)
        - semantic: cat_fine (name), cat_macro (top-level)  (from taxonomy)
        - intent  : intent_last_cat (last cat_macro of user with t' < t)
  7.  Dual view persisted to disk:
        - URM_{train,val,test}.npz   scipy.sparse.csr_matrix (binary)
        - df_{train,val,test}.parquet
      Same user2id / item2id; partition is identical between the two views.
  8.  mappings.json + stats.json next to the views.

No CARS / proposed / fairness here — those are downstream blocks.
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
import pygeohash
import scipy.sparse as sps

from .taxonomy import build_cat_id_to_macro

logger = logging.getLogger(__name__)
TSV_COLS = ["user_id", "venue_id", "cat_id", "cat_name",
            "lat", "lon", "tz_offset", "utc_time"]
UTC_FORMAT = "%a %b %d %H:%M:%S %z %Y"


# ---------------------------------------------------------------------------
# Step 1+2: load + local time
# ---------------------------------------------------------------------------

def _load_and_localize(raw_tsv: Path) -> pd.DataFrame:
    """Read the TSV and produce ``time_local`` (UTC + tz_offset).

    Encoding is ``latin-1`` because the file contains accented venue names
    (e.g. ``Café``) that are not valid UTF-8.
    """
    df = pd.read_csv(
        raw_tsv, sep="\t", header=None, names=TSV_COLS,
        encoding="latin-1",
        dtype={"user_id": np.int32, "venue_id": str, "cat_id": str,
               "cat_name": str, "lat": np.float64, "lon": np.float64,
               "tz_offset": np.int32, "utc_time": str},
    )
    df["utc_time_parsed"] = pd.to_datetime(df["utc_time"], format=UTC_FORMAT,
                                            utc=True)
    df["time_local"] = (
        df["utc_time_parsed"]
        + pd.to_timedelta(df["tz_offset"].astype(int), unit="m")
    )
    # tz-naive local time (we keep tz info in tz_offset column)
    df["time_local"] = df["time_local"].dt.tz_convert(None)
    # drop exact duplicates (same user, venue, second)
    n_before = len(df)
    df = df.drop_duplicates(subset=["user_id", "venue_id", "utc_time"])
    if n_before != len(df):
        logger.info("  dropped %d exact duplicate rows", n_before - len(df))
    return df


# ---------------------------------------------------------------------------
# Step 3: iterative user-item k-core
# ---------------------------------------------------------------------------

def _iterative_kcore(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Iteratively prune users and items with < k unique interactions until
    the set stabilises. Returns a new DataFrame."""
    cur = df
    iteration = 0
    while True:
        n_before = len(cur)
        user_n = cur.groupby("user_id")["venue_id"].nunique()
        item_n = cur.groupby("venue_id")["user_id"].nunique()
        keep_users = user_n[user_n >= k].index
        keep_items = item_n[item_n >= k].index
        cur = cur[cur["user_id"].isin(keep_users) & cur["venue_id"].isin(keep_items)]
        iteration += 1
        logger.info("  k-core iter %d: %d → %d rows  (users=%d, items=%d)",
                    iteration, n_before, len(cur),
                    cur["user_id"].nunique(), cur["venue_id"].nunique())
        if len(cur) == n_before:
            break
    return cur.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4: per-user temporal split (inspired by datarec LeaveRatioLast)
# ---------------------------------------------------------------------------

def _per_user_temporal_split(df: pd.DataFrame,
                              train_ratio: float,
                              val_ratio: float,
                              test_ratio: float) -> pd.DataFrame:
    """Assign each row to ``train``, ``val`` or ``test`` based on its
    position in the user's chronological history.

    Inspired by ``datarec.splitters.user_stratified.temporal.LeaveRatioLast``
    (Caruccio et al. SIGIR 2025) — we do NOT import datarec (its tabular
    reader strips the contextual columns we need), but we follow its method.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "split ratios must sum to 1"
    df = df.sort_values(["user_id", "time_local"], kind="stable").reset_index(drop=True)
    df["_pos"] = df.groupby("user_id").cumcount()
    df["_total"] = df.groupby("user_id")["user_id"].transform("count")
    df["_frac"] = (df["_pos"] + 1) / df["_total"]
    def assign(frac: float) -> str:
        if frac <= train_ratio:
            return "train"
        if frac <= train_ratio + val_ratio:
            return "val"
        return "test"
    df["split"] = df["_frac"].map(assign)
    df = df.drop(columns=["_pos", "_total", "_frac"])
    return df


# ---------------------------------------------------------------------------
# Step 5: cold filter on val + test
# ---------------------------------------------------------------------------

def _cold_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop val/test interactions whose user_id or item_id never appears in
    train. Returns the filtered df + counts of dropped rows per split."""
    train = df[df["split"] == "train"]
    train_users = set(train["user_id"].unique())
    train_items = set(train["venue_id"].unique())
    is_eval = df["split"].isin(("val", "test"))
    cold_mask = is_eval & (
        ~df["user_id"].isin(train_users) | ~df["venue_id"].isin(train_items)
    )
    dropped = df[cold_mask]
    counts = {
        "val_dropped": int(((dropped["split"] == "val")).sum()),
        "test_dropped": int(((dropped["split"] == "test")).sum()),
    }
    out = df[~cold_mask].reset_index(drop=True)
    return out, counts


# ---------------------------------------------------------------------------
# Step 6: contextual features
# ---------------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine in km (lat/lon in degrees)."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df["time_local"]
    df = df.copy()
    df["c_hour"] = t.dt.hour.astype(np.int8)
    df["c_dow"] = t.dt.dayofweek.astype(np.int8)
    df["c_isweekend"] = (df["c_dow"] >= 5).astype(np.int8)
    df["c_month"] = t.dt.month.astype(np.int8)
    return df


def _add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute geohash5/4 and dist_prev (km from previous check-in of the
    same user)."""
    df = df.copy()
    df["geohash5"] = [pygeohash.encode(la, lo, precision=5)
                      for la, lo in zip(df["lat"].values, df["lon"].values)]
    df["geohash4"] = [pygeohash.encode(la, lo, precision=4)
                      for la, lo in zip(df["lat"].values, df["lon"].values)]
    df = df.sort_values(["user_id", "time_local"], kind="stable").reset_index(drop=True)
    lat_prev = df.groupby("user_id")["lat"].shift(1)
    lon_prev = df.groupby("user_id")["lon"].shift(1)
    df["dist_prev"] = _haversine_km(lat_prev.values, lon_prev.values,
                                    df["lat"].values, df["lon"].values)
    df["dist_prev"] = df["dist_prev"].fillna(0.0).astype(np.float32)
    return df


def _add_semantic_features(df: pd.DataFrame, cat2macro: dict[str, str]
                            ) -> tuple[pd.DataFrame, int]:
    """Add cat_fine (= cat_name) and cat_macro (taxonomy roll-up, with
    'Other' fallback for cat_ids not in the legacy taxonomy)."""
    df = df.copy()
    df["cat_fine"] = df["cat_name"]
    df["cat_macro"] = df["cat_id"].map(cat2macro).fillna("Other")
    n_other = int((df["cat_macro"] == "Other").sum())
    return df, n_other


def _add_intent_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """intent_last_cat = cat_macro of the immediately previous interaction
    of the same user. Causal by construction (shift +1 after temporal sort)
    and naturally also causal across the temporal split."""
    df = df.sort_values(["user_id", "time_local"], kind="stable").reset_index(drop=True)
    df["intent_last_cat"] = df.groupby("user_id")["cat_macro"].shift(1)
    df["intent_last_cat"] = df["intent_last_cat"].fillna("None")
    return df


# ---------------------------------------------------------------------------
# Step 7: dual view persistence
# ---------------------------------------------------------------------------

def _build_dual_view(df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Build user2id / item2id, write URM_{train,val,test}.npz and
    df_{train,val,test}.parquet from the same split."""
    user2id = {u: i for i, u in enumerate(sorted(df["user_id"].unique()))}
    item2id = {v: i for i, v in enumerate(sorted(df["venue_id"].unique()))}
    df = df.copy()
    df["u_idx"] = df["user_id"].map(user2id).astype(np.int32)
    df["i_idx"] = df["venue_id"].map(item2id).astype(np.int32)

    n_users = len(user2id)
    n_items = len(item2id)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_summary: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        n = len(sub)
        if n == 0:
            logger.warning("  split %s empty, skipping URM/parquet write", split)
            continue
        urm = sps.csr_matrix(
            (np.ones(n, dtype=np.float32),
             (sub["u_idx"].values, sub["i_idx"].values)),
            shape=(n_users, n_items),
        )
        urm.sum_duplicates()
        urm.data[:] = 1.0
        sps.save_npz(out_dir / f"URM_{split}.npz", urm)

        keep_cols = [
            "u_idx", "i_idx", "user_id", "venue_id", "time_local",
            "cat_fine", "cat_macro",
            "c_hour", "c_dow", "c_isweekend", "c_month",
            "geohash5", "geohash4", "dist_prev",
            "intent_last_cat", "split",
        ]
        sub_out = sub[keep_cols].reset_index(drop=True)
        sub_out.to_parquet(out_dir / f"df_{split}.parquet", index=False)
        splits_summary[split] = {
            "n_interactions": int(n),
            "n_users": int(sub["u_idx"].nunique()),
            "n_items": int(sub["i_idx"].nunique()),
            "urm_density": float(urm.nnz / (n_users * n_items)),
        }
    return {
        "n_users_total": int(n_users),
        "n_items_total": int(n_items),
        "user2id": user2id,
        "item2id": item2id,
        "splits": splits_summary,
    }


# ---------------------------------------------------------------------------
# Step 8: result + mappings + stats
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingResult:
    out_dir: Path
    n_interactions_raw: int
    n_interactions_post_kcore: int
    n_interactions_final: int
    n_users_final: int
    n_items_final: int
    splits: dict[str, Any]
    n_other_macro: int
    n_home_private: int
    cold_dropped: dict[str, int]
    wallclock_s: float

    def short_report(self) -> str:
        lines = [
            f"=== PREPROCESSING REPORT ({self.out_dir.name}) ===",
            f"  raw interactions:           {self.n_interactions_raw:,}",
            f"  after k-core:               {self.n_interactions_post_kcore:,}",
            f"  after cold filter (final):  {self.n_interactions_final:,}",
            f"  final users / items:        {self.n_users_final:,} / {self.n_items_final:,}",
            f"  Home (private) rows:        {self.n_home_private:,}  "
            f"({100*self.n_home_private/max(self.n_interactions_final,1):.2f} %)",
            f"  Other-macro residue:        {self.n_other_macro:,}  "
            f"({100*self.n_other_macro/max(self.n_interactions_final,1):.2f} %)",
            f"  cold-dropped (val/test):    {self.cold_dropped}",
            f"  wallclock:                  {self.wallclock_s:.1f} s",
            "",
            "  per-split summary:",
        ]
        for split, s in self.splits.items():
            lines.append(
                f"    {split:5s}  n={s['n_interactions']:>8,}  "
                f"users={s['n_users']:>5,}  items={s['n_items']:>5,}  "
                f"density={s['urm_density']:.5%}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# top-level entry point
# ---------------------------------------------------------------------------

def preprocess_tsmc2014(raw_tsv: Path | str,
                        out_dir: Path | str,
                        taxonomy_path: Path | str,
                        k_core: int = 10,
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1,
                        test_ratio: float = 0.1,
                        seed: int = 42) -> PreprocessingResult:
    """End-to-end TSMC2014 preprocessing → dual view on disk."""
    raw_tsv = Path(raw_tsv)
    out_dir = Path(out_dir)
    taxonomy_path = Path(taxonomy_path)

    t0 = time.time()
    logger.info("[step01] loading %s ...", raw_tsv)
    df = _load_and_localize(raw_tsv)
    n_raw = len(df)

    logger.info("[step01] iterative k-core (k=%d) ...", k_core)
    df = _iterative_kcore(df, k_core)
    n_post_kcore = len(df)

    logger.info("[step01] per-user temporal split %.0f/%.0f/%.0f ...",
                train_ratio * 100, val_ratio * 100, test_ratio * 100)
    df = _per_user_temporal_split(df, train_ratio, val_ratio, test_ratio)

    logger.info("[step01] cold filter on val+test ...")
    df, cold_counts = _cold_filter(df)

    logger.info("[step01] features: temporal + spatial + semantic + intent ...")
    df = _add_temporal_features(df)
    df = _add_spatial_features(df)
    cat2macro = build_cat_id_to_macro(taxonomy_path)
    df, n_other = _add_semantic_features(df, cat2macro)
    df = _add_intent_proxy(df)

    n_home = int((df["cat_name"] == "Home (private)").sum())
    n_final = len(df)

    logger.info("[step01] building dual view (URM + parquet) ...")
    view_summary = _build_dual_view(df, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    mappings = {
        "user2id": {str(k): v for k, v in view_summary["user2id"].items()},
        "item2id": view_summary["item2id"],
        "params": {
            "raw_tsv": str(raw_tsv),
            "k_core": k_core,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "encoding": "latin-1",
        },
        "n_users_total": view_summary["n_users_total"],
        "n_items_total": view_summary["n_items_total"],
    }
    (out_dir / "mappings.json").write_text(json.dumps(mappings, indent=2))

    stats = {
        "raw_interactions": n_raw,
        "post_kcore_interactions": n_post_kcore,
        "post_cold_interactions": n_final,
        "n_users_final": view_summary["n_users_total"],
        "n_items_final": view_summary["n_items_total"],
        "splits": view_summary["splits"],
        "cold_dropped": cold_counts,
        "n_other_macro": n_other,
        "n_home_private": n_home,
        "taxonomy_cat_id_coverage": {
            "in_taxonomy": int((df["cat_macro"] != "Other").sum()),
            "missing_other": n_other,
        },
        "top10_macro": df["cat_macro"].value_counts().head(10).to_dict(),
        "top10_fine": df["cat_fine"].value_counts().head(10).to_dict(),
        "wallclock_s": time.time() - t0,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str))

    return PreprocessingResult(
        out_dir=out_dir,
        n_interactions_raw=n_raw,
        n_interactions_post_kcore=n_post_kcore,
        n_interactions_final=n_final,
        n_users_final=view_summary["n_users_total"],
        n_items_final=view_summary["n_items_total"],
        splits=view_summary["splits"],
        n_other_macro=n_other,
        n_home_private=n_home,
        cold_dropped=cold_counts,
        wallclock_s=time.time() - t0,
    )
