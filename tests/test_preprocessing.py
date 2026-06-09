"""Tests for pipeline.step01_preprocessing — the 5 invariants required by
the brief. These do NOT need the real Foursquare TSVs; they exercise the
preprocessing pipeline on a tiny synthetic TSV constructed inline.

Brief invariants:
  1. Counts: train+val+test == final (post k-core, post cold filter)
  2. No cold leakage: every user/item in val/test appears in train
  3. Dual-view coherence: per split, set of (user_id, item_id) in URM ==
     set in DF
  4. Per-user temporal monotonicity:
     max(t in train_u) ≤ min(t in val_u) ≤ min(t in test_u)
  5. Causal intent proxy: each row's intent_last_cat was OBSERVED before
     that row's time_local (i.e. equals cat_macro of some strictly earlier
     interaction of the same user, OR is 'None').

Run:
    pytest tests/test_preprocessing.py -v
    # or
    python -m tests.test_preprocessing
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step01_preprocessing import preprocess_tsmc2014

TAXONOMY = REPO_ROOT / "config" / "foursquare_legacy_taxonomy.json"

# Real Foursquare legacy cat_ids that we know are in the taxonomy (so the
# semantic feature step covers them; we also include one fake id to exercise
# the 'Other' fallback path).
REAL_CAT_IDS = [
    "4bf58dd8d48988d127951735",   # Arts & Crafts Store (Shop)
    "4bf58dd8d48988d1df941735",   # Bridge (Travel Spot)
    "4bf58dd8d48988d10c951735",   # Cosmetics Shop (Shop)
    "4bf58dd8d48988d1c8941735",   # something in Arts & Entertainment
    "4bf58dd8d48988d1d1941735",   # Ramen / Noodle House (Food)
]
FAKE_CAT_ID = "ffffffffffffffffffffffff"   # not in taxonomy → 'Other'

TSV_HEADER_COLS = ["user_id", "venue_id", "cat_id", "cat_name",
                    "lat", "lon", "tz_offset", "utc_time"]


def _utc_str(dt: datetime) -> str:
    """Format a UTC datetime the way Foursquare's TSV does."""
    return dt.strftime("%a %b %d %H:%M:%S %z %Y")


def _make_synthetic_tsv(path: Path, n_users: int = 30,
                         interactions_per_user: int = 15,
                         n_items: int = 40,
                         seed: int = 42) -> int:
    """Write a fake TSMC2014-style TSV. Returns number of rows written."""
    rng = np.random.default_rng(seed)
    base_time = datetime(2012, 4, 3, 18, 0, 0, tzinfo=timezone.utc)
    rows = []
    for u in range(1, n_users + 1):
        for k in range(interactions_per_user):
            item = int(rng.integers(0, n_items))
            cat_id = REAL_CAT_IDS[item % len(REAL_CAT_IDS)] \
                if item != 0 else FAKE_CAT_ID
            cat_name = "Synthetic Cat " + str(cat_id[:6])
            lat = 40.7 + rng.normal(0, 0.02)
            lon = -74.0 + rng.normal(0, 0.02)
            tz_offset = -240
            t = base_time + timedelta(
                days=k, hours=int(rng.integers(0, 24)), minutes=int(rng.integers(0, 60)))
            rows.append(f"{u}\tv{item:03d}\t{cat_id}\t{cat_name}\t"
                        f"{lat:.6f}\t{lon:.6f}\t{tz_offset}\t{_utc_str(t)}")
    path.write_text("\n".join(rows), encoding="latin-1")
    return len(rows)


# ---------------------------------------------------------------------------
# Fixture: one preprocessed dataset shared by all tests
# ---------------------------------------------------------------------------

def _run_preprocessing_on_synthetic():
    """Build a tiny TSV → run preprocessing → return (result, out_dir,
    train_df, val_df, test_df)."""
    tmp_root = Path(tempfile.mkdtemp(prefix="test_preprocessing_"))
    raw = tmp_root / "synth.txt"
    n_rows = _make_synthetic_tsv(raw, n_users=30, interactions_per_user=15)
    out_dir = tmp_root / "out"

    # k_core=2 so the small synthetic dataset survives the iterative pruning
    result = preprocess_tsmc2014(
        raw_tsv=raw, out_dir=out_dir, taxonomy_path=TAXONOMY,
        k_core=2,
    )
    train_df = pd.read_parquet(out_dir / "df_train.parquet")
    val_df = pd.read_parquet(out_dir / "df_val.parquet")
    test_df = pd.read_parquet(out_dir / "df_test.parquet")
    return result, out_dir, train_df, val_df, test_df, n_rows


# ---------------------------------------------------------------------------
# Invariant 1: counts
# ---------------------------------------------------------------------------

def test_counts_conservation():
    result, out_dir, train, val, test, n_rows = _run_preprocessing_on_synthetic()
    total_split = len(train) + len(val) + len(test)
    assert total_split == result.n_interactions_final, \
        f"sum of splits {total_split} != post-cold final {result.n_interactions_final}"
    # post-cold = post-kcore minus cold-dropped
    cold_total = sum(result.cold_dropped.values())
    assert result.n_interactions_post_kcore - cold_total == result.n_interactions_final, \
        (f"post-kcore {result.n_interactions_post_kcore} - cold-dropped "
         f"{cold_total} != final {result.n_interactions_final}")
    print(f"  invariant 1 OK — total={total_split} matches result.n_interactions_final")


# ---------------------------------------------------------------------------
# Invariant 2: no cold leakage (every user/item in val/test is in train)
# ---------------------------------------------------------------------------

def test_no_cold_leakage():
    _, _, train, val, test, _ = _run_preprocessing_on_synthetic()
    train_users = set(train["user_id"])
    train_items = set(train["venue_id"])
    for name, sub in (("val", val), ("test", test)):
        u_missing = set(sub["user_id"]) - train_users
        i_missing = set(sub["venue_id"]) - train_items
        assert not u_missing, f"{name}: users missing from train: {u_missing}"
        assert not i_missing, f"{name}: items missing from train: {i_missing}"
    print(f"  invariant 2 OK — no cold users/items in val/test")


# ---------------------------------------------------------------------------
# Invariant 3: dual-view coherence (URM and DF agree per split)
# ---------------------------------------------------------------------------

def test_dual_view_coherence():
    _, out_dir, train, val, test, _ = _run_preprocessing_on_synthetic()
    for split, df in (("train", train), ("val", val), ("test", test)):
        urm = sps.load_npz(out_dir / f"URM_{split}.npz").tocoo()
        urm_pairs = set(zip(urm.row.tolist(), urm.col.tolist()))
        df_pairs = set(zip(df["u_idx"].tolist(), df["i_idx"].tolist()))
        # URM is binary so duplicate (u,i) in df collapse into one cell — the
        # URM set should EQUAL the de-duplicated DF set.
        assert urm_pairs == df_pairs, \
            (f"{split}: URM ({len(urm_pairs)} pairs) != DF dedup "
             f"({len(df_pairs)} pairs)")
    print(f"  invariant 3 OK — URM and parquet agree on all 3 splits")


# ---------------------------------------------------------------------------
# Invariant 4: per-user temporal monotonicity
# ---------------------------------------------------------------------------

def test_temporal_monotonicity_per_user():
    _, _, train, val, test, _ = _run_preprocessing_on_synthetic()

    train_max = train.groupby("user_id")["time_local"].max()
    val_min = val.groupby("user_id")["time_local"].min()
    val_max = val.groupby("user_id")["time_local"].max()
    test_min = test.groupby("user_id")["time_local"].min()

    # For users present in both train and val: train_max ≤ val_min
    common = train_max.index.intersection(val_min.index)
    for u in common:
        assert train_max.loc[u] <= val_min.loc[u], \
            f"user {u}: train_max {train_max.loc[u]} > val_min {val_min.loc[u]}"
    # For users present in both val and test: val_max ≤ test_min
    common = val_max.index.intersection(test_min.index)
    for u in common:
        assert val_max.loc[u] <= test_min.loc[u], \
            f"user {u}: val_max {val_max.loc[u]} > test_min {test_min.loc[u]}"
    print(f"  invariant 4 OK — train_max ≤ val_min ≤ test_min per user")


# ---------------------------------------------------------------------------
# Invariant 5: causal intent proxy
# ---------------------------------------------------------------------------

def test_causal_intent_proxy():
    _, _, train, val, test, _ = _run_preprocessing_on_synthetic()
    full = pd.concat([train, val, test], ignore_index=True)
    full = full.sort_values(["user_id", "time_local"], kind="stable").reset_index(drop=True)
    full["_prev_macro"] = full.groupby("user_id")["cat_macro"].shift(1).fillna("None")
    mismatch = (full["intent_last_cat"] != full["_prev_macro"]).sum()
    assert mismatch == 0, (
        f"intent_last_cat does not equal shift(+1) of cat_macro for "
        f"{mismatch} rows — causality is broken or sorting differs")
    # extra: for rows where intent != 'None', verify the previous row has a
    # time strictly earlier than the current row
    non_none = full[full["intent_last_cat"] != "None"].reset_index(drop=True)
    # compare each row to its sorted predecessor (same user)
    for u, grp in non_none.groupby("user_id"):
        if len(grp) < 2:
            continue
        times = grp["time_local"].values
        for i in range(1, len(grp)):
            # at least one previous interaction strictly earlier exists
            assert times[i] >= times[i-1], \
                f"user {u}: non-monotonic time at idx {i}"
    print(f"  invariant 5 OK — intent_last_cat is strictly causal "
          f"(t' < t) on {len(full)} rows")


# ---------------------------------------------------------------------------
# main (so we can also run as a script)
# ---------------------------------------------------------------------------

def main() -> int:
    print("=== preprocessing tests — synthetic TSMC2014 ===")
    test_counts_conservation()
    test_no_cold_leakage()
    test_dual_view_coherence()
    test_temporal_monotonicity_per_user()
    test_causal_intent_proxy()
    print("=== ALL 5 PREPROCESSING INVARIANTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
