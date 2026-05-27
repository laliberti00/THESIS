"""Read-only inspection of all Gowalla raw sources available in data/GowallaRaw/.

This script does **NOT** preprocess, split, filter, or otherwise transform the
data. It only loads each source, measures schema/scale/quality/distribution,
and writes a JSON dump to stdout (plus a side file) so the report
GOWALLA_DATASETS_COMPARISON.md can cite exact numbers.

Run from the repo root:
    cd ~/Downloads/IntentAwareRS_thesis
    source .venv/bin/activate
    python scripts/inspect_gowalla_raw.py            # ~2-5 min on Apple M2

All distributions use percentiles [5, 25, 50, 75, 95]. The SNAP↔Liu sample-match
uses ``--sample-size 1000`` SNAP rows (seed 42) compared against the join of
Liu's checkins and spots within a temporal/spatial tolerance.

NO output file under data/. Only console + scripts/_inspect_output/inspection.json.
"""

from __future__ import annotations

import argparse
import ast
import json
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "GowallaRaw"
LIU_DIR = RAW_DIR / "22126586"
DCCF_DIR = REPO_ROOT / "data" / "DCCF" / "gowalla"
OUT_DIR = Path(__file__).resolve().parent / "_inspect_output"
SEED = 42


# ---------- helpers --------------------------------------------------------

def banner(msg: str) -> None:
    print()
    print("=" * 78)
    print(f"  {msg}")
    print("=" * 78)


def jsonable(obj):
    """Coerce numpy/pandas types into JSON-serialisable Python primitives."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Period):
        return str(obj)
    return obj


def describe_dist(series: pd.Series, label: str) -> dict:
    """Return mean + percentiles dict."""
    pcts = series.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
    return {
        "label": label,
        "n": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
        "min": float(series.min()),
        "max": float(series.max()),
        "p05": float(pcts[0.05]),
        "p25": float(pcts[0.25]),
        "p50": float(pcts[0.5]),
        "p75": float(pcts[0.75]),
        "p95": float(pcts[0.95]),
    }


# ---------- inspectors -----------------------------------------------------

def inspect_snap() -> dict:
    """SNAP loc-gowalla_totalCheckins.txt — Cho-Myers-Leskovec 2011.

    Schema (TSV, no header): user_id  time  lat  lon  location_id
    """
    banner("SNAP — loc-gowalla_totalCheckins.txt")
    path = RAW_DIR / "loc-gowalla_totalCheckins.txt"
    info: dict[str, Any] = {"source": "snap_totalCheckins", "path": str(path)}

    t0 = time.time()
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["user_id", "time", "lat", "lon", "location_id"],
        dtype={
            "user_id": np.int32,
            "lat": np.float64,
            "lon": np.float64,
            "location_id": np.int64,  # SNAP location_id ranges high
        },
        parse_dates=["time"],
    )
    print(f"[load]   {len(df):,} rows in {time.time()-t0:.1f}s")

    info["schema"] = {
        "format": "TSV without header",
        "columns": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    info["head_10"] = df.head(10).astype(str).to_dict(orient="records")

    # Scale
    info["scale"] = {
        "rows": int(len(df)),
        "users_distinct": int(df.user_id.nunique()),
        "venues_distinct": int(df.location_id.nunique()),
        "time_min": str(df.time.min()),
        "time_max": str(df.time.max()),
    }
    print(f"[scale]  users={info['scale']['users_distinct']:,}  "
          f"venues={info['scale']['venues_distinct']:,}  "
          f"time=[{info['scale']['time_min']} .. {info['scale']['time_max']}]")

    # Quality
    nan_mask = df.isna().any(axis=1)
    zero_zero = (df.lat == 0) & (df.lon == 0)
    lat_oob = (df.lat < -90) | (df.lat > 90)
    lon_oob = (df.lon < -180) | (df.lon > 180)
    info["quality"] = {
        "rows_with_nan": int(nan_mask.sum()),
        "rows_zero_zero_coords": int(zero_zero.sum()),
        "rows_lat_out_of_range": int(lat_oob.sum()),
        "rows_lon_out_of_range": int(lon_oob.sum()),
        "exact_duplicates": int(df.duplicated().sum()),
    }
    print(f"[qual]   NaN={info['quality']['rows_with_nan']}  "
          f"(0,0)={info['quality']['rows_zero_zero_coords']}  "
          f"dup={info['quality']['exact_duplicates']}")

    # Distributions
    user_counts = df.groupby("user_id").size()
    item_counts = df.groupby("location_id").size()
    monthly = df.groupby(df.time.dt.to_period("M")).size()
    info["distributions"] = {
        "checkins_per_user": describe_dist(user_counts, "checkins_per_user"),
        "checkins_per_venue": describe_dist(item_counts, "checkins_per_venue"),
        "monthly": {str(k): int(v) for k, v in monthly.items()},
    }
    print(f"[dist]   ck/user (p50)={info['distributions']['checkins_per_user']['p50']:.0f}, "
          f"ck/venue (p50)={info['distributions']['checkins_per_venue']['p50']:.0f}, "
          f"months={len(monthly)}")

    # Keep the dataframe for the comparison step (sample-match)
    info["_df_handle"] = df
    return info


def inspect_liu_checkins() -> dict:
    """Liu gowalla_checkins.csv — userid, placeid, datetime.

    No spatial coords here; they must be joined from the spots files.
    """
    banner("LIU — gowalla_checkins.csv")
    path = LIU_DIR / "gowalla_checkins.csv"
    info: dict[str, Any] = {"source": "liu_checkins", "path": str(path)}

    t0 = time.time()
    df = pd.read_csv(
        path,
        dtype={"userid": np.int32, "placeid": np.int64},
        parse_dates=["datetime"],
    )
    print(f"[load]   {len(df):,} rows in {time.time()-t0:.1f}s")

    info["schema"] = {
        "format": "CSV with header",
        "columns": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    info["head_10"] = df.head(10).astype(str).to_dict(orient="records")

    info["scale"] = {
        "rows": int(len(df)),
        "users_distinct": int(df.userid.nunique()),
        "placeids_distinct": int(df.placeid.nunique()),
        "time_min": str(df.datetime.min()),
        "time_max": str(df.datetime.max()),
    }
    print(f"[scale]  users={info['scale']['users_distinct']:,}  "
          f"placeids={info['scale']['placeids_distinct']:,}  "
          f"time=[{info['scale']['time_min']} .. {info['scale']['time_max']}]")

    nan_mask = df.isna().any(axis=1)
    info["quality"] = {
        "rows_with_nan": int(nan_mask.sum()),
        "exact_duplicates": int(df.duplicated().sum()),
    }
    print(f"[qual]   NaN={info['quality']['rows_with_nan']}  "
          f"dup={info['quality']['exact_duplicates']}")

    user_counts = df.groupby("userid").size()
    item_counts = df.groupby("placeid").size()
    monthly = df.groupby(df.datetime.dt.to_period("M")).size()
    info["distributions"] = {
        "checkins_per_user": describe_dist(user_counts, "checkins_per_user"),
        "checkins_per_venue": describe_dist(item_counts, "checkins_per_venue"),
        "monthly": {str(k): int(v) for k, v in monthly.items()},
    }
    print(f"[dist]   ck/user (p50)={info['distributions']['checkins_per_user']['p50']:.0f}, "
          f"ck/place (p50)={info['distributions']['checkins_per_venue']['p50']:.0f}, "
          f"months={len(monthly)}")

    info["_df_handle"] = df
    return info


def inspect_liu_spots() -> tuple[dict, dict]:
    """Both gowalla_spots_subset1.csv and gowalla_spots_subset2.csv.

    subset1 has rich attrs incl. spot_categories (Python-literal string).
    subset2 is much smaller and only has name/city_state.
    """
    banner("LIU — gowalla_spots_subset1.csv")
    p1 = LIU_DIR / "gowalla_spots_subset1.csv"
    t0 = time.time()
    s1 = pd.read_csv(
        p1,
        dtype={"id": np.int64, "lat": np.float64, "lng": np.float64,
               "photos_count": np.int32, "checkins_count": np.int32,
               "users_count": np.int32, "radius_meters": np.float32,
               "highlights_count": np.int32, "items_count": np.int32,
               "max_items_count": np.int32, "spot_categories": str},
        parse_dates=["created_at"],
    )
    print(f"[load]   subset1: {len(s1):,} rows in {time.time()-t0:.1f}s")

    has_category = s1["spot_categories"].fillna("").str.strip().ne("").ne("[]")
    info1 = {
        "source": "liu_spots_subset1",
        "path": str(p1),
        "schema": {
            "format": "CSV with header",
            "columns": s1.columns.tolist(),
            "dtypes": {c: str(t) for c, t in s1.dtypes.items()},
        },
        "head_10": s1.head(10).astype(str).to_dict(orient="records"),
        "scale": {
            "rows": int(len(s1)),
            "venues_distinct": int(s1["id"].nunique()),
            "time_min": str(s1.created_at.min()),
            "time_max": str(s1.created_at.max()),
        },
        "quality": {
            "rows_with_nan": int(s1.isna().any(axis=1).sum()),
            "rows_zero_zero_coords": int(((s1.lat == 0) & (s1.lng == 0)).sum()),
            "rows_lat_out_of_range": int(((s1.lat < -90) | (s1.lat > 90)).sum()),
            "rows_lon_out_of_range": int(((s1.lng < -180) | (s1.lng > 180)).sum()),
            "exact_duplicates": int(s1.duplicated().sum()),
            "venues_with_category": int(has_category.sum()),
            "venues_without_category": int((~has_category).sum()),
        },
    }
    print(f"[scale]  venues={info1['scale']['venues_distinct']:,}")
    print(f"[qual]   with_cat={info1['quality']['venues_with_category']:,}  "
          f"without_cat={info1['quality']['venues_without_category']:,}")

    # Distribution of categories
    cat_counter: Counter = Counter()
    sample_parsed = []
    for s in s1.loc[has_category, "spot_categories"].dropna():
        try:
            parsed = ast.literal_eval(s)
            for c in parsed:
                cat_counter[c.get("name", "<noname>")] += 1
            if len(sample_parsed) < 5:
                sample_parsed.append(parsed)
        except Exception:
            pass
    info1["distributions"] = {
        "n_distinct_category_names": len(cat_counter),
        "top_20_categories": cat_counter.most_common(20),
        "sample_parsed_5": sample_parsed,
    }
    print(f"[dist]   distinct cat names: {len(cat_counter)}")

    info1["_df_handle"] = s1

    banner("LIU — gowalla_spots_subset2.csv")
    p2 = LIU_DIR / "gowalla_spots_subset2.csv"
    t0 = time.time()
    # subset2 has trailing comma artifacts; some rows have empty unnamed columns.
    # The file contains non-UTF8 bytes (Latin-1 chars in place names like "café"),
    # so we read with latin-1 which is byte-safe for any byte sequence.
    s2 = pd.read_csv(p2, dtype=str, encoding="latin-1", on_bad_lines="warn")
    print(f"[load]   subset2: {len(s2):,} rows in {time.time()-t0:.1f}s")
    # Try numeric conversion for id, lat, lng
    s2["id"] = pd.to_numeric(s2["id"], errors="coerce").astype("Int64")
    s2["lat"] = pd.to_numeric(s2["lat"], errors="coerce")
    s2["lng"] = pd.to_numeric(s2["lng"], errors="coerce")

    info2 = {
        "source": "liu_spots_subset2",
        "path": str(p2),
        "schema": {
            "format": "CSV with header",
            "columns": s2.columns.tolist(),
            "dtypes": {c: str(t) for c, t in s2.dtypes.items()},
        },
        "head_10": s2.head(10).astype(str).to_dict(orient="records"),
        "scale": {
            "rows": int(len(s2)),
            "venues_distinct": int(s2["id"].nunique()),
        },
        "quality": {
            "rows_with_nan": int(s2.isna().any(axis=1).sum()),
            "rows_zero_zero_coords": int(((s2.lat.fillna(0) == 0) & (s2.lng.fillna(0) == 0)).sum()),
            "rows_lat_out_of_range": int(((s2.lat < -90) | (s2.lat > 90)).fillna(False).sum()),
            "rows_lon_out_of_range": int(((s2.lng < -180) | (s2.lng > 180)).fillna(False).sum()),
            "exact_duplicates": int(s2.duplicated().sum()),
        },
    }
    print(f"[scale]  venues={info2['scale']['venues_distinct']:,}")

    info2["_df_handle"] = s2

    # ---- subset1 vs subset2 set relationship ----
    banner("LIU — subset1 vs subset2 overlap analysis")
    s1_ids = set(s1["id"].dropna().astype("int64").tolist())
    s2_ids = set(s2["id"].dropna().astype("int64").tolist())
    overlap = s1_ids & s2_ids
    only_in_1 = s1_ids - s2_ids
    only_in_2 = s2_ids - s1_ids
    union = s1_ids | s2_ids
    rel = {
        "subset1_n": len(s1_ids),
        "subset2_n": len(s2_ids),
        "intersection_n": len(overlap),
        "only_in_subset1_n": len(only_in_1),
        "only_in_subset2_n": len(only_in_2),
        "union_n": len(union),
        "pct_subset2_in_subset1": (len(overlap) / max(len(s2_ids), 1)) * 100,
        "pct_subset1_in_subset2": (len(overlap) / max(len(s1_ids), 1)) * 100,
    }
    print(f"  subset1: {rel['subset1_n']:,} venues")
    print(f"  subset2: {rel['subset2_n']:,} venues")
    print(f"  intersection: {rel['intersection_n']:,}")
    print(f"  only-in-subset1: {rel['only_in_subset1_n']:,}")
    print(f"  only-in-subset2: {rel['only_in_subset2_n']:,}")
    print(f"  union: {rel['union_n']:,}")
    print(f"  % of subset2 also in subset1: {rel['pct_subset2_in_subset1']:.1f}%")
    info1["subset1_vs_subset2_relationship"] = rel

    return info1, info2


def inspect_liu_categories() -> dict:
    """Parse gowalla_category_structure.json. Recursive: top-level
    'spot_categories' may contain 'spot_categories' children, etc."""
    banner("LIU — gowalla_category_structure.json")
    path = LIU_DIR / "gowalla_category_structure.json"
    info: dict[str, Any] = {"source": "liu_category_structure", "path": str(path)}

    raw = json.loads(path.read_text(encoding="utf-8"))
    top = raw.get("spot_categories", [])

    # Recursive walk
    levels: dict[int, list[str]] = {}
    by_url: dict[str, dict] = {}

    def walk(nodes, depth=0):
        levels.setdefault(depth, [])
        for n in nodes:
            name = n.get("name", "<noname>")
            url = n.get("url", "")
            levels[depth].append(name)
            by_url[url] = {"name": name, "depth": depth, "children": []}
            children = n.get("spot_categories", [])
            for c in children:
                by_url[url]["children"].append(c.get("url", ""))
            walk(children, depth + 1)

    walk(top, 0)

    info["scale"] = {
        "n_categories_total": sum(len(v) for v in levels.values()),
        "n_levels": len(levels),
        "per_level_count": {str(d): len(v) for d, v in sorted(levels.items())},
        "top_level_names": levels.get(0, []),
    }
    print(f"[scale]  levels={info['scale']['n_levels']}  "
          f"total cats={info['scale']['n_categories_total']}")
    for d, names in sorted(levels.items()):
        print(f"  level {d}: {len(names)} categories")

    return info


def inspect_liu_friendship() -> dict:
    banner("LIU — gowalla_friendship.csv")
    path = LIU_DIR / "gowalla_friendship.csv"
    df = pd.read_csv(path, dtype={"userid1": np.int32, "userid2": np.int32})
    print(f"[load]   {len(df):,} rows")
    info = {
        "source": "liu_friendship",
        "path": str(path),
        "schema": {"columns": df.columns.tolist()},
        "scale": {
            "rows_edges": int(len(df)),
            "users_distinct": int(pd.concat([df.userid1, df.userid2]).nunique()),
        },
        "quality": {
            "exact_duplicates": int(df.duplicated().sum()),
            "rows_with_nan": int(df.isna().any(axis=1).sum()),
        },
    }
    print(f"[scale]  edges={info['scale']['rows_edges']:,}  "
          f"users involved={info['scale']['users_distinct']:,}")
    return info


def inspect_liu_userinfo() -> dict:
    banner("LIU — gowalla_userinfo.csv")
    path = LIU_DIR / "gowalla_userinfo.csv"
    df = pd.read_csv(path)
    print(f"[load]   {len(df):,} rows × {df.shape[1]} cols")
    info = {
        "source": "liu_userinfo",
        "path": str(path),
        "schema": {
            "columns": df.columns.tolist(),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        },
        "head_10": df.head(10).astype(str).to_dict(orient="records"),
        "scale": {
            "rows": int(len(df)),
            "users_distinct": int(df["id"].nunique()),
        },
        "quality": {
            "rows_with_nan": int(df.isna().any(axis=1).sum()),
            "exact_duplicates": int(df.duplicated().sum()),
        },
    }
    # checkin_num distribution if present
    if "checkin_num" in df.columns:
        info["distributions"] = {
            "checkin_num": describe_dist(df["checkin_num"].dropna(),
                                          "checkin_num"),
        }
    return info


def inspect_dccf_pkl() -> dict:
    """Reference baseline: Shehzad's pre-split URM .pkl, what we have been
    using in Phase 1-2."""
    banner("REFERENCE — data/DCCF/gowalla/{train,test}.pkl (Shehzad)")
    import scipy.sparse as sps
    info = {"source": "shehzad_pkl", "path": str(DCCF_DIR)}

    with open(DCCF_DIR / "train.pkl", "rb") as f:
        train = pickle.load(f)
    with open(DCCF_DIR / "test.pkl", "rb") as f:
        test = pickle.load(f)

    info["scale"] = {
        "shape": list(train.shape),
        "users": int(train.shape[0]),
        "items": int(train.shape[1]),
        "train_interactions": int(train.nnz),
        "test_interactions": int(test.nnz),
        "total_interactions": int(train.nnz + test.nnz),
        "density_train": float(train.nnz / (train.shape[0] * train.shape[1])),
    }
    print(f"[scale]  {info['scale']['users']:,} users × "
          f"{info['scale']['items']:,} items, "
          f"train={info['scale']['train_interactions']:,}  "
          f"test={info['scale']['test_interactions']:,}")

    csr = train.tocsr()
    per_user = np.asarray((csr != 0).sum(axis=1)).ravel()
    per_item = np.asarray((csr != 0).sum(axis=0)).ravel()
    info["distributions"] = {
        "train_per_user": {
            "min": int(per_user.min()),
            "max": int(per_user.max()),
            "mean": float(per_user.mean()),
            "p05": float(np.quantile(per_user, 0.05)),
            "p25": float(np.quantile(per_user, 0.25)),
            "p50": float(np.quantile(per_user, 0.5)),
            "p75": float(np.quantile(per_user, 0.75)),
            "p95": float(np.quantile(per_user, 0.95)),
        },
        "train_per_item": {
            "min": int(per_item.min()),
            "max": int(per_item.max()),
            "mean": float(per_item.mean()),
            "p05": float(np.quantile(per_item, 0.05)),
            "p25": float(np.quantile(per_item, 0.25)),
            "p50": float(np.quantile(per_item, 0.5)),
            "p75": float(np.quantile(per_item, 0.75)),
            "p95": float(np.quantile(per_item, 0.95)),
        },
    }
    return info


# ---------- SNAP ↔ Liu comparison -----------------------------------------

def compare_snap_vs_liu(snap_info: dict,
                        liu_ckin_info: dict,
                        liu_s1_info: dict,
                        liu_s2_info: dict,
                        sample_size: int = 1000) -> dict:
    """Sample-match: pick `sample_size` rows from SNAP and look for them in Liu
    using (datetime ± tol_seconds, lat ± tol_deg, lon ± tol_deg).

    Liu checkins have no spatial coords, so we first join with spots1+spots2 to
    obtain (userid, placeid, datetime, lat, lng) per Liu row.
    """
    banner("COMPARISON — SNAP ↔ Liu (sample-match)")

    rng = np.random.default_rng(SEED)
    snap = snap_info["_df_handle"]
    liu = liu_ckin_info["_df_handle"]
    s1 = liu_s1_info["_df_handle"]
    s2 = liu_s2_info["_df_handle"]

    # ---------- 1) range overlap (cheap, computed by metadata) ----------
    rng_overlap = {
        "snap_time_range": [snap_info["scale"]["time_min"],
                             snap_info["scale"]["time_max"]],
        "liu_time_range": [liu_ckin_info["scale"]["time_min"],
                            liu_ckin_info["scale"]["time_max"]],
    }
    snap_min = pd.Timestamp(snap_info["scale"]["time_min"])
    snap_max = pd.Timestamp(snap_info["scale"]["time_max"])
    liu_min = pd.Timestamp(liu_ckin_info["scale"]["time_min"])
    liu_max = pd.Timestamp(liu_ckin_info["scale"]["time_max"])
    overlap_start = max(snap_min, liu_min)
    overlap_end = min(snap_max, liu_max)
    rng_overlap["overlap_days"] = max(0, (overlap_end - overlap_start).days)
    rng_overlap["overlap_window"] = (
        [str(overlap_start), str(overlap_end)]
        if rng_overlap["overlap_days"] > 0 else None
    )
    print(f"[range]  SNAP=[{snap_min} .. {snap_max}]")
    print(f"[range]  LIU =[{liu_min} .. {liu_max}]")
    print(f"[range]  overlap days: {rng_overlap['overlap_days']}")

    # ---------- 2) Liu join (checkin × spots) ----------
    print("[join]   building placeid → (lat, lng) map from spots1+spots2 ...")
    # Vectorised dict construction — far faster than iterrows on millions of rows
    s1_clean = s1[["id", "lat", "lng"]].dropna()
    coords = dict(zip(s1_clean["id"].astype("int64").tolist(),
                      zip(s1_clean["lat"].astype(float).tolist(),
                          s1_clean["lng"].astype(float).tolist())))
    s2_clean = s2[["id", "lat", "lng"]].dropna()
    s2_ids = s2_clean["id"].astype("int64").tolist()
    s2_lat = s2_clean["lat"].astype(float).tolist()
    s2_lng = s2_clean["lng"].astype(float).tolist()
    for pid, la, lo in zip(s2_ids, s2_lat, s2_lng):
        if pid not in coords:                       # subset1 wins on conflict
            coords[pid] = (la, lo)
    print(f"[join]   {len(coords):,} distinct placeids with coordinates")

    n_ck = len(liu)
    has_coords_mask = liu["placeid"].isin(coords)
    n_with_coords = int(has_coords_mask.sum())
    print(f"[join]   Liu checkins with placeid resolvable to (lat,lng): "
          f"{n_with_coords:,} / {n_ck:,} ({100*n_with_coords/n_ck:.1f}%)")

    # ---------- 3) Sample SNAP and try matching ----------
    if rng_overlap["overlap_days"] == 0:
        print("[match]  NO temporal overlap — sample-match skipped.")
        return {
            "range_overlap": rng_overlap,
            "join_coverage": {
                "n_liu_checkins": n_ck,
                "n_with_coordinates": n_with_coords,
                "pct_with_coordinates": 100*n_with_coords/n_ck if n_ck else 0,
                "n_distinct_placeids_with_coords": len(coords),
            },
            "sample_match": {
                "sample_size": sample_size,
                "skipped_reason": "no temporal overlap",
            },
        }

    # restrict SNAP to overlap window, then sample
    snap_in_overlap = snap[(snap.time >= overlap_start) & (snap.time <= overlap_end)]
    print(f"[match]  SNAP rows in temporal overlap: {len(snap_in_overlap):,}")
    if len(snap_in_overlap) == 0:
        return {
            "range_overlap": rng_overlap,
            "join_coverage": {"n_liu_checkins": n_ck, "n_with_coordinates": n_with_coords},
            "sample_match": {"sample_size": sample_size,
                              "skipped_reason": "no SNAP rows in overlap window"},
        }

    sample = snap_in_overlap.sample(n=min(sample_size, len(snap_in_overlap)),
                                    random_state=SEED)

    # Build a Liu lookup: filter to overlap window first, then build a dict
    # keyed by minute-rounded datetime → list of (lat,lng) tuples.
    print("[match]  preparing Liu temporal index in overlap window ...")
    liu_in_overlap = liu[(liu.datetime >= overlap_start) & (liu.datetime <= overlap_end)].copy()
    liu_in_overlap = liu_in_overlap[liu_in_overlap["placeid"].isin(coords)]
    print(f"[match]  Liu rows in overlap window with resolvable coords: "
          f"{len(liu_in_overlap):,}")
    # Round datetime to the closest minute. Build the index via numpy/zip,
    # never with iterrows — that would take many minutes on a multi-million row df.
    minute_bucket = liu_in_overlap["datetime"].dt.floor("min").values
    placeid_arr = liu_in_overlap["placeid"].astype("int64").values
    print(f"[match]  building per-minute index ({len(minute_bucket):,} rows) ...")
    liu_by_minute: dict[pd.Timestamp, list[tuple[float, float]]] = {}
    for mb, pid in zip(minute_bucket, placeid_arr):
        lat_lng = coords.get(int(pid))
        if lat_lng is not None:
            liu_by_minute.setdefault(mb, []).append(lat_lng)
    print(f"[match]  {len(liu_by_minute):,} distinct minute-buckets indexed")

    tol_seconds = 600        # ±10 minutes
    tol_lat = 0.001          # ~110 m
    tol_lng = 0.001          # ~85 m at 30°N
    n_match = 0
    examples_match = []
    examples_nomatch = []
    sample_users = sample["user_id"].astype("int64").values
    sample_times = sample["time"].values  # numpy datetime64[ns]
    sample_lats = sample["lat"].astype(float).values
    sample_lngs = sample["lon"].astype(float).values
    for i in range(len(sample)):
        snap_t = pd.Timestamp(sample_times[i])
        snap_lat = sample_lats[i]
        snap_lng = sample_lngs[i]
        # check ±10 min in 1-minute buckets
        base = snap_t.floor("min")
        found = False
        for offset in range(-(tol_seconds // 60), tol_seconds // 60 + 1):
            bucket = base + pd.Timedelta(minutes=offset)
            cands = liu_by_minute.get(np.datetime64(bucket))
            if not cands:
                continue
            for (la, lo) in cands:
                if abs(la - snap_lat) <= tol_lat and abs(lo - snap_lng) <= tol_lng:
                    found = True
                    break
            if found:
                break
        if found:
            n_match += 1
            if len(examples_match) < 3:
                examples_match.append({
                    "snap_user": int(sample_users[i]),
                    "snap_time": str(snap_t),
                    "snap_lat": float(snap_lat),
                    "snap_lng": float(snap_lng),
                })
        else:
            if len(examples_nomatch) < 3:
                examples_nomatch.append({
                    "snap_user": int(sample_users[i]),
                    "snap_time": str(snap_t),
                    "snap_lat": float(snap_lat),
                    "snap_lng": float(snap_lng),
                })

    pct = 100 * n_match / len(sample) if len(sample) else 0
    print(f"[match]  {n_match}/{len(sample)} SNAP rows ({pct:.1f}%) "
          f"matched at least one Liu row within ±10min, ±0.001°")
    if n_match > 0:
        print(f"[match]  examples of MATCH: {examples_match[:2]}")
    if n_match < len(sample):
        print(f"[match]  examples of NO MATCH: {examples_nomatch[:2]}")

    return {
        "range_overlap": rng_overlap,
        "join_coverage": {
            "n_liu_checkins": n_ck,
            "n_with_coordinates": n_with_coords,
            "pct_with_coordinates": 100*n_with_coords/n_ck if n_ck else 0,
            "n_distinct_placeids_with_coords": len(coords),
        },
        "sample_match": {
            "sample_size": int(len(sample)),
            "tolerance_seconds": tol_seconds,
            "tolerance_degrees": tol_lat,
            "matched": int(n_match),
            "pct_matched": float(pct),
            "examples_matched": examples_match[:3],
            "examples_not_matched": examples_nomatch[:3],
        },
    }


# ---------- ID-system comparison ------------------------------------------

def compare_id_systems(snap_info: dict, liu_ckin_info: dict) -> dict:
    """Just look at the numerical ranges of user_id / venue_id to see if they
    could possibly share an indexing convention."""
    banner("ID SYSTEMS — SNAP vs Liu numerical ranges")
    snap = snap_info["_df_handle"]
    liu = liu_ckin_info["_df_handle"]

    snap_users = {"min": int(snap.user_id.min()),
                  "max": int(snap.user_id.max()),
                  "distinct": int(snap.user_id.nunique())}
    liu_users = {"min": int(liu.userid.min()),
                 "max": int(liu.userid.max()),
                 "distinct": int(liu.userid.nunique())}
    snap_venues = {"min": int(snap.location_id.min()),
                   "max": int(snap.location_id.max()),
                   "distinct": int(snap.location_id.nunique())}
    liu_places = {"min": int(liu.placeid.min()),
                  "max": int(liu.placeid.max()),
                  "distinct": int(liu.placeid.nunique())}
    print(f"  SNAP user_id     : {snap_users}")
    print(f"  LIU  userid      : {liu_users}")
    print(f"  SNAP location_id : {snap_venues}")
    print(f"  LIU  placeid     : {liu_places}")

    # Intersection (set-level): identical integer overlap (no semantic mapping)
    snap_uid_set = set(snap.user_id.unique().tolist())
    liu_uid_set = set(liu.userid.unique().tolist())
    snap_iid_set = set(snap.location_id.unique().tolist())
    liu_pid_set = set(liu.placeid.unique().tolist())
    res = {
        "snap_user_id": snap_users,
        "liu_userid": liu_users,
        "snap_location_id": snap_venues,
        "liu_placeid": liu_places,
        "user_id_set_intersection": len(snap_uid_set & liu_uid_set),
        "venue_id_set_intersection": len(snap_iid_set & liu_pid_set),
    }
    print(f"  user_id set intersection : "
          f"{res['user_id_set_intersection']:,} "
          f"(out of SNAP={len(snap_uid_set):,}, LIU={len(liu_uid_set):,})")
    print(f"  venue_id set intersection: "
          f"{res['venue_id_set_intersection']:,} "
          f"(out of SNAP={len(snap_iid_set):,}, LIU={len(liu_pid_set):,})")
    return res


# ---------- main -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=1000,
                        help="SNAP rows to use in the sample-match step")
    parser.add_argument("--skip-comparison", action="store_true",
                        help="Skip SNAP↔Liu sample-match (faster)")
    parser.add_argument("--out", type=Path,
                        default=OUT_DIR / "inspection.json",
                        help="Write the full JSON dump here")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "config": {"seed": SEED, "sample_size": args.sample_size},
    }

    snap_info = inspect_snap()
    report["snap_totalCheckins"] = snap_info
    liu_ckin_info = inspect_liu_checkins()
    report["liu_checkins"] = liu_ckin_info
    liu_s1_info, liu_s2_info = inspect_liu_spots()
    report["liu_spots_subset1"] = liu_s1_info
    report["liu_spots_subset2"] = liu_s2_info
    report["liu_category_structure"] = inspect_liu_categories()
    report["liu_friendship"] = inspect_liu_friendship()
    report["liu_userinfo"] = inspect_liu_userinfo()
    report["shehzad_pkl_reference"] = inspect_dccf_pkl()

    report["id_systems"] = compare_id_systems(snap_info, liu_ckin_info)
    if not args.skip_comparison:
        report["snap_vs_liu"] = compare_snap_vs_liu(
            snap_info, liu_ckin_info, liu_s1_info, liu_s2_info,
            sample_size=args.sample_size,
        )
    else:
        report["snap_vs_liu"] = {"skipped": True}

    # Drop the dataframe handles before serialising
    for k in ("snap_totalCheckins", "liu_checkins",
              "liu_spots_subset1", "liu_spots_subset2"):
        report[k].pop("_df_handle", None)

    banner(f"WRITING JSON OUTPUT → {args.out}")
    args.out.write_text(json.dumps(jsonable(report), indent=2,
                                     default=str), encoding="utf-8")
    print(f"  wrote {args.out.stat().st_size / 1024:.1f} KiB")
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
