"""TIST2015 → TSMC2014-format per-city carver.

The frozen ``pipeline.step01_preprocessing.preprocess_tsmc2014`` expects
the TSMC2014 8-column TSV layout:

    user_id  venue_id  cat_id  cat_name  lat  lon  tz_offset_min  utc_time

The TIST2015 raw files are split:
    - Checkins.txt (4 cols: user, venue, utc_time, tz_offset)
    - POIs.txt    (5 cols: venue, lat, lon, cat_NAME, country)

Crucially POIs has only the category NAME, not the cat_id. The frozen
taxonomy module maps cat_id → macro, so we build a *reverse* lookup
(cat_name → cat_id) by walking the same v2 taxonomy JSON. Unknown
names fall back to a sentinel cat_id reserved for "Other" (the
frozen step01 maps anything not in the taxonomy to macro "Other"
anyway, so this is invariant).

Public API (single function):

    carve_city(city_name, out_path, *, radius_km=35.0,
                window_start='2012-04', window_end='2013-02') -> CarveResult

Streams the global files once per city — for batched carving of many
cities, prefer :func:`carve_cities_batch`.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw"
TAXO_PATH = REPO_ROOT / "config" / "foursquare_legacy_taxonomy.json"
POIS = DATA_DIR / "dataset_TIST2015_POIs.txt"
CHECKINS = DATA_DIR / "dataset_TIST2015_Checkins.txt"
CITIES_FILE = DATA_DIR / "dataset_TIST2015_Cities.txt"

# Fallback cat_id for unmapped POI names. Picked to be lexicographically
# distinct from any real Foursquare cat_id; step01 will map it to "Other".
OTHER_CAT_ID = "00000000000000000000ffff"


# ---------------------------------------------------------------------------
# Taxonomy reverse lookup
# ---------------------------------------------------------------------------

def build_name_to_id(taxonomy_path: Path = TAXO_PATH) -> dict[str, str]:
    """{cat_name (lowercased) → cat_id (hex)}. Walks the v2 tree."""
    data = json.loads(taxonomy_path.read_text(encoding="latin-1"))
    out: dict[str, str] = {}

    def _walk(node: dict) -> None:
        nm = node.get("name")
        cid = node.get("id")
        if nm and cid:
            out[nm.lower()] = cid
        for ch in node.get("categories") or []:
            _walk(ch)

    if isinstance(data, list):
        for top in data:
            _walk(top)
    else:
        for top in data["response"]["categories"]:
            _walk(top)
    return out


# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------

def load_cities(path: Path = CITIES_FILE) -> dict[str, tuple[float, float]]:
    out = {}
    with path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            try:
                out[parts[0]] = (float(parts[1]), float(parts[2]))
            except ValueError:
                continue
    return out


def haversine_km(lat1: float, lon1: float,
                    lat2: float, lon2: float) -> float:
    R = 6371.0088
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Per-city carve
# ---------------------------------------------------------------------------

@dataclass
class CarveResult:
    city: str
    out_path: Path
    n_venues_in_radius: int
    n_checkins_written: int
    n_users_distinct: int
    name_lookup_hits: int
    name_lookup_misses: int
    wallclock_s: float


def _utc_in_window(s: str, ws: str, we: str) -> bool:
    """`s` like 'Tue Apr 03 18:00:06 +0000 2012'.
    `ws` / `we` like '2012-04' / '2013-02' (inclusive, month-level).
    Empty `ws`/`we` → no clip."""
    if not ws and not we:
        return True
    parts = s.split()
    if len(parts) < 6:
        return False
    month_map = {"Jan":"01","Feb":"02","Mar":"03","Apr":"04","May":"05",
                  "Jun":"06","Jul":"07","Aug":"08","Sep":"09","Oct":"10",
                  "Nov":"11","Dec":"12"}
    ym = f"{parts[5]}-{month_map.get(parts[1], '00')}"
    if ws and ym < ws:
        return False
    if we and ym > we:
        return False
    return True


def carve_cities_batch(targets: dict[str, tuple[float, float]],
                          out_dir: Path,
                          *,
                          name_to_id: dict[str, str] | None = None,
                          radius_km: float = 35.0,
                          window_start: str = "2012-04",
                          window_end: str = "2013-02",
                          verbose: bool = True) -> dict[str, CarveResult]:
    """Carve multiple cities in one pass through POIs + Checkins.

    Args:
        targets: city_key → (lat, lon) of city centre.
        out_dir: where to drop ``<city_key>.tsv``.
        name_to_id: cat_name → cat_id mapping (built if None).
        radius_km: max haversine distance from city centre.
        window_start, window_end: inclusive 'YYYY-MM' bounds; empty
            string disables that side.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if name_to_id is None:
        name_to_id = build_name_to_id()

    t0 = time.time()
    # 1. POIs pass — build venue → (city_key, cat_id, cat_name, lat, lon)
    centers = list(targets.items())
    venue_idx: dict[str, tuple[str, str, str, float, float]] = {}
    name_hits = 0; name_miss = 0
    if verbose:
        print(f"[carve] streaming POIs.txt ...")
    with POIS.open() as f:
        for i, line in enumerate(f, 1):
            if verbose and i % 500_000 == 0:
                print(f"  POIs: {i:,} scanned, "
                      f"{len(venue_idx):,} kept ({time.time()-t0:.1f}s)")
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            v_id, lat_s, lon_s, cat_name, _country = parts[:5]
            try:
                lat = float(lat_s); lon = float(lon_s)
            except ValueError:
                continue
            best_key = None; best_d = radius_km + 1.0
            for ckey, (clat, clon) in centers:
                d = haversine_km(lat, lon, clat, clon)
                if d < best_d:
                    best_d = d; best_key = ckey
            if best_key is None:
                continue
            cid = name_to_id.get(cat_name.lower(), OTHER_CAT_ID)
            if cid == OTHER_CAT_ID:
                name_miss += 1
            else:
                name_hits += 1
            venue_idx[v_id] = (best_key, cid, cat_name, lat, lon)
    if verbose:
        print(f"  POIs done: {len(venue_idx):,} venues kept across "
              f"{len(targets)} cities  ({time.time()-t0:.1f}s)")

    # 2. Open one per-city TSV
    handles = {}
    counts = {ck: {"checkins": 0, "users": set()} for ck in targets}
    out_paths = {}
    for ck in targets:
        p = out_dir / f"{ck}.tsv"
        handles[ck] = p.open("w", encoding="utf-8")
        out_paths[ck] = p

    # 3. Checkins streaming
    if verbose:
        print(f"[carve] streaming Checkins.txt (window "
              f"{window_start!r} → {window_end!r}) ...")
    n_rows = 0; n_kept = 0
    with CHECKINS.open() as f:
        for line in f:
            n_rows += 1
            if verbose and n_rows % 2_000_000 == 0:
                print(f"  Checkins: {n_rows:,} scanned, "
                      f"{n_kept:,} written  ({time.time()-t0:.1f}s)")
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            u_id, v_id, utc, tz = parts[:4]
            hit = venue_idx.get(v_id)
            if hit is None:
                continue
            if not _utc_in_window(utc, window_start, window_end):
                continue
            ckey, cid, cat_name, lat, lon = hit
            # TSMC2014 8-col layout
            handles[ckey].write(
                f"{u_id}\t{v_id}\t{cid}\t{cat_name}\t"
                f"{lat:.15f}\t{lon:.15f}\t{tz}\t{utc}\n"
            )
            counts[ckey]["checkins"] += 1
            counts[ckey]["users"].add(u_id)
            n_kept += 1

    for h in handles.values():
        h.close()

    # 4. Build per-city venue-in-radius count
    vinr = {ck: 0 for ck in targets}
    for (ckey, _cid, _cn, _lat, _lon) in venue_idx.values():
        vinr[ckey] += 1

    results = {}
    for ck in targets:
        results[ck] = CarveResult(
            city=ck, out_path=out_paths[ck],
            n_venues_in_radius=vinr[ck],
            n_checkins_written=counts[ck]["checkins"],
            n_users_distinct=len(counts[ck]["users"]),
            name_lookup_hits=name_hits,
            name_lookup_misses=name_miss,
            wallclock_s=time.time() - t0,
        )
    return results


# Convenience wrapper for a single city
def carve_city(city_name: str,
                 out_path: Path,
                 city_center: tuple[float, float] | None = None,
                 **kwargs) -> CarveResult:
    centers = load_cities()
    if city_center is None:
        if city_name not in centers:
            raise ValueError(f"{city_name!r} not in TIST Cities.txt")
        city_center = centers[city_name]
    return carve_cities_batch({city_name: city_center},
                                 out_path.parent, **kwargs)[city_name]


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", required=True,
                          help="Comma list of TIST city names.")
    parser.add_argument("--out", default="data/raw_tist")
    parser.add_argument("--radius", type=float, default=35.0)
    parser.add_argument("--start", default="2012-04")
    parser.add_argument("--end",   default="2013-02")
    args = parser.parse_args()
    centers = load_cities()
    targets = {}
    for raw in args.cities.split(","):
        c = raw.strip()
        if c not in centers:
            print(f"  WARN: {c!r} not in TIST cities; skipping")
            continue
        targets[c.lower().replace(" ", "_")] = centers[c]
    res = carve_cities_batch(targets, REPO_ROOT / args.out,
                                radius_km=args.radius,
                                window_start=args.start,
                                window_end=args.end)
    for k, r in res.items():
        print(f"  {k}: venues={r.n_venues_in_radius:,}  "
              f"checkins_written={r.n_checkins_written:,}  "
              f"distinct_users={r.n_users_distinct:,}")
    print(f"name lookup hits/miss: "
            f"{res[list(targets)[0]].name_lookup_hits} / "
            f"{res[list(targets)[0]].name_lookup_misses}")
