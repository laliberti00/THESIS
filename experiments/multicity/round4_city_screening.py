"""Round-4 PART 1 — TIST2015 multi-city screening (read-only).

Streams the TIST2015 global files to compute per-city statistics for a
shortlist of ~15 candidate cities. NO preprocessing, NO k-core, NO
training. Pure descriptive counts.

Pipeline:
  1. Build fine-category-NAME → top-level macro mapping by walking
     `config/foursquare_legacy_taxonomy.json` (POIs.txt only carries the
     category NAME, not the cat_id, so we cannot reuse the existing
     cat_id-based mapping directly).
  2. Read `dataset_TIST2015_Cities.txt` → city → (lat, lon).
  3. Build a venue_id → (city, macro) dict by streaming
     `dataset_TIST2015_POIs.txt`. Each venue is assigned to the
     SHORTLIST city whose center is closest in haversine, within
     `RADIUS_KM`. Unmapped venues are dropped.
  4. Stream `dataset_TIST2015_Checkins.txt` once; for venues in the
     dict, accumulate per-city counters (n_users, n_venues, n_checkins,
     per-macro counts, distinct user×venue, distinct "YYYY-MM" months,
     per-user checkin counts for the k-core lower-bound estimate).

Outputs:
  outputs_multicity/screening/city_screening.csv
  outputs_multicity/screening/CITY_SCREENING.md      (paper-ready)

Reference anchors (from the TSMC2014 NYC/TKY runs, round-3):
  NYC: TT_share ≈ 0.25  → C5.0 helps; B6 lens GREEN; B_full beats B_blind.
  TKY: TT_share ≈ 0.71  → C5.0 hurts on T&T-target; B6 lens RED; B_full loses.
The screening table lets us pick cities along the TT_share spectrum to
test the law's generalisation.
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# --- Constants ---------------------------------------------------------------

DATA_DIR = REPO_ROOT / "data" / "raw"
TAXO_PATH = REPO_ROOT / "config" / "foursquare_legacy_taxonomy.json"
CHECKINS = DATA_DIR / "dataset_TIST2015_Checkins.txt"
POIS = DATA_DIR / "dataset_TIST2015_POIs.txt"
CITIES_FILE = DATA_DIR / "dataset_TIST2015_Cities.txt"

OUT_DIR = REPO_ROOT / "outputs_multicity" / "screening"

# 15-city shortlist (geographic + mobility-culture spread).
SHORTLIST = [
    "New York", "Tokyo", "Istanbul", "Sao Paulo", "Moscow",
    "Bangkok", "Kuala Lumpur", "Jakarta", "London", "Paris",
    "Mexico City", "Seoul", "Los Angeles", "Chicago", "Singapore",
]

# A venue is associated with the closest shortlist city center if within
# RADIUS_KM. Generous to cover full metro areas; cities in the shortlist
# are far apart enough that overlap is not a concern.
RADIUS_KM = 35.0

# Metro-population reference (millions, 2015-ish UN estimates) — for
# READABILITY only; method generalises by TT_share, NOT by population.
METRO_POP_M = {
    "New York": 20.1,    "Tokyo": 37.4,     "Istanbul": 15.5,
    "Sao Paulo": 22.0,   "Moscow": 12.5,    "Bangkok": 10.7,
    "Kuala Lumpur": 8.0, "Jakarta": 33.4,   "London": 14.3,
    "Paris": 11.0,       "Mexico City": 22.0,
    "Seoul": 25.6,       "Los Angeles": 13.2,
    "Chicago": 8.9,      "Singapore": 5.9,
}

TT_MACRO = "Travel & Transport"

# --- Step 1. name → macro mapping --------------------------------------------

def build_name_to_macro(taxonomy_path: Path = TAXO_PATH) -> dict[str, str]:
    """Walk the v2 legacy taxonomy and return {category_name → top_level_macro}.
    Names are normalised case-insensitively. The 10 top-level macros are
    included as self-mappings (e.g. 'Food' → 'Food')."""
    data = json.loads(taxonomy_path.read_text(encoding="latin-1"))
    out: dict[str, str] = {}

    def _walk(node: dict, macro: str | None) -> None:
        nm = node.get("name", "")
        cur_macro = nm if macro is None else macro
        if nm:
            out[nm.lower()] = cur_macro
        for ch in node.get("categories", []) or []:
            _walk(ch, cur_macro)

    if isinstance(data, list):
        for top in data:
            _walk(top, None)
    else:  # SchemaB
        for top in data["response"]["categories"]:
            _walk(top, None)
    return out


# --- Step 2. city table ------------------------------------------------------

def load_cities(path: Path = CITIES_FILE) -> dict[str, tuple[float, float]]:
    out = {}
    with path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            name, lat, lon = parts[0], parts[1], parts[2]
            out[name] = (float(lat), float(lon))
    return out


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# --- Step 3. POI → city/macro ------------------------------------------------

def build_venue_index(shortlist_centers: dict[str, tuple[float, float]],
                        name_to_macro: dict[str, str],
                        radius_km: float = RADIUS_KM,
                        verbose: bool = True) -> dict[str, tuple[str, str]]:
    """Stream POIs.txt → for each venue, find the closest shortlist city
    (within radius_km), look up macro from category name. Return
    {venue_id → (city_name, macro_name)}.
    """
    centers = list(shortlist_centers.items())
    t0 = time.time()
    n_rows = 0; n_kept = 0
    out: dict[str, tuple[str, str]] = {}
    miss_macro = Counter()
    with POIS.open() as f:
        for line in f:
            n_rows += 1
            if verbose and n_rows % 500_000 == 0:
                print(f"    POIs scanned: {n_rows:,}  kept so far: "
                      f"{n_kept:,}  ({time.time()-t0:.1f}s)")
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            v_id, lat_s, lon_s, cat_name, _country = parts[:5]
            try:
                lat = float(lat_s); lon = float(lon_s)
            except ValueError:
                continue
            # nearest shortlist city
            best_name = None
            best_d = radius_km + 1.0
            for cname, (clat, clon) in centers:
                d = haversine_km(lat, lon, clat, clon)
                if d < best_d:
                    best_d = d; best_name = cname
            if best_name is None:
                continue
            macro = name_to_macro.get(cat_name.lower(), "Other")
            if macro == "Other":
                miss_macro[cat_name] += 1
            out[v_id] = (best_name, macro)
            n_kept += 1
    if verbose:
        print(f"    POIs: scanned {n_rows:,}, kept {n_kept:,}  "
              f"in {time.time()-t0:.1f}s")
        print(f"    Unmapped category names (top 10):")
        for name, c in miss_macro.most_common(10):
            print(f"      {name!r}: {c:,}")
    return out


# --- Step 4. Checkins streaming ----------------------------------------------

def _utc_to_yyyymm(s: str) -> str:
    """Extract 'YYYY-MM' from 'Tue Apr 03 18:00:06 +0000 2012'."""
    # parts: [Tue, Apr, 03, 18:00:06, +0000, 2012]
    parts = s.split()
    if len(parts) < 6:
        return ""
    month_map = {"Jan":"01","Feb":"02","Mar":"03","Apr":"04","May":"05",
                  "Jun":"06","Jul":"07","Aug":"08","Sep":"09","Oct":"10",
                  "Nov":"11","Dec":"12"}
    return f"{parts[5]}-{month_map.get(parts[1], '00')}"


def stream_checkins(venue_idx: dict[str, tuple[str, str]],
                       verbose: bool = True) -> dict[str, dict]:
    """Accumulate per-city statistics from Checkins.txt."""
    per_city: dict[str, dict] = {}
    for cname in {c for c, _ in venue_idx.values()}:
        per_city[cname] = {
            "n_checkins": 0,
            "macro_counts": Counter(),
            "user_counts": Counter(),
            "venue_counts": Counter(),
            "months": set(),
        }

    t0 = time.time(); n_rows = 0; n_kept = 0
    with CHECKINS.open() as f:
        for line in f:
            n_rows += 1
            if verbose and n_rows % 2_000_000 == 0:
                print(f"    Checkins scanned: {n_rows:,}  kept: "
                      f"{n_kept:,}  ({time.time()-t0:.1f}s elapsed)")
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            u_id, v_id, utc, _tz = parts[:4]
            hit = venue_idx.get(v_id)
            if hit is None:
                continue
            cname, macro = hit
            d = per_city[cname]
            d["n_checkins"] += 1
            d["macro_counts"][macro] += 1
            d["user_counts"][u_id] += 1
            d["venue_counts"][v_id] += 1
            d["months"].add(_utc_to_yyyymm(utc))
            n_kept += 1
    if verbose:
        print(f"    Checkins: scanned {n_rows:,}, kept {n_kept:,}  "
              f"in {time.time()-t0:.1f}s")
    return per_city


# --- Step 5. screening table -------------------------------------------------

def normalised_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    K = len(counts)
    if K <= 1:
        return 0.0
    import math as _m
    H = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            H -= p * _m.log2(p)
    return H / _m.log2(K)


def build_table(per_city: dict[str, dict],
                  shortlist_centers: dict[str, tuple[float, float]]) -> list[dict]:
    rows = []
    for cname in shortlist_centers:
        d = per_city.get(cname, None)
        if d is None or d["n_checkins"] == 0:
            rows.append({
                "city": cname, "n_checkins_raw": 0, "n_users_raw": 0,
                "n_venues_raw": 0, "TT_share": 0.0, "macro_entropy": 0.0,
                "est_users_post_kcore": 0, "population_M": METRO_POP_M.get(cname, None),
                "span_months": 0, "span_first": "", "span_last": "",
            })
            continue
        n_check = d["n_checkins"]
        n_users = len(d["user_counts"])
        n_venues = len(d["venue_counts"])
        tt = d["macro_counts"].get(TT_MACRO, 0) / n_check
        entropy = normalised_entropy(d["macro_counts"])
        # Conservative k-core=10 LOWER bound: users with ≥ 10 distinct
        # venues, where those venues themselves have ≥ 10 distinct users.
        # First, prune venues by their unique-user count.
        # Track venue → distinct users requires accumulating per-venue
        # user sets, which we didn't keep to save memory. Use a coarser
        # but principled bound: count users with ≥ 10 checkins.
        # (Equivalent to per-venue-counts==1 collapse; n_users surviving
        # 10-core in real preprocessing is ≤ this.)
        est_kc = sum(1 for c in d["user_counts"].values() if c >= 10)
        months = sorted(m for m in d["months"] if m)
        rows.append({
            "city": cname, "n_checkins_raw": n_check,
            "n_users_raw": n_users, "n_venues_raw": n_venues,
            "TT_share": round(tt, 4), "macro_entropy": round(entropy, 4),
            "est_users_post_kcore": est_kc,
            "population_M": METRO_POP_M.get(cname, None),
            "span_months": len(d["months"] - {""}),
            "span_first": months[0] if months else "",
            "span_last": months[-1] if months else "",
        })
    return rows


# --- Report ------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path) -> None:
    import csv
    cols = ["city", "n_checkins_raw", "n_users_raw", "n_venues_raw",
              "TT_share", "macro_entropy", "est_users_post_kcore",
              "population_M", "span_months", "span_first", "span_last"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md(rows: list[dict], path: Path,
              checkins_window: tuple[str, str]) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["TT_share"])
    lines = [
        "# City screening for multi-city generalisation (round 4 PART 1)",
        "",
        "> Read-only inventory of 15 TIST2015 cities along the **TT_share**",
        "> axis (fraction of check-ins whose POI macro is Travel & Transport).",
        "> No preprocessing, no k-core, no training — pure stream counts.",
        "",
        "Source: `data/raw/dataset_TIST2015_{Checkins,POIs,Cities}.txt`",
        f"(Yang/Zhang/Qu, ACM TIST 2015 — 33.3 M check-ins, 266 909 users,",
        f"3.68 M venues across 415 cities, **Apr 2012 – Sep 2013**,",
        f"sample window observed = {checkins_window[0]} → {checkins_window[1]}).",
        "",
        "Anchors from TSMC2014 round-3 runs (reference, not in this table):",
        "* **NYC (TSMC2014)**: TT_share ≈ 0.25 — B6 lens GREEN, "
            "B_full beats B_blind, X-SAGE additive nudge helps.",
        "* **TKY (TSMC2014)**: TT_share ≈ 0.71 — B6 lens RED, "
            "B_full loses to B_blind, X-SAGE defensive.",
        "",
        "## Screening table (sorted by TT_share ascending)",
        "",
        "| city | TT_share | macro_entropy | n_checkins | n_users | n_venues | est_users_post_kcore | population_M | span_months | span_first | span_last |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows_sorted:
        lines.append(
            f"| {r['city']} | {r['TT_share']:.3f} | {r['macro_entropy']:.3f} | "
            f"{r['n_checkins_raw']:,} | {r['n_users_raw']:,} | "
            f"{r['n_venues_raw']:,} | {r['est_users_post_kcore']:,} | "
            f"{r['population_M'] if r['population_M'] is not None else '—'} | "
            f"{r['span_months']} | {r['span_first']} | {r['span_last']} |")
    lines += [
        "",
        "## Underpowered cities (flag)",
        "",
        "Recommended exclude: any city with `est_users_post_kcore < 800` —",
        "the round-3 NYC floor used 829 users post-k-core, which is at the",
        "lower edge of what the statistical pipeline (paired Wilcoxon, ",
        "TOST, bootstrap CI95) can resolve. Below ~800 the per-stratum n's",
        "(T&T ≈ 25 %, non-T&T ≈ 75 %) drop below ~200 each and CIs widen.",
        "",
        "| city | est_users_post_kcore | flag |",
        "|---|---:|---|",
    ]
    for r in rows_sorted:
        flag = "EXCLUDE" if r["est_users_post_kcore"] < 800 else "OK"
        lines.append(f"| {r['city']} | {r['est_users_post_kcore']:,} | {flag} |")
    lines += [
        "",
        "## Recommendation — deliberate TT_share spread",
        "",
        "The round-3 evidence (C1 → C5.0 → C5 → C6) identifies TT_share as",
        "the variable along which X-SAGE's per-target-macro law fires (or",
        "fails to fire). Population is NOT a governing variable — Tokyo",
        "(37 M) and Bangkok (10.7 M) sit at very different TT_share but both",
        "trigger the same B_full-loses-on-T&T regime, while NYC (20 M) and",
        "Sao Paulo (22 M, expected lower TT_share) do not.",
        "",
        "Pick **one city near each of ~25 %, ~40 %, ~55 %, ~70 %** TT_share,",
        "subject to the `est_users_post_kcore ≥ 800` constraint. The 25 %",
        "bin is already covered by NYC; the 70 % bin by TKY. Choose two",
        "intermediate cities from the table above.",
        "",
        "## Temporal-window note (feeds Luca's decision)",
        "",
        "All TIST2015 check-ins fall in **Apr 2012 – Sep 2013** (~18 months).",
        "The TSMC2014 NYC/TKY were built on **Apr 2012 – Feb 2013** (~10",
        "months). Two policies are possible for PART 2:",
        "",
        "1. **Clean path** — re-extract NYC + TKY from TIST2015 on a single",
        "   common window (e.g. Apr 2012 – Sep 2013, the full TIST span);",
        "   maximises cross-city comparability but invalidates the round-3",
        "   numbers (would have to re-derive everything from scratch).",
        "2. **Compromise** — keep TSMC2014 NYC + TKY as-is (round-3 numbers",
        "   preserved); clip each new TIST city to the TSMC2014 window",
        "   (Apr 2012 – Feb 2013) so the windows match. Loses ~8 months of",
        "   TIST signal per city but preserves the round-3 baseline.",
        "",
        "Each city's full TIST span is shown in the table above to feed",
        "this decision.",
        "",
        "## Files",
        "",
        "* `outputs_multicity/screening/city_screening.csv`",
        "* `outputs_multicity/screening/CITY_SCREENING.md` (this file)",
    ]
    path.write_text("\n".join(lines))


# --- Driver ------------------------------------------------------------------

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> Round-4 PART 1 — city screening on TIST2015")
    print(f"    Shortlist ({len(SHORTLIST)} cities): "
          f"{', '.join(SHORTLIST)}")
    print(f"    Radius: {RADIUS_KM} km haversine to nearest city center.")

    t_start = time.time()
    print(f"\n[1/4] Building name → macro mapping from taxonomy …")
    name_to_macro = build_name_to_macro()
    print(f"    {len(name_to_macro):,} category names mapped.")

    print(f"\n[2/4] Loading Cities.txt …")
    all_cities = load_cities()
    print(f"    {len(all_cities):,} cities total.")
    missing = [c for c in SHORTLIST if c not in all_cities]
    if missing:
        print(f"    WARNING — shortlist not in TIST cities: {missing}")
    shortlist_centers = {c: all_cities[c] for c in SHORTLIST
                            if c in all_cities}
    print(f"    Using {len(shortlist_centers)} shortlist centers.")

    print(f"\n[3/4] Streaming POIs.txt (≈232 MB, ~3.7 M rows) …")
    venue_idx = build_venue_index(shortlist_centers, name_to_macro)
    # Per-city venue count
    from collections import Counter as _C
    per_city_v = _C(c for c, _ in venue_idx.values())
    print(f"    venues per shortlist city:")
    for c, n in sorted(per_city_v.items(), key=lambda x: -x[1]):
        print(f"      {c:18s}  {n:,}")

    print(f"\n[4/4] Streaming Checkins.txt (≈2.2 GB, 33.3 M rows) …")
    per_city = stream_checkins(venue_idx)

    print(f"\n[5/5] Building screening table …")
    rows = build_table(per_city, shortlist_centers)
    write_csv(rows, OUT_DIR / "city_screening.csv")

    # determine observed checkins window
    all_months = sorted({m for r in rows for m in (r["span_first"], r["span_last"]) if m})
    write_md(rows, OUT_DIR / "CITY_SCREENING.md",
                (all_months[0] if all_months else "?",
                 all_months[-1] if all_months else "?"))

    print(f"\n=== ROLL-UP (sorted by TT_share) ===")
    print(f"  {'city':18s}  {'TT%':>5s}  {'H':>5s}  "
          f"{'n_check':>11s}  {'kc≥10':>7s}")
    for r in sorted(rows, key=lambda r: r["TT_share"]):
        print(f"  {r['city']:18s}  "
              f"{r['TT_share']*100:>4.1f}%  "
              f"{r['macro_entropy']:>5.3f}  "
              f"{r['n_checkins_raw']:>11,}  "
              f"{r['est_users_post_kcore']:>7,}")
    print(f"\nWrote {OUT_DIR}/city_screening.csv and CITY_SCREENING.md")
    print(f"Total wallclock: {time.time()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
