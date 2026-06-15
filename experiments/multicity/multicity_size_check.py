"""Round-4 PART 2 — Step 1: carve + k-core size confirmation.

For each candidate city:
  1. Carve from TIST2015 to TSMC2014-format TSV under data/raw_tist/.
  2. Run frozen ``preprocess_tsmc2014`` (k-core=10 + temporal split) to
     produce ``data/processed/<city_key>/``.
  3. Collect post-k-core counts.

Outputs:
  outputs_multicity/selection/size_check_results.json
  outputs_multicity/selection/final_cities.md
  data/raw_tist/<city_key>.tsv
  data/processed/<city_key>/...
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.multicity.tist_carve import (
    carve_cities_batch, load_cities, build_name_to_id,
)
from pipeline.step01_preprocessing import preprocess_tsmc2014

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("size_check")


# Candidates per brief PART-2 Step 1 (round-4 multi-city).
# Compromise window: keep TSMC2014's Apr 2012 – Feb 2013 → clip TIST.
CANDIDATES = {
    # city_key (output name)      TIST official name      TT_share band
    "istanbul":   {"tist": "Istanbul",   "tt_band": "very-low (~0.06)"},
    "bangkok":    {"tist": "Bangkok",    "tt_band": "low (~0.10)"},
    "nyc_tist":   {"tist": "New York",   "tt_band": "low-mid (~0.14)",
                     "round3_bridge": "TSMC NYC (TT~0.25)"},
    "saopaulo":   {"tist": "Sao Paulo",  "tt_band": "low-mid (~0.15)"},
    "tokyo_tist": {"tist": "Tokyo",      "tt_band": "mid-high (~0.45)",
                     "round3_bridge": "TSMC TKY (TT~0.71)"},
}

WINDOW_START = "2012-04"
WINDOW_END   = "2013-02"
KCORE = 10
TAXO_PATH = REPO_ROOT / "config" / "foursquare_legacy_taxonomy.json"


def main() -> int:
    out_sel = REPO_ROOT / "outputs_multicity" / "selection"
    out_sel.mkdir(parents=True, exist_ok=True)
    raw_dir = REPO_ROOT / "data" / "raw_tist"
    raw_dir.mkdir(parents=True, exist_ok=True)

    centers = load_cities()
    targets = {}
    for k, info in CANDIDATES.items():
        if info["tist"] not in centers:
            logger.warning(f"  {info['tist']!r} not found in TIST cities")
            continue
        targets[k] = centers[info["tist"]]

    # 1. Carve (single pass through 2.2 GB Checkins.txt)
    logger.info("[1/3] Carving %d cities from TIST global files "
                  "(window %s → %s) ...",
                  len(targets), WINDOW_START, WINDOW_END)
    carve_results = carve_cities_batch(
        targets, raw_dir,
        name_to_id=build_name_to_id(),
        radius_km=35.0,
        window_start=WINDOW_START, window_end=WINDOW_END,
        verbose=True,
    )
    logger.info("[1/3] Carve done")
    for k, r in carve_results.items():
        logger.info(f"    {k:14s}  venues={r.n_venues_in_radius:>7,}  "
                      f"checkins={r.n_checkins_written:>10,}  "
                      f"users={r.n_users_distinct:>7,}")

    # 2. step01 per city
    logger.info("[2/3] Running preprocess_tsmc2014 per city "
                  "(k-core=%d, 80/10/10 split) ...", KCORE)
    sizes = {}
    for k in CANDIDATES:
        if k not in carve_results:
            continue
        raw_tsv = raw_dir / f"{k}.tsv"
        out_dir = REPO_ROOT / "data" / "processed" / k
        if (out_dir / "metadata.json").exists():
            logger.info(f"  [{k}] processed already exists, reading sizes ...")
            md = json.loads((out_dir / "metadata.json").read_text())
            sizes[k] = {
                "n_users_final": int(md.get("n_users_final", 0)),
                "n_items_final": int(md.get("n_items_final", 0)),
                "n_interactions_final": int(md.get("n_interactions_final", 0)),
                "n_interactions_post_kcore": int(md.get("n_interactions_post_kcore", 0)),
                "from_cache": True,
            }
            continue
        if raw_tsv.stat().st_size < 100_000:
            logger.warning(f"  [{k}] carved TSV is tiny — skipping step01")
            sizes[k] = {"n_users_final": 0, "from_cache": False}
            continue
        t0 = time.time()
        logger.info(f"  [{k}] preprocess_tsmc2014 starting ...")
        try:
            res = preprocess_tsmc2014(
                raw_tsv=raw_tsv,
                out_dir=out_dir,
                taxonomy_path=TAXO_PATH,
                k_core=KCORE,
                seed=42,
            )
            elapsed = time.time() - t0
            sizes[k] = {
                "n_users_final": int(res.n_users_final),
                "n_items_final": int(res.n_items_final),
                "n_interactions_final": int(res.n_interactions_final),
                "n_interactions_post_kcore": int(res.n_interactions_post_kcore),
                "wallclock_s": float(elapsed),
                "from_cache": False,
            }
            logger.info(f"  [{k}] DONE in {elapsed:.1f}s: "
                          f"users={res.n_users_final:,}, "
                          f"items={res.n_items_final:,}, "
                          f"interactions={res.n_interactions_final:,}")
        except Exception as e:
            logger.exception(f"  [{k}] preprocess FAILED: {e}")
            sizes[k] = {"error": str(e), "from_cache": False}

    # 3. Persist + write final_cities.md
    payload = {
        "window_start": WINDOW_START, "window_end": WINDOW_END,
        "k_core": KCORE, "candidates": CANDIDATES,
        "carve": {k: {
            "n_venues_in_radius": r.n_venues_in_radius,
            "n_checkins_written": r.n_checkins_written,
            "n_users_distinct": r.n_users_distinct,
            "wallclock_s": r.wallclock_s,
        } for k, r in carve_results.items()},
        "sizes_post_kcore": sizes,
    }
    (out_sel / "size_check_results.json").write_text(
        json.dumps(payload, indent=2))
    logger.info(f"[3/3] Wrote {out_sel}/size_check_results.json")

    # Markdown final_cities.md
    md = ["# Final candidate sizes (round-4 PART 2, Step 1)", "",
            f"Window: **{WINDOW_START} → {WINDOW_END}** "
            f"(TSMC2014-compatible, 10-mo clip of TIST2015's 18-mo span).",
            f"k-core: **{KCORE}** on user × venue bipartite (iterative).",
            "",
            "| city_key | TIST name | TT_share band | carved checkins | post-k-core users | post-k-core items | post-k-core interactions | flag |",
            "|---|---|---|---:|---:|---:|---:|---|"]
    for k, info in CANDIDATES.items():
        s = sizes.get(k, {})
        c = carve_results.get(k)
        nu = int(s.get("n_users_final", 0))
        flag = "OK" if nu >= 2000 else ("UNDERPOWERED" if nu > 0 else "FAILED")
        bridge = " · " + info.get("round3_bridge", "") if "round3_bridge" in info else ""
        md.append(
            f"| `{k}` | {info['tist']}{bridge} | {info['tt_band']} | "
            f"{(c.n_checkins_written if c else 0):,} | "
            f"{nu:,} | {int(s.get('n_items_final', 0)):,} | "
            f"{int(s.get('n_interactions_final', 0)):,} | **{flag}** |"
        )
    md += ["",
            "## Round-3 bridge cities (provenance probe)",
            "",
            "`nyc_tist` and `tokyo_tist` are the **same real cities** as the",
            "TSMC2014 round-3 anchors (NYC, TKY), seen through a different",
            "collection pipeline. They serve **two distinct roles** that must",
            "be kept clearly separate in all subsequent outputs:",
            "",
            "1. **Spectrum point** — they are simply two of the TIST cities",
            "   along the TT_share axis (NYC-TIST at ~0.14, Tokyo-TIST at",
            "   ~0.45). Treat them as any other TIST city in the cross-city",
            "   analysis.",
            "2. **Provenance probe** — comparing `nyc_tist` vs TSMC `NYC` is",
            "   NOT a city comparison; it is a *same-city, different-dataset*",
            "   comparison. Any differences reflect collection-pipeline",
            "   sensitivity, NOT method instability. Same for `tokyo_tist`",
            "   vs TSMC `TKY`.",
            "",
            "**Never merge** TSMC NYC + TIST NYC (or TSMC TKY + TIST Tokyo)",
            "as if they were one dataset.",
            "",
            "## Acceptance",
            "",
            "Drop any city with post-k-core users < ~2 000 (the round-3 NYC",
            "floor had 829; we set a slightly higher bar here to ensure",
            "per-stratum CIs from B9.1 are credible). Cities flagged OK above",
            "proceed to PART-2 Steps 2-4.",
        ]
    (out_sel / "final_cities.md").write_text("\n".join(md))
    logger.info(f"Wrote {out_sel}/final_cities.md")
    print("\n=== ROLL-UP ===")
    for k in CANDIDATES:
        s = sizes.get(k, {})
        c = carve_results.get(k)
        print(f"  {k:14s}  carved={c.n_checkins_written if c else 0:>10,}  "
              f"post-kcore users={int(s.get('n_users_final', 0)):>6,}  "
              f"items={int(s.get('n_items_final', 0)):>6,}  "
              f"int={int(s.get('n_interactions_final', 0)):>9,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
