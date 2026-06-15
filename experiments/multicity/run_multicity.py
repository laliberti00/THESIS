"""Round-4 PART 2 — multi-city pipeline orchestrator.

Single entry point Luca runs from the VS Code terminal:

    python experiments/multicity/run_multicity.py --cities all --stages all

Features:
  * tqdm progress (per-city outer bar; per-stage inner bar)
  * Live human-readable log to stdout AND outputs_multicity/logs/<run>.log
  * Checkpointing: each (city, stage) writes a .done_<stage> marker
    under outputs_multicity/<city>/; re-running skips done unless --force
  * Graceful failure: a (city, stage) error is logged + .fail_<stage>
    written, and the loop CONTINUES to the next city
  * --smoke: tiny user cap on nyc_tist (smallest carved), full chain
    in <2 min, validates plumbing
  * Reuses frozen pipeline.step01_preprocessing and the X-SAGE / floor
    scripts via subprocess — no forking of frozen modules

Stages: carve, step01, backbones, stageA, stageB, stageC, stageD, all.

Per-city heavy artefacts continue to land under outputs/<city>/ via the
frozen pipeline (same convention as TSMC NYC/TKY, TKY_BAL). The
multi-city dir outputs_multicity/<city>/ stores the per-stage done
markers + the headline summary JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.multicity.tist_carve import (
    carve_cities_batch, load_cities, build_name_to_id,
)
from pipeline.step01_preprocessing import preprocess_tsmc2014


# ----------------------------------------------------------------------------
# City registry — keep in sync with multicity_size_check.py
# ----------------------------------------------------------------------------

CITIES = {
    "istanbul":   {"tist": "Istanbul",   "tt_band": "very-low (~0.06)",
                     "expected_users": 22631},
    "bangkok":    {"tist": "Bangkok",    "tt_band": "low (~0.10)",
                     "expected_users": 6316},
    "nyc_tist":   {"tist": "New York",   "tt_band": "low-mid (~0.14)",
                     "expected_users": 4113,
                     "round3_bridge": "TSMC NYC (TT~0.25)"},
    "saopaulo":   {"tist": "Sao Paulo",  "tt_band": "low-mid (~0.15)",
                     "expected_users": 4395},
    "tokyo_tist": {"tist": "Tokyo",      "tt_band": "mid-high (~0.45)",
                     "expected_users": 7160,
                     "round3_bridge": "TSMC TKY (TT~0.71)"},
}

ALL_STAGES = ("carve", "step01", "backbones",
                  "stageA", "stageB", "stageC", "stageD")

WINDOW_START = "2012-04"
WINDOW_END   = "2013-02"
KCORE = 10
TAXO_PATH = REPO_ROOT / "config" / "foursquare_legacy_taxonomy.json"
MC_ROOT = REPO_ROOT / "outputs_multicity"
LOG_DIR = MC_ROOT / "logs"
PYTHON = sys.executable  # the venv's python


# ----------------------------------------------------------------------------
# Wallclock heuristic — for the upfront ETA
# ----------------------------------------------------------------------------

def estimate_eta_seconds(city_key: str, stages: list[str]) -> int:
    """Round-3 wallclocks (NYC scale) used as the unit; scale linearly
    in users for the dominant per-city cost."""
    n_users = CITIES[city_key].get("expected_users", 1000)
    # baseline: NYC scale (829 users post-kcore) used these times
    base = {
        "carve":     0,  # carve is done once globally, not per city
        "step01":   60,
        "backbones": 60 * 12,  # floor + B_full tuning
        "stageA":   60 * 3,
        "stageB":   30,
        "stageC":   30,
        "stageD":   60 * 5,
    }
    scale = n_users / 829.0
    total = 0
    for s in stages:
        if s == "carve":
            continue
        # rough sublinear: scale ** 0.8 (parts of step01 are O(N), parts
        # are bounded by batch size + d=128 BPR which is closer to O(N^0.7))
        total += int(base.get(s, 60) * (scale ** 0.85))
    return max(60, total)


def hms(seconds: int) -> str:
    h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60)
    if h: return f"{h}h{m:02d}m"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

def setup_logging(run_tag: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logf = LOG_DIR / f"run_{run_tag}.log"
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("multicity")
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers on re-entry
    for h in list(logger.handlers):
        logger.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(logf)
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    logger.info(f"log file: {logf}")
    return logger


# ----------------------------------------------------------------------------
# Stage helpers
# ----------------------------------------------------------------------------

def city_dir(city: str) -> Path:
    p = MC_ROOT / city
    p.mkdir(parents=True, exist_ok=True)
    return p


def done_marker(city: str, stage: str) -> Path:
    return city_dir(city) / f".done_{stage}"


def fail_marker(city: str, stage: str) -> Path:
    return city_dir(city) / f".fail_{stage}"


def write_done(city: str, stage: str, payload: dict | None = None) -> None:
    p = done_marker(city, stage)
    p.write_text(json.dumps(payload or {"ok": True, "at": _now()},
                                indent=2, default=str))
    fail_marker(city, stage).unlink(missing_ok=True)


def write_fail(city: str, stage: str, err: str) -> None:
    fail_marker(city, stage).write_text(err)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def is_done(city: str, stage: str) -> bool:
    return done_marker(city, stage).exists()


def _run_subprocess(cmd: list[str], logger: logging.Logger,
                       desc: str, log_file: Path | None = None) -> int:
    """Run a subprocess streaming stderr+stdout to logger.debug; capture
    full output in `log_file` if provided. Returns return code."""
    logger.info(f"$ {' '.join(cmd)}")
    env = os.environ.copy()
    with subprocess.Popen(cmd, cwd=REPO_ROOT, env=env,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as p:
        lines = []
        for line in p.stdout:
            lines.append(line)
            logger.debug(line.rstrip())
        rc = p.wait()
    if log_file:
        log_file.write_text("".join(lines))
    return rc


# ----------------------------------------------------------------------------
# Stage runners
# ----------------------------------------------------------------------------

def stage_carve(cities: list[str], force: bool,
                  logger: logging.Logger) -> dict[str, str]:
    """Carve ALL cities in one pass through the global files."""
    centers = load_cities()
    name_to_id = build_name_to_id()
    targets = {}
    statuses = {}
    for c in cities:
        info = CITIES[c]
        tsv = REPO_ROOT / "data" / "raw_tist" / f"{c}.tsv"
        if tsv.exists() and not force:
            logger.info(f"[{c}] carve already done ({tsv.stat().st_size//1024**2} MB) — skipping")
            statuses[c] = "skipped"
            continue
        if info["tist"] not in centers:
            logger.warning(f"[{c}] TIST name {info['tist']!r} not found — fail")
            statuses[c] = "failed"
            continue
        targets[c] = centers[info["tist"]]
    if not targets:
        return statuses
    logger.info(f"[carve] streaming TIST global files for "
                  f"{len(targets)} cities (window {WINDOW_START} → {WINDOW_END})")
    t0 = time.time()
    res = carve_cities_batch(
        targets, REPO_ROOT / "data" / "raw_tist",
        name_to_id=name_to_id, radius_km=35.0,
        window_start=WINDOW_START, window_end=WINDOW_END,
        verbose=False,
    )
    for c, r in res.items():
        logger.info(f"[{c}] carve OK: venues={r.n_venues_in_radius:,}  "
                       f"checkins_written={r.n_checkins_written:,}  "
                       f"users={r.n_users_distinct:,}")
        write_done(c, "carve",
                       {"n_venues": r.n_venues_in_radius,
                        "n_checkins": r.n_checkins_written,
                        "n_users_distinct": r.n_users_distinct,
                        "wallclock_s": r.wallclock_s,
                        "window": [WINDOW_START, WINDOW_END]})
        statuses[c] = "ok"
    logger.info(f"[carve] done in {time.time()-t0:.1f}s")
    return statuses


def stage_step01(city: str, args, logger: logging.Logger) -> str:
    raw = REPO_ROOT / "data" / "raw_tist" / f"{city}.tsv"
    out_dir = REPO_ROOT / "data" / "processed" / city
    if (out_dir / "metadata.json").exists() and not args.force:
        logger.info(f"[{city}] step01 already done — skipping")
        return "skipped"
    if not raw.exists():
        logger.error(f"[{city}] step01: missing raw TSV {raw}")
        return "failed"
    # Smoke mode: relax k_core so a small sample doesn't get fully pruned.
    k_core = 3 if args.smoke else KCORE
    logger.info(f"[{city}] step01 starting (kcore={k_core}) ...")
    t0 = time.time()
    try:
        res = preprocess_tsmc2014(
            raw_tsv=raw, out_dir=out_dir,
            taxonomy_path=TAXO_PATH, k_core=k_core, seed=args.seed,
        )
        write_done(city, "step01", {
            "n_users_final": int(res.n_users_final),
            "n_items_final": int(res.n_items_final),
            "n_interactions_post_kcore": int(res.n_interactions_post_kcore),
            "n_interactions_final": int(res.n_interactions_final),
            "wallclock_s": time.time() - t0,
        })
        logger.info(f"[{city}] step01 DONE in {time.time()-t0:.1f}s: "
                       f"users={res.n_users_final:,}, "
                       f"items={res.n_items_final:,}, "
                       f"int={res.n_interactions_final:,}")
        return "ok"
    except Exception as e:
        logger.exception(f"[{city}] step01 FAILED")
        write_fail(city, "step01", repr(e))
        return "failed"


def stage_backbones(city: str, args, logger: logging.Logger) -> str:
    """Floor FM + tuned B_full (via subprocess to existing CLIs)."""
    if is_done(city, "backbones") and not args.force:
        logger.info(f"[{city}] backbones already done — skipping")
        return "skipped"
    log_dir = LOG_DIR / city
    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logger.info(f"[{city}] backbones: floor FM ...")
    rc = _run_subprocess(
        [PYTHON, "-m", "experiments.run_baselines",
          "--city", city, "--models", "FM", "-v"],
        logger, desc="floor FM",
        log_file=log_dir / f"backbones_FM_{_now()}.log",
    )
    if rc != 0:
        logger.error(f"[{city}] floor FM failed rc={rc}")
        write_fail(city, "backbones", f"FM rc={rc}")
        return "failed"
    logger.info(f"[{city}] backbones: B_full tuning ...")
    rc = _run_subprocess(
        [PYTHON, "-m", "experiments.round2_tune_bfull",
          "--city", city, "--max-epochs", "12", "-v"],
        logger, desc="B_full",
        log_file=log_dir / f"backbones_Bfull_{_now()}.log",
    )
    if rc != 0:
        logger.error(f"[{city}] B_full tuning failed rc={rc}")
        write_fail(city, "backbones", f"Bfull rc={rc}")
        return "failed"
    write_done(city, "backbones", {"wallclock_s": time.time() - t0})
    logger.info(f"[{city}] backbones DONE in {time.time()-t0:.1f}s")
    return "ok"


def _run_xsage_stage(city: str, stage_letter: str,
                       args, logger: logging.Logger) -> int:
    log_dir = LOG_DIR / city
    log_dir.mkdir(parents=True, exist_ok=True)
    return _run_subprocess(
        [PYTHON, "-m", "experiments.run_xsage",
          "--city", city, "--stage", stage_letter, "-v"],
        logger, desc=f"X-SAGE Stage {stage_letter}",
        log_file=log_dir / f"stage{stage_letter}_{_now()}.log",
    )


def stage_xsage(city: str, stage_letter: str, args,
                  logger: logging.Logger) -> str:
    stage_key = f"stage{stage_letter}"
    if is_done(city, stage_key) and not args.force:
        logger.info(f"[{city}] {stage_key} already done — skipping")
        return "skipped"
    # Stages B and D require the floor FM scores (Stage A and C do not).
    # If backbones hasn't run for this city, fail fast with a clear message
    # rather than waiting on subprocess startup just to see FileNotFoundError.
    if stage_letter in ("B", "D"):
        fm_hp = (REPO_ROOT / "outputs" / city / "baselines"
                  / "FM.best_hp.json")
        if not fm_hp.exists():
            logger.error(
                f"[{city}] {stage_key}: missing {fm_hp}. "
                f"Stage {stage_letter} requires backbones — run "
                f"`--stages backbones,{stage_key}` first."
            )
            write_fail(city, stage_key, "missing FM.best_hp.json (run backbones)")
            return "failed"
    t0 = time.time()
    rc = _run_xsage_stage(city, stage_letter, args, logger)
    if rc != 0:
        logger.error(f"[{city}] {stage_key} failed rc={rc}")
        write_fail(city, stage_key, f"rc={rc}")
        return "failed"
    # Try to read the produced summary
    summary_path = (REPO_ROOT / "outputs" / city / "xsage" /
                       {"A": "situations/summary.json",
                        "B": "fairness/verdict.json",
                        "C": "projection/verdict.json",
                        "D": "three_way/three_way_summary.json"}[stage_letter])
    payload = {"wallclock_s": time.time() - t0}
    if summary_path.exists():
        try:
            payload["summary"] = json.loads(summary_path.read_text())
        except Exception:
            pass
    write_done(city, stage_key, payload)
    logger.info(f"[{city}] {stage_key} DONE in {time.time()-t0:.1f}s")
    # Sync the headline summary into outputs_multicity/<city>/
    if summary_path.exists():
        target = city_dir(city) / f"summary_{stage_key}.json"
        shutil.copy2(summary_path, target)
    return "ok"


# ----------------------------------------------------------------------------
# Smoke mode
# ----------------------------------------------------------------------------

def make_smoke_tsv(n_users: int, logger: logging.Logger) -> str:
    """Build a tiny smoke TSV from nyc_tist.tsv (smallest carved city)."""
    src = REPO_ROOT / "data" / "raw_tist" / "nyc_tist.tsv"
    if not src.exists():
        raise RuntimeError("nyc_tist.tsv missing — carve first")
    dst = REPO_ROOT / "data" / "raw_tist" / "_smoke.tsv"
    logger.info(f"[smoke] sub-sampling first {n_users} users from {src.name}")
    seen = {}
    out_lines = []
    with src.open() as f:
        for line in f:
            parts = line.split("\t", 1)
            uid = parts[0]
            if uid not in seen and len(seen) >= n_users:
                continue
            seen[uid] = True
            out_lines.append(line)
    dst.write_text("".join(out_lines))
    logger.info(f"[smoke] wrote {dst} ({dst.stat().st_size//1024} KB, "
                  f"{len(out_lines):,} rows, {len(seen)} users)")
    return "_smoke"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_stages(s: str) -> list[str]:
    if s == "all":
        return list(ALL_STAGES)
    out = [x.strip() for x in s.split(",") if x.strip()]
    for x in out:
        if x not in ALL_STAGES:
            raise SystemExit(f"unknown stage {x!r}; allowed: {ALL_STAGES}")
    return out


def parse_cities(s: str) -> list[str]:
    if s == "all":
        return list(CITIES.keys())
    out = []
    for c in (x.strip() for x in s.split(",") if x.strip()):
        if c not in CITIES:
            raise SystemExit(f"unknown city {c!r}; allowed: "
                                f"{list(CITIES.keys())}")
        out.append(c)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cities", default="all",
                          help=f"Comma list or 'all'. Choices: "
                               f"{list(CITIES)} (default: all)")
    parser.add_argument("--stages", default="all",
                          help=f"Comma list or 'all'. Choices: "
                               f"{list(ALL_STAGES)} (default: all)")
    parser.add_argument("--force", action="store_true",
                          help="Redo (city, stage) even if done-marker exists.")
    parser.add_argument("--smoke", action="store_true",
                          help="Tiny end-to-end dry run on a 200-user "
                               "subset of nyc_tist. Validates plumbing.")
    parser.add_argument("--smoke-users", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.smoke:
        run_tag = f"smoke_{run_tag}"
    logger = setup_logging(run_tag)

    if args.smoke:
        smoke_city = make_smoke_tsv(args.smoke_users, logger)
        # Register the smoke city in CITIES so the loop picks it up
        CITIES[smoke_city] = {"tist": "(smoke)", "tt_band": "n/a",
                                  "expected_users": args.smoke_users}
        cities = [smoke_city]
        stages = ["step01", "backbones", "stageA", "stageB", "stageC"]
    else:
        cities = parse_cities(args.cities)
        stages = parse_stages(args.stages)

    # ETA preview
    total = sum(estimate_eta_seconds(c, stages) for c in cities)
    logger.info(f"================================================")
    logger.info(f"Run plan: {len(cities)} city / {len(stages)} stages")
    logger.info(f"Cities: {cities}")
    logger.info(f"Stages: {stages}")
    logger.info(f"Rough ETA (whole run): ~{hms(total)}  "
                  f"(per-city rough breakdown follows)")
    for c in cities:
        logger.info(f"  [{c}]  ~{hms(estimate_eta_seconds(c, stages))}  "
                       f"(expected_users={CITIES[c].get('expected_users', '?')})")
    logger.info(f"--force: {args.force}  --smoke: {args.smoke}  "
                  f"seed: {args.seed}")
    # Up-front dependency check: stages B and D need backbones output.
    if ("stageB" in stages or "stageD" in stages) and "backbones" not in stages:
        missing = []
        for c in cities:
            hp = REPO_ROOT / "outputs" / c / "baselines" / "FM.best_hp.json"
            if not hp.exists():
                missing.append(c)
        if missing:
            logger.warning(
                f"!! Stage B/D requested without `backbones` AND no "
                f"floor-FM artefact yet for: {missing}. "
                f"Those stages will FAIL. Either add 'backbones' to "
                f"--stages, or run `--stages backbones` first."
            )
    logger.info(f"================================================")

    # Carve phase (global one-pass, only if requested AND not smoke)
    statuses: dict[str, dict[str, str]] = {c: {} for c in cities}
    if "carve" in stages and not args.smoke:
        s = stage_carve(cities, args.force, logger)
        for c in cities:
            statuses[c]["carve"] = s.get(c, "n/a")

    per_city_stages = [s for s in stages if s != "carve"]

    # Outer city bar
    for c in tqdm(cities, desc="cities", position=0, leave=True):
        # Inner stages bar
        for s in tqdm(per_city_stages,
                         desc=f"[{c}] stages", position=1, leave=False):
            if s == "step01":
                statuses[c][s] = stage_step01(c, args, logger)
            elif s == "backbones":
                statuses[c][s] = stage_backbones(c, args, logger)
            elif s in {"stageA", "stageB", "stageC", "stageD"}:
                letter = s[-1]
                statuses[c][s] = stage_xsage(c, letter, args, logger)
            else:
                statuses[c][s] = "unknown"

    # Final summary
    logger.info("\n=== FINAL STATUS ===")
    head = "city".ljust(14) + " ".join(s.ljust(10) for s in stages)
    logger.info(head)
    for c in cities:
        row = c.ljust(14) + " ".join(
            (statuses[c].get(s, "—") or "—").ljust(10) for s in stages)
        logger.info(row)

    n_fail = sum(1 for c in cities for s in stages
                    if statuses[c].get(s) == "failed")
    n_ok = sum(1 for c in cities for s in stages
                  if statuses[c].get(s) == "ok")
    n_skip = sum(1 for c in cities for s in stages
                    if statuses[c].get(s) == "skipped")
    logger.info(f"\nOK: {n_ok}  Skipped: {n_skip}  Failed: {n_fail}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
