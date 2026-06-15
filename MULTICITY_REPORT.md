# Multi-city generalisation — round 4

> Cross-city test of the round-3 X-SAGE method (C5.0 per-target-macro
> law, B6 backbone-agnostic lens, B7/B7b exposure trade-off, B8/B8b
> faithfulness) on the TIST2015 global Foursquare check-in dataset.

**Branch**: `round4-multicity`, off the tag `round3-complete` (the
defensible thesis checkpoint, `step02b-round3` @ `f2e8116`).
**Frozen state**: the round-3 work on `outputs/NYC/`, `outputs/TKY/`,
`pipeline/`, `engine/` is **untouched**. All multi-city work lives
under `data/raw_tist/`, `experiments/multicity/`, `outputs_multicity/`.
**Tests**: 19/19 green.

## PART 1 — lockdown + screening (this commit)

Done: tagged `round3-complete`, pushed to origin; branched
`round4-multicity` off the tag; created the separated workspace;
inventoried the TIST2015 raw files; ran a read-only 15-city screening
along the TT_share axis (the variable identified by round-3 C5.0/C6
as the governing axis for the law's sign flip).

### Schema (TIST2015)

* `dataset_TIST2015_Checkins.txt` — 33.3 M rows, 4 cols TSV:
  user_id, venue_id (Foursquare hex), UTC timestamp
  ("Tue Apr 03 18:00:06 +0000 2012"), timezone offset in minutes.
* `dataset_TIST2015_POIs.txt` — 3.68 M venues, 5 cols TSV: venue_id,
  lat, lon, **category name** (NOT a cat_id — different field from
  the legacy taxonomy mapping; required building a separate
  fine-name → macro lookup by walking the v2 taxonomy tree).
  *Note*: the readme says "7 columns" but the file has 5 — readme typo.
* `dataset_TIST2015_Cities.txt` — 415 cities, 6 cols TSV: city name,
  lat, lon, country code, country name, city type. **No city id**;
  venues are linked to cities by nearest-centre haversine within a
  35-km radius (no city overlap concern at metro spacing).
* Temporal span: **Apr 2012 – Sep 2013** (~18 months) — superset of
  the TSMC2014 window (Apr 2012 – Feb 2013, ~10 months).
* NYC (line 143) and Tokyo (line 384) both present.

### Screening table (15 cities, sorted by TT_share)

See `outputs_multicity/screening/CITY_SCREENING.md` for the full
table + per-row commentary. Headline:

|  TT_share band | cities |
|---|---|
| ~0.06 (very low T&T) | Kuala Lumpur (0.055), Istanbul (0.060), Jakarta (0.077), Mexico City (0.078) |
| ~0.10–0.13 | Bangkok (0.103), Moscow (0.106), Chicago (0.112), Los Angeles (0.125) |
| ~0.14–0.16 (NYC-band) | **NYC-TIST (0.141)**, Sao Paulo (0.150), Singapore (0.151) |
| ~0.19–0.24 | Seoul (0.193), Paris (0.213), London (0.236) |
| ~0.45 | **Tokyo-TIST (0.450)** |
| ≥ 0.60 | **none — empty bin** |

All 15 are above the `est_users_post_kcore ≥ 800` power floor.

### CRITICAL FINDING — TIST2015 ≠ TSMC2014 on the same city

The reason this stops at PART 1: re-extracting NYC and Tokyo from
TIST2015 produces materially different TT_share than the TSMC2014
round-3 anchors used.

| city | TSMC2014 round-3 anchor | TIST2015 raw (this screening) |
|---|---|---|
| NYC   | TT_share ≈ **0.25** | **0.141** |
| Tokyo | TT_share ≈ **0.71** | **0.450** |

This is a **data-collection-method effect** (TSMC built from a Twitter
stream filter; TIST is the broader Foursquare API stream), NOT a
temporal-window effect. The 18-month TIST span is a superset of the
10-month TSMC window, so trimming TIST cannot recover the TSMC
distribution.

**Three implications for Luca's decision:**

1. **The TT_share = 0.71 regime is unreachable from TIST.** Highest
   TIST shortlist TT_share is Tokyo at 0.45. To test the law's
   high-T&T end on a TIST city, we'd need a counterfactual upsample
   (inverse of C6's downsample).
2. **The "spread across 25/40/55/70" plan is no longer realistic.**
   The realistic TIST bands are 6–10 %, 11–16 %, 19–24 %, and the
   solitary 45 %.
3. **The compromise window path is now strongly preferred over the
   clean path** (re-extracting NYC/TKY from TIST would discover they
   are different cities, invalidating the round-3 anchors).

### Suggested 3-city PART-2 shortlist (compromise path)

| TT_share band | candidate | n_checkins | est_users_post_kcore | rationale |
|---|---|---|---|---|
| ~0.06 (very low T&T) | **Istanbul** | 2.98 M | 30 452 | biggest, richest, most powered city in TIST |
| ~0.15 (NYC-equivalent) | **Sao Paulo** | 0.81 M | 6 686 | non-Western metro at NYC TT_share — controls for region |
| ~0.45 (mid-high) | **Tokyo (TIST)** | 1.31 M | 9 588 | only TIST city with elevated T&T; reveals where the law starts to bite |

This spans the realistic TIST T&T spectrum end-to-end. All three are
well above the power floor. With the 10-month compromise clip
(Apr 2012 – Feb 2013), each retains > 500 k check-ins.

## What needs Luca's decision before PART 2

1. **City set**: confirm Istanbul + Sao Paulo + Tokyo-TIST, or swap one?
2. **Window policy**: confirm the **compromise** (keep TSMC NYC/TKY
   as round-3 anchors; clip new TIST cities to Apr 2012 – Feb 2013)
   — recommended given the screening finding.
3. **High-T&T regime**: do we run a counterfactual upsampling on one
   TIST city (e.g. lift Sao Paulo or Tokyo-TIST TT_share into 0.6–0.7
   by per-user non-T&T downsample) to recover a high-T&T anchor that
   isn't TSMC TKY? Optional but informative.
4. **Pre-registered predictions**: I'll draft them in PART 2 once the
   cities + window policy are fixed (R8 rule).

## State (this commit)

* `git tag round3-complete` → origin/round3-complete (pushed).
* `git branch round4-multicity` (off `round3-complete`) → origin/round4-multicity (pushed).
* `data/raw_tist/`, `outputs_multicity/`, `experiments/multicity/` created
  and added to `.gitignore` (per the brief's separation rule); only the
  screening script + CSV + MD reports are tracked.
* `outputs/NYC`, `outputs/TKY`, `pipeline/`, `engine/` — **untouched**.

## Files (PART 1)

* `experiments/multicity/round4_city_screening.py`
* `outputs_multicity/screening/city_screening.csv`
* `outputs_multicity/screening/CITY_SCREENING.md`

---

# PART 2 — pipeline built, ready to run (this commit)

PART 2 deliverable per the brief: **infrastructure built and validated by
a smoke test, NOT a full run** — Luca launches the real multi-hour run
from the VS Code terminal himself.

## Step 1 — city sizes confirmed (post-k-core actuals)

`experiments/multicity/multicity_size_check.py` carved each of the 5
candidate cities and ran the frozen `preprocess_tsmc2014` on each
(k-core=10, 80/10/10 temporal split, Apr 2012 – Feb 2013 compromise
window):

| city_key | TIST name | TT_share band | post-kcore users | items | interactions | flag |
|---|---|---|---:|---:|---:|---|
| `istanbul`   | Istanbul   | very-low (~0.06) | **22 631** | 9 305 | 1 258 889 | OK |
| `bangkok`    | Bangkok    | low (~0.10)      | **6 316**  | 5 185 | 407 128   | OK |
| `nyc_tist`   | New York · TSMC NYC bridge | low-mid (~0.14) | **4 113**  | 3 987 | 157 408   | OK |
| `saopaulo`   | Sao Paulo  | low-mid (~0.15)  | **4 395**  | 3 207 | 218 162   | OK |
| `tokyo_tist` | Tokyo · TSMC TKY bridge    | mid-high (~0.45) | **7 160**  | 5 723 | 563 089   | OK |

All 5 cities exceed the ~2 000-user power floor. No substitutions needed.

## Step 2 — pre-registered cross-city predictions (R8)

`outputs_multicity/PREDICTIONS_MULTICITY.md` is committed BEFORE any
Stage run. 9 falsifiable predictions P1–P9 with HIGH/MED confidence
tags:

* P1: ≥4 attractors on Istanbul / Bangkok / SP / NYC-TIST (HIGH each).
* P2: Stage-C ΔF1 ≥ +0.05 on the four low-TT cities (HIGH).
* P3: Tokyo-TIST Stage-C ΔF1 ∈ [−0.05, +0.05] (MED).
* P4: per-target-macro signs preserved on every city — T&T negative,
  non-T&T positive (HIGH, the structural backbone of round-3 C5.0/C6).
* P5: aggregate B_full−B_blind > 0 on the four low-TT cities (HIGH).
* P6: Tokyo-TIST aggregate B_full−B_blind near zero ([−0.01, +0.01]) (MED).
* P7: every city: ≥ 1 inequity sink at Stage B (HIGH).
* P8 / P9: provenance robustness — NYC-TIST behaviour predicted by
  ITS OWN TIST TT_share 0.14 (HIGH); Tokyo-TIST by its OWN 0.45 (MED).
  This is a natural causal test complementary to C6's synthetic one.

## Step 3 — orchestrator built

`experiments/multicity/run_multicity.py` (single command Luca runs).

Features:
* tqdm everywhere (outer city bar, inner stages bar);
* python `logging` to stdout AND
  `outputs_multicity/logs/run_<timestamp>.log`;
* checkpointing per (city, stage) with `.done_<stage>` markers under
  `outputs_multicity/<city>/`; re-run skips done unless `--force`;
* graceful failure: a failing (city, stage) writes `.fail_<stage>`
  and the loop CONTINUES to the next city;
* `--smoke`: 1 500-user dry run on `nyc_tist`, k-core=3, full chain in
  ~18 min (validated);
* upfront per-city ETA estimates printed at run start.

Stages: `carve, step01, backbones, stageA, stageB, stageC, stageD, all`.
All Stage modules are imported / called via subprocess of the frozen
round-3 CLIs (`experiments.run_baselines`, `experiments.round2_tune_bfull`,
`experiments.run_xsage`) — **no frozen code is forked**. The thin
multicity adapter is `experiments/multicity/tist_carve.py` (TIST→TSMC
format conversion + per-city haversine carving + 10-month window clip).

## Step 4 — smoke test PASSED

Smoke command actually executed during PART 2:

```bash
.venv/bin/python -m experiments.multicity.run_multicity --smoke
```

Result on `outputs_multicity/_smoke/`:

| stage | status | headline |
|---|---|---|
| step01    | OK | 1 444 users, 7 538 items, 100 429 interactions (kcore=3) |
| backbones | OK | floor FM ~4 min + B_full tuning ~13 min = ~17 min |
| Stage A   | OK | K=6, ε=0.01, ARI **0.998** cross-seed, 5 attractors |
| Stage B   | OK | **GREEN**, 1 sink at situation 1, KL ratio 1.9× global |
| Stage C   | OK | ΔF1 = **+0.206** (T-based), McNemar p = 0 |

Wallclock: ~18 min end-to-end on a 1 500-user subset of NYC-TIST.
**Plumbing validated**: every stage produced its expected artefacts,
checkpoint markers were written, tqdm/logging behaved.

The smoke result is consistent with PART-2 predictions for low-TT
cities (rich attractor structure, positive ΔF1, GREEN lens).

## The command Luca should run

From the VS Code integrated terminal:

```bash
cd /Users/lucaaliberti/Downloads/IntentAwareRS_thesis

# Smoke (re-validate plumbing locally, ~18 min):
.venv/bin/python -m experiments.multicity.run_multicity --smoke

# Real multi-city full run (5 cities × all stages, ~4–6 h on the M2):
.venv/bin/python -m experiments.multicity.run_multicity \
    --cities all --stages all

# Recommended FIRST pass — only structural stages (no backbones training),
# completes in ~15 min, runs Stage A + Stage C (Stage B needs backbones):
.venv/bin/python -m experiments.multicity.run_multicity \
    --cities all --stages step01,stageA,stageC

# Then add Stage B / D after running backbones (heavy step):
.venv/bin/python -m experiments.multicity.run_multicity \
    --cities all --stages backbones,stageB
```

Logs land in `outputs_multicity/logs/run_<timestamp>.log` (also tee'd
to stdout). To resume after sleep/crash: re-run the same command — the
orchestrator skips completed (city, stage) pairs automatically. To
inspect a specific failure: look for `.fail_<stage>` under
`outputs_multicity/<city>/`. Full runbook: `experiments/multicity/README.md`.

## State (this commit)

* `round4-multicity` branch advanced to PART-2 code; round3-complete
  tag unchanged.
* New files: `experiments/multicity/tist_carve.py`,
  `multicity_size_check.py`, `run_multicity.py`, `README.md`.
* Tracked artefacts (force-added past gitignore):
  `outputs_multicity/PREDICTIONS_MULTICITY.md`,
  `outputs_multicity/selection/final_cities.md`,
  `outputs_multicity/selection/size_check_results.json`,
  the smoke log under `outputs_multicity/logs/`.
* `outputs/NYC`, `outputs/TKY`, `outputs/TKY_BAL`, `pipeline/`,
  `engine/` — **untouched**; 19/19 tests green.
* Heavy per-city artefacts written by frozen modules during the smoke
  live under `outputs/_smoke/` and `data/processed/_smoke/` (both
  gitignored). Real-run per-city artefacts will go to
  `outputs/<city>/` / `data/processed/<city>/` (same convention as
  TSMC NYC/TKY/TKY_BAL).

## Step 5 (PART 3, deferred)

The synthetic ~70% high-T&T anchor (reverse-C6 upsampling) waits until
PART-2's real-city results confirm the law's behaviour. Planned but
not built.
* `MULTICITY_REPORT.md` (this file)
