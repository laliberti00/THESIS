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

## Files

* `experiments/multicity/round4_city_screening.py`
* `outputs_multicity/screening/city_screening.csv`
* `outputs_multicity/screening/CITY_SCREENING.md`
* `MULTICITY_REPORT.md` (this file)
