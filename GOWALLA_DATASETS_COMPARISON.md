# GOWALLA_DATASETS_COMPARISON.md

Fact-only report on the two Gowalla raw sources available in `data/GowallaRaw/`,
produced by [`scripts/inspect_gowalla_raw.py`](scripts/inspect_gowalla_raw.py)
on 2026-05-27 (Apple M2, Python 3.11, numpy 1.26, pandas 2.1).

This document **does not make decisions**. It only describes what is in each
file, what the two raw sources share and where they differ, and how they relate
to Shehzad's pre-split `.pkl` already in use in Phase 1-2.

The full JSON dump that underlies the numbers below is at
[`scripts/_inspect_output/inspection.json`](scripts/_inspect_output/inspection.json)
(28.8 KiB).

---

## 0. Sources inventoried

| Source | Path | Size on disk |
| --- | --- | --- |
| SNAP raw check-ins | `data/GowallaRaw/loc-gowalla_totalCheckins.txt` | 395 MB |
| Liu — check-ins | `data/GowallaRaw/22126586/gowalla_checkins.csv` | 1295 MB |
| Liu — spots subset 1 | `data/GowallaRaw/22126586/gowalla_spots_subset1.csv` | 343 MB |
| Liu — spots subset 2 | `data/GowallaRaw/22126586/gowalla_spots_subset2.csv` | 8 MB |
| Liu — friendship | `data/GowallaRaw/22126586/gowalla_friendship.csv` | 67 MB |
| Liu — user info | `data/GowallaRaw/22126586/gowalla_userinfo.csv` | 17 MB |
| Liu — category taxonomy | `data/GowallaRaw/22126586/gowalla_category_structure.json` | 99 KiB |
| Shehzad reference (already split) | `data/DCCF/gowalla/{train,test,*_index}.pkl` | 31 MB |

---

## 1. SNAP — `loc-gowalla_totalCheckins.txt`

Cho-Myers-Leskovec 2011, distributed by SNAP. TSV without header.

### Schema

| col | dtype | example |
| --- | --- | --- |
| `user_id` | int32 | `0` |
| `time` | datetime, UTC, ISO 8601 with Z | `2010-10-19T23:55:27Z` |
| `lat` | float64 | `30.235909` |
| `lon` | float64 | `-97.795140` |
| `location_id` | int64 | `22847` |

First 3 rows:
```
0   2010-10-19T23:55:27Z   30.2359091   -97.7951396   22847
0   2010-10-18T22:17:43Z   30.2691030   -97.7493954   420315
0   2010-10-17T23:42:03Z   30.2557310   -97.7633858   316637
```

### Scale

- **6,442,892** total rows
- **107,092** distinct `user_id`
- **1,280,969** distinct `location_id`
- Time range: **2009-02-04 05:17:38 UTC** → **2010-10-23 05:22:06 UTC** (≈ 21 months)

### Quality

| issue | count | % of total |
| --- | ---: | ---: |
| rows with any NaN | 0 | 0% |
| rows with `(lat == 0) & (lon == 0)` | 135 | 0.002% |
| rows with `lat` out of `[-90, 90]` | 0 | 0% |
| rows with `lon` out of `[-180, 180]` | 0 | 0% |
| exact-duplicate rows | 601 | 0.009% |

### Distributions

**Check-ins per user** (n=107,092 users):

| stat | value |
| --- | ---: |
| min | 1 |
| p05 | 1 |
| p25 | 7 |
| **p50 (median)** | **25** |
| p75 | 56 |
| p95 | 225 |
| max | 2,175 |
| mean | 60.2 |
| std | 136.2 |

**Check-ins per venue** (n=1,280,969 venues):

| stat | value |
| --- | ---: |
| p25 | 1 |
| **p50 (median)** | **2** |
| p75 | 4 |
| p95 | 18 |

**Monthly volume** (selected, full series in JSON):

| month | check-ins |
| --- | ---: |
| 2010-01 | 311,180 |
| 2010-04 | 549,079 |
| 2010-07 | 707,685 |
| 2010-08 | 861,805 |
| 2010-09 | 973,801 |
| 2010-10 | 742,473 |

The time series ramps up monotonically from 2009-02 to a peak in 2010-09 (974 k
check-ins / month) and then SNAP simply stops at 2010-10-23.

---

## 2. Liu — `gowalla_checkins.csv`

CSV with header. No spatial coordinates — those are in the spots files (§3).

### Schema

| col | dtype | example |
| --- | --- | --- |
| `userid` | int32 | `1338` |
| `placeid` | int64 | `482954` |
| `datetime` | datetime, UTC, ISO 8601 with Z | `2011-06-23T02:24:22Z` |

### Scale

- **36,001,959** total rows (5.59× SNAP)
- **319,063** distinct `userid` (2.98× SNAP)
- **2,844,145** distinct `placeid` (2.22× SNAP venue count)
- Time range: **2009-01-21 16:40:55 UTC** → **2011-08-16 18:54:25 UTC** (≈ 32 months)

### Quality

| issue | count | % of total |
| --- | ---: | ---: |
| rows with any NaN | 0 | 0% |
| exact-duplicate rows | 43,599 | 0.121% |

### Distributions

**Check-ins per user** (n=319,063 users):

| stat | value |
| --- | ---: |
| min | 1 |
| p05 | 1 |
| p25 | 5 |
| **p50 (median)** | **21** |
| p75 | 78 |
| p95 | 443 |
| max | 46,981 |
| mean | 112.8 |
| std | 489.9 |

**Check-ins per place** (n=2,844,145 places):

| stat | value |
| --- | ---: |
| **p50 (median)** | **4** |
| p75 | 12 |
| p95 | 87 |

**Monthly volume** (Liu tail of the time series):

| month | check-ins |
| --- | ---: |
| 2010-09 | (in series, full JSON has it) |
| 2010-11 | 2,155,673 |
| 2010-12 | 2,490,958 |
| 2011-01 | 2,147,037 |
| 2011-02 | 1,938,766 |
| 2011-03 | 2,494,648 |
| 2011-04 | 2,358,150 |
| 2011-05 | 2,232,528 |
| 2011-06 | 1,695,357 |
| 2011-07 | 37,622 |
| 2011-08 | 1,320 |

Liu is ~2-3× denser than SNAP in the months they overlap (e.g. SNAP 2010-09 ≈
974 k vs Liu ~~ ~2-3 M for the same month), and extends ~10 months past SNAP's
cutoff (2010-11 onward). Volume collapses sharply in 2011-07/08, suggesting Gowalla
was shutting down or scraping ended.

---

## 3. Liu spots files

### 3.1 `gowalla_spots_subset1.csv`

CSV with header.

| col | dtype | example |
| --- | --- | --- |
| `id` | int64 | `8904` |
| `created_at` | datetime | `2008-12-06T16:28:53Z` |
| `lng` | float64 | `-94.6074986` |
| `lat` | float64 | `39.0523183` |
| `photos_count` | int32 | `0` |
| `checkins_count` | int32 | `114` |
| `users_count` | int32 | `21` |
| `radius_meters` | float32 | `35` |
| `highlights_count` | int32 | `0` |
| `items_count` | int32 | `10` |
| `max_items_count` | int32 | `10` |
| `spot_categories` | string (Python list of dicts) | `"[{'url': '/categories/89', 'name': 'Craftsman'}]"` |

Schema: this is the **rich** spot table.

**Scale**: 2,724,891 distinct venues. **All 2,724,891 have at least one category
assigned** — 100% category coverage. **630 distinct category names** appear in
the parsed `spot_categories` column.

**Quality**: 0 NaN, 0 zero-zero coords, 0 out-of-range lat/lon, 0 exact duplicates.

**Top 20 categories by frequency**:

| count | category |
| ---: | --- |
| 99,815 | Gas & Automotive |
| 89,727 | Corporate Office |
| 88,596 | Asian |
| 66,892 | Other - Food |
| 64,411 | Apartment |
| 62,902 | Coffee Shop |
| 54,629 | Grocery |
| 51,760 | Other - Shopping |
| 50,707 | Craftsman |
| 45,594 | American |
| 39,591 | Bar |
| 38,819 | Pizza |
| 37,607 | Modern |
| 37,126 | Salon & Barbershop |
| 36,454 | Bank & Financial |
| 35,707 | Mexican |
| 35,544 | Italian |
| 34,333 | Other - Services |
| 31,193 | Pub |
| 30,927 | Modern Hotel |

### 3.2 `gowalla_spots_subset2.csv`

CSV with header. Encoding is **latin-1**, not UTF-8 (some bytes invalid in UTF-8).

| col | dtype | example |
| --- | --- | --- |
| `id` | int | `9813` |
| `lat` | float | `36.6315079` |
| `lng` | float | `-121.9120216` |
| `name` | string | `Pacific Ocean` |
| `city_state` | string | `"Seaside, CA"` |
| (two trailing unnamed columns) | empty | — |

**Scale**: 120,997 rows / **120,958 distinct ids** (39 duplicate rows).

**No category column.** No created_at, photos_count, radius_meters etc.
Subset2 looks like a thinner sidecar — name + city only.

### 3.3 subset1 vs subset2 — overlap analysis ⚠️

The two files share the same `id` namespace, but:

| relation | count |
| --- | ---: |
| `subset1` ∩ `subset2` | **0** |
| only in subset1 | 2,724,891 |
| only in subset2 | 120,958 |
| union | **2,845,849** |

**They are completely disjoint sets** of venue ids. Not a partition either (it
would still be disjoint, just with different cardinalities). The two files are
two independent fragments of the spots universe. Together they form the
"complete" Liu spots catalogue — and that catalogue contains
**2,845,849 distinct venues with coordinates**.

When joined against `gowalla_checkins.csv`, the placeid-to-coordinates lookup
covers **36,001,678 / 36,001,959 check-ins = 100.0% (rounded)** of all Liu
check-ins. Only 281 check-ins have a `placeid` not present in either subset.

### 3.4 `gowalla_category_structure.json`

JSON. Root key `spot_categories`, where each node may recursively contain its
own `spot_categories` child list.

Structure:
- **4 levels** of depth
- **269 total category nodes**
- Level 0 (top): **7** categories
- Level 1 (mid): 134 categories
- Level 2 (sub): 128 categories
- Level 3: 0 categories (level 3 only exists as the empty terminal of recursion)

**Top-level (depth 0) categories**, in order:
1. **Community**
2. **Entertainment**
3. **Food**
4. **Nightlife**
5. **Outdoors**
6. **Shopping**
7. **Travel**

Note that the taxonomy has 269 nodes but `spot_categories` strings in
`subset1.csv` reference 630 distinct names — there are more names "in the wild"
than there are nodes in the canonical taxonomy. This suggests either (a) the JSON
is an abridged version, or (b) some category names in the CSV are aliases /
variations.

---

## 4. Other Liu files (auxiliary)

### 4.1 `gowalla_friendship.csv`

| col | dtype | example |
| --- | --- | --- |
| `userid1` | int32 | `1` |
| `userid2` | int32 | `63488` |

Social graph: 4,418,339 undirected edges (most likely each friendship appears
both as `(a,b)` and `(b,a)`), involving 407,533 distinct users.

407,533 users in friendship vs 319,063 in `gowalla_checkins.csv` — there are
**88,470 users with at least one friend but zero check-ins** in Liu. Symmetric:
some users with check-ins may have no friends.

### 4.2 `gowalla_userinfo.csv`

CSV with header, 407,533 rows × 16 columns. Per-user aggregate stats
(`bookmarked_spots_count, challenge_pin_count, country_pin_count,
highlights_count, items_count, photos_count, pins_count, province_pin_count,
region_pin_count, state_pin_count, trips_count, friends_count, stamps_count,
checkin_num, places_num`).

Useful as a sanity check (Liu's `checkin_num` field should approximately match
what we compute from `gowalla_checkins.csv` per user) but contains no
interaction data on its own.

---

## 5. Shehzad reference (already in the repo)

For comparison, the `.pkl` files in `data/DCCF/gowalla/` (already discussed in
[`DATA_INVENTORY.md`](DATA_INVENTORY.md)) hold:

- **50,821 users × 57,440 items**
- 1,172,425 train interactions + 130,270 test interactions = **1,302,695** total

This is the version we've been using in Phase 1-2. Shehzad obtained it from a
DCCF pre-processing pipeline that is **not public** (no script in the repo).
Both Shehzad and DCCF cite "10-core filtered Gowalla" but the specific raw
source (SNAP, Liu, or otherwise) is not made explicit.

---

## 6. SNAP ↔ Liu comparison

### 6.1 Pure scale

| metric | SNAP | Liu | ratio Liu/SNAP |
| --- | ---: | ---: | ---: |
| check-ins | 6,442,892 | 36,001,959 | 5.59× |
| distinct users | 107,092 | 319,063 | 2.98× |
| distinct venues | 1,280,969 | 2,844,145 | 2.22× |
| temporal span | 21 months | 32 months | — |
| temporal end | 2010-10-23 | 2011-08-16 | Liu +10 mo |
| temporal start | 2009-02-04 | 2009-01-21 | Liu earlier by 14 d |

### 6.2 ID-system comparability

Numerical ranges and integer set intersection:

| field | SNAP range | Liu range | set intersection |
| --- | --- | --- | --- |
| user_id | `[0, 196,585]` | `[1, 2,688,969]` | **43,230** users in both |
| venue/place_id | `[8904, 5,977,757]` | `[8904, 7,715,122]` | **1,246,947** venues in both |

The ids appear to be **the same identifier convention** (both venue ids start at
the same minimum 8904 — too specific to be a coincidence). 1,246,947 / 1,280,969
= **97.3%** of SNAP venues are also present in Liu as the same numerical id.
43,230 / 107,092 = **40.4%** of SNAP users are also present in Liu as the same
numerical id.

### 6.3 Temporal overlap

- Common window: **2009-02-04 05:17:38 → 2010-10-23 05:22:06** = **626 days**
- SNAP rows entirely inside the overlap window: **6,442,892** (100%, since SNAP
  ends inside the overlap)
- Liu rows inside the overlap window with resolvable coords: **17,863,397**
  (i.e. Liu has ~2.77× more check-ins than SNAP for the same period)

### 6.4 Sample-match test ⚠️

**Procedure** — for 1000 SNAP rows sampled with seed 42 from inside the temporal
overlap, search for at least one Liu row matching on:

- `|timestamp_SNAP − timestamp_Liu|` ≤ **10 minutes**
- `|lat_SNAP − lat_Liu|` ≤ **0.001°** (≈ 100 m)
- `|lng_SNAP − lng_Liu|` ≤ **0.001°** (≈ 100 m at 30 °N)

Liu rows used for lookup: 17,863,397 (Liu in the overlap window, with coords
joined from spots1+spots2).

**Result: 0 / 1000 matched. 0.0%.**

Two example non-matches:

```
snap_user=73655  snap_time=2010-02-12 17:12:33  lat= 32.78448  lng= -96.79680
snap_user=3888   snap_time=2010-01-14 19:29:02  lat= 37.80050  lng=-122.40746
```

Despite the temporal overlap of 626 days, 97.3% venue-id intersection, and
17.8 M Liu candidate rows in the window, the strict sample-match recovers zero
matches at the (±10 min, ±100 m) tolerance.

### 6.5 What this means, factually (no interpretation)

The numerical IDs are shared (same convention). The temporal windows overlap
substantially. The venue universes overlap heavily (97.3%). But the
**individual interaction records** — that is, "user X was at venue Y at time T
with coords (lat, lng)" — do **not** match between the two sources at narrow
spatiotemporal tolerance.

Possible factual explanations (do not require a decision now):
- The two sources sample different fractions of Gowalla's check-in firehose
  (e.g. SNAP took an export at one date, Liu at another), each missing rows the
  other has.
- Timestamps may carry different conventions (e.g. one is in UTC, the other is
  in local time at the time of check-in) — but both sources advertise ISO 8601
  with `Z` suffix, which means UTC. Worth verifying.
- Coordinate precision may differ: SNAP coords have 10 decimal places (excessive
  precision); Liu spots coords have 7. At ±0.001° this should not exclude
  matches, but at ±0.0001° it could.
- Rows may have been deduplicated or aggregated in one source but not the
  other.

The sample-match was deliberately strict; relaxing to ±1 hour and ±0.01° (1 km)
is one obvious follow-up if the question becomes important. As-is, the strict
result shows the two sources are **not** straightforwardly "the same data, one
just bigger".

---

## 7. Quick comparison table — facts to keep in mind

| Question | SNAP | Liu | Notes |
| --- | --- | --- | --- |
| Has lat/lon on check-ins? | yes (in main file) | no — must join with spots | extra join step for Liu |
| Has category? | no | yes (subset1 covers all venues) | Liu is richer |
| Has friendship graph? | no | yes (4.4 M edges) | Liu only |
| Has per-user aggregates? | derivable | provided directly | Liu only |
| Time span | 21 months | 32 months | Liu has more recent data |
| Check-in volume | 6.4 M | 36 M | Liu 5.6× |
| Venue catalogue | 1.28 M | 2.84 M | Liu 2.2× |
| Encoding | UTF-8 clean | mixed (subset2 latin-1) | Liu needs encoding care |
| Has a pre-cooked split? | no | no | both raw |
| Used in Shehzad's `.pkl`? | not made explicit | not made explicit | mystery |

---

## 8. Where the numbers came from

- Inspection script: [`scripts/inspect_gowalla_raw.py`](scripts/inspect_gowalla_raw.py)
- Full JSON dump: [`scripts/_inspect_output/inspection.json`](scripts/_inspect_output/inspection.json)
- Console log captured at: `/tmp/inspect_gowalla.log` (not committed, regeneratable by re-running the script)
- Hardware: MacBook Air M2, 16 GB RAM, Python 3.11.0
- Wall-clock: ~7 minutes for the full inspection
- Seed used in sample-match: 42 (fixed in the script)

The script is **read-only**: it never writes inside `data/`, only inside
`scripts/_inspect_output/`. Re-running it is idempotent and produces the same
numbers (modulo wall-clock-dependent `time` reports).
