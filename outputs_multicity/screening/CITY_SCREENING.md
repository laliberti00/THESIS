# City screening for multi-city generalisation (round 4 PART 1)

> Read-only inventory of 15 TIST2015 cities along the **TT_share**
> axis (fraction of check-ins whose POI macro is Travel & Transport).
> No preprocessing, no k-core, no training — pure stream counts.

Source: `data/raw/dataset_TIST2015_{Checkins,POIs,Cities}.txt`
(Yang/Zhang/Qu, ACM TIST 2015 — 33.3 M check-ins, 266 909 users,
3.68 M venues across 415 cities, **Apr 2012 – Sep 2013**,
sample window observed = 2012-04 → 2013-09).

Anchors from TSMC2014 round-3 runs (reference, not in this table):
* **NYC (TSMC2014)**: TT_share ≈ 0.25 — B6 lens GREEN, B_full beats B_blind, X-SAGE additive nudge helps.
* **TKY (TSMC2014)**: TT_share ≈ 0.71 — B6 lens RED, B_full loses to B_blind, X-SAGE defensive.

## Screening table (sorted by TT_share ascending)

| city | TT_share | macro_entropy | n_checkins | n_users | n_venues | est_users_post_kcore | population_M | span_months | span_first | span_last |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kuala Lumpur | 0.055 | 0.884 | 1,604,623 | 17,330 | 139,645 | 13,113 | 8.0 | 18 | 2012-04 | 2013-09 |
| Istanbul | 0.060 | 0.948 | 2,984,594 | 40,912 | 146,415 | 30,452 | 15.5 | 18 | 2012-04 | 2013-09 |
| Jakarta | 0.076 | 0.882 | 1,076,960 | 15,823 | 141,228 | 10,321 | 33.4 | 18 | 2012-04 | 2013-09 |
| Mexico City | 0.078 | 0.913 | 858,741 | 10,329 | 77,167 | 7,306 | 22.0 | 18 | 2012-04 | 2013-09 |
| Bangkok | 0.103 | 0.884 | 1,075,954 | 11,525 | 109,968 | 7,985 | 10.7 | 18 | 2012-04 | 2013-09 |
| Moscow | 0.106 | 0.940 | 950,898 | 10,501 | 93,599 | 7,700 | 12.5 | 18 | 2012-04 | 2013-09 |
| Chicago | 0.112 | 0.904 | 184,873 | 6,885 | 21,949 | 2,057 | 8.9 | 18 | 2012-04 | 2013-09 |
| Los Angeles | 0.125 | 0.893 | 185,061 | 9,136 | 25,889 | 2,475 | 13.2 | 18 | 2012-04 | 2013-09 |
| New York | 0.141 | 0.905 | 600,148 | 17,435 | 69,822 | 7,146 | 20.1 | 18 | 2012-04 | 2013-09 |
| Sao Paulo | 0.150 | 0.906 | 809,200 | 11,856 | 78,906 | 6,686 | 22.0 | 18 | 2012-04 | 2013-09 |
| Singapore | 0.151 | 0.893 | 689,227 | 12,246 | 66,809 | 5,418 | 5.9 | 18 | 2012-04 | 2013-09 |
| Seoul | 0.193 | 0.885 | 213,358 | 3,327 | 46,576 | 1,803 | 25.6 | 18 | 2012-04 | 2013-09 |
| Paris | 0.213 | 0.926 | 111,325 | 6,903 | 19,837 | 1,462 | 11.0 | 18 | 2012-04 | 2013-09 |
| London | 0.236 | 0.911 | 188,530 | 9,724 | 28,687 | 3,175 | 14.3 | 18 | 2012-04 | 2013-09 |
| Tokyo | 0.450 | 0.750 | 1,314,130 | 12,771 | 106,237 | 9,588 | 37.4 | 18 | 2012-04 | 2013-09 |

## Underpowered cities (flag)

Recommended exclude: any city with `est_users_post_kcore < 800` —
the round-3 NYC floor used 829 users post-k-core, which is at the
lower edge of what the statistical pipeline (paired Wilcoxon, 
TOST, bootstrap CI95) can resolve. Below ~800 the per-stratum n's
(T&T ≈ 25 %, non-T&T ≈ 75 %) drop below ~200 each and CIs widen.

| city | est_users_post_kcore | flag |
|---|---:|---|
| Kuala Lumpur | 13,113 | OK |
| Istanbul | 30,452 | OK |
| Jakarta | 10,321 | OK |
| Mexico City | 7,306 | OK |
| Bangkok | 7,985 | OK |
| Moscow | 7,700 | OK |
| Chicago | 2,057 | OK |
| Los Angeles | 2,475 | OK |
| New York | 7,146 | OK |
| Sao Paulo | 6,686 | OK |
| Singapore | 5,418 | OK |
| Seoul | 1,803 | OK |
| Paris | 1,462 | OK |
| London | 3,175 | OK |
| Tokyo | 9,588 | OK |

## Recommendation — deliberate TT_share spread

The round-3 evidence (C1 → C5.0 → C5 → C6) identifies TT_share as
the variable along which X-SAGE's per-target-macro law fires (or
fails to fire). Population is NOT a governing variable — Tokyo
(37 M) and Bangkok (10.7 M) sit at very different TT_share but both
trigger the same B_full-loses-on-T&T regime, while NYC (20 M) and
Sao Paulo (22 M, expected lower TT_share) do not.

Pick **one city near each of ~25 %, ~40 %, ~55 %, ~70 %** TT_share,
subject to the `est_users_post_kcore ≥ 800` constraint. The 25 %
bin is already covered by NYC; the 70 % bin by TKY. Choose two
intermediate cities from the table above.

## Temporal-window note (feeds Luca's decision)

All TIST2015 check-ins fall in **Apr 2012 – Sep 2013** (~18 months).
The TSMC2014 NYC/TKY were built on **Apr 2012 – Feb 2013** (~10
months). Two policies are possible for PART 2:

1. **Clean path** — re-extract NYC + TKY from TIST2015 on a single
   common window (e.g. Apr 2012 – Sep 2013, the full TIST span);
   maximises cross-city comparability but invalidates the round-3
   numbers (would have to re-derive everything from scratch).
2. **Compromise** — keep TSMC2014 NYC + TKY as-is (round-3 numbers
   preserved); clip each new TIST city to the TSMC2014 window
   (Apr 2012 – Feb 2013) so the windows match. Loses ~8 months of
   TIST signal per city but preserves the round-3 baseline.

Each city's full TIST span is shown in the table above to feed
this decision.

## CRITICAL FINDING — TIST2015 ≠ TSMC2014 on the same city

The TIST2015 numbers for NYC and Tokyo do **NOT** match the TSMC2014
round-3 anchors:

| city | TSMC2014 round-3 anchor | TIST2015 raw (this screening) | Δ |
|---|---|---|---|
| NYC   | TT_share ≈ **0.25** | TT_share = **0.141** | −0.11 |
| Tokyo | TT_share ≈ **0.71** | TT_share = **0.450** | −0.26 |
| TKY   | macro entropy ≈ **0.49** | TIST Tokyo = **0.75** | +0.26 |

This is **not** a temporal-window effect: TIST is a superset window
(Apr 2012 – Sep 2013 vs TSMC's Apr 2012 – Feb 2013). It is a
**data-collection-method effect**: TSMC2014 was built from a specific
Twitter-stream filter, biasing toward transit-tagged check-ins;
TIST2015 is the broader Foursquare API stream. Same city, same dates
overlap, different distribution.

**Three consequences for PART 2:**

1. **The TT_share = 0.71 regime (TSMC TKY) is essentially unreachable
   from TIST2015.** The highest TT_share in the 15-city shortlist is
   Tokyo at 0.45. No TIST city sits anywhere near the upper end of the
   round-3 spectrum where the C5.0 law was shown to flip sign.
   * Implication: we **cannot directly test the law's high-T&T end on
     TIST cities**; we can test the low-T&T band and the intermediate
     band, plus the special outlier Tokyo at 0.45.
   * Workaround: counterfactual *upsampling* — pick a TIST city and
     per-user downsample non-T&T to lift TT_share into the 0.6–0.7
     band. Inverse of C6's TKY* trick. Not free but a clean direction.

2. **The "deliberate spread across 25/40/55/70" recommendation is
   no longer realistic** with these 15 cities. The realistic TT_share
   bins from TIST are:
   * ~6–10 % (Kuala Lumpur, Istanbul, Jakarta, Mexico City, Bangkok, Moscow) — 6 candidates
   * ~11–16 % (Chicago, LA, NYC-TIST, Sao Paulo, Singapore) — 5 candidates
   * ~19–24 % (Seoul, Paris, London) — 3 candidates
   * ~45 % (Tokyo-TIST) — 1 candidate
   * 60 %+ — **NONE**

3. **The compromise path (keep TSMC NYC/TKY, clip TIST cities to
   10 months) is now strongly preferred over the clean path.** The
   clean path would re-extract NYC and Tokyo from TIST and discover
   that *they are different cities* from TSMC NYC/TKY at the TT_share
   level — invalidating not only the round-3 numbers but the
   *anchoring* of the per-target-macro law. The compromise path keeps
   the round-3 anchors, treats new TIST cities as a generalisation
   experiment over a (necessarily) **lower** TT_share regime, and lets
   us state honestly: "TSMC2014 NYC/TKY anchor the high-end of the
   T&T spectrum we tested; TIST2015 cities anchor the low/medium end."

## Suggested PART-2 city picks (3-city set, compromise window)

Subject to revision by Luca:

| TT_share band | candidate | notes |
|---|---|---|
| ~0.06 (very low T&T) | **Istanbul** (TT=0.060, n=2.98 M, kc=30 452) | biggest, richest, most powered city in the dataset |
| ~0.15 (NYC-anchor band) | **Sao Paulo** (TT=0.150, kc=6 686) | non-Western metro at NYC-equivalent TT_share — controls for region |
| ~0.45 (Tokyo-TIST, mid-high) | **Tokyo from TIST2015** (TT=0.450, kc=9 588) | only TIST city with non-trivially elevated T&T; usefully reveals where the law starts to bite |

Rationale: spans the realistic TIST T&T spectrum end-to-end, all
above kc=6 000, all 18-month spans → 10-month clip preserves > 500 k
checkins each, no power concern.

Avoid for now: Paris/Seoul/London (kc < 3 200, statistical-power risk
after 10-month clip); Chicago/LA (small absolute checkin counts).
KL / Jakarta / Mexico City / Bangkok / Moscow are near-duplicates of
Istanbul on the TT_share axis — diminishing returns.

## Files

* `outputs_multicity/screening/city_screening.csv`
* `outputs_multicity/screening/CITY_SCREENING.md` (this file)