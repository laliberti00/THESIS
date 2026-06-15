# Final candidate sizes (round-4 PART 2, Step 1)

Window: **2012-04 → 2013-02** (TSMC2014-compatible, 10-mo clip of TIST2015's 18-mo span).
k-core: **10** on user × venue bipartite (iterative).

| city_key | TIST name | TT_share band | carved checkins | post-k-core users | post-k-core items | post-k-core interactions | flag |
|---|---|---|---:|---:|---:|---:|---|
| `istanbul` | Istanbul | very-low (~0.06) | 1,787,843 | 22,631 | 9,305 | 1,258,889 | **OK** |
| `bangkok` | Bangkok | low (~0.10) | 774,319 | 6,316 | 5,185 | 407,128 | **OK** |
| `nyc_tist` | New York · TSMC NYC (TT~0.25) | low-mid (~0.14) | 446,626 | 4,113 | 3,987 | 157,408 | **OK** |
| `saopaulo` | Sao Paulo | low-mid (~0.15) | 602,054 | 4,395 | 3,207 | 218,162 | **OK** |
| `tokyo_tist` | Tokyo · TSMC TKY (TT~0.71) | mid-high (~0.45) | 918,120 | 7,160 | 5,723 | 563,089 | **OK** |

## Round-3 bridge cities (provenance probe)

`nyc_tist` and `tokyo_tist` are the **same real cities** as the
TSMC2014 round-3 anchors (NYC, TKY), seen through a different
collection pipeline. They serve **two distinct roles** that must
be kept clearly separate in all subsequent outputs:

1. **Spectrum point** — they are simply two of the TIST cities
   along the TT_share axis (NYC-TIST at ~0.14, Tokyo-TIST at
   ~0.45). Treat them as any other TIST city in the cross-city
   analysis.
2. **Provenance probe** — comparing `nyc_tist` vs TSMC `NYC` is
   NOT a city comparison; it is a *same-city, different-dataset*
   comparison. Any differences reflect collection-pipeline
   sensitivity, NOT method instability. Same for `tokyo_tist`
   vs TSMC `TKY`.

**Never merge** TSMC NYC + TIST NYC (or TSMC TKY + TIST Tokyo)
as if they were one dataset.

## Acceptance

Drop any city with post-k-core users < ~2 000 (the round-3 NYC
floor had 829; we set a slightly higher bar here to ensure
per-stratum CIs from B9.1 are credible). Cities flagged OK above
proceed to PART-2 Steps 2-4.