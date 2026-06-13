# Dataset candidates for the third city (round 3 D1)

> Memo prepared as part of round-3 session 3. Selection rule, screening
> procedure, and pre-registered prediction (D2) recorded here.

## 1. Foursquare TIST2015 — **primary candidate**

* **Citation**: Yang, D., Zhang, D., Qu, B. (2014). "Participatory Cultural
  Mapping based on Collective Behavior Data in Location Based Social
  Networks." *ACM Transactions on Intelligent Systems and Technology*.
  Also referenced as the "Foursquare global-scale check-in dataset".
* **Official page**: <https://sites.google.com/site/yangdingqi/home/foursquare-dataset>
  (Section "Global-scale Check-in Dataset"). Mirrors: Figshare for the
  WWW'19 extended version (22 months + raw 90 M check-ins), Kaggle.
* **Scale**: ~33.3 M check-ins · 266 909 users · 3.68 M venues · 415 cities
  / 77 countries · period Apr 2012 – Sep 2013 · ~775 MB ZIP. The full
  release is "dataset_TIST2015_*.tsv" (Checkins, POIs, Cities).
* **Schema match against TSMC2014**:
  | aspect | TSMC2014 | TIST2015 | delta |
  |---|---|---|---|
  | per-checkin row | (user, venue, cat_id, lat, lon, tz, utc_time) | (user, venue, utc_time, tz_offset) — categories live in POI file | join needed |
  | venue → city | NYC/TKY pre-filtered | cities.txt provides venue→city assignment by coordinates | apply mapping |
  | category | Foursquare v2 IDs at level-3 | arbitrary hierarchy level (~84 % level-2) | extend fine→macro map to cover level-2 IDs |
  | timezone | tz column | tz_offset column | identical handling |
* **Adaptation effort**: ~half a day. Re-use `config/foursquare_legacy_
  taxonomy.json`; extend the `cat_id → macro` map with the level-2 codes
  that appear in TIST2015 but not TSMC2014.

### Shortlist of TIST2015 cities for screening

The selection rule (see §2) wants a city with **interior T&T share**
(~40-55 %), deliberately between NYC's 24.6 % and TKY's 73.2 %, and a
post-k-core count in 1k-3k users. The Section-2 screening table is
computed once we have the raw extract on disk; the shortlist below is
the reasonable candidate pool, ordered by *a priori* expected mobility
culture distance from NYC and TKY:

| city | rationale (a priori) |
|---|---|
| **Istanbul** | high transit (metro, ferries) but heavier coffee-house culture than Tokyo; expected interior T&T share |
| **São Paulo** | dense bus network + nightlife / food culture |
| **Bangkok** | strong street-food + transit mix; tourist confound |
| **Moscow** | metro-dense + strong shopping pattern |
| **Kuala Lumpur** | KL transit + restaurants |
| **Jakarta** | dense traffic, mixed mobility |

## 2. City screening procedure (the selection rule)

For each shortlist city, compute from raw (before any preprocessing):

1. **T&T check-in share**: count of check-ins whose POI cat_id maps to
   the Travel & Transport macro / total check-ins.
2. **Normalized macro entropy**: H(macro distribution) / log2(K_macro).
3. **Estimated post-k-core user count**: from per-user, per-venue
   in-degree histograms, conservatively estimate users surviving k-core
   = 10. (Pre-actual run; reasonably close in practice.)

Selection: pick a city with **T&T ∈ [40 %, 55 %]** AND post-k-core
users ∈ [1k, 3k]. If two candidates meet both: prefer the one with the
larger post-k-core user count to maximise statistical power.

Output of this step: `outputs/round3/D1/screening.csv` once the raw
extract is available.

## 3. Gowalla post-mortem

The `gowalla-dataset-comparison` branch (`e5167b2`) carried an audit of
Gowalla. Reading from `outputs/.../GOWALLA_DATASETS_COMPARISON.md` (the
inventory document on that branch) and the `inspect_gowalla_raw.py`
script:

* **Why it was parked**: Gowalla check-ins come WITHOUT category tags in
  the public release. The X-SAGE framework requires `cat_macro` for the
  intent proxy AND for the situation-machinery macro graph (W's nodes).
  Without categories there is no W, no attractors, no situation in our
  sense. The fix would require a venue→category enrichment from an
  external source (Foursquare API by venue id, OpenStreetMap by
  coordinates) — at the time of the audit, the API access was not
  available and the coordinate-based fallback gave ~30 % coverage.
* **Is the blocker fixable?** Partially. With OpenStreetMap's NaPTAN /
  amenity / leisure tags, a coordinate-based join would give a higher
  coverage today than the audit numbers reported. But the result would
  introduce an OSM-categorization dependency that is heterogeneous
  across regions (Europe well-tagged, Asia patchy). Not recommended as
  the *primary* third-city choice. Reasonable as a future cross-domain
  test if a high-coverage Foursquare-API enrichment becomes available.

## 4. Weeplaces

Foursquare-derived categories already, decent size. Historical
availability problem: most mirrors went offline after 2017. As of
session-3 writing, no stable source has been verified. The dataset is
RETAINED as a candidate only if a stable mirror (Internet Archive, a
re-publication by a research group) is identified before D2.

## 5. Last.fm-1K — non-POI generality candidate (future round)

* **Citation**: Celma, Ò. (2010). Music Recommendation and Discovery.
  Section 4.6 lists Last.fm-1K processing.
* **Official page**:
  <https://ocelma.net/MusicRecommendationDataset/lastfm-1K.html>.
  Zenodo record 6090214 mirrors the original release; GitHub parquet
  conversions exist.
* **Scale**: ~19 M timestamped listening events, 992 users, 2005-2009.
* **Adaptation effort**: ~1 day. The macro-category for music = genre;
  artist→genre comes from MusicBrainz. The intent proxy = previous-track
  genre. The protocol per-request next-item is directly applicable.
* **Why retained, not primary now**: it is the non-POI generality
  candidate. Including it in the paper as a "the framework transfers
  outside POI" experiment is a STRONG cross-domain claim if it works,
  but it doubles the engineering work this round. Defer to a future
  round; flag it in the paper as work-in-progress.
* **Note: LFM-360K is NOT suitable** — that release is aggregated (user,
  artist, count) without timestamps; no sequences, no intent proxy.

## 6. Frappe — REJECTED as third city (retained for trustworthiness-only transfer)

* **Citation**: Baltrunas, L., Church, K., Karatzoglou, A., Oliver, N.
  (2015). "Frappe: Understanding the Usage and Perception of Mobile App
  Recommendations In-The-Wild." Processed split: 651 users / 1 127
  items / 84 373 interactions.
* **Why rejected**: rich context one-hots (daytime, weather, weekend,
  home/work) but **no event-level timestamps** — the public release is
  aggregated (user × item × context, `cnt`). Consequences:
  * **No per-user sequences** → no intent proxy, no recency profile m,
    no transition matrix W, no attractors, no L3 projection.
  * **No temporal split** — only random split is possible.
* **Retained for**: a future trustworthiness-only domain transfer (just
  the situation = rough k-means on c̃ from the context one-hots, plus
  the lens + sink re-ranking) on a canonical CARS benchmark. The
  missing-intent limitation must be declared upfront in any such study.
  Requires re-joining the app-category metadata (which is in the
  separate `meta.csv` companion file).

## 7. Recommendation

**Primary**: TIST2015 — Istanbul or São Paulo, whichever meets the
selection rule's screening criteria when run.

**Secondary** (future round): Last.fm-1K, framed as the non-POI generality
test. Frappe and Weeplaces are explicitly NOT for this round.

Gowalla is parked, with a "may be revisited if OSM coverage improves".
