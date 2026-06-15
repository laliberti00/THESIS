# Math walkthrough — operationalisation of the X-SAGE framework

> **Purpose.** Source-of-truth extraction of every mathematical primitive
> as implemented, organised against the two-layer framework figure
> (top: SENSING → PERCEPTION → COMPREHENSION → PROJECTION; bottom:
> Data Preparation → Data Representation → Recommendation → Evaluation,
> the Trustworthy principles). For each level we report (a) the source
> file(s)/function(s), (b) the exact operation in notation, (c)
> inputs/outputs with dimensions, (d) parameters and how they are set,
> (e) the trustworthiness measure adopted to operationalise that level,
> (f) any discrepancy with the verbal description used so far.
>
> Status: READ-ONLY documentation, no code changes. Reads the round-3
> frozen state on branch `step02b-round3` (≡ `pipeline/` content of
> branch `round4-multicity`).

## Notation conventions used below

| symbol | type | meaning |
|---|---|---|
| `u, U` | int, set | user index; user set (|U| = N) |
| `i, I` | int, set | item index; item set (|I| = M) |
| `q = (u, t)` | request | a (user, time) pair to be recommended on |
| `B` | int | batch / number of requests in a split |
| `K_mac` | int | number of macro categories (Foursquare top-level, 10) |
| `K` | int | number of situations |
| `D` | int | descriptor dimension (= A + K_mac) |
| `A` | int | number of context attributes (= 6) |
| `[·‖·]` | op | concatenation along last axis |

---

# §1. L0 — SENSING + DATA PREPARATION

Operationalises the framework's **SENSING** box and accounts for the
**Data Preparation** trustworthy principle (data cleaning, correction,
debias, causal split).

## §1.1 Preprocessing pipeline — `pipeline/step01_preprocessing/__init__.py`

Public entry: `preprocess_tsmc2014(raw_tsv, out_dir, taxonomy_path,
k_core=10, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)`.

Steps in order (function pointers in parentheses):

### §1.1.1 Load + UTC→local time (`_load_and_localize`)

Reads the 8-column TSV (latin-1) — `(user_id, venue_id, cat_id,
cat_name, lat, lon, tz_offset, utc_time)` — parses `utc_time` with
format `"%a %b %d %H:%M:%S %z %Y"`, computes

$$t_{\text{local}} \;=\; t_{\text{UTC}} + \Delta_{\text{tz}}$$

with `tz_offset` interpreted in minutes. The local time is stored
tz-naive (the tz info is preserved in the separate column). Exact
duplicates `(user_id, venue_id, utc_time)` are dropped before any
further processing.

### §1.1.2 Iterative user × item k-core (`_iterative_kcore`)

For default `k = 10`: repeat

$$U' \;=\; \{u : |\{i : (u,i) \in D\}| \ge k\},\quad
  I' \;=\; \{i : |\{u : (u,i) \in D\}| \ge k\}$$

$$D \;\leftarrow\; D \cap (U' \times I')$$

**until** `|D|` stops shrinking. Convergence is exact: the loop
breaks when `len(cur) == n_before`. **Both users AND items** are
filtered at each iteration on **unique-interaction count** (not raw
row count); duplicates collapse via `nunique`.

### §1.1.3 Per-user temporal 80/10/10 split (`_per_user_temporal_split`)

Inspired by `datarec.LeaveRatioLast` (Caruccio et al., SIGIR 2025).
Per user, sort by `time_local` (stable sort), assign

$$\text{frac}(u, p) \;=\; \frac{p+1}{n_u}, \quad
  \text{split}(u,p) \;=\;
  \begin{cases}
  \text{train} & \text{if } \text{frac} \le 0.8\\
  \text{val} & \text{if } 0.8 < \text{frac} \le 0.9\\
  \text{test} & \text{if } \text{frac} > 0.9
  \end{cases}$$

where `p` is the 0-indexed position in the chronological history and
`n_u` is the user's total count. Boundary handling: chronological
ties broken by stable sort order; users with fewer than ~3
interactions can end up entirely in `train` (no val/test).
**Verified by `tests/test_preprocessing.py` invariant 4**:
`max(t in train_u) ≤ min(t in val_u) ≤ min(t in test_u)` for every
user.

### §1.1.4 Cold filter (`_cold_filter`)

After the split, drop any `(u, i)` in `val ∪ test` such that
`u ∉ train_users ∨ i ∉ train_items`. This is a hard, *additional*
defence on top of k-core, used because the temporal split can isolate
users/items late in time. **Verified by invariant 2**.

### §1.1.5 Causal intent proxy (`_add_intent_proxy`)

Sort by `(user_id, time_local)` (stable); for each row, set

$$\text{intent\_last\_cat}(u, t) \;=\;
  \text{cat\_macro}\big(\text{shift}_{+1, \text{user}=u}\big)$$

i.e. the macro of the user's previous interaction. The pandas
`groupby("user_id").shift(1)` is strictly causal: row p sees row p-1
of the same user only. The first row per user is `"None"`.

**Verified by invariant 5** (`tests/test_preprocessing.py`):

> Each row's `intent_last_cat` was observed at a strictly earlier
> `time_local` than that row, for the same user — OR is `"None"`.

This is the test that has been called "leakage test" in conversation;
its **actual claim** is: per-row strict temporal precedence within
user. It does **not** check across the split boundary (val/test rows
can correctly take their `intent_last_cat` from a train row of the
same user — that is causal, not leaky).

### §1.1.6 Contextual features

* Temporal (`_add_temporal_features`): `c_hour ∈ {0..23}`, `c_dow ∈
  {0..6}`, `c_isweekend = 𝟙[c_dow ≥ 5]`, `c_month ∈ {1..12}`.
* Spatial (`_add_spatial_features`): `geohash5`, `geohash4` (pygeohash);
  `dist_prev` = haversine km between consecutive check-ins of the
  same user, NaN-filled to 0.
* Semantic (`_add_semantic_features`): `cat_fine = cat_name`;
  `cat_macro = build_cat_id_to_macro(taxonomy)[cat_id]` with
  `"Other"` fallback for IDs not in the legacy Foursquare v2 taxonomy.

### §1.1.7 Dual view (`_build_dual_view`)

Two coherent persistence layers on the SAME split (URM + parquet):
`URM_{train,val,test}.npz` (binary `csr_matrix`,
shape `(N, M)`) and `df_{train,val,test}.parquet` (rich row-wise).
Coherence is **invariant 3**: set of `(u_idx, i_idx)` agrees per split.

### §1.1.8 Causal recent window — `pipeline/step02_models/xsage/l0_sensing.py`

For every request `q = (u, t)` produce the user's **strictly
prior** last-`n` history (`build_recent_window`):

$$H_n(u, t) \;=\; \big\{ \big( \text{macro}_j, \Delta t_j \big) \big\}_{j=0}^{L-1},
  \qquad L = \min(n, |\{r : r.u = u,\ r.t < t\}|)$$

where `Δt_j = (t − t_j)` in minutes, "most recent first". History is
padded with `-1` macro / `NaN` deltas to fixed width `n`.

**History rule (strict causality across splits):**
- target row in `train` → history drawn from `train` rows strictly before;
- target row in `val` → history drawn from full `train`;
- target row in `test` → history drawn from `train ∪ val`
  (refit step).

This is implemented by passing different `history_df` to the same
`build_recent_window` function.

## §1.2 Trustworthiness measures enabled at L0

| TW principle | what we implemented |
|---|---|
| **Data cleaning** | exact-duplicate drop on `(u, i, utc_time)` |
| **Data correction** | tz-aware time localisation; haversine NaN→0 |
| **Data debias** (anti-leakage) | per-user 80/10/10 chronological split + cold filter — verified by 5 invariants (`tests/test_preprocessing.py`) |
| **Data debias** (causal context) | strict-prior intent proxy + per-split history rule for `H_n` |
| **Decentralization** | n/a in round-3; future federated angle deferred |

The 5 invariants are the auditable, asserted-on-every-run guarantee
that this level delivers "trustworthy input data" in the figure's
sense.

## §1.3 Discrepancies with verbal descriptions

| said verbally | actually does |
|---|---|
| "k-core" | iterative until convergence; thresholds users AND items on *unique-interaction count*; NOT a single pass |
| "temporal split" | per-user-fractional on chronological position; NOT a global cut date |
| "leakage test" | per-row strict-prior assertion within user; does NOT test cross-split (that is correct by design) |

---

# §2a. L1a — LOW-LEVEL PERCEPTION (context contribution)

Operationalises the **Low-Level Perception** (ϕ₀: raw → C₀) box. Maps
each request's raw attributes into a **context state vector c̃**
expressing per-attribute informativeness.

## §2a.1 Contribution functions — `pipeline/step02_models/xsage/l1_perception.py`

### §2a.1.1 The shallow-tree estimator (`fit_contribution_functions`)

For each attribute `a` in

$$\mathcal{A} = (\text{c\_hour},\ \text{c\_dow},\ \text{c\_isweekend},\ 
  \text{c\_month},\ \text{prev\_geohash5},\ \text{intent\_last\_cat\_idx})$$

(thus `A = 6`), fit a sklearn `DecisionTreeClassifier(max_depth=3,
min_samples_leaf=200, random_state=42)` predicting **the next-row
`cat_macro`** (`cat_target` in `L0Output`) from `a` alone. The
multi-valued string attributes (`prev_geohash5`, `intent_last_cat_idx`)
are pre-mapped to ints via a train-only vocabulary; numeric attributes
are passed as-is.

For every leaf `b` of every tree, compute the leaf-empirical
next-macro distribution `p_b`, its (base-2) Shannon entropy `H(p_b)`,
and define the leaf-level **contribution score**

$$\theta_{a,b} \;=\; 1 \;-\; \frac{H(p_b)}{\log_2 K_{\text{mac}}}
  \;\in\; [0, 1].$$

This is the **CST attribute-contribution form** stated in the brief
(`S(X) = Σ w_i contrᵢ(x_i)` family), with the choice of `1 −
normalised-entropy` as the per-attribute contribution and equal
attribute weights at this stage (no learned `w_i` yet — see notes
below).

### §2a.1.2 Inference (`ContributionModel.transform`)

For a batch of `B` rows, the tree's `apply` routes each row to a leaf
per attribute, and the model returns the stack

$$\tilde c \in [0, 1]^{B \times A},
  \qquad \tilde c_{b, a} \;=\; \theta_{a, \,\text{leaf}_a(x_a)}.$$

**Note (verbal-vs-actual):** the verbal description "c̃ is per-attribute"
is correct. `c̃` is **NOT** summed/aggregated into a scalar `S(X)` —
the framework consumes the **full A-vector** downstream (see §3.1).
The CST "weighted-sum" form is the per-leaf decision-tree training
target (entropy reduction), not a separate aggregation step in the
forward pass.

## §2a.2 Trustworthiness measures enabled at L1a

| TW principle | what we implemented |
|---|---|
| **Data debias** (no future use) | each tree is fit on `df_train` only; the trees consume only train-known leaves at inference; the geohash vocabulary is train-only |
| **Explainability** (axis 1) | shallow trees (depth=3) are inspectable; each `θ_{a,b}` is an *attribute-level informativeness* score in [0, 1] — directly interpretable |
| **Robustness** | `min_samples_leaf = 200` floor on leaves prevents over-fragmentation; entropy normalisation by `log₂K_mac` makes the score scale-free across cities with different `K_mac` |

## §2a.3 Discrepancies

| said verbally | actually does |
|---|---|
| "c̃ is a context score" | c̃ is a **vector** in [0,1]^A (per-attribute), not a single number |
| "CST-style S(X) = Σ w_i contr(x_i)" | the CST aggregation happens **inside each per-attribute tree** (entropy minimisation); the forward output is the per-attribute contribution stack, consumed as a multi-dimensional descriptor — not summed |

---

# §2b. L1b — HIGH-LEVEL PERCEPTION (intent via transition graph)

Operationalises **High-Level Perception** (ϕ₁: (c₀, u) → C₁). Adds the
user-conditioned **intent vector e** by walking the macro-transition
graph from the user's recency profile.

## §2b.1 Macro-transition matrix W — `estimate_macro_transition`

Sort `df_train` by `(user_id, time_local)`. Let `m[r]` be the
mapped-to-int macro of row `r` and `u[r]` its user. Walk consecutive
pairs `(r, r+1)` with `u[r] = u[r+1]`:

$$N_{c \to c'} \;=\; \big|\{r : m[r] = c,\ m[r+1] = c',\ u[r] = u[r+1]\}\big|$$

Apply add-one (Laplace) smoothing and row-normalise:

$$W_{c, c'} \;=\; \frac{N_{c \to c'} + 1}{\sum_{c''} (N_{c \to c''} + 1)},
  \qquad W \in \mathbb{R}^{K_{\text{mac}} \times K_{\text{mac}}},\ \ 
  \sum_{c'} W_{c, c'} = 1.$$

**Self-loops are preserved** (counts of `c → c`). The matrix is
**row-stochastic** by construction.

**Round-3 C2 transit-aware variant.** A `transit_mode` knob:

* `keep` (default): formula above.
* `collapse`: same W; attractor identification later excludes the
  named transit macros (see §2b.2).
* `mask`: the named transit macros are **contracted out of each user
  sequence before counting**. Specifically: drop every row whose
  `cat_macro ∈ transit_macros`, then re-form consecutive pairs on
  the surviving (still-chronological) sequence. A subsequence
  `A → Transit → B` becomes a direct `A → B` count with the
  **observed-from-data** probability (not the matrix product
  `W_{A,T} · W_{T,B}`). This is "graph surgery on data", not on W.

## §2b.2 Attractors — `find_attractors`

$$\text{indeg}(c) \;=\; \sum_{c'} W_{c', c}, \qquad
  \mathcal{A} \;=\; \big\{ c : \text{indeg}(c) \ge \overline{\text{indeg}} \big\}$$

i.e. **column sums** (incoming mass, weighted by row probabilities) ≥
mean column sum. Returned as a boolean mask of length `K_mac`.

Under `collapse` mode the caller passes
`exclude_indices = [idx of transit macros]`; those indices are
**forced to False** in the returned mask, regardless of their column
sum.

## §2b.3 Recency profile m — `compute_profile`

For each request with recent window `recent_macro[b, :]` of length `n`
(most recent at j=0; `-1` padding for short histories):

$$m_{b, c} \;=\; \frac{ \sum_{j=0}^{n-1} \gamma^{j}\,
  \mathbb{1}[\text{recent\_macro}[b,j] = c \wedge \text{valid}] }
  { \sum_{c'} \sum_{j} \gamma^{j} \mathbb{1}[\dots = c'] }
  \;\in\; \mathbb{R}^{B \times K_{\text{mac}}}.$$

* `valid` excludes padding and (under transit `mask`) the transit
  macros. Defaults: `γ = 0.6`, `n = 5`.
* **Empty histories** (`n_prior = 0`) get a *uniform* row
  `m_{b, c} = 1 / K_mac`.
* Otherwise rows sum to 1.

## §2b.4 Reachability / intent vector e — `compute_intent`

Build a finite, β-discounted power sum of W:

$$\mathcal{R} \;=\; \sum_{k=1}^{H} \beta^{k}\, W^{k},
  \qquad W^{1} := W,\ \ W^{k+1} := W \cdot W^{k},\ \ \beta \in (0,1).$$

Project the recency profile through `R` and project onto the attractor
subset:

$$\tilde e_{b, c} \;=\; \big( m_{b, :}\, \mathcal{R} \big)_{c} \cdot
  \mathbb{1}[c \in \mathcal{A}]$$

(elementwise mask in the **default `hard` mode**). Row-normalise:

$$e_{b, c} \;=\; \frac{\tilde e_{b, c}}{\sum_{c'} \tilde e_{b, c'}}
  \qquad \text{(or 0 if the row sum is 0)}.$$

Defaults: `H = 2`, `β = 0.7`. Dimension `(B, K_mac)`; on `hard` mode
the entries outside `\mathcal{A}` are exactly 0.

**Round-2 1.4 variants** of the projection step:

* `hard` (default): the indicator mask above. Default for the paper.
* `all`: no mask — every macro dimension is kept, weighted by raw
  reachability. Useful when the attractor cut is too aggressive
  (TKY shrinks to 2 dims).
* `soft_topr`: per row, zero everything outside the top-r reachable
  macros, then renormalise.

## §2b.5 Trustworthiness measures enabled at L1b

| TW principle | what we implemented |
|---|---|
| **Trustworthy data representation** | row-stochastic W with add-one smoothing → no zero-probability pathological rows; β finite-horizon prevents unbounded reachability growth |
| **Explainability** (axis 1) | every dimension of e is a real macro category; "attractor ↔ inflow above average" is human-readable |
| **Robustness** (round-3 C2) | the transit-mode knob is a documented graph-surgery option that can be turned on/off without rebuilding the descriptor — used in the round-3 ablation that defended the TKY narrative |
| **Causality** | m, W, attractors all fit on train only; e at inference re-uses the same fixed W |

## §2b.6 Discrepancies

| said verbally | actually does |
|---|---|
| "e = m · Σ β^k W^k" | exact, with `k = 1..H`, finite **H = 2**, β = 0.7; then **masked to attractors** (hard mode) and renormalised. Verbal version omitted the mask/renorm |
| "attractors are 'typical destinations'" | mathematically: macros with **above-average in-degree** in the user-walk transition graph; in-degree is **column sum**, not raw count |
| "γ recency, β transition" | both are scalars `< 1` but they act on different objects — γ on the recency-window history (`m`), β on the power-of-W expansion (`e`) |

---

# §3. L2 — COMPREHENSION (rough k-means situations)

Operationalises the **COMPREHENSION** box
`f: (C₁, u, Σ, G) → {(S_1, p_1), …, (S_K, p_K)}`. The output is the
situation distribution with confidence weights.

## §3.1 The descriptor v

Concatenation:

$$\mathbf{v}_b \;=\; \big[\,\tilde c_b \,\Vert\, e_b\,\big] \;\in\; \mathbb{R}^{D},
  \qquad D \;=\; A + K_{\text{mac}}.$$

For the round-3 NYC mask config `(A, K_mac) = (6, 9)` ⇒ `D = 15`.
For TKY keep `(6, 8)` ⇒ `D = 14`.

**Note (verbal-vs-actual):** the descriptor is **NOT standardised or
rescaled** before clustering. `c̃` lives in `[0, 1]^A`; `e` is a
row-stochastic vector on the attractor subset (values in `[0, 1]`).
The two blocks already share scale by construction; no z-score or
PCA normalisation step exists in the code.

## §3.2 Rough k-means (Lingras–West) — `pipeline/step02_models/xsage/l2_comprehension.py::fit_rough_kmeans`

### §3.2.1 Initialisation

k-means++ seeding (`_init_kmeanspp`) with `seed`. Returns
`μ ∈ ℝ^{K × D}`.

### §3.2.2 Assignment (`_assign`)

For each batch row `v_b`, compute Euclidean distances `d_{b, k} =
‖v_b − μ_k‖`. Let `k^*(b) = argmin_k d_{b, k}` and
`d^*(b) = d_{b, k^*}`. The **competing set** is

$$T(v_b) \;=\; \big\{ k : d_{b, k} - d^*(b) \le \varepsilon \big\}$$

where `ε` is the boundary threshold in **distance units** (not a
relative fraction). Then

* core (lower-approx-only): `|T| = 1`, i.e. `is_boundary[b] = False`.
* boundary: `|T| > 1`, i.e. `is_boundary[b] = True`.

The **membership** vector returned to downstream stages is

$$r_{b, k} \;=\;
  \begin{cases}
  \mathbb{1}[k = k^*(b)] & \text{if core (} |T| = 1 \text{)}\\
  \mathbb{1}[k \in T(v_b)] \,/\, |T(v_b)| & \text{if boundary}
  \end{cases}$$

⇒ on core requests `r` is a one-hot at `k^*`; on boundary requests it
is uniform over the competing cluster set.

### §3.2.3 Prototype update (`_update`)

For each cluster `k`:

* "lower-only" pool: `L_k = {b : k^* = k,\ \neg is\_boundary[b]}`.
* "upper-only" pool: `U_k \setminus L_k = {b : k \in T(v_b),\ is\_boundary[b]}`.

Then

$$\mu_k \;\leftarrow\; w_l \cdot \overline{v}_{L_k} \;+\; w_b \cdot \overline{v}_{U_k \setminus L_k}$$

with **constraints `w_l > w_b`, `w_l + w_b = 1`** (asserted at the top
of `fit_rough_kmeans`). Defaults: `w_l = 0.7`, `w_b = 0.3`.

Edge case: if **either** pool is empty, the update reduces to the
non-empty pool's mean; if **both** are empty, the cluster is "dead"
and the prior `μ_k` is kept (a "healing" pass in the main loop).

### §3.2.4 Convergence

Repeat until `‖μ_new − μ_old‖_F < tol = 1e-4` OR `max_iter = 50/80`.

### §3.2.5 (K, ε) selection — `pipeline/step02_models/xsage/orchestrator.py::_tune_K_and_eps`

Joint grid search:

$$K \in \{4, 6, 8\},\qquad \varepsilon \in \{0.010,\ 0.020,\ 0.030,\ 0.050\}.$$

For each cell, fit twice with seeds `(42, 43)`, compute

* `ARI = adjusted_rand_score(r_seed42.core_label, r_seed43.core_label)`
* `boundary_fraction = is_boundary.mean()` (under seed 42).

**Objective.** Pick the cell with the **highest ARI** among those
whose `boundary_fraction ∈ [0.10, 0.30]`. If none fall in band,
fallback: pick the cell maximising
`ARI − |boundary_fraction − 0.20|`.

The ARI implementation is local (Hubert & Arabie 1985, `comb`-based;
no sklearn dep).

## §3.3 Trustworthiness measures enabled at L2

| TW principle | what we implemented |
|---|---|
| **Robust representation** | rough k-means (Lingras–West) handles ambiguity *natively*: requests near a Voronoi edge become **boundary** with uniform membership instead of arbitrary hard assignment; downstream stages see this via the `is_boundary` flag and a uniform `r` |
| **Robustness via cross-seed stability** | (K, ε) is selected on **two-seed ARI**, not on a within-fit objective. The chosen `(K, ε)` is the one with highest ARI in the [10%, 30%] boundary band |
| **Explainability** | every centroid `μ_k` is a vector in the same descriptor space as `v` — the per-attribute and per-attractor components are directly readable (this is what B11's situation cards visualise) |
| **Adaptivity** | the boundary fraction band tolerates city-specific shifts (NYC ε=0.03, TKY ε=0.05 in round-3); no hand-tuning per city |

## §3.4 Discrepancies

| said verbally | actually does |
|---|---|
| "(K, ε) chosen by silhouette / inertia" | chosen by **cross-seed ARI** with a boundary-band constraint; silhouette is **not** used as the selection objective |
| "boundary points get half the weight" | the **prototype** update gives `w_b = 0.3` to the boundary-only pool (vs `w_l = 0.7` to the lower-only pool); the **membership** `r` gives boundary requests a uniform `1/|T|` over the competing clusters |
| "rough k-means = soft k-means" | NOT the same. Soft k-means assigns *graded* weights to all K clusters via softmax of distances; rough k-means uses a **hard threshold ε** to define a SET of competing clusters and assigns *uniform* weight inside that set, zero outside |

---

# §4. L3 — PROJECTION + RECOMMENDATION + EVALUATION

Operationalises the framework's **PROJECTION** box
`π: (C₁, S, p, u) → (L, (S⁺_j, p⁺_j))`. This is also where the
**Recommendation** (fair, transparent) and **Evaluation** trustworthy
principles enter.

## §4.1 Situation transition matrix T — `pipeline/step02_models/xsage/l3_projection.py::estimate_transition`

Per user, sort by time and produce the sequence of core labels
`z_u = (z_1, z_2, …, z_{n_u})`. Walk consecutive pairs:

$$N^{(s)}_{k, k'} \;=\; \sum_{u} \sum_{p=1}^{n_u - 1}
  \mathbb{1}[z_{p} = k]\,\mathbb{1}[z_{p+1} = k'].$$

Add-one smoothing + row-normalise:

$$T_{k, k'} \;=\; \frac{N^{(s)}_{k, k'} + 1}{\sum_{k''} (N^{(s)}_{k, k''} + 1)}
  \;\in\; \mathbb{R}^{K \times K}.$$

## §4.2 Stage-C next-situation prediction — `predict_next_situation`

Given a query `z_t` (the situation index of the previous request of
the same user — `−1` if none), the prediction is

$$\hat z_{t+1} \;=\; \operatorname*{argmax}_{k'} T_{z_t,\,k'}.$$

The comparison baseline is `time_only_prior`: at train time build
`P(z | c_hour)` as the most common `z` observed at each of 24 hours
(`np.argmax`, ties broken by lower index); at test time predict
`hat z = most[c_hour]`.

The Stage-C headline number is

$$\Delta F_1 \;=\; F_1^{\text{macro}}(T\text{-based}) \;-\;
                  F_1^{\text{macro}}(\text{time-only}),$$

with macro-F1 averaged across K classes (zero-support classes
contribute 0 to the mean). The paired McNemar test (and continuity-
corrected χ²₁ p-value) is also computed inside Stage-C.

## §4.3 Boundary feedback (eq.18) — `boundary_disambiguate`

For a request with previous-step situation `z_prev` (an int per row;
`−1` if none) and current membership `r ∈ ℝ^K`:

$$\tilde r_k \;=\; \frac{r_k \cdot T_{z_{\text{prev}}, k}}
                          {\sum_{k'} r_{k'} \cdot T_{z_{\text{prev}}, k'}}$$

i.e. the elementwise product with the row of T indexed by the previous
situation, then row-renormalise. Rows with `z_prev = −1` are left
unchanged. **The result `\tilde r` is the membership consumed
downstream by recommendation when boundary feedback is enabled.**

## §4.4 Dynamic fairness (eq.19) — `dynamic_fairness`

Given the per-situation long-tail ratio vector `LT ∈ ℝ^K`:

$$\bar{LT}_{t:t+\tau}(k_0) \;=\; \big(T^{\tau}\big)_{k_0,\,:} \cdot LT,
  \qquad \tau \in \{1, 2, 3\}.$$

The (K, τ) output predicts the τ-step-ahead expected long-tail rate
from each starting situation.

## §4.5 Backbones — `backbone.py`, `backbone_full.py`

Two classes of backbone scores live alongside the X-SAGE head:

* **B_blind** — vanilla Foursquare-floor matrix-factorisation
  recommender (`FMRecommender` from
  `engine.Recommenders.FactorizationMachines`) refit on
  `URM_train ∪ URM_val` with HPs from the floor's Bayesian search
  (`outputs/<city>/baselines/FM.best_hp.json`). Output: `scores_B ∈
  ℝ^{N × M}` (one full per-user / per-item score matrix).
* **B_full** — context-aware Rendle FM (`ContextAwareFM` in
  `backbone_full.py`), trained with BPR loss. Multi-hot feature groups
  per active row: `user, item, cat_macro, cat_fine, c_hour, c_dow,
  c_isweekend, c_month, prev_geohash5, intent_last_cat` (10 active
  indices). Order-2 score formula:

$$\text{score}(\text{row}) \;=\; w_0 \;+\; \sum_{f \in \text{active}} w_f
  \;+\; \tfrac{1}{2}\Big( \big(\textstyle\sum_f e_f\big)^2
                          \,-\, \textstyle\sum_f e_f^2 \Big).$$

  Trained with BPR (positive vs sampled negative). At inference,
  `score_full_catalogue` factorises into a "context block" + "item
  block" so the `(B, I, d)` tensor never materialises.

## §4.6 Per-situation category biases — `recommendation.py::fit_situation_biases_z`

For each cluster `k` and macro `c` (from `df_train` only):

$$\hat b^{(k)}_c \;=\; \log\!\bigg(
  \frac{N_{k, c} + \lambda \cdot p_c}
       {\sum_{c'} (N_{k, c'} + \lambda \cdot p_{c'})}
  \bigg) \;-\; \log p_c,$$

with `λ = 50` (round-2 default) and `p_c` the global empirical macro
distribution from train. Then **z-score within situation**:

$$\tilde b^{(k)}_c \;=\; \frac{\hat b^{(k)}_c - \overline{\hat b^{(k)}}}
                              {\operatorname{std}(\hat b^{(k)})}
  \;\in\; \mathbb{R}^{K \times K_{\text{mac}}}.$$

The z-score (round-2 1.5) makes the additive scale **comparable
across situations**: `κ_S` of the combiner can be interpreted as "a
nudge of size ≈ κ standard deviations".

## §4.7 The additive combiner — `recommendation.py::additive_combine_scores`

For an item `i` with macro `c(i)` and a request `b` with membership
`r_b`:

$$\boxed{\quad \hat s_b(i) \;=\; s^{B}_b(i)
   \;+\; \kappa_S \;\gamma_S(b)\;
   \sum_{k=1}^{K} r_{b, k}\, \tilde b^{(k)}_{c(i)} \quad}$$

with:

* `s^B_b(i)` — the backbone score for `(b, i)` (typically B_blind = `FMRecommender`);
* `κ_S` — a scalar mixing coefficient (the "knob" — common values 0,
  0.1, 0.25, 0.5, 1.0);
* `γ_S(b)` — the **certainty gate** from L2:

$$\gamma_S(b) \;=\; \begin{cases} 1 & \text{if core } (|T(v_b)| = 1)\\
   1 / |T(v_b)| & \text{if boundary} \end{cases}$$

  i.e. **`γ_S` is the *inverse of |T|*, NOT 0 on boundary** — the
  uniform-over-competing rule of L2 is propagated;
* `\tilde b^{(k)}_{c(i)}` — the z-scored per-situation bias of §4.6.

**Matched-OFF identity.** The function returns `scores_B` unchanged
when `κ_S = 0.0`:

```python
if kappa == 0.0:
    return scores_B.astype(np.float32)
```

i.e. the algebraic identity `κ = 0 ⇒ \hat s = s_B` is **enforced by
the implementation** (the nudge term is bypassed). B8.3 of round-3
verified this empirically on 76 934 long-tail entries with zero
violations (max abs err 2.4·10⁻⁷ = float32 ε).

**Important note for the paper.** The round-2 1.5 *additive* combiner
in §4.7 is the one used in B7/B7b/B8 / multi-city. There is also an
**older harmonic combiner** in the codebase (`harmonic_combine`,
eq.15 of the original brief) that survived for historical reasons but
is **not** the round-3 headline. The paper should formalise the
additive form.

## §4.8 Fairness re-ranking (round-3 Track-2 long-tail boost)

Implemented inline in `experiments/round3_b5_anatomy.py` (and
identically in B7, B7b, B11). The rule:

1. Define the **short-head split** on `pop = (URM_train + URM_val).sum(axis=0)`:

$$G_0 \;=\; \text{top}_{20\%}\text{ by popularity},
  \qquad G_1 \;=\; \text{everything else (long-tail)}.$$

   (`metrics.py::long_tail_groups`, default `short_head_share = 0.20`.)

2. Identify **inequity sinks** in Stage-B as those situations `k`
   whose KL divergence exceeds ~2× the global mean
   (`fairness/verdict.json::inequity_sinks`).

3. Apply the **gated boost** per request:

$$s^{B}_b(i)
   \;\leftarrow\; s^{B}_b(i)
   \;+\; \mathbb{1}[\,\text{is\_core\_sink}(b)\,] \;\cdot\;
         \kappa_{\text{fair}} \;\cdot\; \mathbb{1}[\,i \in G_1\,]$$

   where the **boolean gate** is

$$\text{is\_core\_sink}(b) \;=\; \big(z_b \in \text{sinks}\big)
   \wedge \big(\lnot is\_boundary[b]\big).$$

4. Re-rank with the boosted scores to get the new top-K.

Defaults: `κ_fair = 1.0` on NYC (mask, sinks {6, 7}), `2.0` on TKY
(keep, sinks {4, 5}) — these are the knee operating points selected
on **val** in A4 / A1bis.

**Key algebraic property** (used by B8.3 faithfulness verification):
on touched rows the score delta is EXACTLY `κ_fair · G_1[i]`, on
untouched rows it is identically 0. Removing the boost regenerates
the OFF list bit-for-bit.

## §4.9 Trustworthiness measures enabled at L3

| TW principle | what we implemented |
|---|---|
| **Robustness** (recommendation) | the situational head is a **score-level adder** on top of any trained backbone (`B_blind`, `EASE^R`, or `B_full`). No retraining is needed to plug a different backbone — the round-3 B6 result ("backbone-agnostic lens") rests on this |
| **Fairness** (recommendation) | the long-tail boost of §4.8 is **gated by sink-ness AND core-ness**, so it never touches lists outside the documented per-situation inequity zones; the choice of `κ_fair` is selected on val (A1bis) and applied on test (selection→deployment causal split) |
| **Explainability** (recommendation) | the additive form gives a **decomposition** `\hat s = s_B + (nudge)` whose nudge term is the per-item contribution licensed by the situation; this is what B8.3 verifies as algebraic identity; B11's HTML showcase reads each request's nudge directly from the score arrays |
| **Adaptivity** | `κ_S` (mixing) and `κ_fair` (fairness boost) are scalar knobs the operator can change without retraining; the MATCHED-OFF identity `κ = 0 ⇒ \hat s = s_B` provides a guaranteed regression-safe fallback |
| **Technical evaluation** | round-3 statistical framework (paired Wilcoxon + Holm, TOST equivalence, bootstrap CI95, permutation, McNemar, Wilson CI) covers every reported claim — see `STATISTICAL_VALIDATION_SUMMARY.md` |
| **Ethical evaluation** | provider-side fairness (per-situation LT, KL) and user-side fairness (Gini of long-tail-received per user, B4) are reported alongside accuracy in every per-city report |

## §4.10 Discrepancies

| said verbally | actually does |
|---|---|
| "ŝ = s_B + κ · γ_S · b" | exact (with `b = Σ_k r_k \tilde b^{(k)}_{c(i)}`, **z-scored** per situation), in the **additive** combiner from round-2 1.5 — NOT the older harmonic combiner of eq.15 (which still exists but isn't the round-3 default) |
| "γ_S = 1 for core, 0 for boundary" | actually `γ_S = 1` on core, **`1/|T|`** (NOT zero) on boundary — i.e. the situational nudge is *attenuated* on boundary, not silenced |
| "the bias `b` is learned" | the bias is **closed-form** in `fit_situation_biases_z`: shrunk log-odds vs the global macro distribution, λ = 50, then z-scored per situation. No gradient training |
| "fairness re-rank is global" | the boost is **doubly-gated** (sink AND core); the matched-OFF identity holds for **every** untouched row |
| "boundary feedback drops boundary points" | actually multiplies `r` by the row of `T` indexed by `z_prev` and renormalises — `r̃ ∝ r ⊙ T[z_prev, :]` — a Bayesian-style transition prior, NOT a discard |

---

# §5. Single notation table (paper-ready)

| symbol | meaning | dim | code variable |
|---|---|---|---|
| `q = (u, t)` | request: user + time | scalar pair | `(u_idx[b], time_local[b])` |
| `N, M` | #users, #items | scalar | `n_users, n_items` |
| `K_mac` | #macro categories (Foursquare top-level) | scalar = 10 | `n_macros` |
| `K` | #situations | scalar (4/6/8 sweep) | `K` |
| `A` | #context attributes for c̃ | scalar = 6 | `len(DEFAULT_ATTRIBUTES)` |
| `D = A + K_mac` | descriptor dim | scalar | `v.shape[1]` |
| `n` | recency-window length | scalar = 5 | `args.n` |
| `H` | reachability horizon | scalar = 2 | `args.H` |
| `γ` | recency decay (in `m`) | scalar = 0.6 | `args.gamma` |
| `β` | transition discount (in `e`) | scalar = 0.7 | `args.beta` |
| `c̃` | context state, per-attribute contributions | (B, A) ∈ [0,1] | `c_train, c_val, c_test` |
| `W` | macro transition matrix, row-stochastic | (K_mac, K_mac) | `W` |
| `m` | recency profile, row-stochastic on (K_mac) | (B, K_mac) | `m_train, m_val, m_test` |
| `𝒜` | attractor set | bool mask, len K_mac | `attractors` |
| `e` | intent vector (β-reachability, masked to 𝒜, renormalised) | (B, K_mac) | `e_train, e_val, e_test` |
| `v = [c̃ ‖ e]` | request descriptor | (B, D) | `v_train, v_val, v_test` |
| `μ_k` | situation prototype | (K, D) | `prototypes` |
| `k*(b)` | argmin-distance situation (the **core label**) | (B,) | `core_label_*` |
| `T(v_b)` | competing-cluster set within ε | bool (B, K) | `competing_sets` |
| `r_{b, k}` | situation membership (one-hot on core, 1/|T| on boundary) | (B, K) | `membership_*` |
| `is_boundary[b]` | `|T(v_b)| > 1`? | bool (B,) | `is_boundary_*` |
| `γ_S(b)` | certainty gate = 1 (core) / 1/|T| (boundary) | scalar per b | derived from `is_boundary, competing_sets` |
| `ε` | rough-k-means boundary threshold (distance units) | scalar | `eps` |
| `w_l, w_b` | prototype-update weights (lower vs boundary pool) | scalar, 0.7/0.3 | `w_l, w_b` |
| `T` (capital) | **situation** transition matrix | (K, K) | `T` |
| `T^τ` | τ-step transition power | (K, K) | derived |
| `LT` | per-situation long-tail rate | (K,) | `lt_per_situation` |
| `G_0, G_1` | item short-head / long-tail mask | bool (M,) | `G0_mask, G1_mask` |
| `s^B_b(i)` | backbone score | (B, M) | `scores_B, scores_blind` |
| `\tilde b^{(k)}_c` | z-scored shrunk-log-odds per (situation, macro) | (K, K_mac) | `biases_z_per_situation` |
| `\hat s_b(i)` | combined score | (B, M) | output of `additive_combine_scores` |
| `κ_S` | situational mixing scalar | scalar (0.0, 0.1, 0.25, 0.5, 1.0) | `kappa` |
| `κ_fair` | long-tail-boost scalar (re-rank track) | scalar (1.0 NYC / 2.0 TKY) | `knee_kappa` |
| `λ` | Bayesian-shrinkage pseudo-count for biases | scalar = 50 | `lam` |

---

# §6. Data-flow summary (single pass, raw → final score)

The implementation realises this DAG (paths are tags into the
notation table above):

```
                                  ┌──────────── L0 (Sensing, Data Prep) ─────────────┐
raw TSV  ──→ load+UTC↓local  ──→  k-core(=10)  ──→  per-user temporal split 80/10/10
                                       ↓
                                cold filter (val/test ⊆ train users/items)
                                       ↓
                  contextual features (temporal, spatial, semantic, intent_last)
                                       ↓
                              dual view  URM_*.npz + df_*.parquet
                                       ↓
                          causal recent window H_n(u, t)        (l0_sensing.py)
                                       ↓
                                  ┌──────────── L1a (Low-level Perception) ──────────┐
                  fit_contribution_functions on train  →  per-attribute shallow trees
                                       ↓
                                       c̃ ∈ [0,1]^{B × A}            (l1_perception.py)
                                       ↓
                                  ┌──────────── L1b (High-level Perception) ─────────┐
       train sequences  →  W (row-stochastic, +1 smoothing)   (l1_perception.py)
       in-degree column-sum  →  𝒜 (attractors)
       recent_macro + γ        →  m (recency profile)
       m × Σ_{k=1..H} β^k W^k  →  mask to 𝒜, renorm  →  e ∈ [0,1]^{B × K_mac}
                                       ↓
                                v = [c̃ ‖ e]  ∈  ℝ^{B × D}
                                       ↓
                                  ┌──────────── L2 (Comprehension) ──────────────────┐
     (K, ε) sweep on (ARI seeds {42, 43}, boundary∈[0.10, 0.30])  →  best (K, ε)
     fit_rough_kmeans(K, ε) on v_train   →  μ_k, k^*, T(v), r, is_boundary
                                       ↓
                       apply μ_k to v_val, v_test  (assign-only re-use)
                                       ↓
                                  ┌──────────── L3 (Projection + Recommendation) ────┐
                       core-label sequences per user  →  T (situation, K × K)
                       Stage-C:  argmax-T vs hour-only prior  →  ΔF1, McNemar
                       dynamic-fairness:  T^τ · LT
                                       ↓
                       boundary_disambiguate(r, T, z_prev)   (eq.18, on demand)
                                       ↓
            backbone refit on URM_train ∪ URM_val      →  s^B  ∈ ℝ^{N × M}   (B_blind)
            (alt: B_full = context-aware FM, trained with BPR, ranks per request)
                                       ↓
            fit_situation_biases_z on (z_train, cat_macro_train, λ=50)
                                       ↓     ⊕  z-score per situation  →  \tilde b^{(k)}_c
                                       ↓
    additive_combine_scores(s^B, r, \tilde b, c(i), κ_S, γ_S)   (round-2 1.5)
                              \hat s_b(i) = s^B_b(i) + κ_S · γ_S(b) · Σ_k r_{b,k} · \tilde b^{(k)}_{c(i)}
                                       ↓
       (round-3 Track-2) fairness re-rank:    s_b(i) += 𝟙[is_core_sink(b)] · κ_fair · 𝟙[i ∈ G_1]
                                       ↓
                        top-K from \hat s  →  L  →  evaluator (R@K, NDCG@K, LT@K, KL)
```

Causality is preserved at every arrow: each estimator (`W`, `m`,
`μ_k`, `T`, `\tilde b`, `s^B`) is fit on data strictly prior to or
within `train` (val for refit), and the inference path reads only
those plus the request's own attributes.

---

# §7. Master discrepancy list

Consolidated from the per-section flags above — every place where the
conversational description was inexact relative to the actual code.

1. **L0 — k-core**: not a single pass, iterative until convergence; thresholds users AND items on **unique-interaction counts**, not raw row counts.
2. **L0 — temporal split**: per-user-fractional on chronological position; not a global cut date.
3. **L0 — "leakage test"**: per-row strict-prior assertion within user; does **not** test cross-split independence (intent-from-train into val/test is *correct* by design).
4. **L1a — c̃**: a **vector** in [0, 1]^A (per-attribute), not a scalar context "score".
5. **L1a — CST aggregation**: happens *inside each per-attribute tree* (entropy reduction); the forward pass keeps the per-attribute stack — there is no learned `Σ w_i contr(x_i)` collapse step.
6. **L1b — e**: hard-attractor mask + renormalisation is part of the default forward path (not an optional post-step).
7. **L1b — attractor**: in-degree = **column sum** of row-stochastic W ≥ mean; not "raw count" / "out-degree".
8. **L1b — γ vs β**: γ is the **recency** decay (in `m`); β is the **transition** discount (in `e`). They act on different objects.
9. **L2 — (K, ε) selection**: **cross-seed ARI** with a [10%, 30%] boundary band, NOT silhouette / inertia / elbow.
10. **L2 — boundary semantics**: prototypes update with `w_l = 0.7` on lower-only and `w_b = 0.3` on boundary-only; membership `r` is one-hot on core and **uniform `1/|T|` on boundary** — not the same as soft k-means.
11. **L3 — boundary feedback**: `r̃ ∝ r ⊙ T[z_prev, :]` then renorm — a **Bayesian prior**, not a "boundary drop".
12. **L3 — combiner**: the round-3 default is the **additive** `\hat s = s_B + κ_S · γ_S · Σ_k r_k \tilde b^{(k)}` (round-2 1.5), NOT the older harmonic combiner.
13. **L3 — γ_S on boundary**: `1/|T|`, NOT 0. The nudge is *attenuated*, not silenced.
14. **L3 — bias b**: **closed-form** shrunk log-odds + z-score, λ=50; no gradient training.
15. **L3 — matched-OFF identity**: hard-coded short-circuit `κ_S = 0 ⇒ return scores_B`. The "100 % faithfulness" empirically verified in B8.3 is the verification that **this short-circuit is observed by the score arrays**.
16. **L3 — fairness re-rank**: doubly gated (`sink ∧ core`); applied as `+κ_fair · 𝟙[i ∈ G_1]` only on those rows; on every untouched row the score delta is identically 0.
17. **L3 — backbone**: B_blind = `FMRecommender` (URM-only vanilla MF) refit on train+val with floor-tuned HP; **not** a CAMF and **not** an EASE^R unless the operator opts in. EASE^R lives in `backbone.py` but is used only as a robustness check in the round-2 / B6 lens audit.

---

## What this document is for

Every equation that goes into the paper's "Approach" section should
be traceable back to a (file, function) pointer above. When the paper
formalises one of the boxes in the framework figure, it should:

1. cite the **operation** (notation in §1–§4),
2. cite the **trustworthiness role** it plays (the corresponding §x.2
   sub-section, mapped to the figure's bottom layer),
3. mention any **discrepancy** consolidated in §7 that affects the
   reader's intuition.

If a future modification (round-4 or beyond) changes one of these
primitives, this document is the place to update first, before the
paper's equations diverge from the code.
