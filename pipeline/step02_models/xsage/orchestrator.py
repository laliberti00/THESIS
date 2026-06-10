"""Stage runners that wire L0 → L1 → L2 → … on top of the city dataset.

Each ``run_stage_x`` is its own entry point so the CLI can call them
individually and so a half-implemented stage doesn't import-fail the others.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

from .l0_sensing import L0Output, build_recent_window, _macro_to_index
from .l1_perception import (DEFAULT_ATTRIBUTES, compute_intent, compute_profile,
                              estimate_macro_transition,
                              fit_contribution_functions, find_attractors)
from .l2_comprehension import (adjusted_rand_score, auto_epsilon,
                                  fit_rough_kmeans)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared dataset assembly
# ---------------------------------------------------------------------------

def _processed_dir(city: str) -> Path:
    return (Path(__file__).resolve().parents[3]
             / "data" / "processed" / city)


def _add_derived_columns(df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``prev_geohash5`` (the previous check-in's geohash, from the same
    user) and ``intent_last_cat_idx`` (integer version of ``intent_last_cat``).

    Both are causal: ``prev_geohash5`` is read from rows in ``history_df``
    strictly earlier than the current row's ``time_local``.
    """
    h = (history_df[["user_id", "time_local", "geohash5"]]
         .sort_values(["user_id", "time_local"]).reset_index(drop=True))
    prev_gh = np.array(["__NONE__"] * len(df), dtype=object)
    users = df["user_id"].values
    times = df["time_local"].values.astype("datetime64[ns]")
    by_user: dict[int, dict[str, np.ndarray]] = {}
    for u, g in h.groupby("user_id", sort=False):
        by_user[int(u)] = {
            "t": g["time_local"].values.astype("datetime64[ns]"),
            "gh": g["geohash5"].values,
        }
    for b in range(len(df)):
        rec = by_user.get(int(users[b]))
        if rec is None:
            continue
        cut = np.searchsorted(rec["t"], times[b], side="left")
        if cut == 0:
            continue
        prev_gh[b] = rec["gh"][cut - 1]
    out = df.copy()
    out["prev_geohash5"] = prev_gh
    return out


def _add_intent_last_cat_idx(df: pd.DataFrame,
                               macro_to_idx: dict[str, int]) -> pd.DataFrame:
    """Map ``intent_last_cat`` to an int (the ``'None'`` value goes to -1)."""
    out = df.copy()
    out["intent_last_cat_idx"] = np.array(
        [macro_to_idx.get(x, -1) for x in df["intent_last_cat"].values],
        dtype=np.int32,
    )
    return out


def _load_city(city: str) -> dict:
    """Read parquet + URM. Add the two derived columns the gate needs. Build
    a stable macro vocabulary across all three splits.
    """
    p = _processed_dir(city)
    df_train = pd.read_parquet(p / "df_train.parquet")
    df_val = pd.read_parquet(p / "df_val.parquet")
    df_test = pd.read_parquet(p / "df_test.parquet")
    urm_train = sps.load_npz(p / "URM_train.npz").tocsr()
    urm_val = sps.load_npz(p / "URM_val.npz").tocsr()
    urm_test = sps.load_npz(p / "URM_test.npz").tocsr()

    macro_to_idx = _macro_to_index(pd.concat([df_train, df_val, df_test]))
    n_macros = len(macro_to_idx)
    n_users, n_items = urm_train.shape

    # train rows use train as their own history; val rows use train; test rows
    # use train ∪ val (refit step).
    history_for_val = df_train
    history_for_test = pd.concat([df_train, df_val], ignore_index=True)

    df_train_d = _add_intent_last_cat_idx(
        _add_derived_columns(df_train, df_train), macro_to_idx)
    df_val_d = _add_intent_last_cat_idx(
        _add_derived_columns(df_val, history_for_val), macro_to_idx)
    df_test_d = _add_intent_last_cat_idx(
        _add_derived_columns(df_test, history_for_test), macro_to_idx)

    # The contribution-functions target: predict the current row's cat_macro
    # from the (causal) attributes we observe BEFORE that row. Renaming for
    # symmetry with L0 output naming.
    for d in (df_train_d, df_val_d, df_test_d):
        d["cat_target"] = np.array(
            [macro_to_idx[c] for c in d["cat_macro"].values],
            dtype=np.int32,
        )

    return {
        "df_train": df_train_d,
        "df_val": df_val_d,
        "df_test": df_test_d,
        "urm_train": urm_train,
        "urm_val": urm_val,
        "urm_test": urm_test,
        "macro_to_idx": macro_to_idx,
        "idx_to_macro": {i: m for m, i in macro_to_idx.items()},
        "n_users": n_users, "n_items": n_items, "n_macros": n_macros,
        "history_for_val": history_for_val,
        "history_for_test": history_for_test,
    }


# ---------------------------------------------------------------------------
# Stage A — situations
# ---------------------------------------------------------------------------

def _build_perception(ds: dict, n: int, gamma: float, H: int, beta: float,
                       attributes: tuple[str, ...] = DEFAULT_ATTRIBUTES,
                       verbose: bool = False) -> dict:
    """Build c̃, W, attractors, m, e for train / val / test under causal rules.

    Returns a dict with per-split:
        c_tilde:    (B, A)
        m:          (B, n_macros)
        e:          (B, n_macros)  — restricted to attractors, others zero
        v:          (B, A + n_macros)
        L0:         dataclass with the raw window + cat_target
    Plus shared:
        W, attractors, contribution_model, attribute_names
    """
    macro_to_idx = ds["macro_to_idx"]
    n_macros = ds["n_macros"]

    # Build L0 per split using the right history (per the brief).
    l0_train = build_recent_window(ds["df_train"], ds["df_train"],
                                     macro_to_idx, n=n)
    l0_val = build_recent_window(ds["df_val"], ds["df_train"],
                                   macro_to_idx, n=n)
    l0_test = build_recent_window(ds["df_test"], ds["history_for_test"],
                                    macro_to_idx, n=n)
    if verbose:
        print(f"  L0: train={len(l0_train)} val={len(l0_val)} test={len(l0_test)}")

    # Contribution functions (fit on train; the attributes the gate reads are
    # all present in the derived dataframe).
    contrib = fit_contribution_functions(ds["df_train"], macro_to_idx,
                                            attributes=attributes,
                                            max_depth=3, min_leaf=200)
    c_train = contrib.transform(ds["df_train"])
    c_val = contrib.transform(ds["df_val"])
    c_test = contrib.transform(ds["df_test"])

    # Macro transition + attractors (estimated on train only).
    W = estimate_macro_transition(ds["df_train"], macro_to_idx)
    attractors = find_attractors(W)
    if verbose:
        print(f"  attractors: "
              f"{[ds['idx_to_macro'][i] for i in np.where(attractors)[0]]}")

    # Recency profile m and intent vector e.
    m_train = compute_profile(l0_train.recent_macro, l0_train.n_prior,
                                n_macros, gamma=gamma)
    m_val = compute_profile(l0_val.recent_macro, l0_val.n_prior,
                              n_macros, gamma=gamma)
    m_test = compute_profile(l0_test.recent_macro, l0_test.n_prior,
                               n_macros, gamma=gamma)
    e_train = compute_intent(m_train, W, attractors, H=H, beta=beta)
    e_val = compute_intent(m_val, W, attractors, H=H, beta=beta)
    e_test = compute_intent(m_test, W, attractors, H=H, beta=beta)

    v_train = np.concatenate([c_train, e_train], axis=1).astype(np.float32)
    v_val = np.concatenate([c_val, e_val], axis=1).astype(np.float32)
    v_test = np.concatenate([c_test, e_test], axis=1).astype(np.float32)

    return {
        "train": {"L0": l0_train, "c": c_train, "m": m_train, "e": e_train, "v": v_train},
        "val": {"L0": l0_val, "c": c_val, "m": m_val, "e": e_val, "v": v_val},
        "test": {"L0": l0_test, "c": c_test, "m": m_test, "e": e_test, "v": v_test},
        "W": W, "attractors": attractors,
        "contrib": contrib, "attributes": tuple(attributes),
    }


def _pca_2d(X: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Plain PCA → 2D + the projection matrix (rows = PCs)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T, Vt[:2]


def _archetype_name(prototype: np.ndarray,
                     attribute_names: tuple[str, ...],
                     idx_to_macro: dict[int, str],
                     attractors_mask: np.ndarray,
                     n_macros: int,
                     top_attr: int = 2) -> str:
    """Build a readable label: top contribution attributes + top attractor."""
    c_part = prototype[: len(attribute_names)]
    e_part = prototype[len(attribute_names):]
    top_attr_idx = np.argsort(c_part)[::-1][:top_attr]
    bits = [f"{attribute_names[i]}↑" for i in top_attr_idx]
    if e_part.sum() > 0:
        valid = np.where(attractors_mask)[0]
        if valid.size:
            top_e = int(valid[np.argmax(e_part[valid])])
            bits.append(f"→{idx_to_macro[top_e]}")
    return " · ".join(bits) if bits else "(uniform)"


def _save_archetypes_csv(out_path: Path, prototypes: np.ndarray,
                           core_label: np.ndarray, is_boundary: np.ndarray,
                           attribute_names: tuple[str, ...],
                           idx_to_macro: dict[int, str],
                           attractors_mask: np.ndarray,
                           cat_targets: np.ndarray, n_macros: int) -> None:
    K = prototypes.shape[0]
    rows = []
    for k in range(K):
        c = (core_label == k) & ~is_boundary
        b = (core_label == k) & is_boundary
        cat_in_situation = cat_targets[c | b]
        if cat_in_situation.size:
            cat_counts = np.bincount(cat_in_situation, minlength=n_macros).astype(float)
            cat_counts /= max(cat_counts.sum(), 1.0)
            top_cats = ", ".join(
                f"{idx_to_macro[int(i)]}:{cat_counts[i]:.2f}"
                for i in np.argsort(cat_counts)[::-1][:3])
        else:
            top_cats = ""
        rows.append({
            "id": k,
            "name": _archetype_name(prototypes[k], attribute_names,
                                      idx_to_macro, attractors_mask, n_macros),
            "n_core": int(c.sum()), "n_boundary": int(b.sum()),
            "top_context_bins": ", ".join(
                f"{attribute_names[i]}={prototypes[k, i]:.2f}"
                for i in np.argsort(prototypes[k, :len(attribute_names)])[::-1][:3]),
            "top_attractors": ", ".join(
                f"{idx_to_macro[int(i)]}={prototypes[k, len(attribute_names) + i]:.2f}"
                for i in (np.argsort(prototypes[k, len(attribute_names):])[::-1][:3])
                if attractors_mask[int(i)]),
            "top_target_cat_macros": top_cats,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _save_examples_csv(out_path: Path, ds: dict, perception: dict,
                         result, split: str = "test", n_per_situation: int = 10) -> None:
    """Pull `n_per_situation` rows per situation from the chosen split,
    with their `c̃`, `e`, label, boundary flag, and the actual next macro/venue.
    """
    df = ds[f"df_{split}"]
    P = perception[split]
    K = int(result.core_label.max() + 1)
    rng = np.random.default_rng(42)
    chunks = []
    for k in range(K):
        idx_all = np.where(result.core_label == k)[0]
        if idx_all.size == 0:
            continue
        choice = rng.choice(idx_all, size=min(n_per_situation, idx_all.size),
                             replace=False)
        for j in choice:
            row = {
                "situation_id": k,
                "u_idx": int(df.iloc[j]["u_idx"]),
                "time_local": str(df.iloc[j]["time_local"]),
                "is_boundary": bool(result.is_boundary[j]),
                "next_cat_macro": str(df.iloc[j]["cat_macro"]),
                "next_venue_id": str(df.iloc[j]["venue_id"]),
                "next_cat_fine": str(df.iloc[j]["cat_fine"]),
            }
            for ai, an in enumerate(perception["attributes"]):
                row[f"c_{an}"] = float(P["c"][j, ai])
            for em in range(P["e"].shape[1]):
                row[f"e_{ds['idx_to_macro'][em]}"] = float(P["e"][j, em])
            chunks.append(row)
    pd.DataFrame(chunks).to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Stage A entry
# ---------------------------------------------------------------------------

def _tune_K_and_eps(v: np.ndarray, seed_a: int = 42, seed_b: int = 43,
                      K_grid: tuple[int, ...] = (4, 6, 8),
                      eps_grid: tuple[float, ...] = (0.010, 0.020, 0.030, 0.050),
                      target_boundary: tuple[float, float] = (0.10, 0.30)) -> tuple[dict, list[dict]]:
    """Grid-search over ``(K, ε)`` using two seeds; pick the configuration
    with highest ARI that also lands the boundary fraction in
    ``target_boundary``. Falls back to the highest-ARI config overall if none
    sits in the band.
    """
    rows = []
    for K in K_grid:
        for eps in eps_grid:
            r1 = fit_rough_kmeans(v, K=K, eps=eps, seed=seed_a, max_iter=80)
            r2 = fit_rough_kmeans(v, K=K, eps=eps, seed=seed_b, max_iter=80)
            ari = adjusted_rand_score(r1.core_label, r2.core_label)
            b_frac = float(r1.is_boundary.mean())
            rows.append({"K": K, "eps": eps, "ari": float(ari),
                            "boundary_fraction": b_frac,
                            "n_iters_seed_a": int(r1.n_iters)})
    in_band = [r for r in rows
                  if target_boundary[0] <= r["boundary_fraction"] <= target_boundary[1]]
    if in_band:
        winner = max(in_band, key=lambda r: r["ari"])
    else:
        # Fallback: pick the one with the best ARI whose boundary is closest
        # to the mid-band centre.
        centre = 0.5 * (target_boundary[0] + target_boundary[1])
        winner = max(rows, key=lambda r: (r["ari"] - abs(r["boundary_fraction"] - centre)))
    return winner, rows


def run_stage_a(city: str, out_root: Path, args) -> None:
    """L0 → L2, stability ARI, archetypes, examples, 2D PCA scatter.

    If ``--K`` is at the default (6) and ``--eps`` is unset, sweep
    ``K ∈ {4, 6, 8}`` and ``ε ∈ {0.010, 0.020, 0.030, 0.050}`` and pick the
    config with the highest ARI(seed=42, seed=43) whose boundary fraction
    is in [10%, 30%].
    """
    t0 = time.time()
    out_stage = out_root / "situations"
    out_stage.mkdir(parents=True, exist_ok=True)
    log = logger.info if args.verbose else (lambda *a, **k: None)

    print(f"  loading {city} ...")
    ds = _load_city(city)
    print(f"  n_users={ds['n_users']} n_items={ds['n_items']} "
          f"n_macros={ds['n_macros']}")

    print(f"  building perception (n={args.n}, H={args.H}, "
          f"gamma={args.gamma}, beta={args.beta}) ...")
    P = _build_perception(ds, n=args.n, gamma=args.gamma, H=args.H,
                            beta=args.beta, verbose=args.verbose)

    # --- (K, ε) selection ----------------------------------------------------
    do_sweep = (args.K == 6 and args.eps is None)
    if do_sweep:
        print(f"  sweeping K ∈ {{4, 6, 8}} × ε grid for stable, in-band config ...")
        winner, tuning_rows = _tune_K_and_eps(P["train"]["v"], seed_a=args.seed,
                                                seed_b=args.seed + 1)
        K = int(winner["K"]); eps = float(winner["eps"])
        pd.DataFrame(tuning_rows).to_csv(out_stage / "tuning_log.csv", index=False)
        print(f"  → winner: K={K}, ε={eps:.4f}  (ARI={winner['ari']:.3f}, "
              f"boundary={winner['boundary_fraction']:.2%})")
    else:
        K = int(args.K)
        eps = args.eps if args.eps is not None else auto_epsilon(P["train"]["v"], K=K)
        print(f"  using user-specified K={K}, ε={eps:.4f}")

    print(f"  fitting rough k-means: K={K}, ε={eps:.4f}, seed={args.seed}")
    res_train = fit_rough_kmeans(P["train"]["v"], K=K, eps=eps,
                                    seed=args.seed, max_iter=80)
    log(f"    converged in {res_train.n_iters} iters")
    # apply to val / test using a tiny 0-iter "fit" trick: reuse prototypes.
    from .l2_comprehension import _assign
    d_v, k_v, comp_v, isb_v = _assign(P["val"]["v"], res_train.prototypes, eps)
    d_t, k_t, comp_t, isb_t = _assign(P["test"]["v"], res_train.prototypes, eps)
    membership_val = _membership_from_assign(k_v, comp_v, isb_v, K)
    membership_test = _membership_from_assign(k_t, comp_t, isb_t, K)

    # Stability: rerun with a different seed on train, ARI on core labels.
    res_train_b = fit_rough_kmeans(P["train"]["v"], K=K, eps=eps,
                                      seed=args.seed + 1, max_iter=80)
    ari = adjusted_rand_score(res_train.core_label, res_train_b.core_label)
    boundary_frac_train = float(res_train.is_boundary.mean())
    boundary_frac_test = float(isb_t.mean())
    print(f"  ARI(seed={args.seed},seed={args.seed+1}) = {ari:.3f}")
    print(f"  boundary fraction: train={boundary_frac_train:.2%} "
          f"test={boundary_frac_test:.2%}")

    # Dump artefacts: archetypes, examples, 2D PCA
    print(f"  writing artefacts → {out_stage}")
    _save_archetypes_csv(out_stage / "archetypes.csv",
                          res_train.prototypes,
                          res_train.core_label, res_train.is_boundary,
                          P["attributes"], ds["idx_to_macro"],
                          P["attractors"], P["train"]["L0"].cat_target,
                          ds["n_macros"])
    # For examples, use TEST split — that's what the user sees.
    examples_res = type(res_train)(
        prototypes=res_train.prototypes, core_label=k_t.astype(np.int32),
        competing_sets=comp_t, membership=membership_test,
        is_boundary=isb_t, distances=d_t.astype(np.float32),
        n_iters=0, converged=True,
    )
    _save_examples_csv(out_stage / "examples.csv", ds, P, examples_res,
                         split="test", n_per_situation=10)

    # Cluster artefacts as .npy for downstream stages
    np.savez(out_stage / "fit.npz",
              prototypes=res_train.prototypes,
              core_label_train=res_train.core_label,
              membership_train=res_train.membership,
              is_boundary_train=res_train.is_boundary,
              core_label_val=k_v.astype(np.int32),
              membership_val=membership_val,
              is_boundary_val=isb_v,
              core_label_test=k_t.astype(np.int32),
              membership_test=membership_test,
              is_boundary_test=isb_t,
              W=P["W"], attractors=P["attractors"],
              v_train=P["train"]["v"], v_val=P["val"]["v"], v_test=P["test"]["v"])

    # PCA scatter (train + prototypes)
    from .viz import plot_clusters_2d
    X2_train, basis = _pca_2d(P["train"]["v"])
    proto_2d = (res_train.prototypes - P["train"]["v"].mean(axis=0)) @ basis.T
    plot_clusters_2d(X2_train, res_train.core_label, res_train.is_boundary,
                       proto_2d, out_stage / "clusters_2d.png",
                       title=f"{city} — situations (PCA of v, train)")

    # Summary
    summary = {
        "city": city, "stage": "A", "K": int(K), "eps": float(eps),
        "K_swept": do_sweep,
        "n": int(args.n), "H": int(args.H), "gamma": float(args.gamma),
        "beta": float(args.beta),
        "ari_seeds": float(ari),
        "boundary_fraction_train": boundary_frac_train,
        "boundary_fraction_test": boundary_frac_test,
        "attractors": [ds["idx_to_macro"][int(i)]
                        for i in np.where(P["attractors"])[0]],
        "wallclock_s": time.time() - t0,
    }
    (out_stage / "summary.json").write_text(json.dumps(summary, indent=2,
                                                          default=str),
                                              encoding="utf-8")

    success = ari >= 0.6
    print(f"\n>>> Stage A on {city}: "
          f"ARI={ari:.3f}  boundary={boundary_frac_test:.2%}  "
          f"→ {'SUCCESS (stable)' if success else 'KILL (unstable, reconsider K/eps/n)'}")


def _membership_from_assign(k_star: np.ndarray, competing: np.ndarray,
                              is_boundary: np.ndarray, K: int) -> np.ndarray:
    """Build the (B, K) membership matrix from a fresh _assign() output."""
    B = len(k_star)
    out = np.zeros((B, K), dtype=np.float32)
    core_mask = ~is_boundary
    if core_mask.any():
        out[core_mask, k_star[core_mask]] = 1.0
    bnd_idx = np.where(is_boundary)[0]
    if bnd_idx.size:
        T_sizes = competing[bnd_idx].sum(axis=1).astype(np.float32)
        out[bnd_idx] = competing[bnd_idx].astype(np.float32) / T_sizes[:, None]
    return out


# ---------------------------------------------------------------------------
# Placeholders for the remaining stages (filled in as we implement them)
# ---------------------------------------------------------------------------

def run_stage_b(city: str, out_root: Path, args) -> None:
    """Cardinal check 1 — per-situation fairness lens on the backbone's top-K.

    Layers:
        1. Load Stage A artefacts (test situation labels, boundary flag).
        2. Refit (or load cached) FM-vanilla on train+val → full score matrix.
        3. For each test request, build the top-K list under the floor's
           ``exclude_seen=True`` rule (mask the user's train+val history).
        4. Compute per-situation ``LT`` (long-tail ratio, eq.11) and ``KL``
           against the global top-K item distribution (eq.12), separately for
           all / core / boundary requests.
        5. Per-situation available long-tail share (structural control, see
           brief §5 Stage B): the long-tail rate among items the typical user
           in this situation *could* still get (i.e. has not yet seen).
        6. Decision rule: GREEN if at least one situation's KL ≥ 1.5 × global
           AND its top-LT is not fully explained by the structural baseline.

    Outputs:
        fairness/per_situation.csv
        fairness/top_items_by_situation.csv  (top 10 items per situation, for
                                                 inspection)
        fairness/verdict.json
    """
    import json
    from .backbone import excluded_mask, load_or_refit
    from .metrics import (kl_divergence, long_tail_groups, long_tail_ratio,
                           topk_from_scores)

    K_top = 20
    SHORT_HEAD_SHARE = 0.20
    LENS_THRESHOLD_RATIO = 1.5

    t0 = time.time()
    out_stage = out_root / "fairness"
    out_stage.mkdir(parents=True, exist_ok=True)
    sit_dir = out_root / "situations"
    if not (sit_dir / "fit.npz").exists():
        raise FileNotFoundError(
            f"Stage A artefacts missing at {sit_dir / 'fit.npz'} — "
            f"run Stage A first.")

    print(f"  loading Stage A fit + dataset ...")
    fit = np.load(sit_dir / "fit.npz", allow_pickle=True)
    ds = _load_city(city)

    # Backbone scores
    print(f"  loading / refitting backbone (FM-vanilla refit on train+val) ...")
    backbone_scores = load_or_refit(city, model_name="FM", verbose=args.verbose)
    excl = excluded_mask(city, ds["n_items"])

    # Item popularity from train+val (= 1 row sum of the URM)
    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(axis=0)).ravel()
    G0_mask, G1_mask = long_tail_groups(pop, short_head_share=SHORT_HEAD_SHARE)
    print(f"  long-tail split: G0 (head) = {int(G0_mask.sum())} items, "
          f"G1 (tail) = {int(G1_mask.sum())}  ({SHORT_HEAD_SHARE:.0%} cut-off)")

    # Per-request top-K
    df_test = ds["df_test"]
    n_test = len(df_test)
    u_test = df_test["u_idx"].values.astype(np.int32)
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)
    K_sit = int(z_test.max() + 1)
    n_items = int(ds["n_items"])

    print(f"  computing top-{K_top} lists for {n_test} test requests ...")
    top_per_request = np.zeros((n_test, K_top), dtype=np.int32)
    # Cache user-wise excluded indices for speed.
    user_excl: dict[int, np.ndarray] = {}
    for q in range(n_test):
        u = int(u_test[q])
        if u not in user_excl:
            user_excl[u] = excl.indices[excl.indptr[u]:excl.indptr[u + 1]]
        s = backbone_scores[u].copy()
        cols = user_excl[u]
        if len(cols):
            s[cols] = -np.inf
        top_per_request[q] = topk_from_scores(s, K_top)

    # Global reference distribution and global LT
    all_items = top_per_request.flatten()
    global_dist = np.bincount(all_items, minlength=n_items).astype(np.float64)
    global_dist /= max(global_dist.sum(), 1.0)
    global_LT = float(G1_mask[all_items].mean())
    print(f"  global LT(top-{K_top}) across test requests = {global_LT:.3f}")

    # Per-situation LT / KL
    print(f"  per-situation lens (K_sit={K_sit}) ...")
    rows = []
    avail_LT_per_situation = {}
    for k in range(K_sit):
        for split_name, mask in (
            ("all", z_test == k),
            ("core", (z_test == k) & ~isb_test),
            ("boundary", (z_test == k) & isb_test),
        ):
            n_req = int(mask.sum())
            if n_req == 0:
                rows.append({"situation": k, "split": split_name,
                             "n_requests": 0, "LT": None, "KL": None})
                continue
            items = top_per_request[mask].flatten()
            d = np.bincount(items, minlength=n_items).astype(np.float64)
            d /= max(d.sum(), 1.0)
            lt = float(G1_mask[items].mean())
            kl = kl_divergence(d, global_dist)
            rows.append({"situation": k, "split": split_name,
                         "n_requests": n_req, "LT": lt, "KL": kl,
                         "n_items_used": int(len(items))})

        # Structural control: among items the typical user in situation k could
        # still receive (i.e. not in train∪val), what's the long-tail share?
        users_k = np.unique(u_test[z_test == k])
        shares = []
        for u in users_k:
            seen = excl.indices[excl.indptr[u]:excl.indptr[u + 1]]
            allowed = np.ones(n_items, dtype=bool); allowed[seen] = False
            if allowed.any():
                shares.append(float(G1_mask[allowed].mean()))
        avail_LT_per_situation[k] = float(np.mean(shares)) if shares else None

    # Save per-situation table.
    df_rows = pd.DataFrame(rows)
    df_rows["available_LT"] = df_rows["situation"].map(avail_LT_per_situation)
    df_rows["global_LT"] = global_LT
    df_rows["LT_minus_available"] = df_rows["LT"] - df_rows["available_LT"]
    df_rows.to_csv(out_stage / "per_situation.csv", index=False)

    # Top items per situation (inspection).
    top_inspect = []
    for k in range(K_sit):
        items = top_per_request[z_test == k].flatten()
        if not len(items): continue
        c = np.bincount(items, minlength=n_items)
        order = np.argsort(c)[::-1][:10]
        for rank, i in enumerate(order, 1):
            top_inspect.append({"situation": k, "rank_in_situation": rank,
                                "item_idx": int(i),
                                "frequency_in_topK": int(c[i]),
                                "is_long_tail": bool(G1_mask[int(i)]),
                                "popularity_count_in_train_val": int(pop[int(i)])})
    pd.DataFrame(top_inspect).to_csv(out_stage / "top_items_by_situation.csv",
                                       index=False)

    # Verdict — GREEN if any situation's KL ≥ 1.5 × global mean AND that
    # situation's LT exceeds the structural-availability baseline by ≥ 5 pp.
    global_KL_mean = df_rows.loc[df_rows["split"] == "all", "KL"].mean()
    candidates_green = []
    for _, r in df_rows.iterrows():
        if r["split"] != "all" or r["KL"] is None: continue
        kl_ratio = float(r["KL"] / max(global_KL_mean, 1e-9))
        lt_excess = float(r["LT"] - r["available_LT"]) if r["available_LT"] is not None else 0.0
        if kl_ratio >= LENS_THRESHOLD_RATIO and abs(lt_excess) >= 0.05:
            candidates_green.append({
                "situation": int(r["situation"]),
                "KL": float(r["KL"]), "KL_ratio_vs_global": kl_ratio,
                "LT": float(r["LT"]), "available_LT": float(r["available_LT"]),
                "lt_excess_over_available": lt_excess,
            })
    verdict = "GREEN" if candidates_green else "FLAT"
    print(f"\n>>> Stage B on {city}: lens verdict = {verdict}")
    if candidates_green:
        for c in candidates_green:
            print(f"     situation {c['situation']}: "
                  f"KL={c['KL']:.3f} (×{c['KL_ratio_vs_global']:.2f} global)  "
                  f"LT={c['LT']:.3f} vs available={c['available_LT']:.3f}  "
                  f"(excess {c['lt_excess_over_available']:+.3f})")
    else:
        print(f"     per-situation KL hovers around global (max ratio "
              f"{(df_rows.loc[df_rows['split']=='all','KL']/max(global_KL_mean,1e-9)).max():.2f}); "
              "no inequity sink that beats the structural-scarcity control.")

    summary = {
        "city": city, "stage": "B", "K_sit": K_sit, "K_top": K_top,
        "short_head_share": SHORT_HEAD_SHARE,
        "global_LT": global_LT,
        "global_KL_mean_all_split": float(global_KL_mean),
        "verdict": verdict,
        "inequity_sinks": candidates_green,
        "wallclock_s": time.time() - t0,
    }
    (out_stage / "verdict.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

def run_stage_c(city: str, out_root: Path, args) -> None:
    """Cardinal check 2 — projection (L3).

    Estimate the situation transition matrix ``T`` (eq.17) from consecutive
    ``(z_t, z_{t+1})`` pairs in (train ∪ val), use it to predict the next
    situation for each test row (whose predecessor in train ∪ val is its
    ``z_prev``), and compare against a *time-only* baseline that predicts
    ``argmax_k P(z = k | c_hour)`` estimated on (train ∪ val).

    Outputs:
        projection/transition_matrix.csv  +  transition_heatmap.png
        projection/next_situation_f1.csv  (F1 transition vs time-only)
        projection/dynamic_fairness.csv   +  dynamic_fairness.png

    Decision rule (brief §5.C): GREEN if transition-based macro F1 beats the
    time-only prior by a meaningful, paired-significant margin.
    """
    import json
    from .l3_projection import (dynamic_fairness, estimate_transition,
                                  macro_f1, predict_next_situation,
                                  time_only_prior)
    from .viz import plot_dynamic_fairness, plot_transition_heatmap

    t0 = time.time()
    out_stage = out_root / "projection"
    out_stage.mkdir(parents=True, exist_ok=True)

    sit_dir = out_root / "situations"
    fit = np.load(sit_dir / "fit.npz", allow_pickle=True)
    K_sit = int(max(np.asarray(fit["core_label_train"]).max(),
                      np.asarray(fit["core_label_val"]).max(),
                      np.asarray(fit["core_label_test"]).max()) + 1)
    print(f"  K_sit = {K_sit}")

    ds = _load_city(city)
    df_train = ds["df_train"]; df_val = ds["df_val"]; df_test = ds["df_test"]
    z_train = np.asarray(fit["core_label_train"]).astype(np.int32)
    z_val = np.asarray(fit["core_label_val"]).astype(np.int32)
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)

    # --- 1. Build per-user time-ordered z sequences over (train ∪ val) ----
    tv = pd.concat([
        df_train.assign(_z=z_train),
        df_val.assign(_z=z_val),
    ], ignore_index=True)
    tv = tv.sort_values(["user_id", "time_local"]).reset_index(drop=True)
    sequences = []
    for u, g in tv.groupby("user_id", sort=False):
        sequences.append(g["_z"].values.astype(np.int32))
    print(f"  estimated T from {len(sequences)} per-user sequences "
          f"({sum(len(s)-1 for s in sequences if len(s)>1)} pairs)")

    T, raw_counts = estimate_transition(sequences, K=K_sit, add_one_smoothing=True)

    # --- 2. For each test row, find the previous z in (train ∪ val) -------
    by_user_tv: dict[int, dict[str, np.ndarray]] = {}
    for u, g in tv.groupby("user_id", sort=False):
        by_user_tv[int(u)] = {
            "t": g["time_local"].values.astype("datetime64[ns]"),
            "z": g["_z"].values.astype(np.int32),
        }
    n_test = len(df_test)
    z_prev_test = np.full(n_test, -1, dtype=np.int32)
    users_test = df_test["user_id"].values.astype(np.int64)
    times_test = df_test["time_local"].values.astype("datetime64[ns]")
    for q in range(n_test):
        rec = by_user_tv.get(int(users_test[q]))
        if rec is None:
            continue
        cut = np.searchsorted(rec["t"], times_test[q], side="left")
        if cut > 0:
            z_prev_test[q] = rec["z"][cut - 1]
    valid = z_prev_test >= 0
    print(f"  z_prev resolved for {valid.sum()}/{n_test} test requests")

    # --- 3. Predict next situation: transition-based vs time-only --------
    # Transition-based: argmax T[z_prev]
    z_pred_T = predict_next_situation(T, z_prev_test[valid])
    # Time-only: from (train+val) build P(z | hour) and pick argmax
    z_pred_time = time_only_prior(
        z_train=tv["_z"].values.astype(np.int32),
        hour_train=tv["c_hour"].values.astype(np.int32),
        hour_test=df_test.loc[valid, "c_hour"].values.astype(np.int32),
    )
    y_true = z_test[valid]

    f1_T = macro_f1(y_true, z_pred_T, K_sit)
    f1_time = macro_f1(y_true, z_pred_time, K_sit)

    # Paired comparison on per-request accuracy (McNemar-style 2×2)
    correct_T = (z_pred_T == y_true)
    correct_time = (z_pred_time == y_true)
    n10 = int((correct_T & ~correct_time).sum())  # T correct, time wrong
    n01 = int((~correct_T & correct_time).sum())  # T wrong, time correct
    # McNemar with continuity correction
    if n10 + n01 > 0:
        mcnemar_stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    else:
        mcnemar_stat = 0.0
    # Chi-square df=1; p-value via scipy if available, else None.
    try:
        from scipy.stats import chi2
        mcnemar_p = float(1 - chi2.cdf(mcnemar_stat, df=1))
    except Exception:
        mcnemar_p = None

    print(f"  next-situation macro F1: "
          f"T-based={f1_T:.3f}  time-only={f1_time:.3f}  "
          f"Δ={f1_T - f1_time:+.3f}")
    print(f"  paired (McNemar): T-only-correct={n10}  time-only-correct={n01}  "
          f"p={mcnemar_p:.4f}" if mcnemar_p is not None else
          f"  paired (McNemar): T-only-correct={n10}  time-only-correct={n01}")

    # --- 4. Dynamic fairness over τ ∈ {1, 2, 3} --------------------------
    # Need LT per situation from Stage B. Read per_situation.csv if present.
    fair_csv = out_root / "fairness" / "per_situation.csv"
    if fair_csv.exists():
        fair = pd.read_csv(fair_csv)
        fair_all = fair[fair["split"] == "all"].set_index("situation")
        lt_per_situation = np.array(
            [float(fair_all.loc[k, "LT"]) if k in fair_all.index else 0.0
             for k in range(K_sit)],
            dtype=np.float64,
        )
    else:
        print("  (Stage B not run yet — dynamic fairness will use zero LTs)")
        lt_per_situation = np.zeros(K_sit, dtype=np.float64)

    LT_curve = dynamic_fairness(T, lt_per_situation, tau_max=3)
    pd.DataFrame(LT_curve,
                  columns=["tau=1", "tau=2", "tau=3"]).assign(
        starting_situation=range(K_sit)
    ).to_csv(out_stage / "dynamic_fairness.csv", index=False)

    # --- 5. Dump T + heatmap + F1 + plots --------------------------------
    pd.DataFrame(T,
                  columns=[f"to_s{j}" for j in range(K_sit)],
                  index=[f"from_s{i}" for i in range(K_sit)]).to_csv(
        out_stage / "transition_matrix.csv")
    pd.DataFrame(raw_counts,
                  columns=[f"to_s{j}" for j in range(K_sit)],
                  index=[f"from_s{i}" for i in range(K_sit)]).to_csv(
        out_stage / "transition_counts.csv")

    pd.DataFrame({
        "method": ["T-based", "time-only"],
        "macro_F1": [f1_T, f1_time],
        "n_test_evaluated": [int(valid.sum())] * 2,
    }).to_csv(out_stage / "next_situation_f1.csv", index=False)

    plot_transition_heatmap(T, out_stage / "transition_heatmap.png",
                              title=f"{city} — situation transition T")
    plot_dynamic_fairness(LT_curve, out_stage / "dynamic_fairness.png",
                            title=f"{city} — LT̄ vs τ per starting situation")

    # --- 6. Verdict ------------------------------------------------------
    delta = float(f1_T - f1_time)
    meaningful = delta >= 0.02
    significant = (mcnemar_p is not None and mcnemar_p < 0.05) or (n10 > n01 + 5)
    verdict = "GREEN" if (meaningful and significant) else "FLAT"

    summary = {
        "city": city, "stage": "C", "K_sit": K_sit,
        "macro_F1_transition": float(f1_T),
        "macro_F1_time_only": float(f1_time),
        "delta_F1": delta,
        "mcnemar_T_only_correct": n10,
        "mcnemar_time_only_correct": n01,
        "mcnemar_p_value": mcnemar_p,
        "verdict": verdict,
        "n_test_evaluated": int(valid.sum()),
        "wallclock_s": time.time() - t0,
    }
    (out_stage / "verdict.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\n>>> Stage C on {city}: projection verdict = {verdict}  "
          f"(ΔF1={delta:+.3f})")

def _vocab_from_split(df_train: pd.DataFrame, df_val: pd.DataFrame,
                        df_test: pd.DataFrame, col: str) -> dict:
    """Stable string → int mapping over the union of splits."""
    vals = sorted(set(df_train[col]).union(df_val[col]).union(df_test[col]))
    return {v: i for i, v in enumerate(vals)}


def _next_item_metrics_per_user(scores: np.ndarray,
                                  target_items: np.ndarray,
                                  exclude: sps.csr_matrix,
                                  users: np.ndarray,
                                  cutoffs: tuple[int, ...] = (1, 5, 10, 20, 40, 50, 100)
                                  ) -> dict[int, dict[str, np.ndarray]]:
    """For each request (row in ``scores``), compute next-item Recall/NDCG/
    MAP/MRR/Precision at the given cutoffs, then aggregate per user as the
    mean across the user's requests.

    ``scores``       (B, n_items) float
    ``target_items`` (B,) int — the row's target i_idx
    ``exclude``      (n_users, n_items) CSR — per-user items to mask
    ``users``        (B,) int — user index per row (for exclusion + grouping)
    Returns ``{user_ids, RECALL_K, NDCG_K, ...}`` per the standard schema.
    """
    B, I = scores.shape
    per_req = {K: {m: np.zeros(B, dtype=np.float32) for m in ("RECALL", "NDCG", "PRECISION", "MAP", "MRR")}
                for K in cutoffs}
    # Build exclusion-applied scores then rank
    for b in range(B):
        u = int(users[b])
        s = scores[b].copy()
        cols = exclude.indices[exclude.indptr[u]:exclude.indptr[u + 1]]
        if len(cols):
            s[cols] = -np.inf
        target = int(target_items[b])
        # Rank = 1 + count of items strictly above target
        ts = s[target]
        rank = int((s > ts).sum()) + 1
        for K in cutoffs:
            if rank <= K:
                per_req[K]["RECALL"][b] = 1.0
                per_req[K]["NDCG"][b] = 1.0 / np.log2(rank + 1)
                per_req[K]["PRECISION"][b] = 1.0 / K
                per_req[K]["MAP"][b] = 1.0 / rank
                per_req[K]["MRR"][b] = 1.0 / rank
    # Aggregate per user
    uniq = np.unique(users); uniq.sort()
    out: dict[int, dict[str, np.ndarray]] = {
        K: {m: np.zeros(len(uniq), dtype=np.float32) for m in per_req[K]}
        for K in cutoffs
    }
    order = np.argsort(users); sorted_u = users[order]
    bd = np.searchsorted(sorted_u, uniq); bd = np.append(bd, len(users))
    for ui, u in enumerate(uniq):
        rows = order[bd[ui]:bd[ui + 1]]
        for K in cutoffs:
            for m in per_req[K]:
                out[K][m][ui] = per_req[K][m][rows].mean()
    return {"user_ids": uniq.astype(np.int64), "metrics": out}


def _export_npz(out_path: Path, per_user: dict, cutoffs) -> None:
    payload = {"user_ids": per_user["user_ids"]}
    for K in cutoffs:
        for m in ("RECALL", "NDCG", "PRECISION", "MAP", "MRR"):
            payload[f"{m}_{K}"] = per_user["metrics"][K][m]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)


def _changed_topk(scores_off: np.ndarray, scores_on: np.ndarray,
                    target_users: np.ndarray, exclude: sps.csr_matrix,
                    K: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """For each request, return (delta_K, changed_bool).

    Δ_K = 1 − |top_off ∩ top_on| / K (eq.16).
    """
    B = scores_off.shape[0]
    deltas = np.zeros(B, dtype=np.float32)
    changed = np.zeros(B, dtype=bool)
    for b in range(B):
        u = int(target_users[b])
        cols = exclude.indices[exclude.indptr[u]:exclude.indptr[u + 1]]
        s_off = scores_off[b].copy(); s_on = scores_on[b].copy()
        if len(cols):
            s_off[cols] = -np.inf; s_on[cols] = -np.inf
        # argpartition for top-K, then convert to a set
        top_off = np.argpartition(s_off, -K)[-K:]
        top_on = np.argpartition(s_on, -K)[-K:]
        inter = len(np.intersect1d(top_off, top_on, assume_unique=False))
        deltas[b] = 1.0 - inter / K
        changed[b] = inter < K
    return deltas, changed


def run_stage_d(city: str, out_root: Path, args) -> None:
    """Stage D — three-way comparison + modulation log + matched-pair tests.

    Layers (per the brief §5.D):
        1.  Load Stage A artefacts; load (or refit) B_blind scores; train (or
            load) B_full scores. Build per-situation category biases from
            train ∪ val.
        2.  For every κ in the sweep, compute the X-SAGE p̂ via harmonic
            combine (eq.15) per test request, top-K, next-item metrics.
        3.  Δ_K (eq.16) for X-SAGE vs B_blind, separated for core vs
            boundary requests.
        4.  Export per-user .npz (standard schema) for B_blind, B_full, and
            X-SAGE at the best κ on val RECALL@20.
        5.  Matched-pair Wilcoxon (a TOST surrogate) on the per-user arrays:
            X-SAGE vs B_blind, X-SAGE vs B_full.
    """
    import json
    from scipy.stats import wilcoxon
    import torch

    from .backbone import excluded_mask, load_or_refit
    from .backbone_full import (ContextAwareFM, FeatureSpec, _build_request_features,
                                  _catalogue_indices, score_all_per_request, train_b_full)
    from .metrics import topk_from_scores
    from .recommendation import (additive_combine_scores, backbone_confidence,
                                    fit_situation_biases, fit_situation_biases_z,
                                    harmonic_combine, situation_confidence,
                                    situational_item_scores, softmax_scores)
    from .viz import plot_modulation_summary

    combiner = getattr(args, "combiner", "harmonic")
    t0 = time.time()
    out_stage = out_root / (
        "recommendation" if combiner == "harmonic"
        else f"recommendation_{combiner}"
    )
    out_stage.mkdir(parents=True, exist_ok=True)
    print(f"  combiner = {combiner}")

    sit_dir = out_root / "situations"
    fit = np.load(sit_dir / "fit.npz", allow_pickle=True)
    ds = _load_city(city)
    df_train = ds["df_train"]; df_val = ds["df_val"]; df_test = ds["df_test"]
    n_items = ds["n_items"]; n_macros = ds["n_macros"]
    macro_to_idx = ds["macro_to_idx"]

    # --- backbones ----------------------------------------------------------
    print(f"  loading / refitting B_blind (FM-vanilla) on {city} ...")
    scores_blind_uitem = load_or_refit(city, model_name="FM", verbose=args.verbose)
    # Broadcast user-level scores to per-request rows
    n_test = len(df_test); u_test = df_test["u_idx"].values.astype(np.int64)
    scores_blind_per_req = scores_blind_uitem[u_test]                  # (n_test, n_items)

    excl = excluded_mask(city, n_items)

    # --- B_full -------------------------------------------------------------
    bfull_dir = out_root / "backbone"
    bfull_scores_path = bfull_dir / "Bfull.scores.npy"
    if bfull_scores_path.exists():
        print(f"  using cached B_full scores: {bfull_scores_path}")
        scores_full_per_req = np.load(bfull_scores_path)
    else:
        print(f"  training B_full (context-aware FM) on {city} ...")
        # Build vocabularies
        fine_to_idx = _vocab_from_split(df_train, df_val, df_test, "cat_fine")
        # Geohash5: derived columns guarantee 'prev_geohash5' exists; sentinel = '__NONE__'.
        prev_vals = sorted(set(df_train["prev_geohash5"]).union(df_val["prev_geohash5"])
                            .union(df_test["prev_geohash5"]))
        # index 0 is reserved for "__NONE__"; assign others 1..G
        prev_vals_clean = [v for v in prev_vals if v != "__NONE__"]
        geo_to_idx = {"__NONE__": 0,
                        **{v: i + 1 for i, v in enumerate(prev_vals_clean)}}
        n_geo = len(prev_vals_clean)
        spec = FeatureSpec(n_users=ds["n_users"], n_items=n_items,
                            n_macros=n_macros, n_fine=len(fine_to_idx),
                            n_geo=n_geo, n_intent_last=n_macros)
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        torch.manual_seed(args.seed)
        model = ContextAwareFM(spec, d=64).to(device)
        # Train on train ∪ val
        df_tv = pd.concat([df_train, df_val], ignore_index=True)
        df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
        feats_tv = _build_request_features(df_tv, spec, macro_to_idx,
                                              fine_to_idx, geo_to_idx)
        macro_per_item, fine_per_item = _catalogue_indices(df_all, spec,
                                                              macro_to_idx, fine_to_idx)
        mask = (ds["urm_train"] + ds["urm_val"]).tocsr(); mask.data[:] = 1.0
        rep = train_b_full(model, feats_tv, mask, macro_per_item, fine_per_item,
                            device=device, n_epochs=10, batch_size=4096,
                            seed=args.seed, verbose=args.verbose)
        print(f"    B_full trained in {rep['wallclock_s']:.1f}s")
        # Score all test requests
        feats_test = _build_request_features(df_test, spec, macro_to_idx,
                                                fine_to_idx, geo_to_idx)
        print(f"    scoring B_full on {n_test} test requests ...")
        scores_full_per_req = score_all_per_request(
            model, feats_test, macro_per_item, fine_per_item,
            device=device, batch_size=128,
        )
        bfull_dir.mkdir(parents=True, exist_ok=True)
        np.save(bfull_scores_path, scores_full_per_req)

    # --- X-SAGE: per-situation biases + harmonic combine -------------------
    z_train = np.asarray(fit["core_label_train"]).astype(np.int32)
    z_val = np.asarray(fit["core_label_val"]).astype(np.int32)
    K_sit = int(max(z_train.max(), z_val.max(),
                      np.asarray(fit["core_label_test"]).max()) + 1)
    z_tv = np.concatenate([z_train, z_val])
    macro_train = np.array([macro_to_idx[c] for c in df_train["cat_macro"].values],
                            dtype=np.int64)
    macro_val = np.array([macro_to_idx[c] for c in df_val["cat_macro"].values],
                          dtype=np.int64)
    macro_tv = np.concatenate([macro_train, macro_val])
    biases = fit_situation_biases(z_tv, macro_tv, K=K_sit,
                                     n_macros=n_macros, lam=50.0)
    biases_z = fit_situation_biases_z(z_tv, macro_tv, K=K_sit,
                                          n_macros=n_macros, lam=50.0)
    # Item → cat_macro for situational scoring
    item_macro = ds["urm_train"].copy()  # placeholder shape — we want a vector
    # Use the build function: catalogue from train+val
    # Reuse the helper from data assembly:
    macro_per_item_xsage = _build_item_cat_macro_local(df_train, df_val, df_test,
                                                          n_items, macro_to_idx)

    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)
    membership_test = np.asarray(fit["membership_test"]).astype(np.float32)
    gamma_per_request = np.where(isb_test, 1.0 / np.maximum(
        (membership_test > 0).sum(axis=1), 1), 1.0).astype(np.float32)

    # softmax distributions (used by the harmonic combiner)
    p_B = softmax_scores(scores_blind_per_req, tau=1.0)
    s_S = situational_item_scores(membership_test, biases, macro_per_item_xsage)
    p_S = softmax_scores(s_S, tau=1.0)
    c_B = backbone_confidence(p_B)

    cutoffs = (1, 5, 10, 20, 40, 50, 100)
    i_target = df_test["i_idx"].values.astype(np.int64)

    # --- κ sweep ----------------------------------------------------------
    rows_three_way: list[dict] = []
    matched_pair_rows: list[dict] = []
    print(f"  computing matched-pair metrics for κ ∈ {args.kappa} ...")
    # Pre-compute B_blind matched-pair metrics (once)
    res_blind = _next_item_metrics_per_user(scores_blind_per_req, i_target, excl,
                                              u_test, cutoffs=cutoffs)
    res_full = _next_item_metrics_per_user(scores_full_per_req, i_target, excl,
                                              u_test, cutoffs=cutoffs)
    _export_npz(out_stage / "Bblind.npz", res_blind, cutoffs)
    _export_npz(out_stage / "Bfull.npz", res_full, cutoffs)

    rows_three_way.append({
        "variant": "B_blind",
        "R_at_20": float(res_blind["metrics"][20]["RECALL"].mean()),
        "N_at_20": float(res_blind["metrics"][20]["NDCG"].mean()),
        "kappa": "—",
    })

    # X-SAGE sweep
    best_kappa = None; best_metric = -1.0
    xsage_per_kappa: dict[float, np.ndarray] = {}    # κ → scores per-request
    res_xsage_by_kappa: dict[float, dict] = {}
    for kappa in args.kappa:
        if kappa == 0.0:
            # Matched OFF: exactly = backbone. Just reuse.
            res_xsage = res_blind
            scores_xsage = scores_blind_per_req
        elif combiner == "harmonic":
            c_S = situation_confidence(kappa, gamma_per_request)
            p_hat = harmonic_combine(p_B, p_S, c_B, c_S)
            scores_xsage = p_hat
            res_xsage = _next_item_metrics_per_user(scores_xsage, i_target,
                                                       excl, u_test,
                                                       cutoffs=cutoffs)
        elif combiner == "additive":
            scores_xsage = additive_combine_scores(
                scores_blind_per_req, membership_test, biases_z,
                macro_per_item_xsage, kappa=kappa,
                gamma_per_request=gamma_per_request,
            )
            res_xsage = _next_item_metrics_per_user(scores_xsage, i_target,
                                                       excl, u_test,
                                                       cutoffs=cutoffs)
        else:
            raise ValueError(f"Unknown combiner: {combiner!r}")
        xsage_per_kappa[kappa] = scores_xsage
        res_xsage_by_kappa[kappa] = res_xsage
        r20 = float(res_xsage["metrics"][20]["RECALL"].mean())
        n20 = float(res_xsage["metrics"][20]["NDCG"].mean())
        rows_three_way.append({"variant": "X-SAGE", "R_at_20": r20,
                                  "N_at_20": n20, "kappa": kappa})
        if r20 > best_metric:
            best_metric = r20; best_kappa = kappa
        print(f"    κ={kappa:.2f}  R@20={r20:.4f}  N@20={n20:.4f}")

    rows_three_way.append({
        "variant": "B_full",
        "R_at_20": float(res_full["metrics"][20]["RECALL"].mean()),
        "N_at_20": float(res_full["metrics"][20]["NDCG"].mean()),
        "kappa": "—",
    })

    print(f"  best X-SAGE κ = {best_kappa} (R@20 = {best_metric:.4f})")
    # Export X-SAGE at best κ
    _export_npz(out_stage / f"XSAGE_kappa{best_kappa}.npz",
                  res_xsage_by_kappa[best_kappa], cutoffs)

    # --- matched-pair tests ------------------------------------------------
    def _paired_wilcoxon(a: np.ndarray, b: np.ndarray):
        if np.allclose(a, b):
            return {"stat": 0.0, "p": 1.0}
        try:
            res = wilcoxon(a, b, zero_method="pratt", alternative="two-sided")
            return {"stat": float(res.statistic), "p": float(res.pvalue)}
        except Exception as e:
            return {"stat": None, "p": None, "error": str(e)}

    r_blind = res_blind["metrics"][20]["RECALL"]
    r_full = res_full["metrics"][20]["RECALL"]
    n_blind = res_blind["metrics"][20]["NDCG"]
    n_full = res_full["metrics"][20]["NDCG"]
    matched_pair_rows.append({"pair": "X-SAGE_vs_B_blind",
                                 "delta_R20": best_metric - rows_three_way[0]["R_at_20"],
                                 **_paired_wilcoxon(res_xsage_by_kappa[best_kappa]["metrics"][20]["RECALL"],
                                                       r_blind)})
    matched_pair_rows.append({"pair": "X-SAGE_vs_B_full",
                                 "delta_R20": best_metric - rows_three_way[-1]["R_at_20"],
                                 **_paired_wilcoxon(res_xsage_by_kappa[best_kappa]["metrics"][20]["RECALL"],
                                                       r_full)})

    # --- modulation log ---------------------------------------------------
    print(f"  building modulation log for the κ sweep ...")
    mod_summary: list[dict] = []
    for kappa in args.kappa:
        scores_on = xsage_per_kappa[kappa]
        if kappa == 0.0:
            deltas = np.zeros(n_test, dtype=np.float32)
            changed = np.zeros(n_test, dtype=bool)
        else:
            deltas, changed = _changed_topk(scores_blind_per_req, scores_on,
                                              u_test, excl, K=20)
        frac_core = float(changed[~isb_test].mean()) if (~isb_test).any() else 0.0
        frac_bnd = float(changed[isb_test].mean()) if isb_test.any() else 0.0
        mod_summary.append({"kappa": kappa,
                              "frac_changed_core": frac_core,
                              "frac_changed_boundary": frac_bnd,
                              "n_changed_core_abs": int(changed[~isb_test].sum()),
                              "n_changed_boundary_abs": int(changed[isb_test].sum()),
                              "delta_K_mean_core": float(deltas[~isb_test].mean()),
                              "delta_K_mean_boundary": float(deltas[isb_test].mean())})

    pd.DataFrame(rows_three_way).to_csv(out_stage / "three_way.csv", index=False)
    pd.DataFrame(matched_pair_rows).to_csv(out_stage / "matched_pair.csv", index=False)
    pd.DataFrame(mod_summary).to_csv(out_stage / "modulation_summary.csv", index=False)

    # Modulation log per request at best κ
    if best_kappa and best_kappa > 0.0:
        deltas, changed = _changed_topk(scores_blind_per_req,
                                          xsage_per_kappa[best_kappa],
                                          u_test, excl, K=20)
        mod_log_df = pd.DataFrame({
            "u_idx": u_test, "time_local": df_test["time_local"].values,
            "z": z_test, "is_boundary": isb_test,
            "kappa": best_kappa,
            "delta_K": deltas,
            "changed_topK": changed,
        })
        mod_log_df.to_csv(out_stage / "modulation_log.csv", index=False)

    plot_modulation_summary(
        np.array([m["kappa"] for m in mod_summary], dtype=np.float32),
        np.array([m["frac_changed_core"] for m in mod_summary]),
        np.array([m["frac_changed_boundary"] for m in mod_summary]),
        out_stage / "modulation_summary.png",
        title=f"{city} — fraction of top-K changed vs B_blind by κ_S",
    )

    summary = {
        "city": city, "stage": "D", "K_sit": K_sit,
        "best_kappa": best_kappa, "best_R20": best_metric,
        "B_blind_R20": rows_three_way[0]["R_at_20"],
        "B_full_R20": rows_three_way[-1]["R_at_20"],
        "matched_pair_tests": matched_pair_rows,
        "modulation_summary": mod_summary,
        "wallclock_s": time.time() - t0,
    }
    (out_stage / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                              encoding="utf-8")
    print(f"\n>>> Stage D on {city} — three-way at best κ={best_kappa}:")
    print(f"     B_blind R@20 = {rows_three_way[0]['R_at_20']:.4f}")
    print(f"     X-SAGE  R@20 = {best_metric:.4f}  "
          f"(Δ vs blind = {best_metric - rows_three_way[0]['R_at_20']:+.4f})")
    print(f"     B_full  R@20 = {rows_three_way[-1]['R_at_20']:.4f}  "
          f"(Δ vs blind = {rows_three_way[-1]['R_at_20'] - rows_three_way[0]['R_at_20']:+.4f})")


def _build_item_cat_macro_local(df_train, df_val, df_test, n_items, macro_to_idx):
    big = pd.concat([df_train, df_val, df_test], ignore_index=True)
    c = big.groupby(["i_idx", "cat_macro"]).size().reset_index(name="n")
    best = c.sort_values(["i_idx", "n"], ascending=[True, False]) \
             .drop_duplicates("i_idx", keep="first")
    out = np.zeros(n_items, dtype=np.int64)
    for _, r in best.iterrows():
        out[int(r["i_idx"])] = macro_to_idx[r["cat_macro"]]
    return out

def run_stage_e(city: str, out_root: Path, args) -> None:
    raise NotImplementedError("Stage E is optional and not yet implemented.")
