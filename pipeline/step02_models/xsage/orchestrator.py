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
    raise NotImplementedError("Stage B will land in commit X7.")

def run_stage_c(city: str, out_root: Path, args) -> None:
    raise NotImplementedError("Stage C will land in commit X8.")

def run_stage_d(city: str, out_root: Path, args) -> None:
    raise NotImplementedError("Stage D will land in commit X10.")

def run_stage_e(city: str, out_root: Path, args) -> None:
    raise NotImplementedError("Stage E is optional and not yet implemented.")
