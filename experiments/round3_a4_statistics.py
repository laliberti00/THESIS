"""Round-3 A4 — statistics package.

Delivers:
  1. TOST equivalence (δ = 0.005 absolute R@20, sensitivity δ = 0.0025) for:
       - matched-OFF parity:        X-SAGE@κ=0  vs  B_blind  (sanity)
       - X-SAGE@κ=0.1 (additive) vs B_blind
  2. User-level bootstrap CIs (B = 10 000) on every per-user mean used in
     the round-2 / round-3 headlines.
  3. step04 wiring on the three-way per-user triple {B_blind, X-SAGE@best,
     B_full} per city — uses the heritage Wilcoxon + HMP + Holm functions
     directly.
  4. A1 operating-point adjudication: bootstrap CI on test ΔR@20 at the
     val-selected κ = 4 sink re-rank.

Outputs:
  outputs/<city>/xsage/round3/A4/tost.json
  outputs/<city>/xsage/round3/A4/bootstrap_ci.json
  outputs/<city>/xsage/round3/A4/a1_oppoint_adjudication.json
  outputs/<city>/xsage/round3/A4/step04_threeway/{primary.tsv, descriptive.tsv}
  outputs/<city>/xsage/round3/A4/verdict.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Heritage step04 functions
from pipeline.step04_statistical_validation.statistical_validation import (
    align_two, harmonic_mean_p, holm_correction, load_per_user_npz,
    paired_wilcoxon, percentile_bootstrap_ci,
)

from pipeline.step02_models.xsage.backbone import excluded_mask, load_or_refit
from pipeline.step02_models.xsage.metrics import long_tail_groups, topk_from_scores
from pipeline.step02_models.xsage.orchestrator import _load_city
from pipeline.step02_models.xsage.recommendation import (
    additive_combine_scores, fit_situation_biases_z, softmax_scores,
)


K_TOP = 20
SHORT_HEAD = 0.20
B_BOOT = 10_000
SEED = 42
DELTA_PRIMARY = 0.005
DELTA_SENSITIVITY = 0.0025
ALPHA = 0.05
KAPPA_OP = 4.0   # A1 val-selected operating point


# ===========================================================================
# Helpers
# ===========================================================================

def _tost(diffs: np.ndarray, delta: float, alpha: float = ALPHA) -> dict:
    """Two one-sided t-tests (paired). H0: |mean diff| ≥ δ. Equivalence
    claimed at level α if BOTH one-sided p-values < α."""
    n = len(diffs)
    if n < 2:
        return {"n": int(n), "mean_diff": float(diffs.mean()),
                  "p_lower": 1.0, "p_upper": 1.0, "equivalent": False,
                  "ci90": [None, None]}
    sd = float(diffs.std(ddof=1))
    se = sd / np.sqrt(n)
    if se == 0:
        # all diffs identical → equivalence iff |diff| < δ
        d = float(diffs.mean())
        return {"n": int(n), "mean_diff": d, "p_lower": 0.0 if d > -delta else 1.0,
                  "p_upper": 0.0 if d < +delta else 1.0,
                  "equivalent": abs(d) < delta,
                  "ci90": [d, d]}
    df = n - 1
    t_lower = (diffs.mean() - (-delta)) / se
    t_upper = (diffs.mean() - (+delta)) / se
    p_lower = float(1 - stats.t.cdf(t_lower, df))      # one-sided H1: > -δ
    p_upper = float(stats.t.cdf(t_upper, df))           # one-sided H1: < +δ
    equivalent = bool((p_lower < alpha) and (p_upper < alpha))
    # 90 % CI (because TOST corresponds to 1-2α CI overlap with [-δ, δ])
    t_crit = float(stats.t.ppf(1 - alpha, df))
    ci_lo = float(diffs.mean() - t_crit * se)
    ci_hi = float(diffs.mean() + t_crit * se)
    return {"n": int(n), "mean_diff": float(diffs.mean()),
              "sd_diff": sd, "se_diff": float(se),
              "p_lower": p_lower, "p_upper": p_upper,
              "equivalent": equivalent,
              "ci90": [ci_lo, ci_hi]}


def _bootstrap_mean_ci(x: np.ndarray, B: int = B_BOOT,
                          seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        means[b] = x[idx].mean()
    return {"n": int(n),
              "mean": float(x.mean()),
              "ci95_lo": float(np.quantile(means, 0.025)),
              "ci95_hi": float(np.quantile(means, 0.975))}


def _bootstrap_diff_ci(x: np.ndarray, y: np.ndarray,
                          B: int = B_BOOT, seed: int = SEED) -> dict:
    """Paired bootstrap on the per-user differences x - y."""
    assert len(x) == len(y)
    diffs = x - y
    return _bootstrap_mean_ci(diffs, B=B, seed=seed)


# ===========================================================================
# Re-compute per-user arrays for specific κ
# ===========================================================================

def _topk(scores: np.ndarray, exclude: sps.csr_matrix, users: np.ndarray,
            K: int = K_TOP) -> np.ndarray:
    n = scores.shape[0]
    out = np.zeros((n, K), dtype=np.int32)
    for b in range(n):
        u = int(users[b])
        s = scores[b].copy()
        cols = exclude.indices[exclude.indptr[u]:exclude.indptr[u + 1]]
        if len(cols):
            s[cols] = -np.inf
        out[b] = topk_from_scores(s, K)
    return out


def _per_user_r20_n20(scores: np.ndarray, targets: np.ndarray,
                          exclude: sps.csr_matrix, users: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_req = scores.shape[0]
    per_req_r = np.zeros(n_req, dtype=np.float32)
    per_req_n = np.zeros(n_req, dtype=np.float32)
    for b in range(n_req):
        u = int(users[b])
        s = scores[b].copy()
        cols = exclude.indices[exclude.indptr[u]:exclude.indptr[u + 1]]
        if len(cols):
            s[cols] = -np.inf
        target = int(targets[b])
        ts = s[target]
        rank = int((s > ts).sum()) + 1
        if rank <= K_TOP:
            per_req_r[b] = 1.0
            per_req_n[b] = 1.0 / np.log2(rank + 1)
    # Aggregate per user
    uniq = np.unique(users); uniq.sort()
    out_r = np.zeros(len(uniq), dtype=np.float32)
    out_n = np.zeros(len(uniq), dtype=np.float32)
    order = np.argsort(users); sorted_u = users[order]
    bd = np.searchsorted(sorted_u, uniq); bd = np.append(bd, len(users))
    for ui in range(len(uniq)):
        rows = order[bd[ui]:bd[ui + 1]]
        out_r[ui] = per_req_r[rows].mean()
        out_n[ui] = per_req_n[rows].mean()
    return out_r, out_n, uniq.astype(np.int64)


def _additive_for_kappa(city: str, ds: dict, fit: np.ndarray,
                            scores_blind_per_req: np.ndarray,
                            kappa: float, scope: str = "all") -> np.ndarray:
    """Apply additive combiner (round-2 1.5) for the given κ.

    ``scope`` selects WHERE the nudge is applied:
        "all"           : apply to all requests with γ_S = 1 on core, 1/|T|
                          on boundary (the original X-SAGE additive)
        "sink_core"     : apply only to core requests of Stage-B-flagged sinks
                          with boost_LT = 1[i ∈ G_1] (the round-2 2.1 fairness
                          re-ranking, used for the A1 op-point adjudication)
    """
    n_macros = ds["n_macros"]; n_items = ds["n_items"]
    macro_to_idx = ds["macro_to_idx"]
    z_train = np.asarray(fit["core_label_train"]).astype(np.int32)
    z_val = np.asarray(fit["core_label_val"]).astype(np.int32)
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)
    membership_test = np.asarray(fit["membership_test"]).astype(np.float32)
    K_sit = int(max(z_train.max(), z_val.max(), z_test.max()) + 1)

    if scope == "all":
        # X-SAGE additive (round-2 1.5)
        macro_train = np.array([macro_to_idx[c] for c in ds["df_train"]["cat_macro"]],
                                dtype=np.int64)
        macro_val = np.array([macro_to_idx[c] for c in ds["df_val"]["cat_macro"]],
                              dtype=np.int64)
        z_tv = np.concatenate([z_train, z_val])
        macro_tv = np.concatenate([macro_train, macro_val])
        biases_z = fit_situation_biases_z(z_tv, macro_tv, K=K_sit,
                                              n_macros=n_macros, lam=50.0)
        big = pd.concat([ds["df_train"], ds["df_val"], ds["df_test"]],
                          ignore_index=True)
        c = big.groupby(["i_idx", "cat_macro"]).size().reset_index(name="n")
        best = c.sort_values(["i_idx", "n"], ascending=[True, False]) \
                 .drop_duplicates("i_idx", keep="first")
        item_macro = np.zeros(n_items, dtype=np.int64)
        for _, r in best.iterrows():
            item_macro[int(r["i_idx"])] = macro_to_idx[r["cat_macro"]]
        gamma = np.where(isb_test,
                            1.0 / np.maximum((membership_test > 0).sum(axis=1), 1),
                            1.0).astype(np.float32)
        return additive_combine_scores(
            scores_blind_per_req, membership_test, biases_z,
            item_macro, kappa=kappa, gamma_per_request=gamma,
        )
    elif scope == "sink_core":
        # Round-2 2.1 fairness re-rank using val-flagged sinks from A1
        a1 = json.loads(
            (REPO_ROOT / "outputs" / city / "xsage" / "round3" / "A1"
              / "sink_comparison.json").read_text())
        sinks = a1["val_flagged"]
        sink_mask = np.isin(z_test, sinks)
        core_sink = sink_mask & ~isb_test
        pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(axis=0)).ravel()
        _, G1_mask = long_tail_groups(pop, short_head_share=SHORT_HEAD)
        boost = G1_mask.astype(np.float32)
        nudge = np.zeros_like(scores_blind_per_req)
        idx = np.where(core_sink)[0]
        nudge[idx] = kappa * boost[None, :]
        return scores_blind_per_req + nudge
    raise ValueError(scope)


# ===========================================================================
# Step04 wiring
# ===========================================================================

def step04_threeway(city: str, npz_paths: dict[str, Path],
                       out_dir: Path) -> dict:
    """Run the heritage Wilcoxon + HMP + Holm pipeline on the {B_blind,
    X-SAGE@best, B_full} triple. We use the heritage functions directly
    rather than the CLI, so the comparison set is exactly the three
    pairwise tests we care about (no placeholder surrogates).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pivot = "X-SAGE"
    pu = {}
    for name, p in npz_paths.items():
        pu[name] = load_per_user_npz(p, model=name, dataset=city)

    rows_descr = []; rows_test = []; pvals = []; labels = []
    for cutoff in (20,):
        for metric in ("RECALL", "NDCG"):
            for name, r in pu.items():
                arr = r.arrays.get((metric, cutoff))
                if arr is None: continue
                ci = percentile_bootstrap_ci(arr)
                rows_descr.append({
                    "city": city, "model": name, "metric": metric,
                    "cutoff": cutoff, "n": len(arr),
                    "mean": float(np.mean(arr)),
                    "ci95_lo": float(ci[0]) if isinstance(ci, tuple) else None,
                    "ci95_hi": float(ci[1]) if isinstance(ci, tuple) else None,
                })
            opp_names = [n for n in pu if n != pivot]
            for opp in opp_names:
                a, b = align_two(pu[pivot], pu[opp], metric, cutoff)
                w = paired_wilcoxon(a, b)
                rows_test.append({
                    "city": city, "comparison": f"{pivot}_vs_{opp}",
                    "metric": metric, "cutoff": cutoff,
                    "n": len(a),
                    "mean_diff": float(a.mean() - b.mean()),
                    "wilcoxon_stat": w.get("statistic", w.get("stat")),
                    "wilcoxon_p": w.get("p_value", w.get("p")),
                })
                pvals.append(w.get("p_value", w.get("p", 1.0)))
                labels.append(f"{city}|{pivot}_vs_{opp}|{metric}@{cutoff}")
    holm_df = holm_correction(pvals, labels, alpha=ALPHA)
    for i, r in enumerate(rows_test):
        r["holm_p"] = float(holm_df.iloc[i]["p_holm"])
        r["holm_reject"] = bool(holm_df.iloc[i].get("reject", False))
    pd.DataFrame(rows_descr).to_csv(out_dir / "descriptive.tsv", sep="\t",
                                       index=False)
    pd.DataFrame(rows_test).to_csv(out_dir / "primary.tsv", sep="\t",
                                      index=False)
    return {"descriptive": rows_descr, "tests": rows_test,
              "holm_alpha": ALPHA}


# ===========================================================================
# Per-city driver
# ===========================================================================

def run_city(city: str, verbose: bool = False) -> dict:
    print(f"\n>>> A4 on {city}")
    out_dir = REPO_ROOT / "outputs" / city / "xsage" / "round3" / "A4"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = _load_city(city)
    sit_dir = REPO_ROOT / "outputs" / city / "xsage" / "situations"
    fit = np.load(sit_dir / "fit.npz", allow_pickle=True)

    # Backbone scores
    scores_blind_uitem = load_or_refit(city, model_name="FM")
    u_test = ds["df_test"]["u_idx"].values.astype(np.int64)
    i_test = ds["df_test"]["i_idx"].values.astype(np.int64)
    scores_blind = scores_blind_uitem[u_test]
    excl_test = (ds["urm_train"] + ds["urm_val"]).tocsr(); excl_test.data[:] = 1.0
    excl_val = ds["urm_train"].tocsr(); excl_val.data[:] = 1.0
    n_items = ds["n_items"]

    # Per-user arrays for the three core variants (using cached npz)
    rec_dir = REPO_ROOT / "outputs" / city / "xsage" / "recommendation_additive"
    npz = {
        "B_blind": rec_dir / "Bblind.npz",
        "B_full": rec_dir / "Bfull.npz",
        "X-SAGE": rec_dir / "XSAGE_kappa0.0.npz",   # = B_blind by construction
    }
    print(f"  per-user arrays loaded from {rec_dir}")

    # === 1. TOST equivalence ============================================
    print(f"  computing TOST (matched-OFF + X-SAGE κ=0.1) ...")
    pu_b = load_per_user_npz(npz["B_blind"], model="B_blind", dataset=city)
    pu_x0 = load_per_user_npz(npz["X-SAGE"], model="X-SAGE@kappa=0", dataset=city)
    a, b = align_two(pu_b, pu_x0, "RECALL", 20)
    tost_matchedoff_r20 = _tost(a - b, DELTA_PRIMARY)
    tost_matchedoff_r20_sens = _tost(a - b, DELTA_SENSITIVITY)
    a, b = align_two(pu_b, pu_x0, "NDCG", 20)
    tost_matchedoff_n20 = _tost(a - b, DELTA_PRIMARY)

    # Recompute X-SAGE @ κ=0.1 (additive)
    print(f"  recomputing X-SAGE@κ=0.1 per-user metrics ...")
    s_kappa01 = _additive_for_kappa(city, ds, fit, scores_blind, 0.1, "all")
    r01, n01, u_per = _per_user_r20_n20(s_kappa01, i_test, excl_test, u_test)
    r_b, n_b, _ = _per_user_r20_n20(scores_blind, i_test, excl_test, u_test)
    tost_xsage01_r20 = _tost(r01 - r_b, DELTA_PRIMARY)
    tost_xsage01_r20_sens = _tost(r01 - r_b, DELTA_SENSITIVITY)
    tost_xsage01_n20 = _tost(n01 - n_b, DELTA_PRIMARY)

    (out_dir / "tost.json").write_text(json.dumps({
        "delta_primary": DELTA_PRIMARY,
        "delta_sensitivity": DELTA_SENSITIVITY,
        "matched_off_R20": tost_matchedoff_r20,
        "matched_off_R20_sensitivity": tost_matchedoff_r20_sens,
        "matched_off_N20": tost_matchedoff_n20,
        "xsage_kappa01_vs_blind_R20": tost_xsage01_r20,
        "xsage_kappa01_vs_blind_R20_sensitivity": tost_xsage01_r20_sens,
        "xsage_kappa01_vs_blind_N20": tost_xsage01_n20,
    }, indent=2), encoding="utf-8")
    print(f"    matched-OFF R@20 mean diff={tost_matchedoff_r20['mean_diff']:+.5f} "
          f"equivalent={tost_matchedoff_r20['equivalent']}")
    print(f"    X-SAGE@0.1 R@20 mean diff={tost_xsage01_r20['mean_diff']:+.5f} "
          f"equivalent={tost_xsage01_r20['equivalent']}")

    # === 2. Bootstrap CIs ==============================================
    print(f"  bootstrap CIs (B={B_BOOT}) ...")
    pu_full = load_per_user_npz(npz["B_full"], model="B_full", dataset=city)
    a_full, b_full = align_two(pu_full, pu_b, "RECALL", 20)
    bootstrap = {
        "B_blind_R20": _bootstrap_mean_ci(r_b),
        "B_blind_N20": _bootstrap_mean_ci(n_b),
        "B_full_R20": _bootstrap_mean_ci(np.asarray(a_full)),
        "B_full_minus_B_blind_R20": _bootstrap_diff_ci(np.asarray(a_full),
                                                            np.asarray(b_full)),
        "X-SAGE@0.1_R20": _bootstrap_mean_ci(r01),
        "X-SAGE@0.1_minus_B_blind_R20": _bootstrap_diff_ci(r01, r_b),
    }

    # === 3. A1 op-point adjudication ===================================
    print(f"  A1 op-point adjudication @ κ={KAPPA_OP} ...")
    s_op = _additive_for_kappa(city, ds, fit, scores_blind, KAPPA_OP, "sink_core")
    r_op, n_op, _ = _per_user_r20_n20(s_op, i_test, excl_test, u_test)
    diff_op = r_op - r_b
    op_boot = _bootstrap_mean_ci(diff_op)
    covers_zero = (op_boot["ci95_lo"] <= 0.0 <= op_boot["ci95_hi"])
    op_decision = ("noise — val rule stands"
                      if covers_zero else
                      "selection rule overfits — switch to val knee-rule")
    a1_adjud = {
        "city": city,
        "kappa_op_val_rule": KAPPA_OP,
        "mean_test_delta_R20_per_user_at_val_kappa": float(diff_op.mean()),
        "bootstrap_ci95_at_val_kappa": op_boot,
        "ci_covers_zero_at_val_kappa": bool(covers_zero),
        "decision": op_decision,
    }
    print(f"    val-rule κ={KAPPA_OP}  test ΔR@20={float(diff_op.mean()):+.4f}  "
          f"CI95=[{op_boot['ci95_lo']:+.4f}, {op_boot['ci95_hi']:+.4f}]  "
          f"covers 0? {covers_zero}")

    # Knee-rule recompute (always done for transparency; binding only when
    # the val rule's CI excludes 0).
    a1_csv = REPO_ROOT / "outputs" / city / "xsage" / "round3" / "A1" / "reranking_tradeoff_valflag.csv"
    tr = pd.read_csv(a1_csv).sort_values("kappa_fair").reset_index(drop=True)
    # Knee on the val (sink_LT_on_val, r20_val_delta) frontier: maximum
    # perpendicular distance from the line joining the two endpoints
    # (Kneedle, simplified).
    xs = tr["sink_LT_on_val"].values
    ys = tr["r20_val_delta"].values
    if len(xs) >= 3 and xs[-1] != xs[0]:
        p0 = np.array([xs[0], ys[0]]); p1 = np.array([xs[-1], ys[-1]])
        seg = p1 - p0; seg /= np.linalg.norm(seg)
        dists = []
        for i in range(len(xs)):
            pt = np.array([xs[i], ys[i]])
            v = pt - p0
            proj = np.dot(v, seg)
            perp = v - proj * seg
            dists.append(np.linalg.norm(perp))
        knee_idx = int(np.argmax(dists))
        knee_kappa = float(tr.loc[knee_idx, "kappa_fair"])
    else:
        knee_idx = -1; knee_kappa = 0.0
    print(f"    knee-rule pick κ={knee_kappa}")
    if knee_kappa != KAPPA_OP and knee_kappa > 0:
        s_knee = _additive_for_kappa(city, ds, fit, scores_blind,
                                          knee_kappa, "sink_core")
        r_knee, n_knee, _ = _per_user_r20_n20(s_knee, i_test, excl_test, u_test)
        diff_knee = r_knee - r_b
        knee_boot = _bootstrap_mean_ci(diff_knee)
        covers_zero_knee = (knee_boot["ci95_lo"] <= 0.0 <= knee_boot["ci95_hi"])
    else:
        diff_knee = diff_op
        knee_boot = op_boot
        covers_zero_knee = covers_zero
    a1_adjud.update({
        "kappa_op_knee_rule": knee_kappa,
        "mean_test_delta_R20_per_user_at_knee_kappa": float(diff_knee.mean()),
        "bootstrap_ci95_at_knee_kappa": knee_boot,
        "ci_covers_zero_at_knee_kappa": bool(covers_zero_knee),
        "binding_rule": ("val" if covers_zero else "knee"),
        "binding_kappa": (KAPPA_OP if covers_zero else knee_kappa),
        "binding_test_delta_R20": (float(diff_op.mean()) if covers_zero
                                       else float(diff_knee.mean())),
        "binding_test_delta_CI95": (op_boot if covers_zero else knee_boot),
    })
    (out_dir / "a1_oppoint_adjudication.json").write_text(
        json.dumps(a1_adjud, indent=2), encoding="utf-8")
    print(f"    knee-rule κ={knee_kappa}  test ΔR@20={float(diff_knee.mean()):+.4f}  "
          f"CI95=[{knee_boot['ci95_lo']:+.4f}, {knee_boot['ci95_hi']:+.4f}]  "
          f"covers 0? {covers_zero_knee}")
    print(f"    BINDING rule: {a1_adjud['binding_rule']} (κ={a1_adjud['binding_kappa']})")

    (out_dir / "bootstrap_ci.json").write_text(json.dumps(bootstrap, indent=2),
                                                  encoding="utf-8")

    # === 4. Step04 three-way (Wilcoxon + HMP + Holm) ====================
    print(f"  step04 three-way Wilcoxon + Holm ...")
    s04 = step04_threeway(city, npz, out_dir / "step04_threeway")

    # Overall verdict
    verdict = {
        "city": city,
        "matched_off_equivalent_at_005": tost_matchedoff_r20["equivalent"],
        "matched_off_equivalent_at_0025": tost_matchedoff_r20_sens["equivalent"],
        "xsage_kappa01_equivalent_at_005": tost_xsage01_r20["equivalent"],
        "xsage_kappa01_equivalent_at_0025": tost_xsage01_r20_sens["equivalent"],
        "B_full_R20_CI95": bootstrap["B_full_R20"],
        "B_full_minus_B_blind_R20_CI95": bootstrap["B_full_minus_B_blind_R20"],
        "a1_oppoint_decision": op_decision,
        "a1_binding_rule": a1_adjud["binding_rule"],
        "a1_binding_kappa": a1_adjud["binding_kappa"],
        "a1_binding_delta_R20": a1_adjud["binding_test_delta_R20"],
        "a1_binding_ci95_excludes_zero": not (
            a1_adjud["binding_test_delta_CI95"]["ci95_lo"] <= 0.0
            <= a1_adjud["binding_test_delta_CI95"]["ci95_hi"]),
        "step04_n_comparisons": len(s04["tests"]),
    }
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2),
                                              encoding="utf-8")
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", choices=["NYC", "TKY", "both"], default="both")
    args = parser.parse_args()
    cities = ["NYC", "TKY"] if args.city == "both" else [args.city]
    for c in cities:
        run_city(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
