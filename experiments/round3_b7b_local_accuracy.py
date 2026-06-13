"""Round-3 B7b — accuracy restricted to touched requests.

B7 reported ΔR@5 ≈ -0.0005 (global) → "head-safe". But the global delta
is diluted by ~93% of requests that are untouched and contribute 0. The
honest statement requires the accuracy delta computed on the touched
subset (sink ∩ core requests where the list actually changed).

This task:
  1. identify touched subset (B5/B7 definition: list materially differs)
  2. on touched only: R@5/10/20, NDCG@5/10/20 for B_blind, X-SAGE,
     and global-rerank (κ-boost without sink-gating)
  3. paired bootstrap CI95 on ΔR@20 and ΔNDCG@10 (B=10000 resamples)
  4. displaced-rank decomposition: where in B_blind did the test target
     sit on touched requests, and was it pushed out?

Outputs:
  outputs/<city>/<sit_root>/round3/B7b/{
      touched_accuracy.csv,
      displaced_rank_analysis.csv,
      verdict.json,
  }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.xsage.backbone import excluded_mask, load_or_refit
from pipeline.step02_models.xsage.metrics import long_tail_groups, topk_from_scores
from pipeline.step02_models.xsage.orchestrator import _load_city


K_TOP = 20
SHORT_HEAD = 0.20
CUTOFFS = (5, 10, 20)
N_BOOT = 10000
RNG_SEED = 12345

CONFIG = {
    "NYC": {
        "sit_dir": "outputs/NYC/xsage_transit_mask/situations",
        "out_root": "outputs/NYC/xsage_transit_mask/round3/B7b",
        "sinks": [6, 7],
        "knee_kappa": 1.0,
        "mode_label": "mask",
    },
    "TKY": {
        "sit_dir": "outputs/TKY/xsage/situations",
        "out_root": "outputs/TKY/xsage/round3/B7b",
        "sinks": [4, 5],
        "knee_kappa": 2.0,
        "mode_label": "keep",
    },
}


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


def _per_request_metrics(top: np.ndarray, targets: np.ndarray,
                         cutoffs=CUTOFFS) -> dict[str, np.ndarray]:
    """For each request, hit@k (∈ {0,1}) and NDCG@k for the single
    held-out target. Returns dict keyed by 'R@k' / 'NDCG@k'."""
    n = top.shape[0]
    out: dict[str, np.ndarray] = {}
    # rank of target in top-K (1..K, or 0 if absent)
    rank_in_topK = np.zeros(n, dtype=np.int32)
    for b in range(n):
        t = int(targets[b])
        hits = np.where(top[b] == t)[0]
        rank_in_topK[b] = int(hits[0]) + 1 if len(hits) else 0
    for k in cutoffs:
        hit = (rank_in_topK > 0) & (rank_in_topK <= k)
        out[f"R@{k}"] = hit.astype(np.float32)
        # NDCG@k with single relevant: 1/log2(rank+1) if rank ≤ k else 0
        ndcg = np.zeros(n, dtype=np.float32)
        ix = np.where(hit)[0]
        if len(ix):
            ndcg[ix] = 1.0 / np.log2(rank_in_topK[ix] + 1)
        out[f"NDCG@{k}"] = ndcg
    out["_rank_in_top20"] = rank_in_topK
    return out


def _bootstrap_ci(diff: np.ndarray, B: int = N_BOOT,
                   seed: int = RNG_SEED) -> tuple[float, float, float]:
    """Paired bootstrap on the mean of `diff` (per-request delta).
    Resamples requests with replacement. Returns (mean, lo, hi) at 95%."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return (0.0, 0.0, 0.0)
    boot_means = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = diff[idx].mean()
    return (float(diff.mean()),
            float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)))


def run_city(city: str) -> dict:
    cfg = CONFIG[city]
    out_dir = REPO_ROOT / cfg["out_root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> B7b on {city} ({cfg['mode_label']}, sinks={cfg['sinks']}, "
          f"κ={cfg['knee_kappa']})")

    fit = np.load(REPO_ROOT / cfg["sit_dir"] / "fit.npz", allow_pickle=True)
    ds = _load_city(city)
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)
    df_test = ds["df_test"]
    u_test = df_test["u_idx"].values.astype(np.int64)
    i_target = df_test["i_idx"].values.astype(np.int64)
    n_items = ds["n_items"]

    scores_blind_uitem = load_or_refit(city, model_name="FM")
    scores_blind = scores_blind_uitem[u_test]
    excl = excluded_mask(city, n_items)

    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(axis=0)).ravel()
    G0_mask, G1_mask = long_tail_groups(pop, short_head_share=SHORT_HEAD)
    boost = G1_mask.astype(np.float32)

    sink_mask = np.isin(z_test, cfg["sinks"])
    core_sink = sink_mask & ~isb_test
    n_test = len(u_test)

    # B_blind / X-SAGE / global-rerank top-20
    top_off = _topk(scores_blind, excl, u_test)
    scores_xsage = scores_blind.copy()
    scores_xsage[np.where(core_sink)[0]] += cfg["knee_kappa"] * boost[None, :]
    top_xsage = _topk(scores_xsage, excl, u_test)

    scores_global = scores_blind + cfg["knee_kappa"] * boost[None, :]
    top_global = _topk(scores_global, excl, u_test)

    # touched mask under X-SAGE (definition matches B5)
    touched_xsage = np.zeros(n_test, dtype=bool)
    for b in range(n_test):
        if not np.array_equal(top_off[b], top_xsage[b]):
            inter = len(np.intersect1d(top_off[b], top_xsage[b],
                                       assume_unique=False))
            if inter < K_TOP:
                touched_xsage[b] = True
    n_touched = int(touched_xsage.sum())
    print(f"  touched (X-SAGE) = {n_touched} / {n_test}  "
          f"({n_touched / n_test:.4f})")

    # touched mask under GLOBAL rerank, for completeness (sanity)
    touched_global = np.zeros(n_test, dtype=bool)
    for b in range(n_test):
        if not np.array_equal(top_off[b], top_global[b]):
            inter = len(np.intersect1d(top_off[b], top_global[b],
                                       assume_unique=False))
            if inter < K_TOP:
                touched_global[b] = True
    n_touched_global = int(touched_global.sum())

    # ----------------- per-request metrics -----------------
    m_off = _per_request_metrics(top_off, i_target)
    m_xsage = _per_request_metrics(top_xsage, i_target)
    m_global = _per_request_metrics(top_global, i_target)

    # ----------------- aggregate on touched subset ---------
    rows = []
    for label, scope_mask, n_scope in [
        ("touched_xsage", touched_xsage, n_touched),
        ("global_all", np.ones(n_test, bool), n_test),
        ("touched_global", touched_global, n_touched_global),
    ]:
        if n_scope == 0:
            continue
        for k in CUTOFFS:
            r_off = float(m_off[f"R@{k}"][scope_mask].mean())
            r_xs  = float(m_xsage[f"R@{k}"][scope_mask].mean())
            r_glob = float(m_global[f"R@{k}"][scope_mask].mean())
            n_off = float(m_off[f"NDCG@{k}"][scope_mask].mean())
            n_xs  = float(m_xsage[f"NDCG@{k}"][scope_mask].mean())
            n_glob = float(m_global[f"NDCG@{k}"][scope_mask].mean())
            rows.append({
                "scope": label, "n_requests": int(n_scope), "k": k,
                "R_off": r_off,
                "R_xsage": r_xs, "dR_xsage": r_xs - r_off,
                "R_global_rerank": r_glob, "dR_global_rerank": r_glob - r_off,
                "NDCG_off": n_off,
                "NDCG_xsage": n_xs, "dNDCG_xsage": n_xs - n_off,
                "NDCG_global_rerank": n_glob, "dNDCG_global_rerank": n_glob - n_off,
            })
    df_acc = pd.DataFrame(rows)
    df_acc.to_csv(out_dir / "touched_accuracy.csv", index=False)

    # ----------------- bootstrap CI on touched X-SAGE ------
    ci = {}
    if n_touched > 0:
        for metric in ("R@20", "NDCG@10"):
            diff = (m_xsage[metric] - m_off[metric])[touched_xsage]
            mean, lo, hi = _bootstrap_ci(diff, B=N_BOOT, seed=RNG_SEED)
            ci[f"xsage_d{metric}_mean"] = mean
            ci[f"xsage_d{metric}_ci95_lo"] = lo
            ci[f"xsage_d{metric}_ci95_hi"] = hi
            ci[f"xsage_d{metric}_covers_zero"] = bool(lo <= 0.0 <= hi)
            # also CI for global rerank on the SAME touched subset
            diff_g = (m_global[metric] - m_off[metric])[touched_xsage]
            mean_g, lo_g, hi_g = _bootstrap_ci(diff_g, B=N_BOOT, seed=RNG_SEED + 1)
            ci[f"global_d{metric}_mean"] = mean_g
            ci[f"global_d{metric}_ci95_lo"] = lo_g
            ci[f"global_d{metric}_ci95_hi"] = hi_g

    # ----------------- displaced-rank decomposition --------
    # For each touched request, the rank of i_target in top_off:
    #   0   → target not in B_blind top-20 (was already a miss)
    #   1-13 → "shallow rank" hit
    #   14-20 → "deep rank" hit  (the boost-vulnerable zone)
    # Also: did X-SAGE still hit at top-20?
    rank_off = m_off["_rank_in_top20"]
    rank_xs = m_xsage["_rank_in_top20"]

    deep_band = (14, 20)
    shallow_band = (1, 13)

    bins = []
    bin_defs = [
        ("miss_in_blind", lambda r: r == 0),
        ("shallow_hit_in_blind", lambda r: shallow_band[0] <= r <= shallow_band[1]),
        ("deep_hit_in_blind", lambda r: deep_band[0] <= r <= deep_band[1]),
    ]
    for b_label, b_fn in bin_defs:
        scope = touched_xsage & np.array([b_fn(int(r)) for r in rank_off])
        n_bin = int(scope.sum())
        # within this bin, how many lose the hit under X-SAGE?
        hit_off = m_off["R@20"][scope].astype(bool)
        hit_xs = m_xsage["R@20"][scope].astype(bool)
        lost = int((hit_off & ~hit_xs).sum())
        gained = int((~hit_off & hit_xs).sum())
        # rank shift among bin members that retain a hit
        retained = hit_off & hit_xs  # length sum(scope)
        n_ret = int(retained.sum())
        if n_ret:
            rank_off_scope = rank_off[scope]      # length sum(scope)
            rank_xs_scope = rank_xs[scope]        # length sum(scope)
            rank_off_ret = rank_off_scope[retained]
            rank_xs_ret = rank_xs_scope[retained]
            mean_drank = float((rank_xs_ret - rank_off_ret).mean())
        else:
            mean_drank = 0.0
        bins.append({
            "rank_band_in_B_blind": b_label,
            "n_touched_in_band": n_bin,
            "share_of_touched": (n_bin / n_touched) if n_touched else 0.0,
            "hits_lost_under_xsage": lost,
            "hits_gained_under_xsage": gained,
            "net_hits": gained - lost,
            "n_retained_hits": n_ret,
            "mean_rank_shift_on_retained": mean_drank,
        })
    df_band = pd.DataFrame(bins)
    df_band.to_csv(out_dir / "displaced_rank_analysis.csv", index=False)

    # ----------------- verdict -----------------------------
    dR20 = ci.get("xsage_dR@20_mean", 0.0)
    lo20 = ci.get("xsage_dR@20_ci95_lo", 0.0)
    hi20 = ci.get("xsage_dR@20_ci95_hi", 0.0)
    dN10 = ci.get("xsage_dNDCG@10_mean", 0.0)
    loN = ci.get("xsage_dNDCG@10_ci95_lo", 0.0)
    hiN = ci.get("xsage_dNDCG@10_ci95_hi", 0.0)
    covers_zero_R20 = ci.get("xsage_dR@20_covers_zero", False)
    covers_zero_N10 = ci.get("xsage_dNDCG@10_covers_zero", False)

    # Tiering:
    #   negligible: BOTH metrics cover 0 OR BOTH |Δ| < 0.005
    #   small:      one metric significantly nonzero but |Δ| < 0.02
    #   large:      any |Δ| ≥ 0.02
    abs_max = max(abs(dR20), abs(dN10))
    if (covers_zero_R20 and covers_zero_N10) or abs_max < 0.005:
        verdict_label = "local_cost_negligible"
    elif abs_max < 0.02:
        verdict_label = "local_cost_small"
    else:
        verdict_label = "local_cost_large"

    verdict = {
        "city": city,
        "mode": cfg["mode_label"],
        "sinks": cfg["sinks"],
        "knee_kappa": cfg["knee_kappa"],
        "n_test": int(n_test),
        "n_touched_xsage": int(n_touched),
        "touched_share_xsage": float(n_touched / n_test),
        "n_touched_global_rerank": int(n_touched_global),
        "touched_accuracy_rows": rows,
        "ci_touched_xsage": ci,
        "displaced_rank_analysis": bins,
        "verdict": verdict_label,
    }
    with open(out_dir / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    # ----------------- console -----------------------------
    print(f"  TOUCHED-ONLY ΔR@20 (X-SAGE): {dR20:+.4f}  "
          f"CI95 [{lo20:+.4f}, {hi20:+.4f}]  "
          f"covers 0: {covers_zero_R20}")
    print(f"  TOUCHED-ONLY ΔNDCG@10 (X-SAGE): {dN10:+.4f}  "
          f"CI95 [{loN:+.4f}, {hiN:+.4f}]  "
          f"covers 0: {covers_zero_N10}")
    print(f"  GLOBAL    ΔR@20 (X-SAGE): "
          f"{next(r['dR_xsage'] for r in rows if r['scope']=='global_all' and r['k']==20):+.5f}")
    print(f"  CONTRAST  global-rerank on SAME touched subset: ΔR@20 "
          f"{ci.get('global_dR@20_mean', 0):+.4f} "
          f"CI95 [{ci.get('global_dR@20_ci95_lo', 0):+.4f}, "
          f"{ci.get('global_dR@20_ci95_hi', 0):+.4f}]")
    print(f"  displaced rank bands (B_blind):")
    for r in bins:
        print(f"    {r['rank_band_in_B_blind']:25s}  "
              f"n={r['n_touched_in_band']:5d}  "
              f"lost={r['hits_lost_under_xsage']:4d}  "
              f"gained={r['hits_gained_under_xsage']:4d}  "
              f"Δrank_retained={r['mean_rank_shift_on_retained']:+.2f}")
    print(f"  verdict: {verdict_label}")
    return verdict


def main() -> int:
    out_root = REPO_ROOT / "outputs" / "round3" / "B7b"
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for city in ("NYC", "TKY"):
        summary[city] = run_city(city)
    with open(out_root / "verdict.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== B7b ROLL-UP ===")
    for c, v in summary.items():
        ci = v["ci_touched_xsage"]
        print(f"  {c}: verdict={v['verdict']}  "
              f"n_touched={v['n_touched_xsage']}  "
              f"ΔR@20={ci.get('xsage_dR@20_mean',0):+.4f}"
              f"[{ci.get('xsage_dR@20_ci95_lo',0):+.4f},"
              f"{ci.get('xsage_dR@20_ci95_hi',0):+.4f}]  "
              f"ΔNDCG@10={ci.get('xsage_dNDCG@10_mean',0):+.4f}"
              f"[{ci.get('xsage_dNDCG@10_ci95_lo',0):+.4f},"
              f"{ci.get('xsage_dNDCG@10_ci95_hi',0):+.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
