"""Round-3 A1bis — val-flag recheck for the NYC MASK sinks.

C2 changed the NYC sink set under mask: {6} (round-2 hard) → {6, 7} (round-3
mask). The A1 anti-leak guarantee (val-flagged = test-flagged) was
established only for keep-mode. This script:

  1. Computes Stage-B lens on VAL requests under NYC MASK situations
     (URM_train excluded).
  2. Flags sinks with the unchanged rule.
  3. Compares to test-flagged {6, 7}.
  4. Re-runs the re-ranking sweep on TEST using val-flagged sinks; the
     operating point uses the **knee rule** (from A4) on the val curve.

TKY needs no action: two-perspective decision keeps the lens on keep-mode,
whose sinks {4, 5} were validated in session 1.
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

from pipeline.step02_models.xsage.backbone import load_or_refit
from pipeline.step02_models.xsage.metrics import (kl_divergence, long_tail_groups,
                                                       topk_from_scores)
from pipeline.step02_models.xsage.orchestrator import _load_city


K_TOP = 20
SHORT_HEAD = 0.20
LENS_RATIO = 1.5
LT_EXCESS_MIN = 0.05
EPS_VAL_ACC = 0.001
KAPPA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


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


def _r20(scores: np.ndarray, targets: np.ndarray,
            exclude: sps.csr_matrix, users: np.ndarray,
            K: int = K_TOP) -> float:
    hits = 0; n = scores.shape[0]
    for b in range(n):
        u = int(users[b])
        s = scores[b].copy()
        cols = exclude.indices[exclude.indptr[u]:exclude.indptr[u + 1]]
        if len(cols):
            s[cols] = -np.inf
        if (s > s[int(targets[b])]).sum() < K:
            hits += 1
    return hits / max(1, n)


def main() -> int:
    city = "NYC"
    sit_dir = REPO_ROOT / "outputs" / city / "xsage_transit_mask" / "situations"
    out_dir = REPO_ROOT / "outputs" / city / "xsage_transit_mask" / "round3" / "A1bis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> A1bis: NYC mask sink val-flag recheck")
    if not (sit_dir / "fit.npz").exists():
        raise FileNotFoundError(f"missing {sit_dir / 'fit.npz'} — run C2 first")
    fit = np.load(sit_dir / "fit.npz", allow_pickle=True)
    ds = _load_city(city)
    n_items = ds["n_items"]

    scores_uitem = load_or_refit(city, model_name="FM")
    df_val = ds["df_val"]; df_test = ds["df_test"]
    u_val = df_val["u_idx"].values.astype(np.int64)
    i_val = df_val["i_idx"].values.astype(np.int64)
    u_test = df_test["u_idx"].values.astype(np.int64)
    i_test = df_test["i_idx"].values.astype(np.int64)
    scores_val = scores_uitem[u_val]
    scores_test = scores_uitem[u_test]
    excl_val = ds["urm_train"].tocsr(); excl_val.data[:] = 1.0
    excl_test = (ds["urm_train"] + ds["urm_val"]).tocsr(); excl_test.data[:] = 1.0

    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(axis=0)).ravel()
    _, G1 = long_tail_groups(pop, short_head_share=SHORT_HEAD)

    # Val lens
    z_val = np.asarray(fit["core_label_val"]).astype(np.int32)
    isb_val = np.asarray(fit["is_boundary_val"]).astype(bool)
    print(f"  val lens on {len(u_val)} requests, K_sit={int(z_val.max() + 1)}")
    top_val_off = _topk(scores_val, excl_val, u_val)
    all_items_val = top_val_off.flatten()
    global_dist = np.bincount(all_items_val, minlength=n_items).astype(np.float64)
    global_dist /= max(global_dist.sum(), 1.0)
    global_LT_val = float(G1[all_items_val].mean())

    val_rows = []
    K_sit = int(z_val.max() + 1)
    for k in range(K_sit):
        for split, m in (("all", z_val == k),
                            ("core", (z_val == k) & ~isb_val),
                            ("boundary", (z_val == k) & isb_val)):
            n_req = int(m.sum())
            if n_req == 0:
                val_rows.append({"situation": k, "split": split, "n_requests": 0,
                                   "LT": None, "KL": None, "available_LT": None})
                continue
            items = top_val_off[m].flatten()
            d = np.bincount(items, minlength=n_items).astype(np.float64); d /= max(d.sum(), 1.0)
            lt = float(G1[items].mean())
            kl = kl_divergence(d, global_dist)
            val_rows.append({"situation": k, "split": split, "n_requests": n_req,
                              "LT": lt, "KL": kl})
        users_k = np.unique(u_val[z_val == k])
        avails = []
        for u in users_k:
            seen = excl_val.indices[excl_val.indptr[u]:excl_val.indptr[u + 1]]
            allowed = np.ones(n_items, dtype=bool); allowed[seen] = False
            if allowed.any():
                avails.append(float(G1[allowed].mean()))
        avail = float(np.mean(avails)) if avails else None
        for r in val_rows[-3:]:
            r["available_LT"] = avail
    pd.DataFrame(val_rows).to_csv(out_dir / "per_situation_val.csv", index=False)

    # Flag sinks
    global_KL_mean = float(np.mean([r["KL"] for r in val_rows
                                          if r["split"] == "all" and r["KL"] is not None]))
    val_sinks = []
    for r in val_rows:
        if r["split"] != "all" or r["KL"] is None: continue
        kl_r = r["KL"] / max(global_KL_mean, 1e-9)
        lt_ex = (r["LT"] - r["available_LT"]) if r["available_LT"] is not None else 0.0
        if kl_r >= LENS_RATIO and abs(lt_ex) >= LT_EXCESS_MIN:
            val_sinks.append({"situation": int(r["situation"]), "KL": r["KL"],
                                "KL_ratio_vs_global": kl_r,
                                "LT": r["LT"], "available_LT": r["available_LT"],
                                "lt_excess_over_available": lt_ex})
    set_val = sorted({s["situation"] for s in val_sinks})
    set_test = [6, 7]
    print(f"  val-flagged sinks: {set_val}")
    print(f"  test-flagged sinks (C2 NYC mask): {set_test}")
    match_exact = set_val == set_test
    val_subset_test = set(set_val).issubset(set(set_test))
    test_subset_val = set(set_test).issubset(set(set_val))
    print(f"  match exact: {match_exact}  val⊆test: {val_subset_test}  test⊆val: {test_subset_val}")

    sink_comparison = {
        "val_flagged": set_val, "test_flagged": set_test,
        "match_exact": match_exact, "val_subset_of_test": val_subset_test,
        "test_subset_of_val": test_subset_val,
        "global_LT_val": global_LT_val, "global_KL_mean_val": global_KL_mean,
    }
    (out_dir / "sink_comparison.json").write_text(json.dumps(sink_comparison, indent=2),
                                                       encoding="utf-8")

    # Re-ranking with val-flagged sinks
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)
    sinks_used = set_val if set_val else set_test
    sink_mask_test = np.isin(z_test, list(sinks_used))
    core_sink_test = sink_mask_test & ~isb_test
    sink_mask_val = np.isin(z_val, list(sinks_used))
    core_sink_val = sink_mask_val & ~isb_val
    print(f"  using sinks {sinks_used} for re-ranking (val-flagged)")

    top_test_off = _topk(scores_test, excl_test, u_test)
    sink_LT_off_test = (float(G1[top_test_off[sink_mask_test].flatten()].mean())
                          if sink_mask_test.any() else float("nan"))
    sink_LT_off_val = (float(G1[top_val_off[sink_mask_val].flatten()].mean())
                          if sink_mask_val.any() else float("nan"))
    r20_test_off = _r20(scores_test, i_test, excl_test, u_test)
    r20_val_off = _r20(scores_val, i_val, excl_val, u_val)

    rows = []; boost = G1.astype(np.float32)
    for kappa in KAPPA_GRID:
        nudge_t = np.zeros_like(scores_test)
        nudge_v = np.zeros_like(scores_val)
        if kappa > 0:
            nudge_t[np.where(core_sink_test)[0]] = kappa * boost[None, :]
            nudge_v[np.where(core_sink_val)[0]] = kappa * boost[None, :]
        s_on_t = scores_test + nudge_t; s_on_v = scores_val + nudge_v
        top_on_t = _topk(s_on_t, excl_test, u_test); top_on_v = _topk(s_on_v, excl_val, u_val)
        sink_LT_on_t = float(G1[top_on_t[sink_mask_test].flatten()].mean()) if sink_mask_test.any() else float("nan")
        sink_LT_on_v = float(G1[top_on_v[sink_mask_val].flatten()].mean()) if sink_mask_val.any() else float("nan")
        r20_on_t = _r20(s_on_t, i_test, excl_test, u_test)
        r20_on_v = _r20(s_on_v, i_val, excl_val, u_val)
        rows.append({"kappa_fair": kappa,
                       "sink_LT_off_val": sink_LT_off_val, "sink_LT_on_val": sink_LT_on_v,
                       "r20_val_delta": r20_on_v - r20_val_off,
                       "sink_LT_off_test": sink_LT_off_test, "sink_LT_on_test": sink_LT_on_t,
                       "r20_test_delta": r20_on_t - r20_test_off})
        print(f"  κ={kappa:>5.2f}  val LT {sink_LT_on_v:.3f} R@20Δ {r20_on_v - r20_val_off:+.4f}  "
              f"test LT {sink_LT_off_test:.3f}→{sink_LT_on_t:.3f} R@20Δ {r20_on_t - r20_test_off:+.4f}")
    df = pd.DataFrame(rows); df.to_csv(out_dir / "reranking_tradeoff_valflag.csv", index=False)

    # Knee operating point on val (sink_LT vs r20_delta curve)
    xs = df["sink_LT_on_val"].values; ys = df["r20_val_delta"].values
    knee_kappa = None
    if len(xs) >= 3 and xs[-1] != xs[0]:
        p0 = np.array([xs[0], ys[0]]); p1 = np.array([xs[-1], ys[-1]])
        seg = (p1 - p0) / np.linalg.norm(p1 - p0)
        dists = [np.linalg.norm(np.array([xs[i], ys[i]]) - p0 - np.dot(np.array([xs[i], ys[i]]) - p0, seg) * seg) for i in range(len(xs))]
        knee_idx = int(np.argmax(dists))
        knee_kappa = float(df.loc[knee_idx, "kappa_fair"])
    op_row = df[df["kappa_fair"] == knee_kappa].iloc[0] if knee_kappa is not None else df.iloc[-1]
    print(f"  knee operating point: κ={knee_kappa}  test LT={op_row['sink_LT_on_test']:.3f}  test R@20Δ={op_row['r20_test_delta']:+.4f}")

    verdict = ("PASS — val⊇test or =" if (match_exact or test_subset_val)
                  else "PARTIAL — s7 not val-validated; paper headline stays on s6 only"
                  if 7 not in set_val and 6 in set_val
                  else "MISMATCH — rewrite headline on val-flagged set")
    payload = {"city": city, "intent_transit": "mask",
                 "sink_comparison": sink_comparison,
                 "knee_kappa_val": knee_kappa,
                 "operating_point_test_sink_LT": float(op_row["sink_LT_on_test"]),
                 "operating_point_test_r20_delta": float(op_row["r20_test_delta"]),
                 "verdict": verdict}
    (out_dir / "verdict.json").write_text(json.dumps(payload, indent=2, default=str),
                                              encoding="utf-8")
    print(f"\n  verdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
