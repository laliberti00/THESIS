"""Round-3 A1 — sink identification on validation (fix the selection leak).

Stage B currently flags inequity sinks by stratifying B_blind's *test*
top-K. The fairness re-ranking is then evaluated on the same test set —
selection and evaluation are coupled. This script:

    1. Computes the Stage B lens (LT/KL/available_LT) on VALIDATION
       requests using B_blind scores (URM_train excluded).
    2. Flags sinks on val with the same rule (KL ≥ 1.5 × global mean AND
       |LT − available_LT| ≥ 0.05).
    3. Reports val-flagged vs test-flagged sink set difference per city.
    4. Re-runs the 2.1 re-ranking sweep on test using the **val-flagged**
       sink set. Also selects κ_fair on val (best sink LT subject to
       ΔR@20_val ≥ −ε with ε = 0.001) and reports test numbers at that κ.

Outputs under outputs/<city>/xsage/round3/A1/:
    per_situation_val.csv
    sink_comparison.json
    reranking_tradeoff_valflag.csv
    verdict.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.xsage.backbone import load_or_refit
from pipeline.step02_models.xsage.metrics import (kl_divergence,
                                                       long_tail_groups,
                                                       topk_from_scores)
from pipeline.step02_models.xsage.orchestrator import _load_city


K_TOP = 20
SHORT_HEAD_SHARE = 0.20
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


def _recall20(scores: np.ndarray, targets: np.ndarray,
                 exclude: sps.csr_matrix, users: np.ndarray,
                 K: int = K_TOP) -> float:
    hits = 0; n = scores.shape[0]
    for b in range(n):
        u = int(users[b])
        s = scores[b].copy()
        cols = exclude.indices[exclude.indptr[u]:exclude.indptr[u + 1]]
        if len(cols):
            s[cols] = -np.inf
        target = int(targets[b])
        ts = s[target]
        if (s > ts).sum() < K:
            hits += 1
    return hits / max(1, n)


def lens_on_split(scores_per_req: np.ndarray, u_split: np.ndarray,
                    z_split: np.ndarray, isb_split: np.ndarray,
                    exclude: sps.csr_matrix, G1_mask: np.ndarray,
                    n_items: int) -> tuple[list[dict], float, float]:
    """Stratified lens: return rows + global LT + global KL mean (all split)."""
    top = _topk(scores_per_req, exclude, u_split, K=K_TOP)
    all_items = top.flatten()
    global_dist = np.bincount(all_items, minlength=n_items).astype(np.float64)
    global_dist /= max(global_dist.sum(), 1.0)
    global_LT = float(G1_mask[all_items].mean())
    K_sit = int(z_split.max() + 1)
    rows = []
    for k in range(K_sit):
        for split_name, mask in (("all", z_split == k),
                                    ("core", (z_split == k) & ~isb_split),
                                    ("boundary", (z_split == k) & isb_split)):
            n_req = int(mask.sum())
            if n_req == 0:
                rows.append({"situation": k, "split": split_name,
                              "n_requests": 0, "LT": None, "KL": None})
                continue
            items = top[mask].flatten()
            d = np.bincount(items, minlength=n_items).astype(np.float64)
            d /= max(d.sum(), 1.0)
            lt = float(G1_mask[items].mean())
            kl = kl_divergence(d, global_dist)
            rows.append({"situation": k, "split": split_name,
                          "n_requests": n_req, "LT": lt, "KL": kl})
        users_k = np.unique(u_split[z_split == k])
        shares = []
        for u in users_k:
            seen = exclude.indices[exclude.indptr[u]:exclude.indptr[u + 1]]
            allowed = np.ones(n_items, dtype=bool); allowed[seen] = False
            if allowed.any():
                shares.append(float(G1_mask[allowed].mean()))
        avail = float(np.mean(shares)) if shares else None
        for r in rows[-3:]:
            r["available_LT"] = avail
    df = pd.DataFrame(rows)
    global_KL_mean = float(df.loc[df["split"] == "all", "KL"].mean())
    df["global_LT"] = global_LT
    df["global_KL_mean"] = global_KL_mean
    return df.to_dict("records"), global_LT, global_KL_mean


def find_sinks(rows: list[dict], global_KL_mean: float,
                  kl_ratio: float = LENS_RATIO,
                  lt_excess_min: float = LT_EXCESS_MIN) -> list[dict]:
    sinks = []
    for r in rows:
        if r["split"] != "all" or r["KL"] is None: continue
        kl_r = float(r["KL"] / max(global_KL_mean, 1e-9))
        lt_ex = (float(r["LT"] - r["available_LT"])
                  if r["available_LT"] is not None else 0.0)
        if kl_r >= kl_ratio and abs(lt_ex) >= lt_excess_min:
            sinks.append({"situation": int(r["situation"]),
                            "KL": float(r["KL"]),
                            "KL_ratio_vs_global": kl_r,
                            "LT": float(r["LT"]),
                            "available_LT": float(r["available_LT"]),
                            "lt_excess_over_available": lt_ex})
    return sinks


def run_city(city: str) -> dict:
    out_dir = REPO_ROOT / "outputs" / city / "xsage" / "round3" / "A1"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> A1 on {city}")
    ds = _load_city(city)
    sit_dir = REPO_ROOT / "outputs" / city / "xsage" / "situations"
    fit = np.load(sit_dir / "fit.npz", allow_pickle=True)

    # Backbone scores (cached)
    scores_uitem = load_or_refit(city, model_name="FM")
    df_val = ds["df_val"]; df_test = ds["df_test"]
    u_val = df_val["u_idx"].values.astype(np.int64)
    u_test = df_test["u_idx"].values.astype(np.int64)
    i_val = df_val["i_idx"].values.astype(np.int64)
    i_test = df_test["i_idx"].values.astype(np.int64)
    scores_val = scores_uitem[u_val]
    scores_test = scores_uitem[u_test]

    # Exclude sets
    excl_val = ds["urm_train"].tocsr()
    excl_val.data[:] = 1.0
    excl_test = (ds["urm_train"] + ds["urm_val"]).tocsr()
    excl_test.data[:] = 1.0

    # Long-tail
    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(axis=0)).ravel()
    _, G1_mask = long_tail_groups(pop, short_head_share=SHORT_HEAD_SHARE)

    # Val lens
    z_val = np.asarray(fit["core_label_val"]).astype(np.int32)
    isb_val = np.asarray(fit["is_boundary_val"]).astype(bool)
    print(f"  val lens ({len(u_val)} requests) ...")
    val_rows, val_LT, val_KL_mean = lens_on_split(
        scores_val, u_val, z_val, isb_val, excl_val, G1_mask, ds["n_items"])
    sinks_val = find_sinks(val_rows, val_KL_mean)
    pd.DataFrame(val_rows).to_csv(out_dir / "per_situation_val.csv", index=False)

    # Existing test lens (read from Stage B output)
    test_verdict_path = REPO_ROOT / "outputs" / city / "xsage" / "fairness" / "verdict.json"
    test_verdict = json.loads(test_verdict_path.read_text())
    sinks_test = test_verdict["inequity_sinks"]

    # Compare
    set_val = {s["situation"] for s in sinks_val}
    set_test = {int(s["situation"]) for s in sinks_test}
    overlap = set_val & set_test
    only_val = set_val - set_test
    only_test = set_test - set_val
    print(f"  val-flagged sinks: {sorted(set_val)}  "
          f"test-flagged sinks: {sorted(set_test)}")
    sink_comparison = {
        "val_flagged": sorted(set_val),
        "test_flagged": sorted(set_test),
        "overlap": sorted(overlap),
        "only_val": sorted(only_val),
        "only_test": sorted(only_test),
        "match_exact": set_val == set_test,
        "val_subset_of_test": set_val.issubset(set_test),
        "test_subset_of_val": set_test.issubset(set_val),
        "global_LT_val": val_LT,
        "global_KL_mean_val": val_KL_mean,
        "global_LT_test": float(test_verdict["global_LT"]),
    }
    (out_dir / "sink_comparison.json").write_text(
        json.dumps(sink_comparison, indent=2), encoding="utf-8")

    # Re-ranking with val-flagged sinks
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)
    sink_mask_test = np.isin(z_test, list(set_val))
    sink_mask_val = np.isin(z_val, list(set_val))
    core_sink_test = sink_mask_test & ~isb_test
    core_sink_val = sink_mask_val & ~isb_val

    # Top-off (baseline) on both splits
    top_test_off = _topk(scores_test, excl_test, u_test, K=K_TOP)
    top_val_off = _topk(scores_val, excl_val, u_val, K=K_TOP)
    sink_LT_off_test = (
        float(G1_mask[top_test_off[sink_mask_test].flatten()].mean())
        if sink_mask_test.any() else float("nan"))
    sink_LT_off_val = (
        float(G1_mask[top_val_off[sink_mask_val].flatten()].mean())
        if sink_mask_val.any() else float("nan"))
    r20_test_off = _recall20(scores_test, i_test, excl_test, u_test, K=K_TOP)
    r20_val_off = _recall20(scores_val, i_val, excl_val, u_val, K=K_TOP)

    rows = []
    boost = G1_mask.astype(np.float32)
    for kappa in KAPPA_GRID:
        # apply on TEST with val-flagged sinks
        nudge_t = np.zeros_like(scores_test)
        if kappa > 0:
            idx = np.where(core_sink_test)[0]
            nudge_t[idx] = kappa * boost[None, :]
        scores_on_test = scores_test + nudge_t
        top_on_test = _topk(scores_on_test, excl_test, u_test, K=K_TOP)
        sink_on_items_test = top_on_test[sink_mask_test].flatten()
        sink_LT_on_test = (float(G1_mask[sink_on_items_test].mean())
                            if len(sink_on_items_test) else float("nan"))
        r20_on_test = _recall20(scores_on_test, i_test, excl_test, u_test, K=K_TOP)

        # also val LT/R@20 at this kappa for operating point pick
        nudge_v = np.zeros_like(scores_val)
        if kappa > 0:
            idx = np.where(core_sink_val)[0]
            nudge_v[idx] = kappa * boost[None, :]
        scores_on_val = scores_val + nudge_v
        top_on_val = _topk(scores_on_val, excl_val, u_val, K=K_TOP)
        sink_on_items_val = top_on_val[sink_mask_val].flatten()
        sink_LT_on_val = (float(G1_mask[sink_on_items_val].mean())
                            if len(sink_on_items_val) else float("nan"))
        r20_on_val = _recall20(scores_on_val, i_val, excl_val, u_val, K=K_TOP)

        rows.append({
            "kappa_fair": kappa,
            "sink_LT_off_val": sink_LT_off_val,
            "sink_LT_on_val": sink_LT_on_val,
            "r20_val_off": r20_val_off, "r20_val_on": r20_on_val,
            "r20_val_delta": r20_on_val - r20_val_off,
            "sink_LT_off_test": sink_LT_off_test,
            "sink_LT_on_test": sink_LT_on_test,
            "r20_test_off": r20_test_off, "r20_test_on": r20_on_test,
            "r20_test_delta": r20_on_test - r20_test_off,
        })
        print(f"  κ={kappa:>5.2f}  val: LT {sink_LT_on_val:.3f} R@20 Δ {r20_on_val - r20_val_off:+.4f}  "
              f"test: LT {sink_LT_off_test:.3f}→{sink_LT_on_test:.3f}  R@20 Δ {r20_on_test - r20_test_off:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "reranking_tradeoff_valflag.csv", index=False)

    # Operating point: best sink_LT_on_val subject to r20_val_delta ≥ -EPS_VAL_ACC.
    eligible = df[df["r20_val_delta"] >= -EPS_VAL_ACC]
    if len(eligible):
        op = eligible.loc[eligible["sink_LT_on_val"].idxmax()]
    else:
        op = df.loc[df["sink_LT_on_val"].idxmax()]
    op_dict = op.to_dict()
    print(f"  operating point κ={op['kappa_fair']:.2f}  val LT={op['sink_LT_on_val']:.3f}  "
          f"test LT={op['sink_LT_on_test']:.3f}  test R@20 Δ={op['r20_test_delta']:+.4f}")

    # Verdict
    accept_match = (sink_comparison["val_subset_of_test"]
                      or sink_comparison["match_exact"])
    verdict = "PASS" if accept_match else "MISMATCH — rewrite headline on val-flagged set"
    payload = {
        "city": city,
        "sink_comparison": sink_comparison,
        "operating_point": op_dict,
        "verdict": verdict,
        "epsilon_val_acc": EPS_VAL_ACC,
        "kappa_grid": list(KAPPA_GRID),
    }
    (out_dir / "verdict.json").write_text(json.dumps(payload, indent=2,
                                                          default=str),
                                              encoding="utf-8")
    print(f"  verdict: {verdict}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", choices=["NYC", "TKY", "both"], default="both")
    args = parser.parse_args()
    cities = ["NYC", "TKY"] if args.city == "both" else [args.city]
    overall = {}
    for c in cities:
        overall[c] = run_city(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
