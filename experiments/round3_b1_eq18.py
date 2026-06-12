"""Round-3 B1 — eq.18 boundary disambiguation evaluation.

Measures whether the projection feedback (r̃_k ∝ r_k · T[z_prev, k]) buys
us anything on boundary test requests:

  (a) Next-situation F1 on boundary rows: predicted z_t with vs without
      feedback. Without feedback = argmax of original r (= core_label).
      With feedback = argmax of r̃.
  (b) Stage-D effect at κ=0.1 additive: re-run the additive combiner with
      disambiguated r̃ and compare Δ_K + R@20 to the standard r path.

Run on NYC and TKY; on TKY also under intent-mode `all` since round-2 1.4
identified that as TKY's current best.

If flat/negative → eq.18 demoted to "design extension" in the tex.
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

from pipeline.step02_models.xsage.backbone import excluded_mask, load_or_refit
from pipeline.step02_models.xsage.l3_projection import (
    boundary_disambiguate, estimate_transition, macro_f1, predict_next_situation,
)
from pipeline.step02_models.xsage.metrics import topk_from_scores
from pipeline.step02_models.xsage.orchestrator import _load_city
from pipeline.step02_models.xsage.recommendation import (
    additive_combine_scores, fit_situation_biases_z,
)


K_TOP = 20


def _z_prev_test(ds: dict, fit: np.ndarray) -> np.ndarray:
    """Compute z_prev for each test row: the user's most recent (train ∪ val)
    z label strictly before the test row's time_local. Returns -1 where no
    history exists.
    """
    df_train = ds["df_train"]; df_val = ds["df_val"]; df_test = ds["df_test"]
    z_train = np.asarray(fit["core_label_train"]).astype(np.int32)
    z_val = np.asarray(fit["core_label_val"]).astype(np.int32)
    tv = pd.concat([df_train.assign(_z=z_train), df_val.assign(_z=z_val)],
                     ignore_index=True).sort_values(["user_id", "time_local"]) \
                                          .reset_index(drop=True)
    by_user: dict[int, dict[str, np.ndarray]] = {}
    for u, g in tv.groupby("user_id", sort=False):
        by_user[int(u)] = {
            "t": g["time_local"].values.astype("datetime64[ns]"),
            "z": g["_z"].values.astype(np.int32),
        }
    users_test = df_test["user_id"].values.astype(np.int64)
    times_test = df_test["time_local"].values.astype("datetime64[ns]")
    out = np.full(len(df_test), -1, dtype=np.int32)
    for q in range(len(df_test)):
        rec = by_user.get(int(users_test[q]))
        if rec is None: continue
        cut = np.searchsorted(rec["t"], times_test[q], side="left")
        if cut > 0:
            out[q] = rec["z"][cut - 1]
    return out


def _topk_metrics(scores: np.ndarray, exclude: sps.csr_matrix,
                     users: np.ndarray, K: int = K_TOP) -> np.ndarray:
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


def run_city(city: str, intent_mode: str = "hard") -> dict:
    print(f"\n>>> B1 on {city} (mode={intent_mode})")
    if intent_mode == "hard":
        sit_root = REPO_ROOT / "outputs" / city / "xsage"
    else:
        sit_root = REPO_ROOT / "outputs" / city / f"xsage_intent_{intent_mode}"
    sit_dir = sit_root / "situations"
    out_dir = sit_root / "round3" / "B1"
    out_dir.mkdir(parents=True, exist_ok=True)

    fit = np.load(sit_dir / "fit.npz", allow_pickle=True)
    ds = _load_city(city)
    z_train = np.asarray(fit["core_label_train"]).astype(np.int32)
    z_val = np.asarray(fit["core_label_val"]).astype(np.int32)
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)
    membership_test = np.asarray(fit["membership_test"]).astype(np.float32)
    K_sit = int(max(z_train.max(), z_val.max(), z_test.max()) + 1)

    # Transition T from (train ∪ val) sequences
    print(f"  estimating T from train+val sequences ...")
    tv = pd.concat([
        ds["df_train"].assign(_z=z_train),
        ds["df_val"].assign(_z=z_val),
    ], ignore_index=True).sort_values(["user_id", "time_local"]) \
                              .reset_index(drop=True)
    sequences = [g["_z"].values.astype(np.int32) for _, g in tv.groupby("user_id", sort=False)]
    T, _ = estimate_transition(sequences, K=K_sit, add_one_smoothing=True)

    # z_prev for test rows
    z_prev = _z_prev_test(ds, fit)
    valid = z_prev >= 0; valid_boundary = valid & isb_test
    print(f"  test rows: {len(z_test)}  valid z_prev: {int(valid.sum())}  "
          f"boundary with z_prev: {int(valid_boundary.sum())}")

    # === (a) Next-situation F1 on boundary rows =====================
    # The brief asks for "next-situation F1". We predict the NEXT test row's
    # z (z_{t+1} of the same user). z_pred_no_fb = argmax(r) of the current
    # row; z_pred_with_fb = argmax(r̃). Without feedback the boundary
    # assignment is the closest-prototype tie-break; with feedback it is
    # disambiguated by z_prev.
    r = membership_test.astype(np.float64)
    r_tilde = boundary_disambiguate(r, T, z_prev)
    z_t_pred_no_fb = r.argmax(axis=1).astype(np.int32)
    z_t_pred_with_fb = r_tilde.argmax(axis=1).astype(np.int32)

    # Compute "next-z" target: for each test row q, find the user's next test
    # row's core_label. Rows without a next row are excluded.
    df_test = ds["df_test"]
    next_z = np.full(len(df_test), -1, dtype=np.int32)
    order = df_test.sort_values(["user_id", "time_local"]).index.values
    z_sorted = z_test[order]
    u_sorted = df_test["user_id"].values[order]
    for j in range(len(order) - 1):
        if u_sorted[j] == u_sorted[j + 1]:
            next_z[order[j]] = z_sorted[j + 1]
    valid_next = next_z >= 0
    valid_boundary_next = valid_boundary & valid_next
    print(f"  boundary rows with both z_prev AND next-z: {int(valid_boundary_next.sum())}")

    if int(valid_boundary_next.sum()) > 0:
        f1_no_fb = macro_f1(next_z[valid_boundary_next],
                                z_t_pred_no_fb[valid_boundary_next], K_sit)
        f1_with_fb = macro_f1(next_z[valid_boundary_next],
                                  z_t_pred_with_fb[valid_boundary_next], K_sit)
    else:
        f1_no_fb = f1_with_fb = float("nan")
    print(f"  boundary next-situation F1: no fb={f1_no_fb:.3f}  "
          f"with fb={f1_with_fb:.3f}  Δ={f1_with_fb - f1_no_fb:+.3f}")

    # === (b) Stage-D additive at κ=0.1, r vs r̃ =====================
    # Build the scoring path
    print(f"  building Stage-D variants at κ=0.1 ...")
    macro_to_idx = ds["macro_to_idx"]
    n_macros = ds["n_macros"]; n_items = ds["n_items"]
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
    cm = big.groupby(["i_idx", "cat_macro"]).size().reset_index(name="n")
    best = cm.sort_values(["i_idx", "n"], ascending=[True, False]) \
                .drop_duplicates("i_idx", keep="first")
    item_macro = np.zeros(n_items, dtype=np.int64)
    for _, rrow in best.iterrows():
        item_macro[int(rrow["i_idx"])] = macro_to_idx[rrow["cat_macro"]]
    scores_blind_uitem = load_or_refit(city, model_name="FM")
    u_test = ds["df_test"]["u_idx"].values.astype(np.int64)
    scores_blind = scores_blind_uitem[u_test]
    excl_test = (ds["urm_train"] + ds["urm_val"]).tocsr(); excl_test.data[:] = 1.0
    i_test = ds["df_test"]["i_idx"].values.astype(np.int64)

    gamma_no_fb = np.where(isb_test,
                              1.0 / np.maximum((r > 0).sum(axis=1), 1),
                              1.0).astype(np.float32)
    gamma_with_fb = np.where(isb_test,
                                 1.0 / np.maximum((r_tilde > 0).sum(axis=1), 1),
                                 1.0).astype(np.float32)

    s_no_fb = additive_combine_scores(scores_blind, r.astype(np.float32),
                                            biases_z, item_macro, kappa=0.1,
                                            gamma_per_request=gamma_no_fb)
    s_with_fb = additive_combine_scores(scores_blind, r_tilde.astype(np.float32),
                                              biases_z, item_macro, kappa=0.1,
                                              gamma_per_request=gamma_with_fb)
    # Δ_K vs B_blind (top-K change rate)
    top_blind = _topk_metrics(scores_blind, excl_test, u_test)
    top_no_fb = _topk_metrics(s_no_fb, excl_test, u_test)
    top_with_fb = _topk_metrics(s_with_fb, excl_test, u_test)
    def delta_k(t_on: np.ndarray, t_off: np.ndarray) -> np.ndarray:
        n = t_on.shape[0]
        out = np.zeros(n, dtype=np.float32)
        for b in range(n):
            inter = len(np.intersect1d(t_on[b], t_off[b], assume_unique=False))
            out[b] = 1.0 - inter / K_TOP
        return out
    d_no_fb = delta_k(top_no_fb, top_blind)
    d_with_fb = delta_k(top_with_fb, top_blind)
    # Average per request and on boundary only
    avg_delta_no_fb_boundary = float(d_no_fb[isb_test].mean()) if isb_test.any() else 0.0
    avg_delta_with_fb_boundary = float(d_with_fb[isb_test].mean()) if isb_test.any() else 0.0
    # R@20
    r20_blind = _r20(scores_blind, i_test, excl_test, u_test)
    r20_no_fb = _r20(s_no_fb, i_test, excl_test, u_test)
    r20_with_fb = _r20(s_with_fb, i_test, excl_test, u_test)

    print(f"  Stage-D effect (κ=0.1 additive, boundary requests):")
    print(f"    Δ_K boundary  no fb={avg_delta_no_fb_boundary:.4f}  "
          f"with fb={avg_delta_with_fb_boundary:.4f}  "
          f"Δ={avg_delta_with_fb_boundary - avg_delta_no_fb_boundary:+.4f}")
    print(f"    R@20 global   B_blind={r20_blind:.4f}  no fb={r20_no_fb:.4f}  "
          f"with fb={r20_with_fb:.4f}  Δ(fb−no fb)={r20_with_fb - r20_no_fb:+.4f}")

    # Verdict
    delta_f1 = float(f1_with_fb - f1_no_fb) if not np.isnan(f1_no_fb) else 0.0
    delta_r20 = float(r20_with_fb - r20_no_fb)
    helpful_f1 = delta_f1 >= 0.01
    helpful_acc = delta_r20 >= 0.0
    verdict = "PASS — feedback improves at least F1 by ≥ 0.01" if helpful_f1 \
                else ("NEUTRAL — no F1 gain ≥ 0.01" if abs(delta_f1) < 0.01
                          else "FAIL — feedback hurts F1")

    payload = {
        "city": city,
        "intent_mode": intent_mode,
        "K_sit": K_sit,
        "n_boundary_with_z_prev": int(valid_boundary.sum()),
        "boundary_F1_no_fb": float(f1_no_fb),
        "boundary_F1_with_fb": float(f1_with_fb),
        "delta_F1": delta_f1,
        "delta_K_boundary_no_fb": avg_delta_no_fb_boundary,
        "delta_K_boundary_with_fb": avg_delta_with_fb_boundary,
        "R20_B_blind": r20_blind,
        "R20_xsage_no_fb": r20_no_fb,
        "R20_xsage_with_fb": r20_with_fb,
        "delta_R20_fb": delta_r20,
        "verdict": verdict,
        "tex_decision": ("keep eq.18 in main" if helpful_f1
                           else "demote eq.18 to design extension"),
    }
    pd.DataFrame([payload]).to_csv(out_dir / "feedback_eval.csv", index=False)
    (out_dir / "verdict.json").write_text(json.dumps(payload, indent=2),
                                              encoding="utf-8")
    print(f"  verdict: {verdict}")
    print(f"  tex decision: {payload['tex_decision']}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", choices=["NYC", "TKY", "both"], default="both")
    args = parser.parse_args()
    cities = ["NYC", "TKY"] if args.city == "both" else [args.city]
    results = []
    for c in cities:
        results.append(run_city(c, intent_mode="hard"))
        if c == "TKY":
            results.append(run_city(c, intent_mode="all"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
