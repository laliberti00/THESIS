"""Round-3 B4 — user × situation profiling suite.

Pure aggregation on Stage A fit + backbone scores + the A1/A1bis sink
re-ranking at the knee operating point.

Outputs (per city):
  outputs/round3/B4/<city>/user_profiles.parquet
  outputs/round3/B4/<city>/archetypes.csv
  outputs/round3/B4/<city>/archetypes_2d.png
  outputs/round3/B4/<city>/user_fairness.csv
  outputs/round3/B4/<city>/summary.md

NYC uses mask-mode situations (sinks {6, 7}, knee κ=1.0).
TKY uses keep-mode situations (sinks {4, 5}, knee κ=2.0) — the lens
perspective per C2 two-perspective decision.
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
from pipeline.step02_models.xsage.metrics import long_tail_groups, topk_from_scores
from pipeline.step02_models.xsage.orchestrator import _load_city


K_TOP = 20
CONFIG = {
    "NYC": {"sit_dir": "outputs/NYC/xsage_transit_mask/situations",
              "sinks": [6, 7], "knee_kappa": 1.0, "mode": "mask"},
    "TKY": {"sit_dir": "outputs/TKY/xsage/situations",
              "sinks": [4, 5], "knee_kappa": 2.0, "mode": "keep"},
}


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a 1D non-negative array."""
    if len(values) == 0: return 0.0
    sorted_v = np.sort(values).astype(np.float64)
    n = len(sorted_v); total = sorted_v.sum()
    if total == 0: return 0.0
    cum = np.cumsum(sorted_v)
    return float((n + 1 - 2 * cum.sum() / total) / n)


def _topk(scores: np.ndarray, exclude: sps.csr_matrix,
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


def run_city(city: str) -> dict:
    cfg = CONFIG[city]
    out_dir = REPO_ROOT / "outputs" / "round3" / "B4" / city
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> B4 on {city} ({cfg['mode']} mode, sinks={cfg['sinks']}, "
          f"κ={cfg['knee_kappa']})")

    ds = _load_city(city)
    fit = np.load(REPO_ROOT / cfg["sit_dir"] / "fit.npz", allow_pickle=True)
    df_test = ds["df_test"]; df_val = ds["df_val"]
    u_test = df_test["u_idx"].values.astype(np.int64)
    u_val = df_val["u_idx"].values.astype(np.int64)
    i_test = df_test["i_idx"].values.astype(np.int64)
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    z_val = np.asarray(fit["core_label_val"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)
    isb_val = np.asarray(fit["is_boundary_val"]).astype(bool)
    K_sit = int(max(z_test.max(), z_val.max()) + 1)

    n_items = ds["n_items"]

    # (1) Per-user situation mix π_u — aggregate test + val for stability
    all_u = np.concatenate([u_val, u_test])
    all_z = np.concatenate([z_val, z_test])
    all_b = np.concatenate([isb_val, isb_test])
    unique_users = np.unique(all_u)
    pi_u = np.zeros((len(unique_users), K_sit), dtype=np.float32)
    boundary_rate = np.zeros(len(unique_users), dtype=np.float32)
    n_requests = np.zeros(len(unique_users), dtype=np.int32)
    sink_set = set(cfg["sinks"])
    sink_exposure_share = np.zeros(len(unique_users), dtype=np.float32)
    for i, u in enumerate(unique_users):
        m = all_u == u
        counts = np.bincount(all_z[m], minlength=K_sit).astype(np.float32)
        if counts.sum() > 0:
            pi_u[i] = counts / counts.sum()
        boundary_rate[i] = float(all_b[m].mean())
        n_requests[i] = int(m.sum())
        in_sink = np.isin(all_z[m], list(sink_set))
        sink_exposure_share[i] = float(in_sink.mean())

    # (2) User archetypes via k-means cosine, k by silhouette
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    # L2-normalise rows for cosine
    norms = np.linalg.norm(pi_u, axis=1, keepdims=True)
    pi_u_norm = pi_u / np.maximum(norms, 1e-9)
    best_k = None; best_sil = -1.0
    for kk in range(3, 9):
        km = KMeans(n_clusters=kk, n_init=10, random_state=42)
        labs = km.fit_predict(pi_u_norm)
        if len(set(labs)) < 2: continue
        sil = float(silhouette_score(pi_u_norm, labs, metric="cosine"))
        if sil > best_sil:
            best_sil = sil; best_k = kk
    print(f"  silhouette sweep: best k = {best_k}, silhouette = {best_sil:.3f}")
    silhouette_acceptable = best_sil >= 0.15
    if silhouette_acceptable:
        km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        archetype_labels = km.fit_predict(pi_u_norm)
        archetype_centers = km.cluster_centers_
    else:
        print("  silhouette < 0.15 — declaring 'no stable user archetypes'")
        archetype_labels = np.zeros(len(unique_users), dtype=np.int32)
        archetype_centers = pi_u_norm.mean(axis=0, keepdims=True)

    # Save archetype centers + names (top-2 situations + boundary rate)
    archetype_rows = []
    if silhouette_acceptable:
        for a in range(best_k):
            cm = archetype_labels == a
            n_in = int(cm.sum())
            top = np.argsort(archetype_centers[a])[::-1][:2]
            archetype_rows.append({
                "archetype_id": int(a),
                "n_users": n_in,
                "top_situations": top.tolist(),
                "top_shares": [float(archetype_centers[a, top[0]]),
                                  float(archetype_centers[a, top[1]])],
                "mean_boundary_rate": float(boundary_rate[cm].mean()),
                "mean_sink_exposure": float(sink_exposure_share[cm].mean()),
            })
    pd.DataFrame(archetype_rows).to_csv(out_dir / "archetypes.csv", index=False)

    # (3) User profiles parquet
    profiles_df = pd.DataFrame({
        "user_id_internal": unique_users,
        "n_requests": n_requests,
        "boundary_rate": boundary_rate,
        "sink_exposure_share": sink_exposure_share,
        "archetype": archetype_labels,
    })
    for k in range(K_sit):
        profiles_df[f"pi_s{k}"] = pi_u[:, k]
    profiles_df.to_parquet(out_dir / "user_profiles.parquet")

    # (4) Per-user fairness — LT received per user before and after the
    # sink re-ranking at the knee operating point
    scores_blind_uitem = load_or_refit(city, model_name="FM")
    scores_blind = scores_blind_uitem[u_test]
    excl_test = excluded_mask(city, n_items)
    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(axis=0)).ravel()
    _, G1_mask = long_tail_groups(pop, short_head_share=0.20)
    boost = G1_mask.astype(np.float32)

    sink_mask_t = np.isin(z_test, cfg["sinks"])
    core_sink_t = sink_mask_t & ~isb_test
    scores_on = scores_blind.copy()
    scores_on[np.where(core_sink_t)[0]] += cfg["knee_kappa"] * boost[None, :]

    print(f"  computing top-20 lists (OFF and ON) ...")
    top_off = _topk(scores_blind, excl_test, u_test)
    top_on = _topk(scores_on, excl_test, u_test)
    lt_off_per_request = G1_mask[top_off].mean(axis=1)
    lt_on_per_request = G1_mask[top_on].mean(axis=1)

    # Aggregate per user
    user_lt_off = np.zeros(len(unique_users), dtype=np.float32)
    user_lt_on = np.zeros(len(unique_users), dtype=np.float32)
    user_n_test = np.zeros(len(unique_users), dtype=np.int32)
    for i, u in enumerate(unique_users):
        m = u_test == u
        if m.any():
            user_lt_off[i] = float(lt_off_per_request[m].mean())
            user_lt_on[i] = float(lt_on_per_request[m].mean())
            user_n_test[i] = int(m.sum())
    # Only over users with ≥ 1 test request
    valid = user_n_test > 0
    gini_off = _gini(user_lt_off[valid])
    gini_on = _gini(user_lt_on[valid])
    print(f"  Gini of LT-received: OFF = {gini_off:.4f}  ON = {gini_on:.4f}  "
          f"Δ = {gini_on - gini_off:+.4f}")

    fairness_rows = []
    for i, u in enumerate(unique_users):
        if not valid[i]: continue
        fairness_rows.append({
            "user_id_internal": int(u),
            "n_test_requests": int(user_n_test[i]),
            "sink_exposure_share": float(sink_exposure_share[i]),
            "lt_received_off": float(user_lt_off[i]),
            "lt_received_on": float(user_lt_on[i]),
            "lt_delta": float(user_lt_on[i] - user_lt_off[i]),
            "archetype": int(archetype_labels[i]),
        })
    pd.DataFrame(fairness_rows).to_csv(out_dir / "user_fairness.csv", index=False)

    # 2D plot of archetypes (PCA on pi_u)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pi_centered = pi_u - pi_u.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(pi_centered, full_matrices=False)
    X2 = pi_centered @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.tab10
    for a in range(int(archetype_labels.max() + 1)):
        m = archetype_labels == a
        ax.scatter(X2[m, 0], X2[m, 1], s=20, alpha=0.7,
                     color=cmap(a % 10), label=f"archetype {a} (n={int(m.sum())})")
    ax.set_xlabel("PC 1 of π_u"); ax.set_ylabel("PC 2 of π_u")
    ax.set_title(f"{city} — user archetypes (silhouette {best_sil:.3f})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / "archetypes_2d.png", dpi=140); plt.close(fig)

    summary = {
        "city": city, "mode": cfg["mode"], "sinks": cfg["sinks"],
        "knee_kappa": cfg["knee_kappa"],
        "n_users": int(len(unique_users)),
        "K_sit": K_sit,
        "best_archetype_k": int(best_k) if best_k is not None else 0,
        "best_silhouette": best_sil,
        "stable_archetypes": bool(silhouette_acceptable),
        "mean_boundary_rate": float(boundary_rate.mean()),
        "mean_sink_exposure_share": float(sink_exposure_share.mean()),
        "gini_lt_off": gini_off,
        "gini_lt_on": gini_on,
        "delta_gini": gini_on - gini_off,
        "mean_lt_off": float(user_lt_off[valid].mean()),
        "mean_lt_on": float(user_lt_on[valid].mean()),
        "delta_mean_lt": float(user_lt_on[valid].mean() - user_lt_off[valid].mean()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
    (out_dir / "summary.md").write_text(
        f"# B4 profiling summary — {city}\n\n"
        f"- mode: {cfg['mode']}, sinks: {cfg['sinks']}, knee κ: {cfg['knee_kappa']}\n"
        f"- users profiled: {summary['n_users']}\n"
        f"- silhouette best k = {summary['best_archetype_k']} "
        f"(silhouette {summary['best_silhouette']:.3f})  "
        f"→ {'STABLE' if summary['stable_archetypes'] else 'no stable archetypes'}\n"
        f"- mean boundary rate per user: {summary['mean_boundary_rate']:.3f}\n"
        f"- mean sink-exposure share per user: {summary['mean_sink_exposure_share']:.3f}\n\n"
        f"## User-side fairness (LT received per user)\n\n"
        f"| metric | OFF (B_blind) | ON (X-SAGE rerank) | Δ |\n"
        f"|---|---:|---:|---:|\n"
        f"| mean LT@20 per user | {summary['mean_lt_off']:.4f} | "
        f"{summary['mean_lt_on']:.4f} | {summary['delta_mean_lt']:+.4f} |\n"
        f"| Gini of LT-received | {summary['gini_lt_off']:.4f} | "
        f"{summary['gini_lt_on']:.4f} | {summary['delta_gini']:+.4f} |\n",
        encoding="utf-8")
    print(f"  wrote {out_dir}")
    return summary


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
