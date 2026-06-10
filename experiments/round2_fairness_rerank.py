"""Round-2 2.1 — situation-gated long-tail re-ranking.

Uses the additive mechanism, pointed at exposure inequities in the
inequity-sink situations flagged by Stage B (s6 on NYC; s4, s5 on TKY).
The fairness nudge is applied **only on core requests** (γ_S=1) of sink
situations:

    ŝ_fair(u, i) = s_B(u, i) + κ_fair · γ_S(v) · sink(z) · boost_LT(i)

with ``boost_LT(i) = 1[i ∈ G_1]`` (long-tail indicator). Sweep κ_fair and
plot the fairness–accuracy trade-off curve.

Headline to deliver:
    "in sink situations, long-tail exposure rises from X% to Y% at a cost
     of Z R@20."
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.xsage.backbone import excluded_mask, load_or_refit
from pipeline.step02_models.xsage.metrics import (
    kl_divergence, long_tail_groups, long_tail_ratio, topk_from_scores,
)
from pipeline.step02_models.xsage.orchestrator import _load_city


def _topk_per_request(scores: np.ndarray, exclude: sps.csr_matrix,
                        users: np.ndarray, K: int = 20) -> np.ndarray:
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
                 exclude: sps.csr_matrix, users: np.ndarray, K: int = 20) -> float:
    hits = 0; n = scores.shape[0]
    for b in range(n):
        u = int(users[b])
        s = scores[b].copy()
        cols = exclude.indices[exclude.indptr[u]:exclude.indptr[u + 1]]
        if len(cols):
            s[cols] = -np.inf
        target = int(targets[b])
        ts = s[target]
        rank = int((s > ts).sum()) + 1
        if rank <= K:
            hits += 1
    return hits / max(1, n)


def run_rerank(city: str, kappa_grid=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
                 K_top: int = 20, verbose: bool = False) -> dict:
    out_dir = REPO_ROOT / "outputs" / city / "xsage" / "fairness"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage A artefacts
    sit_dir = REPO_ROOT / "outputs" / city / "xsage" / "situations"
    fit = np.load(sit_dir / "fit.npz", allow_pickle=True)
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)

    # Load Stage B sink list
    lens_path = REPO_ROOT / "outputs" / city / "xsage" / "fairness" / "verdict.json"
    if not lens_path.exists():
        raise FileNotFoundError(f"Stage B verdict missing at {lens_path}; "
                                  "run Stage B first.")
    lens = json.loads(lens_path.read_text())
    sinks = [int(s["situation"]) for s in lens.get("inequity_sinks", [])]
    if not sinks:
        raise RuntimeError(f"{city}: Stage B flagged NO sinks — rerank has no target.")
    print(f"  flagged sinks for {city}: {sinks}")

    # Backbone
    print(f"  loading backbone (FM-vanilla) scores ...")
    scores_B_uitem = load_or_refit(city, model_name="FM")
    ds = _load_city(city)
    u_test = ds["df_test"]["u_idx"].values.astype(np.int64)
    i_target = ds["df_test"]["i_idx"].values.astype(np.int64)
    scores_B = scores_B_uitem[u_test]                                 # (n_test, n_items)

    excl = excluded_mask(city, ds["n_items"])
    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(axis=0)).ravel()
    G0_mask, G1_mask = long_tail_groups(pop, short_head_share=0.20)
    boost_LT = G1_mask.astype(np.float32)                            # (n_items,)

    sink_mask = np.isin(z_test, sinks)
    core_sink_mask = sink_mask & ~isb_test                            # apply on core sinks only

    # OFF top-K
    top_off = _topk_per_request(scores_B, excl, u_test, K=K_top)

    # global metrics
    global_dist = np.bincount(top_off.flatten(),
                                 minlength=ds["n_items"]).astype(np.float64)
    global_dist /= max(global_dist.sum(), 1.0)
    global_LT = float(G1_mask[top_off.flatten()].mean())

    rows = []
    for kappa in kappa_grid:
        nudge = np.zeros_like(scores_B)
        if kappa > 0:
            # ŝ_fair = s_B + κ · γ_S · sink · boost_LT
            nudge_rows = np.where(core_sink_mask)[0]
            nudge[nudge_rows] = kappa * boost_LT[None, :]
        scores_on = scores_B + nudge
        top_on = _topk_per_request(scores_on, excl, u_test, K=K_top)

        # Sink-only LT and KL
        sink_off_items = top_off[sink_mask].flatten()
        sink_on_items = top_on[sink_mask].flatten()
        sink_LT_off = float(G1_mask[sink_off_items].mean()) if len(sink_off_items) else float("nan")
        sink_LT_on = float(G1_mask[sink_on_items].mean()) if len(sink_on_items) else float("nan")
        sink_dist_on = np.bincount(sink_on_items,
                                       minlength=ds["n_items"]).astype(np.float64)
        sink_dist_on /= max(sink_dist_on.sum(), 1.0)
        sink_KL_on = kl_divergence(sink_dist_on, global_dist)

        # Accuracy cost: R@20 globally and on sinks
        r20_global = _recall20(scores_on, i_target, excl, u_test, K=K_top)
        r20_sink = _recall20(scores_on[sink_mask], i_target[sink_mask],
                                excl, u_test[sink_mask], K=K_top)
        r20_blind_global = _recall20(scores_B, i_target, excl, u_test, K=K_top)
        r20_blind_sink = _recall20(scores_B[sink_mask], i_target[sink_mask],
                                       excl, u_test[sink_mask], K=K_top)
        rows.append({
            "kappa_fair": kappa,
            "sink_LT_off": sink_LT_off, "sink_LT_on": sink_LT_on,
            "sink_LT_delta": sink_LT_on - sink_LT_off,
            "sink_KL_on": sink_KL_on,
            "r20_global": r20_global, "r20_global_delta": r20_global - r20_blind_global,
            "r20_sink": r20_sink, "r20_sink_delta": r20_sink - r20_blind_sink,
        })
        print(f"  κ={kappa:>5.2f}  sink LT: {sink_LT_off:.3f} → {sink_LT_on:.3f}  "
              f"sink R@20: {r20_blind_sink:.4f} → {r20_sink:.4f} "
              f"({r20_sink-r20_blind_sink:+.4f})  global R@20 Δ={r20_global-r20_blind_global:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "reranking_tradeoff.csv", index=False)

    # Plot trade-off
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["kappa_fair"], df["sink_LT_on"], marker="o", label="sink LT@20 (ON)")
    ax.axhline(df["sink_LT_off"].iloc[0], linestyle="--", color="grey",
                 label=f"sink LT (OFF) = {df['sink_LT_off'].iloc[0]:.3f}")
    ax.set_xlabel("κ_fair")
    ax.set_ylabel("long-tail exposure in sinks")
    ax.set_title(f"{city} — sink LT vs κ_fair")
    ax2 = ax.twinx()
    ax2.plot(df["kappa_fair"], df["r20_global_delta"], marker="x",
              color="darkred", linestyle=":", label="Δ R@20 global")
    ax2.plot(df["kappa_fair"], df["r20_sink_delta"], marker="s",
              color="indianred", linestyle="-.", label="Δ R@20 sink")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("ΔR@20 vs B_blind")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "reranking_tradeoff.png", dpi=140)
    plt.close(fig)
    print(f"  saved {out_dir / 'reranking_tradeoff.csv'} + .png")
    return {"sinks": sinks, "table": df.to_dict("records")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", choices=["NYC", "TKY", "both"], default="both")
    parser.add_argument("--kappa", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    cities = ["NYC", "TKY"] if args.city == "both" else [args.city]
    for city in cities:
        print(f"\n>>> Fairness re-ranking on {city}")
        run_rerank(city, kappa_grid=tuple(args.kappa), verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
