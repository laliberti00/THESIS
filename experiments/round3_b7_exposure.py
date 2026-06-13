"""Round-3 B7 — position-weighted exposure of the sink re-ranking.

For each city, at the val-selected knee operating point (same as B5):
  - NYC: mask-mode (sinks {6, 7}), knee κ_fair = 1.0
  - TKY: keep-mode (sinks {4, 5}), knee κ_fair = 2.0

Joins B_blind test top-20 with the fairness-re-ranked top-20 and reports
**position-discounted** long-tail exposure metrics — answers the reviewer
question "is the long-tail you grant *effective* exposure, or just
rank-tail filler?".

Outputs (per city):
  outputs/<city>/<sit_root>/round3/B7/{
      exposure_metrics.csv,
      exposure_at_k.png,
      entry_rank_hist.png,
      provider_coverage.csv,
      verdict.json,
  }

`<sit_root>` matches B5 / B4 convention:
  NYC → xsage_transit_mask    TKY → xsage
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.xsage.backbone import excluded_mask, load_or_refit
from pipeline.step02_models.xsage.metrics import long_tail_groups, topk_from_scores
from pipeline.step02_models.xsage.orchestrator import _load_city


K_TOP = 20
K_HEAD = 5
SHORT_HEAD = 0.20

# Position-discount: NDCG/Singh-Joachims style log discount.
RANK_WEIGHTS = 1.0 / np.log2(np.arange(1, K_TOP + 1) + 1)
W_SUM = RANK_WEIGHTS.sum()

EXPOSURE_AT_K = [3, 5, 10, 15, 20]


CONFIG = {
    "NYC": {
        "sit_dir": "outputs/NYC/xsage_transit_mask/situations",
        "out_root": "outputs/NYC/xsage_transit_mask/round3/B7",
        "sinks": [6, 7],
        "knee_kappa": 1.0,
        "mode_label": "mask",
    },
    "TKY": {
        "sit_dir": "outputs/TKY/xsage/situations",
        "out_root": "outputs/TKY/xsage/round3/B7",
        "sinks": [4, 5],
        "knee_kappa": 2.0,
        "mode_label": "keep",
    },
}


# ----------------------------- helpers ---------------------------------------

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


def _ndcg_at(top_k_items: np.ndarray, target: int, k: int) -> float:
    """NDCG@k with single relevant item: 1/log2(rank+1) if hit, else 0."""
    arr = top_k_items[:k]
    hits = np.where(arr == target)[0]
    if len(hits) == 0:
        return 0.0
    return float(1.0 / np.log2(hits[0] + 2))


def _flat_lt(top: np.ndarray, G1: np.ndarray) -> np.ndarray:
    """Per-request flat LT@20 = #(top-20 items ∈ G1) / 20."""
    return G1[top].mean(axis=1)


def _disc_lt(top: np.ndarray, G1: np.ndarray) -> np.ndarray:
    """Per-request position-discounted LT exposure (NDCG-style)."""
    masks = G1[top].astype(np.float32)
    num = (masks * RANK_WEIGHTS[None, :]).sum(axis=1)
    return num / W_SUM


def _lt_at_k(top: np.ndarray, G1: np.ndarray, k: int) -> np.ndarray:
    return G1[top[:, :k]].mean(axis=1)


# ----------------------------- main per-city ---------------------------------

def run_city(city: str) -> dict:
    cfg = CONFIG[city]
    out_dir = REPO_ROOT / cfg["out_root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> B7 on {city} ({cfg['mode_label']}, sinks={cfg['sinks']}, "
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
    print(f"  test n={n_test}  sink share={sink_mask.mean():.3f}  "
          f"core∩sink share={core_sink.mean():.3f}")

    top_off = _topk(scores_blind, excl, u_test)
    scores_on = scores_blind.copy()
    scores_on[np.where(core_sink)[0]] += cfg["knee_kappa"] * boost[None, :]
    top_on = _topk(scores_on, excl, u_test)

    # actually-touched mask (matches B5 — TKY has core∩sink items where the
    # kappa boost is insufficient to displace any popular item; those lists
    # are unchanged. We define `touched` as "list materially differs", not
    # "list was eligible for the boost".
    changed = np.zeros(n_test, dtype=bool)
    for b in range(n_test):
        if not np.array_equal(top_off[b], top_on[b]):
            inter = len(np.intersect1d(top_off[b], top_on[b], assume_unique=False))
            if inter < K_TOP:
                changed[b] = True
    n_touched = int(changed.sum())
    print(f"  touched lists: {n_touched} / {n_test}  "
          f"({n_touched / n_test:.4f})")

    # ----------------- (1) flat LT@20 + (2) discounted LT@20 -----------------
    flat_off = _flat_lt(top_off, G1_mask)
    flat_on = _flat_lt(top_on, G1_mask)
    disc_off = _disc_lt(top_off, G1_mask)
    disc_on = _disc_lt(top_on, G1_mask)

    metrics_rows = []
    for label, mask in [("global_all", np.ones(n_test, bool)),
                        ("touched_only", changed)]:
        if mask.sum() == 0:
            continue
        metrics_rows.append({
            "scope": label,
            "n": int(mask.sum()),
            "flat_LT20_off": float(flat_off[mask].mean()),
            "flat_LT20_on": float(flat_on[mask].mean()),
            "flat_LT20_delta": float((flat_on - flat_off)[mask].mean()),
            "disc_LT20_off": float(disc_off[mask].mean()),
            "disc_LT20_on": float(disc_on[mask].mean()),
            "disc_LT20_delta": float((disc_on - disc_off)[mask].mean()),
        })
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(out_dir / "exposure_metrics.csv", index=False)

    # ratio (touched-only is where the action lives)
    touched_row = next((r for r in metrics_rows
                        if r["scope"] == "touched_only"), None)
    if touched_row and touched_row["flat_LT20_delta"] != 0:
        ratio = touched_row["disc_LT20_delta"] / touched_row["flat_LT20_delta"]
    else:
        ratio = float("nan")

    # ----------------- (3) exposure-at-k curve ------------------------------
    rows_at_k = []
    for k in EXPOSURE_AT_K:
        lt_off_k = _lt_at_k(top_off, G1_mask, k)
        lt_on_k = _lt_at_k(top_on, G1_mask, k)
        rows_at_k.append({
            "k": k,
            "LT_off_global": float(lt_off_k.mean()),
            "LT_on_global": float(lt_on_k.mean()),
            "delta_global": float((lt_on_k - lt_off_k).mean()),
            "LT_off_touched": float(lt_off_k[changed].mean()) if n_touched else 0.0,
            "LT_on_touched": float(lt_on_k[changed].mean()) if n_touched else 0.0,
            "delta_touched": float((lt_on_k - lt_off_k)[changed].mean())
                              if n_touched else 0.0,
        })
    df_at_k = pd.DataFrame(rows_at_k)
    df_at_k.to_csv(out_dir / "exposure_at_k.csv", index=False)

    # find separation point: smallest k where touched-only Δ exceeds 1/20
    sep_k = None
    for r in rows_at_k:
        if r["delta_touched"] >= 1.0 / K_TOP:
            sep_k = r["k"]
            break

    # plot
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ks = [r["k"] for r in rows_at_k]
    ax.plot(ks, [r["LT_off_touched"] for r in rows_at_k],
            "o-", label="B_blind (touched)", color="#444")
    ax.plot(ks, [r["LT_on_touched"] for r in rows_at_k],
            "o-", label="X-SAGE rerank (touched)", color="#c33")
    ax.plot(ks, [r["LT_off_global"] for r in rows_at_k],
            "s--", label="B_blind (global)", color="#888")
    ax.plot(ks, [r["LT_on_global"] for r in rows_at_k],
            "s--", label="X-SAGE rerank (global)", color="#e88")
    ax.set_xlabel("cutoff k")
    ax.set_ylabel("LT@k (fraction of long-tail items in top-k)")
    ax.set_title(f"{city} — exposure-at-k ({cfg['mode_label']}, "
                 f"sinks={cfg['sinks']}, κ={cfg['knee_kappa']})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "exposure_at_k.png", dpi=130)
    plt.close(fig)

    # ----------------- (4) entry-rank distribution --------------------------
    entry_ranks = []  # ranks 1..20 of LT items that ENTER under rerank
    exit_ranks = []   # ranks 1..20 of items that EXIT under rerank
    for b in np.where(changed)[0]:
        off = list(top_off[b])
        on = list(top_on[b])
        set_off = set(map(int, off))
        set_on = set(map(int, on))
        for r, it in enumerate(on):
            if int(it) not in set_off:
                if G1_mask[int(it)]:
                    entry_ranks.append(r + 1)
        for r, it in enumerate(off):
            if int(it) not in set_on:
                exit_ranks.append(r + 1)

    median_entry_rank = float(np.median(entry_ranks)) if entry_ranks else None
    median_exit_rank = float(np.median(exit_ranks)) if exit_ranks else None

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bins = np.arange(0.5, 21.5, 1.0)
    ax.hist(entry_ranks, bins=bins, alpha=0.6, label=f"entries (LT) n={len(entry_ranks)}",
            color="#c33")
    ax.hist(exit_ranks, bins=bins, alpha=0.45, label=f"exits n={len(exit_ranks)}",
            color="#444")
    ax.set_xlabel("rank in rer-ranked list")
    ax.set_ylabel("count")
    ax.set_title(f"{city} — entry/exit rank distribution (touched only)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "entry_rank_hist.png", dpi=130)
    plt.close(fig)

    # ----------------- (5) provider coverage --------------------------------
    def _distinct_lt(top: np.ndarray, scope: np.ndarray) -> int:
        if scope.sum() == 0:
            return 0
        items = np.unique(top[scope].reshape(-1))
        return int(G1_mask[items].sum())

    cov_rows = [
        {"scope": "global_all",
         "n_requests": int(n_test),
         "distinct_LT_in_top20_off": _distinct_lt(top_off, np.ones(n_test, bool)),
         "distinct_LT_in_top20_on":  _distinct_lt(top_on,  np.ones(n_test, bool))},
        {"scope": "touched_only",
         "n_requests": int(n_touched),
         "distinct_LT_in_top20_off": _distinct_lt(top_off, changed),
         "distinct_LT_in_top20_on":  _distinct_lt(top_on,  changed)},
        {"scope": "core_sink",
         "n_requests": int(core_sink.sum()),
         "distinct_LT_in_top20_off": _distinct_lt(top_off, core_sink),
         "distinct_LT_in_top20_on":  _distinct_lt(top_on,  core_sink)},
    ]
    cov_rows.append({
        "scope": "total_LT_universe",
        "n_requests": int(G1_mask.sum()),
        "distinct_LT_in_top20_off": int(G1_mask.sum()),
        "distinct_LT_in_top20_on": int(G1_mask.sum()),
    })
    df_cov = pd.DataFrame(cov_rows)
    df_cov["delta"] = (df_cov["distinct_LT_in_top20_on"]
                       - df_cov["distinct_LT_in_top20_off"])
    df_cov.to_csv(out_dir / "provider_coverage.csv", index=False)

    # ----------------- (6) head-accuracy R@5 / NDCG@5 -----------------------
    # head accuracy (off): is target in top-5 of B_blind?
    hit5_off = np.array([int(i_target[b]) in set(top_off[b, :K_HEAD])
                         for b in range(n_test)], dtype=bool)
    hit5_on = np.array([int(i_target[b]) in set(top_on[b, :K_HEAD])
                        for b in range(n_test)], dtype=bool)
    ndcg5_off = np.array([_ndcg_at(top_off[b], int(i_target[b]), K_HEAD)
                          for b in range(n_test)])
    ndcg5_on = np.array([_ndcg_at(top_on[b], int(i_target[b]), K_HEAD)
                         for b in range(n_test)])

    head_metrics = {
        "R5_off_global": float(hit5_off.mean()),
        "R5_on_global": float(hit5_on.mean()),
        "R5_delta_global": float((hit5_on.astype(int) - hit5_off.astype(int)).mean()),
        "NDCG5_off_global": float(ndcg5_off.mean()),
        "NDCG5_on_global": float(ndcg5_on.mean()),
        "NDCG5_delta_global": float((ndcg5_on - ndcg5_off).mean()),
        "R5_off_touched": float(hit5_off[changed].mean()) if n_touched else 0.0,
        "R5_on_touched": float(hit5_on[changed].mean()) if n_touched else 0.0,
        "R5_delta_touched": (float((hit5_on.astype(int) - hit5_off.astype(int))[changed].mean())
                              if n_touched else 0.0),
    }

    # ----------------- verdict ---------------------------------------------
    flat_d = touched_row["flat_LT20_delta"] if touched_row else 0.0
    disc_d = touched_row["disc_LT20_delta"] if touched_row else 0.0
    if abs(flat_d) < 1e-9:
        verdict_label = "no_intervention"
    elif ratio is not None and not np.isnan(ratio):
        # Position-robust: discounted Δ retains ≥ 60% of flat Δ.
        # Tail-only: discounted Δ retains < 25% of flat Δ.
        if ratio >= 0.60:
            verdict_label = "position_robust"
        elif ratio < 0.25:
            verdict_label = "tail_only"
        else:
            verdict_label = "mixed"
    else:
        verdict_label = "undefined"

    verdict = {
        "city": city,
        "mode": cfg["mode_label"],
        "sinks": cfg["sinks"],
        "knee_kappa": cfg["knee_kappa"],
        "n_test": int(n_test),
        "n_touched": int(n_touched),
        "touched_share": float(n_touched / n_test),
        "metrics": metrics_rows,
        "exposure_at_k": rows_at_k,
        "separation_k": sep_k,
        "median_entry_rank": median_entry_rank,
        "median_exit_rank": median_exit_rank,
        "ratio_disc_over_flat_touched": (float(ratio)
                                         if not np.isnan(ratio) else None),
        "provider_coverage": cov_rows,
        "head_metrics": head_metrics,
        "verdict": verdict_label,
    }
    with open(out_dir / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    # ----------------- console summary -------------------------------------
    print(f"  flat LT@20 (touched): {flat_off[changed].mean():.4f} → "
          f"{flat_on[changed].mean():.4f} "
          f"(Δ {flat_d:+.4f})" if n_touched else "  no touched lists")
    print(f"  disc LT@20 (touched): {disc_off[changed].mean():.4f} → "
          f"{disc_on[changed].mean():.4f} "
          f"(Δ {disc_d:+.4f})" if n_touched else "")
    print(f"  ratio disc/flat = {ratio:.3f}  →  verdict = {verdict_label}")
    print(f"  separation k = {sep_k}")
    if median_entry_rank is not None:
        print(f"  median entry rank (LT) = {median_entry_rank}  "
              f"median exit rank = {median_exit_rank}")
    print(f"  R@5  : {head_metrics['R5_off_global']:.4f} → "
          f"{head_metrics['R5_on_global']:.4f}  "
          f"(Δ {head_metrics['R5_delta_global']:+.5f})")
    print(f"  NDCG@5: {head_metrics['NDCG5_off_global']:.4f} → "
          f"{head_metrics['NDCG5_on_global']:.4f}  "
          f"(Δ {head_metrics['NDCG5_delta_global']:+.5f})")
    return verdict


def main() -> int:
    out_root = REPO_ROOT / "outputs" / "round3" / "B7"
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for city in ("NYC", "TKY"):
        summary[city] = run_city(city)
    # roll-up
    with open(out_root / "verdict.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== B7 ROLL-UP ===")
    for c, v in summary.items():
        print(f"  {c}: verdict={v['verdict']}  ratio={v['ratio_disc_over_flat_touched']}  "
              f"sep_k={v['separation_k']}  med_entry={v['median_entry_rank']}  "
              f"ΔR@5={v['head_metrics']['R5_delta_global']:+.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
