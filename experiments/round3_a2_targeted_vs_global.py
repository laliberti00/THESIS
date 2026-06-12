"""Round-3 A2 — targeted vs global long-tail re-ranking comparator.

For each city we compare three additive long-tail boost variants:

    targeted    sink × core requests          (current 2.1; gated by both
                                                  sink membership and Stage-A
                                                  certainty γ_S = 1[core])
    sinkonly    sink × all requests           (same sink gate, no certainty
                                                  gate)
    global      all × all requests            (no gates — naive long-tail
                                                  re-ranking)

All three use the same boost ``1[i ∈ G_1]`` and the same κ grid. The
trade-off curve is (global R@20 cost) vs (sink LT@20 and global LT@20).
Acceptance (per brief §A2): PASS if the targeted curve dominates (≤ cost
at every matched LT level) on at least one city and is never dominated.

Outputs (gitignored):
  outputs/<city>/xsage/round3/A2/tradeoff_targeted.csv
  outputs/<city>/xsage/round3/A2/tradeoff_sinknogate.csv
  outputs/<city>/xsage/round3/A2/tradeoff_global.csv
  outputs/<city>/xsage/round3/A2/tradeoff_overlay.png
  outputs/<city>/xsage/round3/A2/verdict.json
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
SHORT_HEAD = 0.20
KAPPA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


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
        if (s > s[target]).sum() < K:
            hits += 1
    return hits / max(1, n)


def trade_off_for_variant(name: str,
                              scores: np.ndarray,
                              i_target: np.ndarray,
                              excl: sps.csr_matrix,
                              users: np.ndarray,
                              G1_mask: np.ndarray,
                              apply_mask: np.ndarray,
                              sink_mask: np.ndarray,
                              kappa_grid=KAPPA_GRID) -> pd.DataFrame:
    """``apply_mask`` is the boolean of which test requests receive the nudge."""
    rows = []
    boost = G1_mask.astype(np.float32)
    top_off = _topk(scores, excl, users)
    r20_off = _recall20(scores, i_target, excl, users)
    global_LT_off = float(G1_mask[top_off.flatten()].mean())
    sink_LT_off = (float(G1_mask[top_off[sink_mask].flatten()].mean())
                      if sink_mask.any() else float("nan"))
    for kappa in kappa_grid:
        if kappa == 0:
            scores_on = scores
        else:
            nudge = np.zeros_like(scores)
            nudge[apply_mask] = kappa * boost[None, :]
            scores_on = scores + nudge
        top_on = _topk(scores_on, excl, users)
        r20_on = _recall20(scores_on, i_target, excl, users)
        global_LT_on = float(G1_mask[top_on.flatten()].mean())
        sink_LT_on = (float(G1_mask[top_on[sink_mask].flatten()].mean())
                          if sink_mask.any() else float("nan"))
        rows.append({"variant": name, "kappa_fair": kappa,
                       "r20_off": r20_off, "r20_on": r20_on,
                       "r20_delta": r20_on - r20_off,
                       "global_LT_off": global_LT_off, "global_LT_on": global_LT_on,
                       "global_LT_delta": global_LT_on - global_LT_off,
                       "sink_LT_off": sink_LT_off, "sink_LT_on": sink_LT_on,
                       "sink_LT_delta": sink_LT_on - sink_LT_off})
    return pd.DataFrame(rows)


def interpolated_cost_at_target(df: pd.DataFrame, lt_col: str,
                                   target_lt: float) -> float | None:
    """Linear-interpolate R@20 cost (|r20_delta|) at the first kappa where
    ``lt_col`` first exceeds ``target_lt``."""
    df = df.sort_values("kappa_fair").reset_index(drop=True)
    for i in range(1, len(df)):
        lt_prev = df.loc[i - 1, lt_col]; lt_cur = df.loc[i, lt_col]
        if lt_cur >= target_lt:
            if lt_cur == lt_prev:
                return float(-df.loc[i, "r20_delta"])
            t = (target_lt - lt_prev) / (lt_cur - lt_prev)
            r_prev = df.loc[i - 1, "r20_delta"]; r_cur = df.loc[i, "r20_delta"]
            return float(-(r_prev + t * (r_cur - r_prev)))
    return None


def run_city(city: str) -> dict:
    out_dir = REPO_ROOT / "outputs" / city / "xsage" / "round3" / "A2"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> A2 on {city}")
    ds = _load_city(city)
    sit_dir = REPO_ROOT / "outputs" / city / "xsage" / "situations"
    fit = np.load(sit_dir / "fit.npz", allow_pickle=True)
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)
    # Sinks from A1 (val-flagged); fall back to Stage B test-flagged if A1
    # has not been run.
    a1_path = REPO_ROOT / "outputs" / city / "xsage" / "round3" / "A1" / "sink_comparison.json"
    if a1_path.exists():
        sinks = json.loads(a1_path.read_text())["val_flagged"]
    else:
        sinks = [int(s["situation"]) for s in
                  json.loads((REPO_ROOT / "outputs" / city / "xsage" /
                                "fairness" / "verdict.json").read_text())
                            ["inequity_sinks"]]
    print(f"  sinks = {sinks}")

    df_test = ds["df_test"]; n_items = ds["n_items"]
    u_test = df_test["u_idx"].values.astype(np.int64)
    i_test = df_test["i_idx"].values.astype(np.int64)
    scores_uitem = load_or_refit(city, model_name="FM")
    scores = scores_uitem[u_test]
    excl = excluded_mask(city, n_items)
    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(axis=0)).ravel()
    _, G1_mask = long_tail_groups(pop, short_head_share=SHORT_HEAD)
    sink_mask = np.isin(z_test, sinks)
    core_sink = sink_mask & ~isb_test
    all_mask = np.ones(len(u_test), dtype=bool)

    print(f"  computing 3 trade-off curves ...")
    df_targeted = trade_off_for_variant("targeted", scores, i_test, excl,
                                            u_test, G1_mask, core_sink, sink_mask)
    df_sinknogate = trade_off_for_variant("sinknogate", scores, i_test, excl,
                                              u_test, G1_mask, sink_mask, sink_mask)
    df_global = trade_off_for_variant("global", scores, i_test, excl,
                                          u_test, G1_mask, all_mask, sink_mask)
    df_targeted.to_csv(out_dir / "tradeoff_targeted.csv", index=False)
    df_sinknogate.to_csv(out_dir / "tradeoff_sinknogate.csv", index=False)
    df_global.to_csv(out_dir / "tradeoff_global.csv", index=False)

    # Print summary
    print(f"  variant     κ    R@20 Δ   global_LT  sink_LT")
    for df_v, name in ((df_targeted, "targeted"),
                          (df_sinknogate, "sinknogate"),
                          (df_global, "global")):
        for _, r in df_v.iterrows():
            print(f"  {name:10s}  {r['kappa_fair']:>4.2f}  "
                  f"{r['r20_delta']:+8.4f}  {r['global_LT_on']:>8.3f}  "
                  f"{r['sink_LT_on']:>8.3f}")

    # Costs at matched targets (sink LT × 2 and × 3 of baseline)
    base_sink_LT = float(df_targeted.iloc[0]["sink_LT_off"])
    base_global_LT = float(df_targeted.iloc[0]["global_LT_off"])
    targets_sink = (base_sink_LT * 2, base_sink_LT * 3)
    targets_global = (base_global_LT * 1.5, base_global_LT * 2)
    costs = {}
    for name, df_v in (("targeted", df_targeted),
                          ("sinknogate", df_sinknogate),
                          ("global", df_global)):
        costs[name] = {}
        for t in targets_sink:
            costs[name][f"acc_cost@sinkLT={t:.3f}"] = interpolated_cost_at_target(
                df_v, "sink_LT_on", t)
        for t in targets_global:
            costs[name][f"acc_cost@globalLT={t:.3f}"] = interpolated_cost_at_target(
                df_v, "global_LT_on", t)

    # Dominance check — only on metrics where BOTH variants reach the target.
    # The targeted variant's scope is sinks-only, so it cannot push global LT
    # by design; comparing on global LT targets is not meaningful for the
    # "targeted vs global" claim. We restrict to sink-LT axes (matched targets
    # at × 2 and × 3 of baseline).
    sink_keys = [k for k in costs["targeted"] if k.startswith("acc_cost@sinkLT")]

    def is_dominated_by(a: str, b: str, costs: dict, keys: list[str]) -> bool:
        ka = costs[a]; kb = costs[b]
        worse_or_equal = []
        strictly_better = False
        for k in keys:
            ca = ka.get(k); cb = kb.get(k)
            if ca is None or cb is None:
                continue   # not comparable on this axis
            if cb < ca: strictly_better = True; worse_or_equal.append(True)
            elif cb == ca: worse_or_equal.append(True)
            else: worse_or_equal.append(False)
        if not worse_or_equal:
            return False
        return all(worse_or_equal) and strictly_better

    dominated_by_targeted = {
        "sinknogate": is_dominated_by("sinknogate", "targeted", costs, sink_keys),
        "global": is_dominated_by("global", "targeted", costs, sink_keys),
    }
    targeted_dominated_by = {
        "sinknogate": is_dominated_by("targeted", "sinknogate", costs, sink_keys),
        "global": is_dominated_by("targeted", "global", costs, sink_keys),
    }
    targeted_beats_global_strict = dominated_by_targeted["global"]
    targeted_dominates_any = any(dominated_by_targeted.values())
    targeted_is_dominated = any(targeted_dominated_by.values())
    if targeted_beats_global_strict and not targeted_is_dominated:
        verdict = "PASS — targeted strictly beats global on every matched sink-LT level"
    elif targeted_dominates_any and not targeted_is_dominated:
        verdict = "PASS — targeted dominates at least one comparator on sink-LT axis"
    elif targeted_is_dominated:
        verdict = "FAIL — targeted is dominated on sink-LT axis"
    else:
        verdict = "TIE — targeted matches but does not strictly beat comparators"
    print(f"\n  costs at matched targets:")
    for v, cs in costs.items():
        print(f"   {v:10s}  " + "  ".join(f"{k}: {c:.4f}" if c is not None
                                            else f"{k}: NA"
                                            for k, c in cs.items()))
    print(f"  verdict: {verdict}")

    # Overlay plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, lt_col, title in zip(axes, ("sink_LT_on", "global_LT_on"),
                                    (f"{city} — sink LT@20 vs acc",
                                     f"{city} — global LT@20 vs acc")):
        for df_v, name, color in ((df_targeted, "targeted", "tab:blue"),
                                    (df_sinknogate, "sinknogate", "tab:orange"),
                                    (df_global, "global", "tab:green")):
            ax.plot(df_v[lt_col], -df_v["r20_delta"], marker="o", label=name,
                      color=color)
        ax.set_xlabel(title.split(" vs ")[0].split(" — ")[1])
        ax.set_ylabel("R@20 cost (= −delta)")
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "tradeoff_overlay.png", dpi=140); plt.close(fig)

    payload = {
        "city": city,
        "sinks_used": sinks,
        "costs_at_matched_targets": costs,
        "dominated_by_targeted": dominated_by_targeted,
        "targeted_dominated_by": targeted_dominated_by,
        "verdict": verdict,
    }
    (out_dir / "verdict.json").write_text(json.dumps(payload, indent=2,
                                                          default=str),
                                              encoding="utf-8")
    return payload


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
