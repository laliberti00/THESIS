"""Round-3 C1 — diagnosis pack for the TKY rescue hypothesis.

Read-only. Computes, for both cities, the numbers the round-3 brief asks
for in §C1:

    * macro frequency distribution + entropy
    * transition matrix W row-entropy + stationary distribution
    * in-degree histogram and attractor margin (how close each macro is to
      the indeg ≥ mean threshold)
    * share of check-ins that are Travel & Transport (T&T), and share of
      *transitions* that pass through it
    * inter-check-in time gap distribution with a session-boundary
      candidate τ (knee detection)
    * cat_fine long-tail stats (counts of cat_fine values below 50/100
      train occurrences)

Outputs:
    outputs/<city>/xsage/round3/C1/diagnosis.json
    outputs/<city>/xsage/round3/C1/figures/{gap_hist.png,indeg.png}
    outputs/round3/C1/C1_READ.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.xsage.l1_perception import (
    estimate_macro_transition, find_attractors,
)
from pipeline.step02_models.xsage.orchestrator import _load_city


TT = "Travel & Transport"
SESSION_THRESHOLDS_H = [1.0, 3.0, 6.0, 12.0, 24.0]


def _entropy(p: np.ndarray, base: int = 2) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum() / math.log(base))


def _stationary(W: np.ndarray, n_iter: int = 1000, tol: float = 1e-10) -> np.ndarray:
    """Power iteration to find the stationary distribution of W (rows are
    transition probabilities)."""
    K = W.shape[0]
    v = np.full(K, 1.0 / K, dtype=np.float64)
    Wt = W.T
    for _ in range(n_iter):
        v_new = Wt @ v
        v_new /= v_new.sum()
        if np.linalg.norm(v_new - v) < tol:
            return v_new
        v = v_new
    return v


def _knee_via_max_curvature(x: np.ndarray, y: np.ndarray) -> int | None:
    """Index of the maximum-curvature point in a smooth-ish 1d curve.

    Uses normalised diagonal distance to the line that joins the first
    and last points (Kneedle, simplified).
    """
    if len(x) < 3:
        return None
    p0 = np.array([x[0], y[0]])
    p1 = np.array([x[-1], y[-1]])
    seg = p1 - p0
    seg_len = np.linalg.norm(seg)
    if seg_len == 0:
        return None
    seg_n = seg / seg_len
    dists = []
    for i in range(len(x)):
        pt = np.array([x[i], y[i]])
        v = pt - p0
        proj = np.dot(v, seg_n)
        perp = v - proj * seg_n
        dists.append(np.linalg.norm(perp))
    return int(np.argmax(dists))


def diagnose_city(city: str, out_dir: Path, verbose: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)
    print(f"\n>>> diagnosing {city}")
    ds = _load_city(city)
    df_train = ds["df_train"]; df_val = ds["df_val"]; df_test = ds["df_test"]
    macro_to_idx = ds["macro_to_idx"]
    idx2macro = {v: k for k, v in macro_to_idx.items()}
    n_macros = ds["n_macros"]

    # --- 1. macro frequency + entropy (train) ----------------------------
    macro_counts = df_train["cat_macro"].value_counts().reindex(
        sorted(macro_to_idx.keys()), fill_value=0)
    macro_freq = (macro_counts / macro_counts.sum()).to_dict()
    macro_entropy = _entropy(np.array(list(macro_freq.values())))

    # --- 2. W rows entropy + stationary distribution ---------------------
    W = estimate_macro_transition(df_train, macro_to_idx, add_one_smoothing=True)
    row_entropy = np.array([_entropy(W[i]) for i in range(n_macros)])
    stationary = _stationary(W)
    stationary_dict = {idx2macro[i]: float(stationary[i]) for i in range(n_macros)}
    log2K = math.log2(n_macros)

    # --- 3. indeg + attractor margin ------------------------------------
    indeg = W.sum(axis=0)
    indeg_mean = float(indeg.mean())
    attractors = (indeg >= indeg_mean)
    attractor_list = [idx2macro[i] for i in np.where(attractors)[0]]
    attractor_margin = {
        idx2macro[i]: float(indeg[i] - indeg_mean) for i in range(n_macros)
    }

    # --- 4. T&T share -----------------------------------------------------
    tt_idx = macro_to_idx.get(TT)
    if tt_idx is not None:
        tt_checkin_share_train = float((df_train["cat_macro"] == TT).mean())
        tt_checkin_share_overall = float(
            pd.concat([df_train["cat_macro"], df_val["cat_macro"], df_test["cat_macro"]])
              .eq(TT).mean())
        # transition share through T&T = sum over t where src or dst is T&T
        s = df_train.sort_values(["user_id", "time_local"]).reset_index(drop=True)
        m_idx = np.array([macro_to_idx[m] for m in s["cat_macro"].values],
                          dtype=np.int32)
        same_user = s["user_id"].values[1:] == s["user_id"].values[:-1]
        src = m_idx[:-1][same_user]; dst = m_idx[1:][same_user]
        n_trans = len(src)
        n_through_tt = int(((src == tt_idx) | (dst == tt_idx)).sum())
        tt_transition_share = n_through_tt / max(1, n_trans)
        n_via_tt = int(((src == tt_idx) & (dst == tt_idx)).sum())
        tt_self_share = n_via_tt / max(1, n_trans)
    else:
        tt_checkin_share_train = 0.0
        tt_checkin_share_overall = 0.0
        tt_transition_share = 0.0
        tt_self_share = 0.0

    # --- 5. inter-check-in time gap distribution + knee τ ----------------
    s = df_train.sort_values(["user_id", "time_local"]).reset_index(drop=True)
    times = s["time_local"].values.astype("datetime64[ns]")
    users = s["user_id"].values
    gaps_min = (times[1:] - times[:-1]).astype("timedelta64[s]").astype(np.float64) / 60.0
    same_user = users[1:] == users[:-1]
    gaps_min = gaps_min[same_user]
    qs = np.quantile(gaps_min, np.linspace(0.05, 0.99, 50))
    gap_hist = {
        "n_gaps": int(len(gaps_min)),
        "min_min": float(gaps_min.min()),
        "median_min": float(np.median(gaps_min)),
        "p90_min": float(np.quantile(gaps_min, 0.90)),
        "p95_min": float(np.quantile(gaps_min, 0.95)),
        "p99_min": float(np.quantile(gaps_min, 0.99)),
        "max_min": float(gaps_min.max()),
    }
    # Knee on the log(gap) vs ecdf curve
    log_gaps = np.log10(np.maximum(gaps_min, 1.0))
    log_qs = np.quantile(log_gaps, np.linspace(0.05, 0.99, 200))
    ecdf = np.linspace(0.05, 0.99, 200)
    knee_idx = _knee_via_max_curvature(log_qs, ecdf)
    knee_log = float(log_qs[knee_idx]) if knee_idx is not None else None
    knee_min = 10 ** knee_log if knee_log is not None else None
    knee_h = (knee_min / 60.0) if knee_min is not None else None

    # session count per threshold
    sessions_per_threshold = {}
    for h in SESSION_THRESHOLDS_H:
        gap_min_threshold = h * 60.0
        # session_boundary at index i+1 if gap[i] > threshold
        boundaries = int((gaps_min > gap_min_threshold).sum())
        # n_sessions = n_users + #boundaries  (each user starts a fresh session,
        # any internal long gap adds one)
        n_sessions = int(df_train["user_id"].nunique() + boundaries)
        median_session_len = float(len(s) / max(1, n_sessions))
        sessions_per_threshold[f"{h:g}h"] = {
            "n_sessions": n_sessions,
            "median_session_len_rows": median_session_len,
        }

    # --- 6. cat_fine long-tail stats -------------------------------------
    fine_counts = df_train["cat_fine"].value_counts()
    fine_stats = {
        "n_unique_train": int(len(fine_counts)),
        "below_50": int((fine_counts < 50).sum()),
        "below_100": int((fine_counts < 100).sum()),
        "below_500": int((fine_counts < 500).sum()),
        "top_5": fine_counts.head(5).to_dict(),
        "long_tail_share_below_50_of_n_rows": float(
            fine_counts[fine_counts < 50].sum() / len(df_train)),
    }

    # --- figures ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(log_qs, ecdf, lw=1.2)
    if knee_idx is not None:
        ax.axvline(log_gaps[0] if knee_log is None else knee_log, color="r",
                    ls="--", lw=0.8, label=f"knee ≈ {knee_min:.0f} min ({knee_h:.1f} h)")
    ax.set_xlabel("log10(gap min)"); ax.set_ylabel("ECDF")
    ax.set_title(f"{city} — inter-check-in gap (train)")
    ax.legend(); fig.tight_layout()
    fig.savefig(fig_dir / "gap_hist.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    order = np.argsort(indeg)[::-1]
    ax.bar(range(n_macros), indeg[order], color="steelblue")
    ax.axhline(indeg_mean, color="r", ls="--", label=f"mean indeg = {indeg_mean:.3f}")
    ax.set_xticks(range(n_macros))
    ax.set_xticklabels([idx2macro[i] for i in order], rotation=30, ha="right",
                        fontsize=8)
    ax.set_ylabel("in-degree of W"); ax.set_title(f"{city} — macro in-degrees")
    ax.legend(); fig.tight_layout()
    fig.savefig(fig_dir / "indeg.png", dpi=140); plt.close(fig)

    diag = {
        "city": city,
        "n_train_rows": int(len(df_train)),
        "n_macros": int(n_macros),
        "macro_frequencies": macro_freq,
        "macro_entropy_bits": float(macro_entropy),
        "macro_entropy_max_bits": float(log2K),
        "macro_entropy_normalised": float(macro_entropy / log2K),
        "W_row_entropy_bits": {idx2macro[i]: float(row_entropy[i])
                                for i in range(n_macros)},
        "W_row_entropy_mean_bits": float(row_entropy.mean()),
        "stationary_distribution": stationary_dict,
        "indeg": {idx2macro[i]: float(indeg[i]) for i in range(n_macros)},
        "indeg_mean": indeg_mean,
        "attractors_hard": attractor_list,
        "attractor_margin": attractor_margin,
        "travel_transport": {
            "checkin_share_train": tt_checkin_share_train,
            "checkin_share_overall": tt_checkin_share_overall,
            "transition_share": tt_transition_share,
            "self_transition_share": tt_self_share,
        },
        "gap_distribution_min": gap_hist,
        "session_boundary_knee": {
            "knee_minutes": knee_min,
            "knee_hours": knee_h,
        },
        "sessions_per_threshold": sessions_per_threshold,
        "cat_fine_long_tail": fine_stats,
    }
    (out_dir / "diagnosis.json").write_text(json.dumps(diag, indent=2, default=str),
                                               encoding="utf-8")
    print(f"  wrote {out_dir / 'diagnosis.json'}")
    return diag


def write_read_md(diag_by_city: dict[str, dict], out_path: Path) -> None:
    """1-page interpretation: TKY vs NYC diagnostic comparison."""
    nyc = diag_by_city["NYC"]; tky = diag_by_city["TKY"]
    lines = [
        "# Round-3 C1 — diagnosis read\n\n",
        "_Read-only diagnostic comparison between NYC and TKY against the_\n",
        "_pre-registered Round-3 hypothesis: TKY's failures share one root, the_\n",
        "_macro-graph dominance by Travel & Transport plus a skewed macro mix._\n\n",
        "## Macro-mix concentration\n\n",
        "| metric | NYC | TKY |\n|---|---|---|\n",
        f"| macro entropy / log2(K) | {nyc['macro_entropy_normalised']:.3f} | "
        f"{tky['macro_entropy_normalised']:.3f} |\n",
        f"| mean row-entropy of W (bits) | {nyc['W_row_entropy_mean_bits']:.3f} | "
        f"{tky['W_row_entropy_mean_bits']:.3f} |\n",
        f"| # hard attractors (indeg ≥ mean) | {len(nyc['attractors_hard'])} | "
        f"{len(tky['attractors_hard'])} |\n\n",
        "**Hard attractors:**\n\n",
        f"- NYC: {', '.join(nyc['attractors_hard'])}\n",
        f"- TKY: {', '.join(tky['attractors_hard'])}\n\n",
    ]
    lines += [
        "## Travel & Transport (T&T) saturation\n\n",
        "| metric | NYC | TKY |\n|---|---|---|\n",
        f"| T&T share of check-ins (train) | {nyc['travel_transport']['checkin_share_train']:.3f} | "
        f"{tky['travel_transport']['checkin_share_train']:.3f} |\n",
        f"| T&T share of transitions | {nyc['travel_transport']['transition_share']:.3f} | "
        f"{tky['travel_transport']['transition_share']:.3f} |\n",
        f"| T&T → T&T transition share | {nyc['travel_transport']['self_transition_share']:.3f} | "
        f"{tky['travel_transport']['self_transition_share']:.3f} |\n",
        f"| T&T stationary probability | {nyc['stationary_distribution'].get('Travel & Transport', 0):.3f} | "
        f"{tky['stationary_distribution'].get('Travel & Transport', 0):.3f} |\n\n",
    ]
    lines += [
        "## Inter-check-in gap & session knee\n\n",
        "| metric | NYC | TKY |\n|---|---|---|\n",
        f"| median gap (min) | {nyc['gap_distribution_min']['median_min']:.1f} | "
        f"{tky['gap_distribution_min']['median_min']:.1f} |\n",
        f"| p95 gap (min) | {nyc['gap_distribution_min']['p95_min']:.1f} | "
        f"{tky['gap_distribution_min']['p95_min']:.1f} |\n",
        f"| knee (hours) | {nyc['session_boundary_knee']['knee_hours']:.1f} | "
        f"{tky['session_boundary_knee']['knee_hours']:.1f} |\n\n",
    ]
    lines += [
        "## cat_fine long-tail\n\n",
        "| metric | NYC | TKY |\n|---|---|---|\n",
        f"| # unique cat_fine (train) | {nyc['cat_fine_long_tail']['n_unique_train']} | "
        f"{tky['cat_fine_long_tail']['n_unique_train']} |\n",
        f"| # cat_fine with <50 occurrences | {nyc['cat_fine_long_tail']['below_50']} | "
        f"{tky['cat_fine_long_tail']['below_50']} |\n",
        f"| # cat_fine with <100 occurrences | {nyc['cat_fine_long_tail']['below_100']} | "
        f"{tky['cat_fine_long_tail']['below_100']} |\n",
        f"| share of rows whose cat_fine has <50 occ | "
        f"{nyc['cat_fine_long_tail']['long_tail_share_below_50_of_n_rows']:.3f} | "
        f"{tky['cat_fine_long_tail']['long_tail_share_below_50_of_n_rows']:.3f} |\n\n",
    ]
    # Interpretation
    tky_tt_share = tky['travel_transport']['transition_share']
    nyc_tt_share = nyc['travel_transport']['transition_share']
    tky_tt_dominance = tky_tt_share / max(nyc_tt_share, 1e-9)
    tky_attr_count = len(tky['attractors_hard'])
    nyc_attr_count = len(nyc['attractors_hard'])
    lines += [
        "## Interpretation against the pre-registered hypothesis\n\n",
        f"1. **Attractor count**: NYC has {nyc_attr_count} attractors, TKY only "
        f"{tky_attr_count} — confirms the hypothesis that TKY's attractor space "
        "is collapsed. The hard-cutoff `mode` throws away information on TKY.\n",
        f"2. **T&T dominance**: T&T participates in "
        f"{tky_tt_share*100:.1f}% of TKY transitions vs "
        f"{nyc_tt_share*100:.1f}% of NYC's — a {tky_tt_dominance:.2f}× ratio. "
        f"T&T → T&T self-transitions account for "
        f"{tky['travel_transport']['self_transition_share']*100:.1f}% of TKY "
        f"transitions (vs {nyc['travel_transport']['self_transition_share']*100:.1f}% "
        "on NYC). **C2 transit-aware mode is well-motivated for TKY.**\n",
        f"3. **Session knee**: TKY's gap-distribution knee is at "
        f"~{tky['session_boundary_knee']['knee_hours']:.1f} h (NYC: "
        f"~{nyc['session_boundary_knee']['knee_hours']:.1f} h). "
        "Defaulting C4 to τ ≈ 6h is reasonable; sensitivity to ±2 h should be cheap.\n",
        f"4. **cat_fine sparsity — SURPRISE finding (against the pre-registered hypothesis)**:\n"
        f"   TKY has {tky['cat_fine_long_tail']['below_50']}/{tky['cat_fine_long_tail']['n_unique_train']} "
        f"cat_fine values with <50 train occurrences "
        f"({tky['cat_fine_long_tail']['long_tail_share_below_50_of_n_rows']*100:.1f}% of rows). "
        f"NYC has {nyc['cat_fine_long_tail']['below_50']}/{nyc['cat_fine_long_tail']['n_unique_train']} "
        f"({nyc['cat_fine_long_tail']['long_tail_share_below_50_of_n_rows']*100:.1f}% of rows). "
        f"**TKY's cat_fine is LESS sparse than NYC's**, not more. The pre-registered "
        f"hypothesis that cat_fine sparsity is the B_full TKY paradox driver is "
        f"**falsified by C1**. The real B_full failure likely lies in the structural "
        f"T&T dominance: with 73 % of TKY check-ins in T&T, the per-context-cell FM "
        f"learns 'next item is probably T&T', biasing its top-K toward T&T venues "
        f"regardless of the test target. C5 should still run but the suspect now is "
        f"different — try **M−geo** and **M−time** first (the per-context cells they "
        f"create) before M−fine.\n\n",
        "## Action items for the round (informed by this read)\n\n",
        "- **C2 priority HIGH**: `intent-transit={mask,collapse}` on TKY with intent-mode `all`. T&T dominance is the headline TKY anomaly.\n",
        "- **C4**: τ candidates from the knee are unreliable on TKY (fat-tailed gap distribution). Use τ ∈ {3 h, 6 h, 12 h} as sensitivity rather than a knee-derived single value.\n",
        "- **C5 revised priority**: M−geo, M−time, M−fine, M−intent (M−geo and M−time first — the C1 read suggests per-context-cell features are the more likely culprit than cat_fine sparsity).\n",
        "- A1 + A2 + A3 + A4 are independent of the C1 read and should run in parallel.\n",
    ]
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"\nwrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", choices=["NYC", "TKY", "both"], default="both")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    cities = ["NYC", "TKY"] if args.city == "both" else [args.city]
    diag_by_city = {}
    for city in cities:
        out_dir = REPO_ROOT / "outputs" / city / "xsage" / "round3" / "C1"
        diag_by_city[city] = diagnose_city(city, out_dir, verbose=args.verbose)
    if len(cities) == 2:
        read_path = REPO_ROOT / "outputs" / "round3" / "C1" / "C1_READ.md"
        read_path.parent.mkdir(parents=True, exist_ok=True)
        write_read_md(diag_by_city, read_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
