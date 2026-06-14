"""Round-3 B9 — statistical hardening for the two HIGH-priority gaps.

Closes the audit (STATISTICAL_VALIDATION_AUDIT.md) gap items:

  B9.1 (HIGH)  C6 P-iv per-target-macro paired tests + propagated
               aggregate CI from per-stratum SEs (TKY_BAL).
               Also re-runs the same hardening for the original C5.0
               law on NYC and TKY for completeness.

  B9.2 (HIGH)  C5 ablation paired Wilcoxon + Holm step-down across
               5 variants vs B_blind, using the per-request arrays
               saved by the re-run of round3_c5_bfull_ablation.

Both subtasks consume per-request hit arrays only — no retraining
beyond C5's already-running variant re-run.

Outputs:
  outputs/round3/B9/{c6_pivot_paired.json, c5_paired_holm.json,
                       c50_paired_holm.json, summary.md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.xsage.backbone import excluded_mask, load_or_refit
from pipeline.step02_models.xsage.orchestrator import _load_city
from pipeline.step04_statistical_validation.statistical_validation import (
    paired_wilcoxon, percentile_bootstrap_ci, harmonic_mean_p
)

K_TOP = 20
TT_MACRO = "Travel & Transport"


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _per_request_R20(scores: np.ndarray, u_test: np.ndarray,
                       i_target: np.ndarray, excl: sps.csr_matrix,
                       K: int = K_TOP) -> np.ndarray:
    """Per-request R@20 hit (0/1) array given per-request score matrix.
    `scores` shape (B, n_items)."""
    out = np.zeros(scores.shape[0], dtype=np.float32)
    for b in range(scores.shape[0]):
        u = int(u_test[b])
        s = scores[b].copy()
        cols = excl.indices[excl.indptr[u]:excl.indptr[u + 1]]
        if len(cols):
            s[cols] = -np.inf
        tgt = int(i_target[b])
        if (s > s[tgt]).sum() < K:
            out[b] = 1.0
    return out


def _per_request_N20(scores: np.ndarray, u_test: np.ndarray,
                       i_target: np.ndarray, excl: sps.csr_matrix,
                       K: int = K_TOP) -> np.ndarray:
    out = np.zeros(scores.shape[0], dtype=np.float32)
    for b in range(scores.shape[0]):
        u = int(u_test[b])
        s = scores[b].copy()
        cols = excl.indices[excl.indptr[u]:excl.indptr[u + 1]]
        if len(cols):
            s[cols] = -np.inf
        tgt = int(i_target[b])
        rank = int((s > s[tgt]).sum()) + 1
        if rank <= K:
            out[b] = float(1.0 / np.log2(rank + 1))
    return out


def _holm_step_down(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down on a list of raw p-values.
    Returns adjusted p in the original order."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adj = [0.0] * n
    prev = 0.0
    for rank, i in enumerate(order):
        a = (n - rank) * p_values[i]
        a = min(1.0, max(prev, a))
        prev = a
        adj[i] = a
    return adj


def _paired_test_block(diff: np.ndarray, label: str) -> dict:
    """Wilcoxon + bootstrap CI95 + signed t-stat on a paired diff array."""
    n = len(diff)
    mean = float(diff.mean())
    if n == 0:
        return {"label": label, "n": 0, "mean": 0.0, "wilcoxon_p": 1.0,
                "ci95_lo": 0.0, "ci95_hi": 0.0}
    ci_lo, ci_hi = percentile_bootstrap_ci(diff, n_boot=10_000)
    # paired Wilcoxon (sign-rank)
    w = paired_wilcoxon(diff, np.zeros_like(diff))
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "label": label, "n": n, "mean": mean, "se": se,
        "ci95_lo": float(ci_lo), "ci95_hi": float(ci_hi),
        "wilcoxon_stat": float(w["statistic"]),
        "wilcoxon_p": float(w["p_value"]),
    }


# -----------------------------------------------------------------------------
# B9.1 — C5.0 / C6 per-macro paired tests + propagated aggregate
# -----------------------------------------------------------------------------

def stratified_c50_or_c6(city: str, label_for_report: str) -> dict:
    """For one city (NYC / TKY / TKY_BAL):
      * load B_blind FM scores (cached) and B_full per-request scores
      * compute per-request R@20 hits for each
      * paired Wilcoxon + bootstrap CI95 per stratum (T&T / non-T&T)
      * Holm step-down across the 2 strata + the aggregate (3 tests)
      * propagate per-stratum SEs to a closed-form CI95 on the aggregate
        Δ via Δ_agg = s_TT · Δ_TT + s_nonTT · Δ_nonTT
      * compare to the bootstrap CI95 on the aggregate (sanity check)
    """
    print(f"\n>>> B9.1 stratified paired tests on {label_for_report}")
    ds = _load_city(city)
    df_test = ds["df_test"]
    u_test = df_test["u_idx"].values.astype(np.int64)
    i_target = df_test["i_idx"].values.astype(np.int64)
    cat_target = df_test["cat_macro"].values.astype(str)
    n_items = ds["n_items"]
    excl = excluded_mask(city, n_items)

    scores_blind = load_or_refit(city, model_name="FM")[u_test]
    bfull_path = REPO_ROOT / "outputs" / city / "xsage" / "backbone" / "Bfull.scores.npy"
    scores_full = np.load(bfull_path)
    r20_b = _per_request_R20(scores_blind, u_test, i_target, excl)
    r20_f = _per_request_R20(scores_full, u_test, i_target, excl)

    tt_mask = cat_target == TT_MACRO
    diff = r20_f - r20_b  # B_full - B_blind, per-request

    blocks = {}
    blocks["TT"] = _paired_test_block(diff[tt_mask], f"{label_for_report}_TT")
    blocks["nonTT"] = _paired_test_block(diff[~tt_mask],
                                            f"{label_for_report}_nonTT")
    blocks["all"] = _paired_test_block(diff, f"{label_for_report}_all")

    raw_ps = [blocks["TT"]["wilcoxon_p"],
                blocks["nonTT"]["wilcoxon_p"],
                blocks["all"]["wilcoxon_p"]]
    adj_ps = _holm_step_down(raw_ps)
    blocks["TT"]["holm_p"] = adj_ps[0]
    blocks["nonTT"]["holm_p"] = adj_ps[1]
    blocks["all"]["holm_p"] = adj_ps[2]
    for k in ("TT", "nonTT", "all"):
        blocks[k]["holm_reject_at_0.05"] = bool(blocks[k]["holm_p"] < 0.05)

    # Pool shares
    s_TT = float(tt_mask.mean())
    s_nonTT = 1.0 - s_TT
    # Closed-form propagated aggregate from per-macro means + SEs
    pred_agg_mean = s_TT * blocks["TT"]["mean"] + s_nonTT * blocks["nonTT"]["mean"]
    pred_agg_se = float(np.sqrt(
        (s_TT * blocks["TT"]["se"]) ** 2 +
        (s_nonTT * blocks["nonTT"]["se"]) ** 2)) if blocks["TT"]["se"] == blocks["TT"]["se"] else float("nan")
    pred_agg_ci_lo = pred_agg_mean - 1.96 * pred_agg_se
    pred_agg_ci_hi = pred_agg_mean + 1.96 * pred_agg_se

    out = {
        "city": city,
        "label": label_for_report,
        "n_test": int(len(u_test)),
        "share_TT": s_TT,
        "share_nonTT": s_nonTT,
        "per_macro": blocks,
        "law_check": {
            "predicted_aggregate_mean": pred_agg_mean,
            "predicted_aggregate_se": pred_agg_se,
            "predicted_aggregate_ci95": [pred_agg_ci_lo, pred_agg_ci_hi],
            "measured_aggregate_mean": blocks["all"]["mean"],
            "measured_aggregate_ci95": [blocks["all"]["ci95_lo"],
                                            blocks["all"]["ci95_hi"]],
            "abs_difference": float(abs(pred_agg_mean
                                          - blocks["all"]["mean"])),
        },
    }
    for k in ("TT", "nonTT", "all"):
        b = blocks[k]
        print(f"  {k:6s}: n={b['n']:5d}  Δ={b['mean']:+.4f}  "
              f"CI95 [{b['ci95_lo']:+.4f}, {b['ci95_hi']:+.4f}]  "
              f"Wilcoxon p={b['wilcoxon_p']:.2e}  Holm p={b['holm_p']:.2e}  "
              f"reject={b['holm_reject_at_0.05']}")
    print(f"  LAW: predicted Δ_agg = {pred_agg_mean:+.5f} "
          f"± 1.96·{pred_agg_se:.5f}  "
          f"CI95 [{pred_agg_ci_lo:+.5f}, {pred_agg_ci_hi:+.5f}]")
    print(f"      measured Δ_agg  = {blocks['all']['mean']:+.5f}  "
          f"CI95 [{blocks['all']['ci95_lo']:+.5f}, "
          f"{blocks['all']['ci95_hi']:+.5f}]  "
          f"|diff| = {abs(pred_agg_mean - blocks['all']['mean']):.5f}")
    return out


# -----------------------------------------------------------------------------
# B9.2 — C5 ablation paired Wilcoxon + Holm
# -----------------------------------------------------------------------------

def c5_paired_holm() -> dict:
    """For TKY, paired Wilcoxon (vs B_blind) + Holm step-down over the 5
    ablated variants. Requires per-request arrays saved by the re-run of
    round3_c5_bfull_ablation."""
    print(f"\n>>> B9.2 C5 ablation paired Wilcoxon + Holm on TKY")
    perreq_dir = REPO_ROOT / "outputs" / "round3" / "C5" / "per_request"
    variants = ("M_full", "M_minus_time", "M_minus_geo",
                  "M_minus_fine", "M_minus_intent")

    # Need TKY B_blind per-request R@20
    city = "TKY"
    ds = _load_city(city)
    df_test = ds["df_test"]
    u_test = df_test["u_idx"].values.astype(np.int64)
    i_target = df_test["i_idx"].values.astype(np.int64)
    n_items = ds["n_items"]
    excl = excluded_mask(city, n_items)
    scores_blind = load_or_refit(city, model_name="FM")[u_test]
    r20_blind = _per_request_R20(scores_blind, u_test, i_target, excl)
    n20_blind = _per_request_N20(scores_blind, u_test, i_target, excl)
    cat_target = df_test["cat_macro"].values.astype(str)
    tt_mask = cat_target == TT_MACRO

    # Load each variant
    results = {}
    raw_ps_r20 = []
    raw_ps_n20 = []
    blocks_per_variant = {}
    for v in variants:
        p = perreq_dir / f"TKY_{v}.npz"
        if not p.exists():
            print(f"  WARN missing {p} — C5 re-run not finished yet")
            return {"status": "WAIT_FOR_C5_RERUN", "missing": str(p)}
        f = np.load(p)
        r20_v = np.asarray(f["R20_per_request"]).astype(np.float32)
        n20_v = np.asarray(f["N20_per_request"]).astype(np.float32)
        diff_r = r20_v - r20_blind
        diff_n = n20_v - n20_blind
        block_r = _paired_test_block(diff_r, f"{v}_vs_blind_R20_all")
        block_r["mean_v"] = float(r20_v.mean())
        block_r["mean_b"] = float(r20_blind.mean())
        block_n = _paired_test_block(diff_n, f"{v}_vs_blind_N20_all")
        block_n["mean_v"] = float(n20_v.mean())
        block_n["mean_b"] = float(n20_blind.mean())
        # Per-stratum (T&T vs non-T&T) — for understanding the mechanism
        block_r_TT = _paired_test_block(diff_r[tt_mask],
                                            f"{v}_vs_blind_R20_TT")
        block_r_nonTT = _paired_test_block(diff_r[~tt_mask],
                                                f"{v}_vs_blind_R20_nonTT")
        blocks_per_variant[v] = {
            "R20_all": block_r, "N20_all": block_n,
            "R20_TT": block_r_TT, "R20_nonTT": block_r_nonTT,
        }
        raw_ps_r20.append(block_r["wilcoxon_p"])
        raw_ps_n20.append(block_n["wilcoxon_p"])

    adj_r20 = _holm_step_down(raw_ps_r20)
    adj_n20 = _holm_step_down(raw_ps_n20)
    for i, v in enumerate(variants):
        blocks_per_variant[v]["R20_all"]["holm_p"] = adj_r20[i]
        blocks_per_variant[v]["R20_all"]["holm_reject_at_0.05"] = bool(
            adj_r20[i] < 0.05)
        blocks_per_variant[v]["N20_all"]["holm_p"] = adj_n20[i]
        blocks_per_variant[v]["N20_all"]["holm_reject_at_0.05"] = bool(
            adj_n20[i] < 0.05)

    # Print summary
    print(f"  {'variant':18s}  {'Δ R@20':>9s}  CI95-R@20             "
          f"wilcoxon p   Holm p     reject?")
    for v in variants:
        b = blocks_per_variant[v]["R20_all"]
        print(f"  {v:18s}  {b['mean']:>+9.4f}  "
              f"[{b['ci95_lo']:+.4f}, {b['ci95_hi']:+.4f}]   "
              f"{b['wilcoxon_p']:.2e}   {b['holm_p']:.2e}   "
              f"{b['holm_reject_at_0.05']}")
    return {
        "status": "OK",
        "city": "TKY",
        "blind_R20": float(r20_blind.mean()),
        "blind_N20": float(n20_blind.mean()),
        "variants": blocks_per_variant,
        "raw_p_R20": raw_ps_r20, "holm_p_R20": adj_r20,
        "raw_p_N20": raw_ps_n20, "holm_p_N20": adj_n20,
    }


# -----------------------------------------------------------------------------

def main() -> int:
    out_dir = REPO_ROOT / "outputs" / "round3" / "B9"
    out_dir.mkdir(parents=True, exist_ok=True)

    # B9.1a — original C5.0 law on NYC and TKY (uses saved B_full)
    c50_nyc = stratified_c50_or_c6("NYC", "C5.0_NYC")
    c50_tky = stratified_c50_or_c6("TKY", "C5.0_TKY")
    # B9.1b — C6 P-iv on TKY_BAL
    c6_bal = stratified_c50_or_c6("TKY_BAL", "C6_TKY_BAL")

    with open(out_dir / "c50_paired_holm.json", "w") as f:
        json.dump({"NYC": c50_nyc, "TKY": c50_tky}, f, indent=2, default=str)
    with open(out_dir / "c6_pivot_paired.json", "w") as f:
        json.dump(c6_bal, f, indent=2, default=str)

    # B9.2 — C5 ablation (waits for the re-run if not done)
    c5_paired = c5_paired_holm()
    with open(out_dir / "c5_paired_holm.json", "w") as f:
        json.dump(c5_paired, f, indent=2, default=str)

    # summary.md
    lines = ["# B9 statistical hardening summary", ""]
    lines.append("## B9.1 — C5.0 / C6 per-target-macro paired tests (Holm-adjusted)")
    lines.append("")
    for label, blob in [("NYC (C5.0)", c50_nyc),
                          ("TKY (C5.0)", c50_tky),
                          ("TKY_BAL (C6)", c6_bal)]:
        lines.append(f"### {label}  (n={blob['n_test']}, T&T share={blob['share_TT']:.3f})")
        lines.append("")
        lines.append("| stratum | n | mean Δ | CI95 | Wilcoxon p | Holm p | reject |")
        lines.append("|---|---|---|---|---|---|---|")
        for k in ("TT", "nonTT", "all"):
            b = blob["per_macro"][k]
            lines.append(
                f"| {k} | {b['n']} | {b['mean']:+.5f} | "
                f"[{b['ci95_lo']:+.5f}, {b['ci95_hi']:+.5f}] | "
                f"{b['wilcoxon_p']:.2e} | {b['holm_p']:.2e} | "
                f"**{b['holm_reject_at_0.05']}** |"
            )
        lc = blob["law_check"]
        lines.append("")
        lines.append(
            f"**Law-check.** Predicted aggregate Δ (from per-macro means × "
            f"pool shares) = {lc['predicted_aggregate_mean']:+.5f} ± 1.96·"
            f"{lc['predicted_aggregate_se']:.5f}  → CI95 "
            f"[{lc['predicted_aggregate_ci95'][0]:+.5f}, "
            f"{lc['predicted_aggregate_ci95'][1]:+.5f}]. Measured = "
            f"{lc['measured_aggregate_mean']:+.5f}, CI95 "
            f"[{lc['measured_aggregate_ci95'][0]:+.5f}, "
            f"{lc['measured_aggregate_ci95'][1]:+.5f}]. "
            f"|prediction − measurement| = **{lc['abs_difference']:.5f}**."
        )
        lines.append("")

    lines.append("## B9.2 — C5 ablation paired Wilcoxon + Holm on TKY")
    lines.append("")
    if c5_paired.get("status") == "WAIT_FOR_C5_RERUN":
        lines.append(f"_C5 re-run not finished yet ({c5_paired['missing']})_")
    else:
        lines.append(f"B_blind R@20 = {c5_paired['blind_R20']:.4f}  "
                       f"(NDCG@20 = {c5_paired['blind_N20']:.4f})")
        lines.append("")
        lines.append("| variant | Δ R@20 | CI95 | Wilcoxon p | Holm p | reject H₀ |")
        lines.append("|---|---|---|---|---|---|")
        for v, blocks in c5_paired["variants"].items():
            r = blocks["R20_all"]
            lines.append(
                f"| {v} | {r['mean']:+.4f} | "
                f"[{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] | "
                f"{r['wilcoxon_p']:.2e} | {r['holm_p']:.2e} | "
                f"**{r['holm_reject_at_0.05']}** |"
            )
        lines.append("")
        lines.append("Per-stratum mechanism (R@20 vs B_blind, stratified):")
        lines.append("")
        lines.append("| variant | Δ T&T | CI95 T&T | Δ non-T&T | CI95 non-T&T |")
        lines.append("|---|---|---|---|---|")
        for v, blocks in c5_paired["variants"].items():
            t = blocks["R20_TT"]; n_ = blocks["R20_nonTT"]
            lines.append(
                f"| {v} | {t['mean']:+.4f} | "
                f"[{t['ci95_lo']:+.4f}, {t['ci95_hi']:+.4f}] | "
                f"{n_['mean']:+.4f} | "
                f"[{n_['ci95_lo']:+.4f}, {n_['ci95_hi']:+.4f}] |"
            )
    (out_dir / "summary.md").write_text("\n".join(lines))
    print(f"\nwrote {out_dir}/summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
