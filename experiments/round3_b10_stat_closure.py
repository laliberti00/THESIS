"""Round-3 B10 — statistical-validation closure.

Closes the remaining audit gaps (STATISTICAL_VALIDATION_AUDIT.md):

  B10.1 (HIGH/MED)   Permutation test for B6 (lens backbone-agnostic).
                     Builds the null by random sink-relabel preserving
                     each model's |sinks|; reports observed mean pair-
                     wise Jaccard + p-value over 10 000 permutations.

  B10.2 (PRIORITY)   Family-level statistical-validation summary
                     (STATISTICAL_VALIDATION_SUMMARY.md / .csv) — every
                     family with test used, n, within-family correction,
                     headline result, and the explicit cross-family
                     non-correction policy + deterministic-verifications
                     subsection.

  B10.3 (MED)        Triv add-ons:
                     - B7 ratio CI (bootstrap on disc/flat per request)
                     - B8b Wilson CI95 on cross-seed name-match rates
                     - C2 / Stage-C McNemar p-values (already computed in
                       stage C verdict.json — just collected here)

Outputs:
  outputs/round3/B10/{b6_permutation.json, b7_ratio_ci.json,
                       b8b_wilson_ci.json, c2_mcnemar.json}
  STATISTICAL_VALIDATION_SUMMARY.md
  STATISTICAL_VALIDATION_SUMMARY.csv
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.xsage.backbone import excluded_mask, load_or_refit
from pipeline.step02_models.xsage.metrics import (long_tail_groups,
                                                       topk_from_scores)
from pipeline.step02_models.xsage.orchestrator import _load_city
from pipeline.step04_statistical_validation.statistical_validation import (
    percentile_bootstrap_ci,
)


K_TOP = 20
SHORT_HEAD = 0.20
RANK_WEIGHTS = 1.0 / np.log2(np.arange(1, K_TOP + 1) + 1)
W_SUM = RANK_WEIGHTS.sum()


# -----------------------------------------------------------------------------
# B10.1 — Permutation test on B6 sink agreement
# -----------------------------------------------------------------------------

def _mean_pairwise_jaccard(sink_sets: list[set]) -> float:
    """Mean Jaccard between all unordered pairs of M model sink sets.
    Jaccard(∅, ∅) = 1 by convention; |A∪B|=0 ⇒ set as 1 (won't happen
    here because we exclude Random which is the only one with empty
    sinks)."""
    M = len(sink_sets)
    if M < 2:
        return 1.0
    s = 0.0; n = 0
    for i in range(M):
        for j in range(i + 1, M):
            a = sink_sets[i]; b = sink_sets[j]
            u = a | b; ix = a & b
            jac = len(ix) / len(u) if u else 1.0
            s += jac; n += 1
    return s / n


def permutation_b6(city: str, n_perm: int = 10_000,
                       seed: int = 13) -> dict:
    """Null: each model independently relabels its sinks by drawing
    |sinks| values uniformly at random from {0..K-1}. Random model is
    excluded (it flags ∅ — negative control, not part of the agreement
    family).

    Statistic = mean pairwise Jaccard across the (M-1) non-Random
    models' sink sets.
    """
    print(f"\n>>> B10.1 permutation test for B6 on {city}")
    over = json.loads((REPO_ROOT / "outputs" / "round3" / "B6" / city /
                          "sink_overlap.json").read_text())
    per_model = over["per_model_sinks"]
    # Discover K from union; safer: from situations
    fit = np.load(REPO_ROOT / "outputs" / city / "xsage" / "situations" /
                    "fit.npz", allow_pickle=True)
    K = int(np.asarray(fit["prototypes"]).shape[0])

    models = [m for m in per_model if m != "Random"]
    sink_sets_obs = [set(per_model[m]) for m in models]
    sizes = [len(s) for s in sink_sets_obs]
    obs = _mean_pairwise_jaccard(sink_sets_obs)

    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_perm, dtype=np.float64)
    pool = np.arange(K)
    for t in range(n_perm):
        sets_t = [set(rng.choice(pool, size=k, replace=False).tolist())
                    if k > 0 else set()
                    for k in sizes]
        null_stats[t] = _mean_pairwise_jaccard(sets_t)
    # one-sided p (testing for AGREEMENT > chance)
    n_ge = int((null_stats >= obs).sum())
    # Add 1 in num+denom for finite-sample valid p (Phipson & Smyth).
    p_value = (n_ge + 1) / (n_perm + 1)

    out = {
        "city": city,
        "K_situations": K,
        "models_used": models,
        "per_model_sink_sizes": sizes,
        "n_perm": int(n_perm),
        "seed": int(seed),
        "observed_mean_pairwise_jaccard": float(obs),
        "null_mean": float(null_stats.mean()),
        "null_median": float(np.median(null_stats)),
        "null_p99": float(np.quantile(null_stats, 0.99)),
        "null_p999": float(np.quantile(null_stats, 0.999)),
        "p_value_one_sided_greater": float(p_value),
        "n_perm_ge_observed": int(n_ge),
    }
    print(f"  K={K}  models={models}  sizes={sizes}")
    print(f"  observed = {obs:.4f}  null mean = {null_stats.mean():.4f}  "
          f"null p99 = {np.quantile(null_stats, 0.99):.4f}")
    print(f"  p (one-sided greater) = {p_value:.6f}  (n_perm_ge = {n_ge}/{n_perm})")
    return out


# -----------------------------------------------------------------------------
# B10.3a — B7 disc/flat ratio bootstrap CI on touched subset
# -----------------------------------------------------------------------------

CONFIG_B7 = {
    "NYC": {"sit_dir": "outputs/NYC/xsage_transit_mask/situations",
             "sinks": [6, 7], "knee_kappa": 1.0},
    "TKY": {"sit_dir": "outputs/TKY/xsage/situations",
             "sinks": [4, 5], "knee_kappa": 2.0},
}


def b7_ratio_ci(city: str, n_boot: int = 10_000, seed: int = 17) -> dict:
    """Per-request paired bootstrap CI95 on (disc_LT@20 Δ) and on the
    ratio (disc_LT@20 Δ) / (flat_LT@20 Δ), restricted to touched lists.
    Confirms the headline "position-robust" interval brackets 1.

    Per-request arrays are re-derived inline from saved backbone scores
    (no retraining).
    """
    print(f"\n>>> B10.3a B7 ratio CI on {city}")
    cfg = CONFIG_B7[city]
    fit = np.load(REPO_ROOT / cfg["sit_dir"] / "fit.npz", allow_pickle=True)
    ds = _load_city(city)
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)
    df_test = ds["df_test"]
    u_test = df_test["u_idx"].values.astype(np.int64)
    n_items = ds["n_items"]
    scores_blind = load_or_refit(city, model_name="FM")[u_test]
    excl = excluded_mask(city, n_items)

    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(axis=0)).ravel()
    _, G1 = long_tail_groups(pop, short_head_share=SHORT_HEAD)
    boost = G1.astype(np.float32)

    sink_mask = np.isin(z_test, cfg["sinks"])
    core_sink = sink_mask & ~isb_test
    n_test = len(u_test)
    scores_on = scores_blind.copy()
    scores_on[np.where(core_sink)[0]] += cfg["knee_kappa"] * boost[None, :]

    # top-K
    def _topk(s):
        out = np.zeros((s.shape[0], K_TOP), dtype=np.int32)
        for b in range(s.shape[0]):
            ss = s[b].copy()
            u = int(u_test[b])
            cols = excl.indices[excl.indptr[u]:excl.indptr[u + 1]]
            if len(cols):
                ss[cols] = -np.inf
            out[b] = topk_from_scores(ss, K_TOP)
        return out
    top_off = _topk(scores_blind); top_on = _topk(scores_on)

    flat_off = G1[top_off].mean(axis=1).astype(np.float32)
    flat_on = G1[top_on].mean(axis=1).astype(np.float32)
    disc_off = (G1[top_off].astype(np.float32) * RANK_WEIGHTS[None, :]).sum(axis=1) / W_SUM
    disc_on = (G1[top_on].astype(np.float32) * RANK_WEIGHTS[None, :]).sum(axis=1) / W_SUM

    touched = np.zeros(n_test, dtype=bool)
    for b in range(n_test):
        if not np.array_equal(top_off[b], top_on[b]):
            inter = len(np.intersect1d(top_off[b], top_on[b],
                                          assume_unique=False))
            if inter < K_TOP:
                touched[b] = True
    n_t = int(touched.sum())
    d_flat = (flat_on - flat_off)[touched]
    d_disc = (disc_on - disc_off)[touched]
    mean_flat = float(d_flat.mean()); mean_disc = float(d_disc.mean())
    ratio_point = mean_disc / mean_flat if mean_flat != 0 else float("nan")

    # bootstrap CI on flat Δ, disc Δ, and the ratio
    rng = np.random.default_rng(seed)
    boot_flat = np.empty(n_boot); boot_disc = np.empty(n_boot)
    boot_ratio = np.empty(n_boot)
    for t in range(n_boot):
        idx = rng.integers(0, n_t, size=n_t)
        a = float(d_flat[idx].mean()); b = float(d_disc[idx].mean())
        boot_flat[t] = a; boot_disc[t] = b
        boot_ratio[t] = b / a if a != 0 else float("nan")
    flat_ci = (float(np.percentile(boot_flat, 2.5)),
                float(np.percentile(boot_flat, 97.5)))
    disc_ci = (float(np.percentile(boot_disc, 2.5)),
                float(np.percentile(boot_disc, 97.5)))
    finite = np.isfinite(boot_ratio)
    ratio_ci = (float(np.percentile(boot_ratio[finite], 2.5)),
                  float(np.percentile(boot_ratio[finite], 97.5)))
    out = {
        "city": city, "n_touched": n_t, "n_boot": n_boot, "seed": seed,
        "flat_LT20_delta_mean": mean_flat,
        "flat_LT20_delta_ci95": flat_ci,
        "disc_LT20_delta_mean": mean_disc,
        "disc_LT20_delta_ci95": disc_ci,
        "ratio_disc_over_flat_point": ratio_point,
        "ratio_disc_over_flat_ci95": ratio_ci,
        "ratio_ci_brackets_one": bool(ratio_ci[0] <= 1.0 <= ratio_ci[1]),
    }
    print(f"  ratio point = {ratio_point:.4f}  CI95 [{ratio_ci[0]:.4f}, "
          f"{ratio_ci[1]:.4f}]  brackets 1? {out['ratio_ci_brackets_one']}")
    return out


# -----------------------------------------------------------------------------
# B10.3b — B8b Wilson CI95 on cross-seed name-match rates
# -----------------------------------------------------------------------------

def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score CI95 on a binomial proportion. z = Phi^-1(0.975)."""
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    rad = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - rad), min(1.0, center + rad))


def b8b_wilson() -> dict:
    """Wilson CI95 on each cross-seed stability rate from B8b."""
    print(f"\n>>> B10.3b B8b Wilson CI95 on stability rates")
    src = REPO_ROOT / "outputs" / "round3" / "B8b" / "summary.json"
    summary = json.loads(src.read_text())
    out = {}
    for city, blob in summary.items():
        K = int(blob["K"])
        # alt seeds count by counting per_seed entries
        per_seed = blob["stability_per_seed"]
        n_alt = len(per_seed)
        n_total = K * n_alt
        # aggregate matches across seeds
        agg = {}
        for kind in ("strict", "intent", "time"):
            k_total = int(round(sum(per_seed[s][kind] for s in per_seed) * K))
            lo, hi = _wilson_ci(k_total, n_total)
            agg[kind] = {"matches": k_total, "trials": n_total,
                          "rate": k_total / n_total if n_total else 0.0,
                          "wilson_ci95_lo": lo, "wilson_ci95_hi": hi}
        out[city] = {"K": K, "n_alt_seeds": n_alt, "n_trials": n_total,
                       **agg}
        print(f"  {city} (K={K}, alt seeds={n_alt}, n={n_total}):")
        for kind in ("strict", "intent", "time"):
            a = agg[kind]
            print(f"    {kind:8s}: {a['matches']}/{a['trials']} = "
                  f"{a['rate']:.3f}  CI95 [{a['wilson_ci95_lo']:.3f}, "
                  f"{a['wilson_ci95_hi']:.3f}]")
    return out


# -----------------------------------------------------------------------------
# B10.3c — C2 / Stage-C McNemar (read from existing Stage C verdicts)
# -----------------------------------------------------------------------------

def c2_mcnemar() -> dict:
    """Collect McNemar p-values for T-based vs time-only projection from
    Stage C verdict.json across the 4 paper-lead configs."""
    print(f"\n>>> B10.3c Stage C McNemar collection")
    out = {}
    for tag, path in [
        ("NYC_keep",  "outputs/NYC/xsage/projection/verdict.json"),
        ("NYC_mask",  "outputs/NYC/xsage_transit_mask/projection/verdict.json"),
        ("TKY_keep",  "outputs/TKY/xsage/projection/verdict.json"),
        ("TKY_mask",  "outputs/TKY/xsage_intent_all_transit_mask/projection/verdict.json"),
    ]:
        p = REPO_ROOT / path
        if not p.exists():
            print(f"  {tag}: missing — skip")
            continue
        d = json.loads(p.read_text())
        out[tag] = {
            "delta_F1": float(d["delta_F1"]),
            "n_T_only_correct": int(d["mcnemar_T_only_correct"]),
            "n_time_only_correct": int(d["mcnemar_time_only_correct"]),
            "mcnemar_p_value": float(d["mcnemar_p_value"]),
            "n_test_evaluated": int(d["n_test_evaluated"]),
            "verdict": d.get("verdict", ""),
        }
        print(f"  {tag}: ΔF1={d['delta_F1']:+.3f}  "
              f"T-only={d['mcnemar_T_only_correct']}  "
              f"time-only={d['mcnemar_time_only_correct']}  "
              f"McNemar p={d['mcnemar_p_value']:.4e}")
    return out


# -----------------------------------------------------------------------------
# B10.2 — Family-level summary doc + CSV
# -----------------------------------------------------------------------------

def family_summary(b6_perm: dict, b7_ratio: dict, b8b_wil: dict,
                     c2_mcn: dict) -> tuple[str, pd.DataFrame]:
    rows: list[dict] = []

    # A4 three-way TOST equivalence (matched-OFF + X-SAGE κ=0.1)
    rows.append({
        "family": "A4 three-way TOST",
        "city": "NYC + TKY",
        "test": "TOST (two one-sided)",
        "n_comparisons": 6,
        "within_family_correction": "δ-primary + δ-sensitivity",
        "alpha": 0.05,
        "headline": "matched-OFF identity equivalent at δ=0.005 AND δ=0.0025; X-SAGE κ=0.1 equivalent at δ=0.005, NOT at δ=0.0025 on NYC",
        "p_or_ci": "p_lower / p_upper, both < 0.05 ⇒ equivalent",
        "outputs": "outputs/{NYC,TKY}/xsage/round3/A4/tost.json",
    })
    # A4 three-way Wilcoxon + Holm (vs B_blind and vs B_full)
    rows.append({
        "family": "A4 three-way Wilcoxon",
        "city": "NYC + TKY",
        "test": "paired Wilcoxon",
        "n_comparisons": 4,
        "within_family_correction": "Holm step-down",
        "alpha": 0.05,
        "headline": "NYC X-SAGE vs B_full: Holm p=7.4e-4; TKY: Holm p=1.1e-5",
        "p_or_ci": "Holm-adjusted p < 0.05 on B_full comparisons; B_blind null",
        "outputs": "outputs/{NYC,TKY}/xsage/round3/A4/step04_threeway/primary.tsv",
    })
    rows.append({
        "family": "A4 paired bootstrap CI",
        "city": "NYC + TKY",
        "test": "percentile bootstrap CI95 (B=10 000)",
        "n_comparisons": 2,
        "within_family_correction": "n/a (interval estimates)",
        "alpha": 0.05,
        "headline": "NYC ΔR@20 B_full−B_blind = +0.0150 CI95 [+0.0039, +0.0259] (excludes 0)",
        "p_or_ci": "CI95",
        "outputs": "outputs/NYC/xsage/round3/A4/bootstrap_ci.json",
    })

    # B7b paired CI on touched
    rows.append({
        "family": "B7b touched-subset accuracy",
        "city": "NYC + TKY",
        "test": "paired bootstrap CI95 (B=10 000)",
        "n_comparisons": 4,
        "within_family_correction": "n/a (interval estimates)",
        "alpha": 0.05,
        "headline": "TKY ΔR@20 touched CI95 [−0.017, −0.008] excludes 0; NYC ΔR@20 [−0.010, +0.000] marginal; both ΔNDCG@10 exclude 0",
        "p_or_ci": "CI95",
        "outputs": "outputs/{NYC,TKY}/<sit_root>/round3/B7b/verdict.json",
    })

    # B9.1 per-target-macro paired tests
    rows.append({
        "family": "B9.1 per-target-macro paired tests",
        "city": "NYC + TKY + TKY_BAL",
        "test": "paired Wilcoxon + percentile bootstrap CI95",
        "n_comparisons": 9,  # 3 cities × (T&T, non-T&T, all)
        "within_family_correction": "Holm step-down (per city, 3 strata)",
        "alpha": 0.05,
        "headline": "C6 promotes BOTH halves of the law to Holm-reject simultaneously on TKY_BAL (T&T p_Holm=2.5e-6, non-T&T p_Holm=8.2e-3); NYC + TKY each have one Holm-rejected stratum + the aggregate",
        "p_or_ci": "Holm-adjusted p; Wald CI from per-stratum reconstructs aggregate bootstrap CI to 5 decimals",
        "outputs": "outputs/round3/B9/{c50_paired_holm.json, c6_pivot_paired.json, summary.md}",
    })

    # B9.2 C5 ablation
    rows.append({
        "family": "B9.2 C5 ablation",
        "city": "TKY",
        "test": "paired Wilcoxon",
        "n_comparisons": 5,
        "within_family_correction": "Holm step-down",
        "alpha": 0.05,
        "headline": "5/5 ablation variants Holm-reject vs B_blind at α=0.05; best (M_minus_geo) p_Holm=8.0e-6",
        "p_or_ci": "Holm-adjusted p < 0.05 on all variants",
        "outputs": "outputs/round3/B9/c5_paired_holm.json",
    })

    # B10.1 permutation test
    if b6_perm:
        for city, blob in b6_perm.items():
            rows.append({
                "family": "B10.1 lens-permutation",
                "city": city,
                "test": f"permutation (preserving |sinks|, n={blob['n_perm']})",
                "n_comparisons": 1,
                "within_family_correction": "n/a (single test per city)",
                "alpha": 0.05,
                "headline": (f"observed mean pairwise Jaccard = "
                              f"{blob['observed_mean_pairwise_jaccard']:.3f}; "
                              f"null mean = {blob['null_mean']:.3f}; "
                              f"p = {blob['p_value_one_sided_greater']:.6f}"),
                "p_or_ci": f"p (one-sided) = {blob['p_value_one_sided_greater']:.6f}",
                "outputs": "outputs/round3/B10/b6_permutation.json",
            })

    # B10.3a B7 ratio CI
    for city, blob in b7_ratio.items():
        rows.append({
            "family": "B10.3a B7 disc/flat ratio",
            "city": city,
            "test": "percentile bootstrap CI95 on per-request ratio (B=10 000)",
            "n_comparisons": 1,
            "within_family_correction": "n/a (interval estimate)",
            "alpha": 0.05,
            "headline": (f"ratio point {blob['ratio_disc_over_flat_point']:.3f} "
                          f"CI95 [{blob['ratio_disc_over_flat_ci95'][0]:.3f}, "
                          f"{blob['ratio_disc_over_flat_ci95'][1]:.3f}]; "
                          f"brackets 1? {blob['ratio_ci_brackets_one']}"),
            "p_or_ci": "CI95",
            "outputs": "outputs/round3/B10/b7_ratio_ci.json",
        })

    # B10.3b B8b Wilson CI
    for city, blob in b8b_wil.items():
        for kind in ("strict", "intent", "time"):
            agg = blob[kind]
            rows.append({
                "family": "B10.3b B8b name-stability Wilson",
                "city": f"{city} [{kind}]",
                "test": "Wilson score CI95 (binomial)",
                "n_comparisons": 1,
                "within_family_correction": "n/a (interval estimate)",
                "alpha": 0.05,
                "headline": (f"{agg['matches']}/{agg['trials']} = "
                              f"{agg['rate']:.3f}  CI95 "
                              f"[{agg['wilson_ci95_lo']:.3f}, "
                              f"{agg['wilson_ci95_hi']:.3f}]"),
                "p_or_ci": "Wilson CI95",
                "outputs": "outputs/round3/B10/b8b_wilson_ci.json",
            })

    # B10.3c Stage C McNemar
    for tag, blob in c2_mcn.items():
        rows.append({
            "family": "B10.3c Stage-C / C2 McNemar",
            "city": tag,
            "test": "paired McNemar (continuity-corrected χ²₁)",
            "n_comparisons": 1,
            "within_family_correction": "n/a (single per config)",
            "alpha": 0.05,
            "headline": (f"ΔF1 = {blob['delta_F1']:+.3f}; "
                          f"T-only correct {blob['n_T_only_correct']}, "
                          f"time-only correct {blob['n_time_only_correct']}; "
                          f"McNemar p = {blob['mcnemar_p_value']:.2e}"),
            "p_or_ci": f"p = {blob['mcnemar_p_value']:.2e}",
            "outputs": "outputs/<city>/<sit_root>/projection/verdict.json",
        })

    df = pd.DataFrame(rows)

    md_lines = [
        "# Statistical-validation summary (X-SAGE round 3)",
        "",
        "> Paper-ready enumeration of every statistical-test family used "
        "in this round, with the within-family correction applied and "
        "the cross-family policy stated explicitly. Backbone of the "
        "paper's \"Statistical validation\" subsection.",
        "",
        "## 1. Test families",
        "",
    ]
    md_lines.append("| family | city / scope | test | n | within-family correction | headline | p / CI |")
    md_lines.append("|---|---|---|---:|---|---|---|")
    for r in rows:
        md_lines.append(
            f"| {r['family']} | {r['city']} | {r['test']} | "
            f"{r['n_comparisons']} | {r['within_family_correction']} | "
            f"{r['headline']} | {r['p_or_ci']} |"
        )
    md_lines += [
        "",
        "## 2. Cross-family correction policy",
        "",
        "**No cross-family multiple-comparison correction is applied.** "
        "Rationale: each family above tests a *distinct hypothesis* "
        "(equivalence of matched-OFF; three-way accuracy; touched-subset "
        "local cost; per-target-macro law; ablation vs blind; lens "
        "permutation; exposure ratio; name stability; projection vs "
        "clock). Within-family Holm step-down controls family-wise "
        "Type-I error at α = 0.05 for each. Pooling these into a single "
        "family would conflate independent questions and inflate the "
        "correction unnecessarily. This is the standard rationale (see "
        "Rubin 2017, *Stat. Sci.*); the trade-off is full transparency "
        "of test inventory, which this table provides.",
        "",
        "## 3. Deterministic verifications (NOT statistical tests)",
        "",
        "These claims are exact identities or pre-registered match-or-"
        "miss outcomes; they correctly carry NO p-value:",
        "",
        "* **Matched-OFF identity** — X-SAGE with κ=0 must equal "
        "B_blind by construction; the A4 TOST is a regression "
        "guarantee, not inference.",
        "* **B5 touched-share identity** — `touched == sink ∩ core` "
        "holds by construction on NYC (PASS exact); the 58 TKY "
        "deviations are sub-threshold core-sinks where the additive "
        "boost is insufficient to displace any popular item — "
        "*property* of the additive rule, not a bug.",
        "* **B8.3 faithfulness** — `score_on − score_blind = κ·G1` is "
        "an algebraic identity. Empirically verified: 0 violations on "
        "76 934 long-tail entries (max abs err = 2.4 × 10⁻⁷ = float32 "
        "ε). This is *verification* of the identity, not inference.",
        "* **A1 / A1bis val=test sink-set match** — NYC mask sinks "
        "{6, 7} are exactly val-flagged and test-flagged; the equality "
        "is set-membership exact, not a statistical claim.",
        "* **C6 R8 pre-registration outcomes** — the predictions in "
        "`PREDICTION_C6.md` were committed BEFORE the measured runs; "
        "the four outcomes (3 MISS + 1 MATCH, with the MATCH on the "
        "structural law P-iv) are pre-specified match/miss results, "
        "not post-hoc tests.",
        "",
        "These should NOT be reported with synthetic p-values: doing "
        "so would be a category error (a deterministic 100 % is *not* "
        "p < 10⁻⁶, it is identity).",
        "",
        "## 4. Tooling",
        "",
        "* `pipeline/step04_statistical_validation/statistical_validation.py` "
        "(phase-2 protocol package) — paired Wilcoxon, bootstrap CI95, "
        "HMP combination, Holm step-down. Re-used unchanged.",
        "* `experiments/round3_a4_statistics.py` (A4 stat suite).",
        "* `experiments/round3_b7b_local_accuracy.py` (paired bootstrap).",
        "* `experiments/round3_b9_stat_hardening.py` (per-macro paired "
        "Wilcoxon + Holm + propagated CI; C5 ablation paired tests).",
        "* `experiments/round3_b10_stat_closure.py` (permutation test, "
        "ratio CI, Wilson CI, McNemar collection, this summary).",
        "",
    ]
    return "\n".join(md_lines), df


# -----------------------------------------------------------------------------

def main() -> int:
    out_dir = REPO_ROOT / "outputs" / "round3" / "B10"
    out_dir.mkdir(parents=True, exist_ok=True)

    # B10.1
    perm = {city: permutation_b6(city) for city in ("NYC", "TKY")}
    (out_dir / "b6_permutation.json").write_text(
        json.dumps(perm, indent=2, default=str))

    # B10.3a
    b7r = {city: b7_ratio_ci(city) for city in ("NYC", "TKY")}
    (out_dir / "b7_ratio_ci.json").write_text(
        json.dumps(b7r, indent=2, default=str))

    # B10.3b
    b8w = b8b_wilson()
    (out_dir / "b8b_wilson_ci.json").write_text(
        json.dumps(b8w, indent=2, default=str))

    # B10.3c
    c2 = c2_mcnemar()
    (out_dir / "c2_mcnemar.json").write_text(
        json.dumps(c2, indent=2, default=str))

    # B10.2 family summary
    md, df = family_summary(perm, b7r, b8w, c2)
    (REPO_ROOT / "STATISTICAL_VALIDATION_SUMMARY.md").write_text(md)
    df.to_csv(REPO_ROOT / "STATISTICAL_VALIDATION_SUMMARY.csv", index=False)
    print(f"\nwrote STATISTICAL_VALIDATION_SUMMARY.{{md,csv}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
