"""Statistical validation script for the thesis (Phase 2).

Compares a "model under test" against a configurable set of baselines using
the per-user metric vectors produced by the THESIS-patched EvaluatorHoldout
(see EVALUATOR_PATCH.md). Implements the protocol described in
STATISTICAL_PROTOCOL.md:

- Descriptive: mean ± std per model on Recall@K, NDCG@K
- Percentile bootstrap 95 % confidence intervals (n_boot = 10000)
- Paired Wilcoxon signed-rank test for every primary comparison
- For stochastic opponents (multiple seeds): per-seed Wilcoxon then Harmonic
  Mean P-value (Wilson 2019, PNAS) for combination (handles dependence)
- Holm step-down correction applied ONLY across the primary family

Designed to run end-to-end on a placeholder configuration (RP3β as pivot,
4 surrogate comparisons) so we can verify mechanics on Gowalla before the
real situation-aware FM model exists in Phase 3.

Usage:
    python statistical_validation.py \\
        --dataset gowalla \\
        --per-user-dir repro_check_results/per_user \\
        --metric RECALL --cutoff 20 \\
        --out-descr results_phase2/descriptive.tsv \\
        --out-tests results_phase2/tests.tsv

References:
    Wilson, D.J. (2019) "The harmonic mean p-value for combining dependent
    tests". PNAS 116(4):1195-1200. https://doi.org/10.1073/pnas.1814092116
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================================
# 1. Data loading
# ============================================================================

@dataclass
class PerUserResult:
    """One model's per-user metric arrays on one dataset, one seed."""
    model: str
    dataset: str
    seed: int | None  # None for deterministic models
    user_ids: np.ndarray
    arrays: dict[tuple[str, int], np.ndarray]  # (metric, cutoff) -> 1-D array

    def vec(self, metric: str, cutoff: int) -> np.ndarray:
        return self.arrays[(metric, cutoff)]


def load_per_user_npz(npz_path: Path, model: str, dataset: str,
                      seed: int | None = None) -> PerUserResult:
    """Load a .npz file produced by repro_check_baseline.py --save-per-user."""
    data = np.load(npz_path)
    user_ids = data["user_ids"]
    arrays = {}
    for key in data.files:
        if key == "user_ids":
            continue
        # Keys are "<METRIC>_<CUTOFF>", e.g. "RECALL_20"
        metric, cutoff_str = key.rsplit("_", 1)
        cutoff = int(cutoff_str)
        arrays[(metric, cutoff)] = data[key]
    return PerUserResult(model=model, dataset=dataset, seed=seed,
                         user_ids=user_ids, arrays=arrays)


def align_two(a: PerUserResult, b: PerUserResult, metric: str, cutoff: int
              ) -> tuple[np.ndarray, np.ndarray]:
    """Return paired arrays restricted to the intersection of user_ids."""
    if np.array_equal(a.user_ids, b.user_ids):
        return a.vec(metric, cutoff), b.vec(metric, cutoff)
    # Otherwise align on intersection
    common, ai, bi = np.intersect1d(a.user_ids, b.user_ids, return_indices=True)
    return a.vec(metric, cutoff)[ai], b.vec(metric, cutoff)[bi]


# ============================================================================
# 2. Descriptive statistics + bootstrap CI
# ============================================================================

def percentile_bootstrap_ci(values: np.ndarray, n_boot: int = 10_000,
                            alpha: float = 0.05, rng_seed: int = 42
                            ) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean.

    Why percentile (not BCa): for n >> 1 and approximately symmetric
    user-metric distributions (Recall, NDCG bounded in [0,1]), percentile
    bootstrap is well-calibrated and trivial to explain.
    """
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    # vectorized: sample n_boot × n indices, mean over axis 1
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return lo, hi


def descriptive_row(model_label: str, vec: np.ndarray, n_boot: int = 10_000
                    ) -> dict:
    lo, hi = percentile_bootstrap_ci(vec, n_boot=n_boot)
    return {
        "model": model_label,
        "n_users": len(vec),
        "mean": float(vec.mean()),
        "std": float(vec.std(ddof=1)),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "pct_zero": float((vec == 0).mean()),
    }


# ============================================================================
# 3. Paired Wilcoxon
# ============================================================================

def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> dict:
    """Wilcoxon signed-rank test of (x − y).

    H_0: median of paired differences is zero.
    Alternative: two-sided.

    zero_method='pratt' is used (instead of the default 'wilcox') because
    in our top-K setting more than half of the users have x = y = 0, and
    dropping ties before ranking ('wilcox') loses statistical power. The
    Pratt method keeps tied differences in the ranking and is the modern
    recommendation.
    """
    diff = x - y
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0, "n_pairs": len(x),
                "n_nonzero_pairs": 0, "mean_diff": 0.0}
    res = stats.wilcoxon(x, y, zero_method="pratt", alternative="two-sided",
                         correction=False, mode="auto")
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "n_pairs": int(len(x)),
        "n_nonzero_pairs": int((diff != 0).sum()),
        "mean_diff": float(diff.mean()),
    }


# ============================================================================
# 4. Harmonic Mean P-value (Wilson 2019 PNAS) for multi-seed combination
# ============================================================================

def harmonic_mean_p(pvalues: Iterable[float],
                    weights: Iterable[float] | None = None) -> dict:
    """Wilson (2019, PNAS) Harmonic Mean P-value combined test.

    Combines L p-values from non-independent tests of the same null. Unlike
    Fisher's combined probability test, HMP is robust to arbitrary
    dependence between the L tests, which is exactly our case: K independent
    seeds of the *same* model evaluated on the *same* test set against the
    *same* opponent — the per-seed Wilcoxon p-values share the user-level
    noise structure.

    Returns the HMP statistic and the asymptotic combined p-value
    (HMP × e × ln(L), clipped to [0,1]). For L ≤ ~20 this asymptotic
    calibration is what Wilson recommends in the supplementary material
    (it is also conservative; a more permissive Bayesian-calibrated value
    exists but is not needed here).
    """
    p = np.asarray(list(pvalues), dtype=float)
    L = len(p)
    if L == 0:
        raise ValueError("HMP requires at least one p-value")
    if L == 1:
        return {"hmp": float(p[0]), "p_combined": float(p[0]), "n": 1}
    if weights is None:
        w = np.full(L, 1.0 / L)
    else:
        w = np.asarray(list(weights), dtype=float)
        w = w / w.sum()
    # avoid div-by-zero on perfect ties
    p_clipped = np.clip(p, 1e-300, 1.0)
    hmp = 1.0 / float(np.sum(w / p_clipped))
    p_combined = float(min(1.0, hmp * np.e * np.log(L)))
    return {"hmp": float(hmp), "p_combined": p_combined, "n": int(L)}


# ============================================================================
# 5. Holm step-down correction (family-wise)
# ============================================================================

def holm_correction(pvalues: list[float], labels: list[str], alpha: float = 0.05
                    ) -> pd.DataFrame:
    """Holm step-down procedure on a *family* of K p-values.

    Returns a DataFrame with raw and adjusted p-values, and a "reject"
    column at α. Adjusted p-values are non-decreasing along the original
    sorted order, computed as
        p_adj_(i) = max_{j<=i} ((K - j + 1) * p_(j))
    then capped at 1.
    """
    K = len(pvalues)
    order = np.argsort(pvalues)
    p_sorted = np.array(pvalues, dtype=float)[order]
    adj_sorted = np.zeros(K)
    running_max = 0.0
    for j in range(K):
        candidate = (K - j) * p_sorted[j]
        running_max = max(running_max, candidate)
        adj_sorted[j] = min(1.0, running_max)
    # restore original order
    adj = np.empty(K)
    adj[order] = adj_sorted
    return pd.DataFrame({
        "comparison": labels,
        "p_raw": pvalues,
        "p_holm": adj.tolist(),
        "reject_at_0.05": (adj <= alpha).tolist(),
    })


# ============================================================================
# 6. Placeholder configuration: 4 primary comparisons, RP3β-pivot
# ============================================================================

@dataclass
class Comparison:
    """One primary comparison.

    ``model_under_test_label``: e.g. "RP3β" (will be "FM-situ" in Phase 3).
    ``opponent_label``: e.g. "ItemKNN" (will be "FM-flat", "DCCF", "BIGCF").
    ``model_npz``: path to per-user .npz of the model under test.
    ``opponent_npzs``: list of one or more .npz; one for deterministic,
    several for stochastic (multi-seed) opponents.
    """
    name: str
    model_under_test_label: str
    opponent_label: str
    model_npz: Path
    opponent_npzs: list[Path]
    pseudo_seed_noise: float = 0.0  # if >0 we synthesize K seeds via Gaussian
                                    # noise stub on a single deterministic .npz
    n_pseudo_seeds: int = 1


def build_placeholder_comparisons(per_user_dir: Path, dataset: str
                                  ) -> list[Comparison]:
    """The 4 placeholder comparisons that mimic the Phase 3 primaries.

    Real Phase 3 layout            →  Placeholder Phase 2 surrogate
    ------------------------------------------------------------
    FM-situ vs FM-flat             →  RP3β vs ItemKNN  (deterministic)
    FM-situ vs best non-neural     →  RP3β vs P3α      (deterministic)
    FM-situ vs DCCF (5 seeds)      →  RP3β vs UserKNN  +  Gaussian-noise stub
    FM-situ vs BIGCF (5 seeds)     →  RP3β vs TopPop   +  Gaussian-noise stub
    """
    def p(name: str) -> Path:
        return per_user_dir / f"{dataset}_{name}.npz"

    rp3b = p("RP3beta")

    return [
        Comparison(
            name="primary_1__vs_FM_flat_surrogate",
            model_under_test_label="RP3β (pivot)",
            opponent_label="ItemKNN (placeholder for FM-flat)",
            model_npz=rp3b, opponent_npzs=[p("ItemKNN")],
        ),
        Comparison(
            name="primary_2__vs_best_baseline_surrogate",
            model_under_test_label="RP3β (pivot)",
            opponent_label="P3α (placeholder for best non-neural)",
            model_npz=rp3b, opponent_npzs=[p("P3alpha")],
        ),
        Comparison(
            name="primary_3__vs_DCCF_surrogate",
            model_under_test_label="RP3β (pivot)",
            opponent_label="UserKNN ×5 (placeholder for DCCF, Gaussian-noise stub)",
            model_npz=rp3b, opponent_npzs=[p("UserKNN")],
            pseudo_seed_noise=0.01, n_pseudo_seeds=5,
        ),
        Comparison(
            name="primary_4__vs_BIGCF_surrogate",
            model_under_test_label="RP3β (pivot)",
            opponent_label="TopPop ×5 (placeholder for BIGCF, Gaussian-noise stub)",
            model_npz=rp3b, opponent_npzs=[p("TopPop")],
            pseudo_seed_noise=0.005, n_pseudo_seeds=5,
        ),
    ]


def _maybe_pseudo_seeds(opponent: PerUserResult, metric: str, cutoff: int,
                        noise_sd: float, n_seeds: int, rng_seed: int = 0
                        ) -> list[np.ndarray]:
    """Materialize K stochastic versions of a deterministic per-user vector
    by adding i.i.d. Gaussian noise. Used only for the *placeholder*
    Phase-2 surrogate of stochastic opponents (DCCF, BIGCF). In Phase 3
    these will be replaced with actual per-user vectors from the K real
    seeds of the stochastic model.
    """
    base = opponent.vec(metric, cutoff)
    rng = np.random.default_rng(rng_seed)
    out = []
    for k in range(n_seeds):
        perturbed = base + rng.normal(0.0, noise_sd, size=base.shape)
        # clip back to [0, 1] since Recall/NDCG are bounded
        perturbed = np.clip(perturbed, 0.0, 1.0)
        out.append(perturbed)
    return out


# ============================================================================
# 7. Driver
# ============================================================================

def run(args):
    per_user_dir = Path(args.per_user_dir)
    out_descr = Path(args.out_descr)
    out_tests = Path(args.out_tests)
    out_descr.parent.mkdir(parents=True, exist_ok=True)
    out_tests.parent.mkdir(parents=True, exist_ok=True)

    metric = args.metric.upper()
    cutoff = int(args.cutoff)

    print(f"=== Statistical validation — {args.dataset} / {metric}@{cutoff} ===")

    comps = build_placeholder_comparisons(per_user_dir, args.dataset)

    # -----------------------------------------------------------------------
    # Descriptive table
    # -----------------------------------------------------------------------
    descr_rows: list[dict] = []
    seen_models: set[str] = set()
    # First, the "model under test" — same across all 4 placeholder comparisons
    pivot = load_per_user_npz(comps[0].model_npz,
                              model=comps[0].model_under_test_label,
                              dataset=args.dataset)
    descr_rows.append(descriptive_row(pivot.model, pivot.vec(metric, cutoff)))
    seen_models.add(pivot.model)

    # Then each opponent
    for comp in comps:
        opp = load_per_user_npz(comp.opponent_npzs[0],
                                model=comp.opponent_label,
                                dataset=args.dataset)
        if comp.opponent_label in seen_models:
            continue
        if comp.pseudo_seed_noise > 0:
            seed_vecs = _maybe_pseudo_seeds(opp, metric, cutoff,
                                            comp.pseudo_seed_noise,
                                            comp.n_pseudo_seeds)
            # report the mean-across-seeds vector as the "representative" for
            # the descriptive table (just for human reading; the actual test
            # uses per-seed vectors, see below)
            mean_vec = np.mean(seed_vecs, axis=0)
            row = descriptive_row(comp.opponent_label, mean_vec)
            row["model"] = comp.opponent_label
            row["mean_across_seeds"] = True
            descr_rows.append(row)
        else:
            descr_rows.append(descriptive_row(comp.opponent_label,
                                              opp.vec(metric, cutoff)))
        seen_models.add(comp.opponent_label)

    descr_df = pd.DataFrame(descr_rows)
    descr_df.to_csv(out_descr, sep="\t", index=False, float_format="%.6f")
    print(f"[descr] wrote {out_descr}")
    print(descr_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # Primary tests
    # -----------------------------------------------------------------------
    raw_p_values: list[float] = []
    labels: list[str] = []
    per_comp_details: list[dict] = []

    for comp in comps:
        opp = load_per_user_npz(comp.opponent_npzs[0],
                                model=comp.opponent_label,
                                dataset=args.dataset)
        x, y = align_two(pivot, opp, metric, cutoff)

        if comp.pseudo_seed_noise > 0:
            # multi-seed branch: per-seed Wilcoxon, then HMP
            seed_vecs = _maybe_pseudo_seeds(opp, metric, cutoff,
                                            comp.pseudo_seed_noise,
                                            comp.n_pseudo_seeds)
            per_seed_p = []
            for k, vec_k in enumerate(seed_vecs):
                wres = paired_wilcoxon(x, vec_k)
                per_seed_p.append(wres["p_value"])
            hmp = harmonic_mean_p(per_seed_p)
            primary_p = hmp["p_combined"]
            details = {
                "comparison": comp.name,
                "mode": "multi-seed-HMP",
                "n_seeds": len(per_seed_p),
                "per_seed_p_values": per_seed_p,
                "hmp_statistic": hmp["hmp"],
                "hmp_combined_p": hmp["p_combined"],
                "primary_p": primary_p,
            }
        else:
            wres = paired_wilcoxon(x, y)
            primary_p = wres["p_value"]
            details = {
                "comparison": comp.name,
                "mode": "single-Wilcoxon",
                "wilcoxon_statistic": wres["statistic"],
                "n_pairs": wres["n_pairs"],
                "n_nonzero_pairs": wres["n_nonzero_pairs"],
                "mean_diff": wres["mean_diff"],
                "primary_p": primary_p,
            }
        raw_p_values.append(primary_p)
        labels.append(comp.name)
        per_comp_details.append(details)

    # Holm correction across the 4 primary p-values
    holm_df = holm_correction(raw_p_values, labels, alpha=0.05)

    # Attach details
    for i, comp in enumerate(comps):
        holm_df.loc[i, "model_under_test"] = comp.model_under_test_label
        holm_df.loc[i, "opponent"] = comp.opponent_label
        holm_df.loc[i, "details_json"] = json.dumps(per_comp_details[i],
                                                    default=float)

    holm_df = holm_df[["comparison", "model_under_test", "opponent",
                       "p_raw", "p_holm", "reject_at_0.05", "details_json"]]
    holm_df.to_csv(out_tests, sep="\t", index=False, float_format="%.6g")
    print(f"\n[tests] wrote {out_tests}")
    print(holm_df.drop(columns=["details_json"]).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="gowalla",
                        choices=["gowalla", "amazonBook", "tmall"])
    parser.add_argument("--per-user-dir", default="repro_check_results/per_user",
                        help="Directory containing <dataset>_<model>.npz files")
    parser.add_argument("--metric", default="RECALL",
                        choices=["RECALL", "NDCG", "PRECISION", "MAP", "MRR"])
    parser.add_argument("--cutoff", default=20, type=int)
    parser.add_argument("--out-descr", default="results_phase2/descriptive.tsv")
    parser.add_argument("--out-tests", default="results_phase2/tests.tsv")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
