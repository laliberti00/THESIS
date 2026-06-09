"""Smoke test — does the scaffolded repo run end-to-end on a synthetic URM?

This is the permanent replacement for the old Gowalla-specific equivalence
test. It does NOT need any external dataset: a small synthetic URM is built
in-memory and exercised through the engine + statistical_validation
pipeline. It must pass on any clean checkout.

Run from the repo root:
    source .venv/bin/activate
    python -m tests.test_smoke              # one-shot script
    pytest tests/test_smoke.py -v           # also valid via pytest

Covers:
  1. engine.Evaluation.EvaluatorHoldout returns aggregate metrics on a fitted
     baseline (TopPop and ItemKNN).
  2. save_per_user=True populates per_user_metrics with arrays of the right
     length, AND aggregates remain bit-identical to save_per_user=False.
  3. pipeline.step04_statistical_validation.statistical_validation's
     paired_wilcoxon, harmonic_mean_p, holm_correction, align_two and
     percentile_bootstrap_ci all run on synthetic .npz vectors.
  4. experiments.run_baselines.run_baselines wires engine + evaluator
     correctly when fed a synthetic URM.

If any assertion fails, the repo is structurally broken — fix imports / paths.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

# Make the repo root importable so `engine`, `pipeline`, `experiments` work
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ----------------------------------------------------------------------
# Synthetic URM builder
# ----------------------------------------------------------------------

def make_synthetic_urm(n_users: int = 100, n_items: int = 200,
                      density_train: float = 0.05,
                      density_test: float = 0.01,
                      seed: int = 42):
    """Generate a deterministic train/test URM pair.

    Test cells are a strict subset of (a different mask from) train cells.
    Each user has a non-empty test row by construction (assert later).
    """
    rng = np.random.default_rng(seed)

    # Train: dense Bernoulli mask, kept as int
    train_dense = (rng.random((n_users, n_items)) < density_train).astype(np.float32)
    # Each user must have ≥ 1 train interaction
    for u in range(n_users):
        if train_dense[u].sum() == 0:
            train_dense[u, rng.integers(0, n_items)] = 1.0
    URM_train = sps.csr_matrix(train_dense)

    # Test: independent Bernoulli; ensure ≥1 test item per user, disjoint
    test_dense = (rng.random((n_users, n_items)) < density_test).astype(np.float32)
    # remove overlap with train
    test_dense[train_dense.astype(bool)] = 0
    for u in range(n_users):
        if test_dense[u].sum() == 0:
            # pick the first item not in train
            free = np.where(train_dense[u] == 0)[0]
            if len(free) > 0:
                test_dense[u, free[rng.integers(0, len(free))]] = 1.0
    URM_test = sps.csr_matrix(test_dense)
    return URM_train, URM_test


# ----------------------------------------------------------------------
# Test 1 — engine TopPop + ItemKNN on synthetic URM
# ----------------------------------------------------------------------

def test_engine_baselines_smoke():
    from engine.Evaluation.Evaluator import EvaluatorHoldout
    from engine.Recommenders.NonPersonalizedRecommender import TopPop
    from engine.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

    URM_train, URM_test = make_synthetic_urm()

    # TopPop
    rec = TopPop(URM_train)
    rec.fit()
    scores = rec._compute_item_score(np.arange(5))
    assert scores.shape == (5, URM_train.shape[1])
    assert np.all(np.isfinite(scores))

    # ItemKNN
    rec2 = ItemKNNCFRecommender(URM_train)
    rec2.fit(topK=20, similarity="cosine")
    scores2 = rec2._compute_item_score(np.arange(5))
    assert scores2.shape == (5, URM_train.shape[1])

    # Evaluator aggregates
    ev = EvaluatorHoldout(URM_test, [5, 10, 20], exclude_seen=True, verbose=False)
    df, _ = ev.evaluateRecommender(rec)
    assert "RECALL" in df.columns
    assert "NDCG" in df.columns
    assert 0.0 <= df.loc[20, "RECALL"] <= 1.0
    print(f"  TopPop  R@20={df.loc[20, 'RECALL']:.4f}  N@20={df.loc[20, 'NDCG']:.4f}")


# ----------------------------------------------------------------------
# Test 2 — save_per_user invariants
# ----------------------------------------------------------------------

def test_save_per_user_invariants():
    from engine.Evaluation.Evaluator import EvaluatorHoldout
    from engine.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

    URM_train, URM_test = make_synthetic_urm()
    rec = ItemKNNCFRecommender(URM_train)
    rec.fit(topK=20, similarity="cosine")

    # Path A: save_per_user=False
    ev_a = EvaluatorHoldout(URM_test, [10, 20], exclude_seen=True,
                            verbose=False, save_per_user=False)
    df_a, _ = ev_a.evaluateRecommender(rec)
    assert not hasattr(ev_a, "per_user_metrics")

    # Path B: save_per_user=True
    ev_b = EvaluatorHoldout(URM_test, [10, 20], exclude_seen=True,
                            verbose=False, save_per_user=True)
    df_b, _ = ev_b.evaluateRecommender(rec)
    assert hasattr(ev_b, "per_user_metrics")
    assert hasattr(ev_b, "per_user_user_ids")

    # Invariant 1: aggregate(False) == aggregate(True), bit-identical
    for cutoff in (10, 20):
        for metric in df_a.columns:
            a = float(df_a.loc[cutoff, metric])
            b = float(df_b.loc[cutoff, metric])
            assert abs(a - b) <= 1e-12, \
                f"Aggregate diverges @ cutoff={cutoff}, metric={metric}: {a} vs {b}"

    # Invariant 2: mean(per_user) == aggregate, for PRECISION/RECALL/NDCG/MAP/MRR
    for cutoff in (10, 20):
        for metric in ("PRECISION", "RECALL", "NDCG", "MAP", "MRR"):
            arr = ev_b.per_user_metrics[cutoff][metric]
            assert arr.shape == (len(ev_b.per_user_user_ids),)
            mean = float(arr.mean())
            aggr = float(df_b.loc[cutoff, metric])
            assert abs(mean - aggr) <= 1e-12, \
                f"per_user.mean != aggregate @ cutoff={cutoff}, metric={metric}"
    print(f"  per_user invariants OK on n={len(ev_b.per_user_user_ids)} users")


# ----------------------------------------------------------------------
# Test 3 — statistical_validation primitives
# ----------------------------------------------------------------------

def test_statistical_validation_primitives():
    from pipeline.step04_statistical_validation import statistical_validation as sv

    rng = np.random.default_rng(0)
    n = 500
    a = rng.uniform(0, 1, size=n)
    b = a + rng.normal(0, 0.05, size=n)   # b is a noisy version of a → strong dependence

    # Bootstrap CI
    lo, hi = sv.percentile_bootstrap_ci(a, n_boot=2000)
    assert lo < a.mean() < hi

    # Wilcoxon paired (a vs b) — small effect
    wres = sv.paired_wilcoxon(a, b)
    assert 0.0 <= wres["p_value"] <= 1.0
    assert wres["n_pairs"] == n

    # Harmonic mean p (5 random pvalues)
    pvals = rng.uniform(1e-10, 1.0, size=5).tolist()
    hmp = sv.harmonic_mean_p(pvals)
    assert 0.0 <= hmp["p_combined"] <= 1.0

    # Holm correction on 4 pvalues
    holm_df = sv.holm_correction([1e-10, 1e-5, 0.04, 0.5],
                                  ["c1", "c2", "c3", "c4"])
    assert "p_holm" in holm_df.columns
    assert holm_df["p_holm"].is_monotonic_increasing  # after sort
    assert (holm_df["p_holm"] >= holm_df["p_raw"]).all()

    print(f"  stat-val primitives OK (wilcoxon p={wres['p_value']:.3g}, "
          f"hmp={hmp['p_combined']:.3g})")


# ----------------------------------------------------------------------
# Test 4 — align_two contract on shuffled user_ids
# ----------------------------------------------------------------------

def test_align_two_pairing_contract():
    from pipeline.step04_statistical_validation import statistical_validation as sv

    n = 50
    rng = np.random.default_rng(7)
    user_ids_a = np.arange(n)
    user_ids_b = rng.permutation(n)
    a_vals = rng.uniform(0, 1, n)
    # b_vals are the same per-user value as a, but stored under b's ordering
    b_vals = np.empty(n, dtype=float)
    b_vals[np.argsort(user_ids_b)] = a_vals  # so b[user_ids_b == uid] == a[uid]

    PU = sv.PerUserResult
    A = PU(model="A", dataset="syn", seed=None,
           user_ids=user_ids_a,
           arrays={("RECALL", 20): a_vals})
    B = PU(model="B", dataset="syn", seed=None,
           user_ids=user_ids_b,
           arrays={("RECALL", 20): b_vals})

    xa, yb = sv.align_two(A, B, "RECALL", 20)
    # After alignment, xa[i] and yb[i] should refer to the SAME user, so by
    # construction they should be identical.
    assert np.allclose(xa, yb), \
        "align_two failed to pair by user_id — got different values for the same user"
    print(f"  align_two OK on shuffled user_ids (n={n})")


# ----------------------------------------------------------------------
# Test 5 — experiments.run_baselines.run_baselines on synthetic URM
# ----------------------------------------------------------------------

def test_run_baselines_function_call():
    from experiments.run_baselines import run_baselines

    URM_train, URM_test = make_synthetic_urm()
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = run_baselines(
            URM_train, URM_test,
            models=["TopPop", "ItemKNN"],
            fit_params_by_model={"ItemKNN": {"topK": 20, "similarity": "cosine"}},
            cutoffs=[10, 20],
            save_per_user=True,
            out_dir=Path(tmpdir),
            verbose=False,
        )
        assert len(summary) == 2
        for row in summary:
            assert "recall_at_20" in row
            assert 0.0 <= row["recall_at_20"] <= 1.0
            # tsv was written
            assert (Path(tmpdir) / f"{row['model']}.tsv").exists()
            # per_user .npz was written
            assert (Path(tmpdir) / "per_user" / f"{row['model']}.npz").exists()
    print(f"  run_baselines OK on {len(summary)} models, persisted to tmpdir")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> int:
    print("=== smoke test — synthetic URM ===")
    test_engine_baselines_smoke()
    test_save_per_user_invariants()
    test_statistical_validation_primitives()
    test_align_two_pairing_contract()
    test_run_baselines_function_call()
    print("=== ALL SMOKE TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
