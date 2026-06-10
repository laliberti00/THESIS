"""Smoke tests for the X-SAGE module — shape / row-sum / matched-OFF invariants.

These do NOT need real data; they work off small synthetic tensors so they run
in well under a second and don't depend on the parquet being present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.xsage.l1_perception import (
    compute_intent, compute_profile, find_attractors,
)
from pipeline.step02_models.xsage.l2_comprehension import (
    adjusted_rand_score, fit_rough_kmeans,
)
from pipeline.step02_models.xsage.l3_projection import (
    estimate_transition, predict_next_situation,
)
from pipeline.step02_models.xsage.metrics import (
    list_displacement, long_tail_groups, long_tail_ratio, topk_from_scores,
)
from pipeline.step02_models.xsage.recommendation import (
    backbone_confidence, fit_situation_biases, harmonic_combine,
    situation_confidence, situational_item_scores,
)


# ---------------------------------------------------------------------------
# L1: row-sums (m a row-prob; e restricted to attractors → sum=1 if any attr)
# ---------------------------------------------------------------------------

def test_profile_row_sums_to_one():
    rng = np.random.default_rng(0)
    n_macros = 9
    B, n = 50, 5
    recent = rng.integers(0, n_macros, size=(B, n)).astype(np.int32)
    # randomly pad
    n_prior = rng.integers(1, n + 1, size=B).astype(np.int32)
    for b in range(B):
        recent[b, n_prior[b]:] = -1
    m = compute_profile(recent, n_prior, n_macros, gamma=0.6)
    assert m.shape == (B, n_macros)
    assert np.allclose(m.sum(axis=1), 1.0, atol=1e-5)
    assert (m >= 0).all()


def test_intent_normalisation_over_attractors():
    K = 9
    # Make a proper stochastic W
    rng = np.random.default_rng(1)
    W = rng.random((K, K)) + 0.1
    W /= W.sum(axis=1, keepdims=True)
    attractors = np.zeros(K, dtype=bool); attractors[[2, 4, 7]] = True
    B = 32
    m = np.full((B, K), 1.0 / K, dtype=np.float32)
    e = compute_intent(m, W, attractors, H=2, beta=0.7)
    assert e.shape == (B, K)
    # Non-attractor coords are exactly zero
    assert np.allclose(e[:, ~attractors], 0.0)
    # Rows that have any attractor mass sum to 1
    s = e.sum(axis=1)
    assert np.allclose(s, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# L2: rough k-means — membership rows sum to 1; ARI of identical labels is 1.
# ---------------------------------------------------------------------------

def test_rough_kmeans_membership_rows_sum_to_one():
    rng = np.random.default_rng(2)
    X = np.concatenate([
        rng.normal(loc=[+2, +2], scale=0.4, size=(60, 2)),
        rng.normal(loc=[-2, +2], scale=0.4, size=(60, 2)),
        rng.normal(loc=[+0, -2], scale=0.4, size=(60, 2)),
    ]).astype(np.float32)
    res = fit_rough_kmeans(X, K=3, eps=0.5, seed=42)
    sums = res.membership.sum(axis=1)
    # On core rows: exactly 1. On boundary rows: 1/|T| × |T| == 1.
    assert np.allclose(sums, 1.0, atol=1e-5)
    assert res.core_label.shape == (180,)


def test_adjusted_rand_identical():
    a = np.array([0, 0, 1, 1, 2, 2, 2])
    assert adjusted_rand_score(a, a) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# L3: transition T row-sums
# ---------------------------------------------------------------------------

def test_transition_rows_sum_to_one():
    rng = np.random.default_rng(3)
    K = 4
    seqs = [rng.integers(0, K, size=rng.integers(2, 10)) for _ in range(20)]
    T, raw = estimate_transition(seqs, K=K)
    assert T.shape == (K, K)
    assert np.allclose(T.sum(axis=1), 1.0, atol=1e-9)
    # predict_next_situation returns the argmax per row
    z = np.array([0, 1, 2, 3])
    pred = predict_next_situation(T, z)
    assert pred.shape == (K,)


# ---------------------------------------------------------------------------
# Metrics: list displacement and long-tail
# ---------------------------------------------------------------------------

def test_list_displacement_bounds():
    L = np.array([1, 2, 3, 4, 5])
    assert list_displacement(L, L, K=5) == pytest.approx(0.0)
    assert list_displacement(L, np.array([6, 7, 8, 9, 10]), K=5) \
           == pytest.approx(1.0)
    # half overlap
    assert list_displacement(np.array([1, 2, 6, 7, 8]),
                              np.array([1, 2, 3, 4, 5]), K=5) \
           == pytest.approx(0.6, abs=1e-6)


def test_long_tail_group_split():
    pop = np.array([100, 90, 80, 50, 20, 10, 5, 2, 1])
    G0, G1 = long_tail_groups(pop, short_head_share=0.2)
    # 20% of 9 items = 2 (ceil)
    assert int(G0.sum()) == 2
    assert int(G1.sum()) == 7
    # LT ratio of a list of all G1 items = 1.0
    assert long_tail_ratio(np.where(G1)[0][:3], G1) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Recommendation: kappa=0 ⇒ ON == OFF exactly
# ---------------------------------------------------------------------------

def test_kappa_zero_recovers_backbone():
    rng = np.random.default_rng(4)
    B, I, K_sit, n_macros = 7, 20, 5, 9
    membership = rng.random((B, K_sit)).astype(np.float32)
    membership /= membership.sum(axis=1, keepdims=True)
    biases = rng.random((K_sit, n_macros)).astype(np.float32)
    item_macro = rng.integers(0, n_macros, size=I).astype(np.int32)
    p_B = rng.random((B, I)).astype(np.float32); p_B /= p_B.sum(axis=1, keepdims=True)
    s_S = situational_item_scores(membership, biases, item_macro)
    # softmax for p_S
    from pipeline.step02_models.xsage.recommendation import softmax_scores
    p_S = softmax_scores(s_S)
    c_B = backbone_confidence(p_B)
    c_S = situation_confidence(0.0, gamma_per_request=np.ones(B))
    p_hat = harmonic_combine(p_B, p_S, c_B, c_S)
    # When c_S = 0 the combiner must collapse to p_B (up to normalisation).
    assert np.allclose(p_hat / p_hat.sum(axis=1, keepdims=True),
                        p_B / p_B.sum(axis=1, keepdims=True), atol=1e-5)


def test_topk_from_scores_argpartition():
    rng = np.random.default_rng(5)
    s = rng.random(50)
    K = 7
    out = topk_from_scores(s, K)
    assert out.shape == (K,)
    # The K returned items are the largest
    top_vals = s[out]
    rest = np.sort(s)[::-1]
    assert np.allclose(top_vals, rest[:K])
