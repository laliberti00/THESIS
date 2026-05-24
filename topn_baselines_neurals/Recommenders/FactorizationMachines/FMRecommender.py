"""Factorisation Machine — vanilla (user+item only) baseline.

Thesis Phase 2 implementation. PyTorch CPU/MPS, BPR pairwise loss,
no contextual features.

When only the two one-hot features (user-id, item-id) are present in a
Rendle 2010 Factorisation Machine, the second-order interaction term
reduces exactly to <P[u], Q[i]>, and the first-order term reduces to
b_u[u] + b_i[i]; together with the constant w_0 the prediction is

    f(u, i) = w_0 + b_u[u] + b_i[i] + <P[u], Q[i]>,

i.e. identical to a Bias-MF / BPR-MF model. We therefore inherit from
the shared `_BPRMFBase` backbone. In Phase 3, this class will be
extended with explicit pairwise cross-feature interactions
<v_f, v_g> · x_f · x_g for the contextual features f, g != user, item.
"""

from topn_baselines_neurals.Recommenders.FactorizationMachines._BPRMFBase import _BPRMFBase


class FMRecommender(_BPRMFBase):
    """Vanilla Factorisation Machine (only user-id and item-id features)."""

    RECOMMENDER_NAME = "FMRecommender"
