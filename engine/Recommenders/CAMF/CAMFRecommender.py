"""Context-Aware Matrix Factorisation — vanilla baseline.

Thesis Phase 2 implementation. PyTorch CPU/MPS, BPR pairwise loss,
no contextual features (yet).

The CAMF family (Baltrunas, Ludwig & Ricci, RecSys 2011) augments a
biased matrix-factorisation model with *context-conditional* bias terms
B_{i,c} (CAMF-CI) or B_{u,c} (CAMF-CU). In its vanilla form — that is,
without any contextual feature available — these context-bias terms
vanish, and the model collapses to

    f(u, i) = b_global + b_u[u] + b_i[i] + <P[u], Q[i]>,

i.e. a Bias-MF / BPR-MF baseline. Numerically this is identical to
`FMRecommender` in the same vanilla setting: both classes share the
`_BPRMFBase` backbone. We keep them as distinct classes because they
will diverge in Phase 3 along different axes (FM: second-order cross
features, CAMF: context-conditional biases).

References:
  Baltrunas, L., Ludwig, B., & Ricci, F. (2011). Matrix factorization
  techniques for context aware recommendation. RecSys 2011.
"""

from engine.Recommenders.FactorizationMachines._BPRMFBase import _BPRMFBase


class CAMFRecommender(_BPRMFBase):
    """Vanilla CAMF (Bias-MF backbone; no context features yet)."""

    RECOMMENDER_NAME = "CAMFRecommender"
