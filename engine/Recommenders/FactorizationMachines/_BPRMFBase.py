"""Shared BPR-MF backbone for the thesis FM and CAMF vanilla recommenders.

Both `FMRecommender` and `CAMFRecommender` reduce, in their *vanilla*
(no contextual features) form, to the same model:

    score(u, i) = b_global + b_u[u] + b_i[i] + <P[u], Q[i]>

trained with the BPR (Rendle et al. 2009) pairwise objective:

    L = - mean_over_triplets log σ( score(u, i_pos) - score(u, i_neg) )
        + α_u ‖P[u]‖² + α_i ‖Q[i]‖²

i.e. plain matrix factorisation with explicit user/item bias. The classes
diverge in Phase 3 when contextual features arrive: FM will add 2-way
cross-feature interactions (Rendle 2010), CAMF will add context-conditional
bias terms (Baltrunas et al. 2011, the CAMF-CI / CAMF-CU variants).

Design notes:
  - PyTorch implementation, CPU-only by default (works on macOS arm64).
    Picks ``mps`` automatically if available, ``cuda`` if present.
  - Negative sampling: uniform over items not in the user's training set.
    No popularity-biased sampling (kept simple, matches LightFM 'bpr' loss).
  - Optimizer: Adam, mini-batches of (u, i_pos, i_neg) triplets sampled
    once per epoch from the URM_train interactions.
  - Compatible with engine.Evaluation.EvaluatorHoldout via
    `_compute_item_score(user_id_array, items_to_compute)`.

This module is *not* exported publicly. Use `FMRecommender` or
`CAMFRecommender` instead.
"""

from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sps

import torch
from torch import nn

from engine.Recommenders.BaseRecommender import BaseRecommender


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _BPRMFModule(nn.Module):
    """The torch model itself: bias terms + low-rank user/item factors."""

    def __init__(self, n_users: int, n_items: int, n_components: int):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_components)
        self.item_factors = nn.Embedding(n_items, n_components)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        # standard small init
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Return (batch,) tensor of scores for the given pairs."""
        pu = self.user_factors(user_idx)
        qi = self.item_factors(item_idx)
        bu = self.user_bias(user_idx).squeeze(-1)
        bi = self.item_bias(item_idx).squeeze(-1)
        dot = (pu * qi).sum(-1)
        return self.global_bias + bu + bi + dot

    def score_user_all_items(self, user_idx: torch.Tensor) -> torch.Tensor:
        """Return (batch, n_items) of scores for given users vs every item."""
        pu = self.user_factors(user_idx)                       # (B, K)
        bu = self.user_bias(user_idx)                          # (B, 1)
        all_q = self.item_factors.weight                       # (I, K)
        all_bi = self.item_bias.weight.squeeze(-1)             # (I,)
        dot = pu @ all_q.t()                                   # (B, I)
        return self.global_bias + bu + all_bi.unsqueeze(0) + dot


class _BPRMFBase(BaseRecommender):
    """Shared training/predict logic. Subclasses set their own
    RECOMMENDER_NAME and may override `_init_extras` / `_extra_loss`."""

    RECOMMENDER_NAME = "_BPRMFBase"

    # Default HP space documented for the future Bayesian search
    # (Shehzad's run_hyperparameter_search infrastructure). These are
    # documented values, not used at runtime unless an HP search is launched.
    HP_SEARCH_SPACE = {
        "n_components":  "Categorical([32, 64, 128, 256])",
        "learning_rate": "Real(1e-4, 1e-1, prior='log-uniform')",
        "user_alpha":    "Real(1e-6, 1e-2, prior='log-uniform')",
        "item_alpha":    "Real(1e-6, 1e-2, prior='log-uniform')",
        "n_epochs":      "Integer(5, 50)",
        "batch_size":    "Categorical([1024, 4096, 16384])",
        "loss":          "Categorical(['bpr'])  # fixed in Phase 2",
    }

    def __init__(self, URM_train, verbose: bool = True):
        super().__init__(URM_train, verbose=verbose)
        self.device = _select_device()
        self._print(f"using device: {self.device.type}")

    # ----- training -----------------------------------------------------

    def fit(self,
            n_components: int = 64,
            learning_rate: float = 0.01,
            user_alpha: float = 1e-5,
            item_alpha: float = 1e-5,
            n_epochs: int = 10,
            batch_size: int = 4096,
            negative_sampling_seed: int = 2026,
            verbose_every: int = 1):
        """Train the model with BPR pairwise loss.

        :param n_components: latent dimension (no_components in LightFM).
        :param learning_rate: Adam learning rate.
        :param user_alpha, item_alpha: L2 regularisation strength on the
            user/item factor matrices (matches LightFM naming).
        :param n_epochs: number of full passes over the interactions.
        :param batch_size: triplets per gradient step.
        :param negative_sampling_seed: RNG seed for negative item draws.
            Set this to 2022, 2023, 42, 0, 1 to produce the 5 seeds the
            statistical-validation script expects for stochastic opponents.
        """
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.user_alpha = user_alpha
        self.item_alpha = item_alpha
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # CSR row pointers, used both for sampling positives and to test
        # whether a candidate negative is actually unseen.
        urm = sps.csr_matrix(self.URM_train)
        positives = urm.nonzero()
        pos_users = positives[0].astype(np.int64)
        pos_items = positives[1].astype(np.int64)
        n_pos = len(pos_users)
        rng = np.random.default_rng(negative_sampling_seed)

        # Build a list-of-sets representation for O(1) "is item in user's
        # training profile" checks during negative rejection sampling.
        user_pos_sets: list[set[int]] = [set() for _ in range(self.n_users)]
        for u, i in zip(pos_users.tolist(), pos_items.tolist()):
            user_pos_sets[u].add(i)

        self._init_extras()  # hook for subclasses

        self.model = _BPRMFModule(self.n_users, self.n_items, n_components).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(n_epochs):
            t0 = time.time()
            # Shuffle the positive triplets each epoch
            shuffle_idx = rng.permutation(n_pos)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_pos, batch_size):
                end = min(start + batch_size, n_pos)
                idx = shuffle_idx[start:end]
                u_np = pos_users[idx]
                ipos_np = pos_items[idx]
                # uniform negative sampling with simple rejection
                ineg_np = rng.integers(0, self.n_items, size=len(idx)).astype(np.int64)
                for k in range(len(idx)):
                    while ineg_np[k] in user_pos_sets[u_np[k]]:
                        ineg_np[k] = int(rng.integers(0, self.n_items))

                u_t = torch.from_numpy(u_np).to(self.device)
                ipos_t = torch.from_numpy(ipos_np).to(self.device)
                ineg_t = torch.from_numpy(ineg_np).to(self.device)

                opt.zero_grad()
                score_pos = self.model.score(u_t, ipos_t)
                score_neg = self.model.score(u_t, ineg_t)
                diff = score_pos - score_neg
                bpr_loss = -torch.nn.functional.logsigmoid(diff).mean()
                # L2 regularisation on the embeddings touched in this batch
                reg = (
                    user_alpha * self.model.user_factors(u_t).pow(2).sum(-1).mean()
                    + item_alpha * self.model.item_factors(ipos_t).pow(2).sum(-1).mean()
                    + item_alpha * self.model.item_factors(ineg_t).pow(2).sum(-1).mean()
                )
                loss = bpr_loss + reg + self._extra_loss(u_t, ipos_t, ineg_t)
                loss.backward()
                opt.step()
                epoch_loss += float(bpr_loss.item())
                n_batches += 1
            dt = time.time() - t0
            if self.verbose and (epoch == 0 or (epoch + 1) % verbose_every == 0
                                 or epoch + 1 == n_epochs):
                self._print(
                    f"epoch {epoch+1}/{n_epochs}  bpr_loss={epoch_loss/n_batches:.5f}  "
                    f"{dt:.1f}s ({n_pos/dt:.0f} triplets/s)"
                )

        # cache once: factors, biases (as numpy arrays for fast _compute_item_score)
        self.model.eval()
        with torch.no_grad():
            self._user_factors_np = self.model.user_factors.weight.detach().cpu().numpy()
            self._item_factors_np = self.model.item_factors.weight.detach().cpu().numpy()
            self._user_bias_np = self.model.user_bias.weight.detach().cpu().numpy().ravel()
            self._item_bias_np = self.model.item_bias.weight.detach().cpu().numpy().ravel()
            self._global_bias_np = float(self.model.global_bias.detach().cpu().numpy())

    # ----- scoring (BaseRecommender / EvaluatorHoldout interface) -------

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """Score a batch of users against all items (numpy back-end for speed)."""
        users = np.asarray(user_id_array, dtype=np.int64)
        P = self._user_factors_np[users]                       # (B, K)
        bu = self._user_bias_np[users].reshape(-1, 1)          # (B, 1)
        Q = self._item_factors_np                              # (I, K)
        bi = self._item_bias_np.reshape(1, -1)                 # (1, I)
        scores = self._global_bias_np + bu + bi + P @ Q.T      # (B, I)
        if items_to_compute is not None:
            mask = np.full(self.n_items, -np.inf, dtype=np.float32)
            mask[np.asarray(items_to_compute, dtype=np.int64)] = 0.0
            scores = scores + mask  # set non-target items to -inf via broadcast
        return scores

    # ----- subclass hooks ----------------------------------------------

    def _init_extras(self):
        """Override in subclass to initialise context-specific parameters."""
        pass

    def _extra_loss(self, u_t, ipos_t, ineg_t):
        """Override in subclass to add context-specific regularisation."""
        return torch.zeros(1, device=self.device)
