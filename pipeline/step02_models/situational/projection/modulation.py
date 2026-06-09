"""L3.b — situational modulation.

Given the FM base score ``ŷ0`` (per (request, candidate-item)), the modulation
adds a per-situation category bias and a ``s_k``-modulated user-item term:

    ŷ(u, i) = ŷ0(u, i)  +  Σ_{k=1..K} π_k · ( b^{(k)}_{cat(i)} + <s_k ⊙ p_u, q_i> )

The gate ``π`` is computed once per request and broadcast across candidates.

Parameters:
    B    : per-situation, per-macro category bias  B ∈ R^{K × n_macros}
    S    : per-situation scaling vector            S ∈ R^{K × d}

If we keep ``B`` and ``S`` zero-initialised, the OFF variant (K=1 with these
params untrained or the situation gate disabled) returns ``ŷ0`` exactly. The
matched pair is therefore the SAME module with two flags.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SituationalModulation(nn.Module):
    """Adds Σ π_k (b^(k)_{cat(i)} + <s_k ⊙ p_u, q_i>) to ŷ0."""

    def __init__(self, K: int, n_macros: int, d: int) -> None:
        super().__init__()
        self.K = K
        self.n_macros = n_macros
        self.d = d
        # Zero init → start matched to OFF; gradient descent decides if to move.
        self.B = nn.Parameter(torch.zeros(K, n_macros))
        self.S = nn.Parameter(torch.zeros(K, d))

    # ---------------------------------------------------------------
    # Per-pair (BPR train step)
    # ---------------------------------------------------------------
    def forward(self,
                y0: torch.Tensor,
                pi: torch.Tensor,
                pu: torch.Tensor,
                qi: torch.Tensor,
                cat_idx: torch.Tensor) -> torch.Tensor:
        """Modulate one candidate per row.

        Args:
            y0:      (B,)
            pi:      (B, K)
            pu, qi:  (B, d) user / candidate-item latents
            cat_idx: (B,) long — candidate cat_macro index

        Returns:
            (B,) modulated score.
        """
        # Category bias term per situation, picking the candidate's macro col.
        b_per_k = self.B[:, cat_idx].t()                     # (B, K)
        cat_term = (pi * b_per_k).sum(-1)                    # (B,)

        # <s_k ⊙ p_u, q_i> = Σ_d s_{k,d} p_{u,d} q_{i,d}
        # Per-batch, per-K. (B, K, d) Hadamard then dot.
        s_pu = pu.unsqueeze(1) * self.S.unsqueeze(0)         # (B, K, d)
        ip_term = (s_pu * qi.unsqueeze(1)).sum(-1)           # (B, K)
        ip_term = (pi * ip_term).sum(-1)                     # (B,)

        return y0 + cat_term + ip_term

    # ---------------------------------------------------------------
    # Per-request full ranking (used by the ranker)
    # ---------------------------------------------------------------
    def score_all_items(self,
                        y0_all: torch.Tensor,
                        pi: torch.Tensor,
                        pu: torch.Tensor,
                        Q: torch.Tensor,
                        item_cat_macro: torch.Tensor) -> torch.Tensor:
        """Apply modulation across the whole catalogue.

        Args:
            y0_all:         (B, n_items) FM base scores.
            pi:             (B, K).
            pu:             (B, d) user latents.
            Q:              (n_items, d) item latents (= FMBase.Q.weight).
            item_cat_macro: (n_items,) long mapping item → cat_macro index.

        Returns:
            (B, n_items) modulated scores.
        """
        # Category bias: pick the catalogue's macro per item, then weight by π.
        # B: (K, n_macros) → take per-item column → (K, I)
        b_per_item = self.B[:, item_cat_macro]                # (K, I)
        cat_term = pi @ b_per_item                            # (B, I)

        # Interaction term:
        # <s_k ⊙ p_u, q_i>  =  Σ_d s_{k,d} p_{u,d} q_{i,d}
        # → (B, K, I) but cheaper:
        #    (B, K, d) = pu.unsqueeze(1) * S.unsqueeze(0)
        #    (B, K, I) = einsum that with Q
        s_pu = pu.unsqueeze(1) * self.S.unsqueeze(0)          # (B, K, d)
        ip_term = torch.einsum("bkd,id->bki", s_pu, Q)        # (B, K, I)
        ip_term = (pi.unsqueeze(-1) * ip_term).sum(dim=1)     # (B, I)

        return y0_all + cat_term + ip_term
