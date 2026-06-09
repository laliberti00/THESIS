"""L2 — comprehension.

Bilinear gate over (context ``c``, intent ``e``) → logits over ``K`` latent
situations:

    ℓ_k = β_k + a_kᵀ c + eᵀ M_k c          for k = 1..K,
    π   = softmax(ℓ),   z = argmax_k π_k.

Each ``M_k`` is parameterised by a low-rank factorisation ``M_k = U_k V_kᵀ``
of rank ``r`` so the bilinear cost is ``O(K · r · (d_c + d_e))`` per request
instead of ``O(K · d_c · d_e)``.

The ``use_intent`` toggle drops the bilinear term so the gate becomes purely
context-driven (``ℓ_k = β_k + a_kᵀ c``). Setting ``use_situation = False`` at
the model level bypasses this module entirely.

Inputs:
    c: (B, d_c) context vector.
    e: (B, d_e) intent vector (zero or no-intent for empty windows).
    use_intent: bool — whether to add the bilinear term.

Output:
    pi:    (B, K) softmax probabilities.
    z:     (B,)   argmax indices.
    logits:(B, K) raw logits (useful for entropy regularisation).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SituationGate(nn.Module):
    """Low-rank bilinear gate."""

    def __init__(self, d_c: int, d_e: int, K: int, r: int = 4) -> None:
        super().__init__()
        if K < 1:
            raise ValueError("K must be ≥ 1")
        if r < 1:
            raise ValueError("r must be ≥ 1")
        self.K = K
        self.r = r
        self.d_c = d_c
        self.d_e = d_e

        self.beta = nn.Parameter(torch.zeros(K))
        # K × d_c context-linear weights (a_k).
        self.A = nn.Parameter(torch.empty(K, d_c))
        nn.init.normal_(self.A, std=0.05)

        # K × r × d_e and K × r × d_c factors of M_k = U_k V_kᵀ.
        self.U = nn.Parameter(torch.empty(K, r, d_e))
        self.V = nn.Parameter(torch.empty(K, r, d_c))
        nn.init.normal_(self.U, std=0.05)
        nn.init.normal_(self.V, std=0.05)

    def forward(self, c: torch.Tensor, e: torch.Tensor,
                use_intent: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Context-linear term : (B, K) = (B, d_c) @ (d_c, K)
        lin = c @ self.A.t()
        logits = self.beta.unsqueeze(0) + lin

        if use_intent:
            # eᵀ M_k c = eᵀ (U_kᵀ V_k) c = <U_k e, V_k c>  per k.
            # (B, K, r) = (1, K, r, d_e) @ (B, 1, d_e, 1) → squeeze
            Ue = torch.einsum("krd,bd->bkr", self.U, e)
            Vc = torch.einsum("krd,bd->bkr", self.V, c)
            bilin = (Ue * Vc).sum(dim=-1)
            logits = logits + bilin

        pi = torch.softmax(logits, dim=-1)
        z = pi.argmax(dim=-1)
        return pi, z, logits
