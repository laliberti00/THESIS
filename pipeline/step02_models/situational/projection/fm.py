"""L3.a — FM base score ``ŷ0``.

We use Rendle 2010's order-2 Factorisation Machine over four feature groups:

    * user identity              one-hot, n_users embeddings  P  ∈ R^{n_u × d}
    * candidate venue            one-hot, n_items embeddings  Q  ∈ R^{n_i × d}
    * candidate ``cat_macro``    one-hot, n_macros embeddings C  ∈ R^{n_m × d}
    * request cyclic time c_t    7 dense features         W_t ∈ R^{7   × d}

Linear (first-order) terms:
    w0  + b_u[u] + b_i[i] + b_m[cat_macro(i)] + l_tᵀ c_t

Second-order term:
    1/2 · Σ_d  ( (Σ_f v_{f,d} x_f)² − Σ_f v_{f,d}² x_f² ).

For our four blocks this evaluates to dense pairwise interactions

    ŷ0 = w0 + b_u[u] + b_i[i] + b_m[cat(i)] + l_tᵀ c_t
       + <P[u], Q[i]> + <P[u], C[cat(i)]> + <Q[i], C[cat(i)]>
       + <P[u]+Q[i]+C[cat(i)], W_t c_t>
       (i.e. each one-hot embedding interacts with the dense time block too).

We expose ``P`` and ``Q`` so the modulation layer (L3.b) can apply
``s_k ⊙ P[u]`` on top.

The matched OFF variant (situation off OR K=1 with zeroed modulation params)
returns ``ŷ0`` unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FMBase(nn.Module):
    """Rendle-style order-2 FM, four feature blocks."""

    def __init__(self, n_users: int, n_items: int, n_macros: int,
                 d: int, n_time_feats: int = 7) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_macros = n_macros
        self.d = d
        self.n_time_feats = n_time_feats

        # First-order
        self.w0 = nn.Parameter(torch.zeros(1))
        self.b_u = nn.Embedding(n_users, 1)
        self.b_i = nn.Embedding(n_items, 1)
        self.b_m = nn.Embedding(n_macros, 1)
        self.l_t = nn.Parameter(torch.zeros(n_time_feats))
        nn.init.zeros_(self.b_u.weight)
        nn.init.zeros_(self.b_i.weight)
        nn.init.zeros_(self.b_m.weight)

        # Second-order embeddings
        self.P = nn.Embedding(n_users, d)
        self.Q = nn.Embedding(n_items, d)
        self.C = nn.Embedding(n_macros, d)
        self.W_t = nn.Parameter(torch.empty(n_time_feats, d))
        nn.init.normal_(self.P.weight, std=0.05)
        nn.init.normal_(self.Q.weight, std=0.05)
        nn.init.normal_(self.C.weight, std=0.05)
        nn.init.normal_(self.W_t, std=0.05)

    # ---------------------------------------------------------------
    # Per-pair (one item per row)
    # ---------------------------------------------------------------
    def forward(self,
                u: torch.Tensor, i: torch.Tensor, m: torch.Tensor,
                c_t: torch.Tensor) -> torch.Tensor:
        """Score ŷ0 for *one* candidate per row.

        Args:
            u, i, m: long (B,) — user, venue and candidate ``cat_macro`` index.
            c_t:     float (B, n_time_feats).

        Returns:
            (B,) tensor.
        """
        pu = self.P(u)
        qi = self.Q(i)
        cm = self.C(m)
        wt_ct = c_t @ self.W_t                                # (B, d)

        first = (self.w0
                 + self.b_u(u).squeeze(-1)
                 + self.b_i(i).squeeze(-1)
                 + self.b_m(m).squeeze(-1)
                 + c_t @ self.l_t)

        pair = ((pu * qi).sum(-1)
                + (pu * cm).sum(-1)
                + (qi * cm).sum(-1)
                + ((pu + qi + cm) * wt_ct).sum(-1))
        return first + pair

    # ---------------------------------------------------------------
    # Per-request full ranking : score ALL items for one user/time slot.
    # ---------------------------------------------------------------
    def score_all_items(self,
                        u: torch.Tensor,
                        c_t: torch.Tensor,
                        item_cat_macro: torch.Tensor) -> torch.Tensor:
        """For a batch of ``B`` requests, return scores for every item.

        Args:
            u:               (B,) long, the user per request.
            c_t:             (B, n_time_feats) float, time block per request.
            item_cat_macro:  (n_items,) long, mapping item → cat_macro idx.
                             Fixed across requests (catalogue level).

        Returns:
            (B, n_items) tensor of ``ŷ0`` scores.
        """
        B = u.shape[0]
        I = self.n_items

        pu = self.P(u)                                       # (B, d)
        qi = self.Q.weight                                   # (I, d)
        bu = self.b_u(u).squeeze(-1)                         # (B,)
        bi = self.b_i.weight.squeeze(-1)                     # (I,)
        # cat embeddings indexed by *each item*'s category
        cm_per_item = self.C(item_cat_macro)                 # (I, d)
        bm_per_item = self.b_m(item_cat_macro).squeeze(-1)   # (I,)

        # time block per request
        wt_ct = c_t @ self.W_t                               # (B, d)
        l_term = c_t @ self.l_t                              # (B,)

        # First-order: w0 + b_u(u) + b_i(i) + b_m(cat(i)) + l_t·c_t
        first = (self.w0
                 + bu.unsqueeze(-1)                          # (B,1)
                 + bi.unsqueeze(0)                           # (1,I)
                 + bm_per_item.unsqueeze(0)                  # (1,I)
                 + l_term.unsqueeze(-1))                     # (B,1)

        # Pairwise terms
        pu_qi = pu @ qi.t()                                  # (B,I)
        pu_cm = pu @ cm_per_item.t()                         # (B,I) per item
        qi_cm = (qi * cm_per_item).sum(-1)                   # (I,)  (item-level)
        wt_qi = wt_ct @ qi.t()                               # (B,I)
        wt_cm = wt_ct @ cm_per_item.t()                      # (B,I)
        wt_pu = (wt_ct * pu).sum(-1)                         # (B,)

        pair = (pu_qi + pu_cm + qi_cm.unsqueeze(0)
                + wt_pu.unsqueeze(-1) + wt_qi + wt_cm)

        return first + pair

    # ---------------------------------------------------------------
    # Convenience: get the per-user / per-item latent factors used by
    # the modulation layer.
    # ---------------------------------------------------------------
    def latents(self, u: torch.Tensor, i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.P(u), self.Q(i)
