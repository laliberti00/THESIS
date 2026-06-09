"""L1.a — context encoding.

Cyclic encoding of (hour, day-of-week, month) + a `c_isweekend` flag give a
fixed 7-dim time vector ``c_t``. The previous check-in's ``geohash5`` is mapped
through a learnable embedding ``E_g`` of dimension ``d_g`` to give ``c_s``.

The model never reads the *current* row's ``geohash5`` — that is leakage. The
``previous`` location is the strongest legitimate spatial signal at request
time and is allowed.

Output: ``c = [c_t ‖ c_s] ∈ R^{7 + d_g}``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def cyclic(values: torch.Tensor, period: int) -> torch.Tensor:
    """``γ(p,P) = (sin 2πp/P, cos 2πp/P)`` for an integer batch tensor.

    Args:
        values: long tensor of shape (B,) in ``[0, period)``.
        period: positive integer.

    Returns:
        Tensor of shape (B, 2).
    """
    ang = 2.0 * math.pi * values.float() / float(period)
    return torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)


class ContextEncoder(nn.Module):
    """Builds ``c = [c_t ‖ c_s]`` per request.

    Args:
        n_geo: number of distinct ``geohash5`` cells in the city. One extra slot
               is reserved for the ``no-previous-location`` sentinel (index 0).
        d_g:   geohash embedding dimensionality (default 8 per brief).

    Inputs to ``forward``:
        c_hour:       long tensor (B,) in [0, 24)
        c_dow:        long tensor (B,) in [0, 7)
        c_month:      long tensor (B,) in [0, 12)
        c_isweekend:  long tensor (B,) in {0, 1}
        prev_geo_idx: long tensor (B,) in [0, n_geo+1) where 0 = no-prev
    """

    DIM_CT = 2 + 2 + 2 + 1  # γ(hour)+γ(dow)+γ(month)+isweekend

    def __init__(self, n_geo: int, d_g: int = 8) -> None:
        super().__init__()
        if n_geo < 1:
            raise ValueError("n_geo must be ≥ 1")
        # +1 sentinel row at index 0 for the no-previous-location case.
        self.geo_emb = nn.Embedding(num_embeddings=n_geo + 1,
                                    embedding_dim=d_g,
                                    padding_idx=0)
        self.d_g = d_g
        self.n_geo = n_geo
        self.out_dim = self.DIM_CT + d_g

    def forward(self,
                c_hour: torch.Tensor,
                c_dow: torch.Tensor,
                c_month: torch.Tensor,
                c_isweekend: torch.Tensor,
                prev_geo_idx: torch.Tensor) -> torch.Tensor:
        gh = cyclic(c_hour, 24)
        gd = cyclic(c_dow, 7)
        gm = cyclic(c_month, 12)
        we = c_isweekend.float().unsqueeze(-1)
        c_t = torch.cat([gh, gd, gm, we], dim=-1)
        c_s = self.geo_emb(prev_geo_idx)
        return torch.cat([c_t, c_s], dim=-1)
