"""Top-level assembly of the situation-aware FM.

Variants (toggles, not separate code paths):
    V0 — context-aware FM, ``use_situation = False``                ⇒ ŷ = ŷ0
    V1 — situation ON, intent ON,    ``use_situation = True, use_intent = True``
    V2 — situation ON, intent OFF,   ``use_situation = True, use_intent = False``

The matched OFF guarantees: when ``use_situation`` is False, the modulation
layer is bypassed completely, so ``ŷ = ŷ0`` exactly (no parameter drift).

Inputs to ``forward_pair`` (BPR training step) are dense feature columns. Inputs
to ``score_full_catalogue`` (ranker) are the same plus a fixed
``item_cat_macro`` lookup.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .perception.context import ContextEncoder
from .perception.intent import IntentEncoder
from .comprehension.situation import SituationGate
from .projection.fm import FMBase
from .projection.modulation import SituationalModulation


@dataclass
class SituationalConfig:
    n_users: int
    n_items: int
    n_macros: int
    n_geo: int

    K: int = 4
    d: int = 32                  # latent dim (FM)
    d_g: int = 8                 # geohash embed dim
    d_e: int = 8                 # intent embed dim
    r: int = 4                   # bilinear rank
    n_time_feats: int = 7        # γ(hour,dow,month) + isweekend

    use_situation: bool = True
    use_intent: bool = True


class SituationalModel(nn.Module):
    """The full L1→L3 stack."""

    def __init__(self, cfg: SituationalConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.context = ContextEncoder(n_geo=cfg.n_geo, d_g=cfg.d_g)
        self.intent = IntentEncoder(n_macros=cfg.n_macros, d_e=cfg.d_e)
        self.gate = SituationGate(d_c=self.context.out_dim, d_e=cfg.d_e,
                                   K=cfg.K, r=cfg.r)
        self.fm = FMBase(n_users=cfg.n_users, n_items=cfg.n_items,
                          n_macros=cfg.n_macros, d=cfg.d,
                          n_time_feats=cfg.n_time_feats)
        self.modulation = SituationalModulation(K=cfg.K, n_macros=cfg.n_macros,
                                                  d=cfg.d)

        # Buffer for catalogue-level item → cat_macro lookup (set externally).
        self.register_buffer("item_cat_macro",
                              torch.zeros(cfg.n_items, dtype=torch.long))

    # ---------------------------------------------------------------
    # Setters used by the ranker
    # ---------------------------------------------------------------
    def set_item_cat_macro(self, mapping: torch.Tensor) -> None:
        assert mapping.dtype == torch.long
        assert mapping.shape == (self.cfg.n_items,)
        self.item_cat_macro = mapping.to(self.item_cat_macro.device)

    # ---------------------------------------------------------------
    # Perception → comprehension (request-level, item-independent)
    # ---------------------------------------------------------------
    def _request_state(self,
                        c_hour: torch.Tensor,
                        c_dow: torch.Tensor,
                        c_month: torch.Tensor,
                        c_isweekend: torch.Tensor,
                        prev_geo_idx: torch.Tensor,
                        intent_feat: torch.Tensor,
                        intent_empty: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (c, pi, z) per request."""
        c = self.context(c_hour, c_dow, c_month, c_isweekend, prev_geo_idx)
        e = self.intent(intent_feat, intent_empty)
        if not self.cfg.use_situation:
            B = c.shape[0]
            K = self.cfg.K
            # When the gate is disabled, π becomes the K=1 indicator. We still
            # carry K dims to keep tensor shapes uniform — modulation is bypassed
            # anyway in this branch.
            pi = torch.zeros(B, K, device=c.device)
            pi[:, 0] = 1.0
            z = torch.zeros(B, dtype=torch.long, device=c.device)
            return c, pi, z
        pi, z, _ = self.gate(c, e, use_intent=self.cfg.use_intent)
        return c, pi, z

    # ---------------------------------------------------------------
    # Training-time : one candidate (positive or negative) per row
    # ---------------------------------------------------------------
    def forward_pair(self,
                      u: torch.Tensor,
                      i: torch.Tensor,
                      m: torch.Tensor,
                      c_t: torch.Tensor,
                      pi: torch.Tensor) -> torch.Tensor:
        y0 = self.fm(u, i, m, c_t)
        if not self.cfg.use_situation:
            return y0
        pu, qi = self.fm.latents(u, i)
        return self.modulation(y0, pi, pu, qi, m)

    # ---------------------------------------------------------------
    # Per-request full catalogue scoring (used by the ranker)
    # ---------------------------------------------------------------
    @torch.no_grad()
    def score_full_catalogue(self,
                              u: torch.Tensor,
                              c_t: torch.Tensor,
                              pi: torch.Tensor) -> torch.Tensor:
        """Score every item for each row in the batch.

        Args:
            u:   (B,) long
            c_t: (B, n_time_feats) float
            pi:  (B, K) float (output of `_request_state`)

        Returns:
            (B, n_items) tensor of scores.
        """
        y0_all = self.fm.score_all_items(u, c_t, self.item_cat_macro)
        if not self.cfg.use_situation:
            return y0_all
        pu = self.fm.P(u)
        return self.modulation.score_all_items(y0_all, pi, pu,
                                                 self.fm.Q.weight,
                                                 self.item_cat_macro)
