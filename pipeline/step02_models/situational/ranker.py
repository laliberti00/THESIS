"""Per-request full-catalogue ranking → per-user ``.npz`` in the standard
schema (so step04 statistical_validation runs unchanged).

For each split row we build the request (timestamp, prev-geo, intent
features), score every item, mask the user's train history (so revisits of
already-seen items aren't ranked), find the rank of the target item, and
compute next-item metrics. Per-user values are the **mean** of the metric
over the user's requests in that split.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sps
import torch

from .data import CityDataset, SplitArrays
from .model import SituationalModel


METRIC_NAMES = ("PRECISION", "RECALL", "MAP", "MRR", "NDCG")


@dataclass
class PerUserResults:
    """Per-user mean metrics + side info."""
    user_ids: np.ndarray                       # (U_eval,) int — IN test set
    metrics: dict[int, dict[str, np.ndarray]]  # cutoff → metric → (U_eval,)
    z_per_request: np.ndarray | None           # (B,) chosen situation per req


def _build_time_block(split: SplitArrays,
                       idx: np.ndarray,
                       device: torch.device) -> torch.Tensor:
    """Reconstruct the dense (B, 7) time block for the FM.

    Matches the layout of ``ContextEncoder.cyclic(...)`` so the FM and the gate
    see the same numbers.
    """
    import math
    B = idx.shape[0]
    c_t = np.zeros((B, 7), dtype=np.float32)
    for col, (vals, period) in enumerate([
        (split.c_hour[idx].astype(np.float32), 24),
        (split.c_dow[idx].astype(np.float32), 7),
        (split.c_month[idx].astype(np.float32), 12),
    ]):
        ang = 2.0 * math.pi * vals / period
        c_t[:, 2 * col] = np.sin(ang)
        c_t[:, 2 * col + 1] = np.cos(ang)
    c_t[:, 6] = split.c_isweekend[idx].astype(np.float32)
    return torch.from_numpy(c_t).to(device)


def _rank_of_target(scores: torch.Tensor,
                     targets: torch.Tensor) -> torch.Tensor:
    """For each row in ``scores`` (B, I), return the 1-indexed rank of
    ``targets`` (B,). NaN / +inf -inf already handled by caller.

    Rank = 1 + number of items with strictly higher score (random tie-break
    implicit — no ties matter in floats with these embeddings, but if they
    did we'd be conservative).
    """
    target_scores = scores.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    higher = (scores > target_scores.unsqueeze(-1)).sum(dim=-1)
    return (higher + 1).long()


def _request_metrics(rank: int, cutoffs: Sequence[int]) -> dict[int, dict[str, float]]:
    """Next-item metrics for a single request given the target's 1-indexed rank.

    Standard relations when there is exactly one relevant item:
        HR@K = Recall@K = 1[rank ≤ K]
        Precision@K     = HR@K / K
        NDCG@K          = 1/log2(rank+1) if rank ≤ K else 0
        MRR@K           = 1/rank        if rank ≤ K else 0
        MAP@K           = 1/rank        if rank ≤ K else 0   (single relevant)
    """
    import math
    out: dict[int, dict[str, float]] = {}
    for K in cutoffs:
        if rank <= K:
            r = 1.0
            p = 1.0 / K
            n = 1.0 / math.log2(rank + 1)
            m = 1.0 / rank
        else:
            r = 0.0; p = 0.0; n = 0.0; m = 0.0
        out[K] = {"RECALL": r, "PRECISION": p, "NDCG": n, "MRR": m, "MAP": m}
    return out


@torch.no_grad()
def rank_split(model: SituationalModel,
                ds: CityDataset,
                split: SplitArrays,
                cutoffs: Sequence[int] = (1, 5, 10, 20, 40, 50, 100),
                batch_size: int = 512,
                exclude_mask: sps.csr_matrix | None = None,
                device: torch.device | None = None,
                save_z: bool = False) -> PerUserResults:
    """Score every request in ``split`` and aggregate per-user.

    Args:
        exclude_mask: items to exclude from ranking *per user* (typically
                      ``ds.urm_train`` for val; ``ds.urm_train + ds.urm_val``
                      for test). The target item is always restored even if
                      it is also in the user's history.
        save_z: if True also save the gate's argmax per request (for the
                situation diagnostics).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    B_total = len(split.u)
    if exclude_mask is None:
        exclude_mask = ds.urm_train

    # Pre-fetch arrays
    u = split.u; i = split.i; m = split.m
    prev_geo = split.prev_geo
    intent_feat = split.intent_feat; intent_empty = split.intent_empty
    c_hour_a = split.c_hour; c_dow_a = split.c_dow
    c_month_a = split.c_month; c_isw_a = split.c_isweekend

    # Per-request metrics buckets
    per_req_metric: dict[int, dict[str, np.ndarray]] = {
        K: {met: np.zeros(B_total, dtype=np.float32) for met in METRIC_NAMES}
        for K in cutoffs
    }
    z_buf = np.zeros(B_total, dtype=np.int32) if save_z else None

    # Catalogue exclusion is per-user; convert exclude_mask to dense once
    # only if it's small enough — we'll just index into it per batch.

    for start in range(0, B_total, batch_size):
        end = min(start + batch_size, B_total)
        idx = np.arange(start, end)

        u_t = torch.from_numpy(u[idx].astype(np.int64)).to(device)
        i_t = torch.from_numpy(i[idx].astype(np.int64)).to(device)
        c_t = _build_time_block(split, idx, device)

        ce_h = torch.from_numpy(c_hour_a[idx].astype(np.int64)).to(device)
        ce_d = torch.from_numpy(c_dow_a[idx].astype(np.int64)).to(device)
        ce_m = torch.from_numpy(c_month_a[idx].astype(np.int64)).to(device)
        ce_w = torch.from_numpy(c_isw_a[idx].astype(np.int64)).to(device)
        pg = torch.from_numpy(prev_geo[idx].astype(np.int64)).to(device)
        feat = torch.from_numpy(intent_feat[idx]).to(device)
        emp = torch.from_numpy(intent_empty[idx]).to(device)

        _, pi, z = model._request_state(ce_h, ce_d, ce_m, ce_w, pg, feat, emp)
        scores = model.score_full_catalogue(u_t, c_t, pi)        # (B, I)

        # Mask per-user excluded items: -inf so they fall to the bottom.
        users_np = u[idx]
        for row, ur in enumerate(users_np):
            cols = exclude_mask.indices[exclude_mask.indptr[ur]:
                                        exclude_mask.indptr[ur + 1]]
            if len(cols):
                scores[row, cols] = float("-inf")

        # If the target is in the excluded mask (rare revisit), don't restore
        # it — per the brief §7, we score venues "not in the user's train
        # history", so a revisit is treated as a miss.

        ranks = _rank_of_target(scores, i_t).cpu().numpy()

        for r_in_batch, rank in enumerate(ranks):
            req_idx = start + r_in_batch
            mvals = _request_metrics(int(rank), cutoffs)
            for K, mm in mvals.items():
                for met, v in mm.items():
                    per_req_metric[K][met][req_idx] = v

        if save_z:
            z_buf[start:end] = z.cpu().numpy().astype(np.int32)

    # Aggregate per user (mean over the user's requests).
    eval_users = np.unique(u)
    eval_users.sort()
    metrics_per_user: dict[int, dict[str, np.ndarray]] = {
        K: {met: np.zeros(len(eval_users), dtype=np.float32) for met in METRIC_NAMES}
        for K in cutoffs
    }
    # Build user -> row indices map once.
    order = np.argsort(u)
    sorted_u = u[order]
    boundaries = np.searchsorted(sorted_u, eval_users)
    boundaries = np.append(boundaries, len(u))
    for ui, ur in enumerate(eval_users):
        s, e = boundaries[ui], boundaries[ui + 1]
        rows = order[s:e]
        for K in cutoffs:
            for met in METRIC_NAMES:
                metrics_per_user[K][met][ui] = per_req_metric[K][met][rows].mean()

    return PerUserResults(
        user_ids=eval_users.astype(np.int64),
        metrics=metrics_per_user,
        z_per_request=z_buf,
    )


def export_npz(per_user: PerUserResults,
                path: Path,
                z_per_request: np.ndarray | None = None) -> None:
    """Write ``.npz`` with the *standard schema* expected by step04.

    Layout:
        user_ids
        RECALL_K, NDCG_K, PRECISION_K, MAP_K, MRR_K  for every K in metrics
        z_per_request  (optional side array)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"user_ids": per_user.user_ids}
    for K, mm in per_user.metrics.items():
        for met in METRIC_NAMES:
            payload[f"{met}_{K}"] = mm[met]
    if z_per_request is not None:
        payload["z_per_request"] = z_per_request
    np.savez_compressed(path, **payload)


def aggregate_for_summary(per_user: PerUserResults,
                            cutoff_key: int = 20) -> dict[str, float]:
    """Quick scalar summary (mean over users) for the requested cutoff."""
    out = {}
    if cutoff_key not in per_user.metrics:
        return out
    for met in METRIC_NAMES:
        out[met] = float(per_user.metrics[cutoff_key][met].mean())
    return out
