"""BPR pairwise trainer for the situation-aware FM.

Iterates over rows of the training split (one row = one positive interaction
plus its request context). Samples one negative item per positive uniformly
from items the user has not seen in the training URM. Optimises:

    L = -log σ(ŷ⁺ - ŷ⁻) + λ_emb · ‖embeddings‖²₂.

Early stopping on val RECALL@20, patience configurable.

Notes:
* the gate ``π`` is computed once per request and reused for both positive and
  negative candidates — matches the eval-time semantics where the gate is item-
  independent.
* negative sampling is rejection-based (uniform draw, reject if the item is in
  the user's history mask). Catalogue is small enough (~3 k) that this is fast.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
import torch
import torch.nn.functional as F

from .data import CityDataset, SplitArrays
from .model import SituationalModel
from .ranker import rank_split


@dataclass
class TrainConfig:
    lr: float = 1e-2
    weight_decay: float = 1e-5
    batch_size: int = 1024
    max_epochs: int = 50
    patience: int = 5
    eval_every: int = 1
    eval_batch_size: int = 512
    seed: int = 42
    verbose: bool = False


def _sample_negatives_for_batch(rng: np.random.Generator,
                                  u_batch: np.ndarray,
                                  mask: sps.csr_matrix,
                                  n_items: int,
                                  max_tries: int = 8) -> np.ndarray:
    """Vectorised rejection sampling: 1 negative per positive."""
    B = u_batch.shape[0]
    neg = rng.integers(0, n_items, size=B)
    for _ in range(max_tries):
        bad = np.zeros(B, dtype=bool)
        for j, ur in enumerate(u_batch):
            row_start = mask.indptr[ur]
            row_end = mask.indptr[ur + 1]
            if neg[j] in mask.indices[row_start:row_end]:
                bad[j] = True
        if not bad.any():
            return neg
        neg[bad] = rng.integers(0, n_items, size=int(bad.sum()))
    return neg


def _build_time_block_train(split: SplitArrays,
                              idx: np.ndarray,
                              device: torch.device) -> torch.Tensor:
    """Same as ranker._build_time_block but on a SplitArrays already in memory."""
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


def train_situational(model: SituationalModel,
                        ds: CityDataset,
                        cfg: TrainConfig,
                        device: torch.device | None = None) -> dict:
    """Train ``model`` on ``ds.train`` using BPR with val early stopping.

    Returns a small training report dict.
    """
    if device is None:
        device = next(model.parameters()).device
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    train_split = ds.train
    val_split = ds.val
    n_items = ds.n_items
    n_train = len(train_split.u)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)

    item_macro_t = torch.from_numpy(ds.item_cat_macro.astype(np.int64)).to(device)
    model.set_item_cat_macro(item_macro_t)

    best_recall = -float("inf")
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0
    history = []

    t_start = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = rng.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            B = len(idx)

            u_np = train_split.u[idx]
            i_pos_np = train_split.i[idx]
            m_pos_np = train_split.m[idx]

            i_neg_np = _sample_negatives_for_batch(rng, u_np, ds.urm_train, n_items)
            m_neg_np = ds.item_cat_macro[i_neg_np].astype(np.int64)

            u = torch.from_numpy(u_np.astype(np.int64)).to(device)
            i_pos = torch.from_numpy(i_pos_np.astype(np.int64)).to(device)
            m_pos = torch.from_numpy(m_pos_np.astype(np.int64)).to(device)
            i_neg = torch.from_numpy(i_neg_np.astype(np.int64)).to(device)
            m_neg = torch.from_numpy(m_neg_np).to(device)

            c_t = _build_time_block_train(train_split, idx, device)
            ce_h = torch.from_numpy(train_split.c_hour[idx].astype(np.int64)).to(device)
            ce_d = torch.from_numpy(train_split.c_dow[idx].astype(np.int64)).to(device)
            ce_m = torch.from_numpy(train_split.c_month[idx].astype(np.int64)).to(device)
            ce_w = torch.from_numpy(train_split.c_isweekend[idx].astype(np.int64)).to(device)
            pg = torch.from_numpy(train_split.prev_geo[idx].astype(np.int64)).to(device)
            feat = torch.from_numpy(train_split.intent_feat[idx]).to(device)
            emp = torch.from_numpy(train_split.intent_empty[idx]).to(device)

            _, pi, _ = model._request_state(ce_h, ce_d, ce_m, ce_w, pg, feat, emp)

            y_pos = model.forward_pair(u, i_pos, m_pos, c_t, pi)
            y_neg = model.forward_pair(u, i_neg, m_neg, c_t, pi)
            loss = -F.logsigmoid(y_pos - y_neg).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)

        # ---- val early stopping -----------------------------------------
        if epoch % cfg.eval_every == 0:
            t0 = time.time()
            val_res = rank_split(
                model, ds, val_split,
                cutoffs=(20,),
                batch_size=cfg.eval_batch_size,
                exclude_mask=ds.urm_train,
                device=device, save_z=False,
            )
            val_rec = float(val_res.metrics[20]["RECALL"].mean())
            eval_s = time.time() - t0
            history.append({"epoch": epoch, "loss": avg_loss,
                             "val_recall20": val_rec, "eval_s": eval_s})
            if cfg.verbose:
                print(f"    e{epoch:02d}  loss={avg_loss:.4f}  "
                      f"val R@20={val_rec:.4f}  ({eval_s:.1f}s)")
            improved = val_rec > best_recall + 1e-6
            if improved:
                best_recall = val_rec
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                best_epoch = epoch
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.patience:
                    if cfg.verbose:
                        print(f"    early stop at epoch {epoch} "
                              f"(best {best_epoch}, R@20={best_recall:.4f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_recall20": best_recall,
        "best_epoch": best_epoch,
        "history": history,
        "wallclock_s": time.time() - t_start,
    }


def refit_train_plus_val(model: SituationalModel,
                           ds: CityDataset,
                           cfg: TrainConfig,
                           n_epochs: int,
                           device: torch.device | None = None) -> dict:
    """Reset the optimiser and train on (train ∪ val) for a fixed n_epochs.

    Used for the final test evaluation, mirroring the floor's refit step.
    """
    if device is None:
        device = next(model.parameters()).device
    torch.manual_seed(cfg.seed + 1)
    rng = np.random.default_rng(cfg.seed + 1)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)

    # Fuse train + val rows
    def stack(a, b):
        return np.concatenate([a, b], axis=0)

    s = ds.train; v = ds.val
    fused_u = stack(s.u, v.u)
    fused_i = stack(s.i, v.i)
    fused_m = stack(s.m, v.m)
    fused_c_hour = stack(s.c_hour, v.c_hour)
    fused_c_dow = stack(s.c_dow, v.c_dow)
    fused_c_month = stack(s.c_month, v.c_month)
    fused_isw = stack(s.c_isweekend, v.c_isweekend)
    fused_prev = stack(s.prev_geo, v.prev_geo)
    fused_feat = stack(s.intent_feat, v.intent_feat)
    fused_empty = stack(s.intent_empty, v.intent_empty)
    n_rows = len(fused_u)

    fused_mask = (ds.urm_train + ds.urm_val).tocsr()
    fused_mask.data[:] = 1.0

    item_macro_t = torch.from_numpy(ds.item_cat_macro.astype(np.int64)).to(device)
    model.set_item_cat_macro(item_macro_t)

    t_start = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        order = rng.permutation(n_rows)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_rows, cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            u_np = fused_u[idx]
            i_pos_np = fused_i[idx]
            m_pos_np = fused_m[idx]

            i_neg_np = _sample_negatives_for_batch(rng, u_np, fused_mask, ds.n_items)
            m_neg_np = ds.item_cat_macro[i_neg_np].astype(np.int64)

            u = torch.from_numpy(u_np.astype(np.int64)).to(device)
            i_pos = torch.from_numpy(i_pos_np.astype(np.int64)).to(device)
            m_pos = torch.from_numpy(m_pos_np.astype(np.int64)).to(device)
            i_neg = torch.from_numpy(i_neg_np.astype(np.int64)).to(device)
            m_neg = torch.from_numpy(m_neg_np).to(device)

            # Time block via NumPy (same routine as eval).
            import math
            c_t_np = np.zeros((len(idx), 7), dtype=np.float32)
            for col, (vals, period) in enumerate([
                (fused_c_hour[idx].astype(np.float32), 24),
                (fused_c_dow[idx].astype(np.float32), 7),
                (fused_c_month[idx].astype(np.float32), 12),
            ]):
                ang = 2.0 * math.pi * vals / period
                c_t_np[:, 2 * col] = np.sin(ang)
                c_t_np[:, 2 * col + 1] = np.cos(ang)
            c_t_np[:, 6] = fused_isw[idx].astype(np.float32)
            c_t = torch.from_numpy(c_t_np).to(device)

            ce_h = torch.from_numpy(fused_c_hour[idx].astype(np.int64)).to(device)
            ce_d = torch.from_numpy(fused_c_dow[idx].astype(np.int64)).to(device)
            ce_m = torch.from_numpy(fused_c_month[idx].astype(np.int64)).to(device)
            ce_w = torch.from_numpy(fused_isw[idx].astype(np.int64)).to(device)
            pg = torch.from_numpy(fused_prev[idx].astype(np.int64)).to(device)
            feat = torch.from_numpy(fused_feat[idx]).to(device)
            emp = torch.from_numpy(fused_empty[idx]).to(device)

            _, pi, _ = model._request_state(ce_h, ce_d, ce_m, ce_w, pg, feat, emp)

            y_pos = model.forward_pair(u, i_pos, m_pos, c_t, pi)
            y_neg = model.forward_pair(u, i_neg, m_neg, c_t, pi)
            loss = -F.logsigmoid(y_pos - y_neg).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        if cfg.verbose:
            print(f"    refit e{epoch:02d}  loss={epoch_loss/max(1,n_batches):.4f}")

    return {"wallclock_s": time.time() - t_start, "n_epochs": n_epochs}
