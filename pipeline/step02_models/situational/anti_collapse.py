"""Part B — anti-collapse experiments for the situational gate.

We **do not** modify the existing module (it's loaded into the running TKY
process); instead we provide a thin layer that:

    * re-initialises the gate weights at a smaller scale (B1),
    * applies a softmax temperature (B1),
    * adds an MoE load-balancing loss (B2),
    * trains a warm-up phase with the gate disabled (B3).

The training loop is a fork of ``trainer.train_situational`` with these knobs
plugged in. The matched-pair guarantee is preserved: when all knobs are at
their default the loop is BPR-only and matches the original behaviour.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import CityDataset
from .model import SituationalModel
from .ranker import rank_split
from .trainer import (TrainConfig, _build_time_block_train,
                        _sample_negatives_for_batch)


@dataclass
class AntiCollapseConfig:
    init_scale: float = 5e-2          # gate weight init std (original: 5e-2)
    temperature: float = 1.0           # softmax temperature; π = softmax(ℓ/T)
    balance_weight: float = 0.0        # λ for load-balance loss
    warmup_epochs: int = 0             # epochs with gate forced to uniform
    anneal_temperature: bool = False   # if True, T linearly 2T_start → T_end
    temperature_anneal_to: float = 1.0


# ---------------------------------------------------------------------------
# Re-initialisations
# ---------------------------------------------------------------------------

def apply_balanced_init(model: SituationalModel, scale: float = 1e-3) -> None:
    """Re-init the gate weights at a small scale so initial logits ≈ 0.

    With the original ``5e-2`` init, the gate's ``c`` term alone already
    produces logits of order 0.5 — large enough that one situation can pull
    away over training. Shrinking to ``1e-3`` makes the gate start ≈ uniform
    *and* keeps it harder to break out of uniformity without informative
    signal.
    """
    with torch.no_grad():
        model.gate.beta.zero_()
        nn.init.normal_(model.gate.A, std=scale)
        nn.init.normal_(model.gate.U, std=scale)
        nn.init.normal_(model.gate.V, std=scale)


# ---------------------------------------------------------------------------
# Helpers reused at train + eval time
# ---------------------------------------------------------------------------

def _entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return -(p * torch.clamp(p, eps, 1.0).log()).sum(dim=-1)


def _balance_loss(pi: torch.Tensor) -> torch.Tensor:
    """L = log K − H(p̄); p̄ = batch-mean π. Zero ⇔ p̄ uniform.

    Pushes the *average* usage of situations to uniform; does NOT force each
    request's π to be uniform.
    """
    K = pi.shape[-1]
    p_bar = pi.mean(dim=0)
    H = -(p_bar * torch.clamp(p_bar, 1e-12, 1.0).log()).sum()
    return float(math.log(K)) - H


def _request_state_with_T(model: SituationalModel,
                            ce_h, ce_d, ce_m, ce_w, pg, feat, emp,
                            temperature: float,
                            force_uniform: bool = False
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward perception + gate with a softmax temperature.

    Returns: ``(c, pi, z, logits)``. If ``force_uniform=True`` (warm-up),
    ``pi`` is replaced by ``1/K``.
    """
    c = model.context(ce_h, ce_d, ce_m, ce_w, pg)
    e = model.intent(feat, emp)
    _, _, logits = model.gate(c, e, use_intent=model.cfg.use_intent)
    if force_uniform:
        B = c.shape[0]; K = model.cfg.K
        pi = torch.full((B, K), 1.0 / K, device=c.device)
        z = torch.zeros(B, dtype=torch.long, device=c.device)
        return c, pi, z, logits
    pi = torch.softmax(logits / temperature, dim=-1)
    z = pi.argmax(dim=-1)
    return c, pi, z, logits


# ---------------------------------------------------------------------------
# Main: train with the anti-collapse knobs
# ---------------------------------------------------------------------------

def train_with_anti_collapse(model: SituationalModel,
                                ds: CityDataset,
                                t_cfg: TrainConfig,
                                anti: AntiCollapseConfig,
                                device: torch.device | None = None) -> dict:
    """Fork of ``train_situational`` with: temperature, load-balance, warm-up.

    Returns the same shape of report as the original trainer.
    """
    if device is None:
        device = next(model.parameters()).device
    torch.manual_seed(t_cfg.seed)
    rng = np.random.default_rng(t_cfg.seed)

    train_split = ds.train; val_split = ds.val
    n_items = ds.n_items
    n_train = len(train_split.u)

    optim = torch.optim.Adam(model.parameters(), lr=t_cfg.lr,
                              weight_decay=t_cfg.weight_decay)

    item_macro_t = torch.from_numpy(ds.item_cat_macro.astype(np.int64)).to(device)
    model.set_item_cat_macro(item_macro_t)

    best_recall = -float("inf")
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0
    history = []

    t_start = time.time()
    for epoch in range(1, t_cfg.max_epochs + 1):
        model.train()
        order = rng.permutation(n_train)
        epoch_loss = 0.0
        epoch_bpr = 0.0
        epoch_balance = 0.0
        epoch_pi_entropy = 0.0
        epoch_logit_mag = 0.0
        n_batches = 0

        in_warmup = (epoch <= anti.warmup_epochs)

        # Temperature for this epoch (with optional anneal)
        if anti.anneal_temperature and t_cfg.max_epochs > 1:
            # half-linear: T_start → T_end across first half, then T_end
            half = max(1, t_cfg.max_epochs // 2)
            if epoch <= half:
                frac = (epoch - 1) / max(1, half - 1)
                T_epoch = anti.temperature * (1 - frac) \
                          + anti.temperature_anneal_to * frac
            else:
                T_epoch = anti.temperature_anneal_to
        else:
            T_epoch = anti.temperature

        for start in range(0, n_train, t_cfg.batch_size):
            idx = order[start:start + t_cfg.batch_size]
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

            c, pi, z, logits = _request_state_with_T(
                model, ce_h, ce_d, ce_m, ce_w, pg, feat, emp,
                temperature=T_epoch, force_uniform=in_warmup,
            )

            y_pos = model.forward_pair(u, i_pos, m_pos, c_t, pi)
            y_neg = model.forward_pair(u, i_neg, m_neg, c_t, pi)
            bpr = -F.logsigmoid(y_pos - y_neg).mean()

            if anti.balance_weight > 0 and not in_warmup:
                bal = _balance_loss(pi)
            else:
                bal = torch.zeros((), device=device)

            loss = bpr + anti.balance_weight * bal

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            epoch_loss += float(loss.item())
            epoch_bpr += float(bpr.item())
            epoch_balance += float(bal.item()) if torch.is_tensor(bal) else float(bal)
            epoch_pi_entropy += float(_entropy(pi).mean().item())
            epoch_logit_mag += float(logits.abs().mean().item())
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        avg_pi_ent = epoch_pi_entropy / max(1, n_batches)
        avg_logit = epoch_logit_mag / max(1, n_batches)

        # Val eval — uses the SAME knobs for consistency. The ranker uses the
        # model's own _request_state, so we need to install the temperature
        # there too: monkey-patch via a wrapper at eval time.
        val_rec = _val_recall20_with_T(model, ds, val_split, T_epoch,
                                          device=device, in_warmup=in_warmup,
                                          batch_size=t_cfg.eval_batch_size)
        history.append({
            "epoch": epoch, "loss": avg_loss,
            "bpr": epoch_bpr / max(1, n_batches),
            "balance": epoch_balance / max(1, n_batches),
            "pi_entropy": avg_pi_ent,
            "logit_abs_mean": avg_logit,
            "val_recall20": val_rec,
            "T": T_epoch, "warmup": in_warmup,
        })
        if t_cfg.verbose:
            print(f"    e{epoch:02d}  loss={avg_loss:.4f}  "
                  f"H(π)={avg_pi_ent:.3f}  |ℓ|={avg_logit:.2f}  "
                  f"val R@20={val_rec:.4f}  T={T_epoch:.2f}"
                  f"{' [warmup]' if in_warmup else ''}")
        improved = val_rec > best_recall + 1e-6
        if improved:
            best_recall = val_rec
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            best_epoch = epoch
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= t_cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_val_recall20": best_recall,
        "best_epoch": best_epoch,
        "history": history,
        "wallclock_s": time.time() - t_start,
    }


def _val_recall20_with_T(model, ds, val_split, temperature: float,
                          device, in_warmup: bool, batch_size: int) -> float:
    """Quick val RECALL@20 with the same gate temperature/warm-up active.

    We can't just call the original rank_split because the model uses
    ``_request_state`` internally with no temperature. Cheap workaround:
    score using a small custom loop that mirrors the trainer.
    """
    model.eval()
    K = model.cfg.K
    n_items = ds.n_items
    n_v = len(val_split.u)
    exclude = ds.urm_train.tocsr()
    item_macro_t = torch.from_numpy(ds.item_cat_macro.astype(np.int64)).to(device)
    model.set_item_cat_macro(item_macro_t)

    hits = 0
    with torch.no_grad():
        for s in range(0, n_v, batch_size):
            idx = np.arange(s, min(s + batch_size, n_v))
            u = torch.from_numpy(val_split.u[idx].astype(np.int64)).to(device)
            i_target = torch.from_numpy(val_split.i[idx].astype(np.int64)).to(device)
            c_t = _build_time_block_train(val_split, idx, device)
            ce_h = torch.from_numpy(val_split.c_hour[idx].astype(np.int64)).to(device)
            ce_d = torch.from_numpy(val_split.c_dow[idx].astype(np.int64)).to(device)
            ce_m = torch.from_numpy(val_split.c_month[idx].astype(np.int64)).to(device)
            ce_w = torch.from_numpy(val_split.c_isweekend[idx].astype(np.int64)).to(device)
            pg = torch.from_numpy(val_split.prev_geo[idx].astype(np.int64)).to(device)
            feat = torch.from_numpy(val_split.intent_feat[idx]).to(device)
            emp = torch.from_numpy(val_split.intent_empty[idx]).to(device)
            _, pi, _, _ = _request_state_with_T(
                model, ce_h, ce_d, ce_m, ce_w, pg, feat, emp,
                temperature=temperature, force_uniform=in_warmup,
            )
            scores = model.score_full_catalogue(u, c_t, pi)
            users_np = val_split.u[idx]
            for row, ur in enumerate(users_np):
                cols = exclude.indices[exclude.indptr[ur]:exclude.indptr[ur+1]]
                if len(cols):
                    scores[row, cols] = float("-inf")
            target_scores = scores.gather(1, i_target.unsqueeze(-1)).squeeze(-1)
            ranks = (scores > target_scores.unsqueeze(-1)).sum(-1) + 1
            hits += int((ranks <= 20).sum().item())
    return hits / max(1, n_v)


# ---------------------------------------------------------------------------
# Convenience: eval a trained anti-collapse model on test using same T
# ---------------------------------------------------------------------------

def eval_test_with_T(model: SituationalModel,
                      ds: CityDataset,
                      temperature: float,
                      device,
                      batch_size: int = 512) -> dict:
    """Per-user metrics on test with the gate at the given temperature.

    Returns aggregated metrics + per-request z + π entropy stats.
    """
    test = ds.test
    n = len(test.u)
    exclude = (ds.urm_train + ds.urm_val).tocsr()
    exclude.data[:] = 1.0
    item_macro_t = torch.from_numpy(ds.item_cat_macro.astype(np.int64)).to(device)
    model.set_item_cat_macro(item_macro_t)
    model.eval()

    n_users = ds.n_users
    cutoffs = (1, 5, 10, 20, 40, 50, 100)
    from .ranker import METRIC_NAMES, PerUserResults, _request_metrics

    per_req = {K: {met: np.zeros(n, dtype=np.float32) for met in METRIC_NAMES}
                for K in cutoffs}
    z_buf = np.zeros(n, dtype=np.int32)
    pi_ent_buf = np.zeros(n, dtype=np.float32)

    with torch.no_grad():
        for s in range(0, n, batch_size):
            idx = np.arange(s, min(s + batch_size, n))
            u = torch.from_numpy(test.u[idx].astype(np.int64)).to(device)
            i_target = torch.from_numpy(test.i[idx].astype(np.int64)).to(device)
            c_t = _build_time_block_train(test, idx, device)
            ce_h = torch.from_numpy(test.c_hour[idx].astype(np.int64)).to(device)
            ce_d = torch.from_numpy(test.c_dow[idx].astype(np.int64)).to(device)
            ce_m = torch.from_numpy(test.c_month[idx].astype(np.int64)).to(device)
            ce_w = torch.from_numpy(test.c_isweekend[idx].astype(np.int64)).to(device)
            pg = torch.from_numpy(test.prev_geo[idx].astype(np.int64)).to(device)
            feat = torch.from_numpy(test.intent_feat[idx]).to(device)
            emp = torch.from_numpy(test.intent_empty[idx]).to(device)
            _, pi, z, _ = _request_state_with_T(
                model, ce_h, ce_d, ce_m, ce_w, pg, feat, emp,
                temperature=temperature, force_uniform=False,
            )
            scores = model.score_full_catalogue(u, c_t, pi)
            users_np = test.u[idx]
            for row, ur in enumerate(users_np):
                cols = exclude.indices[exclude.indptr[ur]:exclude.indptr[ur+1]]
                if len(cols):
                    scores[row, cols] = float("-inf")
            target_scores = scores.gather(1, i_target.unsqueeze(-1)).squeeze(-1)
            ranks = (scores > target_scores.unsqueeze(-1)).sum(-1) + 1
            for r_in_batch, rank in enumerate(ranks.cpu().numpy()):
                req_idx = s + r_in_batch
                mvals = _request_metrics(int(rank), cutoffs)
                for K, mm in mvals.items():
                    for met, v in mm.items():
                        per_req[K][met][req_idx] = v
            z_buf[s:s + len(idx)] = z.cpu().numpy().astype(np.int32)
            ent = _entropy(pi).cpu().numpy()
            pi_ent_buf[s:s + len(idx)] = ent.astype(np.float32)

    eval_users = np.unique(test.u); eval_users.sort()
    metrics_per_user = {K: {met: np.zeros(len(eval_users), dtype=np.float32)
                             for met in METRIC_NAMES} for K in cutoffs}
    order = np.argsort(test.u)
    sorted_u = test.u[order]
    bd = np.searchsorted(sorted_u, eval_users)
    bd = np.append(bd, len(test.u))
    for ui, ur in enumerate(eval_users):
        rows = order[bd[ui]:bd[ui + 1]]
        for K in cutoffs:
            for met in METRIC_NAMES:
                metrics_per_user[K][met][ui] = per_req[K][met][rows].mean()

    summary = {
        "RECALL_20": float(metrics_per_user[20]["RECALL"].mean()),
        "NDCG_20": float(metrics_per_user[20]["NDCG"].mean()),
        "MAP_20": float(metrics_per_user[20]["MAP"].mean()),
        "MRR_20": float(metrics_per_user[20]["MRR"].mean()),
    }
    z_dist = np.bincount(z_buf, minlength=model.cfg.K).astype(float)
    z_dist /= max(1, z_dist.sum())
    return {
        "user_ids": eval_users,
        "metrics_per_user": metrics_per_user,
        "summary": summary,
        "z_distribution": z_dist.tolist(),
        "pi_entropy_mean": float(pi_ent_buf.mean()),
        "K": model.cfg.K,
    }
