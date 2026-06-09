"""Part A diagnostics for the NYC situational collapse.

Runs on CPU (the TKY run on PID 3742 is hogging MPS — we don't fight it).
Writes ``outputs/NYC/situational/diagnostics_report.json`` and prints a
console summary closed by a decision gate:

    BUG / FEATURE PROBLEM      → fix and re-run before any anti-collapse work.
    GENUINE MoE COLLAPSE       → proceed to Part B.

Sections (mirror the brief):
    A1  intent-window coverage (empty fraction, length stats, e=no-intent share)
    A2  gate-input variance (per-dim std of c and e)
    A3  routing dynamics (per-epoch batch-mean π entropy + gate-logit scale)
    A4  modulation usage (‖b^(k)‖, ‖s_k‖, modulation magnitude vs ŷ0)
    A5  gradient norms reaching gate + modulation parameters
    A6  V1 vs V0 RECALL@20 / NDCG@20 (already known from prior run, restated)

Run:
    .venv/bin/python -m experiments.diagnose_situational_nyc
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.situational.data import (build_city_dataset,
                                                          SplitArrays)
from pipeline.step02_models.situational.model import (SituationalConfig,
                                                          SituationalModel)
from pipeline.step02_models.situational.perception.intent import \
    precompute_intent_features
from pipeline.step02_models.situational.ranker import rank_split
from pipeline.step02_models.situational.trainer import (
    TrainConfig, _build_time_block_train, _sample_negatives_for_batch,
)

import pandas as pd
import scipy.sparse as sps


DEVICE = torch.device("cpu")
NYC = REPO_ROOT / "data" / "processed" / "NYC"
OUT = REPO_ROOT / "outputs" / "NYC" / "situational"
OUT.mkdir(parents=True, exist_ok=True)

# Best HP found by the orchestrator's V0 grid (see RESULTS.md).
HP = dict(d=64, lr=5e-3, weight_decay=1e-5, batch_size=1024,
            max_epochs=8, patience=20, eval_every=1)
K_FOR_V1 = 6              # the sweep winner; the value we're diagnosing
N_DIAG_EPOCHS = 6         # short — we just need the trajectory


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise entropy of a (..., K) probability tensor."""
    return -(p * torch.clamp(p, eps, 1.0).log()).sum(dim=-1)


def shannon_log_K(K: int) -> float:
    return float(np.log(K))


def encode_request_inputs(model: SituationalModel, split: SplitArrays,
                            device: torch.device,
                            batch_size: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    """Run perception over a split → return (c, e) tensors stacked across all rows."""
    B = len(split.u)
    c_chunks, e_chunks = [], []
    with torch.no_grad():
        for s in range(0, B, batch_size):
            idx = np.arange(s, min(s + batch_size, B))
            ce_h = torch.from_numpy(split.c_hour[idx].astype(np.int64)).to(device)
            ce_d = torch.from_numpy(split.c_dow[idx].astype(np.int64)).to(device)
            ce_m = torch.from_numpy(split.c_month[idx].astype(np.int64)).to(device)
            ce_w = torch.from_numpy(split.c_isweekend[idx].astype(np.int64)).to(device)
            pg = torch.from_numpy(split.prev_geo[idx].astype(np.int64)).to(device)
            feat = torch.from_numpy(split.intent_feat[idx]).to(device)
            emp = torch.from_numpy(split.intent_empty[idx]).to(device)
            c = model.context(ce_h, ce_d, ce_m, ce_w, pg)
            e = model.intent(feat, emp)
            c_chunks.append(c); e_chunks.append(e)
    return torch.cat(c_chunks, 0), torch.cat(e_chunks, 0)


def gate_pi_and_logits(model: SituationalModel, c: torch.Tensor, e: torch.Tensor):
    pi, z, logits = model.gate(c, e, use_intent=model.cfg.use_intent)
    return pi, z, logits


# ---------------------------------------------------------------------------
# A1 — intent-window coverage
# ---------------------------------------------------------------------------

def a1_intent_window(ds, df_train, df_val, df_test) -> dict:
    """Recompute the *un-standardised* intent features so we can read raw lengths."""
    raw_train, empty_train = precompute_intent_features(
        df_train, df_train, ds.macro_to_idx)
    raw_test, empty_test = precompute_intent_features(
        df_test, pd.concat([df_train, df_val], ignore_index=True),
        ds.macro_to_idx)

    n_macros = ds.n_macros
    out = {}
    for name, raw, empty in [("train", raw_train, empty_train),
                              ("test", raw_test, empty_test)]:
        lengths = raw[:, n_macros + 1]                    # ℓ column
        non_empty = lengths > 0
        out[name] = {
            "n_rows": int(len(lengths)),
            "fraction_empty_window": float((~non_empty).mean()),
            "window_length": {
                "min": int(lengths[non_empty].min()) if non_empty.any() else 0,
                "median": float(np.median(lengths[non_empty])) if non_empty.any() else 0.0,
                "max": int(lengths[non_empty].max()) if non_empty.any() else 0,
                "fraction_full_10": float((lengths == 10).mean()),
            },
        }
    return out


# ---------------------------------------------------------------------------
# A2 — gate-input variance + sample vectors
# ---------------------------------------------------------------------------

def a2_input_variance(model: SituationalModel, ds, split: SplitArrays,
                       label: str = "test") -> dict:
    """std per dim of c and e across split rows; near-constant detection."""
    c, e = encode_request_inputs(model, split, DEVICE)
    c_std = c.std(dim=0).cpu().numpy()
    e_std = e.std(dim=0).cpu().numpy()

    def summarise(std: np.ndarray, vecs: torch.Tensor, name: str) -> dict:
        near_const = float((std < 1e-4).sum())
        samples = vecs[:5].cpu().tolist()
        return {
            f"{name}_dim": int(std.shape[0]),
            f"{name}_std_min": float(std.min()),
            f"{name}_std_mean": float(std.mean()),
            f"{name}_std_max": float(std.max()),
            f"{name}_n_near_constant": near_const,
            f"{name}_samples_first5": samples,
        }
    out = {"label": label}
    out.update(summarise(c_std, c, "c"))
    out.update(summarise(e_std, e, "e"))
    # Distance of average non-empty e to the learned no_intent vector
    # (proxy for "does intent collapse to the no-intent constant?")
    non_empty = ~torch.from_numpy(split.intent_empty)
    if non_empty.any():
        e_mean_nonempty = e[non_empty].mean(dim=0)
        no_intent = model.intent.no_intent.detach()
        diff = (e_mean_nonempty - no_intent).norm().item()
        ref = max(no_intent.norm().item(), 1e-6)
        out["e_nonempty_minus_no_intent_norm"] = float(diff)
        out["no_intent_vector_norm"] = float(no_intent.norm().item())
        out["relative_collapse_to_no_intent"] = float(diff / ref)
    return out


# ---------------------------------------------------------------------------
# A3 — routing dynamics (per-epoch π entropy + logit scale)
#        also doubles as A5 — gradient norms reach the gate / modulation
# ---------------------------------------------------------------------------

def a3_a5_training_trajectory(model: SituationalModel, ds,
                                cfg: TrainConfig,
                                n_epochs: int) -> dict:
    """Train V1 for `n_epochs` and log per-epoch:
        - batch-mean π entropy (nats)
        - mean abs(logit)
        - grad-norms reaching gate + modulation params
    Returns the trajectory.
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)
    item_macro_t = torch.from_numpy(ds.item_cat_macro.astype(np.int64))
    model.set_item_cat_macro(item_macro_t)

    train = ds.train
    K = model.cfg.K
    n_train = len(train.u)

    # ---- A3 init (epoch 0): π entropy and logit scale BEFORE any training step
    with torch.no_grad():
        c_all, e_all = encode_request_inputs(model, train, DEVICE)
        pi0, z0, logits0 = gate_pi_and_logits(model, c_all, e_all)
        ent0 = entropy(pi0).mean().item()
        logit_scale0 = logits0.abs().mean().item()
        mean_pi0 = pi0.mean(dim=0).cpu().tolist()
    traj = [{
        "epoch": 0, "stage": "init",
        "pi_entropy": ent0,
        "logit_abs_mean": logit_scale0,
        "mean_pi": mean_pi0,
        "log_K": shannon_log_K(K),
    }]

    grad_groups = {
        "gate.beta": [model.gate.beta],
        "gate.A": [model.gate.A],
        "gate.U": [model.gate.U],
        "gate.V": [model.gate.V],
        "modulation.B": [model.modulation.B],
        "modulation.S": [model.modulation.S],
        "context.geo_emb": [model.context.geo_emb.weight],
        "intent.linear": [model.intent.linear.weight, model.intent.linear.bias],
    }

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = rng.permutation(n_train)
        grads_acc = {k: 0.0 for k in grad_groups}
        n_logged = 0
        pi_ent_acc = 0.0
        logit_abs_acc = 0.0
        loss_acc = 0.0
        n_batches = 0

        for start in range(0, n_train, cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            u_np = train.u[idx]
            i_pos_np = train.i[idx]
            m_pos_np = train.m[idx]
            i_neg_np = _sample_negatives_for_batch(rng, u_np, ds.urm_train, ds.n_items)
            m_neg_np = ds.item_cat_macro[i_neg_np].astype(np.int64)

            u = torch.from_numpy(u_np.astype(np.int64))
            i_pos = torch.from_numpy(i_pos_np.astype(np.int64))
            m_pos = torch.from_numpy(m_pos_np.astype(np.int64))
            i_neg = torch.from_numpy(i_neg_np.astype(np.int64))
            m_neg = torch.from_numpy(m_neg_np)
            c_t = _build_time_block_train(train, idx, DEVICE)
            ce_h = torch.from_numpy(train.c_hour[idx].astype(np.int64))
            ce_d = torch.from_numpy(train.c_dow[idx].astype(np.int64))
            ce_m = torch.from_numpy(train.c_month[idx].astype(np.int64))
            ce_w = torch.from_numpy(train.c_isweekend[idx].astype(np.int64))
            pg = torch.from_numpy(train.prev_geo[idx].astype(np.int64))
            feat = torch.from_numpy(train.intent_feat[idx])
            emp = torch.from_numpy(train.intent_empty[idx])

            c = model.context(ce_h, ce_d, ce_m, ce_w, pg)
            e = model.intent(feat, emp)
            pi, z, logits = model.gate(c, e, use_intent=True)

            pi_ent_acc += entropy(pi).mean().item()
            logit_abs_acc += logits.abs().mean().item()

            y_pos = model.forward_pair(u, i_pos, m_pos, c_t, pi)
            y_neg = model.forward_pair(u, i_neg, m_neg, c_t, pi)
            loss = -F.logsigmoid(y_pos - y_neg).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()

            # Capture gradient norms BEFORE optim.step()
            if n_logged < 3:
                for name, params in grad_groups.items():
                    norm = 0.0
                    for p in params:
                        if p.grad is not None:
                            norm += float(p.grad.norm().item()) ** 2
                    grads_acc[name] += math.sqrt(norm)
                n_logged += 1

            optim.step()
            loss_acc += float(loss.item())
            n_batches += 1

        avg_grads = {k: grads_acc[k] / max(1, n_logged) for k in grads_acc}
        traj.append({
            "epoch": epoch,
            "stage": "train",
            "pi_entropy": pi_ent_acc / max(1, n_batches),
            "logit_abs_mean": logit_abs_acc / max(1, n_batches),
            "loss": loss_acc / max(1, n_batches),
            "grad_norms_avg_first3_batches": avg_grads,
        })

    return {
        "K": K,
        "log_K": shannon_log_K(K),
        "trajectory": traj,
    }


# ---------------------------------------------------------------------------
# A4 — modulation usage
# ---------------------------------------------------------------------------

def a4_modulation_usage(model: SituationalModel, ds) -> dict:
    """Norm of b^(k), s_k; magnitude of the modulation term vs ŷ0."""
    B = model.modulation.B.detach()                       # (K, n_macros)
    S = model.modulation.S.detach()                       # (K, d)
    out = {
        "B_norm_per_situation": B.norm(dim=1).cpu().tolist(),
        "S_norm_per_situation": S.norm(dim=1).cpu().tolist(),
        "B_mean_abs": float(B.abs().mean()),
        "S_mean_abs": float(S.abs().mean()),
    }
    # Compute ŷ0 and ŷ on a batch of test requests, all candidate items
    test = ds.test
    n = min(200, len(test.u))
    with torch.no_grad():
        # Slice
        u_t = torch.from_numpy(test.u[:n].astype(np.int64))
        c_t = _build_time_block_train(test, np.arange(n), DEVICE)
        ce_h = torch.from_numpy(test.c_hour[:n].astype(np.int64))
        ce_d = torch.from_numpy(test.c_dow[:n].astype(np.int64))
        ce_m = torch.from_numpy(test.c_month[:n].astype(np.int64))
        ce_w = torch.from_numpy(test.c_isweekend[:n].astype(np.int64))
        pg = torch.from_numpy(test.prev_geo[:n].astype(np.int64))
        feat = torch.from_numpy(test.intent_feat[:n])
        emp = torch.from_numpy(test.intent_empty[:n])
        c, pi, z = model._request_state(ce_h, ce_d, ce_m, ce_w, pg, feat, emp)

        y0 = model.fm.score_all_items(u_t, c_t, model.item_cat_macro)
        y_full = model.score_full_catalogue(u_t, c_t, pi)
        delta = (y_full - y0)
    out["yhat_abs_mean"] = float(y_full.abs().mean())
    out["y0_abs_mean"] = float(y0.abs().mean())
    out["modulation_term_abs_mean"] = float(delta.abs().mean())
    out["modulation_over_y0_ratio"] = float(delta.abs().mean() / max(y0.abs().mean(), 1e-9))
    # Per-situation usage on test
    out["z_distribution_test"] = np.bincount(z.cpu().numpy(),
                                              minlength=model.cfg.K).astype(float)
    out["z_distribution_test"] /= max(1, out["z_distribution_test"].sum())
    out["z_distribution_test"] = out["z_distribution_test"].tolist()
    return out


# ---------------------------------------------------------------------------
# A6 — V1 vs V0 from prior run
# ---------------------------------------------------------------------------

def a6_v1_vs_v0() -> dict:
    """Read the standard .npz files written by the orchestrator (seed=42 only —
    for the diagnostic that's enough; full table is in RESULTS.md)."""
    v0_path = OUT / "V0_seed42.npz"; v1_path = OUT / "V1_seed42.npz"
    if not (v0_path.exists() and v1_path.exists()):
        return {"warning": "previous .npz not found — run the orchestrator first"}
    d0 = np.load(v0_path); d1 = np.load(v1_path)
    return {
        "V0_recall20_mean_seed42": float(d0["RECALL_20"].mean()),
        "V1_recall20_mean_seed42": float(d1["RECALL_20"].mean()),
        "V0_ndcg20_mean_seed42": float(d0["NDCG_20"].mean()),
        "V1_ndcg20_mean_seed42": float(d1["NDCG_20"].mean()),
        "delta_recall20_seed42": float(d1["RECALL_20"].mean() - d0["RECALL_20"].mean()),
        "delta_ndcg20_seed42": float(d1["NDCG_20"].mean() - d0["NDCG_20"].mean()),
    }


# ---------------------------------------------------------------------------
# Decision gate
# ---------------------------------------------------------------------------

def decide(report: dict) -> dict:
    """Classify the failure mode from the metrics."""
    a1 = report["A1"]
    a2_init = report["A2_init"]
    a3 = report["A3_A5"]
    a4 = report["A4"]
    a6 = report["A6"]

    reasons = []

    # Tests
    most_windows_empty = (a1["train"]["fraction_empty_window"] > 0.5
                          or a1["test"]["fraction_empty_window"] > 0.5)
    c_near_const = a2_init["c_n_near_constant"] >= a2_init["c_dim"] - 1
    e_near_const = a2_init["e_n_near_constant"] >= a2_init["e_dim"] - 1
    e_collapsed = a2_init.get("relative_collapse_to_no_intent", 1.0) < 0.1
    init_collapsed = (a3["trajectory"][0]["pi_entropy"]
                      < 0.5 * a3["log_K"])
    final_collapsed = (a3["trajectory"][-1]["pi_entropy"]
                       < 0.5 * a3["log_K"])
    grad_zero_gate = False
    for ep in a3["trajectory"][1:]:
        g = ep["grad_norms_avg_first3_batches"]
        if (g["gate.A"] < 1e-6 and g["gate.U"] < 1e-6 and g["gate.V"] < 1e-6):
            grad_zero_gate = True; break
    modulation_dead = a4["modulation_over_y0_ratio"] < 1e-3
    accuracy_tied = abs(a6.get("delta_recall20_seed42", 0.0)) < 0.01

    if most_windows_empty:
        reasons.append("A1: most windows empty — intent vector is the no-intent constant for most requests")
    if c_near_const:
        reasons.append("A2: context vector c is near-constant — wiring/encoding bug")
    if e_near_const:
        reasons.append("A2: intent vector e is near-constant across non-empty requests — projection bug or feature collapse")
    if grad_zero_gate:
        reasons.append("A5: gradients to gate params are ≈0 — detached graph")
    if init_collapsed:
        reasons.append("A3: π already collapsed at init — bad init scale or wired softmax")

    is_bug = (most_windows_empty or c_near_const or e_near_const
              or grad_zero_gate or init_collapsed)

    if is_bug:
        label = "BUG / FEATURE PROBLEM"
        next_step = "Fix the issues above and rerun V1 NYC before any anti-collapse work."
    elif final_collapsed and not init_collapsed and not grad_zero_gate:
        label = "GENUINE MoE COLLAPSE"
        next_step = ("π starts ~uniform and collapses during training — proceed "
                     "to Part B (balanced init, temperature, load-balance loss, warmup).")
        if modulation_dead:
            reasons.append("A4: modulation magnitude ≈ 0 relative to ŷ0 — gate has no incentive to differentiate")
    else:
        label = "INCONCLUSIVE — REPORT MANUALLY"
        next_step = "Check the JSON; the symptom doesn't fit either canonical pattern."

    return {"label": label, "reasons": reasons, "next_step": next_step}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"=== NYC situational diagnostics (CPU; TKY MPS run untouched) ===")
    t0 = time.time()
    ds = build_city_dataset(NYC, verbose=False)
    df_train = pd.read_parquet(NYC / "df_train.parquet")
    df_val = pd.read_parquet(NYC / "df_val.parquet")
    df_test = pd.read_parquet(NYC / "df_test.parquet")
    print(f"  data: n_users={ds.n_users}  n_items={ds.n_items}  "
          f"n_macros={ds.n_macros}  n_geo={ds.n_geo}")

    # --- A1 ----------------------------------------------------------------
    print("\n[A1] intent-window coverage")
    a1 = a1_intent_window(ds, df_train, df_val, df_test)
    for split, d in a1.items():
        print(f"  {split:5s}: empty_window={d['fraction_empty_window']*100:.2f}%  "
              f"window_len median={d['window_length']['median']:.0f}  "
              f"fraction_at_n10={d['window_length']['fraction_full_10']*100:.1f}%")

    # --- model init ---------------------------------------------------------
    cfg = SituationalConfig(
        n_users=ds.n_users, n_items=ds.n_items,
        n_macros=ds.n_macros, n_geo=ds.n_geo,
        K=K_FOR_V1, d=HP["d"], d_g=8, d_e=8, r=4,
        use_situation=True, use_intent=True,
    )
    torch.manual_seed(42)
    model = SituationalModel(cfg).to(DEVICE)
    item_macro_t = torch.from_numpy(ds.item_cat_macro.astype(np.int64))
    model.set_item_cat_macro(item_macro_t)

    # --- A2 at init ---------------------------------------------------------
    print("\n[A2 init] gate-input variance on test (untrained model)")
    a2_init = a2_input_variance(model, ds, ds.test, "test_init")
    print(f"  c: std min/mean/max = "
          f"{a2_init['c_std_min']:.4f}/{a2_init['c_std_mean']:.4f}/{a2_init['c_std_max']:.4f}  "
          f"near-constant dims = {int(a2_init['c_n_near_constant'])}/{a2_init['c_dim']}")
    print(f"  e: std min/mean/max = "
          f"{a2_init['e_std_min']:.4f}/{a2_init['e_std_mean']:.4f}/{a2_init['e_std_max']:.4f}  "
          f"near-constant dims = {int(a2_init['e_n_near_constant'])}/{a2_init['e_dim']}")
    if "relative_collapse_to_no_intent" in a2_init:
        print(f"  ‖mean(e_nonempty) − no_intent‖ / ‖no_intent‖ = "
              f"{a2_init['relative_collapse_to_no_intent']:.4f}")

    # --- A3 + A5: trajectory ------------------------------------------------
    print(f"\n[A3+A5] training trajectory over {N_DIAG_EPOCHS} epochs (K={cfg.K})")
    t_cfg = TrainConfig(lr=HP["lr"], weight_decay=HP["weight_decay"],
                         batch_size=HP["batch_size"], max_epochs=N_DIAG_EPOCHS,
                         patience=N_DIAG_EPOCHS, eval_every=N_DIAG_EPOCHS,
                         eval_batch_size=512, seed=42, verbose=False)
    a3 = a3_a5_training_trajectory(model, ds, t_cfg, N_DIAG_EPOCHS)
    log_K = a3["log_K"]
    print(f"  log K = {log_K:.4f}")
    for row in a3["trajectory"]:
        if row["stage"] == "init":
            print(f"    init    : π_entropy={row['pi_entropy']:.3f}  "
                  f"|logit|_mean={row['logit_abs_mean']:.3f}  "
                  f"mean π={['{:.2f}'.format(x) for x in row['mean_pi']]}")
        else:
            g = row["grad_norms_avg_first3_batches"]
            print(f"    e{row['epoch']:02d}     : π_entropy={row['pi_entropy']:.3f}  "
                  f"|logit|_mean={row['logit_abs_mean']:.3f}  "
                  f"loss={row['loss']:.4f}  "
                  f"grad(gate.A)={g['gate.A']:.4f}  "
                  f"grad(modulation.B)={g['modulation.B']:.4f}")

    # --- A2 after training --------------------------------------------------
    print("\n[A2 trained] gate-input variance on test (after training)")
    a2_trained = a2_input_variance(model, ds, ds.test, "test_trained")
    print(f"  c: std min/mean/max = "
          f"{a2_trained['c_std_min']:.4f}/{a2_trained['c_std_mean']:.4f}/{a2_trained['c_std_max']:.4f}")
    print(f"  e: std min/mean/max = "
          f"{a2_trained['e_std_min']:.4f}/{a2_trained['e_std_mean']:.4f}/{a2_trained['e_std_max']:.4f}")

    # --- A4: modulation usage ----------------------------------------------
    print(f"\n[A4] modulation usage (after {N_DIAG_EPOCHS} epochs)")
    a4 = a4_modulation_usage(model, ds)
    print(f"  ‖b^(k)‖ per situation: {[f'{x:.3f}' for x in a4['B_norm_per_situation']]}")
    print(f"  ‖s_k‖ per situation : {[f'{x:.3f}' for x in a4['S_norm_per_situation']]}")
    print(f"  |modulation| / |ŷ0| = {a4['modulation_over_y0_ratio']:.4f}")
    print(f"  z-distribution test : {[f'{x:.3f}' for x in a4['z_distribution_test']]}")

    # --- A6: V1 vs V0 from prior run ---------------------------------------
    print(f"\n[A6] V1 vs V0 (prior run, seed=42)")
    a6 = a6_v1_vs_v0()
    if "warning" in a6:
        print(f"  {a6['warning']}")
    else:
        print(f"  V0 R@20={a6['V0_recall20_mean_seed42']:.4f}  "
              f"V1 R@20={a6['V1_recall20_mean_seed42']:.4f}  "
              f"Δ={a6['delta_recall20_seed42']:+.4f}")
        print(f"  V0 N@20={a6['V0_ndcg20_mean_seed42']:.4f}  "
              f"V1 N@20={a6['V1_ndcg20_mean_seed42']:.4f}  "
              f"Δ={a6['delta_ndcg20_seed42']:+.4f}")

    # --- Decision -----------------------------------------------------------
    report = {
        "elapsed_s": time.time() - t0,
        "hp": HP, "K_for_V1": K_FOR_V1, "n_diag_epochs": N_DIAG_EPOCHS,
        "A1": a1, "A2_init": a2_init, "A2_trained": a2_trained,
        "A3_A5": a3, "A4": a4, "A6": a6,
    }
    verdict = decide(report)
    report["verdict"] = verdict

    out_path = OUT / "diagnostics_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str),
                          encoding="utf-8")
    print("\n=== verdict ===")
    print(f"  {verdict['label']}")
    if verdict["reasons"]:
        for r in verdict["reasons"]:
            print(f"   • {r}")
    print(f"  → {verdict['next_step']}")
    print(f"\n(report written to {out_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
