"""Light HP sweep for B_full (the comparison context-aware FM).

The original Stage-D B_full was trained at d=64, lr=5e-3, L2=1e-5, fixed
10 epochs — same architecture as the floor's FM-vanilla but **untuned**,
whereas B_blind enjoyed the floor's full Bayesian search. The comparison
is unfair, especially on TKY where B_full landed below B_blind.

This module:
    1. Trains a copy of ``ContextAwareFM`` for each ``(d, L2)`` cell on
       ``TRAIN`` only, evaluates val R@20 per epoch, early-stops with
       patience.
    2. Picks the (d, L2, n_epochs) with the best val R@20.
    3. Refits the winner on ``TRAIN ∪ VAL`` for the same n_epochs.
    4. Dumps the final test score matrix to
       ``outputs/<city>/xsage/backbone/Bfull.scores.npy`` (overwriting the
       untuned cache) and an audit trail to
       ``outputs/<city>/xsage/backbone/Bfull.tuning_log.csv``.

We deliberately keep the grid small (3 × 3 = 9 cells × patience-bounded
epochs) so this completes in a few minutes per city.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch

from .backbone_full import (ContextAwareFM, FeatureSpec, _build_request_features,
                              _catalogue_indices, score_all_per_request,
                              train_b_full)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _val_recall20(scores: np.ndarray,
                    target_items: np.ndarray,
                    exclude_mask: sps.csr_matrix,
                    users: np.ndarray,
                    K: int = 20) -> float:
    """Per-request mean R@K (= hit rate) on val. Cheap enough at val size."""
    hits = 0
    n = scores.shape[0]
    for b in range(n):
        u = int(users[b])
        cols = exclude_mask.indices[exclude_mask.indptr[u]:exclude_mask.indptr[u + 1]]
        s = scores[b].copy()
        if len(cols):
            s[cols] = -np.inf
        target = int(target_items[b])
        ts = s[target]
        rank = int((s > ts).sum()) + 1
        if rank <= K:
            hits += 1
    return hits / max(1, n)


def tune_and_refit(city: str,
                     dataset_loader,
                     d_grid=(32, 64, 128),
                     l2_grid=(1e-5, 1e-4, 1e-3),
                     max_epochs: int = 20,
                     patience: int = 3,
                     batch_size: int = 4096,
                     lr: float = 5e-3,
                     seed: int = 42,
                     verbose: bool = False) -> dict:
    """Sweep (d, L2), pick winner on val R@20, refit on train+val, dump scores.

    ``dataset_loader`` is a ``city -> dict`` callable returning the orchestrator's
    standard bundle (``df_train``, ``df_val``, ``df_test``, ``urm_*``,
    ``macro_to_idx``, ``n_users``, ``n_items``, ...).
    """
    out_dir = REPO_ROOT / "outputs" / city / "xsage" / "backbone"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = dataset_loader(city)
    df_train = ds["df_train"]; df_val = ds["df_val"]; df_test = ds["df_test"]
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # Build the multi-hot vocabularies (same as Stage D)
    fine_vals = sorted(set(df_train["cat_fine"]).union(df_val["cat_fine"])
                        .union(df_test["cat_fine"]))
    fine_to_idx = {v: i for i, v in enumerate(fine_vals)}
    prev_vals = sorted(set(df_train["prev_geohash5"]).union(df_val["prev_geohash5"])
                        .union(df_test["prev_geohash5"]))
    prev_clean = [v for v in prev_vals if v != "__NONE__"]
    geo_to_idx = {"__NONE__": 0,
                    **{v: i + 1 for i, v in enumerate(prev_clean)}}
    n_geo = len(prev_clean)
    spec = FeatureSpec(n_users=ds["n_users"], n_items=ds["n_items"],
                        n_macros=ds["n_macros"], n_fine=len(fine_to_idx),
                        n_geo=n_geo, n_intent_last=ds["n_macros"])

    feats_train = _build_request_features(df_train, spec, ds["macro_to_idx"],
                                              fine_to_idx, geo_to_idx)
    feats_val = _build_request_features(df_val, spec, ds["macro_to_idx"],
                                            fine_to_idx, geo_to_idx)
    feats_test = _build_request_features(df_test, spec, ds["macro_to_idx"],
                                              fine_to_idx, geo_to_idx)
    macro_per_item, fine_per_item = _catalogue_indices(df_all, spec,
                                                          ds["macro_to_idx"],
                                                          fine_to_idx)

    mask_train = ds["urm_train"].tocsr().copy(); mask_train.data[:] = 1.0
    # Eval val with exclude_seen=URM_train (consistent with the floor)
    val_users = feats_val["u_idx"]; val_targets = feats_val["i_idx"]

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"  device = {device}, training on TRAIN, ES on val R@20 "
          f"(max_epochs={max_epochs}, patience={patience})")

    tuning_log: list[dict] = []
    best_cell = None
    for d in d_grid:
        for l2 in l2_grid:
            torch.manual_seed(seed)
            model = ContextAwareFM(spec, d=d).to(device)
            history: list[dict] = []
            best_val = -1.0; best_epoch = 0; bad = 0
            t0 = time.time()
            # Manual training loop with per-epoch val eval
            optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
            rng = np.random.default_rng(seed)
            n_train = len(feats_train["u_idx"])
            for epoch in range(1, max_epochs + 1):
                model.train()
                order = rng.permutation(n_train)
                losses = []
                for s in range(0, n_train, batch_size):
                    idx = order[s:s + batch_size]
                    u_np = feats_train["u_idx"][idx]
                    i_pos_np = feats_train["i_idx"][idx]
                    i_neg_np = rng.integers(0, ds["n_items"], size=len(idx))
                    # Rejection
                    for _ in range(4):
                        bad_neg = np.zeros(len(idx), dtype=bool)
                        for j, ur in enumerate(u_np):
                            cols = mask_train.indices[mask_train.indptr[ur]:mask_train.indptr[ur + 1]]
                            if i_neg_np[j] in cols:
                                bad_neg[j] = True
                        if not bad_neg.any(): break
                        i_neg_np[bad_neg] = rng.integers(0, ds["n_items"], size=int(bad_neg.sum()))
                    macro_pos = macro_per_item[i_pos_np]
                    macro_neg = macro_per_item[i_neg_np]
                    fine_pos = fine_per_item[i_pos_np]
                    fine_neg = fine_per_item[i_neg_np]
                    u = torch.from_numpy(u_np).to(device)
                    i_pos = torch.from_numpy(i_pos_np).to(device)
                    i_neg = torch.from_numpy(i_neg_np).to(device)
                    m_pos = torch.from_numpy(macro_pos.astype(np.int64)).to(device)
                    m_neg = torch.from_numpy(macro_neg.astype(np.int64)).to(device)
                    f_pos = torch.from_numpy(fine_pos.astype(np.int64)).to(device)
                    f_neg = torch.from_numpy(fine_neg.astype(np.int64)).to(device)
                    c_hour = torch.from_numpy(feats_train["c_hour"][idx]).to(device)
                    c_dow = torch.from_numpy(feats_train["c_dow"][idx]).to(device)
                    c_isw = torch.from_numpy(feats_train["c_isw"][idx]).to(device)
                    c_month = torch.from_numpy(feats_train["c_month"][idx]).to(device)
                    prev_geo = torch.from_numpy(feats_train["prev_geo_idx"][idx]).to(device)
                    intent_last = torch.from_numpy(feats_train["intent_last_idx"][idx]).to(device)
                    idx_pos = model._row_indices_for_item(
                        u, i_pos, m_pos, f_pos,
                        c_hour, c_dow, c_isw, c_month, prev_geo, intent_last)
                    idx_neg = model._row_indices_for_item(
                        u, i_neg, m_neg, f_neg,
                        c_hour, c_dow, c_isw, c_month, prev_geo, intent_last)
                    y_pos = model.forward_row(idx_pos)
                    y_neg = model.forward_row(idx_neg)
                    loss = -torch.nn.functional.logsigmoid(y_pos - y_neg).mean()
                    optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
                    losses.append(float(loss.item()))
                # Val R@20
                val_scores = score_all_per_request(model, feats_val, macro_per_item,
                                                     fine_per_item, device=device,
                                                     batch_size=128)
                val_r20 = _val_recall20(val_scores, val_targets, mask_train, val_users)
                history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                                  "val_r20": val_r20})
                if verbose:
                    print(f"    d={d} l2={l2} e{epoch:02d} loss={np.mean(losses):.4f} "
                          f"val R@20={val_r20:.4f}")
                if val_r20 > best_val + 1e-6:
                    best_val = val_r20; best_epoch = epoch; bad = 0
                else:
                    bad += 1
                    if bad >= patience: break
            wall = time.time() - t0
            row = {"d": d, "L2": l2, "best_val_r20": best_val,
                     "best_epoch": best_epoch, "wallclock_s": wall}
            tuning_log.append(row)
            print(f"  d={d:>3d} L2={l2:>1.0e}  best val R@20={best_val:.4f}  "
                  f"@ epoch {best_epoch}  ({wall:.1f}s)")
            if (best_cell is None) or (best_val > best_cell["best_val_r20"]):
                best_cell = row

    print(f"\n  winner: d={best_cell['d']} L2={best_cell['L2']:.0e} "
          f"epochs={best_cell['best_epoch']} (val R@20 = {best_cell['best_val_r20']:.4f})")

    # Refit on TRAIN ∪ VAL with the winner's HPs
    print(f"  refitting on train+val ...")
    feats_tv = {k: np.concatenate([feats_train[k], feats_val[k]])
                for k in feats_train}
    mask_tv = (ds["urm_train"] + ds["urm_val"]).tocsr(); mask_tv.data[:] = 1.0
    torch.manual_seed(seed + 1)
    model = ContextAwareFM(spec, d=best_cell["d"]).to(device)
    rep = train_b_full(model, feats_tv, mask_tv, macro_per_item, fine_per_item,
                          device=device,
                          lr=lr, weight_decay=best_cell["L2"],
                          batch_size=batch_size,
                          n_epochs=int(best_cell["best_epoch"]),
                          seed=seed + 1, verbose=False)
    print(f"  refit wall-clock = {rep['wallclock_s']:.1f}s")

    # Score on test
    print(f"  scoring on test ...")
    test_scores = score_all_per_request(model, feats_test, macro_per_item,
                                            fine_per_item, device=device,
                                            batch_size=128)
    np.save(out_dir / "Bfull.scores.npy", test_scores)
    pd.DataFrame(tuning_log).to_csv(out_dir / "Bfull.tuning_log.csv", index=False)
    # Save meta
    import json
    (out_dir / "Bfull.meta.json").write_text(json.dumps({
        "city": city, "tuned": True,
        "winner": {k: float(v) if isinstance(v, np.floating) else v
                    for k, v in best_cell.items()},
        "wallclock_total_s": rep["wallclock_s"],
    }, indent=2), encoding="utf-8")

    return {"best_cell": best_cell, "tuning_log": tuning_log,
              "test_scores_shape": test_scores.shape}
