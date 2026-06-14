"""Round-3 C5 — B_full feature ablation on TKY (revised priorities per C5.0).

Variants in priority order: M-time, M-geo, M-fine+meso (without meso for
now), M-intent. Base = tuned B_full (TKY test 0.0417), comparator B_blind
(0.0500). Early-stop on val R@20.

Per C5.0, the per-target-macro effect is what matters: a variant that
keeps the non-T&T gain and removes the T&T harm would beat B_blind
overall on TKY. We report each variant stratified by target macro.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.xsage.backbone_full import (
    ContextAwareFM, FeatureSpec, _build_request_features, _catalogue_indices,
)
from pipeline.step02_models.xsage.backbone import excluded_mask
from pipeline.step02_models.xsage.orchestrator import _load_city
from pipeline.step02_models.xsage.metrics import topk_from_scores


K_TOP = 20
SHORT_HEAD = 0.20

# Mapping from variant name to active feature groups
ALL_GROUPS = ("user", "item", "macro", "fine",
                "hour", "dow", "isw", "month",
                "prev_geo", "intent_last")
VARIANTS = {
    "M_full":      ALL_GROUPS,
    "M_minus_time": tuple(g for g in ALL_GROUPS
                          if g not in {"hour", "dow", "isw", "month"}),
    "M_minus_geo": tuple(g for g in ALL_GROUPS if g not in {"prev_geo"}),
    "M_minus_fine": tuple(g for g in ALL_GROUPS if g not in {"fine"}),
    "M_minus_intent": tuple(g for g in ALL_GROUPS if g not in {"intent_last"}),
}


class AblatedFM(ContextAwareFM):
    """Same model as ContextAwareFM but skips a subset of feature groups."""

    def __init__(self, spec: FeatureSpec, d: int, active_groups: tuple[str, ...]):
        super().__init__(spec, d=d)
        self.active_groups = active_groups
        self.context_groups = tuple(g for g in
                                          ("user", "hour", "dow", "isw", "month",
                                           "prev_geo", "intent_last")
                                          if g in active_groups)
        self.item_groups = tuple(g for g in ("item", "macro", "fine")
                                    if g in active_groups)

    def _row_indices_ablated(self, u, i, macro_of_i, fine_of_i,
                                c_hour, c_dow, c_isw, c_month, prev_geo, intent_last):
        offs = self.offsets
        all_vals = {
            "user": u + offs["user"],
            "item": i + offs["item"],
            "macro": macro_of_i + offs["macro"],
            "fine": fine_of_i + offs["fine"],
            "hour": c_hour + offs["hour"],
            "dow": c_dow + offs["dow"],
            "isw": c_isw + offs["isw"],
            "month": c_month + offs["month"],
            "prev_geo": prev_geo + offs["prev_geo"],
            "intent_last": intent_last + offs["intent_last"],
        }
        active_tensors = [all_vals[g] for g in self.active_groups]
        return torch.stack(active_tensors, dim=-1)

    def score_full_catalogue_ablated(self, ctx_idx, item_idx_in_emb,
                                          macro_idx_in_emb, fine_idx_in_emb):
        """Ablated catalogue scoring; ctx_idx has only the active context cols."""
        # Context-side
        e_ctx = self.emb(ctx_idx); w_ctx = self.lin(ctx_idx).squeeze(-1).sum(dim=-1)
        sum_ctx = e_ctx.sum(dim=-2); sumsq_ctx = e_ctx.pow(2).sum(dim=-2)
        # Item-side: pick active item embeddings
        item_embs = []; item_lins = []
        if "item" in self.item_groups:
            item_embs.append(self.emb(item_idx_in_emb))
            item_lins.append(self.lin(item_idx_in_emb).squeeze(-1))
        if "macro" in self.item_groups:
            item_embs.append(self.emb(macro_idx_in_emb))
            item_lins.append(self.lin(macro_idx_in_emb).squeeze(-1))
        if "fine" in self.item_groups:
            item_embs.append(self.emb(fine_idx_in_emb))
            item_lins.append(self.lin(fine_idx_in_emb).squeeze(-1))
        sum_item = sum(item_embs)
        sumsq_item = sum(e.pow(2) for e in item_embs)
        w_item = sum(item_lins)
        sum_total = sum_ctx.unsqueeze(1) + sum_item.unsqueeze(0)
        sumsq_total = sumsq_ctx.unsqueeze(1) + sumsq_item.unsqueeze(0)
        order2 = 0.5 * (sum_total.pow(2).sum(dim=-1) - sumsq_total.sum(dim=-1))
        linear = self.bias + w_ctx.unsqueeze(-1) + w_item.unsqueeze(0)
        return linear + order2


def _build_ablated_ctx(feats: dict, idx, offs: dict, active_groups: tuple,
                          device) -> torch.Tensor:
    name_to_val = {
        "user": feats["u_idx"][idx],
        "hour": feats["c_hour"][idx], "dow": feats["c_dow"][idx],
        "isw": feats["c_isw"][idx], "month": feats["c_month"][idx],
        "prev_geo": feats["prev_geo_idx"][idx],
        "intent_last": feats["intent_last_idx"][idx],
    }
    cols = []
    for g in ("user", "hour", "dow", "isw", "month", "prev_geo", "intent_last"):
        if g not in active_groups:
            continue
        cols.append(torch.from_numpy(name_to_val[g]).to(device) + offs[g])
    return torch.stack(cols, dim=-1)


def train_and_score(city: str, variant_name: str, active_groups: tuple,
                       max_epochs: int = 12, batch_size: int = 4096,
                       lr: float = 5e-3, weight_decay: float = 1e-5,
                       d: int = 128, seed: int = 42, val_patience: int = 3,
                       verbose: bool = False) -> dict:
    print(f"\n>>> {city} {variant_name} (active: {len(active_groups)} groups)")
    t_start = time.time()
    ds = _load_city(city)
    df_train = ds["df_train"]; df_val = ds["df_val"]; df_test = ds["df_test"]
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    fine_vals = sorted(set(df_train["cat_fine"]).union(df_val["cat_fine"])
                          .union(df_test["cat_fine"]))
    fine_to_idx = {v: i for i, v in enumerate(fine_vals)}
    prev_vals = sorted(set(df_train["prev_geohash5"]).union(df_val["prev_geohash5"])
                          .union(df_test["prev_geohash5"]))
    prev_clean = [v for v in prev_vals if v != "__NONE__"]
    geo_to_idx = {"__NONE__": 0, **{v: i + 1 for i, v in enumerate(prev_clean)}}
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
                                                              ds["macro_to_idx"], fine_to_idx)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"  device={device}")
    torch.manual_seed(seed)
    model = AblatedFM(spec, d=d, active_groups=active_groups).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mask_train = ds["urm_train"].tocsr().copy(); mask_train.data[:] = 1.0
    n_items = ds["n_items"]
    rng = np.random.default_rng(seed)

    # Training with val ES
    best_val_r20 = -1.0; best_epoch = 0; bad = 0
    best_state = None
    n_train = len(feats_train["u_idx"])
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(n_train)
        losses = []
        for s in range(0, n_train, batch_size):
            idx = order[s:s + batch_size]
            u_np = feats_train["u_idx"][idx]
            i_pos_np = feats_train["i_idx"][idx]
            i_neg_np = rng.integers(0, n_items, size=len(idx))
            for _ in range(4):
                bad_neg = np.zeros(len(idx), dtype=bool)
                for j, ur in enumerate(u_np):
                    cols = mask_train.indices[mask_train.indptr[ur]:mask_train.indptr[ur + 1]]
                    if i_neg_np[j] in cols:
                        bad_neg[j] = True
                if not bad_neg.any(): break
                i_neg_np[bad_neg] = rng.integers(0, n_items, size=int(bad_neg.sum()))
            u = torch.from_numpy(u_np).to(device)
            i_pos = torch.from_numpy(i_pos_np).to(device)
            i_neg = torch.from_numpy(i_neg_np).to(device)
            macro_pos = torch.from_numpy(macro_per_item[i_pos_np].astype(np.int64)).to(device)
            macro_neg = torch.from_numpy(macro_per_item[i_neg_np].astype(np.int64)).to(device)
            fine_pos = torch.from_numpy(fine_per_item[i_pos_np].astype(np.int64)).to(device)
            fine_neg = torch.from_numpy(fine_per_item[i_neg_np].astype(np.int64)).to(device)
            c_hour = torch.from_numpy(feats_train["c_hour"][idx]).to(device)
            c_dow = torch.from_numpy(feats_train["c_dow"][idx]).to(device)
            c_isw = torch.from_numpy(feats_train["c_isw"][idx]).to(device)
            c_month = torch.from_numpy(feats_train["c_month"][idx]).to(device)
            prev_geo = torch.from_numpy(feats_train["prev_geo_idx"][idx]).to(device)
            intent_last = torch.from_numpy(feats_train["intent_last_idx"][idx]).to(device)
            idx_pos = model._row_indices_ablated(
                u, i_pos, macro_pos, fine_pos,
                c_hour, c_dow, c_isw, c_month, prev_geo, intent_last)
            idx_neg = model._row_indices_ablated(
                u, i_neg, macro_neg, fine_neg,
                c_hour, c_dow, c_isw, c_month, prev_geo, intent_last)
            y_pos = model.forward_row(idx_pos); y_neg = model.forward_row(idx_neg)
            loss = -F.logsigmoid(y_pos - y_neg).mean()
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
            losses.append(float(loss.item()))

        # Val eval
        model.eval()
        with torch.no_grad():
            offs = model.offsets
            item_idx_in_emb = torch.arange(n_items, device=device) + offs["item"]
            macro_idx_in_emb = torch.from_numpy(macro_per_item).to(device) + offs["macro"]
            fine_idx_in_emb = torch.from_numpy(fine_per_item).to(device) + offs["fine"]
            val_scores = np.zeros((len(feats_val["u_idx"]), n_items), dtype=np.float32)
            bs = 128
            for s in range(0, len(feats_val["u_idx"]), bs):
                vidx = np.arange(s, min(s + bs, len(feats_val["u_idx"])))
                ctx = _build_ablated_ctx(feats_val, vidx, offs, active_groups, device)
                sc = model.score_full_catalogue_ablated(
                    ctx, item_idx_in_emb, macro_idx_in_emb, fine_idx_in_emb)
                val_scores[vidx] = sc.cpu().numpy()
            # Apply train-mask exclude
            hits = 0
            for b in range(len(feats_val["u_idx"])):
                ur = int(feats_val["u_idx"][b])
                s_row = val_scores[b].copy()
                cols = mask_train.indices[mask_train.indptr[ur]:mask_train.indptr[ur + 1]]
                if len(cols): s_row[cols] = -np.inf
                tgt = int(feats_val["i_idx"][b])
                if (s_row > s_row[tgt]).sum() < K_TOP: hits += 1
            val_r20 = hits / max(1, len(feats_val["u_idx"]))
        if verbose:
            print(f"    e{epoch:02d}  loss={np.mean(losses):.4f}  val R@20={val_r20:.4f}")
        if val_r20 > best_val_r20 + 1e-6:
            best_val_r20 = val_r20; best_epoch = epoch; bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= val_patience: break

    model.load_state_dict(best_state)
    # Refit on train+val (same n_epochs as best)
    print(f"  refitting on train+val for {best_epoch} epochs (val R@20={best_val_r20:.4f}) ...")
    torch.manual_seed(seed + 1)
    model = AblatedFM(spec, d=d, active_groups=active_groups).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    feats_tv = {k: np.concatenate([feats_train[k], feats_val[k]]) for k in feats_train}
    mask_tv = (ds["urm_train"] + ds["urm_val"]).tocsr(); mask_tv.data[:] = 1.0
    n_tv = len(feats_tv["u_idx"])
    for ep in range(1, best_epoch + 1):
        model.train()
        order = rng.permutation(n_tv)
        for s in range(0, n_tv, batch_size):
            idx = order[s:s + batch_size]
            u_np = feats_tv["u_idx"][idx]
            i_pos_np = feats_tv["i_idx"][idx]
            i_neg_np = rng.integers(0, n_items, size=len(idx))
            for _ in range(4):
                bad_neg = np.zeros(len(idx), dtype=bool)
                for j, ur in enumerate(u_np):
                    cols = mask_tv.indices[mask_tv.indptr[ur]:mask_tv.indptr[ur + 1]]
                    if i_neg_np[j] in cols: bad_neg[j] = True
                if not bad_neg.any(): break
                i_neg_np[bad_neg] = rng.integers(0, n_items, size=int(bad_neg.sum()))
            u = torch.from_numpy(u_np).to(device)
            i_pos = torch.from_numpy(i_pos_np).to(device)
            i_neg = torch.from_numpy(i_neg_np).to(device)
            macro_pos = torch.from_numpy(macro_per_item[i_pos_np].astype(np.int64)).to(device)
            macro_neg = torch.from_numpy(macro_per_item[i_neg_np].astype(np.int64)).to(device)
            fine_pos = torch.from_numpy(fine_per_item[i_pos_np].astype(np.int64)).to(device)
            fine_neg = torch.from_numpy(fine_per_item[i_neg_np].astype(np.int64)).to(device)
            c_hour = torch.from_numpy(feats_tv["c_hour"][idx]).to(device)
            c_dow = torch.from_numpy(feats_tv["c_dow"][idx]).to(device)
            c_isw = torch.from_numpy(feats_tv["c_isw"][idx]).to(device)
            c_month = torch.from_numpy(feats_tv["c_month"][idx]).to(device)
            prev_geo = torch.from_numpy(feats_tv["prev_geo_idx"][idx]).to(device)
            intent_last = torch.from_numpy(feats_tv["intent_last_idx"][idx]).to(device)
            idx_pos = model._row_indices_ablated(
                u, i_pos, macro_pos, fine_pos,
                c_hour, c_dow, c_isw, c_month, prev_geo, intent_last)
            idx_neg = model._row_indices_ablated(
                u, i_neg, macro_neg, fine_neg,
                c_hour, c_dow, c_isw, c_month, prev_geo, intent_last)
            y_pos = model.forward_row(idx_pos); y_neg = model.forward_row(idx_neg)
            loss = -F.logsigmoid(y_pos - y_neg).mean()
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()

    # Score test
    model.eval()
    with torch.no_grad():
        offs = model.offsets
        item_idx_in_emb = torch.arange(n_items, device=device) + offs["item"]
        macro_idx_in_emb = torch.from_numpy(macro_per_item).to(device) + offs["macro"]
        fine_idx_in_emb = torch.from_numpy(fine_per_item).to(device) + offs["fine"]
        n_test = len(feats_test["u_idx"])
        test_scores = np.zeros((n_test, n_items), dtype=np.float32)
        bs = 128
        for s in range(0, n_test, bs):
            tidx = np.arange(s, min(s + bs, n_test))
            ctx = _build_ablated_ctx(feats_test, tidx, offs, active_groups, device)
            sc = model.score_full_catalogue_ablated(
                ctx, item_idx_in_emb, macro_idx_in_emb, fine_idx_in_emb)
            test_scores[tidx] = sc.cpu().numpy()

    excl_test = (ds["urm_train"] + ds["urm_val"]).tocsr(); excl_test.data[:] = 1.0
    # Per-request R@20 and N@20
    r20_per_req = np.zeros(n_test); n20_per_req = np.zeros(n_test)
    for b in range(n_test):
        ur = int(feats_test["u_idx"][b])
        s_row = test_scores[b].copy()
        cols = excl_test.indices[excl_test.indptr[ur]:excl_test.indptr[ur + 1]]
        if len(cols): s_row[cols] = -np.inf
        tgt = int(feats_test["i_idx"][b])
        ts = s_row[tgt]
        rank = int((s_row > ts).sum()) + 1
        if rank <= K_TOP:
            r20_per_req[b] = 1.0
            n20_per_req[b] = 1.0 / np.log2(rank + 1)

    cat_target = df_test["cat_macro"].values.astype(str)
    tt_mask = cat_target == "Travel & Transport"

    # Save per-request arrays (for B9 stat hardening: paired Wilcoxon).
    perreq_dir = REPO_ROOT / "outputs" / "round3" / "C5" / "per_request"
    perreq_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(perreq_dir / f"{city}_{variant_name}.npz",
                          R20_per_request=r20_per_req.astype(np.float32),
                          N20_per_request=n20_per_req.astype(np.float32),
                          tt_mask=tt_mask)

    result = {
        "city": city, "variant": variant_name,
        "active_groups": list(active_groups),
        "best_epoch": best_epoch, "best_val_r20": best_val_r20,
        "R20_all": float(r20_per_req.mean()),
        "N20_all": float(n20_per_req.mean()),
        "R20_TT": float(r20_per_req[tt_mask].mean()) if tt_mask.any() else float("nan"),
        "N20_TT": float(n20_per_req[tt_mask].mean()) if tt_mask.any() else float("nan"),
        "R20_nonTT": float(r20_per_req[~tt_mask].mean()) if (~tt_mask).any() else float("nan"),
        "N20_nonTT": float(n20_per_req[~tt_mask].mean()) if (~tt_mask).any() else float("nan"),
        "wallclock_s": time.time() - t_start,
    }
    print(f"  R@20 all={result['R20_all']:.4f}  T&T={result['R20_TT']:.4f}  "
          f"non-T&T={result['R20_nonTT']:.4f}  ({result['wallclock_s']:.1f}s)")
    return result


def main() -> int:
    out_dir = REPO_ROOT / "outputs" / "round3" / "C5"
    out_dir.mkdir(parents=True, exist_ok=True)
    city = "TKY"
    rows = []
    for vname in ("M_full", "M_minus_time", "M_minus_geo",
                    "M_minus_fine", "M_minus_intent"):
        rows.append(train_and_score(city, vname, VARIANTS[vname],
                                          max_epochs=12, d=128, lr=5e-3,
                                          weight_decay=1e-5, val_patience=3,
                                          verbose=False))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation_results.csv", index=False)

    # Stratified table
    print("\n=== TKY ablation summary ===")
    print(f"{'variant':18s}  {'R@20 all':>9s}  {'R@20 T&T':>9s}  {'R@20 non-T&T':>13s}  Δ vs B_blind 0.0500")
    print("-" * 85)
    for r in rows:
        delta = r["R20_all"] - 0.0500
        print(f"{r['variant']:18s}  {r['R20_all']:>9.4f}  {r['R20_TT']:>9.4f}  "
              f"{r['R20_nonTT']:>13.4f}  {delta:>+8.4f}")

    # Verdict
    full_r20 = next(r for r in rows if r["variant"] == "M_full")["R20_all"]
    best_variant = max(rows, key=lambda r: r["R20_all"])
    found_recovery = best_variant["R20_all"] >= 0.0500
    verdict = ("investigative narrative — variant recovers ≥ B_blind"
                  if found_recovery
                  else "defensive narrative — no reduced B_full beats B_blind on TKY")
    payload = {"city": city,
                 "base_M_full_R20": full_r20,
                 "B_blind_R20": 0.0500,
                 "best_variant": best_variant["variant"],
                 "best_variant_R20": best_variant["R20_all"],
                 "best_variant_R20_TT": best_variant["R20_TT"],
                 "best_variant_R20_nonTT": best_variant["R20_nonTT"],
                 "recovery_found": found_recovery,
                 "verdict": verdict,
                 "rows": rows}
    (out_dir / "verdict.json").write_text(json.dumps(payload, indent=2,
                                                          default=str),
                                              encoding="utf-8")
    print(f"\nverdict: {verdict}")
    print(f"best variant: {best_variant['variant']} R@20={best_variant['R20_all']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
