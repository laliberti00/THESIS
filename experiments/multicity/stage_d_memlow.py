"""Memory-efficient Stage D for big cities (round 4 Istanbul fix).

The frozen `pipeline.step02_models.xsage.orchestrator.run_stage_d`
materialises B_blind, B_full, p_B, p_S, and ALL kappa-sweep X-SAGE
scores as (n_test, n_items) float32 matrices simultaneously. For
Istanbul (n_test ≈ 136k, n_items ≈ 9.3k) that is ~36 GB peak —
SIGKILL on an 8 GB MacBook.

This shim reproduces the **same three_way.csv headline numbers**
(B_blind R@20, B_full R@20, X-SAGE R@20 / NDCG@20 for each κ) by:
  - keeping scores_blind only as the per-USER matrix (n_users, n_items)
    — for Istanbul that is 22 631 × 9 305 × 4B = ~840 MB (manageable);
  - mmap'ing Bfull.scores.npy from disk (no resident RAM until read);
  - iterating test requests in batches of `BATCH_SIZE` (default 1024),
    computing the additive X-SAGE nudge per batch per κ on the fly,
    accumulating sum / count for R@20 and NDCG@20;
  - never holding two full kappa matrices simultaneously.

Per-batch peak memory: ~150 MB. Total Stage D for Istanbul: < 2 GB.

Output: SAME `three_way.csv` schema as the frozen Stage D + a
`stage_d_memlow_summary.json`. Skips the matched-pair Wilcoxon and
modulation log to keep the shim minimal — those require per-user
arrays we deliberately do not materialise. They can be regenerated
later from the saved Bblind.npz / Bfull.npz / XSAGE.npz per-user
metrics if needed.

CLI:

    python -m experiments.multicity.stage_d_memlow --city istanbul

Writes done-marker compatible with the orchestrator's checkpointing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.xsage.backbone import excluded_mask, load_or_refit
from pipeline.step02_models.xsage.orchestrator import _load_city
from pipeline.step02_models.xsage.recommendation import (
    fit_situation_biases_z,
)


KAPPA_SWEEP = (0.0, 0.1, 0.25, 0.5, 1.0)
CUTOFF = 20
BATCH_SIZE = 1024


# ---------------------------------------------------------------------------
# Per-request metric — pure NumPy, batch-safe
# ---------------------------------------------------------------------------

def _hits_at_k_and_ndcg(scores_batch: np.ndarray,
                          u_batch: np.ndarray,
                          i_target_batch: np.ndarray,
                          excl,
                          K: int = CUTOFF) -> tuple[np.ndarray, np.ndarray]:
    """For each row in the batch:
       hit@K = 1 if rank of target ≤ K else 0
       ndcg@K = 1/log2(rank+1) if hit else 0

    Replicates *exactly* the frozen Stage D's `_next_item_metrics_per_user`:
    copy the row, set excluded items to -inf (so a re-visited target also
    gets -inf → automatic miss), then count items strictly above target.
    """
    B = scores_batch.shape[0]
    hits = np.zeros(B, dtype=np.float32)
    ndcg = np.zeros(B, dtype=np.float32)
    for b in range(B):
        u = int(u_batch[b])
        s = scores_batch[b].copy()           # ~37 KB / row — cheap
        cols = excl.indices[excl.indptr[u]:excl.indptr[u + 1]]
        if len(cols):
            s[cols] = -np.inf
        tgt = int(i_target_batch[b])
        ts = s[tgt]
        rank = int((s > ts).sum()) + 1
        if rank <= K:
            hits[b] = 1.0
            ndcg[b] = 1.0 / np.log2(rank + 1)
    return hits, ndcg


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_memlow_stage_d(city: str, batch_size: int = BATCH_SIZE,
                          kappa_sweep: tuple = KAPPA_SWEEP,
                          verbose: bool = True) -> dict:
    t_start = time.time()
    out_stage = REPO_ROOT / "outputs" / city / "xsage" / "recommendation"
    out_stage.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[memlow Stage D] city={city}  batch={batch_size}")
        print(f"  kappa sweep = {list(kappa_sweep)}")

    # --- Load Stage A artefacts -------------------------------------------
    sit_dir = REPO_ROOT / "outputs" / city / "xsage" / "situations"
    fit = np.load(sit_dir / "fit.npz", allow_pickle=True)
    z_train = np.asarray(fit["core_label_train"]).astype(np.int32)
    z_val = np.asarray(fit["core_label_val"]).astype(np.int32)
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)
    membership_test = np.asarray(fit["membership_test"]).astype(np.float32)
    K_sit = int(max(z_train.max(), z_val.max(), z_test.max()) + 1)

    # --- Dataset and macros -----------------------------------------------
    ds = _load_city(city)
    df_train = ds["df_train"]; df_val = ds["df_val"]; df_test = ds["df_test"]
    n_items = ds["n_items"]; n_macros = ds["n_macros"]
    macro_to_idx = ds["macro_to_idx"]

    u_test = df_test["u_idx"].values.astype(np.int64)
    i_target = df_test["i_idx"].values.astype(np.int64)
    n_test = len(u_test)

    # Per-item macro index — small array (n_items,)
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    item_macro = np.full(n_items, -1, dtype=np.int32)
    for i, m in zip(df_all["i_idx"].values, df_all["cat_macro"].values):
        item_macro[int(i)] = macro_to_idx.get(m, 0)
    if (item_macro < 0).any():
        item_macro[item_macro < 0] = 0           # safe default for missing

    # γ per request: 1 on core, 1/|T| on boundary
    competing = (membership_test > 0).sum(axis=1).astype(np.int32)
    gamma_per_req = np.where(isb_test, 1.0 / np.maximum(competing, 1), 1.0
                                ).astype(np.float32)

    # --- Bias z-scored per situation (small: K_sit × n_macros) -----------
    z_tv = np.concatenate([z_train, z_val])
    m_tr = np.array([macro_to_idx[c] for c in df_train["cat_macro"].values],
                       dtype=np.int64)
    m_va = np.array([macro_to_idx[c] for c in df_val["cat_macro"].values],
                       dtype=np.int64)
    macro_tv = np.concatenate([m_tr, m_va])
    biases_z = fit_situation_biases_z(z_tv, macro_tv, K=K_sit,
                                           n_macros=n_macros, lam=50.0)   # (K_sit, n_macros)

    # --- Backbone scores --------------------------------------------------
    if verbose:
        print(f"  loading backbone B_blind (per-user, full matrix) ...")
    scores_blind_uitem = load_or_refit(city, model_name="FM", verbose=False)
    # Don't broadcast to n_test rows — we'll index per batch.
    if verbose:
        size_mb = scores_blind_uitem.nbytes / 1024**2
        print(f"    B_blind shape={scores_blind_uitem.shape}  "
              f"size={size_mb:.1f} MB")

    bfull_path = (REPO_ROOT / "outputs" / city / "xsage" / "backbone"
                    / "Bfull.scores.npy")
    if not bfull_path.exists():
        raise FileNotFoundError(
            f"{bfull_path} missing — run `backbones` stage first")
    if verbose:
        print(f"  mmap'ing B_full from {bfull_path} ...")
    scores_full_mmap = np.load(bfull_path, mmap_mode="r")
    if verbose:
        size_mb = scores_full_mmap.nbytes / 1024**2
        print(f"    B_full shape={scores_full_mmap.shape}  "
              f"size_on_disk={size_mb:.1f} MB (mmap)")

    excl = excluded_mask(city, n_items)

    # --- Batched evaluation ----------------------------------------------
    # Per-request hit / ndcg arrays (small: n_test × float32 per variant).
    variants = ["B_blind", "B_full"] + [f"X-SAGE_k{k:.2f}" for k in kappa_sweep]
    hit_r20 = {v: np.zeros(n_test, dtype=np.float32) for v in variants}
    hit_n20 = {v: np.zeros(n_test, dtype=np.float32) for v in variants}
    n_seen = 0

    if verbose:
        print(f"  batched eval: n_test={n_test:,}  batch_size={batch_size}")
    t_eval = time.time()
    for start in range(0, n_test, batch_size):
        end = min(n_test, start + batch_size)
        idx = slice(start, end)
        u_b = u_test[idx]
        i_b = i_target[idx]
        # B_blind batch: per-user indexing → (B, n_items)
        sb_batch = scores_blind_uitem[u_b]                       # (B, n_items)
        # B_full batch from mmap (per-request)
        sf_batch = np.asarray(scores_full_mmap[idx])             # (B, n_items)
        gamma_b = gamma_per_req[idx]                              # (B,)
        memb_b = membership_test[idx]                             # (B, K_sit)

        # B_blind — keep the (h, n) arrays; reuse for κ=0 (matched-OFF).
        h_blind, n_blind = _hits_at_k_and_ndcg(sb_batch, u_b, i_b, excl)
        hit_r20["B_blind"][start:end] = h_blind
        hit_n20["B_blind"][start:end] = n_blind
        # B_full
        h_full, n_full = _hits_at_k_and_ndcg(sf_batch, u_b, i_b, excl)
        hit_r20["B_full"][start:end] = h_full
        hit_n20["B_full"][start:end] = n_full

        # X-SAGE per-κ (additive combiner; same formula as round-2 1.5).
        s_per_macro = memb_b @ biases_z                           # (B, n_macros)
        s_per_item = s_per_macro[:, item_macro]                   # (B, n_items)
        for k in kappa_sweep:
            key = f"X-SAGE_k{k:.2f}"
            if k == 0.0:
                # Matched-OFF identity: X-SAGE(κ=0) = B_blind exactly.
                hk, nk = h_blind, n_blind
            else:
                # ŝ = s_B + κ · γ_S · Σ_k r_k · b̃^{(k)}_{c(i)}
                nudge = (k * gamma_b[:, None]).astype(np.float32) * s_per_item
                sk = sb_batch + nudge
                hk, nk = _hits_at_k_and_ndcg(sk, u_b, i_b, excl)
            hit_r20[key][start:end] = hk
            hit_n20[key][start:end] = nk

        n_seen += (end - start)
        if verbose and (start // batch_size) % 20 == 0:
            print(f"    {n_seen:,}/{n_test:,}  "
                  f"({time.time()-t_eval:.1f}s elapsed)")
    if verbose:
        print(f"  batched eval done in {time.time()-t_eval:.1f}s")

    # --- Aggregate (macro-user average to match frozen Stage D) ----------
    # Per-user mean of per-request metrics, then mean across users.
    unique_u, inv_idx = np.unique(u_test, return_inverse=True)
    n_users = len(unique_u)
    counts_per_user = np.zeros(n_users, dtype=np.float32)
    np.add.at(counts_per_user, inv_idx, 1.0)

    def _macro_user_avg(arr: np.ndarray) -> float:
        sums = np.zeros(n_users, dtype=np.float64)
        np.add.at(sums, inv_idx, arr.astype(np.float64))
        per_user = sums / counts_per_user
        return float(per_user.mean())

    results = []
    results.append({"variant": "B_blind",
                       "R_at_20": _macro_user_avg(hit_r20["B_blind"]),
                       "N_at_20": _macro_user_avg(hit_n20["B_blind"]),
                       "kappa": "—"})
    for k in kappa_sweep:
        key = f"X-SAGE_k{k:.2f}"
        results.append({"variant": "X-SAGE",
                           "R_at_20": _macro_user_avg(hit_r20[key]),
                           "N_at_20": _macro_user_avg(hit_n20[key]),
                           "kappa": k})
    results.append({"variant": "B_full",
                       "R_at_20": _macro_user_avg(hit_r20["B_full"]),
                       "N_at_20": _macro_user_avg(hit_n20["B_full"]),
                       "kappa": "—"})

    # --- Output ------------------------------------------------------------
    three_way_path = out_stage / "three_way.csv"
    pd.DataFrame(results).to_csv(three_way_path, index=False)
    if verbose:
        print(f"  wrote {three_way_path}")

    # Find best κ
    xsage_rows = [r for r in results if r["variant"] == "X-SAGE"]
    best = max(xsage_rows, key=lambda r: r["R_at_20"])

    payload = {
        "city": city, "stage": "D_memlow",
        "n_test": int(n_test), "n_items": int(n_items),
        "K_sit": K_sit, "kappa_sweep": list(kappa_sweep),
        "batch_size": int(batch_size),
        "B_blind_R20": results[0]["R_at_20"],
        "B_blind_N20": results[0]["N_at_20"],
        "B_full_R20": results[-1]["R_at_20"],
        "B_full_N20": results[-1]["N_at_20"],
        "best_xsage_kappa": best["kappa"],
        "best_xsage_R20": best["R_at_20"],
        "best_xsage_N20": best["N_at_20"],
        "delta_Bfull_vs_Bblind_R20": results[-1]["R_at_20"] - results[0]["R_at_20"],
        "wallclock_s": time.time() - t_start,
        "note": "matched-pair Wilcoxon + modulation log skipped (require per-user arrays)",
    }
    (out_stage / "stage_d_memlow_summary.json").write_text(
        json.dumps(payload, indent=2))

    # Touch checkpoint marker compatible with run_multicity orchestrator
    mc_city_dir = REPO_ROOT / "outputs_multicity" / city
    mc_city_dir.mkdir(parents=True, exist_ok=True)
    (mc_city_dir / ".done_stageD").write_text(json.dumps({
        "ok": True, "via": "stage_d_memlow",
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_test": int(n_test),
        "wallclock_s": payload["wallclock_s"],
    }, indent=2))
    # Remove any stale fail marker
    fail = mc_city_dir / ".fail_stageD"
    if fail.exists():
        fail.unlink()
    # Copy summary into the multicity per-city dir as well
    import shutil
    shutil.copy2(out_stage / "stage_d_memlow_summary.json",
                  mc_city_dir / "summary_stageD.json")

    if verbose:
        print(f"\n=== Three-way (memlow) for {city} ===")
        for r in results:
            kappa_str = f"κ={r['kappa']}" if r["variant"] == "X-SAGE" else "—"
            print(f"  {r['variant']:8s} {kappa_str:>8s}  "
                  f"R@20={r['R_at_20']:.4f}  N@20={r['N_at_20']:.4f}")
        print(f"  best κ = {best['kappa']}  R@20 = {best['R_at_20']:.4f}")
        print(f"  Δ B_full − B_blind R@20 = "
              f"{payload['delta_Bfull_vs_Bblind_R20']:+.4f}")
        print(f"  wallclock = {payload['wallclock_s']:.1f}s")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    run_memlow_stage_d(args.city, batch_size=args.batch_size,
                          verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
