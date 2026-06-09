"""Part B — anti-collapse sweep on NYC (CPU; doesn't touch the TKY MPS run).

Configurations swept:
    C0  baseline           : original init, T=1.0, λ=0,    warmup=0  (= V1 as before)
    C1  balanced init      : init_scale=1e-3, T=1.0, λ=0,  warmup=0
    C2  C1 + balance loss  : init_scale=1e-3, T=1.0, λ=0.1, warmup=0
    C3  C1 + balance + warm: init_scale=1e-3, T=1.0, λ=0.1, warmup=5
    Cλ  λ sweep            : C1 + λ ∈ {0.01, 0.5, 1.0}                (around 0.1)
    CT  T sweep            : C1 + T ∈ {1.5, 2.0} (anneal to 1.0)

For each: z-distribution on test, π entropy mean, R@20 + NDCG@20 (1 seed for
screening). Best config re-run on 2 extra seeds; we also retrain V0 with
identical protocol for a fair internal comparison (V0 from the prior run
used train+val refit which gives it a small unfair edge).

Classification: BUG FOUND / FIXABLE & USEFUL / FORCED BUT USELESS.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.situational.anti_collapse import (
    AntiCollapseConfig, apply_balanced_init, eval_test_with_T,
    train_with_anti_collapse,
)
from pipeline.step02_models.situational.data import build_city_dataset
from pipeline.step02_models.situational.model import (SituationalConfig,
                                                          SituationalModel)
from pipeline.step02_models.situational.trainer import (TrainConfig,
                                                            train_situational)
from pipeline.step02_models.situational.ranker import (
    METRIC_NAMES, rank_split,
)


DEVICE = torch.device("cpu")
NYC = REPO_ROOT / "data" / "processed" / "NYC"
OUT = REPO_ROOT / "outputs" / "NYC" / "situational"
OUT.mkdir(parents=True, exist_ok=True)

# Same HP as Part A / the orchestrator's V0 winner.
HP = dict(d=64, lr=5e-3, weight_decay=1e-5, batch_size=1024,
            max_epochs=16, patience=6, eval_every=1)
K = 6


def _v0_baseline_train_only(ds, seed: int, verbose: bool = False) -> dict:
    """Train V0 (situation OFF) train-only — for a fair internal comparison."""
    cfg = SituationalConfig(n_users=ds.n_users, n_items=ds.n_items,
                              n_macros=ds.n_macros, n_geo=ds.n_geo,
                              K=1, d=HP["d"], d_g=8, d_e=8, r=4,
                              use_situation=False, use_intent=False)
    torch.manual_seed(seed)
    model = SituationalModel(cfg).to(DEVICE)
    t_cfg = TrainConfig(lr=HP["lr"], weight_decay=HP["weight_decay"],
                         batch_size=HP["batch_size"], max_epochs=HP["max_epochs"],
                         patience=HP["patience"], eval_every=1,
                         eval_batch_size=512, seed=seed, verbose=verbose)
    rep = train_situational(model, ds, t_cfg, device=DEVICE)
    # Eval on test
    exclude = (ds.urm_train + ds.urm_val).tocsr()
    exclude.data[:] = 1.0
    res = rank_split(model, ds, ds.test, cutoffs=(1, 5, 10, 20, 40, 50, 100),
                      exclude_mask=exclude, device=DEVICE, save_z=False,
                      batch_size=512)
    return {
        "summary": {
            "RECALL_20": float(res.metrics[20]["RECALL"].mean()),
            "NDCG_20": float(res.metrics[20]["NDCG"].mean()),
        },
        "best_val_recall20": rep["best_val_recall20"],
        "best_epoch": rep["best_epoch"],
        "wallclock_s": rep["wallclock_s"],
    }


def run_config(name: str, ds, seed: int,
                init_scale: float, temperature: float, balance_weight: float,
                warmup_epochs: int, anneal_temperature: bool = False,
                verbose: bool = False) -> dict:
    """One Part-B variant on NYC."""
    print(f"\n>>> [{name}] seed={seed}  init={init_scale}  T={temperature}  "
          f"λ={balance_weight}  warmup={warmup_epochs}"
          f"{' anneal' if anneal_temperature else ''}")
    cfg = SituationalConfig(n_users=ds.n_users, n_items=ds.n_items,
                              n_macros=ds.n_macros, n_geo=ds.n_geo,
                              K=K, d=HP["d"], d_g=8, d_e=8, r=4,
                              use_situation=True, use_intent=True)
    torch.manual_seed(seed)
    model = SituationalModel(cfg).to(DEVICE)
    apply_balanced_init(model, scale=init_scale)
    anti = AntiCollapseConfig(init_scale=init_scale, temperature=temperature,
                                balance_weight=balance_weight,
                                warmup_epochs=warmup_epochs,
                                anneal_temperature=anneal_temperature,
                                temperature_anneal_to=1.0)
    t_cfg = TrainConfig(lr=HP["lr"], weight_decay=HP["weight_decay"],
                         batch_size=HP["batch_size"], max_epochs=HP["max_epochs"],
                         patience=HP["patience"], eval_every=1,
                         eval_batch_size=512, seed=seed, verbose=verbose)
    t0 = time.time()
    rep = train_with_anti_collapse(model, ds, t_cfg, anti, device=DEVICE)
    res = eval_test_with_T(model, ds, temperature=temperature, device=DEVICE)
    wall = time.time() - t0
    return {
        "name": name, "seed": seed,
        "anti": anti.__dict__,
        "train_report": rep,
        "test": {
            "summary": res["summary"],
            "z_distribution": res["z_distribution"],
            "pi_entropy_mean": res["pi_entropy_mean"],
            "K_used": int(np.sum(np.array(res["z_distribution"]) > 0)),
        },
        "wallclock_s": wall,
    }


def classify(c0_v1, c_best, v0_train_only_R20, v0_train_only_N20) -> dict:
    """BUG FOUND / FIXABLE & USEFUL / FORCED BUT USELESS."""
    best_R = c_best["test"]["summary"]["RECALL_20"]
    best_N = c_best["test"]["summary"]["NDCG_20"]
    base_R = c0_v1["test"]["summary"]["RECALL_20"]
    delta_vs_v0 = best_R - v0_train_only_R20
    z_dist = np.array(c_best["test"]["z_distribution"])
    non_degenerate = (z_dist.max() < 0.80) and (np.sum(z_dist > 0.01) >= 2)

    if non_degenerate and delta_vs_v0 > 0.001:
        label = "FIXABLE & USEFUL"
        reason = (f"best Part-B config breaks degeneracy "
                  f"(max π = {z_dist.max():.3f}, "
                  f"{int((z_dist>0.01).sum())} situations used > 1%) "
                  f"AND beats V0 train-only R@20 by Δ={delta_vs_v0:+.4f}.")
    elif non_degenerate:
        label = "FORCED BUT USELESS"
        reason = (f"non-degenerate (max π = {z_dist.max():.3f}, "
                  f"{int((z_dist>0.01).sum())} situations used > 1%) "
                  f"BUT accuracy Δ vs V0 train-only R@20 = {delta_vs_v0:+.4f} ≤ 0.001 — "
                  "partition doesn't earn its keep on NYC.")
    else:
        label = "STILL COLLAPSING"
        reason = (f"even the best anti-collapse config keeps "
                  f"max π = {z_dist.max():.3f} > 0.8 — settings too weak.")
    return {"label": label, "reason": reason,
            "delta_R20_vs_V0_traineonly": delta_vs_v0,
            "best_R20": best_R, "best_N20": best_N,
            "V0_R20": v0_train_only_R20, "V0_N20": v0_train_only_N20,
            "z_max": float(z_dist.max()),
            "n_used_situations": int((z_dist > 0.01).sum())}


def main() -> int:
    print(f"=== Part B — anti-collapse sweep on NYC (CPU) ===")
    t0 = time.time()
    ds = build_city_dataset(NYC, verbose=False)
    print(f"  data: n_users={ds.n_users} n_items={ds.n_items} K={K}")

    # ---- V0 train-only baseline (fair comparator) ----
    print(f"\n=== V0 train-only baseline (seed=42) ===")
    v0 = _v0_baseline_train_only(ds, seed=42, verbose=True)
    print(f"  V0 train-only R@20={v0['summary']['RECALL_20']:.4f}  "
          f"N@20={v0['summary']['NDCG_20']:.4f}")

    # ---- Screening (1 seed) ----
    results: list[dict] = []
    SCREEN_SEED = 42

    results.append(run_config("C0_baseline_K6", ds, SCREEN_SEED,
                               init_scale=5e-2, temperature=1.0,
                               balance_weight=0.0, warmup_epochs=0,
                               verbose=True))
    results.append(run_config("C1_balanced_init", ds, SCREEN_SEED,
                               init_scale=1e-3, temperature=1.0,
                               balance_weight=0.0, warmup_epochs=0,
                               verbose=True))
    results.append(run_config("C2_C1_plus_balance0.1", ds, SCREEN_SEED,
                               init_scale=1e-3, temperature=1.0,
                               balance_weight=0.1, warmup_epochs=0,
                               verbose=True))
    results.append(run_config("C3_C2_plus_warm5", ds, SCREEN_SEED,
                               init_scale=1e-3, temperature=1.0,
                               balance_weight=0.1, warmup_epochs=5,
                               verbose=True))
    # λ sweep around 0.1
    results.append(run_config("Cλ_0.01", ds, SCREEN_SEED,
                               init_scale=1e-3, temperature=1.0,
                               balance_weight=0.01, warmup_epochs=0,
                               verbose=True))
    results.append(run_config("Cλ_0.5", ds, SCREEN_SEED,
                               init_scale=1e-3, temperature=1.0,
                               balance_weight=0.5, warmup_epochs=0,
                               verbose=True))
    results.append(run_config("Cλ_1.0", ds, SCREEN_SEED,
                               init_scale=1e-3, temperature=1.0,
                               balance_weight=1.0, warmup_epochs=0,
                               verbose=True))
    # T anneal
    results.append(run_config("CT_T1.5_anneal", ds, SCREEN_SEED,
                               init_scale=1e-3, temperature=1.5,
                               balance_weight=0.1, warmup_epochs=0,
                               anneal_temperature=True, verbose=True))

    # Summary table
    print("\n=== Screening summary (seed=42) ===")
    print(f"{'name':28s}  {'R@20':>7s}  {'N@20':>7s}  {'H(π)':>5s}  "
          f"{'z_max':>5s}  {'#used':>5s}")
    print(f"{'V0 train-only (K=1)':28s}  "
          f"{v0['summary']['RECALL_20']:>7.4f}  {v0['summary']['NDCG_20']:>7.4f}  "
          f"{'--':>5s}  {'--':>5s}  {'1':>5s}")
    for r in results:
        zd = np.array(r["test"]["z_distribution"])
        print(f"{r['name']:28s}  "
              f"{r['test']['summary']['RECALL_20']:>7.4f}  "
              f"{r['test']['summary']['NDCG_20']:>7.4f}  "
              f"{r['test']['pi_entropy_mean']:>5.2f}  "
              f"{zd.max():>5.2f}  "
              f"{int((zd>0.01).sum()):>5d}")

    # Pick best by R@20 among the configs that are non-degenerate (z_max < 0.8)
    candidates = [r for r in results
                  if np.array(r["test"]["z_distribution"]).max() < 0.80
                  and r["name"] != "C0_baseline_K6"]
    if candidates:
        best = max(candidates, key=lambda r: r["test"]["summary"]["RECALL_20"])
    else:
        best = max(results, key=lambda r: r["test"]["summary"]["RECALL_20"])
    print(f"\n>>> Best non-degenerate config: {best['name']}")

    # Optional: 2 extra seeds for the best config (stability)
    extra: list[dict] = []
    for seed in (13, 2024):
        extra.append(run_config(best["name"] + f"_seed{seed}", ds, seed,
                                 init_scale=best["anti"]["init_scale"],
                                 temperature=best["anti"]["temperature"],
                                 balance_weight=best["anti"]["balance_weight"],
                                 warmup_epochs=best["anti"]["warmup_epochs"],
                                 anneal_temperature=best["anti"]["anneal_temperature"],
                                 verbose=False))

    # Classification
    c0 = next(r for r in results if r["name"] == "C0_baseline_K6")
    verdict = classify(c0, best, v0["summary"]["RECALL_20"],
                        v0["summary"]["NDCG_20"])
    print(f"\n=== classification ===")
    print(f"  {verdict['label']}")
    print(f"  {verdict['reason']}")

    report = {
        "elapsed_s": time.time() - t0,
        "hp": HP, "K": K,
        "v0_train_only": v0,
        "screening": [r for r in results],
        "best_config_name": best["name"],
        "extra_seeds": extra,
        "verdict": verdict,
    }
    (OUT / "part_b_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n(report written to {OUT / 'part_b_report.json'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
