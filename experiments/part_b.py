"""Part B — anti-collapse sweep, parameterised by city. Mirrors part_b_nyc.py
but allows ``--city NYC|TKY|both`` and ``--device cpu|mps``.

For each config: z-distribution, π entropy, R@20/NDCG@20 (1 screening seed),
then 2 extra seeds for the best non-degenerate config. Compares against a V0
trained with the **same train-only protocol** (fair internal baseline).
"""
from __future__ import annotations

import argparse
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
from pipeline.step02_models.situational.ranker import rank_split


HP = dict(d=64, lr=5e-3, weight_decay=1e-5, batch_size=1024,
            max_epochs=16, patience=6, eval_every=1)
K = 6


def _v0_baseline_train_only(ds, seed: int, device, verbose: bool = False) -> dict:
    cfg = SituationalConfig(n_users=ds.n_users, n_items=ds.n_items,
                              n_macros=ds.n_macros, n_geo=ds.n_geo,
                              K=1, d=HP["d"], d_g=8, d_e=8, r=4,
                              use_situation=False, use_intent=False)
    torch.manual_seed(seed)
    model = SituationalModel(cfg).to(device)
    t_cfg = TrainConfig(lr=HP["lr"], weight_decay=HP["weight_decay"],
                         batch_size=HP["batch_size"], max_epochs=HP["max_epochs"],
                         patience=HP["patience"], eval_every=1,
                         eval_batch_size=512, seed=seed, verbose=verbose)
    rep = train_situational(model, ds, t_cfg, device=device)
    exclude = (ds.urm_train + ds.urm_val).tocsr(); exclude.data[:] = 1.0
    res = rank_split(model, ds, ds.test, cutoffs=(1, 5, 10, 20, 40, 50, 100),
                      exclude_mask=exclude, device=device, save_z=False,
                      batch_size=512)
    return {"summary": {
                "RECALL_20": float(res.metrics[20]["RECALL"].mean()),
                "NDCG_20": float(res.metrics[20]["NDCG"].mean()),
            },
            "best_val_recall20": rep["best_val_recall20"],
            "best_epoch": rep["best_epoch"],
            "wallclock_s": rep["wallclock_s"]}


def run_config(name: str, ds, seed: int, device,
                init_scale: float, temperature: float, balance_weight: float,
                warmup_epochs: int, anneal_temperature: bool = False,
                verbose: bool = False) -> dict:
    print(f"\n>>> [{name}] seed={seed}  init={init_scale}  T={temperature}  "
          f"λ={balance_weight}  warmup={warmup_epochs}"
          f"{' anneal' if anneal_temperature else ''}")
    cfg = SituationalConfig(n_users=ds.n_users, n_items=ds.n_items,
                              n_macros=ds.n_macros, n_geo=ds.n_geo,
                              K=K, d=HP["d"], d_g=8, d_e=8, r=4,
                              use_situation=True, use_intent=True)
    torch.manual_seed(seed)
    model = SituationalModel(cfg).to(device)
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
    rep = train_with_anti_collapse(model, ds, t_cfg, anti, device=device)
    res = eval_test_with_T(model, ds, temperature=temperature, device=device)
    return {"name": name, "seed": seed, "anti": anti.__dict__,
            "train_report": rep,
            "test": {"summary": res["summary"],
                       "z_distribution": res["z_distribution"],
                       "pi_entropy_mean": res["pi_entropy_mean"],
                       "K_used": int(np.sum(np.array(res["z_distribution"]) > 0))},
            "wallclock_s": time.time() - t0}


def classify(c0_v1, c_best, v0_R20, v0_N20) -> dict:
    best_R = c_best["test"]["summary"]["RECALL_20"]
    best_N = c_best["test"]["summary"]["NDCG_20"]
    delta = best_R - v0_R20
    zd = np.array(c_best["test"]["z_distribution"])
    non_degenerate = (zd.max() < 0.80) and (np.sum(zd > 0.01) >= 2)
    if non_degenerate and delta > 0.001:
        label = "FIXABLE & USEFUL"
        reason = (f"breaks degeneracy (max π = {zd.max():.3f}, "
                  f"{int((zd>0.01).sum())} used > 1%) AND Δ vs V0 train-only "
                  f"R@20 = {delta:+.4f}.")
    elif non_degenerate:
        label = "FORCED BUT USELESS"
        reason = (f"non-degenerate (max π = {zd.max():.3f}, "
                  f"{int((zd>0.01).sum())} used > 1%) BUT Δ vs V0 R@20 = "
                  f"{delta:+.4f} ≤ 0.001 — partition doesn't earn its keep.")
    else:
        label = "STILL COLLAPSING"
        reason = f"even best config: max π = {zd.max():.3f} > 0.8."
    return {"label": label, "reason": reason,
            "delta_R20_vs_V0_traineonly": float(delta),
            "best_R20": float(best_R), "best_N20": float(best_N),
            "V0_R20": float(v0_R20), "V0_N20": float(v0_N20),
            "z_max": float(zd.max()),
            "n_used_situations": int((zd > 0.01).sum())}


def run_part_b_for_city(city: str, device: torch.device,
                          verbose_train: bool = False) -> dict:
    proc = REPO_ROOT / "data" / "processed" / city
    out = REPO_ROOT / "outputs" / city / "situational"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Part B — anti-collapse sweep on {city} (device={device}) ===")
    t0 = time.time()
    ds = build_city_dataset(proc, verbose=False)
    print(f"  data: n_users={ds.n_users} n_items={ds.n_items} K={K}")

    print(f"\n=== V0 train-only baseline (seed=42) ===")
    v0 = _v0_baseline_train_only(ds, seed=42, device=device, verbose=True)
    print(f"  V0 train-only R@20={v0['summary']['RECALL_20']:.4f}  "
          f"N@20={v0['summary']['NDCG_20']:.4f}")

    results: list[dict] = []
    SCREEN_SEED = 42
    configs = [
        ("C0_baseline_K6",      dict(init_scale=5e-2, temperature=1.0, balance_weight=0.0,  warmup_epochs=0)),
        ("C1_balanced_init",    dict(init_scale=1e-3, temperature=1.0, balance_weight=0.0,  warmup_epochs=0)),
        ("C2_C1_plus_balance0.1",dict(init_scale=1e-3, temperature=1.0, balance_weight=0.1, warmup_epochs=0)),
        ("C3_C2_plus_warm5",    dict(init_scale=1e-3, temperature=1.0, balance_weight=0.1, warmup_epochs=5)),
        ("Cλ_0.01",             dict(init_scale=1e-3, temperature=1.0, balance_weight=0.01,warmup_epochs=0)),
        ("Cλ_0.5",              dict(init_scale=1e-3, temperature=1.0, balance_weight=0.5, warmup_epochs=0)),
        ("Cλ_1.0",              dict(init_scale=1e-3, temperature=1.0, balance_weight=1.0, warmup_epochs=0)),
        ("CT_T1.5_anneal",      dict(init_scale=1e-3, temperature=1.5, balance_weight=0.1, warmup_epochs=0, anneal_temperature=True)),
    ]
    for name, params in configs:
        results.append(run_config(name, ds, SCREEN_SEED, device,
                                    verbose=verbose_train, **params))

    print(f"\n=== Screening summary {city} (seed=42) ===")
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

    candidates = [r for r in results
                  if np.array(r["test"]["z_distribution"]).max() < 0.80
                  and r["name"] != "C0_baseline_K6"]
    best = max(candidates if candidates else results,
                 key=lambda r: r["test"]["summary"]["RECALL_20"])
    print(f"\n>>> Best non-degenerate config on {city}: {best['name']}")

    extra: list[dict] = []
    for seed in (13, 2024):
        extra.append(run_config(best["name"] + f"_seed{seed}", ds, seed, device,
                                 verbose=False, **{k: v for k, v in best["anti"].items()
                                                    if k in ("init_scale","temperature",
                                                              "balance_weight","warmup_epochs",
                                                              "anneal_temperature")}))

    c0 = next(r for r in results if r["name"] == "C0_baseline_K6")
    verdict = classify(c0, best, v0["summary"]["RECALL_20"],
                        v0["summary"]["NDCG_20"])
    print(f"\n=== classification {city} ===")
    print(f"  {verdict['label']}")
    print(f"  {verdict['reason']}")

    report = {"elapsed_s": time.time() - t0,
                "city": city, "hp": HP, "K": K,
                "v0_train_only": v0,
                "screening": results,
                "best_config_name": best["name"],
                "extra_seeds": extra,
                "verdict": verdict}
    (out / "part_b_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"(report written to {out / 'part_b_report.json'})")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=["NYC","TKY","both"], default="TKY")
    parser.add_argument("--device", choices=["cpu","mps"], default="cpu")
    parser.add_argument("-v","--verbose", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but unavailable — falling back to CPU.")
        device = torch.device("cpu")

    cities = ["NYC","TKY"] if args.city == "both" else [args.city]
    out = {}
    for c in cities:
        out[c] = run_part_b_for_city(c, device, verbose_train=args.verbose)
    summary_path = REPO_ROOT / "outputs" / "part_b_summary.json"
    summary_path.write_text(json.dumps(
        {c: r["verdict"] for c, r in out.items()},
        indent=2), encoding="utf-8")
    print(f"\nfinal summary → {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
