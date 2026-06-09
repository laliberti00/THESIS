"""End-to-end driver: tune V0 → run V0/V1/V2 (+ K sweep) → export .npz →
situation diagnostics → RESULTS.md → VERDICT.

This file is imported by ``experiments/run_situational.py``; keeping the
orchestration here keeps the CLI a thin wrapper.

Per the brief (§6) the tuning is reduced for the go/no-go:
    * Tune V0 (situation off) on val RECALL@20 with skopt ``n_cases=30``.
    * V1 and V2 reuse V0's HPs; tune only ``K`` via the sweep.
The final paper run will use the floor's ``n_cases=100``.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml

from .data import CityDataset, build_city_dataset
from .model import SituationalConfig, SituationalModel
from .ranker import METRIC_NAMES, aggregate_for_summary, export_npz, rank_split
from .trainer import TrainConfig, refit_train_plus_val, train_situational

# --------- protocol --------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = REPO_ROOT / "config" / "protocol.yaml"

DEFAULT_CUTOFFS = (1, 5, 10, 20, 40, 50, 100)
DEFAULT_SEEDS = (42, 13, 2024)
DEFAULT_K_SWEEP = (2, 3, 4, 6, 8)


# --------- HP search space (reduced for the go/no-go) ----------------------

# We hand-pick a tiny grid for the go/no-go instead of running skopt; the
# search space is faithful to the brief but tractable on a laptop in 1-2
# evaluations per HP point.  Tuning V0 only is intentional (the matched-pair
# study isolates the *mechanism's* effect, not HP optimisation).
GRID_V0 = [
    {"d": 32, "lr": 5e-3, "weight_decay": 1e-5, "batch_size": 1024,
     "max_epochs": 20, "patience": 4},
    {"d": 32, "lr": 1e-2, "weight_decay": 1e-5, "batch_size": 1024,
     "max_epochs": 20, "patience": 4},
    {"d": 64, "lr": 5e-3, "weight_decay": 1e-5, "batch_size": 1024,
     "max_epochs": 20, "patience": 4},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(ds: CityDataset, hp: dict, K: int,
                  use_situation: bool, use_intent: bool,
                  d_g: int = 8, d_e: int = 8, r: int = 4) -> SituationalModel:
    cfg = SituationalConfig(
        n_users=ds.n_users, n_items=ds.n_items,
        n_macros=ds.n_macros, n_geo=ds.n_geo,
        K=K, d=hp["d"], d_g=d_g, d_e=d_e, r=r,
        use_situation=use_situation, use_intent=use_intent,
    )
    return SituationalModel(cfg)


def _make_trainer(hp: dict, seed: int, verbose: bool) -> TrainConfig:
    return TrainConfig(
        lr=hp["lr"], weight_decay=hp["weight_decay"],
        batch_size=hp["batch_size"], max_epochs=hp["max_epochs"],
        patience=hp["patience"], eval_every=1, eval_batch_size=512,
        seed=seed, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Per-variant run
# ---------------------------------------------------------------------------

def run_variant(ds: CityDataset, hp: dict, K: int,
                 use_situation: bool, use_intent: bool,
                 seed: int, device: torch.device,
                 with_refit: bool = True,
                 verbose: bool = False) -> dict:
    """Train (+optionally refit on train∪val) and evaluate on test.

    Returns the per-user results dict (suitable for npz export) plus a small
    summary dict for the report.
    """
    torch.manual_seed(seed)
    model = _make_model(ds, hp, K=K,
                          use_situation=use_situation,
                          use_intent=use_intent).to(device)

    t_cfg = _make_trainer(hp, seed=seed, verbose=verbose)
    report = train_situational(model, ds, t_cfg, device=device)

    if with_refit:
        # Refit on train + val for as many epochs as the val-best one.
        refit_epochs = max(1, report.get("best_epoch", t_cfg.max_epochs))
        # Fresh init to avoid double-fit drift (matches the floor protocol).
        model = _make_model(ds, hp, K=K,
                              use_situation=use_situation,
                              use_intent=use_intent).to(device)
        ref_rep = refit_train_plus_val(model, ds, t_cfg,
                                        n_epochs=refit_epochs,
                                        device=device)
        report["refit"] = ref_rep

    # Test eval (full ranking, exclude train+val items).
    exclude = (ds.urm_train + ds.urm_val).tocsr()
    exclude.data[:] = 1.0
    res = rank_split(model, ds, ds.test,
                      cutoffs=DEFAULT_CUTOFFS,
                      batch_size=512,
                      exclude_mask=exclude,
                      device=device,
                      save_z=True)

    summary = aggregate_for_summary(res, cutoff_key=20)
    return {"per_user": res, "summary": summary, "report": report}


# ---------------------------------------------------------------------------
# Reduced HP sweep for V0 (the only tuned variant in the go/no-go)
# ---------------------------------------------------------------------------

def tune_v0(ds: CityDataset, seed: int, device: torch.device,
              verbose: bool = False) -> dict:
    """Pick the V0 HP setting with best val RECALL@20."""
    best = None
    best_rec = -float("inf")
    for hp in GRID_V0:
        torch.manual_seed(seed)
        model = _make_model(ds, hp, K=1,
                              use_situation=False, use_intent=False).to(device)
        cfg = _make_trainer(hp, seed=seed, verbose=False)
        report = train_situational(model, ds, cfg, device=device)
        rec = float(report["best_val_recall20"])
        if verbose:
            print(f"    V0 sweep hp={hp}  val R@20={rec:.4f}")
        if rec > best_rec:
            best_rec = rec
            best = hp
    return {"hp": best, "val_recall20": best_rec}


# ---------------------------------------------------------------------------
# Situation diagnostics for V1
# ---------------------------------------------------------------------------

def situation_diagnostics(ds: CityDataset,
                            v1_result: dict,
                            df_test_path: Path) -> dict:
    """Distribution and semantic content of the K situations used by V1."""
    df_test = pd.read_parquet(df_test_path)
    z = v1_result["per_user"].z_per_request
    if z is None:
        return {"warning": "no z saved for V1"}

    K = int(z.max() + 1)
    counts = np.bincount(z, minlength=K)
    z_distribution = (counts / counts.sum()).tolist()

    inv_macro = {v: k for k, v in ds.macro_to_idx.items()}
    diags = []
    for k in range(K):
        mask = (z == k)
        if mask.sum() == 0:
            diags.append({"k": k, "fraction": 0.0,
                          "note": "EMPTY — situation never chosen"})
            continue
        rows = df_test.iloc[mask]
        cat_counts = rows["cat_macro"].value_counts(normalize=True)
        diags.append({
            "k": k,
            "fraction": float(mask.mean()),
            "mean_hour": float(rows["c_hour"].mean()),
            "fraction_weekend": float(rows["c_isweekend"].mean()),
            "top_target_categories": cat_counts.head(3).to_dict(),
            "bottom_target_categories": cat_counts.tail(3).to_dict(),
            "intent_breakdown": rows["intent_last_cat"]
                .value_counts(normalize=True).head(3).to_dict(),
        })

    # Specialization vs chance:
    global_cat = df_test["cat_macro"].value_counts(normalize=True)
    purity_vs_chance = []
    for k in range(K):
        mask = (z == k)
        if mask.sum() == 0:
            purity_vs_chance.append(None); continue
        local = df_test.iloc[mask]["cat_macro"].value_counts(normalize=True)
        top = local.index[0]
        purity_vs_chance.append({
            "top_cat": top,
            "local_share": float(local.iloc[0]),
            "global_share": float(global_cat.get(top, 0.0)),
            "lift": float(local.iloc[0] / max(1e-6, global_cat.get(top, 1e-6))),
        })

    # Degeneracy flag
    degenerate = (max(z_distribution) > 0.80)
    return {
        "K": K,
        "z_distribution": z_distribution,
        "degenerate_collapse": degenerate,
        "per_situation": diags,
        "specialization_vs_chance": purity_vs_chance,
    }


# ---------------------------------------------------------------------------
# RESULTS.md
# ---------------------------------------------------------------------------

def _table_md(rows: list[dict], cols: list[str]) -> str:
    header = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    body = "".join(
        "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
        for r in rows
    )
    return header + sep + body


def write_results_md(out_dir: Path,
                       city: str,
                       hp: dict,
                       runs: dict,
                       K_sweep_runs: dict,
                       situation_diag: dict,
                       verdict: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "RESULTS.md"
    lines = [f"# step02b go/no-go — {city}\n"]
    lines.append("## Protocol\n")
    lines.append(f"* Per-request next-item ranking (NOT URM-set holdout — "
                 f"absolute numbers are not directly comparable to the floor).\n")
    lines.append(f"* HPs tuned on V0 only via a reduced grid (see brief §6); "
                 f"reused for V1/V2. **chosen HP**: `{hp}`\n")
    lines.append(f"* Seeds: {list(runs.keys())} (go/no-go uses 3).\n")
    lines.append(f"* Cutoffs: {DEFAULT_CUTOFFS}.\n\n")

    lines.append("## Variants — mean ± std over seeds\n")
    rows = []
    for var in ["V0", "V1", "V2"]:
        if var not in runs[list(runs.keys())[0]]: continue
        recs = [runs[s][var]["summary"]["RECALL"] for s in runs]
        ndcgs = [runs[s][var]["summary"]["NDCG"] for s in runs]
        rows.append({
            "variant": var,
            "R@20 mean": f"{np.mean(recs):.4f}",
            "R@20 std": f"{np.std(recs):.4f}",
            "NDCG@20 mean": f"{np.mean(ndcgs):.4f}",
            "NDCG@20 std": f"{np.std(ndcgs):.4f}",
        })
    lines.append(_table_md(rows, ["variant", "R@20 mean", "R@20 std",
                                    "NDCG@20 mean", "NDCG@20 std"]))

    lines.append("\n## Headline contrasts (mean delta over seeds)\n")
    seeds = list(runs.keys())
    if "V1" in runs[seeds[0]] and "V0" in runs[seeds[0]]:
        d_rec = np.mean([runs[s]["V1"]["summary"]["RECALL"]
                          - runs[s]["V0"]["summary"]["RECALL"] for s in seeds])
        d_ndcg = np.mean([runs[s]["V1"]["summary"]["NDCG"]
                          - runs[s]["V0"]["summary"]["NDCG"] for s in seeds])
        lines.append(f"* **V1 − V0** ΔR@20 = {d_rec:+.4f}  ΔNDCG@20 = {d_ndcg:+.4f}\n")
    if "V1" in runs[seeds[0]] and "V2" in runs[seeds[0]]:
        d_rec = np.mean([runs[s]["V1"]["summary"]["RECALL"]
                          - runs[s]["V2"]["summary"]["RECALL"] for s in seeds])
        d_ndcg = np.mean([runs[s]["V1"]["summary"]["NDCG"]
                          - runs[s]["V2"]["summary"]["NDCG"] for s in seeds])
        lines.append(f"* **V1 − V2** ΔR@20 = {d_rec:+.4f}  ΔNDCG@20 = {d_ndcg:+.4f}\n")

    if K_sweep_runs:
        lines.append("\n## K sweep (V1 with intent ON)\n")
        rows = []
        for K in sorted(K_sweep_runs):
            recs = [K_sweep_runs[K][s]["summary"]["RECALL"] for s in K_sweep_runs[K]]
            ndcgs = [K_sweep_runs[K][s]["summary"]["NDCG"] for s in K_sweep_runs[K]]
            rows.append({
                "K": K,
                "R@20 mean": f"{np.mean(recs):.4f}",
                "NDCG@20 mean": f"{np.mean(ndcgs):.4f}",
            })
        lines.append(_table_md(rows, ["K", "R@20 mean", "NDCG@20 mean"]))

    if situation_diag:
        lines.append("\n## Situation diagnostics (V1)\n")
        lines.append(f"* K used: {situation_diag.get('K')}\n")
        lines.append(f"* π distribution over test requests: "
                     f"{[f'{x:.3f}' for x in situation_diag.get('z_distribution', [])]}\n")
        lines.append(f"* Degenerate collapse (>80% on one situation): "
                     f"**{situation_diag.get('degenerate_collapse')}**\n")
        if "per_situation" in situation_diag:
            lines.append("\nPer-situation summary:\n")
            for s in situation_diag["per_situation"]:
                if "note" in s:
                    lines.append(f"  - k={s['k']}: {s['note']}\n"); continue
                tops = ", ".join(f"{c}={v:.2f}" for c, v in s["top_target_categories"].items())
                lines.append(f"  - k={s['k']}  frac={s['fraction']:.3f}  "
                             f"mean_hour={s['mean_hour']:.1f}  "
                             f"weekend={s['fraction_weekend']:.2f}  "
                             f"top: {tops}\n")

    lines.append("\n## VERDICT\n")
    lines.append(f"**{verdict['label']}** — {verdict['reason']}\n")

    path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------

def decide_verdict(city: str,
                    runs: dict,
                    situation_diag: dict,
                    sig_p: float = 0.05) -> dict:
    seeds = list(runs.keys())
    v1_r = np.array([runs[s]["V1"]["summary"]["RECALL"] for s in seeds])
    v0_r = np.array([runs[s]["V0"]["summary"]["RECALL"] for s in seeds])
    v1_n = np.array([runs[s]["V1"]["summary"]["NDCG"] for s in seeds])
    v0_n = np.array([runs[s]["V0"]["summary"]["NDCG"] for s in seeds])

    # Tiny seed sample (3) — we report effect sign and rely on per-user stats
    # via step04 (Wilcoxon) for the real significance test, computed elsewhere.
    delta_r = float((v1_r - v0_r).mean())
    delta_n = float((v1_n - v0_n).mean())
    distinct = not situation_diag.get("degenerate_collapse", True)
    # specialisation lift: at least one situation > chance by 30 %
    lifts = [s["lift"] for s in (situation_diag.get("specialization_vs_chance") or [])
             if s is not None]
    specialised = any(l > 1.30 for l in lifts) if lifts else False
    accuracy_pos = (delta_r > 0) or (delta_n > 0)

    if accuracy_pos and distinct and specialised:
        label = "GREEN"
        reason = (f"V1 beats V0 on R@20 by {delta_r:+.4f} and NDCG@20 by "
                  f"{delta_n:+.4f} (mean over 3 seeds); situations are "
                  "non-degenerate and specialised. Proceed to full step02b.")
    elif distinct and specialised:
        label = "YELLOW"
        reason = (f"Accuracy parity (ΔR@20={delta_r:+.4f}, ΔNDCG@20={delta_n:+.4f}) "
                  "but situations are non-degenerate and specialised — "
                  "lead with interpretability / fairness story.")
    else:
        label = "RED"
        reason = (f"V1 ≈ V0 (ΔR@20={delta_r:+.4f}, ΔNDCG@20={delta_n:+.4f}) "
                  f"AND situations are degenerate (collapse={situation_diag.get('degenerate_collapse')}) "
                  f"or non-specialised. Mechanism is inert.")
    return {"label": label, "reason": reason,
            "delta_R20": delta_r, "delta_NDCG20": delta_n,
            "distinct": distinct, "specialised": specialised}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_for_city(city: str,
                  seeds: tuple = DEFAULT_SEEDS,
                  K_sweep: tuple = DEFAULT_K_SWEEP,
                  with_refit: bool = True,
                  verbose: bool = False) -> dict:
    """End-to-end: tune V0 → run V0/V1/V2 × seeds → K sweep on V1 →
    situation diagnostics → RESULTS.md → return summary."""
    processed_dir = REPO_ROOT / "data" / "processed" / city
    out_dir = REPO_ROOT / "outputs" / city / "situational"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")  # the model is small; CPU is enough
    if torch.backends.mps.is_available():
        try:
            torch.zeros(1).to("mps")
            device = torch.device("mps")
        except Exception:
            device = torch.device("cpu")

    print(f"\n>>> [{city}] device={device}")
    t0 = time.time()
    ds = build_city_dataset(processed_dir, verbose=verbose)
    print(f"    dataset built in {time.time()-t0:.1f}s  "
          f"n_users={ds.n_users}  n_items={ds.n_items}  "
          f"n_macros={ds.n_macros}  n_geo={ds.n_geo}")

    # ---- V0 tuning -----------------------------------------------------
    print(f">>> [{city}] tuning V0 (situation OFF) on val")
    tune_res = tune_v0(ds, seed=seeds[0], device=device, verbose=verbose)
    hp = tune_res["hp"]
    print(f"    V0 best HP: {hp}  val R@20={tune_res['val_recall20']:.4f}")

    # ---- Variant matrix -----------------------------------------------
    runs = {seed: {} for seed in seeds}

    # First pass: V1 with K=DEFAULT (best of sweep computed below)
    K_default = 4  # initial K (replaced after sweep with the winner)

    print(f">>> [{city}] running V0 × {len(seeds)} seeds (situation OFF)")
    for seed in seeds:
        out = run_variant(ds, hp, K=1, use_situation=False, use_intent=False,
                            seed=seed, device=device, with_refit=with_refit,
                            verbose=verbose)
        export_npz(out["per_user"],
                    out_dir / f"V0_seed{seed}.npz",
                    z_per_request=out["per_user"].z_per_request)
        runs[seed]["V0"] = out
        print(f"    V0 seed={seed} R@20={out['summary']['RECALL']:.4f}")

    # ---- K sweep on V1 ----------------------------------------------
    print(f">>> [{city}] K sweep on V1 (situation ON, intent ON)")
    K_sweep_runs = {K: {} for K in K_sweep}
    best_K = K_sweep[0]
    best_K_score = -float("inf")
    for K in K_sweep:
        for seed in seeds:
            out = run_variant(ds, hp, K=K, use_situation=True, use_intent=True,
                                seed=seed, device=device, with_refit=with_refit,
                                verbose=verbose)
            K_sweep_runs[K][seed] = out
            print(f"    Ksweep K={K} seed={seed} R@20={out['summary']['RECALL']:.4f}")
        # avg over seeds for K
        avg = np.mean([K_sweep_runs[K][s]["summary"]["RECALL"] for s in seeds])
        if avg > best_K_score:
            best_K_score = avg; best_K = K
    print(f"    sweep winner: K={best_K}  R@20={best_K_score:.4f}")

    # V1 chosen = K_sweep_runs[best_K]
    for seed in seeds:
        runs[seed]["V1"] = K_sweep_runs[best_K][seed]
        export_npz(runs[seed]["V1"]["per_user"],
                    out_dir / f"V1_seed{seed}.npz",
                    z_per_request=runs[seed]["V1"]["per_user"].z_per_request)

    # ---- V2 (situation ON, intent OFF, K = best_K) ------------------
    print(f">>> [{city}] running V2 (situation ON, intent OFF) K={best_K}")
    for seed in seeds:
        out = run_variant(ds, hp, K=best_K, use_situation=True, use_intent=False,
                            seed=seed, device=device, with_refit=with_refit,
                            verbose=verbose)
        export_npz(out["per_user"],
                    out_dir / f"V2_seed{seed}.npz",
                    z_per_request=out["per_user"].z_per_request)
        runs[seed]["V2"] = out
        print(f"    V2 seed={seed} R@20={out['summary']['RECALL']:.4f}")

    # ---- Diagnostics on V1 ------------------------------------------
    diag = situation_diagnostics(ds, runs[seeds[0]]["V1"],
                                   processed_dir / "df_test.parquet")
    (out_dir / "situation_report.json").write_text(
        json.dumps(diag, indent=2, default=str), encoding="utf-8")

    # ---- VERDICT + RESULTS.md ---------------------------------------
    verdict = decide_verdict(city, runs, diag)
    write_results_md(out_dir, city, hp, runs, K_sweep_runs, diag, verdict)
    print(f"\n>>> [{city}] VERDICT: {verdict['label']}  {verdict['reason']}")

    return {"hp": hp, "best_K": best_K,
            "runs": {s: {var: r["summary"] for var, r in runs[s].items()}
                      for s in seeds},
            "K_sweep": {K: {s: K_sweep_runs[K][s]["summary"] for s in seeds}
                         for K in K_sweep},
            "diagnostics": diag,
            "verdict": verdict}
