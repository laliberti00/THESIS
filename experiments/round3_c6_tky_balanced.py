"""Round-3 C6 — TKY* balanced counterfactual.

Construction (per the brief and PREDICTION_C6.md):
  1. From raw TKY, per-user uniform downsample Travel & Transport check-ins
     until global T&T share ≈ 25 % (NYC's level).
  2. Save reduced raw as data/raw/dataset_TSMC2014_TKY_BAL.txt.
  3. Run step01 from the reduced raw (all sequence features recomputed on
     the reduced sequences — never recycled).
  4. Stage A (keep-mode, default hard) → Stage C → B_blind + tuned B_full
     → C5.0-style stratified read → Stage-B lens (curiosity read).

Predictions (committed in PREDICTION_C6.md before this run):
  P-i:   ≥ 4 attractors under keep-mode.
  P-ii:  Stage-C ΔF1 vs clock ≥ 0.
  P-iii: B_full − B_blind aggregate ΔR@20 ≥ 0.
  P-iv:  per-macro signs invariant.

Outcomes recorded next to predictions after the runs.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


TARGET_TT_SHARE = 0.25
MIN_USERS_AFTER_KCORE = 800
TT_MACRO = "Travel & Transport"


def step1_downsample_tt(raw_in: Path, raw_out: Path,
                            taxonomy_path: Path, seed: int = 42) -> dict:
    """Per-user uniform-random downsample of T&T check-ins until global
    T&T share ≈ TARGET_TT_SHARE.
    """
    print(f">>> [C6.1] downsampling T&T in {raw_in.name} (target share={TARGET_TT_SHARE:.0%})")
    # Same parsing as step01
    TSV_COLS = ["user_id", "venue_id", "cat_id", "cat_name",
                 "lat", "lon", "tz_offset", "utc_time"]
    df = pd.read_csv(raw_in, sep="\t", header=None, names=TSV_COLS,
                       encoding="latin-1")
    n_raw = len(df)
    from pipeline.step01_preprocessing.taxonomy import build_cat_id_to_macro
    cat_to_macro = build_cat_id_to_macro(taxonomy_path)
    df["cat_macro"] = df["cat_id"].map(cat_to_macro).fillna("Other")
    n_tt_raw = int((df["cat_macro"] == TT_MACRO).sum())
    share_raw = n_tt_raw / n_raw
    print(f"  raw: {n_raw:,} check-ins, T&T share = {share_raw:.3f}")

    # Target: keep R fraction of T&T such that global share ≈ TARGET_TT_SHARE.
    # Let kept T&T = R * n_tt; new total = R * n_tt + (n - n_tt).
    # new share = R*n_tt / (R*n_tt + n - n_tt) = TARGET → solve for R:
    # R = TARGET * (n - n_tt) / (n_tt * (1 - TARGET))
    R = (TARGET_TT_SHARE * (n_raw - n_tt_raw)
          / (n_tt_raw * (1 - TARGET_TT_SHARE)))
    print(f"  target T&T retention fraction R = {R:.4f}")
    R = max(min(R, 1.0), 0.0)

    # Per-user uniform downsample, preserving chronological order
    rng = np.random.default_rng(seed)
    keep = np.ones(len(df), dtype=bool)
    tt_indices = df.index[df["cat_macro"] == TT_MACRO].tolist()
    n_tt_keep_total = int(round(R * n_tt_raw))
    # uniformly per-user
    user_tt_counts = df.loc[tt_indices, "user_id"].value_counts().to_dict()
    drops = set()
    for u, ct in user_tt_counts.items():
        ct_keep = int(round(R * ct))
        u_tt = df.index[(df["user_id"] == u) & (df["cat_macro"] == TT_MACRO)].tolist()
        if ct_keep < len(u_tt):
            to_drop = rng.choice(u_tt, size=len(u_tt) - ct_keep,
                                       replace=False)
            drops.update(to_drop.tolist())
    keep[list(drops)] = False
    df_out = df.loc[keep].sort_values(["user_id", "utc_time"]).reset_index(drop=True)
    n_kept = len(df_out)
    n_tt_kept = int((df_out["cat_macro"] == TT_MACRO).sum())
    new_share = n_tt_kept / n_kept
    print(f"  kept: {n_kept:,} check-ins, T&T share = {new_share:.3f}")

    # Write reduced TSV (drop the cat_macro column we added)
    df_out = df_out.drop(columns=["cat_macro"])
    df_out.to_csv(raw_out, sep="\t", header=False, index=False, encoding="latin-1")
    print(f"  wrote {raw_out}")
    return {"n_raw": int(n_raw), "n_kept": int(n_kept),
              "n_tt_raw": int(n_tt_raw), "n_tt_kept": int(n_tt_kept),
              "share_raw": float(share_raw), "share_after": float(new_share),
              "R": float(R)}


def step2_preprocess(raw_path: Path, out_dir: Path,
                          taxonomy_path: Path) -> dict:
    print(f"\n>>> [C6.2] step01 preprocessing → {out_dir}")
    from pipeline.step01_preprocessing import preprocess_tsmc2014
    result = preprocess_tsmc2014(
        raw_tsv=raw_path, out_dir=out_dir,
        taxonomy_path=taxonomy_path, k_core=10,
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    print(result.short_report())
    return {"n_users": result.n_users, "n_items": result.n_items,
              "n_interactions_post_kcore": result.n_interactions_post_kcore,
              "n_interactions_final": result.n_interactions_final}


def step3_stage_a_c(city: str) -> dict:
    print(f"\n>>> [C6.3] Stage A + C on {city} (keep-mode, default hard intent)")
    # Use the existing CLI
    env = os.environ.copy()
    out = {}
    for stage in ("A", "C"):
        cmd = [".venv/bin/python", "-m", "experiments.run_xsage",
                 "--city", city, "--stage", stage, "-v"]
        print(f"  running {' '.join(cmd[-3:])}")
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                                       capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout[-2000:]); print(result.stderr[-1000:])
            raise RuntimeError(f"Stage {stage} failed")
        for line in result.stdout.split("\n")[-15:]:
            print(f"    {line}")
        if stage == "A":
            # Parse from summary.json
            sa = json.loads((REPO_ROOT / "outputs" / city / "xsage"
                                  / "situations" / "summary.json").read_text())
            out["stage_a_K"] = int(sa["K"])
            out["stage_a_eps"] = float(sa["eps"])
            out["stage_a_ARI"] = float(sa["ari_seeds"])
            out["stage_a_attractors"] = sa["attractors"]
            out["stage_a_n_attractors"] = len(sa["attractors"])
        elif stage == "C":
            sc = json.loads((REPO_ROOT / "outputs" / city / "xsage"
                                  / "projection" / "verdict.json").read_text())
            out["stage_c_macro_F1_T"] = float(sc["macro_F1_transition"])
            out["stage_c_macro_F1_time"] = float(sc["macro_F1_time_only"])
            out["stage_c_delta_F1"] = float(sc["delta_F1"])
            out["stage_c_mcnemar_p"] = sc["mcnemar_p_value"]
    return out


def step4_backbones_and_stratified(city: str) -> dict:
    print(f"\n>>> [C6.4] B_blind floor recipe + B_full tuned + stratified read")
    # Floor of one model (FM) is needed to provide best_hp.json
    # Use experiments.run_baselines with --models FM for B_blind
    env = os.environ.copy()
    # 1. Floor FM for B_blind HP
    cmd = [".venv/bin/python", "-m", "experiments.run_baselines",
             "--city", city, "--models", "FM", "-v"]
    print(f"  Floor FM (B_blind HP search) ...")
    r = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-1500:])
        raise RuntimeError("Floor FM failed")
    print(f"  Floor FM done.")

    # 2. B_full tune (round-2 1.1)
    cmd = [".venv/bin/python", "-m", "experiments.round2_tune_bfull",
             "--city", city, "--max-epochs", "12", "-v"]
    print(f"  B_full tuning ...")
    r = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-1500:])
        raise RuntimeError("B_full tuning failed")
    print(f"  B_full tuning done.")

    # 3. Stratified read using cached scores
    from pipeline.step02_models.xsage.backbone import excluded_mask, load_or_refit
    from pipeline.step02_models.xsage.orchestrator import _load_city
    ds = _load_city(city)
    df_test = ds["df_test"]
    u_test = df_test["u_idx"].values.astype(np.int64)
    i_target = df_test["i_idx"].values.astype(np.int64)
    cat_target = df_test["cat_macro"].values.astype(str)
    scores_blind = load_or_refit(city, model_name="FM")[u_test]
    bfull_path = REPO_ROOT / "outputs" / city / "xsage" / "backbone" / "Bfull.scores.npy"
    scores_full = np.load(bfull_path)
    excl = excluded_mask(city, ds["n_items"])
    K = 20
    r20_b = np.zeros(len(u_test)); n20_b = np.zeros(len(u_test))
    r20_f = np.zeros(len(u_test)); n20_f = np.zeros(len(u_test))
    for b in range(len(u_test)):
        u = int(u_test[b]); tgt = int(i_target[b])
        for sc, ra, na in ((scores_blind, r20_b, n20_b),
                             (scores_full, r20_f, n20_f)):
            s_row = sc[b].copy()
            cols = excl.indices[excl.indptr[u]:excl.indptr[u + 1]]
            if len(cols): s_row[cols] = -np.inf
            ts = s_row[tgt]
            rank = int((s_row > ts).sum()) + 1
            if rank <= K:
                ra[b] = 1.0
                na[b] = 1.0 / np.log2(rank + 1)
    tt_mask = cat_target == TT_MACRO
    out = {
        "B_blind_R20_all": float(r20_b.mean()), "B_blind_N20_all": float(n20_b.mean()),
        "B_full_R20_all": float(r20_f.mean()),  "B_full_N20_all": float(n20_f.mean()),
        "B_blind_R20_TT": float(r20_b[tt_mask].mean()),
        "B_full_R20_TT":  float(r20_f[tt_mask].mean()),
        "B_blind_R20_nonTT": float(r20_b[~tt_mask].mean()),
        "B_full_R20_nonTT":  float(r20_f[~tt_mask].mean()),
        "delta_R20_all":   float(r20_f.mean() - r20_b.mean()),
        "delta_R20_TT":    float(r20_f[tt_mask].mean() - r20_b[tt_mask].mean()),
        "delta_R20_nonTT": float(r20_f[~tt_mask].mean() - r20_b[~tt_mask].mean()),
        "test_TT_share": float(tt_mask.mean()),
    }
    print(f"  B_blind R@20 = {out['B_blind_R20_all']:.4f}  "
          f"B_full R@20 = {out['B_full_R20_all']:.4f}  "
          f"Δ = {out['delta_R20_all']:+.4f}")
    print(f"    T&T  : Δ = {out['delta_R20_TT']:+.4f}")
    print(f"    nonTT: Δ = {out['delta_R20_nonTT']:+.4f}")
    return out


def step5_lens(city: str) -> dict:
    """Stage-B lens curiosity read."""
    print(f"\n>>> [C6.5] Stage-B lens (curiosity read)")
    cmd = [".venv/bin/python", "-m", "experiments.run_xsage",
             "--city", city, "--stage", "B", "-v"]
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-1000:])
        raise RuntimeError("Stage B failed")
    for line in r.stdout.split("\n")[-12:]:
        print(f"    {line}")
    v = json.loads((REPO_ROOT / "outputs" / city / "xsage"
                          / "fairness" / "verdict.json").read_text())
    return {"global_LT": v["global_LT"], "global_KL_mean": v["global_KL_mean_all_split"],
              "verdict": v["verdict"], "sinks": [s["situation"] for s in v["inequity_sinks"]]}


def main() -> int:
    out_dir = REPO_ROOT / "outputs" / "round3" / "C6"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_in = REPO_ROOT / "data" / "raw" / "dataset_TSMC2014_TKY.txt"
    raw_out = REPO_ROOT / "data" / "raw" / "dataset_TSMC2014_TKY_BAL.txt"
    taxonomy = REPO_ROOT / "config" / "foursquare_legacy_taxonomy.json"

    t0 = time.time()
    s1 = step1_downsample_tt(raw_in, raw_out, taxonomy, seed=42)
    proc_dir = REPO_ROOT / "data" / "processed" / "TKY_BAL"
    s2 = step2_preprocess(raw_out, proc_dir, taxonomy)
    if s2["n_users"] < MIN_USERS_AFTER_KCORE:
        print(f"  GUARD: only {s2['n_users']} users after k-core (< {MIN_USERS_AFTER_KCORE})")
        # Note in summary; do not abort — still run, but flag

    s3 = step3_stage_a_c("TKY_BAL")
    s4 = step4_backbones_and_stratified("TKY_BAL")
    s5 = step5_lens("TKY_BAL")

    # Compare predictions to outcomes
    P_i = s3["stage_a_n_attractors"] >= 4
    P_ii = s3["stage_c_delta_F1"] >= 0
    P_iii = s4["delta_R20_all"] >= 0
    P_iv = (s4["delta_R20_TT"] < 0) and (s4["delta_R20_nonTT"] > 0)

    summary = {
        "downsample": s1, "preprocess": s2,
        "stage_a_c": s3, "backbones_stratified": s4,
        "lens": s5,
        "predictions": {
            "P_i_attractors_ge_4": {"prediction": True, "outcome": int(s3["stage_a_n_attractors"]),
                                          "match": bool(P_i)},
            "P_ii_delta_F1_ge_0": {"prediction": True, "outcome": float(s3["stage_c_delta_F1"]),
                                          "match": bool(P_ii)},
            "P_iii_aggregate_positive": {"prediction": True,
                                                "outcome": float(s4["delta_R20_all"]),
                                                "match": bool(P_iii)},
            "P_iv_per_macro_invariant": {"prediction": True,
                                                "outcome": {"T&T_delta": s4["delta_R20_TT"],
                                                                "nonTT_delta": s4["delta_R20_nonTT"]},
                                                "match": bool(P_iv)},
        },
        "wallclock_s": time.time() - t0,
    }
    (out_dir / "OUTCOMES.json").write_text(json.dumps(summary, indent=2, default=str),
                                                encoding="utf-8")
    print("\n=== TKY* OUTCOMES ===")
    print(f"P-i  (≥4 attractors) : prediction TRUE  outcome {s3['stage_a_n_attractors']}    "
          f"→ {'MATCH' if P_i else 'MISS'}")
    print(f"P-ii (ΔF1 ≥ 0)       : prediction TRUE  outcome {s3['stage_c_delta_F1']:+.3f}  "
          f"→ {'MATCH' if P_ii else 'MISS'}")
    print(f"P-iii (aggregate ≥0) : prediction TRUE  outcome {s4['delta_R20_all']:+.4f} "
          f"→ {'MATCH' if P_iii else 'MISS'}")
    print(f"P-iv  (per-macro inv): prediction TRUE  outcome T&T {s4['delta_R20_TT']:+.4f} / "
          f"nonTT {s4['delta_R20_nonTT']:+.4f}  → {'MATCH' if P_iv else 'MISS'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
