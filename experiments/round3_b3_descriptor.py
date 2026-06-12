"""Round-3 B3 — descriptor transparency pack.

Read-only diagnostics on the L2 descriptor ``v = [c̃ ‖ e]``:

  * per-dimension std/variance of v on train and test (block-wise)
  * block correlation between c̃ and e (max abs Pearson across pairs)
  * train → test drift of situation distribution P(z) and boundary
    fraction (total variation distance + χ² statistic)

Outputs (per city + intent mode):
  outputs/<city>/xsage[_intent_<mode>]/round3/B3/descriptor_stats.csv
  outputs/<city>/xsage[_intent_<mode>]/round3/B3/ctilde_e_correlation.json
  outputs/<city>/xsage[_intent_<mode>]/round3/B3/drift.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.xsage.orchestrator import _load_city


def analyse(city: str, intent_mode: str = "hard") -> dict:
    sit_dir = REPO_ROOT / "outputs" / city / (
        "xsage" if intent_mode == "hard"
        else f"xsage_intent_{intent_mode}") / "situations"
    if not (sit_dir / "fit.npz").exists():
        print(f"  {city} mode={intent_mode}: Stage A artefacts missing, skipping")
        return {}
    out_dir = sit_dir.parent / "round3" / "B3"
    out_dir.mkdir(parents=True, exist_ok=True)
    fit = np.load(sit_dir / "fit.npz", allow_pickle=True)
    v_train = np.asarray(fit["v_train"]).astype(np.float64)
    v_val = np.asarray(fit["v_val"]).astype(np.float64)
    v_test = np.asarray(fit["v_test"]).astype(np.float64)

    # Block split: first columns are c̃ (DEFAULT_ATTRIBUTES from L1) — read
    # from fit metadata if possible; otherwise infer from W shape.
    W = np.asarray(fit["W"])
    K_macros = W.shape[0]
    D = v_train.shape[1]
    n_attr = D - K_macros           # c̃ block dim

    c_train = v_train[:, :n_attr]; e_train = v_train[:, n_attr:]
    c_test = v_test[:, :n_attr]; e_test = v_test[:, n_attr:]

    # Per-dim std
    stats_rows = []
    for split_name, V in (("train", v_train), ("val", v_val), ("test", v_test)):
        std = V.std(axis=0); mean = V.mean(axis=0)
        for d in range(D):
            block = "c_tilde" if d < n_attr else "e"
            stats_rows.append({"split": split_name, "dim": d, "block": block,
                                "mean": float(mean[d]),
                                "std": float(std[d])})
    pd.DataFrame(stats_rows).to_csv(out_dir / "descriptor_stats.csv", index=False)

    block_summary = {}
    for split_name, V in (("train", v_train), ("test", v_test)):
        block_summary[split_name] = {
            "n_dim_c_tilde": int(n_attr),
            "n_dim_e": int(K_macros),
            "c_tilde_std_min": float(V[:, :n_attr].std(axis=0).min()),
            "c_tilde_std_mean": float(V[:, :n_attr].std(axis=0).mean()),
            "c_tilde_std_max": float(V[:, :n_attr].std(axis=0).max()),
            "c_tilde_near_constant_dims": int((V[:, :n_attr].std(axis=0) < 1e-4).sum()),
            "e_std_min": float(V[:, n_attr:].std(axis=0).min()),
            "e_std_mean": float(V[:, n_attr:].std(axis=0).mean()),
            "e_std_max": float(V[:, n_attr:].std(axis=0).max()),
            "e_near_constant_dims": int((V[:, n_attr:].std(axis=0) < 1e-4).sum()),
        }

    # c̃ × e correlation (max abs Pearson across block pairs)
    c_train_c = c_train - c_train.mean(axis=0)
    e_train_c = e_train - e_train.mean(axis=0)
    # cross-block correlation matrix
    norm_c = np.linalg.norm(c_train_c, axis=0); norm_c[norm_c == 0] = 1.0
    norm_e = np.linalg.norm(e_train_c, axis=0); norm_e[norm_e == 0] = 1.0
    cross = (c_train_c.T @ e_train_c) / (norm_c[:, None] * norm_e[None, :])
    corr_summary = {
        "max_abs_pearson": float(np.abs(cross).max()),
        "mean_abs_pearson": float(np.abs(cross).mean()),
        "n_pairs_above_0p5": int((np.abs(cross) > 0.5).sum()),
        "block_correlation_max_abs_per_c_dim": [float(x) for x in np.abs(cross).max(axis=1)],
        "block_correlation_max_abs_per_e_dim": [float(x) for x in np.abs(cross).max(axis=0)],
    }
    (out_dir / "ctilde_e_correlation.json").write_text(
        json.dumps(corr_summary, indent=2), encoding="utf-8")

    # Train → test drift
    z_train = np.asarray(fit["core_label_train"]).astype(np.int64)
    z_test = np.asarray(fit["core_label_test"]).astype(np.int64)
    isb_train = np.asarray(fit["is_boundary_train"]).astype(bool)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)
    K_sit = int(max(z_train.max(), z_test.max()) + 1)
    p_train = np.bincount(z_train, minlength=K_sit).astype(np.float64) / max(len(z_train), 1)
    p_test = np.bincount(z_test, minlength=K_sit).astype(np.float64) / max(len(z_test), 1)
    tvd = 0.5 * float(np.abs(p_train - p_test).sum())
    expected = p_train * len(z_test)
    observed = np.bincount(z_test, minlength=K_sit).astype(np.float64)
    chi2 = float(np.sum(np.where(expected > 0,
                                    (observed - expected) ** 2 / np.maximum(expected, 1e-9),
                                    0.0)))
    drift = {
        "K_sit": K_sit,
        "P_z_train": p_train.tolist(),
        "P_z_test": p_test.tolist(),
        "total_variation_distance": tvd,
        "chi_square_statistic": chi2,
        "boundary_fraction_train": float(isb_train.mean()),
        "boundary_fraction_test": float(isb_test.mean()),
        "boundary_fraction_delta": float(isb_test.mean() - isb_train.mean()),
    }
    (out_dir / "drift.json").write_text(json.dumps(drift, indent=2),
                                            encoding="utf-8")
    (out_dir / "block_summary.json").write_text(json.dumps(block_summary, indent=2),
                                                       encoding="utf-8")

    print(f"  {city} mode={intent_mode}: TVD(P_z train→test)={tvd:.4f}  "
          f"|c̃×e|max={corr_summary['max_abs_pearson']:.3f}  "
          f"c̃ near-constant dims (train)={block_summary['train']['c_tilde_near_constant_dims']}")
    return {"block_summary": block_summary, "correlation": corr_summary, "drift": drift}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", choices=["NYC", "TKY", "both"], default="both")
    parser.add_argument("--intent-mode", choices=["hard", "all", "both"], default="both")
    args = parser.parse_args()
    cities = ["NYC", "TKY"] if args.city == "both" else [args.city]
    modes = ["hard", "all"] if args.intent_mode == "both" else [args.intent_mode]
    for city in cities:
        print(f"\n>>> B3 on {city}")
        for mode in modes:
            analyse(city, intent_mode=mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
