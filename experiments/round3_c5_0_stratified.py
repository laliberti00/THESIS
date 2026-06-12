"""Round-3 C5.0 — stratified accuracy read: B_blind vs B_full by target macro.

Hypothesis (pre-registered): with 73 % T&T mass on TKY, the per-context
features make B_full over-confident in the T&T prior, crowding out
discrimination on non-T&T targets. Prediction: B_full's deficit vs
B_blind on TKY concentrates on non-T&T-target requests.

This script joins per-request top-K scores from B_blind and tuned B_full
with the target item's macro category, reports R@20/N@20 stratified by
target ∈ {T&T, non-T&T}, and per-macro.

NO retrain — pure aggregation of cached score matrices + parquet.

Acceptance. Reported either way; updates C5 retrain priorities.

Outputs:
  outputs/<city>/xsage/round3/C5_0/stratified_accuracy.csv
  outputs/<city>/xsage/round3/C5_0/hypothesis_and_read.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.xsage.backbone import excluded_mask, load_or_refit
from pipeline.step02_models.xsage.orchestrator import _load_city


K_TOP = 20
TT_MACRO = "Travel & Transport"


def _per_request_r20_n20(scores: np.ndarray,
                            targets: np.ndarray,
                            exclude: sps.csr_matrix,
                            users: np.ndarray,
                            K: int = K_TOP
                            ) -> tuple[np.ndarray, np.ndarray]:
    n = scores.shape[0]
    r20 = np.zeros(n, dtype=np.float32)
    n20 = np.zeros(n, dtype=np.float32)
    for b in range(n):
        u = int(users[b])
        s = scores[b].copy()
        cols = exclude.indices[exclude.indptr[u]:exclude.indptr[u + 1]]
        if len(cols):
            s[cols] = -np.inf
        target = int(targets[b])
        ts = s[target]
        rank = int((s > ts).sum()) + 1
        if rank <= K:
            r20[b] = 1.0
            n20[b] = 1.0 / np.log2(rank + 1)
    return r20, n20


def _stratify(name: str, mask: np.ndarray, r_b: np.ndarray, n_b: np.ndarray,
                 r_f: np.ndarray, n_f: np.ndarray) -> dict:
    n_req = int(mask.sum())
    if n_req == 0:
        return {"stratum": name, "n_requests": 0,
                  "B_blind_R20": float("nan"), "B_full_R20": float("nan"),
                  "delta_R20": float("nan"),
                  "B_blind_N20": float("nan"), "B_full_N20": float("nan"),
                  "delta_N20": float("nan")}
    b_r = float(r_b[mask].mean()); f_r = float(r_f[mask].mean())
    b_n = float(n_b[mask].mean()); f_n = float(n_f[mask].mean())
    return {"stratum": name, "n_requests": n_req,
              "B_blind_R20": b_r, "B_full_R20": f_r, "delta_R20": f_r - b_r,
              "B_blind_N20": b_n, "B_full_N20": f_n, "delta_N20": f_n - b_n}


def run_city(city: str) -> dict:
    out_dir = REPO_ROOT / "outputs" / city / "xsage" / "round3" / "C5_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> C5.0 on {city}")
    ds = _load_city(city)
    df_test = ds["df_test"]
    u_test = df_test["u_idx"].values.astype(np.int64)
    i_test = df_test["i_idx"].values.astype(np.int64)
    cat_target = df_test["cat_macro"].values.astype(str)

    # Load cached score matrices
    print(f"  loading B_blind (FM-vanilla) and tuned B_full ...")
    scores_blind = load_or_refit(city, model_name="FM")[u_test]
    bfull_scores_path = (REPO_ROOT / "outputs" / city / "xsage" / "backbone"
                              / "Bfull.scores.npy")
    if not bfull_scores_path.exists():
        raise FileNotFoundError(
            f"{bfull_scores_path} missing — re-run experiments.round2_tune_bfull "
            f"--city {city}")
    scores_full = np.load(bfull_scores_path)
    assert scores_full.shape[0] == len(u_test), \
        f"B_full scores have shape {scores_full.shape} but expected {len(u_test)} requests"

    excl_test = (ds["urm_train"] + ds["urm_val"]).tocsr(); excl_test.data[:] = 1.0
    print(f"  computing per-request R@20 and N@20 ...")
    r_blind, n_blind = _per_request_r20_n20(scores_blind, i_test, excl_test, u_test)
    r_full, n_full = _per_request_r20_n20(scores_full, i_test, excl_test, u_test)

    # Strata: ALL, T&T-target, non-T&T-target, then per macro
    rows = []
    rows.append(_stratify("ALL", np.ones(len(u_test), dtype=bool),
                            r_blind, n_blind, r_full, n_full))
    tt_mask = cat_target == TT_MACRO
    rows.append(_stratify("Travel_and_Transport", tt_mask,
                            r_blind, n_blind, r_full, n_full))
    rows.append(_stratify("non_T&T", ~tt_mask,
                            r_blind, n_blind, r_full, n_full))
    for m in sorted(set(cat_target)):
        mask = cat_target == m
        rows.append(_stratify(m, mask, r_blind, n_blind, r_full, n_full))

    pd.DataFrame(rows).to_csv(out_dir / "stratified_accuracy.csv", index=False)
    print(f"\n  {'stratum':25s}  {'n_req':>6s}  {'B_blind R':>9s}  "
          f"{'B_full R':>9s}  {'Δ R':>8s}  {'Δ N':>8s}")
    for r in rows:
        if r["n_requests"] == 0: continue
        print(f"  {r['stratum']:25s}  {r['n_requests']:>6d}  "
              f"{r['B_blind_R20']:>9.4f}  {r['B_full_R20']:>9.4f}  "
              f"{r['delta_R20']:>+8.4f}  {r['delta_N20']:>+8.4f}")
    return {"city": city, "rows": rows}


def write_read(diag_by_city: dict[str, dict], out_path: Path) -> None:
    nyc = diag_by_city["NYC"]; tky = diag_by_city["TKY"]
    nyc_rows = {r["stratum"]: r for r in nyc["rows"]}
    tky_rows = {r["stratum"]: r for r in tky["rows"]}

    lines = [
        "# Round-3 C5.0 — stratified accuracy read\n\n",
        "## Pre-registered hypothesis\n\n",
        "With 73 % T&T mass (C1 finding) on TKY, the per-context features "
        "make B_full over-confident in the T&T prior, crowding out "
        "discrimination on non-T&T targets. **Prediction: B_full's deficit "
        "vs B_blind on TKY concentrates on non-T&T-target requests.**\n\n",
        "## Headline numbers (test, per-request)\n\n",
        "| stratum | n_req | NYC ΔR@20 | NYC ΔN@20 | TKY ΔR@20 | TKY ΔN@20 |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for s in ("ALL", "Travel_and_Transport", "non_T&T"):
        nr = nyc_rows.get(s, {}); tr = tky_rows.get(s, {})
        lines.append(
            f"| {s} | NYC {nr.get('n_requests', 0)} / TKY {tr.get('n_requests', 0)} | "
            f"{nr.get('delta_R20', float('nan')):+.4f} | "
            f"{nr.get('delta_N20', float('nan')):+.4f} | "
            f"{tr.get('delta_R20', float('nan')):+.4f} | "
            f"{tr.get('delta_N20', float('nan')):+.4f} |\n")

    tky_all = tky_rows["ALL"]
    tky_tt = tky_rows.get("Travel_and_Transport", {})
    tky_nontt = tky_rows.get("non_T&T", {})

    lines += [
        "\n## Interpretation\n\n",
        f"**TKY**. Overall B_full ΔR@20 = {tky_all['delta_R20']:+.4f} vs B_blind. "
        f"On T&T targets, ΔR@20 = {tky_tt.get('delta_R20', 0):+.4f} "
        f"({tky_tt.get('n_requests', 0)} requests). On non-T&T targets, "
        f"ΔR@20 = {tky_nontt.get('delta_R20', 0):+.4f} "
        f"({tky_nontt.get('n_requests', 0)} requests).\n\n",
    ]
    if tky_tt.get("delta_R20", 0) > tky_nontt.get("delta_R20", 0):
        lines.append(
            f"The deficit is concentrated on **non-T&T targets** "
            f"({tky_nontt['delta_R20']:+.4f}); on T&T targets B_full actually "
            f"{'gains' if tky_tt['delta_R20'] > 0 else 'closes the gap'} "
            f"({tky_tt['delta_R20']:+.4f}). **Hypothesis confirmed**: the FM is "
            f"over-confident in the T&T prior at the cost of non-T&T "
            f"discrimination.\n\n")
    else:
        lines.append(
            f"The deficit is NOT specifically concentrated on non-T&T targets. "
            f"**Hypothesis falsified**: B_full's failure on TKY does not localise "
            f"on the T&T/non-T&T axis. The C5 retrains must look elsewhere "
            f"(per-context-cell features are still the primary suspect per C1).\n\n")

    lines += [
        "## Action items for C5\n\n",
    ]
    if tky_tt.get("delta_R20", 0) > tky_nontt.get("delta_R20", 0):
        lines += [
            "Confirmed: B_full's TKY deficit is on non-T&T targets. The per-context "
            "features (geo, time) are likely amplifying the T&T prior — they "
            "create per-cell expectations that fit the dominant macro and "
            "drown out non-T&T signals. **C5 priority sharpened: M−geo and "
            "M−time first** (drop those one-hot blocks and see if non-T&T R@20 "
            "recovers).\n\n",
            "M−fine and M−intent are unlikely culprits (cat_fine is well-spread "
            "per C1, and intent_last_cat encodes the SAME signal as the T&T mass "
            "already in the data — removing it is unlikely to flip the verdict).\n",
        ]
    else:
        lines += [
            "The deficit is not localised on T&T vs non-T&T axis. C5 retrain "
            "priorities revert to the C1 suggestion (M−geo, M−time, M−fine+meso, "
            "M−intent) without expectation of which one will recover. If none "
            "does, the defensive narrative stands.\n",
        ]
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"\nwrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", choices=["NYC", "TKY", "both"], default="both")
    args = parser.parse_args()
    cities = ["NYC", "TKY"] if args.city == "both" else [args.city]
    diag = {}
    for c in cities:
        diag[c] = run_city(c)
    if len(cities) == 2:
        read_path = REPO_ROOT / "outputs" / "round3" / "C5_0" / "hypothesis_and_read.md"
        read_path.parent.mkdir(parents=True, exist_ok=True)
        write_read(diag, read_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
