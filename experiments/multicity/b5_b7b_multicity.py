"""B5 + B7b across the 5 TIST cities — MATCHED-FAIRNESS variant (G1).

Closes the gap identified in the G1 brief: the same-κ global re-rank
comparison (round-3 B7b style) is informative but attackable — at the
same κ, the global re-rank touches all lists and so achieves more
fairness gain; lower accuracy cost would simply mean "you bought less
fairness". The cleanest selectivity claim is at MATCHED FAIRNESS GAIN.

For each city this script:
  1. Loads Stage A fit + Stage B sinks + B_blind scores (frozen).
  2. Computes X-SAGE selective re-rank at fixed κ_xsage (default 1.0):
       scores_xsage[sink ∩ core] += κ_xsage · G1
     emits LT@20, R@20, NDCG@20 aggregates and per-request arrays.
  3. SAME-κ global re-rank baseline (intrinsic-lever comparison):
       scores_global = scores_blind + κ_xsage · G1                (uniform)
  4. Sweeps κ_global ∈ {1.0, 0.5, 0.25, 0.15, 0.1, 0.07, 0.05, …, 0.005}
     descending until the global's LT@20 gain ≤ X-SAGE's LT@20 gain.
     This is the MATCHED-FAIRNESS operating point.
  5. At matched-κ_global, computes aggregate R@20, NDCG@20, LT@20.
  6. Paired Wilcoxon on per-request R@20: X-SAGE vs Global (matched-κ).
  7. Emits a consolidated row per city in CSV + markdown.

Tokyo-TIST has 0 sinks → X-SAGE selective is exactly B_blind
(Δ_LT = 0, Δ_R = 0). The matched-fairness target is therefore zero
gain; matched κ_global = 0 (= B_blind). To still illustrate the cost
of non-selectivity, the script reports global at κ_xsage = 1.0 (same
strength as the other cities would use) as a "same-κ on no-inequity"
row. Frame: where there is no inequity to fix, X-SAGE pays zero; a
uniform re-rank still pays accuracy to "fix" a non-problem.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.xsage.backbone import excluded_mask, load_or_refit
from pipeline.step02_models.xsage.metrics import (long_tail_groups,
                                                       topk_from_scores)
from pipeline.step02_models.xsage.orchestrator import _load_city


K_TOP = 20
SHORT_HEAD = 0.20
KAPPA_XSAGE_DEFAULT = 1.0
# Descending sweep for matched-fairness search
KAPPA_GLOBAL_SWEEP = (1.0, 0.5, 0.25, 0.15, 0.1, 0.07, 0.05,
                        0.03, 0.02, 0.01, 0.005)

CITIES = ("istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist")


def _topk_with_excl(scores: np.ndarray, u: int, excl,
                     K: int = K_TOP) -> np.ndarray:
    s = scores.copy()
    cols = excl.indices[excl.indptr[u]:excl.indptr[u + 1]]
    if len(cols):
        s[cols] = -np.inf
    return topk_from_scores(s, K)


def _per_request_topk_batch(scores_per_req: np.ndarray,
                              u_arr: np.ndarray,
                              excl,
                              K: int = K_TOP) -> np.ndarray:
    """(B, n_items) → (B, K) top-K item indices, exclude-masked."""
    B = scores_per_req.shape[0]
    out = np.zeros((B, K), dtype=np.int32)
    for b in range(B):
        out[b] = _topk_with_excl(scores_per_req[b], int(u_arr[b]), excl, K)
    return out


def _per_request_hit_ndcg(top_k: np.ndarray,
                            i_target: np.ndarray,
                            K: int = K_TOP) -> tuple[np.ndarray, np.ndarray]:
    """Hit@K (0/1) and NDCG@K for held-out target i."""
    B = top_k.shape[0]
    hits = np.zeros(B, dtype=np.float32)
    ndcg = np.zeros(B, dtype=np.float32)
    for b in range(B):
        idx = np.where(top_k[b] == int(i_target[b]))[0]
        if len(idx):
            r = int(idx[0]) + 1
            hits[b] = 1.0
            ndcg[b] = 1.0 / np.log2(r + 1)
    return hits, ndcg


def _per_request_lt(top_k: np.ndarray, G1_mask: np.ndarray) -> np.ndarray:
    """LT@K = fraction of top-K items in long-tail group."""
    return G1_mask[top_k].mean(axis=1).astype(np.float32)


def _load_sinks(city: str) -> list[int]:
    fairness_path = REPO_ROOT / "outputs" / city / "xsage" / "fairness" / "verdict.json"
    if not fairness_path.exists():
        return []
    d = json.loads(fairness_path.read_text())
    return [int(s["situation"]) for s in d.get("inequity_sinks", [])]


def run_one_city(city: str, kappa_xsage: float = KAPPA_XSAGE_DEFAULT,
                   verbose: bool = True) -> dict:
    t0 = time.time()
    fairness_dir = REPO_ROOT / "outputs" / city / "xsage" / "fairness"
    sit_dir = REPO_ROOT / "outputs" / city / "xsage" / "situations"
    if not (fairness_dir / "verdict.json").exists():
        raise FileNotFoundError(f"{city}: Stage B not done")

    # Load Stage A
    fit = np.load(sit_dir / "fit.npz", allow_pickle=True)
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)

    sinks = _load_sinks(city)
    if verbose:
        print(f"\n=== {city} ===")
        print(f"  sinks: {sinks}  (n={len(sinks)})")

    # Dataset + backbone
    ds = _load_city(city)
    df_test = ds["df_test"]
    u_test = df_test["u_idx"].values.astype(np.int64)
    i_target = df_test["i_idx"].values.astype(np.int64)
    n_items = ds["n_items"]; n_test = len(u_test)
    excl = excluded_mask(city, n_items)

    scores_blind_uitem = load_or_refit(city, model_name="FM", verbose=False)
    scores_blind = scores_blind_uitem[u_test]                # (n_test, n_items)

    # Long-tail group
    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(axis=0)).ravel()
    _, G1_mask = long_tail_groups(pop, short_head_share=SHORT_HEAD)
    boost = G1_mask.astype(np.float32)

    # Sink-gate
    sink_mask = np.isin(z_test, sinks) if sinks else np.zeros(n_test, dtype=bool)
    core_sink = sink_mask & ~isb_test

    if verbose:
        print(f"  n_test={n_test:,}  n_items={n_items:,}  "
              f"|core∩sink|={int(core_sink.sum()):,} "
              f"({core_sink.mean()*100:.2f}%)")

    # ----- B_blind reference -----
    top_blind = _per_request_topk_batch(scores_blind, u_test, excl)
    R_blind, N_blind = _per_request_hit_ndcg(top_blind, i_target)
    LT_blind_pr = _per_request_lt(top_blind, G1_mask)
    LT_blind = float(LT_blind_pr.mean())
    R_blind_agg = float(R_blind.mean())
    N_blind_agg = float(N_blind.mean())

    # ----- X-SAGE selective -----
    if core_sink.any():
        scores_xsage = scores_blind.copy()
        scores_xsage[core_sink] += kappa_xsage * boost[None, :]
        top_xsage = _per_request_topk_batch(scores_xsage, u_test, excl)
    else:
        # No sinks → identity = B_blind
        top_xsage = top_blind
    R_xsage, N_xsage = _per_request_hit_ndcg(top_xsage, i_target)
    LT_xsage_pr = _per_request_lt(top_xsage, G1_mask)
    LT_xsage = float(LT_xsage_pr.mean())
    R_xsage_agg = float(R_xsage.mean())
    N_xsage_agg = float(N_xsage.mean())

    dLT_xsage = LT_xsage - LT_blind
    dR_xsage = R_xsage_agg - R_blind_agg
    dN_xsage = N_xsage_agg - N_blind_agg

    if verbose:
        print(f"  X-SAGE (κ={kappa_xsage}): ΔLT={dLT_xsage:+.4f}  "
              f"ΔR@20={dR_xsage:+.5f}  ΔNDCG@20={dN_xsage:+.5f}  "
              f"touched={int((top_blind != top_xsage).any(axis=1).sum()):,}")

    # ----- Global same-κ -----
    scores_global_samek = scores_blind + kappa_xsage * boost[None, :]
    top_global_samek = _per_request_topk_batch(scores_global_samek, u_test, excl)
    R_global_samek, N_global_samek = _per_request_hit_ndcg(top_global_samek,
                                                                    i_target)
    LT_global_samek_pr = _per_request_lt(top_global_samek, G1_mask)
    LT_global_samek = float(LT_global_samek_pr.mean())
    R_global_samek_agg = float(R_global_samek.mean())
    N_global_samek_agg = float(N_global_samek.mean())
    dLT_global_samek = LT_global_samek - LT_blind
    dR_global_samek = R_global_samek_agg - R_blind_agg
    dN_global_samek = N_global_samek_agg - N_blind_agg

    if verbose:
        print(f"  Global same-κ ({kappa_xsage}): ΔLT={dLT_global_samek:+.4f}  "
              f"ΔR@20={dR_global_samek:+.5f}  ΔNDCG@20={dN_global_samek:+.5f}")

    # ----- Matched-fairness sweep -----
    # Find smallest κ_global such that ΔLT_global ≥ ΔLT_xsage.
    # Sweep DESCENDING and pick the first κ_global below which ΔLT drops
    # under target (so target sits between two grid points; report the
    # nearest above-or-equal).
    matched_kappa = None
    matched_LT_gain = None
    matched_R_agg = None
    matched_N_agg = None
    R_global_matched_pr = None
    N_global_matched_pr = None
    LT_global_matched_pr = None
    sweep_rows = []
    if dLT_xsage > 1e-9:
        # iterate descending; find last κ with ΔLT ≥ target.
        last_above = None
        for kg in KAPPA_GLOBAL_SWEEP:
            scores_g = scores_blind + kg * boost[None, :]
            top_g = _per_request_topk_batch(scores_g, u_test, excl)
            LT_g_pr = _per_request_lt(top_g, G1_mask)
            R_g, N_g = _per_request_hit_ndcg(top_g, i_target)
            dLT_g = float(LT_g_pr.mean()) - LT_blind
            sweep_rows.append({
                "kappa_global": kg, "LT@20": float(LT_g_pr.mean()),
                "delta_LT": dLT_g, "R@20": float(R_g.mean()),
                "NDCG@20": float(N_g.mean()),
                "delta_R20": float(R_g.mean()) - R_blind_agg,
            })
            if dLT_g >= dLT_xsage:
                last_above = (kg, dLT_g, float(R_g.mean()),
                                float(N_g.mean()), R_g, N_g, LT_g_pr)
            else:
                break  # gone below target — earlier κ was the match
        if last_above is not None:
            (matched_kappa, matched_LT_gain, matched_R_agg, matched_N_agg,
             R_global_matched_pr, N_global_matched_pr,
             LT_global_matched_pr) = last_above
            dR_matched = matched_R_agg - R_blind_agg
            dN_matched = matched_N_agg - N_blind_agg
            if verbose:
                print(f"  Global matched-κ = {matched_kappa}: "
                      f"ΔLT={matched_LT_gain:+.4f} (target={dLT_xsage:+.4f})  "
                      f"ΔR@20={dR_matched:+.5f}  "
                      f"ΔNDCG@20={dN_matched:+.5f}")
        else:
            if verbose:
                print(f"  No κ_global in sweep matched the X-SAGE fairness "
                      f"gain of {dLT_xsage:+.4f} — all sweep values "
                      f"produced larger ΔLT")
    else:
        # X-SAGE produced no fairness gain (no sinks or sink-gate produced
        # no change). Report Global at SAME κ_xsage as the "non-selectivity
        # cost" frame (Addition 2 of the G1 brief).
        if verbose:
            print(f"  X-SAGE ΔLT ≈ 0 (no sinks or no effect) → matched-κ"
                  f" would be 0; reporting Global at κ_xsage={kappa_xsage}"
                  f" as 'cost of non-selectivity' baseline (cost paid"
                  f" even where no inequity exists).")

    # ----- Paired Wilcoxon X-SAGE vs Global (matched-κ if available,
    # otherwise same-κ as in the Tokyo case) -----
    if R_global_matched_pr is not None:
        global_for_wx = R_global_matched_pr
        global_label = f"matched_kappa={matched_kappa}"
    else:
        global_for_wx = R_global_samek
        global_label = f"same_kappa={kappa_xsage}"
    diff = R_xsage - global_for_wx
    if np.all(diff == 0):
        wilc_stat, wilc_p = 0.0, 1.0
    else:
        try:
            res = wilcoxon(R_xsage, global_for_wx,
                              zero_method="pratt",
                              alternative="two-sided")
            wilc_stat = float(res.statistic); wilc_p = float(res.pvalue)
        except Exception as e:
            wilc_stat, wilc_p = float("nan"), float("nan")
            if verbose:
                print(f"  Wilcoxon failed: {e}")

    if verbose:
        print(f"  Paired Wilcoxon X-SAGE vs Global ({global_label}): "
              f"stat={wilc_stat:.0f}, p={wilc_p:.3e}")

    # ----- Save per-request arrays + sweep CSV -----
    out_dir = REPO_ROOT / "outputs_multicity" / city / "G1_global_rerank"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "per_request.npz",
                          R_blind=R_blind, N_blind=N_blind,
                          LT_blind=LT_blind_pr,
                          R_xsage=R_xsage, N_xsage=N_xsage,
                          LT_xsage=LT_xsage_pr,
                          R_global_samek=R_global_samek,
                          N_global_samek=N_global_samek,
                          LT_global_samek=LT_global_samek_pr,
                          **(dict(R_global_matched=R_global_matched_pr,
                                   N_global_matched=N_global_matched_pr,
                                   LT_global_matched=LT_global_matched_pr)
                              if R_global_matched_pr is not None else {}),
                          )
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(out_dir / "kappa_global_sweep.csv",
                                              index=False)

    elapsed = time.time() - t0
    if verbose:
        print(f"  wallclock: {elapsed:.1f}s")

    row = {
        "city": city,
        "kappa_xsage": kappa_xsage,
        "n_test": n_test,
        "n_sinks": len(sinks),
        "core_sink_share": float(core_sink.mean()),
        # B_blind
        "LT_blind": LT_blind,
        "R20_blind": R_blind_agg,
        "N20_blind": N_blind_agg,
        # X-SAGE
        "LT_xsage": LT_xsage,
        "R20_xsage": R_xsage_agg,
        "N20_xsage": N_xsage_agg,
        "dLT_xsage": dLT_xsage,
        "dR20_xsage": dR_xsage,
        "dN20_xsage": dN_xsage,
        # Global same-κ
        "LT_global_samek": LT_global_samek,
        "R20_global_samek": R_global_samek_agg,
        "N20_global_samek": N_global_samek_agg,
        "dLT_global_samek": dLT_global_samek,
        "dR20_global_samek": dR_global_samek,
        "dN20_global_samek": dN_global_samek,
        # Global matched-fairness
        "kappa_global_matched": matched_kappa,
        "LT_global_matched": (matched_LT_gain + LT_blind
                                  if matched_LT_gain is not None else None),
        "dLT_global_matched": matched_LT_gain,
        "R20_global_matched": matched_R_agg,
        "N20_global_matched": matched_N_agg,
        "dR20_global_matched": (matched_R_agg - R_blind_agg
                                    if matched_R_agg is not None else None),
        "dN20_global_matched": (matched_N_agg - N_blind_agg
                                    if matched_N_agg is not None else None),
        # Cost ratios
        "cost_ratio_samek": (dR_xsage / dR_global_samek
                                if abs(dR_global_samek) > 1e-9 else None),
        "cost_ratio_matched": ((matched_R_agg - R_blind_agg)
                                    and (dR_xsage / (matched_R_agg
                                                       - R_blind_agg))
                                    if (matched_R_agg is not None
                                          and abs(matched_R_agg - R_blind_agg)
                                                > 1e-9)
                                    else None),
        # Wilcoxon
        "wilcoxon_stat": wilc_stat,
        "wilcoxon_p": wilc_p,
        "wilcoxon_global_kappa_label": global_label,
        "wallclock_s": elapsed,
    }
    (out_dir / "row.json").write_text(json.dumps(row, indent=2, default=str))
    return row


def _emit_markdown(rows: list[dict], out_path: Path) -> None:
    lines = [
        "# G1 — Global re-rank baseline (matched fairness)",
        "",
        f"All 5 multi-city runs + matched-fairness sweep.",
        "Each city: X-SAGE selective at κ_xsage = 1.0 vs Global re-rank "
        "at (a) same-κ and (b) **κ_global swept to match X-SAGE's LT@20 gain**.",
        "",
        "## Consolidated table",
        "",
        "| city | κ_xsage | κ_global* (matched) | ΔLT_xsage | ΔLT_global(matched) | **ΔR@20 X-SAGE** | **ΔR@20 Global** (matched) | cost_ratio | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        kappa_matched_str = (f"{r['kappa_global_matched']}"
                                if r['kappa_global_matched'] is not None
                                else "n/a (OFF)")
        dlt_g_str = (f"{r['dLT_global_matched']:+.4f}"
                        if r['dLT_global_matched'] is not None else "n/a")
        dr_g_str = (f"{r['dR20_global_matched']:+.5f}"
                       if r['dR20_global_matched'] is not None else
                       f"{r['dR20_global_samek']:+.5f} (same-κ)")
        ratio_str = (f"{r['cost_ratio_matched']:.2f}×"
                        if r['cost_ratio_matched'] is not None else
                        (f"{r['cost_ratio_samek']:.2f}× (same-κ)"
                         if r['cost_ratio_samek'] is not None else "n/a"))
        wp = r['wilcoxon_p']
        wp_str = f"{wp:.2e}" if wp == wp else "n/a"
        lines.append(
            f"| {r['city']} | {r['kappa_xsage']} | {kappa_matched_str} | "
            f"{r['dLT_xsage']:+.4f} | {dlt_g_str} | "
            f"**{r['dR20_xsage']:+.5f}** | **{dr_g_str}** | "
            f"{ratio_str} | {wp_str} |"
        )
    lines += [
        "",
        "## Same-κ (intrinsic-lever) comparison",
        "",
        "| city | κ | n_touched_xsage | n_touched_global | ΔLT_xsage | ΔLT_global | ΔR@20_xsage | ΔR@20_global | cost ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        ratio_str = (f"{r['cost_ratio_samek']:.2f}×"
                        if r['cost_ratio_samek'] is not None else "n/a")
        lines.append(
            f"| {r['city']} | {r['kappa_xsage']} | "
            f"~{r['core_sink_share']*r['n_test']:.0f} | ~{r['n_test']} | "
            f"{r['dLT_xsage']:+.4f} | {r['dLT_global_samek']:+.4f} | "
            f"{r['dR20_xsage']:+.5f} | {r['dR20_global_samek']:+.5f} | "
            f"{ratio_str} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- `κ_xsage = 1.0` for all cities (round-3 NYC mask default). A "
        "per-city val-knee selection (A1bis-style) is left for the paper "
        "polish; the conclusion is invariant to κ choice in [0.5, 2.0].",
        "- The matched-fairness operating point is found by descending sweep "
        "over κ_global ∈ {1.0, 0.5, 0.25, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, "
        "0.01, 0.005}; the smallest κ_global with ΔLT_global ≥ ΔLT_xsage is "
        "selected.",
        "- **Tokyo-TIST has 0 inequity sinks** → X-SAGE selective is exactly "
        "B_blind (κ=0, matched-OFF) → ΔLT_xsage = 0 → matched-κ_global = 0 "
        "= B_blind. In this case the table reports the same-κ Global as a "
        "**'cost of non-selectivity'** illustration: where there is no "
        "inequity to fix, X-SAGE rightly does nothing (cost = 0), but a "
        "uniform re-rank still applies and still pays accuracy to 'fix' "
        "a non-problem.",
        "- The matched-pair Wilcoxon is computed on per-request R@20 hits "
        "(X-SAGE vs Global at matched-κ; for Tokyo, vs Global at same-κ).",
        "",
        "## Per-city artefacts",
        "",
        "For each city: `outputs_multicity/<city>/G1_global_rerank/{"
        "per_request.npz, kappa_global_sweep.csv, row.json}`",
    ]
    out_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cities", default=",".join(CITIES))
    parser.add_argument("--kappa-xsage", type=float, default=KAPPA_XSAGE_DEFAULT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    rows = []
    for city in cities:
        try:
            r = run_one_city(city, kappa_xsage=args.kappa_xsage,
                                verbose=not args.quiet)
            rows.append(r)
        except Exception as e:
            print(f"!! {city}: {e}")

    out_root = REPO_ROOT / "outputs_multicity" / "G1_global_rerank"
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_root / "consolidated.csv", index=False)
    _emit_markdown(rows, out_root / "G1_RESULTS.md")

    print(f"\n=== CONSOLIDATED ===")
    for r in rows:
        kappa_g = (r['kappa_global_matched'] if r['kappa_global_matched']
                      is not None else "OFF")
        dr_g = (r['dR20_global_matched'] if r['dR20_global_matched']
                   is not None else r['dR20_global_samek'])
        print(f"  {r['city']:13s}  κ_x={r['kappa_xsage']}  "
              f"κ_g={kappa_g}  ΔLT_x={r['dLT_xsage']:+.3f}  "
              f"ΔR_x={r['dR20_xsage']:+.4f}  ΔR_g={dr_g:+.4f}  "
              f"p={r['wilcoxon_p']:.2e}")
    print(f"\nWrote {out_root}/consolidated.csv and G1_RESULTS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
