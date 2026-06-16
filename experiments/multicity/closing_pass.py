"""Closing pass — G1 tidy + G2 fairness coherence + G3 stat hardening
+ G4 projection as disambiguator. All read-only on existing artefacts;
no retraining.

Outputs:
  G1-tidy → outputs_multicity/G1_global_rerank/G1_RESULTS.md (appended)
            + outputs_multicity/G1_global_rerank/identity_check.json
  G2     → outputs_multicity/<city>/fairness_coherence.{csv,md}
  G3     → outputs_multicity/multicity_stats.{csv,md}
  G4     → outputs_multicity/<city>/projection_disambiguation.{csv,md}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.xsage.backbone import excluded_mask, load_or_refit
from pipeline.step02_models.xsage.metrics import (long_tail_groups,
                                                       topk_from_scores)
from pipeline.step02_models.xsage.orchestrator import _load_city
from pipeline.step02_models.xsage.l3_projection import (estimate_transition,
                                                            boundary_disambiguate)
from pipeline.step02_models.xsage.recommendation import (
    fit_situation_biases_z,
)


CITIES = ("istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist")
K_TOP = 20
SHORT_HEAD = 0.20
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gini(values: np.ndarray) -> float:
    """Standard Gini on a non-negative 1D array. Returns 0 for uniform / all-zero."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = len(v); s = v.sum()
    if n == 0 or s == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum() / (n * s)) - (n + 1) / n)


def per_request_R20_and_ranklist(scores_per_req: np.ndarray,
                                     u_arr: np.ndarray,
                                     i_target: np.ndarray,
                                     excl, K: int = K_TOP):
    """Per-request: hit@K (0/1), ndcg@K, and top-K item indices."""
    B = scores_per_req.shape[0]
    hits = np.zeros(B, dtype=np.float32)
    ndcg = np.zeros(B, dtype=np.float32)
    topk_items = np.zeros((B, K), dtype=np.int32)
    for b in range(B):
        u = int(u_arr[b])
        s = scores_per_req[b].copy()
        cols = excl.indices[excl.indptr[u]:excl.indptr[u + 1]]
        if len(cols):
            s[cols] = -np.inf
        tgt = int(i_target[b])
        ts = s[tgt]
        rank = int((s > ts).sum()) + 1
        if rank <= K:
            hits[b] = 1.0
            ndcg[b] = 1.0 / np.log2(rank + 1)
        topk_items[b] = topk_from_scores(s, K)
    return hits, ndcg, topk_items


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down. Returns adjusted p in original order."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adj = [0.0] * n
    prev = 0.0
    for rank, i in enumerate(order):
        a = (n - rank) * p_values[i]
        a = min(1.0, max(prev, a))
        prev = a
        adj[i] = a
    return adj


def paired_bootstrap_ci(diff: np.ndarray, B: int = 10_000,
                          seed: int = 42) -> tuple[float, float, float]:
    """Mean + percentile CI95 of `diff` via paired bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return (0.0, 0.0, 0.0)
    boot = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot[b] = diff[idx].mean()
    return (float(diff.mean()),
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


# ---------------------------------------------------------------------------
# G1 tidy — κ=0 identity check + LT-match tolerance
# ---------------------------------------------------------------------------

def g1_tidy(cities, out_dir: Path):
    print(f"\n{'='*60}\nG1 tidy — κ=0 identity check + LT tolerance\n{'='*60}")
    out_dir.mkdir(parents=True, exist_ok=True)
    identity = []
    lt_tolerance = []
    for city in cities:
        # κ=0 identity: from outputs/<city>/xsage/recommendation/three_way.csv
        twp = REPO_ROOT / "outputs" / city / "xsage" / "recommendation" / "three_way.csv"
        if not twp.exists():
            print(f"  [{city}] no three_way.csv — skipping identity check")
            continue
        df = pd.read_csv(twp)
        rb = df[df["variant"] == "B_blind"].iloc[0]
        rx0 = df[(df["variant"] == "X-SAGE") &
                   (df["kappa"].astype(str) == "0.0")].iloc[0]
        delta_r = float(rb["R_at_20"]) - float(rx0["R_at_20"])
        delta_n = float(rb["N_at_20"]) - float(rx0["N_at_20"])
        ok = (abs(delta_r) < 1e-9) and (abs(delta_n) < 1e-9)
        identity.append({"city": city,
                            "B_blind_R20": float(rb["R_at_20"]),
                            "X-SAGE_k0_R20": float(rx0["R_at_20"]),
                            "delta_R20": delta_r,
                            "delta_N20": delta_n,
                            "identity_holds": ok})
        print(f"  [{city}] B_blind={rb['R_at_20']:.6f}  "
              f"X-SAGE(κ=0)={rx0['R_at_20']:.6f}  "
              f"|Δ|={abs(delta_r):.2e}  {'✓' if ok else '✗'}")

        # LT-match tolerance: from G1 row.json
        row_path = REPO_ROOT / "outputs_multicity" / city / "G1_global_rerank" / "row.json"
        if row_path.exists():
            row = json.loads(row_path.read_text())
            lt_x = row.get("LT_xsage")
            lt_g = row.get("LT_global_matched")
            tol = (abs(lt_g - lt_x) if (lt_x is not None and lt_g is not None)
                      else None)
            lt_tolerance.append({"city": city,
                                    "kappa_global_matched": row.get("kappa_global_matched"),
                                    "LT_xsage": lt_x, "LT_global_matched": lt_g,
                                    "abs_tol": tol})
    n_pass = sum(1 for r in identity if r["identity_holds"])
    print(f"\n  κ=0 identity: {n_pass}/{len(identity)} cities pass exactly")
    (out_dir / "identity_check.json").write_text(json.dumps({
        "identity_check": identity,
        "lt_match_tolerance": lt_tolerance,
        "n_pass": n_pass, "n_total": len(identity),
    }, indent=2, default=str))
    return {"identity": identity, "lt_tolerance": lt_tolerance}


# ---------------------------------------------------------------------------
# G2 — Fairness coherence (LT vs KL vs Gini per situation)
# ---------------------------------------------------------------------------

def g2_one_city(city: str) -> dict:
    print(f"\n[G2 {city}]")
    fair_csv = REPO_ROOT / "outputs" / city / "xsage" / "fairness" / "per_situation.csv"
    if not fair_csv.exists():
        print(f"  no per_situation.csv — skipping")
        return {}
    df_fair = pd.read_csv(fair_csv)
    df_all = df_fair[df_fair["split"] == "all"].copy()

    fit_path = REPO_ROOT / "outputs" / city / "xsage" / "situations" / "fit.npz"
    fit = np.load(fit_path, allow_pickle=True)
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)

    ds = _load_city(city)
    df_test = ds["df_test"]
    u_test = df_test["u_idx"].values.astype(np.int64)
    i_target = df_test["i_idx"].values.astype(np.int64)
    n_items = ds["n_items"]
    excl = excluded_mask(city, n_items)

    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(axis=0)).ravel()
    _, G1_mask = long_tail_groups(pop, short_head_share=SHORT_HEAD)

    # Score requests with B_blind, get per-request top-K
    scores_blind = load_or_refit(city, model_name="FM", verbose=False)[u_test]
    _, _, topk = per_request_R20_and_ranklist(scores_blind, u_test, i_target, excl)

    # Per-situation Gini of item exposure across requests in that situation
    sit_rows = []
    K_sit = int(z_test.max() + 1)
    n_test = len(u_test)
    for k in range(K_sit):
        mask_k = (z_test == k)
        n_k = int(mask_k.sum())
        # exposure histogram: count how many times each item appears in the
        # top-K of requests assigned to situation k
        if n_k == 0:
            sit_rows.append({"situation": k, "n_requests": 0,
                                "LT": 0.0, "KL": 0.0, "Gini": 0.0,
                                "is_sink_LT": False, "is_sink_KL": False,
                                "is_sink_Gini": False})
            continue
        flat = topk[mask_k].ravel()                  # (n_k * K,)
        counts = np.bincount(flat.astype(np.int64), minlength=n_items)
        g = gini(counts)
        # LT and KL from fairness/per_situation.csv (already computed, "all" split)
        row_all = df_all[df_all["situation"] == k].iloc[0]
        sit_rows.append({"situation": k, "n_requests": n_k,
                            "LT": float(row_all["LT"]),
                            "KL": float(row_all["KL"]),
                            "Gini": g,
                            # placeholders for the sink flags, set later
                            })

    # Define sinks per metric:
    # - LT-sink: LT is FAR BELOW available_LT (i.e. low LT-rate vs feasible)
    #   → use the fairness/verdict.json's existing inequity_sinks
    # - KL-sink: KL ≥ ~2× global mean (the round-3 threshold)
    # - Gini-sink: Gini ≥ mean + 1 std across situations
    verdict_path = REPO_ROOT / "outputs" / city / "xsage" / "fairness" / "verdict.json"
    verdict = json.loads(verdict_path.read_text())
    sinks_LT = set(int(s["situation"]) for s in verdict.get("inequity_sinks", []))
    global_kl = verdict.get("global_KL_mean_all_split", 0.0)
    # KL-sink: KL ≥ 2× global_kl
    # Gini-sink: Gini ≥ mean + 1*std across SAME situations
    ginis = np.array([r["Gini"] for r in sit_rows])
    g_mean = ginis.mean(); g_std = ginis.std()
    sinks_Gini = set(r["situation"] for r in sit_rows
                       if r["Gini"] >= g_mean + g_std)
    sinks_KL = set(r["situation"] for r in sit_rows
                     if global_kl > 0 and r["KL"] >= 2.0 * global_kl)
    for r in sit_rows:
        r["is_sink_LT"]   = bool(r["situation"] in sinks_LT)
        r["is_sink_KL"]   = bool(r["situation"] in sinks_KL)
        r["is_sink_Gini"] = bool(r["situation"] in sinks_Gini)

    # Coherence verdict
    all_sinks = sinks_LT | sinks_KL | sinks_Gini
    intersect = sinks_LT & sinks_KL & sinks_Gini if all_sinks else set()
    coherent = (sinks_LT == sinks_KL == sinks_Gini)
    # if all empty, also coherent
    if not all_sinks:
        coherent = True

    print(f"  K={K_sit}  sinks_LT={sinks_LT}  sinks_KL={sinks_KL}  "
          f"sinks_Gini={sinks_Gini}  coherent={coherent}")

    out_dir = REPO_ROOT / "outputs_multicity" / city
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sit_rows).to_csv(out_dir / "fairness_coherence.csv", index=False)

    # Markdown one-pager
    md = [f"# {city} — fairness metric coherence (G2)", "",
            f"Definitions:",
            f"- **LT-sink**: from `fairness/verdict.json::inequity_sinks` (KL ratio ≥ ~2× global)",
            f"- **KL-sink**: KL ≥ 2 × global_KL_mean ({2*global_kl:.3f})",
            f"- **Gini-sink**: per-situation item-exposure Gini ≥ mean + 1·std "
              f"(thresh = {g_mean+g_std:.3f})",
            "",
            f"| situation | n_requests | LT | KL | Gini | LT-sink | KL-sink | Gini-sink |",
            "|---|---:|---:|---:|---:|:---:|:---:|:---:|"]
    for r in sit_rows:
        md.append(f"| s{r['situation']} | {r['n_requests']} | {r['LT']:.4f} | "
                  f"{r['KL']:.4f} | {r['Gini']:.4f} | "
                  f"{'✓' if r['is_sink_LT'] else ''} | "
                  f"{'✓' if r['is_sink_KL'] else ''} | "
                  f"{'✓' if r['is_sink_Gini'] else ''} |")
    md.append("")
    md.append(f"**Coherence verdict**: {'COHERENT' if coherent else 'DISAGREEMENT'}")
    md.append(f"- sinks_LT = {sorted(sinks_LT)}")
    md.append(f"- sinks_KL = {sorted(sinks_KL)}")
    md.append(f"- sinks_Gini = {sorted(sinks_Gini)}")
    if not coherent:
        md.append(f"- **intersect (agree on)**: {sorted(intersect)}")
        md.append(f"- **disagreement**: situations in some but not all metric: "
                  f"{sorted(all_sinks - intersect)}")
    (out_dir / "fairness_coherence.md").write_text("\n".join(md))

    return {"city": city, "K": K_sit,
            "sinks_LT": sorted(sinks_LT),
            "sinks_KL": sorted(sinks_KL),
            "sinks_Gini": sorted(sinks_Gini),
            "coherent": coherent,
            "intersect": sorted(intersect),
            "per_situation": sit_rows}


# ---------------------------------------------------------------------------
# G3 — Multi-city statistical hardening (CI95 + Holm on ΔR@20 and ΔF1)
# ---------------------------------------------------------------------------

def g3_one_city(city: str, B_boot: int = 10_000) -> dict:
    print(f"\n[G3 {city}]")
    ds = _load_city(city)
    df_test = ds["df_test"]
    u_test = df_test["u_idx"].values.astype(np.int64)
    i_target = df_test["i_idx"].values.astype(np.int64)
    n_items = ds["n_items"]
    excl = excluded_mask(city, n_items)

    # B_blind per-request hits
    scores_blind = load_or_refit(city, model_name="FM", verbose=False)[u_test]
    R_blind, _, _ = per_request_R20_and_ranklist(scores_blind, u_test,
                                                       i_target, excl)
    # B_full per-request hits (mmap to keep memory low for Istanbul)
    bfull_path = REPO_ROOT / "outputs" / city / "xsage" / "backbone" / "Bfull.scores.npy"
    if not bfull_path.exists():
        print(f"  missing {bfull_path} — skipping")
        return {}
    scores_full_mmap = np.load(bfull_path, mmap_mode="r")
    # Compute per-request hits in batches for Istanbul-scale (avoid copying)
    R_full = np.zeros(len(u_test), dtype=np.float32)
    BSZ = 1024
    for s in range(0, len(u_test), BSZ):
        e = min(len(u_test), s + BSZ)
        sb = np.asarray(scores_full_mmap[s:e])
        h, _, _ = per_request_R20_and_ranklist(sb, u_test[s:e],
                                                     i_target[s:e], excl)
        R_full[s:e] = h

    # Aggregate per-user
    uniq, inv = np.unique(u_test, return_inverse=True)
    cnt = np.zeros(len(uniq))
    np.add.at(cnt, inv, 1)
    sum_blind = np.zeros(len(uniq))
    sum_full = np.zeros(len(uniq))
    np.add.at(sum_blind, inv, R_blind.astype(np.float64))
    np.add.at(sum_full, inv, R_full.astype(np.float64))
    per_user_blind = sum_blind / cnt
    per_user_full = sum_full / cnt
    diff = per_user_full - per_user_blind

    mean, lo, hi = paired_bootstrap_ci(diff, B=B_boot)
    crosses_zero = (lo <= 0.0 <= hi)
    # Wilcoxon paired test
    try:
        wres = stats.wilcoxon(per_user_full, per_user_blind,
                                  zero_method="pratt",
                                  alternative="two-sided")
        wilc_p = float(wres.pvalue)
    except Exception:
        wilc_p = float("nan")
    print(f"  ΔR@20 (B_full - B_blind) macro-user: "
          f"{mean:+.5f}  CI95 [{lo:+.5f}, {hi:+.5f}]  "
          f"Wilcoxon p={wilc_p:.3e}  crosses0={crosses_zero}")

    # ΔF1 from Stage C verdict.json (no CI computed — McNemar p is reported)
    proj_path = REPO_ROOT / "outputs" / city / "xsage" / "projection" / "verdict.json"
    if proj_path.exists():
        proj = json.loads(proj_path.read_text())
        delta_f1 = float(proj.get("delta_F1", 0.0))
        mc_p = float(proj.get("mcnemar_p_value", float("nan")))
        n10 = int(proj.get("mcnemar_T_only_correct", 0))
        n01 = int(proj.get("mcnemar_time_only_correct", 0))
    else:
        delta_f1 = float("nan"); mc_p = float("nan"); n10 = n01 = 0

    return {"city": city, "n_users": int(len(uniq)),
              "B_blind_R20": float(per_user_blind.mean()),
              "B_full_R20": float(per_user_full.mean()),
              "delta_R20_paired_user_mean": mean,
              "delta_R20_ci95_lo": lo, "delta_R20_ci95_hi": hi,
              "delta_R20_wilcoxon_p": wilc_p,
              "delta_R20_crosses_zero": crosses_zero,
              "delta_F1_T_vs_clock": delta_f1,
              "mcnemar_p": mc_p,
              "mcnemar_n10_T_only_correct": n10,
              "mcnemar_n01_time_only_correct": n01,
              }


def g3_finalize(rows: list[dict], out_root: Path):
    if not rows:
        return
    # Holm on the 5 ΔR@20 Wilcoxon p-values
    p_R = [r["delta_R20_wilcoxon_p"] for r in rows]
    p_R_holm = holm(p_R)
    # Holm on the 5 ΔF1 McNemar p-values
    p_F = [r["mcnemar_p"] for r in rows]
    p_F_holm = holm(p_F)
    for i, r in enumerate(rows):
        r["delta_R20_wilcoxon_p_holm"] = p_R_holm[i]
        r["delta_R20_holm_reject"] = bool(p_R_holm[i] < ALPHA)
        r["delta_F1_mcnemar_p_holm"] = p_F_holm[i]
        r["delta_F1_holm_reject"] = bool(p_F_holm[i] < ALPHA)

    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_root / "multicity_stats.csv", index=False)

    md = ["# G3 — Multi-city statistical hardening", "",
            "Per-city paired bootstrap CI95 + Holm step-down over the 5 cities.",
            "",
            "## ΔR@20 = B_full − B_blind (paired bootstrap on per-USER mean, B=10 000)",
            "",
            "| city | n_users | B_blind R@20 | B_full R@20 | **ΔR@20** | **CI95** | Wilcoxon p | **Holm p** | reject H₀? | crosses 0? |",
            "|---|---:|---:|---:|---:|---|---:|---:|:---:|:---:|"]
    for r in rows:
        ci = f"[{r['delta_R20_ci95_lo']:+.5f}, {r['delta_R20_ci95_hi']:+.5f}]"
        md.append(f"| {r['city']} | {r['n_users']:,} | {r['B_blind_R20']:.4f} | "
                  f"{r['B_full_R20']:.4f} | {r['delta_R20_paired_user_mean']:+.5f} | "
                  f"{ci} | {r['delta_R20_wilcoxon_p']:.2e} | "
                  f"{r['delta_R20_wilcoxon_p_holm']:.2e} | "
                  f"{'✅' if r['delta_R20_holm_reject'] else '❌'} | "
                  f"{'YES' if r['delta_R20_crosses_zero'] else 'no'} |")
    md += ["", "## ΔF1 = macro-F1(T-based) − macro-F1(time-only)", "",
             "Per-city McNemar (already in Stage C); Holm across 5 cities.",
             "",
             "| city | ΔF1 | n10 (T-only) | n01 (time-only) | McNemar p | **Holm p** | reject H₀? |",
             "|---|---:|---:|---:|---:|---:|:---:|"]
    for r in rows:
        md.append(f"| {r['city']} | {r['delta_F1_T_vs_clock']:+.4f} | "
                  f"{r['mcnemar_n10_T_only_correct']:,} | "
                  f"{r['mcnemar_n01_time_only_correct']:,} | "
                  f"{r['mcnemar_p']:.2e} | "
                  f"{r['delta_F1_mcnemar_p_holm']:.2e} | "
                  f"{'✅' if r['delta_F1_holm_reject'] else '❌'} |")
    (out_root / "multicity_stats.md").write_text("\n".join(md))
    print(f"\n[G3] Wrote {out_root}/multicity_stats.{{csv,md}}")


# ---------------------------------------------------------------------------
# G4 — Projection as disambiguator (boundary recognition + do-no-harm)
# ---------------------------------------------------------------------------

def _macro_f1(y_true, y_pred, K):
    f1s = []
    for k in range(K):
        tp = int(((y_true == k) & (y_pred == k)).sum())
        fp = int(((y_true != k) & (y_pred == k)).sum())
        fn = int(((y_true == k) & (y_pred != k)).sum())
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        f1 = 2*p*r / max(1e-9, p + r)
        f1s.append(f1)
    return float(np.mean(f1s))


def g4_one_city(city: str) -> dict:
    print(f"\n[G4 {city}]")
    sit_dir = REPO_ROOT / "outputs" / city / "xsage" / "situations"
    fit = np.load(sit_dir / "fit.npz", allow_pickle=True)
    z_train = np.asarray(fit["core_label_train"]).astype(np.int32)
    z_val = np.asarray(fit["core_label_val"]).astype(np.int32)
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)
    membership_test = np.asarray(fit["membership_test"]).astype(np.float32)
    K_sit = int(max(z_train.max(), z_val.max(), z_test.max()) + 1)

    ds = _load_city(city)
    df_train = ds["df_train"]; df_val = ds["df_val"]; df_test = ds["df_test"]

    # Estimate situation transition matrix T on TRAIN sequences (causal)
    train_seqs = []
    df_t_sorted = df_train.sort_values(["user_id", "time_local"]).reset_index(drop=True)
    # Construct per-user sequences
    u_arr = df_t_sorted["user_id"].values
    z_arr_train = z_train.copy()  # in the same order as df_train pre-sort?
    # Note: z_train is in the order of df_train (the unsorted). We need to re-align.
    # The fit_train uses the same row order as the original df_train.
    # To build sequences in chronological order, we need to (re-)sort by user/time
    # and apply same permutation to z_train. Use df_train index → z_train index.
    df_train_idx = df_train.copy(); df_train_idx["_rownum"] = np.arange(len(df_train))
    df_train_sorted = df_train_idx.sort_values(["user_id", "time_local"]
                                                     ).reset_index(drop=True)
    z_train_sorted = z_train[df_train_sorted["_rownum"].values]
    # build sequences
    user_changes = np.concatenate([[0],
                                       np.where(np.diff(df_train_sorted["user_id"].values) != 0)[0] + 1,
                                       [len(df_train_sorted)]])
    for i in range(len(user_changes) - 1):
        a, b = user_changes[i], user_changes[i+1]
        train_seqs.append(z_train_sorted[a:b])
    T, raw = estimate_transition(train_seqs, K_sit)

    # Build z_prev per TEST request
    #   For each user, chronological list of all interactions across splits.
    #   For each test row, find the most recent prior interaction's z.
    df_all = pd.concat([
        df_train.assign(_split="train", _z=z_train),
        df_val.assign(_split="val", _z=z_val),
        df_test.assign(_split="test", _z=z_test),
    ], ignore_index=True)
    df_all_sorted = df_all.sort_values(["user_id", "time_local"]
                                            ).reset_index(drop=True)
    # For each row whose split=="test", find the previous row of the same user.
    # df_all_sorted is in (user, time) order, so the "previous row" of a user
    # is the one immediately before, if user matches.
    same_user = (df_all_sorted["user_id"].values[1:] ==
                    df_all_sorted["user_id"].values[:-1])
    # Build a mapping: for each row in df_all_sorted, z_prev = z of previous
    # row if same user else -1
    z_prev_full = np.full(len(df_all_sorted), -1, dtype=np.int32)
    z_prev_full[1:][same_user] = df_all_sorted["_z"].values[:-1][same_user].astype(np.int32)

    # Extract z_prev for test rows in df_all_sorted (must match df_test order!)
    # Re-map: df_test rows are in some order. We need z_prev in df_test's order.
    df_test_with_idx = df_test.copy()
    df_test_with_idx["_test_pos"] = np.arange(len(df_test))
    # Merge to get z_prev per test position
    # df_all_sorted contains test rows with the same (user_id, time_local).
    # Pick the test-only subset, sorted by user/time, with z_prev attached, then
    # restore the original df_test order.
    test_mask = df_all_sorted["_split"].values == "test"
    test_subset = df_all_sorted[test_mask].copy()
    test_subset["_z_prev"] = z_prev_full[test_mask]
    # Merge back by (user_id, time_local) to df_test order
    # Use a dict keyed by (user_id, time_local) → z_prev to avoid duplicate
    # explosion on merge. Multiple test rows at the same (u,t) are rare but
    # do occur on Bangkok / Tokyo; we de-duplicate by keeping first.
    test_subset_dedup = test_subset.drop_duplicates(
        subset=["user_id", "time_local"], keep="first")
    zp_lookup = dict(zip(
        zip(test_subset_dedup["user_id"].values,
              test_subset_dedup["time_local"].values),
        test_subset_dedup["_z_prev"].values,
    ))
    z_prev_test = np.array([
        zp_lookup.get((int(u), t), -1)
        for u, t in zip(df_test["user_id"].values, df_test["time_local"].values)
    ], dtype=np.int32)

    # Apply boundary_disambiguate
    r = membership_test.copy()
    r_tilde = boundary_disambiguate(r, T, z_prev_test)

    # Recognition F1 on BOUNDARY cases
    # The "true" recognition target = z_test (the core label, deterministic)
    # BEFORE: argmax of r (for boundary cases, uniform — tie-break by lowest k)
    # AFTER: argmax of r̃ (transition-prior-weighted)
    def _argmax_tiebreak_low(M):
        # numpy argmax already picks lowest index on ties
        return M.argmax(axis=1)

    y_true_b = z_test[isb_test]
    if y_true_b.size == 0:
        print(f"  no boundary cases — skipping recognition F1")
        return {"city": city, "n_boundary": 0,
                  "recognition_F1_before": None, "recognition_F1_after": None,
                  "delta_F1_boundary": None}
    y_before = _argmax_tiebreak_low(r[isb_test])
    y_after = _argmax_tiebreak_low(r_tilde[isb_test])
    f1_before = _macro_f1(y_true_b, y_before, K_sit)
    f1_after  = _macro_f1(y_true_b, y_after, K_sit)
    delta_b = f1_after - f1_before
    acc_before = float((y_before == y_true_b).mean())
    acc_after = float((y_after == y_true_b).mean())

    print(f"  n_boundary={int(isb_test.sum()):,}/{len(z_test):,}  "
          f"recognition F1 BEFORE={f1_before:.4f}  AFTER={f1_after:.4f}  "
          f"Δ={delta_b:+.4f}  (acc {acc_before:.3f} → {acc_after:.3f})")

    # Do-no-harm check: macro-F1 over ALL test rows (boundary + core)
    y_before_all = _argmax_tiebreak_low(r)
    y_after_all = _argmax_tiebreak_low(r_tilde)
    f1_all_before = _macro_f1(z_test, y_before_all, K_sit)
    f1_all_after  = _macro_f1(z_test, y_after_all, K_sit)
    print(f"  do-no-harm (ALL rows): F1 BEFORE={f1_all_before:.4f}  "
          f"AFTER={f1_all_after:.4f}  Δ={f1_all_after-f1_all_before:+.4f}")

    # Honest report of future-prediction signal from Stage C
    proj_path = REPO_ROOT / "outputs" / city / "xsage" / "projection" / "verdict.json"
    if proj_path.exists():
        proj = json.loads(proj_path.read_text())
        future_dF1 = float(proj.get("delta_F1", 0.0))
        n10 = int(proj.get("mcnemar_T_only_correct", 0))
        n01 = int(proj.get("mcnemar_time_only_correct", 0))
        mc_p = float(proj.get("mcnemar_p_value", float("nan")))
    else:
        future_dF1 = float("nan"); n10 = n01 = 0; mc_p = float("nan")

    out_dir = REPO_ROOT / "outputs_multicity" / city
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "city": city,
        "K_sit": K_sit,
        "n_test": int(len(z_test)),
        "n_boundary": int(isb_test.sum()),
        "boundary_share": float(isb_test.mean()),
        "recognition_F1_boundary_before": f1_before,
        "recognition_F1_boundary_after": f1_after,
        "delta_F1_boundary": delta_b,
        "boundary_acc_before": acc_before,
        "boundary_acc_after": acc_after,
        "macro_F1_all_before": f1_all_before,
        "macro_F1_all_after": f1_all_after,
        "do_no_harm_delta": f1_all_after - f1_all_before,
        "future_pred_delta_F1": future_dF1,
        "future_pred_mcnemar_T_correct": n10,
        "future_pred_mcnemar_time_correct": n01,
        "future_pred_mcnemar_p": mc_p,
        "future_pred_macro_F1_favors_T": future_dF1 > 0,
        "future_pred_mcnemar_favors_T": n10 > n01,
    }
    (out_dir / "projection_disambiguation.json").write_text(
        json.dumps(payload, indent=2, default=str))
    pd.DataFrame([payload]).to_csv(
        out_dir / "projection_disambiguation.csv", index=False)
    md = [f"# {city} — projection as disambiguator (G4)", "",
            f"K={K_sit}  n_test={payload['n_test']:,}  "
            f"n_boundary={payload['n_boundary']:,} ({payload['boundary_share']*100:.1f}%)",
            "",
            "## Boundary-case recognition F1 (BEFORE vs AFTER feedback)",
            "",
            f"| | macro-F1 | accuracy |",
            f"|---|---:|---:|",
            f"| BEFORE feedback (r argmax) | {f1_before:.4f} | {acc_before:.4f} |",
            f"| AFTER feedback (r̃ = r ⊙ T[z_prev,:]) | {f1_after:.4f} | {acc_after:.4f} |",
            f"| **Δ** | **{delta_b:+.4f}** | **{acc_after-acc_before:+.4f}** |",
            "",
            "## Do-no-harm (macro-F1 over ALL test rows)",
            "",
            f"| | macro-F1 |",
            f"|---|---:|",
            f"| BEFORE | {f1_all_before:.4f} |",
            f"| AFTER | {f1_all_after:.4f} |",
            f"| **Δ** | **{f1_all_after-f1_all_before:+.4f}** |",
            "",
            "## Honest future-prediction read (from Stage C)",
            "",
            f"- macro-F1 (T-based vs time-only): **ΔF1 = {future_dF1:+.4f}**",
            f"  → favors {'T-based' if future_dF1 > 0 else 'time-only'}",
            f"- McNemar paired: T-only-correct = {n10:,}, time-only-correct = {n01:,}, p = {mc_p:.2e}",
            f"  → favors {'T-based' if n10 > n01 else 'time-only'}",
            f"- **Honest read**: macro-F1 and McNemar "
              f"{'AGREE' if (future_dF1 > 0) == (n10 > n01) else 'DISAGREE'}",
            ]
    (out_dir / "projection_disambiguation.md").write_text("\n".join(md))
    return payload


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    out_root = REPO_ROOT / "outputs_multicity"
    out_root.mkdir(parents=True, exist_ok=True)

    # G1 tidy
    g1_payload = g1_tidy(CITIES,
                            out_dir=REPO_ROOT / "outputs_multicity" / "G1_global_rerank")

    # G2 — per city
    print(f"\n{'='*60}\nG2 — fairness coherence\n{'='*60}")
    g2_rows = []
    for c in CITIES:
        r = g2_one_city(c)
        if r:
            g2_rows.append(r)

    # G2 summary
    md = ["# G2 — Multi-city fairness coherence (cross-metric)", ""]
    md.append("| city | sinks_LT | sinks_KL | sinks_Gini | coherent? |")
    md.append("|---|---|---|---|:---:|")
    for r in g2_rows:
        md.append(f"| {r['city']} | {r['sinks_LT']} | {r['sinks_KL']} | "
                  f"{r['sinks_Gini']} | "
                  f"{'✅' if r['coherent'] else '❌'} |")
    (out_root / "G2_fairness_coherence_summary.md").write_text("\n".join(md))

    # G3
    print(f"\n{'='*60}\nG3 — multi-city stat hardening\n{'='*60}")
    g3_rows = []
    for c in CITIES:
        r = g3_one_city(c)
        if r:
            g3_rows.append(r)
    g3_finalize(g3_rows, out_root)

    # G4 — per city
    print(f"\n{'='*60}\nG4 — projection as disambiguator\n{'='*60}")
    g4_rows = []
    for c in CITIES:
        try:
            r = g4_one_city(c)
            g4_rows.append(r)
        except Exception as e:
            print(f"  [{c}] G4 failed: {e}")

    # G4 summary
    md = ["# G4 — Projection as disambiguator (multi-city)", "",
            "| city | n_boundary | F1_boundary_before | F1_boundary_after | **Δ** | do-no-harm Δ (all) | future_ΔF1 | McNemar p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in g4_rows:
        if not r:
            continue
        nbnd = r.get('n_boundary', 0) or 0
        if nbnd == 0:
            md.append(f"| {r['city']} | 0 | n/a | n/a | n/a | "
                      f"n/a | "
                      f"{r['future_pred_delta_F1']:+.3f} | "
                      f"{r['future_pred_mcnemar_p']:.2e} |")
        else:
            md.append(f"| {r['city']} | {r['n_boundary']:,} | "
                      f"{r['recognition_F1_boundary_before']:.4f} | "
                      f"{r['recognition_F1_boundary_after']:.4f} | "
                      f"**{r['delta_F1_boundary']:+.4f}** | "
                      f"{r['do_no_harm_delta']:+.4f} | "
                      f"{r['future_pred_delta_F1']:+.3f} | "
                      f"{r['future_pred_mcnemar_p']:.2e} |")
    (out_root / "G4_projection_disambiguation_summary.md").write_text("\n".join(md))

    # final summary print
    print(f"\n{'='*60}\nFINAL\n{'='*60}")
    print(f"G1 identity check: {g1_payload['identity']}")
    print(f"G2 coherence: {[r['coherent'] for r in g2_rows]}")
    print(f"G3 ΔR@20 reject (Holm@0.05): "
          f"{[r['city']+'='+str(r['delta_R20_holm_reject']) for r in g3_rows]}")
    print(f"G4 Δ recognition F1 (boundary): "
          f"{[r['city']+'='+str(round(r['delta_F1_boundary'],4)) if r.get('delta_F1_boundary') is not None else r['city']+'=n/a' for r in g4_rows if r]}")


if __name__ == "__main__":
    sys.exit(main() or 0)
