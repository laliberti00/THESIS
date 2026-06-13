"""Round-3 B5 — anatomy of the fairness intervention.

For each city, at the val-selected knee operating point:
  - NYC: mask-mode situations (sinks {6, 7}), knee κ_fair = 1.0  (A1bis)
  - TKY: keep-mode situations (sinks {4, 5}), knee κ_fair = 2.0  (A4)

Joins B_blind test top-20 with the fairness-re-ranked top-20 and reports:

  1. Touched share + identity check (sink ∩ core).
  2. Per touched list: # items swapped, rank depth of exits/entries.
  3. Who enters / exits: popularity percentile, category profile.
  4. Per-situation breakdown.
  5. Three worked examples per city.
  6. Contrast row: B_full / global re-rank / X-SAGE.

Outputs: outputs/round3/B5/<city>/{intervention_anatomy.csv,
                                    depth_profile.png, examples.md,
                                    summary.md}.
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
from pipeline.step02_models.xsage.metrics import long_tail_groups, topk_from_scores
from pipeline.step02_models.xsage.orchestrator import _load_city


K_TOP = 20
SHORT_HEAD = 0.20


CONFIG = {
    "NYC": {
        "sit_dir": "outputs/NYC/xsage_transit_mask/situations",
        "sinks": [6, 7],
        "knee_kappa": 1.0,
        "mode_label": "mask",
    },
    "TKY": {
        "sit_dir": "outputs/TKY/xsage/situations",
        "sinks": [4, 5],
        "knee_kappa": 2.0,
        "mode_label": "keep (round-2)",
    },
}


def _topk(scores: np.ndarray, exclude: sps.csr_matrix, users: np.ndarray,
            K: int = K_TOP) -> np.ndarray:
    n = scores.shape[0]
    out = np.zeros((n, K), dtype=np.int32)
    for b in range(n):
        u = int(users[b])
        s = scores[b].copy()
        cols = exclude.indices[exclude.indptr[u]:exclude.indptr[u + 1]]
        if len(cols):
            s[cols] = -np.inf
        out[b] = topk_from_scores(s, K)
    return out


def _r20(scores, targets, exclude, users):
    hits = 0; n = scores.shape[0]
    for b in range(n):
        u = int(users[b])
        s = scores[b].copy()
        cols = exclude.indices[exclude.indptr[u]:exclude.indptr[u + 1]]
        if len(cols):
            s[cols] = -np.inf
        if (s > s[int(targets[b])]).sum() < K_TOP:
            hits += 1
    return hits / max(1, n)


def run_city(city: str) -> dict:
    cfg = CONFIG[city]
    out_dir = REPO_ROOT / "outputs" / "round3" / "B5" / city
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> B5 on {city} ({cfg['mode_label']}, sinks={cfg['sinks']}, "
          f"κ={cfg['knee_kappa']})")

    fit = np.load(REPO_ROOT / cfg["sit_dir"] / "fit.npz", allow_pickle=True)
    ds = _load_city(city)
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)
    df_test = ds["df_test"]
    u_test = df_test["u_idx"].values.astype(np.int64)
    i_target = df_test["i_idx"].values.astype(np.int64)
    n_items = ds["n_items"]

    scores_blind_uitem = load_or_refit(city, model_name="FM")
    scores_blind = scores_blind_uitem[u_test]
    excl = excluded_mask(city, n_items)
    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(axis=0)).ravel()
    _, G1_mask = long_tail_groups(pop, short_head_share=SHORT_HEAD)
    boost = G1_mask.astype(np.float32)

    sink_mask = np.isin(z_test, cfg["sinks"])
    core_sink = sink_mask & ~isb_test
    print(f"  test n={len(u_test)}  sink share={sink_mask.mean():.3f}  "
          f"core∩sink share={core_sink.mean():.3f}")

    top_off = _topk(scores_blind, excl, u_test)
    scores_on = scores_blind.copy()
    scores_on[np.where(core_sink)[0]] += cfg["knee_kappa"] * boost[None, :]
    top_on = _topk(scores_on, excl, u_test)

    # (1) Touched share + identity
    changed = np.zeros(len(u_test), dtype=bool)
    for b in range(len(u_test)):
        if not np.array_equal(top_off[b], top_on[b]):
            inter = len(np.intersect1d(top_off[b], top_on[b], assume_unique=False))
            if inter < K_TOP:
                changed[b] = True
    touched_share = float(changed.mean())
    # Identity check: changed should equal core_sink (by construction)
    identity_match = bool((changed == core_sink).all())
    only_core_sink_not_changed = int((core_sink & ~changed).sum())
    only_changed_not_core_sink = int((~core_sink & changed).sum())
    print(f"  touched share = {touched_share:.4f}  "
          f"identity (changed == core_sink): {identity_match}")
    if not identity_match:
        print(f"    core_sink ∧ ¬changed = {only_core_sink_not_changed}  "
              f"¬core_sink ∧ changed = {only_changed_not_core_sink}")

    # (2) Per touched list: # items swapped + rank depth of exits / entries
    swap_counts = []; entries_ranks = []; exits_ranks = []
    overlap_at_K = {k: 0 for k in (1, 5, 10, 20)}
    n_touched = int(changed.sum())
    for b in np.where(changed)[0]:
        off = list(top_off[b]); on = list(top_on[b])
        set_off = set(map(int, off)); set_on = set(map(int, on))
        n_swap = K_TOP - len(set_off & set_on)
        swap_counts.append(n_swap)
        for k_eval in overlap_at_K:
            inter = len(set(off[:k_eval]) & set(on[:k_eval]))
            overlap_at_K[k_eval] += inter / k_eval
        for r, it in enumerate(off):
            if int(it) not in set_on:
                exits_ranks.append(r + 1)
        for r, it in enumerate(on):
            if int(it) not in set_off:
                entries_ranks.append(r + 1)
    if n_touched > 0:
        overlap_at_K = {k: v / n_touched for k, v in overlap_at_K.items()}

    # (3) Who enters / exits: popularity percentile + categories
    pop_pct = np.argsort(np.argsort(pop)) / max(len(pop) - 1, 1)
    df_all = pd.concat([ds["df_train"], ds["df_val"], ds["df_test"]],
                         ignore_index=True)
    cm = df_all.groupby(["i_idx", "cat_macro"]).size().reset_index(name="n")
    item_macro = {int(r["i_idx"]): r["cat_macro"]
                    for _, r in cm.sort_values(["i_idx", "n"], ascending=[True, False])
                                       .drop_duplicates("i_idx", keep="first")
                                       .iterrows()}
    cf = df_all.groupby(["i_idx", "cat_fine"]).size().reset_index(name="n")
    item_fine = {int(r["i_idx"]): r["cat_fine"]
                   for _, r in cf.sort_values(["i_idx", "n"], ascending=[True, False])
                                     .drop_duplicates("i_idx", keep="first")
                                     .iterrows()}

    # Aggregate enter/exit profile
    enters_pop, exits_pop = [], []
    enters_macro = []; exits_macro = []
    for b in np.where(changed)[0]:
        set_off = set(map(int, top_off[b])); set_on = set(map(int, top_on[b]))
        for it in set_on - set_off:
            enters_pop.append(pop_pct[it])
            enters_macro.append(item_macro.get(it, "?"))
        for it in set_off - set_on:
            exits_pop.append(pop_pct[it])
            exits_macro.append(item_macro.get(it, "?"))
    enter_pop_mean = float(np.mean(enters_pop)) if enters_pop else 0.0
    exit_pop_mean = float(np.mean(exits_pop)) if exits_pop else 0.0
    enter_macro_dist = pd.Series(enters_macro).value_counts(normalize=True).to_dict() if enters_macro else {}
    exit_macro_dist = pd.Series(exits_macro).value_counts(normalize=True).to_dict() if exits_macro else {}

    # (4) Per-situation breakdown
    per_sit_rows = []
    for s in cfg["sinks"]:
        m = (z_test == s) & ~isb_test  # core sink of this situation
        if not m.any():
            continue
        bs = np.where(m & changed)[0]
        per_sit_rows.append({
            "situation": int(s),
            "n_core_requests": int(m.sum()),
            "n_touched": int(len(bs)),
            "fraction_touched": float(len(bs) / max(m.sum(), 1)),
            "avg_swap": float(np.mean([K_TOP - len(set(top_off[b]) & set(top_on[b]))
                                              for b in bs])) if len(bs) else 0.0,
        })
    pd.DataFrame(per_sit_rows).to_csv(out_dir / "per_situation_anatomy.csv", index=False)

    # (5) Three worked examples
    rng = np.random.default_rng(42)
    example_idx = []
    for s in cfg["sinks"]:
        candidates = np.where((z_test == s) & ~isb_test & changed)[0]
        if len(candidates) == 0: continue
        example_idx.extend(rng.choice(candidates,
                                          size=min(2, len(candidates)),
                                          replace=False).tolist())
    examples_md = [f"# B5 worked examples — {city}\n\n",
                    f"_Operating point: {cfg['mode_label']} situations, sinks={cfg['sinks']}, "
                    f"knee κ_fair = {cfg['knee_kappa']}_\n\n"]
    for q in example_idx[:3]:
        u = int(u_test[q]); z = int(z_test[q]); is_b = bool(isb_test[q])
        off_list = list(top_off[q]); on_list = list(top_on[q])
        set_off = set(map(int, off_list)); set_on = set(map(int, on_list))
        examples_md.append(f"## Example — u={u} at {df_test.iloc[q]['time_local']}\n\n")
        examples_md.append(f"- Situation: **s{z}**  ({'boundary' if is_b else 'core'})\n")
        examples_md.append(f"- Target macro / fine: {df_test.iloc[q]['cat_macro']} / "
                              f"{df_test.iloc[q]['cat_fine']}\n")
        examples_md.append(f"- Top-20 changed at ranks: "
                              f"{[i+1 for i, x in enumerate(on_list) if int(x) not in set_off]}\n\n")
        examples_md.append("| rank | OFF item (pop %ile / macro) | ON item (pop %ile / macro) |\n")
        examples_md.append("|---:|---|---|\n")
        for r in range(K_TOP):
            o = int(off_list[r]); n = int(on_list[r])
            ot = "" if int(o) == int(n) else (
                f"{item_macro.get(o, '?')[:18]} / {pop_pct[o]:.2f}")
            nt = "" if int(o) == int(n) else (
                f"{item_macro.get(n, '?')[:18]} / {pop_pct[n]:.2f}")
            marker = "" if int(o) == int(n) else " ★"
            if int(o) == int(n):
                examples_md.append(f"| {r+1} | (unchanged) | (unchanged) |\n")
            else:
                examples_md.append(f"| {r+1} | item={o}, {ot} | item={n}, {nt} |{marker}\n")
        n_entries = int(K_TOP - len(set_off & set_on))
        long_tail_in = sum(1 for x in (set_on - set_off) if G1_mask[x])
        examples_md.append(f"\n**Reason**: u in s{z} (core); biases boost long-tail; "
                              f"{n_entries} items entered top-20, of which {long_tail_in} "
                              f"are long-tail.\n\n")
    (out_dir / "examples.md").write_text("".join(examples_md), encoding="utf-8")

    # (6) Contrast row: B_full and global re-rank touch rates
    bfull_path = REPO_ROOT / "outputs" / city / "xsage" / "backbone" / "Bfull.scores.npy"
    bfull_scores = np.load(bfull_path) if bfull_path.exists() else None
    if bfull_scores is not None:
        top_bfull = _topk(bfull_scores, excl, u_test)
        bfull_changed = np.array([not np.array_equal(top_off[b], top_bfull[b])
                                       for b in range(len(u_test))])
        bfull_touched_share = float(bfull_changed.mean())
    else:
        bfull_touched_share = float("nan")

    # Global re-rank at same κ_fair as fairness boost but applied to ALL requests
    scores_global = scores_blind + cfg["knee_kappa"] * boost[None, :]
    top_global = _topk(scores_global, excl, u_test)
    global_changed = np.array([not np.array_equal(top_off[b], top_global[b])
                                   for b in range(len(u_test))])
    global_touched_share = float(global_changed.mean())

    # Headline summary
    summary = {
        "city": city, "mode": cfg["mode_label"], "sinks": cfg["sinks"],
        "knee_kappa": cfg["knee_kappa"],
        "n_test_requests": int(len(u_test)),
        "touched_share_xsage": touched_share,
        "n_touched": n_touched,
        "identity_check_passes": identity_match,
        "core_sink_not_changed": only_core_sink_not_changed,
        "changed_not_core_sink": only_changed_not_core_sink,
        "avg_swap_count_per_touched_list": float(np.mean(swap_counts)) if swap_counts else 0.0,
        "median_entry_rank": float(np.median(entries_ranks)) if entries_ranks else 0.0,
        "median_exit_rank": float(np.median(exits_ranks)) if exits_ranks else 0.0,
        "overlap_at_K": overlap_at_K,
        "enter_pop_percentile_mean": enter_pop_mean,
        "exit_pop_percentile_mean": exit_pop_mean,
        "enter_macro_top3": dict(list(enter_macro_dist.items())[:3]),
        "exit_macro_top3": dict(list(exit_macro_dist.items())[:3]),
        "contrast_touched_share": {
            "X-SAGE (this run)": touched_share,
            "B_full (no accounting)": bfull_touched_share,
            "global re-rank (no gate)": global_touched_share,
        },
    }
    (out_dir / "summary.md").write_text(
        "# B5 — anatomy of the intervention\n\n"
        f"## {city} headline\n\n"
        f"- mode: **{cfg['mode_label']}**, sinks: {cfg['sinks']}, "
        f"knee κ = {cfg['knee_kappa']}\n"
        f"- touched share: **{touched_share:.4f}** (vs B_full "
        f"{bfull_touched_share:.4f}, vs global {global_touched_share:.4f})\n"
        f"- identity check (changed == core∩sink): **{identity_match}**\n"
        f"- avg items swapped per touched list: {summary['avg_swap_count_per_touched_list']:.1f}\n"
        f"- median entry rank: {summary['median_entry_rank']:.1f}, "
        f"median exit rank: {summary['median_exit_rank']:.1f}\n"
        f"- overlap@K: " + ", ".join(f"K={k}: {v:.3f}" for k, v in overlap_at_K.items()) + "\n"
        f"- enter pop %ile mean: {enter_pop_mean:.3f}, exit pop %ile mean: {exit_pop_mean:.3f}\n"
        f"- enter macro top-3: {summary['enter_macro_top3']}\n"
        f"- exit macro top-3: {summary['exit_macro_top3']}\n\n"
        f"## Contrast row\n\n"
        f"| variant | touched share | accounting |\n"
        f"|---|---|---|\n"
        f"| **X-SAGE** at knee κ={cfg['knee_kappa']} | {touched_share:.4f} | per-situation, sink-gated, ledger in CSV |\n"
        f"| B_full (tuned) | {bfull_touched_share:.4f} | none (black-box context FM) |\n"
        f"| global re-rank κ={cfg['knee_kappa']} | {global_touched_share:.4f} | none (no targeting) |\n",
        encoding="utf-8")

    # Detailed per-request CSV
    csv_rows = []
    for b in range(len(u_test)):
        csv_rows.append({
            "u_idx": int(u_test[b]),
            "time_local": str(df_test.iloc[b]["time_local"]),
            "z": int(z_test[b]),
            "is_boundary": bool(isb_test[b]),
            "is_sink": bool(sink_mask[b]),
            "is_core_sink": bool(core_sink[b]),
            "changed_topK": bool(changed[b]),
            "n_swap": int(K_TOP - len(set(top_off[b]) & set(top_on[b]))) if changed[b] else 0,
            "target_macro": str(df_test.iloc[b]["cat_macro"]),
        })
    pd.DataFrame(csv_rows).to_csv(out_dir / "intervention_anatomy.csv", index=False)

    # Depth profile PNG
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    if entries_ranks:
        ax.hist(entries_ranks, bins=np.arange(1, K_TOP + 2) - 0.5,
                  alpha=0.7, label="entries", color="tab:blue")
    if exits_ranks:
        ax.hist(exits_ranks, bins=np.arange(1, K_TOP + 2) - 0.5,
                  alpha=0.7, label="exits", color="tab:red")
    ax.set_xlabel("rank in top-20")
    ax.set_ylabel("count")
    ax.set_title(f"{city} — {cfg['mode_label']}, knee κ={cfg['knee_kappa']}: rank depth of entries vs exits")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / "depth_profile.png", dpi=140); plt.close(fig)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                              encoding="utf-8")
    print(f"  wrote {out_dir}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", choices=["NYC", "TKY", "both"], default="both")
    args = parser.parse_args()
    cities = ["NYC", "TKY"] if args.city == "both" else [args.city]
    for c in cities:
        run_city(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
