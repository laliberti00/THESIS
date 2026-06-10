"""Round-2 Track 3 — explainability artifacts.

Produces, per city, three things the brief asked for explicitly:

  3.1  ``situations/contrib_functions.png`` — bar chart per attribute of
       the learned θ_b values (contribution-function leaf outputs).
  3.2  ``situations/situation_cards.md``    — one card per situation:
       auto name, top context bins, top attractors, top observed fine
       categories, 3 sample test requests, readable biases b^{(k)}.
  3.3  ``recommendation/explanations.csv``  — 30 sample test requests
       with the served POI, situation z (core/boundary), and a one-line
       reason derived directly from the readable bias.
  3.4  the transition heatmap from Stage C is already in place; the
       brief's "upgrade" is just renaming if needed.
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

from pipeline.step02_models.xsage.l1_perception import fit_contribution_functions
from pipeline.step02_models.xsage.orchestrator import (DEFAULT_ATTRIBUTES,
                                                            _load_city)
from pipeline.step02_models.xsage.recommendation import fit_situation_biases


def plot_contrib_functions(city: str, intent_mode: str = "hard") -> None:
    """3.1 — one bar chart per learned attribute showing the leaf θ values."""
    out_root = REPO_ROOT / "outputs" / city / (
        "xsage" if intent_mode == "hard" else f"xsage_intent_{intent_mode}")
    out_dir = out_root / "situations"
    ds = _load_city(city)
    contrib = fit_contribution_functions(ds["df_train"], ds["macro_to_idx"],
                                            attributes=DEFAULT_ATTRIBUTES)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(DEFAULT_ATTRIBUTES)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows))
    axes = np.array(axes).reshape(-1)
    for ai, a in enumerate(DEFAULT_ATTRIBUTES):
        ax = axes[ai]
        leaf_theta = contrib.leaf_theta[a]
        ids = sorted(leaf_theta.keys())
        vals = [leaf_theta[i] for i in ids]
        ax.bar(range(len(vals)), vals, color="steelblue")
        ax.set_title(f"{a} (n_leaves={len(vals)})", fontsize=10)
        ax.set_ylabel("θ (1 - normalised leaf entropy)")
        ax.set_ylim(0, max(0.05, max(vals) * 1.1))
        ax.set_xlabel("leaf id (sorted)")
        ax.tick_params(axis="both", labelsize=8)
    for ai in range(len(DEFAULT_ATTRIBUTES), len(axes)):
        axes[ai].axis("off")
    fig.suptitle(f"{city} — learned contribution functions θ per attribute",
                   fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "contrib_functions.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out_dir / 'contrib_functions.png'}")


def situation_cards(city: str, intent_mode: str = "hard",
                       n_per_situation: int = 3) -> None:
    """3.2 — one .md card per situation."""
    out_root = REPO_ROOT / "outputs" / city / (
        "xsage" if intent_mode == "hard" else f"xsage_intent_{intent_mode}")
    sit_dir = out_root / "situations"
    archetypes = pd.read_csv(sit_dir / "archetypes.csv")
    examples = pd.read_csv(sit_dir / "examples.csv")
    fit = np.load(sit_dir / "fit.npz", allow_pickle=True)

    ds = _load_city(city)
    n_macros = ds["n_macros"]
    # Per-situation cat_macro biases (log-odds w.r.t. global)
    z_train = np.asarray(fit["core_label_train"]).astype(np.int32)
    z_val = np.asarray(fit["core_label_val"]).astype(np.int32)
    df_train = ds["df_train"]; df_val = ds["df_val"]
    macro_to_idx = ds["macro_to_idx"]
    macro_train = np.array([macro_to_idx[m] for m in df_train["cat_macro"]],
                            dtype=np.int64)
    macro_val = np.array([macro_to_idx[m] for m in df_val["cat_macro"]],
                          dtype=np.int64)
    z_tv = np.concatenate([z_train, z_val])
    macro_tv = np.concatenate([macro_train, macro_val])
    K = int(z_tv.max() + 1)
    biases = fit_situation_biases(z_tv, macro_tv, K=K, n_macros=n_macros,
                                     lam=50.0)
    idx2macro = {v: k for k, v in macro_to_idx.items()}

    lines = [f"# {city} — situation cards (intent_mode = {intent_mode})\n"]
    for k in range(K):
        row = archetypes[archetypes["id"] == k]
        if row.empty: continue
        r = row.iloc[0]
        # Top-3 boosted and bottom-3 suppressed macros for this situation
        b_k = biases[k]
        idx_high = np.argsort(b_k)[::-1][:3]
        idx_low = np.argsort(b_k)[:3]
        boosts = ", ".join(f"{idx2macro[int(i)]} {b_k[int(i)]:+.2f}" for i in idx_high)
        suppress = ", ".join(f"{idx2macro[int(i)]} {b_k[int(i)]:+.2f}" for i in idx_low)
        # Pull a few sample examples
        ex = examples[examples["situation_id"] == k].head(n_per_situation)

        lines.append(f"\n## Situation s{k} — {r['name']}\n")
        lines.append(f"- **n_core** = {int(r['n_core'])}, **n_boundary** = {int(r['n_boundary'])}\n")
        lines.append(f"- **top context bins**: {r['top_context_bins']}\n")
        lines.append(f"- **top attractors**: {r['top_attractors']}\n")
        lines.append(f"- **observed target macros (top 3)**: {r['top_target_cat_macros']}\n")
        lines.append(f"- **biases (log-odds vs global)**:\n")
        lines.append(f"  - boosts: {boosts}\n")
        lines.append(f"  - suppresses: {suppress}\n")
        if not ex.empty:
            lines.append(f"\n*Sample test requests:*\n\n")
            for _, e in ex.iterrows():
                lines.append(f"- u={int(e['u_idx'])} at {e['time_local']} "
                              f"({'boundary' if e['is_boundary'] else 'core'}) "
                              f"→ visits venue {e['next_venue_id'][:12]}... "
                              f"({e['next_cat_macro']} / {e['next_cat_fine']})\n")
    out_path = sit_dir / "situation_cards.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"  wrote {out_path}")


def explanations(city: str, intent_mode: str = "hard", n_samples: int = 30,
                   kappa: float = 0.5) -> None:
    """3.3 — per-recommendation explanations for ~30 sample test requests."""
    out_root = REPO_ROOT / "outputs" / city / (
        "xsage" if intent_mode == "hard" else f"xsage_intent_{intent_mode}")
    sit_dir = out_root / "situations"
    rec_dir = out_root / "recommendation_additive"
    if not rec_dir.exists():
        rec_dir = out_root / "recommendation"
    rec_dir.mkdir(parents=True, exist_ok=True)
    fit = np.load(sit_dir / "fit.npz", allow_pickle=True)

    ds = _load_city(city)
    macro_to_idx = ds["macro_to_idx"]; idx2macro = {v: k for k, v in macro_to_idx.items()}
    df_test = ds["df_test"]; n_test = len(df_test)
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)
    membership_test = np.asarray(fit["membership_test"]).astype(np.float32)
    K_sit = int(z_test.max() + 1)
    n_macros = ds["n_macros"]; n_items = ds["n_items"]

    # Per-situation bias (log-odds)
    z_train = np.asarray(fit["core_label_train"]).astype(np.int32)
    z_val = np.asarray(fit["core_label_val"]).astype(np.int32)
    df_train = ds["df_train"]; df_val = ds["df_val"]
    macro_train = np.array([macro_to_idx[m] for m in df_train["cat_macro"]],
                            dtype=np.int64)
    macro_val = np.array([macro_to_idx[m] for m in df_val["cat_macro"]],
                          dtype=np.int64)
    z_tv = np.concatenate([z_train, z_val])
    macro_tv = np.concatenate([macro_train, macro_val])
    biases = fit_situation_biases(z_tv, macro_tv, K=K_sit, n_macros=n_macros,
                                     lam=50.0)

    # Score B_blind, find served item per request
    from pipeline.step02_models.xsage.backbone import excluded_mask, load_or_refit
    scores_uitem = load_or_refit(city, model_name="FM")
    u_test = df_test["u_idx"].values.astype(np.int64)
    scores_per_req = scores_uitem[u_test]
    excl = excluded_mask(city, n_items)

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_test, size=n_samples, replace=False)

    # Build item → cat_macro lookup (use train+val+test consensus)
    counts_m = pd.concat([df_train, df_val, df_test], ignore_index=True) \
                 .groupby(["i_idx", "cat_macro"]).size().reset_index(name="n")
    best_m = counts_m.sort_values(["i_idx", "n"], ascending=[True, False]) \
                       .drop_duplicates("i_idx", keep="first")
    item_to_macro = {int(r["i_idx"]): r["cat_macro"] for _, r in best_m.iterrows()}
    counts_f = pd.concat([df_train, df_val, df_test], ignore_index=True) \
                 .groupby(["i_idx", "cat_fine"]).size().reset_index(name="n")
    best_f = counts_f.sort_values(["i_idx", "n"], ascending=[True, False]) \
                       .drop_duplicates("i_idx", keep="first")
    item_to_fine = {int(r["i_idx"]): r["cat_fine"] for _, r in best_f.iterrows()}

    rows = []
    for q in sample_idx:
        u = int(u_test[q])
        z = int(z_test[q])
        is_b = bool(isb_test[q])
        # Apply additive nudge (κ=0.5 by default)
        from pipeline.step02_models.xsage.recommendation import (
            additive_combine_scores, fit_situation_biases_z)
        biases_z = fit_situation_biases_z(z_tv, macro_tv, K=K_sit,
                                              n_macros=n_macros, lam=50.0)
        item_macro_full = np.array(
            [macro_to_idx[item_to_macro[i]] for i in range(n_items)],
            dtype=np.int64)
        gamma = np.array([1.0 / max((membership_test[q] > 0).sum(), 1)
                            if is_b else 1.0], dtype=np.float32)
        # score this single request
        s_one = scores_per_req[q:q + 1].copy()
        nudged = additive_combine_scores(s_one,
                                             membership_test[q:q + 1],
                                             biases_z,
                                             item_macro_full,
                                             kappa=kappa,
                                             gamma_per_request=gamma)
        cols = excl.indices[excl.indptr[u]:excl.indptr[u + 1]]
        if len(cols): nudged[0, cols] = -np.inf
        top1 = int(np.argmax(nudged[0]))
        # Build the reason
        b_k = biases[z]
        top_boosted = np.argsort(b_k)[::-1][:2]
        top_macro_names = [idx2macro[int(i)] for i in top_boosted]
        topb_vals = [f"{b_k[int(i)]:+.2f}" for i in top_boosted]
        rec_macro = item_to_macro.get(top1, "?")
        rec_fine = item_to_fine.get(top1, "?")
        reason = (f"In situation s{z} ({'boundary' if is_b else 'core'}) the "
                    f"biases boost {top_macro_names[0]} ({topb_vals[0]}) and "
                    f"{top_macro_names[1]} ({topb_vals[1]}); served POI is in "
                    f"{rec_macro}.")
        rows.append({
            "u_idx": u,
            "time_local": str(df_test.iloc[q]["time_local"]),
            "z": z, "is_boundary": is_b,
            "served_venue_id": str(df_test.iloc[q]["venue_id"]),
            "served_top1_item_idx": top1,
            "served_top1_macro": rec_macro,
            "served_top1_fine": rec_fine,
            "actual_target_macro": str(df_test.iloc[q]["cat_macro"]),
            "reason": reason,
        })
    df = pd.DataFrame(rows)
    out_path = rec_dir / "explanations.csv"
    df.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", choices=["NYC", "TKY", "both"], default="both")
    parser.add_argument("--intent-mode", choices=["hard", "all"], default="hard")
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--kappa", type=float, default=0.5)
    args = parser.parse_args()

    cities = ["NYC", "TKY"] if args.city == "both" else [args.city]
    for city in cities:
        print(f"\n>>> Explainability artefacts for {city} (mode={args.intent_mode})")
        plot_contrib_functions(city, intent_mode=args.intent_mode)
        situation_cards(city, intent_mode=args.intent_mode)
        explanations(city, intent_mode=args.intent_mode,
                      n_samples=args.n_samples, kappa=args.kappa)
    return 0


if __name__ == "__main__":
    sys.exit(main())
