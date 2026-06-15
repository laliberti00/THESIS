"""Round-3 B11 — situation showcase.

Static figures + interactive HTML for the situations and the X-SAGE
explainability, both rendered from existing artefacts (Stage A, B5,
B7b, B8b, faithfulness). NO new experiments.

Outputs:
  outputs/round3/B11/<city>/cards/s<k>.{png,pdf}
  outputs/round3/B11/<city>/explainability/s<k>.{png,pdf}
  outputs/round3/B11/<city>/{cards_data.json, explainability_data.json}
  outputs/round3/B11/showcase.html             (self-contained)

Configs follow B8b naming exactly (paper-lead):
  NYC: mask-mode (sinks {6,7}); TKY: keep-mode (sinks {4,5}).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.xsage.backbone import excluded_mask, load_or_refit
from pipeline.step02_models.xsage.metrics import (long_tail_groups,
                                                       topk_from_scores)
from pipeline.step02_models.xsage.orchestrator import _load_city

from experiments.round3_b8b_robust_naming import (
    _name_situations_robust, _ensure_distinct, CONFIG as B8B_CONFIG,
    TAU_DOM,
)
from experiments.round3_b8_explainability import (
    MACRO_SHORT, N_ATTRIB,
)

# ----------------------------- constants -------------------------------------

K_TOP = 20
SHORT_HEAD = 0.20

CONFIG = {
    "NYC": {
        "sit_dir": "outputs/NYC/xsage_transit_mask/situations",
        "fairness_csv": "outputs/NYC/xsage_transit_mask/fairness/per_situation.csv",
        "b5_anatomy": "outputs/round3/B5/NYC/intervention_anatomy.csv",
        "sinks": [6, 7],
        "knee_kappa": 1.0,
        "mode_label": "mask",
    },
    "TKY": {
        "sit_dir": "outputs/TKY/xsage/situations",
        "fairness_csv": "outputs/TKY/xsage/fairness/per_situation.csv",
        "b5_anatomy": "outputs/round3/B5/TKY/intervention_anatomy.csv",
        "sinks": [4, 5],
        "knee_kappa": 2.0,
        "mode_label": "keep",
    },
}

DICT_PATH = REPO_ROOT / "config" / "situation_label_dictionary.json"


# ----------------------------- readable label --------------------------------

def _strip_distinctness(name: str) -> tuple[str, str]:
    """Return (head_no_fallback, intent_token).
    Example: 'Morning · Outdoors (+Food) @08h #0' →
             ('Morning', 'Outdoors')
             keeps the BASE tokens, drops the (+X), @HHh, #k.
    """
    # drop distinctness fallbacks
    base = re.sub(r"\s*\(\+[A-Za-z]+\)", "", name)
    base = re.sub(r"\s*@\d{2}h", "", base)
    base = re.sub(r"\s*#\d+", "", base)
    if " · " in base:
        head, intent = base.split(" · ", 1)
    else:
        head, intent = base, ""
    return head.strip(), intent.strip()


def readable_label(time_token: str, intent_token: str,
                     dict_blob: dict) -> tuple[str, str]:
    """Apply the published dictionary. Returns (label, source) where
    source ∈ {'single_intent', 'pair_intent', 'fallback'}."""
    key = f"{time_token}|{intent_token}"
    if key in dict_blob.get("single_intent", {}):
        return dict_blob["single_intent"][key], "single_intent"
    if key in dict_blob.get("pair_intent", {}):
        return dict_blob["pair_intent"][key], "pair_intent"
    # alt: if pair like "A/B", try reversed
    if "/" in intent_token:
        a, b = intent_token.split("/", 1)
        alt_key = f"{time_token}|{b}/{a}"
        if alt_key in dict_blob.get("pair_intent", {}):
            return dict_blob["pair_intent"][alt_key], "pair_intent"
    return "", "fallback"  # empty → use technical name


# ----------------------------- data assembly ---------------------------------

def assemble_city(city: str, dict_blob: dict) -> dict:
    """Load all artefacts and produce a dict of per-situation data."""
    cfg = CONFIG[city]
    fit = np.load(REPO_ROOT / cfg["sit_dir"] / "fit.npz", allow_pickle=True)
    prototypes = np.asarray(fit["prototypes"])
    core_label_train = np.asarray(fit["core_label_train"]).astype(np.int32)
    is_boundary_train = np.asarray(fit["is_boundary_train"]).astype(bool)
    attractors_mask = np.asarray(fit["attractors"]).astype(bool)
    K = prototypes.shape[0]

    ds = _load_city(city)
    n_macros = ds["n_macros"]
    idx_to_macro = ds["idx_to_macro"]
    df_train = ds["df_train"]
    df_test = ds["df_test"]
    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(axis=0)).ravel()
    G0_mask, G1_mask = long_tail_groups(pop, short_head_share=SHORT_HEAD)
    pop_pct = np.argsort(np.argsort(pop)) / max(len(pop) - 1, 1)
    item_macro = df_train.groupby("i_idx")["cat_macro"].first().to_dict()

    # Robust B8b names
    names_raw, traces = _name_situations_robust(
        prototypes, core_label_train, is_boundary_train,
        df_train, attractors_mask, idx_to_macro, n_macros)
    final_names, _ = _ensure_distinct(names_raw, traces, prototypes,
                                          n_macros, attractors_mask,
                                          idx_to_macro)

    # Fairness per-situation
    fairness = pd.read_csv(REPO_ROOT / cfg["fairness_csv"])
    fairness_all = fairness[fairness["split"] == "all"].set_index("situation")

    # Population baselines for the "high/low vs pop" signature
    pop_we_share = float(df_train["c_isweekend"].mean())
    pop_modal_hour = int(np.argmax(np.bincount(
        df_train["c_hour"].values.astype(int), minlength=24)))

    cards = []
    n_test = len(df_test)
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)

    for k in range(K):
        core_mask_tr = (core_label_train == k) & ~is_boundary_train
        bnd_mask_tr = (core_label_train == k) & is_boundary_train
        members_core = df_train.iloc[np.where(core_mask_tr)[0]]
        n_core_train = int(core_mask_tr.sum())
        n_bnd_train = int(bnd_mask_tr.sum())
        n_total_train = n_core_train + n_bnd_train

        # frequency in TEST split (the operating data)
        mask_test = (z_test == k)
        n_test_in_k = int(mask_test.sum())
        n_test_core = int((mask_test & ~isb_test).sum())
        n_test_bnd = int((mask_test & isb_test).sum())
        freq_pct = n_test_in_k / max(n_test, 1)
        certainty_pct = (n_test_core / n_test_in_k) if n_test_in_k else 0.0

        # KL imbalance
        kl = float(fairness_all.loc[k, "KL"]) if k in fairness_all.index else 0.0
        global_kl_mean = float(fairness_all["KL"].mean())
        kl_ratio = kl / global_kl_mean if global_kl_mean > 0 else 0.0
        lt_in_sit = float(fairness_all.loc[k, "LT"]) if k in fairness_all.index else 0.0
        global_lt = float(fairness_all["global_LT"].iloc[0]) if len(fairness_all) else 0.0

        # tokens (from the rule trace) — needed for readable lookup
        tr = traces[k]
        time_token_full = ((tr.get("weekend_token") + " " if tr.get("weekend_token") else "")
                            + tr.get("time_token", "")).strip()
        intent_trace = tr.get("intent", {})
        intent_token = intent_trace.get("token", "")
        readable, source = readable_label(time_token_full, intent_token, dict_blob)
        technical = final_names[k]

        # Signature: modal hour, weekend
        modal_hour = int(tr.get("modal_hour", -1))
        we_share = float(tr.get("weekend_share", 0.0))

        # Intent composition (FULL ranked bars, normalised over attractors)
        e_dims = prototypes[k][N_ATTRIB:N_ATTRIB + n_macros]
        e_attr = np.where(attractors_mask, e_dims, 0.0).astype(np.float64)
        e_attr = np.maximum(e_attr, 0)
        if e_attr.sum() > 0:
            comp = e_attr / e_attr.sum()
        else:
            comp = e_attr
        intent_composition = [
            {"macro": idx_to_macro[int(i)],
             "short": MACRO_SHORT.get(idx_to_macro[int(i)], idx_to_macro[int(i)]),
             "score": float(comp[int(i)]),
             "is_attractor": bool(attractors_mask[int(i)])}
            for i in np.argsort(-comp)
            if attractors_mask[int(i)] or comp[int(i)] > 0
        ]

        # Top-3 target-macro distribution on TEST
        cat_target_test = df_test["cat_macro"].values.astype(str)
        macros_in_test = cat_target_test[mask_test]
        if len(macros_in_test):
            vc = pd.Series(macros_in_test).value_counts(normalize=True)
            top3_target = [{"macro": m, "share": float(s)}
                            for m, s in vc.head(3).items()]
        else:
            top3_target = []

        is_sink = k in cfg["sinks"]
        # Note: distinctness on the readable label is applied after the loop;
        # see `_apply_readable_distinctness` below. We store the raw lookup
        # result here.
        cards.append({
            "id": int(k),
            "technical_name": technical,
            "readable_label": readable,
            "readable_source": source,
            "time_token": time_token_full,
            "intent_token": intent_token,
            "is_sink": bool(is_sink),
            "stats": {
                "freq_pct_of_test": round(freq_pct, 4),
                "n_test_in_situation": n_test_in_k,
                "certainty_pct_core": round(certainty_pct, 4),
                "kl": round(kl, 4),
                "kl_vs_global": round(kl_ratio, 3),
                "lt_in_sit": round(lt_in_sit, 4),
                "global_lt": round(global_lt, 4),
                "modal_hour": modal_hour,
                "weekend_share": round(we_share, 4),
                "pop_modal_hour": pop_modal_hour,
                "pop_weekend_share": round(pop_we_share, 4),
                "n_core_train": n_core_train,
                "n_boundary_train": n_bnd_train,
            },
            "intent_composition": intent_composition,
            "top3_target_macros": top3_target,
        })

    # Distinctness on the readable label: two situations may legitimately
    # map to the same dictionary entry (e.g. NYC s0/s6 both 'Morning · Outdoors'
    # → 'Morning stroll'). Disambiguate deterministically by suffixing  #<id>.
    # This is a rule, not a hand-edit: any reviewer regenerating from the
    # dictionary + per-situation tokens will produce the same suffix.
    seen: dict[str, list[int]] = {}
    for c in cards:
        if c["readable_label"]:
            seen.setdefault(c["readable_label"], []).append(c["id"])
    for lbl, ids in seen.items():
        if len(ids) <= 1:
            continue
        for cid in ids:
            for c in cards:
                if c["id"] == cid:
                    c["readable_label"] = f"{lbl} #{cid}"
                    c["readable_source"] = c["readable_source"] + "+distinctness"

    # Pack data needed by the explainability panel: per-cluster representative
    # touched-or-not request + top-20 OFF/ON.
    rep = _representative_requests(city, cfg, fit, ds, G1_mask, pop_pct,
                                      item_macro, [c["id"] for c in cards])

    return {
        "city": city, "K": K, "mode_label": cfg["mode_label"],
        "sinks": cfg["sinks"], "knee_kappa": cfg["knee_kappa"],
        "global_lt": float(global_lt),
        "global_kl_mean": float(global_kl_mean),
        "cards": cards,
        "explainability": rep,
    }


def _representative_requests(city: str, cfg: dict, fit, ds, G1_mask,
                                pop_pct, item_macro, ks: list[int]) -> list[dict]:
    """Per situation k: pick a representative test request from cluster k
    (touched if k is a sink, else any core member). Compute B_blind /
    X-SAGE top-3 and a small score-decomposition snippet."""
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)
    df_test = ds["df_test"]
    u_test = df_test["u_idx"].values.astype(np.int64)
    n_items = ds["n_items"]
    scores_blind = load_or_refit(city, model_name="FM")[u_test]
    excl = excluded_mask(city, n_items)
    boost = G1_mask.astype(np.float32)
    sink_mask = np.isin(z_test, cfg["sinks"])
    core_sink = sink_mask & ~isb_test
    scores_on = scores_blind.copy()
    scores_on[np.where(core_sink)[0]] += cfg["knee_kappa"] * boost[None, :]

    def _topk(s_row, u):
        s = s_row.copy()
        cols = excl.indices[excl.indptr[u]:excl.indptr[u + 1]]
        if len(cols):
            s[cols] = -np.inf
        return topk_from_scores(s, K_TOP), s
    out = []
    rng = np.random.default_rng(42)
    for k in ks:
        scope = (z_test == k) & ~isb_test
        idxs = np.where(scope)[0]
        if len(idxs) == 0:
            out.append({"id": k, "available": False})
            continue
        if k in cfg["sinks"]:
            # pick a touched request with the median number of swaps
            touched_idx = []
            swaps = []
            for b in idxs:
                u = int(u_test[b])
                top_off, _ = _topk(scores_blind[b], u)
                top_on, _ = _topk(scores_on[b], u)
                inter = len(set(map(int, top_off)) & set(map(int, top_on)))
                if inter < K_TOP:
                    touched_idx.append(int(b))
                    swaps.append(K_TOP - inter)
            if not touched_idx:
                pick = int(rng.choice(idxs))
            else:
                # median-swap request → most "illustrative"
                med = int(np.median(swaps))
                # pick the touched request closest to median swap
                diffs = [abs(s - med) for s in swaps]
                pick = touched_idx[int(np.argmin(diffs))]
        else:
            pick = int(rng.choice(idxs))

        u = int(u_test[pick])
        top_off, s_off_raw = _topk(scores_blind[pick], u)
        top_on, s_on_raw = _topk(scores_on[pick], u)
        # entries / exits
        set_off = set(map(int, top_off))
        set_on = set(map(int, top_on))
        entries_lt = [(r + 1, int(it))
                       for r, it in enumerate(top_on)
                       if int(it) not in set_off and G1_mask[int(it)]]
        # top-5 ON vs OFF for display
        rows_top5 = []
        for r in range(5):
            off_i = int(top_off[r]); on_i = int(top_on[r])
            rows_top5.append({
                "rank": r + 1,
                "off_item": off_i,
                "off_macro": item_macro.get(off_i, "?"),
                "off_pop_pct": round(float(pop_pct[off_i]), 3),
                "on_item": on_i,
                "on_macro": item_macro.get(on_i, "?"),
                "on_pop_pct": round(float(pop_pct[on_i]), 3),
                "changed": off_i != on_i,
            })
        # decomposition for top-1 in OFF/ON
        if k in cfg["sinks"] and entries_lt:
            promoted_rank, promoted_item = entries_lt[0]
            base = float(scores_blind[pick][promoted_item])
            nudge = float(scores_on[pick][promoted_item] - base)
            explainee = {"item": promoted_item, "rank_on": promoted_rank,
                            "macro": item_macro.get(promoted_item, "?"),
                            "score_base": round(base, 3),
                            "score_nudge": round(nudge, 3),
                            "score_total": round(base + nudge, 3),
                            "explanation_kind": "promoted_long_tail"}
        else:
            top1 = int(top_off[0])
            base = float(scores_blind[pick][top1])
            nudge = float(scores_on[pick][top1] - base)
            explainee = {"item": top1, "rank_on": 1,
                            "macro": item_macro.get(top1, "?"),
                            "score_base": round(base, 3),
                            "score_nudge": round(nudge, 3),
                            "score_total": round(base + nudge, 3),
                            "explanation_kind": ("untouched_zero_nudge"
                                                 if not core_sink[pick] else
                                                 "touched_top1")}
        out.append({
            "id": k,
            "available": True,
            "request": {
                "u_idx": u,
                "time_local": str(df_test.iloc[pick]["time_local"]),
                "target_macro": str(df_test.iloc[pick]["cat_macro"]),
                "target_fine": str(df_test.iloc[pick]["cat_fine"]),
                "is_core_sink": bool(core_sink[pick]),
            },
            "top5": rows_top5,
            "entries_lt_in_top20": entries_lt,
            "n_swaps_top20": K_TOP - len(set_off & set_on),
            "explainee": explainee,
        })
    return out


# ----------------------------- card figure -----------------------------------

# Flat aesthetic constants
COL_TEXT = "#1f1f23"
COL_MUTED = "#6c6c74"
COL_BORDER = "#d0d0d6"
COL_BG = "#ffffff"
COL_PANEL = "#f7f7f9"
COL_ACCENT = "#3a6ea5"
COL_SINK = "#c0392b"
COL_GOOD = "#27ae60"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": COL_BORDER,
    "axes.labelcolor": COL_TEXT,
    "axes.titlecolor": COL_TEXT,
    "xtick.color": COL_TEXT,
    "ytick.color": COL_TEXT,
    "savefig.facecolor": COL_BG,
    "figure.facecolor": COL_BG,
})


def _qualitative(value: float, low: float, hi: float) -> str:
    if value < low:
        return "low"
    if value > hi:
        return "high"
    return "med"


def draw_card(card: dict, ax) -> None:
    """Draw a single situation card on a matplotlib Axes."""
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    # outer card border
    border = mpatches.FancyBboxPatch(
        (1.5, 1.5), 97, 97, boxstyle="round,pad=0.0,rounding_size=2.5",
        linewidth=0.7, edgecolor=COL_BORDER, facecolor=COL_BG, zorder=1,
    )
    ax.add_patch(border)

    # Header bar
    title = card["readable_label"] or card["technical_name"]
    subtitle = (f"{card['technical_name']}"
                  + ("" if card["readable_label"]
                       else "  (dictionary fallback)"))
    ax.text(4.0, 92.5, title,
            fontsize=14, fontweight="bold", color=COL_TEXT)
    ax.text(4.0, 87.5, subtitle, fontsize=8.5, color=COL_MUTED)

    if card["is_sink"]:
        # sink badge top-right
        badge = mpatches.FancyBboxPatch(
            (78, 91), 18, 6, boxstyle="round,pad=0.0,rounding_size=1.4",
            linewidth=0.5, edgecolor=COL_SINK, facecolor="#fdecea", zorder=2,
        )
        ax.add_patch(badge)
        ax.text(87, 94, "★ inequity sink", fontsize=8.5, color=COL_SINK,
                ha="center", va="center", fontweight="bold")

    # Stat row (3 tiles)
    tile_y = 73; tile_h = 11
    tile_specs = [
        ("frequency",
          f"{card['stats']['freq_pct_of_test']*100:.1f}%",
          f"{card['stats']['n_test_in_situation']:,} test"),
        ("certainty (core)",
          f"{card['stats']['certainty_pct_core']*100:.0f}%",
          f"{card['stats']['n_core_train']:,}c · "
          f"{card['stats']['n_boundary_train']:,}b train"),
        ("KL × global",
          f"{card['stats']['kl_vs_global']:.2f}",
          f"KL={card['stats']['kl']:.2f}"),
    ]
    for i, (label, big, small) in enumerate(tile_specs):
        x0 = 4.0 + i * 31.0
        tile = mpatches.FancyBboxPatch(
            (x0, tile_y), 29, tile_h,
            boxstyle="round,pad=0.0,rounding_size=1.3",
            linewidth=0.5, edgecolor=COL_BORDER, facecolor=COL_PANEL,
        )
        ax.add_patch(tile)
        ax.text(x0 + 1.0, tile_y + tile_h - 2.3, label.upper(),
                fontsize=7.0, color=COL_MUTED, fontweight="bold")
        # Big value (red on sink for KL tile if KL high)
        big_col = (COL_SINK if (label == "KL × global"
                                  and card["stats"]["kl_vs_global"] > 1.5)
                     else COL_TEXT)
        ax.text(x0 + 1.0, tile_y + 5.5, big, fontsize=15.5,
                fontweight="bold", color=big_col)
        ax.text(x0 + 1.0, tile_y + 1.6, small, fontsize=7.4, color=COL_MUTED)

    # Time-band signature: small bar with low/med/high labels for hour, weekend
    sig_y = 60
    ax.text(4.0, sig_y + 4.0, "SIGNATURE", fontsize=8.0, color=COL_MUTED,
            fontweight="bold")
    sig_specs = [
        ("modal hour",
          f"{card['stats']['modal_hour']:02d}h",
          f"city {card['stats']['pop_modal_hour']:02d}h"),
        ("weekend",
          (f"{card['stats']['weekend_share']*100:.0f}%"),
          f"city {card['stats']['pop_weekend_share']*100:.0f}%"),
    ]
    for i, (label, big, small) in enumerate(sig_specs):
        x0 = 4.0 + i * 46.0
        ax.text(x0, sig_y - 2.0, label, fontsize=8.0, color=COL_MUTED)
        ax.text(x0, sig_y - 5.7, big, fontsize=12.5,
                fontweight="bold", color=COL_TEXT)
        ax.text(x0 + 17, sig_y - 5.7, small, fontsize=8.5, color=COL_MUTED,
                va="center")

    # Intent composition bars
    int_y_top = 48
    ax.text(4.0, int_y_top, "INTENT COMPOSITION", fontsize=8.0,
            color=COL_MUTED, fontweight="bold")
    # show up to top-5 attractor macros
    visible = [c for c in card["intent_composition"]
                if c["is_attractor"] and c["score"] > 0][:5]
    if not visible:
        ax.text(4.0, int_y_top - 4, "(no attractor mass — diffuse situation)",
                fontsize=8.5, color=COL_MUTED, style="italic")
    else:
        bar_h = 4.4
        gap = 1.6
        max_score = max(c["score"] for c in visible)
        for i, c in enumerate(visible):
            yb = int_y_top - 4.5 - i * (bar_h + gap)
            # label
            ax.text(4.0, yb + bar_h / 2.0, c["short"], fontsize=8.5,
                    color=COL_TEXT, va="center")
            # bar
            bar_x0 = 28.0
            bar_w_max = 56.0
            w = bar_w_max * (c["score"] / max_score)
            color_bar = (COL_SINK if card["is_sink"] and i == 0
                          else COL_ACCENT)
            ax.add_patch(mpatches.FancyBboxPatch(
                (bar_x0, yb), w, bar_h,
                boxstyle="round,pad=0.0,rounding_size=0.6",
                linewidth=0, facecolor=color_bar, alpha=0.85,
            ))
            ax.text(bar_x0 + w + 1.5, yb + bar_h / 2.0,
                    f"{c['score']*100:.0f}%", fontsize=8.5, color=COL_TEXT,
                    va="center")

    # Top target macros (next visit's macro distribution in test)
    foot_y = 12
    ax.text(4.0, foot_y + 4.5, "TOP NEXT-VENUE MACROS  (test)",
            fontsize=7.6, color=COL_MUTED, fontweight="bold")
    if card["top3_target_macros"]:
        chip_x = 4.0
        for c in card["top3_target_macros"]:
            short = MACRO_SHORT.get(c["macro"], c["macro"])
            label = f"{short} · {c['share']*100:.0f}%"
            # Chip width — generous so labels don't touch the next chip.
            w = max(15.0, 3.0 + len(label) * 1.55)
            chip = mpatches.FancyBboxPatch(
                (chip_x, foot_y - 2), w, 4.7,
                boxstyle="round,pad=0.0,rounding_size=2.2",
                linewidth=0.4, edgecolor=COL_BORDER, facecolor=COL_PANEL,
            )
            ax.add_patch(chip)
            ax.text(chip_x + w / 2.0, foot_y + 0.5, label, fontsize=8,
                    color=COL_TEXT, ha="center", va="center")
            chip_x += w + 2.0
            if chip_x > 88:  # avoid running off the card border
                break


# ----------------------------- explainability figure -------------------------

def draw_explainability(card: dict, exp: dict, ax) -> None:
    """Draw an explainability panel on a matplotlib Axes."""
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    border = mpatches.FancyBboxPatch(
        (1.5, 1.5), 97, 97, boxstyle="round,pad=0.0,rounding_size=2.5",
        linewidth=0.7, edgecolor=COL_BORDER, facecolor=COL_BG, zorder=1,
    )
    ax.add_patch(border)

    title = card["readable_label"] or card["technical_name"]
    ax.text(4.0, 92, "EXPLAINABILITY  ·  " + title,
            fontsize=11.5, fontweight="bold", color=COL_TEXT)
    ax.text(4.0, 87.5, card["technical_name"], fontsize=8.5, color=COL_MUTED)
    if card["is_sink"]:
        ax.text(96, 92, "★ sink", fontsize=9.0, color=COL_SINK, ha="right",
                fontweight="bold")

    if not exp.get("available"):
        ax.text(50, 50, "no requests in test split", fontsize=10,
                ha="center", color=COL_MUTED, style="italic")
        return

    req = exp["request"]
    ax.text(4.0, 81.5, (f"Request u={req['u_idx']}  at  {req['time_local']}  "
                          f"→ target: {req['target_macro']} / "
                          f"{req['target_fine']}"),
            fontsize=8.5, color=COL_MUTED)

    # Side-by-side top-5 table (rows go from y≈75 down to ~46).
    # Use MACRO_SHORT to keep the per-row text inside its column width.
    table_y_top = 75
    col0_x = 4.0
    col1_x = 52.0
    ax.text(col0_x, table_y_top, "B_blind top-5", fontsize=8.5,
            fontweight="bold", color=COL_TEXT)
    ax.text(col1_x, table_y_top, "X-SAGE top-5  (★ = changed)", fontsize=8.5,
            fontweight="bold", color=COL_TEXT)
    row_h = 5.6
    for i, row in enumerate(exp["top5"]):
        y_row = table_y_top - 3.6 - i * row_h
        off_m = MACRO_SHORT.get(row["off_macro"], row["off_macro"][:14])
        on_m = MACRO_SHORT.get(row["on_macro"], row["on_macro"][:14])
        ax.text(col0_x, y_row, f"{row['rank']}.", fontsize=8.4,
                color=COL_MUTED, va="top")
        ax.text(col0_x + 4.5, y_row,
                (f"item {row['off_item']} · {off_m}   "
                  f"[{row['off_pop_pct']*100:.0f}%pop]"),
                fontsize=7.8, color=COL_TEXT, va="top")
        is_lt_entry = any(eli[1] == row["on_item"]
                            for eli in exp["entries_lt_in_top20"])
        col = COL_GOOD if is_lt_entry else COL_TEXT
        marker = " ★" if row["changed"] else ""
        ax.text(col1_x, y_row, f"{row['rank']}.", fontsize=8.4,
                color=COL_MUTED, va="top")
        ax.text(col1_x + 4.5, y_row,
                (f"item {row['on_item']} · {on_m}{marker}   "
                  f"[{row['on_pop_pct']*100:.0f}%pop]"),
                fontsize=7.8, color=col, va="top",
                fontweight=("bold" if is_lt_entry else "normal"))

    # Single unified box for decomposition + narrative + foot (y = 5..43)
    box_y = 5; box_h = 38
    box = mpatches.FancyBboxPatch(
        (4.0, box_y), 92, box_h,
        boxstyle="round,pad=0.0,rounding_size=1.5",
        linewidth=0.5, edgecolor=COL_BORDER, facecolor=COL_PANEL,
    )
    ax.add_patch(box)

    # 1. score decomposition (top of box)
    ax.text(5.5, box_y + box_h - 3.5, "SCORE DECOMPOSITION",
            fontsize=7.8, color=COL_MUTED, fontweight="bold")
    e = exp["explainee"]
    ax.text(5.5, box_y + box_h - 7.5,
            (f"item {e['item']}  ·  {e['macro']}  "
              f"·  rank {e['rank_on']} in X-SAGE list"),
            fontsize=8.8, color=COL_TEXT, fontweight="bold")
    eq = (f"score_total  =  score_base  +  situational nudge"
            f"  →  {e['score_total']:.3f}  =  "
            f"{e['score_base']:.3f}  +  {e['score_nudge']:+.3f}")
    ax.text(5.5, box_y + box_h - 11.0, eq, fontsize=8.2, color=COL_TEXT,
            family="monospace")

    # 2. narrative (middle of box)
    if e["explanation_kind"] == "promoted_long_tail":
        narrative = (f"Recommended item {e['item']} ({e['macro']}) at rank "
                       f"{e['rank_on']} because you are in '{title}'"
                       f" — an inequity-sink situation, which lifts every "
                       f"long-tail item by +κ.")
    elif e["explanation_kind"] == "untouched_zero_nudge":
        narrative = (f"This situation ('{title}') is not an inequity sink: "
                       f"the recommendations are the base FM's, unmodified.")
    else:
        narrative = (f"Top-1 in '{title}' kept the base FM's choice; "
                       f"long-tail items may be promoted at deeper ranks.")
    ax.text(5.5, box_y + box_h - 15.5, "NATURAL LANGUAGE",
            fontsize=7.8, color=COL_MUTED, fontweight="bold")
    # wrap narrative to fit
    wrapped = _wrap_text(narrative, width=90)
    for j, line in enumerate(wrapped[:4]):
        ax.text(5.5, box_y + box_h - 19.0 - j * 3.2, line,
                fontsize=8.4, color=COL_TEXT)

    # 3. mechanism note
    note = {
        "promoted_long_tail":
            "Removing the nudge term subtracts +κ exactly and the item exits"
            " the top-20 — verified faithful (B8.3: 0 violations / 76 934 LT"
            " entries; max abs err 2.4×10⁻⁷ = float32 ε).",
        "touched_top1":
            "List was re-ranked but the top-1 was unchanged. The nudge"
            " for this item is 0 because it is not long-tail. Faithfulness"
            " (B8.3) holds trivially.",
        "untouched_zero_nudge":
            "Situation is NOT an inequity sink → the nudge is identically 0"
            " on every item; X-SAGE list = B_blind list. The decomposition"
            " still applies (with nudge=0), preserving faithfulness (B8.3).",
    }.get(e["explanation_kind"], "")
    wrapped_note = _wrap_text(note, width=92)
    for j, line in enumerate(wrapped_note[:3]):
        ax.text(5.5, box_y + 4.5 - j * 2.8, line,
                fontsize=7.4, color=COL_MUTED, style="italic")


def _wrap_text(text: str, width: int) -> list[str]:
    """Simple greedy word-wrap for matplotlib text (no external deps)."""
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# ----------------------------- HTML page -------------------------------------

HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>X-SAGE — situation showcase</title>
<style>
:root{--bg:#fafafc;--card:#fff;--text:#1f1f23;--muted:#6c6c74;
--border:#d0d0d6;--panel:#f7f7f9;--accent:#3a6ea5;--sink:#c0392b;
--good:#27ae60;}
@media (prefers-color-scheme: dark){:root{
--bg:#13141a;--card:#1c1d24;--text:#e8e8ec;--muted:#9c9ca6;
--border:#2c2d35;--panel:#22232b;}}
*{box-sizing:border-box}
body{margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,
"Segoe UI",sans-serif;color:var(--text);background:var(--bg);}
header{padding:24px 32px;border-bottom:1px solid var(--border);}
h1{margin:0 0 4px;font-size:18px;font-weight:600;}
header .sub{color:var(--muted);font-size:13px;}
.tabs{display:flex;gap:8px;padding:16px 32px 0;}
.tab{padding:8px 16px;border:1px solid var(--border);border-radius:8px;
background:var(--card);color:var(--muted);font-size:13px;cursor:pointer;}
.tab.active{color:var(--text);background:var(--panel);border-color:var(--accent);}
main{padding:24px 32px;}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));
gap:18px;}
.card{background:var(--card);border:1px solid var(--border);
border-radius:12px;padding:18px;cursor:pointer;transition:.15s;}
.card:hover{border-color:var(--accent);transform:translateY(-1px);}
.card-title{font-size:16px;font-weight:600;}
.card-sub{color:var(--muted);font-size:11.5px;margin-top:2px;}
.badge{display:inline-block;padding:2px 8px;border-radius:6px;
background:#fdecea;color:var(--sink);font-size:11px;font-weight:600;
border:1px solid #f5b7b1;margin-left:8px;}
@media (prefers-color-scheme: dark){.badge{background:#3a1a1a;border-color:#7a3030;}}
.tiles{display:flex;gap:8px;margin:12px 0;}
.tile{flex:1;background:var(--panel);border-radius:8px;padding:8px 10px;
border:1px solid var(--border);}
.tile-l{font-size:9.5px;color:var(--muted);text-transform:uppercase;
font-weight:600;letter-spacing:.04em;}
.tile-v{font-size:18px;font-weight:700;margin-top:1px;}
.tile-s{font-size:10px;color:var(--muted);}
.tile-v.danger{color:var(--sink);}
.sig{margin-top:6px;font-size:11px;color:var(--muted);}
.intent{margin-top:10px;}
.intent h4{margin:6px 0 6px;font-size:9.5px;color:var(--muted);
text-transform:uppercase;letter-spacing:.04em;font-weight:700;}
.intent .row{display:flex;align-items:center;gap:6px;margin:3px 0;}
.intent .lbl{width:74px;font-size:11px;}
.intent .bar{flex:1;height:6px;background:var(--panel);border-radius:4px;
overflow:hidden;}
.intent .bar > div{height:100%;background:var(--accent);}
.intent .row.sink .bar > div{background:var(--sink);}
.intent .pct{font-size:10.5px;color:var(--muted);width:36px;text-align:right;}
.chips{margin-top:10px;display:flex;flex-wrap:wrap;gap:5px;}
.chip{padding:3px 8px;background:var(--panel);border:1px solid var(--border);
border-radius:10px;font-size:10.5px;color:var(--muted);}
.expandable{display:none;margin-top:12px;background:var(--panel);
border-radius:8px;padding:12px;font-size:12px;border:1px solid var(--border);}
.expandable.open{display:block;}
.exp-rows{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:6px;}
.exp-rows h5{margin:0 0 4px;font-size:11px;color:var(--muted);}
.exp-rows ol{margin:0;padding-left:18px;font-size:11.5px;}
.exp-rows .lt{color:var(--good);font-weight:600;}
.dec{margin-top:8px;background:var(--card);border:1px solid var(--border);
padding:8px 10px;border-radius:6px;font-family:ui-monospace,SFMono-Regular,
Menlo,monospace;font-size:11px;}
.note{margin-top:6px;color:var(--muted);font-size:11px;font-style:italic;}
.narrative{margin-top:8px;font-size:12.5px;line-height:1.45;}
.faith{margin-top:10px;font-size:10.5px;color:var(--muted);font-style:italic;
border-top:1px dashed var(--border);padding-top:8px;}
section.city{display:none;}
section.city.active{display:block;}
</style></head>
<body>
<header>
<h1>X-SAGE — situation showcase</h1>
<div class="sub">Click a card to expand the explainability panel. All
content rendered from <code>cards_data.json</code> + <code>
explainability_data.json</code>; same numbers as the PNG/PDF figures.</div>
</header>
<div class="tabs" id="tabs"></div>
<main id="main"></main>
<script>
const DATA = __DATA__;

function renderCity(city){
  const c = DATA[city];
  const el = document.createElement("div");
  el.className = "grid";
  c.cards.forEach((card)=>{
    const exp = c.explainability.find(e=>e.id===card.id);
    const title = card.readable_label || card.technical_name;
    const subtitle = card.readable_label ? card.technical_name :
                      "(dictionary fallback)";
    const sink = card.is_sink ? '<span class="badge">★ sink</span>' : "";
    const kl_danger = card.stats.kl_vs_global > 1.5 ? "danger" : "";
    const visible = (card.intent_composition || []).filter(x=>x.is_attractor && x.score>0).slice(0,5);
    let bars = "";
    if(visible.length===0){
      bars = '<div class="note">(no attractor mass — diffuse situation)</div>';
    } else {
      const maxs = Math.max(...visible.map(x=>x.score));
      bars = visible.map((x,i)=>`<div class="row ${card.is_sink && i===0 ? "sink":""}">
        <div class="lbl">${x.short}</div>
        <div class="bar"><div style="width:${(x.score/maxs*100).toFixed(1)}%"></div></div>
        <div class="pct">${(x.score*100).toFixed(0)}%</div>
      </div>`).join("");
    }
    const chips = (card.top3_target_macros||[]).map(t=>
      `<span class="chip">${(t.macro||"").replace("&","&amp;")}: ${(t.share*100).toFixed(0)}%</span>`).join("");
    // explainability
    let expHtml = "";
    if(exp && exp.available){
      const e = exp.explainee;
      const top5 = exp.top5.map(r=>{
        const isLT = exp.entries_lt_in_top20.some(([_,it])=>it===r.on_item);
        return `<li><span>${r.off_item} · ${r.off_macro} <span class="pct">[${(r.off_pop_pct*100).toFixed(0)}%pop]</span></span></li>`;
      }).join("");
      const top5on = exp.top5.map(r=>{
        const isLT = exp.entries_lt_in_top20.some(([_,it])=>it===r.on_item);
        const mark = r.changed ? " ★" : "";
        return `<li class="${isLT?"lt":""}">${r.on_item} · ${r.on_macro}${mark} <span class="pct">[${(r.on_pop_pct*100).toFixed(0)}%pop]</span></li>`;
      }).join("");
      let note = "";
      let narr = "";
      if(e.explanation_kind==="promoted_long_tail"){
        note = "Removing the nudge term subtracts +κ exactly and item exits the top-20 — verified faithful (B8.3: 0 violations / 76 934).";
        narr = `Recommended item ${e.item} (${e.macro}) at rank ${e.rank_on} because you are in '${title}' — an inequity-sink situation, which lifts every long-tail item by +κ.`;
      } else if(e.explanation_kind==="untouched_zero_nudge"){
        note = "Situation is NOT an inequity sink → the nudge is identically 0 on every item; X-SAGE list = B_blind list. The decomposition still applies (with nudge = 0), preserving faithfulness (B8.3).";
        narr = `This situation ('${title}') is not an inequity sink: the recommendations are the base FM's, unmodified.`;
      } else {
        note = "This list was re-ranked but the top-1 was unchanged.";
        narr = `Top-1 in '${title}' kept the base FM's choice.`;
      }
      const req = exp.request;
      expHtml = `<div class="expandable">
        <div class="card-sub">Request u=${req.u_idx} at ${req.time_local} → target: ${req.target_macro} / ${req.target_fine}</div>
        <div class="exp-rows">
          <div><h5>B_blind top-5</h5><ol>${top5}</ol></div>
          <div><h5>X-SAGE top-5</h5><ol>${top5on}</ol></div>
        </div>
        <div class="dec">item ${e.item} · ${e.macro} · rank ${e.rank_on}<br>
        score_total = score_base + nudge → ${e.score_total} = ${e.score_base} + ${e.score_nudge>=0?"+":""}${e.score_nudge}</div>
        <div class="note">${note}</div>
        <div class="narrative">${narr}</div>
        <div class="faith">Faithfulness verified (B8.3): 0 violations over 76 934 long-tail entries; max abs error 2.4×10⁻⁷ (float32 ε).</div>
      </div>`;
    } else if(exp){
      expHtml = `<div class="expandable"><div class="note">no requests in test split</div></div>`;
    }
    el.insertAdjacentHTML("beforeend", `
      <div class="card" data-id="${card.id}">
        <div class="card-title">${title}${sink}</div>
        <div class="card-sub">${subtitle}</div>
        <div class="tiles">
          <div class="tile"><div class="tile-l">frequency</div>
            <div class="tile-v">${(card.stats.freq_pct_of_test*100).toFixed(1)}%</div>
            <div class="tile-s">${card.stats.n_test_in_situation.toLocaleString()} test</div></div>
          <div class="tile"><div class="tile-l">certainty</div>
            <div class="tile-v">${(card.stats.certainty_pct_core*100).toFixed(0)}%</div>
            <div class="tile-s">core in test</div></div>
          <div class="tile"><div class="tile-l">KL × global</div>
            <div class="tile-v ${kl_danger}">${card.stats.kl_vs_global.toFixed(2)}</div>
            <div class="tile-s">KL=${card.stats.kl.toFixed(2)}</div></div>
        </div>
        <div class="sig">modal hour <b>${String(card.stats.modal_hour).padStart(2,'0')}h</b> (city ${String(card.stats.pop_modal_hour).padStart(2,'0')}h) · weekend <b>${(card.stats.weekend_share*100).toFixed(0)}%</b> (city ${(card.stats.pop_weekend_share*100).toFixed(0)}%)</div>
        <div class="intent"><h4>intent composition</h4>${bars}</div>
        <div class="chips"><span class="chip" style="border:none;background:transparent;color:var(--muted);">top next macros:</span>${chips}</div>
        ${expHtml}
      </div>`);
  });
  return el;
}

function setupTabs(){
  const tabs = document.getElementById("tabs");
  const main = document.getElementById("main");
  Object.keys(DATA).forEach((city,i)=>{
    const t = document.createElement("div");
    t.className = "tab" + (i===0?" active":"");
    t.textContent = city;
    t.onclick = ()=>{
      document.querySelectorAll(".tab").forEach(x=>x.classList.remove("active"));
      t.classList.add("active");
      document.querySelectorAll("section.city").forEach(x=>x.classList.remove("active"));
      document.getElementById("city-"+city).classList.add("active");
    };
    tabs.appendChild(t);
    const sec = document.createElement("section");
    sec.className = "city" + (i===0?" active":"");
    sec.id = "city-"+city;
    sec.appendChild(renderCity(city));
    main.appendChild(sec);
  });
  // expand on click
  main.addEventListener("click",(e)=>{
    const c = e.target.closest(".card");
    if(!c) return;
    const ex = c.querySelector(".expandable");
    if(ex) ex.classList.toggle("open");
  });
}
setupTabs();
</script></body></html>"""


# ----------------------------- main ------------------------------------------

def main() -> int:
    out_root = REPO_ROOT / "outputs" / "round3" / "B11"
    out_root.mkdir(parents=True, exist_ok=True)
    dict_blob = json.loads(DICT_PATH.read_text())

    by_city = {}
    for city in ("NYC", "TKY"):
        print(f"\n>>> B11 on {city}")
        data = assemble_city(city, dict_blob)
        cdir = out_root / city / "cards"
        edir = out_root / city / "explainability"
        cdir.mkdir(parents=True, exist_ok=True)
        edir.mkdir(parents=True, exist_ok=True)

        # cards_data.json
        (out_root / city / "cards_data.json").write_text(
            json.dumps(data["cards"], indent=2, default=str))
        (out_root / city / "explainability_data.json").write_text(
            json.dumps(data["explainability"], indent=2, default=str))

        # per-situation figures
        for card in data["cards"]:
            fig, ax = plt.subplots(figsize=(6.2, 5.7), dpi=160)
            draw_card(card, ax)
            for ext in ("png", "pdf"):
                fig.savefig(cdir / f"s{card['id']}.{ext}",
                              bbox_inches="tight", pad_inches=0.15)
            plt.close(fig)

            exp = next((e for e in data["explainability"]
                          if e["id"] == card["id"]),
                          {"id": card["id"], "available": False})
            fig, ax = plt.subplots(figsize=(7.8, 7.2), dpi=160)
            draw_explainability(card, exp, ax)
            for ext in ("png", "pdf"):
                fig.savefig(edir / f"s{card['id']}.{ext}",
                              bbox_inches="tight", pad_inches=0.15)
            plt.close(fig)
            print(f"  s{card['id']} → "
                  f"{card['readable_label'] or '(fallback)'} | "
                  f"{card['technical_name']}")

        by_city[city] = data

    # write HTML
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(by_city, default=str))
    (out_root / "showcase.html").write_text(html)
    print(f"\nwrote {out_root}/showcase.html ({len(html)//1024} KB)")
    print(f"wrote {out_root}/<city>/cards/s<k>.{{png,pdf}}")
    print(f"wrote {out_root}/<city>/explainability/s<k>.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
