"""Round-3 B8b — robust intent token (dominance-gap rule) + stability re-test.

B8's intent token was a hard argmax over the centroid's reachable-macro
scores, which was brittle to argmax-flip when the top two were near-tied
(strict-name stability 38 % NYC / 0 % TKY across seeds, even though
cluster-level ARI was 0.82 / 0.66).

This task replaces the argmax with a dominance-gap rule:

    sort attractor scores; let p1 ≥ p2 ≥ p3 ≥ ...
    if  p1 - p2 ≥ τ_dom   →  token = m1            (clear dominance)
    elif p2 - p3 ≥ τ_dom  →  token = sorted({m1, m2}) joined "/"
                                                    (co-dominance, order-stable)
    else                  →  token = "Mixed"        (three-way diffuse)

The pair is sorted alphabetically by short-label so swapping p1↔p2 yields
the SAME string ("Food/Shop" not "Shop/Food"). τ_dom = 0.10 is declared
in the rule trace.

Re-runs naming + Hungarian-matched stability with seeds {43, 44, 45}.

Outputs:
  outputs/round3/B8b/<city>/{
      named_situations_robust.csv,
      naming_rule_trace.md,
      name_stability_robust.json,
      worked_examples_named.md,
  }
  outputs/round3/B8b/summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.xsage.backbone import excluded_mask, load_or_refit
from pipeline.step02_models.xsage.metrics import long_tail_groups, topk_from_scores
from pipeline.step02_models.xsage.orchestrator import _load_city
from pipeline.step02_models.xsage.l2_comprehension import fit_rough_kmeans

# Reuse B8 helpers (rule constants + non-intent helpers).
from experiments.round3_b8_explainability import (
    K_TOP, SHORT_HEAD, N_ATTRIB, MACRO_SHORT,
    HOUR_BAND_EDGES, WEEKEND_HI, WEEKEND_LO, BOUNDARY_DIFFUSE_HI,
    CONFIG, _hour_band, _ensure_distinct, _match_clusters,
    _intent_token_of, _time_band_of, _item_macro,
)


# Robust-intent threshold (declared, single value).
TAU_DOM = 0.10


# -----------------------------------------------------------------------------
# Robust intent token
# -----------------------------------------------------------------------------

def _robust_intent_token(prototypes_k: np.ndarray, n_macros: int,
                            attractors_mask: np.ndarray,
                            idx_to_macro: dict[int, str]) -> tuple[str, dict]:
    """Dominance-gap rule on attractor-restricted centroid e-dims.

    Returns (token_string, trace_dict).
    """
    e_dims = prototypes_k[N_ATTRIB:N_ATTRIB + n_macros]
    e_attr = np.where(attractors_mask, e_dims, -np.inf)
    sorted_idx = np.argsort(e_attr)[::-1]
    p = [float(e_attr[int(i)]) for i in sorted_idx]
    m = [str(idx_to_macro[int(i)]) for i in sorted_idx]
    # Discard -inf entries (non-attractors).
    valid = [(p[i], m[i]) for i in range(len(p)) if p[i] > -np.inf]
    valid_p = [v[0] for v in valid]
    valid_m = [v[1] for v in valid]

    n_attr = len(valid_p)
    trace = {"valid_attractor_scores": [round(x, 4) for x in valid_p[:5]],
              "valid_attractor_macros": valid_m[:5],
              "tau_dom": TAU_DOM}

    if n_attr == 0:
        return "Untagged", {**trace, "rule_branch": "no_attractors"}
    if n_attr == 1:
        token = MACRO_SHORT.get(valid_m[0], valid_m[0])
        return token, {**trace, "rule_branch": "single_attractor",
                          "token": token}

    p1, p2 = valid_p[0], valid_p[1]
    p3 = valid_p[2] if n_attr >= 3 else -np.inf
    gap_12 = p1 - p2
    gap_23 = p2 - p3 if n_attr >= 3 else float("inf")

    trace["gap_p1_minus_p2"] = round(float(gap_12), 4)
    trace["gap_p2_minus_p3"] = (round(float(gap_23), 4)
                                  if n_attr >= 3 else None)

    if gap_12 >= TAU_DOM:
        token = MACRO_SHORT.get(valid_m[0], valid_m[0])
        trace["rule_branch"] = "clear_dominance"
        trace["token"] = token
        return token, trace
    if gap_23 >= TAU_DOM:
        # Co-dominance: order-stable pair alphabetically by short label.
        short1 = MACRO_SHORT.get(valid_m[0], valid_m[0])
        short2 = MACRO_SHORT.get(valid_m[1], valid_m[1])
        pair = sorted([short1, short2])
        token = "/".join(pair)
        trace["rule_branch"] = "co_dominance"
        trace["token"] = token
        return token, trace
    # Three-way diffuse → "Mixed".
    trace["rule_branch"] = "diffuse"
    trace["token"] = "Mixed"
    return "Mixed", trace


def name_situation_robust(prototypes_k: np.ndarray,
                            members_df: pd.DataFrame,
                            attractors_mask: np.ndarray,
                            idx_to_macro: dict[int, str],
                            n_macros: int,
                            boundary_share: float) -> tuple[str, dict]:
    """Same shell as B8's name_situation but with the robust intent token."""
    trace: dict = {}
    n = len(members_df)

    # Time token (modal hour band of core members).
    if n == 0:
        time_token = "Untagged"; modal_hour = -1
    else:
        hour_counts = np.bincount(
            members_df["c_hour"].values.astype(int), minlength=24)
        modal_hour = int(np.argmax(hour_counts))
        time_token = _hour_band(modal_hour)
    trace["modal_hour"] = int(modal_hour)
    trace["time_token"] = time_token

    # Weekend token (extreme-only).
    we_share = float(members_df["c_isweekend"].mean()) if n else 0.0
    trace["weekend_share"] = round(we_share, 4)
    if we_share > WEEKEND_HI:
        we_token = "Weekend"
    elif we_share < WEEKEND_LO:
        we_token = "Weekday"
    else:
        we_token = ""
    trace["weekend_token"] = we_token

    # Robust intent token.
    intent_token, intent_trace = _robust_intent_token(
        prototypes_k, n_macros, attractors_mask, idx_to_macro)
    trace["intent"] = intent_trace

    head = (we_token + " " if we_token else "") + time_token
    name = f"{head} · {intent_token}"
    if boundary_share > BOUNDARY_DIFFUSE_HI:
        name += " (diffuse)"
    trace["boundary_share"] = round(boundary_share, 4)
    return name, trace


def _name_situations_robust(prototypes: np.ndarray, core_labels: np.ndarray,
                              is_boundary: np.ndarray, df_train: pd.DataFrame,
                              attractors_mask: np.ndarray,
                              idx_to_macro: dict[int, str],
                              n_macros: int) -> tuple[list[str], list[dict]]:
    K = prototypes.shape[0]
    names, traces = [], []
    for k in range(K):
        core_mask = (core_labels == k) & ~is_boundary
        members = df_train.iloc[np.where(core_mask)[0]]
        n_core = int(core_mask.sum())
        n_clust = int((core_labels == k).sum())
        b_share = 1.0 - (n_core / max(n_clust, 1))
        nm, tr = name_situation_robust(prototypes[k], members,
                                          attractors_mask, idx_to_macro,
                                          n_macros, b_share)
        names.append(nm); traces.append(tr)
    return names, traces


# -----------------------------------------------------------------------------
# Per-city
# -----------------------------------------------------------------------------

def run_city(city: str) -> dict:
    cfg = CONFIG[city]
    out_dir = REPO_ROOT / "outputs" / "round3" / "B8b" / city
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> B8b on {city} (τ_dom={TAU_DOM})")

    fit = np.load(REPO_ROOT / cfg["sit_dir"] / "fit.npz", allow_pickle=True)
    prototypes = np.asarray(fit["prototypes"])
    core_label_train = np.asarray(fit["core_label_train"]).astype(np.int32)
    is_boundary_train = np.asarray(fit["is_boundary_train"]).astype(bool)
    v_train = np.asarray(fit["v_train"])
    attractors_mask = np.asarray(fit["attractors"]).astype(bool)
    K = prototypes.shape[0]

    ds = _load_city(city)
    n_macros = ds["n_macros"]
    idx_to_macro = ds["idx_to_macro"]
    df_train = ds["df_train"]

    # 1. Robust names on the reference seed (42 — saved fit).
    names_raw, traces = _name_situations_robust(
        prototypes, core_label_train, is_boundary_train,
        df_train, attractors_mask, idx_to_macro, n_macros)
    final_names, dist_log = _ensure_distinct(names_raw, traces, prototypes,
                                                n_macros, attractors_mask,
                                                idx_to_macro)
    distinct_ok = len(set(final_names)) == K
    print(f"  K={K}  distinct={distinct_ok}  collisions: "
          f"{len(dist_log['collisions'])}")

    # 2. Stability across seeds {43, 44, 45} — Hungarian-matched.
    alt_seeds = (43, 44, 45)
    stab = {"reference_seed": 42, "reference_names": final_names,
            "tau_dom": TAU_DOM,
            "alternate_seeds": list(alt_seeds), "per_seed": {}}
    for s in alt_seeds:
        res = fit_rough_kmeans(v_train, K=cfg["K"], eps=cfg["eps"],
                                 max_iter=80, seed=int(s))
        match = _match_clusters(prototypes, res.prototypes)  # alt → ref
        alt_raw, alt_tr = _name_situations_robust(
            res.prototypes, res.core_label, res.is_boundary,
            df_train, attractors_mask, idx_to_macro, n_macros)
        alt_names, _ = _ensure_distinct(alt_raw, alt_tr, res.prototypes,
                                          n_macros, attractors_mask,
                                          idx_to_macro)
        ref_to_alt = {int(kr): int(ka) for ka, kr in enumerate(match)
                       if kr >= 0}
        n_strict = n_intent = n_time = 0
        rows = []
        for kref in range(cfg["K"]):
            ref_name = final_names[kref]
            ref_base_intent = traces[kref]["intent"].get("token", "")
            ka = ref_to_alt.get(kref, -1)
            if ka < 0:
                rows.append({"k_ref": kref, "ref_name": ref_name,
                              "matched_alt_id": None,
                              "matched_alt_name": None,
                              "strict_match": False,
                              "intent_match": False,
                              "time_match": False}); continue
            an = alt_names[ka]
            alt_base_intent = alt_tr[ka]["intent"].get("token", "")
            strict = (ref_name == an)
            # Intent match on the BASE intent token from the rule trace
            # (before distinctness fallbacks). Comparing post-fallback
            # names penalises the rule for fallback-induced cosmetic
            # differences (e.g. "+Food" vs "+Shop" when the cluster's
            # secondary attractor was tied and flipped under perturbation).
            intent = (ref_base_intent == alt_base_intent
                       and ref_base_intent != "")
            time_ = (_time_band_of(ref_name) == _time_band_of(an))
            n_strict += int(strict); n_intent += int(intent); n_time += int(time_)
            rows.append({"k_ref": kref, "ref_name": ref_name,
                          "matched_alt_id": ka, "matched_alt_name": an,
                          "ref_base_intent": ref_base_intent,
                          "alt_base_intent": alt_base_intent,
                          "strict_match": strict, "intent_match": intent,
                          "time_match": time_})
        stab["per_seed"][str(s)] = {
            "strict_match_rate": n_strict / cfg["K"],
            "intent_match_rate": n_intent / cfg["K"],
            "time_band_match_rate": n_time / cfg["K"],
            "rows": rows,
        }
        print(f"  seed={s}: strict {n_strict}/{cfg['K']} "
              f"({n_strict/cfg['K']:.2f})  "
              f"intent {n_intent}/{cfg['K']} ({n_intent/cfg['K']:.2f})  "
              f"time {n_time}/{cfg['K']} ({n_time/cfg['K']:.2f})")
    with open(out_dir / "name_stability_robust.json", "w") as f:
        json.dump(stab, f, indent=2, default=str)

    # Aggregate strict/intent/time over the alt seeds.
    avg_strict = float(np.mean(
        [stab["per_seed"][str(s)]["strict_match_rate"] for s in alt_seeds]))
    avg_intent = float(np.mean(
        [stab["per_seed"][str(s)]["intent_match_rate"] for s in alt_seeds]))
    avg_time = float(np.mean(
        [stab["per_seed"][str(s)]["time_band_match_rate"] for s in alt_seeds]))
    print(f"  AVG over {len(alt_seeds)} alt seeds: "
          f"strict {avg_strict:.2f}  intent {avg_intent:.2f}  "
          f"time {avg_time:.2f}")

    # 3. Persist named_situations_robust.csv + rule trace.
    pd.DataFrame([
        {"id": k, "name": final_names[k], "is_sink": k in cfg["sinks"],
         "rule_branch": traces[k]["intent"].get("rule_branch", ""),
         "gap_p1_minus_p2": traces[k]["intent"].get("gap_p1_minus_p2"),
         "gap_p2_minus_p3": traces[k]["intent"].get("gap_p2_minus_p3")}
        for k in range(K)
    ]).to_csv(out_dir / "named_situations_robust.csv", index=False)

    rule_lines = [
        f"# {city} — robust naming rule trace (B8b)",
        "",
        f"Rule (deterministic, no test access):",
        f"  HOUR_BAND_EDGES = {HOUR_BAND_EDGES}",
        f"  WEEKEND_HI = {WEEKEND_HI}, WEEKEND_LO = {WEEKEND_LO}",
        f"  TAU_DOM = {TAU_DOM}",
        f"  BOUNDARY_DIFFUSE_HI = {BOUNDARY_DIFFUSE_HI}",
        "",
        "Intent token branches:",
        "  if (p1 - p2) >= TAU_DOM   →  clear_dominance: token = m1",
        "  elif (p2 - p3) >= TAU_DOM →  co_dominance: token = "
        "sorted({m1, m2})[0]/sorted({m1, m2})[1]",
        "  else                      →  diffuse: token = 'Mixed'",
        "",
        f"Distinctness fallbacks (applied in order): "
        f"second-best attractor → @HHh modal hour → #k cluster id.",
        "",
        "Per-situation firings:",
        "",
    ]
    for k in range(K):
        rule_lines.append(f"### s{k} → **{final_names[k]}**")
        for kk, vv in traces[k].items():
            rule_lines.append(f"  - {kk}: {vv}")
        rule_lines.append("")
    if dist_log["collisions"]:
        rule_lines.append("## Distinctness refinements")
        for c in dist_log["collisions"]:
            rule_lines.append(f"- step {c['step']}: name `{c['name']}` "
                              f"collided across ids {c['ids']}"
                              + (f" — {c['note']}" if c.get("note") else ""))
    (out_dir / "naming_rule_trace.md").write_text("\n".join(rule_lines))

    return {
        "city": city, "K": K, "tau_dom": TAU_DOM,
        "named_situations": final_names,
        "distinct_after_refinement": distinct_ok,
        "n_collisions_total": sum(1 for c in dist_log["collisions"]),
        "stability_avg": {"strict": avg_strict, "intent": avg_intent,
                            "time": avg_time},
        "stability_per_seed": {
            str(s): {
                "strict": stab["per_seed"][str(s)]["strict_match_rate"],
                "intent": stab["per_seed"][str(s)]["intent_match_rate"],
                "time":   stab["per_seed"][str(s)]["time_band_match_rate"],
            } for s in alt_seeds
        },
    }


def _rebuild_worked_examples_named(city: str, final_names: list[str]) -> None:
    """Re-render the worked examples with the NEW robust names. Keeps the
    same picking rule as B8 so the lead examples stay comparable."""
    cfg = CONFIG[city]
    out_dir = REPO_ROOT / "outputs" / "round3" / "B8b" / city

    fit = np.load(REPO_ROOT / cfg["sit_dir"] / "fit.npz", allow_pickle=True)
    core_label_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    is_boundary_test = np.asarray(fit["is_boundary_test"]).astype(bool)
    ds = _load_city(city)
    n_macros = ds["n_macros"]
    idx_to_macro = ds["idx_to_macro"]
    df_test = ds["df_test"]
    df_train = ds["df_train"]
    u_test = df_test["u_idx"].values.astype(np.int64)
    n_items = ds["n_items"]

    scores_blind_uitem = load_or_refit(city, model_name="FM")
    scores_blind = scores_blind_uitem[u_test]
    excl = excluded_mask(city, n_items)
    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(axis=0)).ravel()
    G0, G1 = long_tail_groups(pop, short_head_share=SHORT_HEAD)
    sink_mask = np.isin(core_label_test, cfg["sinks"])
    core_sink = sink_mask & ~is_boundary_test
    scores_on = scores_blind.copy()
    scores_on[np.where(core_sink)[0]] += cfg["knee_kappa"] * G1.astype(np.float32)[None, :]

    def _topk(s):
        out = np.zeros((s.shape[0], K_TOP), dtype=np.int32)
        for b in range(s.shape[0]):
            ss = s[b].copy()
            u = int(u_test[b])
            cols = excl.indices[excl.indptr[u]:excl.indptr[u + 1]]
            if len(cols):
                ss[cols] = -np.inf
            out[b] = topk_from_scores(ss, K_TOP)
        return out
    top_off = _topk(scores_blind); top_on = _topk(scores_on)
    n_test = len(u_test)
    touched = np.zeros(n_test, dtype=bool)
    n_swap = np.zeros(n_test, dtype=int)
    for b in range(n_test):
        set_off = set(map(int, top_off[b])); set_on = set(map(int, top_on[b]))
        inter = len(set_off & set_on)
        n_swap[b] = K_TOP - inter
        if not np.array_equal(top_off[b], top_on[b]) and inter < K_TOP:
            touched[b] = True

    touched_idx = np.where(touched)[0]
    order = sorted(touched_idx.tolist(),
                    key=lambda b: (-int(n_swap[b]), int(u_test[b]),
                                    str(df_test.iloc[b]["time_local"])))
    picks = order[:3]
    pop_pct = np.argsort(np.argsort(pop)) / max(len(pop) - 1, 1)
    cat_macro_test = df_test["cat_macro"].values.astype(str)
    cat_fine_test = df_test["cat_fine"].values.astype(str)
    ex = [f"# {city} — worked examples (B8b robust naming, "
           f"τ_dom = {TAU_DOM})",
           "",
           f"_Operating point: {cfg['mode_label']} situations, "
           f"sinks={cfg['sinks']}, knee κ_fair = {cfg['knee_kappa']}._",
           "",
           f"_Names below are produced by the robust dominance-gap rule "
           f"(B8b.1); no hand-editing._",
           ""]
    for b in picks:
        z = int(core_label_test[b])
        sit_name = final_names[z]
        is_b = bool(is_boundary_test[b])
        cert = "core" if not is_b else "boundary"
        ex.append(f"## Example — u={int(u_test[b])} at "
                   f"{df_test.iloc[b]['time_local']}")
        ex.append(f"- Situation: **{sit_name}** (id s{z}, {cert})")
        ex.append(f"- Target macro / fine: {cat_macro_test[b]} / "
                   f"{cat_fine_test[b]}")
        set_off = set(map(int, top_off[b])); set_on = set(map(int, top_on[b]))
        new_lts = [(r + 1, int(it)) for r, it in enumerate(top_on[b])
                    if int(it) not in set_off and G1[int(it)]]
        ex.append(f"- Long-tail items lifted into top-20: "
                   f"{len(new_lts)} (κ·boost = +{cfg['knee_kappa']:.2f} each)")
        ex.append("")
        ex.append("| rank | OFF item (pop %ile / macro) | ON item (pop %ile / macro) |")
        ex.append("|---:|---|---|")
        for r in range(K_TOP):
            off_i = int(top_off[b, r]); on_i = int(top_on[b, r])
            off_macro = _item_macro(off_i, df_train, idx_to_macro)
            on_macro = _item_macro(on_i, df_train, idx_to_macro)
            mark = "" if off_i == on_i else " ★"
            ex.append(f"| {r+1} | item={off_i}, {off_macro} / "
                       f"{pop_pct[off_i]:.2f} | item={on_i}, {on_macro} / "
                       f"{pop_pct[on_i]:.2f} |{mark}")
        ex.append("")
        if new_lts:
            tlr, tli = min(new_lts)
            tlm = _item_macro(tli, df_train, idx_to_macro)
            ex.append(f"**Explanation**: lifted **item {tli}** ({tlm}) to "
                       f"rank {tlr} because the request falls in "
                       f"**{sit_name}** ({cert}), which boosts every "
                       f"long-tail item by +{cfg['knee_kappa']:.2f} "
                       f"(∑ of {len(new_lts)} long-tail entries among the "
                       f"top-20). Score and nudge are additive — removing "
                       f"the nudge regenerates the OFF list exactly "
                       f"(B8.3, faithfulness = 100 %, unchanged by the "
                       f"robust naming layer of B8b).")
        ex.append("")
    (out_dir / "worked_examples_named.md").write_text("\n".join(ex))


# -----------------------------------------------------------------------------

def main() -> int:
    out_root = REPO_ROOT / "outputs" / "round3" / "B8b"
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for city in ("NYC", "TKY"):
        summary[city] = run_city(city)
        _rebuild_worked_examples_named(city, summary[city]["named_situations"])
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print before-vs-after comparison using B8 summary if available.
    b8_summary_path = REPO_ROOT / "outputs" / "round3" / "B8" / "summary.json"
    b8_summary = (json.loads(b8_summary_path.read_text())
                   if b8_summary_path.exists() else None)
    print("\n=== B8b vs B8 STABILITY (avg over alt seeds) ===")
    print("                strict    intent    time-band")
    for city in ("NYC", "TKY"):
        b8b = summary[city]["stability_avg"]
        if b8_summary and city in b8_summary:
            old = b8_summary[city]["name_stability"]
            # B8 stored per-seed dict {seed: {strict,intent,time}}
            old_strict = float(np.mean([v["strict"] for v in old.values()]))
            old_intent = float(np.mean([v["intent"] for v in old.values()]))
            old_time = float(np.mean([v["time"] for v in old.values()]))
            print(f"  {city}  B8  -> {old_strict:.2f}      {old_intent:.2f}      {old_time:.2f}")
        print(f"  {city}  B8b -> {b8b['strict']:.2f}      {b8b['intent']:.2f}      {b8b['time']:.2f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
