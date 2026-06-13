"""Round-3 B6 — Is the lens backbone-agnostic? Multi-backbone audit.

For each of the 8 step02a floor models, score the test set, run the Stage-B
lens on top-20 lists using the keep-mode situations (the audit
representation), and report:

  - Per-model sink sets (cities × models matrix).
  - Pairwise Jaccard overlap of sink sets across models.
  - Per-situation LT/KL heatmap (models × situations).
  - Global LT of each model.
  - Split verification: G0 = top-20 % of items by train interaction count
                       (Ge convention, repo's split definition).

Reading: if sink sets substantially coincide across models → inequity is
structural; if they diverge → lens performs per-model audits. Either
answer is a paper result.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.xsage.backbone import excluded_mask
from pipeline.step02_models.xsage.metrics import (kl_divergence, long_tail_groups,
                                                       topk_from_scores)
from pipeline.step02_models.xsage.orchestrator import _load_city


K_TOP = 20
SHORT_HEAD = 0.20
LENS_RATIO = 1.5
LT_EXCESS_MIN = 0.05

MODELS = ["Random", "TopPop", "ItemKNN", "UserKNN",
            "P3alpha", "RP3beta", "EASE_R", "FM"]


def _score_model(city: str, model_name: str, urm: sps.csr_matrix) -> np.ndarray:
    """Refit model on URM_train ∪ URM_val with floor best HPs and return the
    full user × item score matrix.
    """
    hp_path = (REPO_ROOT / "outputs" / city / "baselines"
                / f"{model_name}.best_hp.json")
    hp = json.loads(hp_path.read_text(encoding="utf-8")) if hp_path.exists() else {}
    n_users, n_items = urm.shape
    if model_name == "Random":
        rng = np.random.default_rng(int(hp.get("random_seed", 42)))
        return rng.random((n_users, n_items)).astype(np.float32)
    if model_name == "TopPop":
        pop = np.asarray(urm.sum(axis=0)).ravel().astype(np.float32)
        return np.tile(pop, (n_users, 1))
    if model_name == "ItemKNN":
        from engine.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
        rec = ItemKNNCFRecommender(urm, verbose=False); rec.fit(**hp)
        return rec._compute_item_score(np.arange(n_users)).astype(np.float32)
    if model_name == "UserKNN":
        from engine.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
        rec = UserKNNCFRecommender(urm, verbose=False); rec.fit(**hp)
        return rec._compute_item_score(np.arange(n_users)).astype(np.float32)
    if model_name == "P3alpha":
        from engine.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
        rec = P3alphaRecommender(urm, verbose=False); rec.fit(**hp)
        return rec._compute_item_score(np.arange(n_users)).astype(np.float32)
    if model_name == "RP3beta":
        from engine.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
        rec = RP3betaRecommender(urm, verbose=False); rec.fit(**hp)
        return rec._compute_item_score(np.arange(n_users)).astype(np.float32)
    if model_name == "EASE_R":
        from engine.Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
        rec = EASE_R_Recommender(urm); rec.fit(**hp)
        return rec._compute_item_score(np.arange(n_users)).astype(np.float32)
    if model_name == "FM":
        from engine.Recommenders.FactorizationMachines.FMRecommender import FMRecommender
        rec = FMRecommender(urm)
        rec.fit(**hp)
        return rec._compute_item_score(np.arange(n_users)).astype(np.float32)
    raise ValueError(f"Unknown model {model_name!r}")


def _topk_per_request(scores_per_request: np.ndarray, exclude: sps.csr_matrix,
                         users: np.ndarray, K: int = K_TOP) -> np.ndarray:
    n = scores_per_request.shape[0]
    out = np.zeros((n, K), dtype=np.int32)
    for b in range(n):
        u = int(users[b])
        s = scores_per_request[b].copy()
        cols = exclude.indices[exclude.indptr[u]:exclude.indptr[u + 1]]
        if len(cols):
            s[cols] = -np.inf
        out[b] = topk_from_scores(s, K)
    return out


def _lens_for_topk(top: np.ndarray, z: np.ndarray, isb: np.ndarray,
                       users: np.ndarray, exclude: sps.csr_matrix,
                       G1_mask: np.ndarray, n_items: int) -> tuple[list[dict], float, float]:
    """Returns (rows, global_LT, global_KL_mean_all)."""
    all_items = top.flatten()
    global_dist = np.bincount(all_items, minlength=n_items).astype(np.float64)
    global_dist /= max(global_dist.sum(), 1.0)
    global_LT = float(G1_mask[all_items].mean())

    K_sit = int(z.max() + 1)
    rows = []
    for k in range(K_sit):
        for split, mask in (("all", z == k),
                                ("core", (z == k) & ~isb),
                                ("boundary", (z == k) & isb)):
            n_req = int(mask.sum())
            if n_req == 0:
                rows.append({"situation": k, "split": split, "n_requests": 0,
                              "LT": None, "KL": None, "available_LT": None})
                continue
            items = top[mask].flatten()
            d = np.bincount(items, minlength=n_items).astype(np.float64)
            d /= max(d.sum(), 1.0)
            lt = float(G1_mask[items].mean())
            kl = kl_divergence(d, global_dist)
            rows.append({"situation": k, "split": split, "n_requests": n_req,
                          "LT": lt, "KL": kl})
        users_k = np.unique(users[z == k])
        avails = []
        for u in users_k:
            seen = exclude.indices[exclude.indptr[u]:exclude.indptr[u + 1]]
            allowed = np.ones(n_items, dtype=bool); allowed[seen] = False
            if allowed.any():
                avails.append(float(G1_mask[allowed].mean()))
        avail = float(np.mean(avails)) if avails else None
        for r in rows[-3:]:
            r["available_LT"] = avail

    global_KL_mean = float(np.mean([r["KL"] for r in rows
                                          if r["split"] == "all" and r["KL"] is not None]))
    return rows, global_LT, global_KL_mean


def _identify_sinks(rows: list[dict], global_KL_mean: float) -> set[int]:
    sinks = set()
    for r in rows:
        if r["split"] != "all" or r["KL"] is None: continue
        kl_r = r["KL"] / max(global_KL_mean, 1e-9)
        lt_ex = (r["LT"] - r["available_LT"]) if r["available_LT"] is not None else 0.0
        if kl_r >= LENS_RATIO and abs(lt_ex) >= LT_EXCESS_MIN:
            sinks.add(int(r["situation"]))
    return sinks


def jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    inter = len(a & b); union = len(a | b)
    return inter / max(union, 1)


def run_city(city: str) -> dict:
    out_dir = REPO_ROOT / "outputs" / "round3" / "B6" / city
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> B6 on {city}")
    ds = _load_city(city)
    n_items = ds["n_items"]
    df_test = ds["df_test"]
    u_test = df_test["u_idx"].values.astype(np.int64)
    excl = excluded_mask(city, n_items)
    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(axis=0)).ravel()
    _, G1_mask = long_tail_groups(pop, short_head_share=SHORT_HEAD)

    sit_dir = REPO_ROOT / "outputs" / city / "xsage" / "situations"
    fit = np.load(sit_dir / "fit.npz", allow_pickle=True)
    z_test = np.asarray(fit["core_label_test"]).astype(np.int32)
    isb_test = np.asarray(fit["is_boundary_test"]).astype(bool)
    K_sit = int(z_test.max() + 1)
    print(f"  K_sit = {K_sit}, n_test = {len(u_test)}")

    # URM for refit
    urm = (ds["urm_train"] + ds["urm_val"]).tocsr()
    urm.data[:] = 1.0; urm = urm.astype(np.float32)

    all_rows = []  # per-model per-situation lens rows
    per_model_summary = []
    heatmap_KL = np.zeros((len(MODELS), K_sit), dtype=np.float64)
    heatmap_LT = np.zeros((len(MODELS), K_sit), dtype=np.float64)
    sinks_per_model: dict[str, set[int]] = {}

    for mi, model in enumerate(MODELS):
        t0 = time.time()
        print(f"  [{mi+1}/{len(MODELS)}] {model} ...", end="", flush=True)
        try:
            scores_uitem = _score_model(city, model, urm)
        except Exception as e:
            print(f" SKIP ({e.__class__.__name__})")
            per_model_summary.append({"model": model, "status": "skip",
                                          "global_LT": None})
            continue
        # Broadcast user-level scores to per-request rows
        scores = scores_uitem[u_test]
        top = _topk_per_request(scores, excl, u_test)
        rows, global_LT, gkm = _lens_for_topk(top, z_test, isb_test,
                                                    u_test, excl, G1_mask, n_items)
        sinks = _identify_sinks(rows, gkm)
        sinks_per_model[model] = sinks
        for r in rows:
            r["model"] = model
            all_rows.append(r)
            if r["split"] == "all" and r["KL"] is not None:
                k = int(r["situation"])
                heatmap_KL[mi, k] = r["KL"]
                heatmap_LT[mi, k] = r["LT"]
        per_model_summary.append({"model": model, "status": "ok",
                                      "global_LT": global_LT,
                                      "global_KL_mean": gkm,
                                      "sinks": sorted(sinks),
                                      "n_sinks": len(sinks)})
        print(f" sinks={sorted(sinks)}  global_LT={global_LT:.3f}  ({time.time()-t0:.1f}s)")

    pd.DataFrame(all_rows).to_csv(out_dir / "lens_by_model.csv", index=False)
    pd.DataFrame(per_model_summary).to_csv(out_dir / "model_summary.csv", index=False)

    # Jaccard overlap pairwise
    overlap = {}
    for m1 in sinks_per_model:
        overlap[m1] = {}
        for m2 in sinks_per_model:
            overlap[m1][m2] = jaccard(sinks_per_model[m1], sinks_per_model[m2])
    union_all = set().union(*sinks_per_model.values())
    inter_all = set.intersection(*[s for s in sinks_per_model.values() if s]) if any(sinks_per_model.values()) else set()
    (out_dir / "sink_overlap.json").write_text(
        json.dumps({"per_pair_jaccard": overlap,
                       "union_across_models": sorted(union_all),
                       "intersection_across_models": sorted(inter_all),
                       "per_model_sinks": {m: sorted(s) for m, s in sinks_per_model.items()}},
                       indent=2),
        encoding="utf-8")

    # Heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mat, title in zip(axes, (heatmap_KL, heatmap_LT),
                                  (f"{city} — per-situation KL (rows=models)",
                                   f"{city} — per-situation LT (rows=models)")):
        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        ax.set_xticks(range(K_sit)); ax.set_xticklabels([f"s{i}" for i in range(K_sit)])
        ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS)
        ax.set_title(title)
        for i in range(len(MODELS)):
            for j in range(K_sit):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                          fontsize=7, color="black" if mat[i, j] < mat.max() * 0.6 else "white")
        plt.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap.png", dpi=140); plt.close(fig)

    return {"city": city, "sinks_per_model": {m: sorted(s) for m, s in sinks_per_model.items()},
              "union_sinks": sorted(union_all), "intersection_sinks": sorted(inter_all)}


def write_split_verification(out_path: Path) -> None:
    """G0 = top 20 % of items by train interaction count (Ge convention)."""
    lines = ["# B6 split verification\n\n",
             "## Long-tail split definition used in this repo\n\n",
             "`G0` (short-head) = top 20 % of items by **train + val interaction count**.\n",
             "`G1` (long-tail) = the remaining 80 %.\n\n",
             "This is what `pipeline.step02_models.xsage.metrics.long_tail_groups` "
             "produces with `short_head_share = 0.20`. The cut-off is computed from "
             "`(URM_train + URM_val).sum(axis=0)` so that train+val popularity is the "
             "criterion. The convention matches Ge et al. (RecSys 2021) — top-X-% of "
             "items by popularity define the short head; the remainder is the long tail.\n\n",
             "## Comparison to alternative conventions\n\n",
             "- Some papers split by **train-only** popularity. The repo uses train+val, "
             "which is the same as the floor's exclude-seen mask: items the recommender is "
             "allowed to know about at test time.\n",
             "- Some papers use the **80/20 by interaction VOLUME** rather than item count. "
             "The repo uses item count (cleaner and closer to Ge's text). A robustness check "
             "with the 80/20-by-volume convention can be added without re-running anything "
             "heavier than this lens.\n\n",
             "All round-2 and round-3 lens numbers (sinks, LT, KL) reported in the paper "
             "are computed with the **train+val item-count 20% split** as above.\n"]
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", choices=["NYC", "TKY", "both"], default="both")
    args = parser.parse_args()
    cities = ["NYC", "TKY"] if args.city == "both" else [args.city]
    results = {}
    for c in cities:
        results[c] = run_city(c)
    write_split_verification(REPO_ROOT / "outputs" / "round3" / "B6" / "split_verification.md")
    # Top-level summary
    sums = []
    for c, r in results.items():
        for m, s in r["sinks_per_model"].items():
            sums.append({"city": c, "model": m, "sinks": s, "n_sinks": len(s)})
    pd.DataFrame(sums).to_csv(REPO_ROOT / "outputs" / "round3" / "B6"
                                  / "summary.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
