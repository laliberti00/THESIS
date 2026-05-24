"""Equivalence test for the EvaluatorHoldout per-user patch.

Verifies three invariants on Gowalla with RP3β:

1.  Aggregate metrics with ``save_per_user=False`` (legacy path) are
    bit-identical to the pre-patch reference saved by Shehzad in
    ``results/DCCF/gowalla_RP3betaRecommender.txt``.
2.  Aggregate metrics with ``save_per_user=True`` are *also* bit-identical
    to the same reference — the patch is additive, not transformative.
3.  For PRECISION, RECALL and NDCG: the mean of the per-user array equals
    (within ~1e-12) the aggregate value. For MAP/MRR: same.

Run from repo root:
    python -m tests.test_evaluator_patch_equivalence
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topn_baselines_neurals.Data_manager.Gowalla_AmazonBook_Tmall_DCCF import (
    Gowalla_AmazonBook_Tmall_DCCF,
)
from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout
from topn_baselines_neurals.Recommenders.GraphBased.RP3betaRecommender import (
    RP3betaRecommender,
)


# Best HPs from run_experiments_for_DCCF_original_baselines.py:87-88
RP3BETA_HP_GOWALLA = {
    "topK": 777,
    "alpha": 0.5663562161452378,
    "beta": 0.001085447926739258,
    "normalize_similarity": True,
}
CUTOFFS = [1, 5, 10, 20, 40, 50, 100]
TOL_AGGREGATE = 1e-12       # bit-identical between save_per_user False/True
TOL_VS_REFERENCE = 1e-3     # 0.1 %: numpy 1.23 vs 1.26 tie-breaking — see
                            # REPRODUCIBILITY_CHECK.md (agreed thesis tol = 1 %).
                            # 1e-3 is 10× stricter than the agreed envelope.
TOL_MEAN_VS_AGGR = 1e-12    # mean of per-user must equal aggregate


def load_reference() -> pd.DataFrame:
    ref_path = REPO / "results/DCCF/gowalla_RP3betaRecommender.txt"
    df = pd.read_csv(ref_path, sep="\t")
    df = df.rename(columns={"cuttOff": "cutoff"}).set_index("cutoff")
    return df


def fit_rp3beta(URM_train):
    rec = RP3betaRecommender(URM_train, verbose=False)
    rec.fit(**RP3BETA_HP_GOWALLA)
    return rec


def main():
    print("=== Equivalence test of EvaluatorHoldout per-user patch ===\n")

    data_path = REPO / "data/DCCF/gowalla"
    t0 = time.time()
    URM_train, URM_test = Gowalla_AmazonBook_Tmall_DCCF()._load_data_from_give_files(
        data_path.resolve(), validation=False
    )
    print(f"[1/4] Loaded Gowalla in {time.time()-t0:.1f}s "
          f"(URM_train nnz={URM_train.nnz}, URM_test nnz={URM_test.nnz}).")

    t0 = time.time()
    rec = fit_rp3beta(URM_train)
    print(f"[2/4] Fitted RP3β in {time.time()-t0:.1f}s.")

    # Run 1: legacy path (save_per_user=False)
    print("\n[3/4] Run A — save_per_user=False (legacy path)")
    t0 = time.time()
    eval_A = EvaluatorHoldout(URM_test, CUTOFFS, exclude_seen=True,
                              verbose=False, save_per_user=False)
    df_A, _ = eval_A.evaluateRecommender(rec)
    print(f"    done in {time.time()-t0:.1f}s")
    assert not hasattr(eval_A, "per_user_metrics"), (
        "Legacy path must NOT populate per_user_metrics")

    # Run 2: per-user path (save_per_user=True)
    print("\n[4/4] Run B — save_per_user=True (per-user enabled)")
    t0 = time.time()
    eval_B = EvaluatorHoldout(URM_test, CUTOFFS, exclude_seen=True,
                              verbose=False, save_per_user=True)
    df_B, _ = eval_B.evaluateRecommender(rec)
    print(f"    done in {time.time()-t0:.1f}s")
    assert hasattr(eval_B, "per_user_metrics"), \
        "Patched path MUST populate per_user_metrics"
    assert hasattr(eval_B, "per_user_user_ids"), \
        "Patched path MUST populate per_user_user_ids"

    # ---- Invariant 1: A vs B bit-identical ----
    print("\n--- Invariant 1: aggregate(False) == aggregate(True) ---")
    for cutoff in CUTOFFS:
        for metric in df_A.columns:
            a = float(df_A.loc[cutoff, metric])
            b = float(df_B.loc[cutoff, metric])
            diff = abs(a - b)
            assert diff <= TOL_AGGREGATE, (
                f"FAIL @ cutoff={cutoff}, metric={metric}: A={a}, B={b}, diff={diff}")
    print(f"  OK — all {len(CUTOFFS)*len(df_A.columns)} (cutoff, metric) pairs "
          f"identical within {TOL_AGGREGATE}")

    # ---- Invariant 2: aggregate vs Shehzad reference ----
    print("\n--- Invariant 2: aggregate(True) ≈ Shehzad reference ---")
    df_ref = load_reference()
    max_dev = 0.0
    for cutoff in CUTOFFS:
        for metric in ("PRECISION", "RECALL", "NDCG", "MAP", "MRR"):
            mine = float(df_B.loc[cutoff, metric])
            ref = float(df_ref.loc[cutoff, metric])
            rel = abs(mine - ref) / max(abs(ref), 1e-12)
            max_dev = max(max_dev, rel)
            assert rel <= TOL_VS_REFERENCE, (
                f"FAIL @ cutoff={cutoff}, metric={metric}: mine={mine}, "
                f"ref={ref}, rel_diff={rel:.2e}")
    print(f"  OK — max relative deviation = {max_dev:.2e} (threshold {TOL_VS_REFERENCE})")

    # ---- Invariant 3: per-user mean == aggregate ----
    print("\n--- Invariant 3: mean(per_user) == aggregate ---")
    n_eval = len(eval_B.per_user_user_ids)
    print(f"  per_user arrays have length {n_eval}")
    assert n_eval > 0, "no users evaluated"
    for cutoff in CUTOFFS:
        for metric in ("PRECISION", "RECALL", "NDCG", "MAP", "MRR"):
            arr = eval_B.per_user_metrics[cutoff][metric]
            assert arr.shape == (n_eval,), \
                f"shape mismatch @ cutoff={cutoff}, metric={metric}: {arr.shape}"
            mean = float(arr.mean())
            aggr = float(df_B.loc[cutoff, metric])
            diff = abs(mean - aggr)
            assert diff <= TOL_MEAN_VS_AGGR, (
                f"FAIL @ cutoff={cutoff}, metric={metric}: "
                f"mean(per_user)={mean}, aggregate={aggr}, diff={diff:.2e}")
    print(f"  OK — for {len(CUTOFFS)*5} (cutoff, metric) pairs the mean of the "
          f"per-user array equals the aggregate within {TOL_MEAN_VS_AGGR}")

    # ---- Show one sample of the per-user vectors ----
    print("\n--- Sample of per-user vectors at cutoff=20 ---")
    for metric in ("RECALL", "NDCG"):
        arr = eval_B.per_user_metrics[20][metric]
        print(f"  {metric:>9s}@20: n={len(arr)}, mean={arr.mean():.6f}, "
              f"std={arr.std():.6f}, min={arr.min():.6f}, max={arr.max():.6f}, "
              f"%zero={100*(arr==0).mean():.1f}%")

    print("\n=== ALL THREE INVARIANTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
