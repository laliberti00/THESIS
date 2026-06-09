"""Smoke test for the situation-aware FM. Runs end-to-end on NYC with tiny HPs
to catch shape / leakage / matched-OFF bugs *before* the full go/no-go.

Skipped if data/processed/NYC/ isn't there. Designed to complete in < 2 min on
a CPU laptop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.situational.data import build_city_dataset
from pipeline.step02_models.situational.model import (SituationalConfig,
                                                          SituationalModel)
from pipeline.step02_models.situational.trainer import (TrainConfig,
                                                            train_situational)
from pipeline.step02_models.situational.ranker import rank_split


PROCESSED_NYC = REPO_ROOT / "data" / "processed" / "NYC"


def _have_nyc():
    return (PROCESSED_NYC / "df_train.parquet").exists()


@pytest.mark.skipif(not _have_nyc(),
                    reason="data/processed/NYC not present")
def test_dataset_shapes():
    ds = build_city_dataset(PROCESSED_NYC, verbose=False)
    for s, name in [(ds.train, "train"), (ds.val, "val"), (ds.test, "test")]:
        B = len(s.u)
        assert s.i.shape == (B,)
        assert s.m.shape == (B,)
        assert s.c_hour.shape == (B,)
        assert s.prev_geo.shape == (B,)
        assert s.intent_feat.shape == (B, ds.n_macros + 3)
        assert s.intent_empty.shape == (B,)
    # Catalogue lookup is valid everywhere.
    assert (ds.item_cat_macro >= 0).all()
    assert ds.item_cat_macro.shape == (ds.n_items,)


@pytest.mark.skipif(not _have_nyc(),
                    reason="data/processed/NYC not present")
def test_v0_off_returns_y0_exactly():
    """With ``use_situation=False`` the modulation must be bypassed."""
    ds = build_city_dataset(PROCESSED_NYC, verbose=False)
    cfg = SituationalConfig(n_users=ds.n_users, n_items=ds.n_items,
                            n_macros=ds.n_macros, n_geo=ds.n_geo,
                            K=1, d=8, d_g=4, d_e=4, r=2,
                            use_situation=False, use_intent=False)
    model = SituationalModel(cfg)
    model.set_item_cat_macro(torch.from_numpy(ds.item_cat_macro.astype(np.int64)))

    # Random non-zero modulation params so a bug would show up.
    with torch.no_grad():
        model.modulation.B.copy_(torch.randn_like(model.modulation.B) * 0.5)
        model.modulation.S.copy_(torch.randn_like(model.modulation.S) * 0.5)

    u = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    i = torch.tensor([10, 11, 12, 13], dtype=torch.long)
    m = torch.tensor(ds.item_cat_macro[i.numpy()].astype(np.int64))
    c_t = torch.randn(4, 7)
    pi_dummy = torch.zeros(4, 1); pi_dummy[:, 0] = 1.0

    y = model.forward_pair(u, i, m, c_t, pi_dummy)
    y0 = model.fm(u, i, m, c_t)
    assert torch.allclose(y, y0, atol=1e-6), \
        f"V0 (situation OFF) must return y0 exactly; max diff = {(y-y0).abs().max():.3e}"


@pytest.mark.skipif(not _have_nyc(),
                    reason="data/processed/NYC not present")
def test_train_one_epoch_no_nan_and_ranker_works():
    ds = build_city_dataset(PROCESSED_NYC, verbose=False)
    cfg = SituationalConfig(n_users=ds.n_users, n_items=ds.n_items,
                            n_macros=ds.n_macros, n_geo=ds.n_geo,
                            K=3, d=16, d_g=4, d_e=4, r=2,
                            use_situation=True, use_intent=True)
    model = SituationalModel(cfg)
    t_cfg = TrainConfig(lr=5e-3, weight_decay=1e-5, batch_size=1024,
                         max_epochs=1, patience=1, eval_every=1,
                         eval_batch_size=512, seed=0, verbose=False)
    rep = train_situational(model, ds, t_cfg)
    assert np.isfinite(rep["best_val_recall20"]), \
        f"NaN val recall after smoke train: {rep}"

    res = rank_split(model, ds, ds.test, cutoffs=(20,),
                      batch_size=256,
                      exclude_mask=(ds.urm_train + ds.urm_val).tocsr(),
                      save_z=True)
    r20 = res.metrics[20]["RECALL"]
    assert np.isfinite(r20).all()
    assert (r20 >= 0).all() and (r20 <= 1).all()
    assert res.z_per_request is not None
    assert res.z_per_request.shape == (len(ds.test.u),)


def main() -> int:
    print("=== situational smoke tests ===")
    if not _have_nyc():
        print("  SKIPPED (data/processed/NYC missing)")
        return 0
    test_dataset_shapes()
    print("  [shapes] OK")
    test_v0_off_returns_y0_exactly()
    print("  [V0 OFF == y0] OK")
    test_train_one_epoch_no_nan_and_ranker_works()
    print("  [train 1 epoch + rank] OK")
    print("=== smoke OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
