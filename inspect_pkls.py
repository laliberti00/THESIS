"""Inspect Shehzad's .pkl files for the three DCCF datasets we keep.

This is a one-shot diagnostic, run during Phase 1 of the thesis cleanup.
Its output is captured into DATA_INVENTORY.md. Safe to delete after Phase 1.
"""

import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sps


REPO = Path(__file__).resolve().parent
DATASETS = ["gowalla", "amazonBook", "tmall"]


def describe(name: str, obj) -> None:
    print(f"  [{name}]")
    print(f"    python type:    {type(obj).__module__}.{type(obj).__name__}")
    if sps.issparse(obj):
        coo = obj.tocoo()
        print(f"    sparse format:  {obj.format}")
        print(f"    shape:          {obj.shape}")
        print(f"    nnz:            {obj.nnz}")
        print(f"    dtype:          {obj.dtype}")
        print(f"    data sample:    {coo.data[:10]}")
        print(f"    row sample:     {coo.row[:10]}")
        print(f"    col sample:     {coo.col[:10]}")
        print(f"    unique data:    {np.unique(coo.data)[:10]}")
    elif isinstance(obj, np.ndarray):
        print(f"    ndarray shape:  {obj.shape}")
        print(f"    dtype:          {obj.dtype}")
        print(f"    sample:         {obj.ravel()[:10]}")
    elif isinstance(obj, (list, tuple)):
        print(f"    len:            {len(obj)}")
        if obj:
            print(f"    first item:     {repr(obj[0])[:200]}")
    elif isinstance(obj, dict):
        print(f"    n keys:         {len(obj)}")
        sample_keys = list(obj.keys())[:5]
        for k in sample_keys:
            v = obj[k]
            v_repr = repr(v)[:120]
            print(f"      key={k!r:>10} -> {type(v).__name__}: {v_repr}")
    else:
        print(f"    repr:           {repr(obj)[:300]}")


def stats(name: str, mat) -> None:
    if not sps.issparse(mat):
        return
    csr = mat.tocsr()
    n_users, n_items = csr.shape
    nnz = csr.nnz
    per_user = np.asarray((csr != 0).sum(axis=1)).ravel()
    per_item = np.asarray((csr != 0).sum(axis=0)).ravel()
    print(f"    --- stats ({name}) ---")
    print(f"    n_users={n_users}, n_items={n_items}, interactions={nnz}")
    print(
        f"    interactions/user:  min={per_user.min()}, "
        f"max={per_user.max()}, mean={per_user.mean():.2f}"
    )
    print(
        f"    interactions/item:  min={per_item.min()}, "
        f"max={per_item.max()}, mean={per_item.mean():.2f}"
    )
    n_empty_users = int((per_user == 0).sum())
    print(f"    users with 0 interactions: {n_empty_users}")


def main():
    for ds in DATASETS:
        data_dir = REPO / "data" / "DCCF" / ds
        print(f"\n========================================")
        print(f"  DATASET: {ds}  ({data_dir})")
        print(f"========================================")
        for pkl in sorted(data_dir.glob("*.pkl")):
            size_mb = pkl.stat().st_size / (1024 * 1024)
            print(f"\n- File: {pkl.name} ({size_mb:.2f} MB)")
            with open(pkl, "rb") as f:
                obj = pickle.load(f)
            describe(pkl.name, obj)
            if sps.issparse(obj):
                stats(pkl.name, obj)


if __name__ == "__main__":
    main()
