# step01 — Preprocessing

Adapter Foursquare TSMC2014 → dual-view (sparse URM for CF baselines + per-interaction
DataFrame for CARS/proposed). Same user/item indexing, same train/val/test partition.

**Not yet implemented.** Will land in the next brief (preprocessing step).

Expected outputs in `data/processed/<city>/`:
- `train_urm.npz`, `val_urm.npz`, `test_urm.npz` — scipy sparse for CF baselines
- `train_df.parquet`, `val_df.parquet`, `test_df.parquet` — per-interaction with contextual features
- `mappings.json` — user2id, item2id, category2id, hparams
