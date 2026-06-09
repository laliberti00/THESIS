# step02 / cars — Context-Aware Recommenders

CARS that extend `engine/Recommenders/` with contextual features.

**Not yet implemented.** Planned files (later brief):
- `fm_context.py` — FMRecommender (extends `engine.Recommenders.FactorizationMachines._BPRMFBase`)
  with one-hot contextual features + Rendle 2010 second-order interactions
- `camf_context.py` — CAMFRecommender with additive contextual bias terms
- `tfm.py` — Tensor Factorization Machine (Karatzoglou et al. 2010)

All PyTorch CPU/MPS, BPR pairwise loss. Compatible with `engine.Evaluation.EvaluatorHoldout`
via `_compute_item_score(user_id_array, items_to_compute=None)`.
