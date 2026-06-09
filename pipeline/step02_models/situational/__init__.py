"""step02b — proposed *Situation-Aware* recommender (the thesis contribution).

Internal organisation mirrors the Situation-Awareness loop:

    perception (L1)  →  comprehension (L2)  →  projection (L3)

* ``perception.context``    : cyclic time encoding + previous-location geohash
                              embedding → context vector **c**.
* ``perception.intent``     : causal recency window over the user's history →
                              intent vector **e**.
* ``comprehension.situation`` : bilinear gate over (**c**, **e**) → softmax
                              ``π`` over ``K`` latent situations (argmax → z).
* ``projection.fm``         : Rendle-style FM over {user, item, candidate
                              ``cat_macro``, request cyclic time} → base score
                              ``ŷ0``.
* ``projection.modulation`` : adds per-situation category bias and
                              ``s_k``-modulated user-item interaction → ``ŷ``.

The whole stack is assembled in :mod:`pipeline.step02_models.situational.model`.
Training, ranking and per-user ``.npz`` export live in :mod:`trainer` and
:mod:`ranker`. The CLI wrapper is :mod:`experiments.run_situational`.

The orchestrator routine ``run_situational_for_city`` lives in this package
once the individual phases are implemented; see the brief in the
``step02b-goNogo`` branch for the contract.
"""
