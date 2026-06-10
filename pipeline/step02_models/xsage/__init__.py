"""X-SAGE — *step02b* situation-aware layer on top of the floor (step02a).

The layer **models** situations from intent + context (L0→L2), **projects** how
they evolve (L3), and **modulates** a context-blind POI backbone *only when the
situation is certain* (recommendation). It is SARE-faithful by design: the
backbone is context-blind, so context only enters the model through the
interpretable X-SAGE head.

Levels:
    L0 — :mod:`l0_sensing`        causal recent window per request
    L1 — :mod:`l1_perception`     learned contribution functions c̃, macro
                                   transition W, attractors, recency profile
                                   m, intent vector e
    L2 — :mod:`l2_comprehension`  descriptor v = [c̃ ‖ e]; rough k-means
                                   (Lingras–West) → core/boundary situations
    L3 — :mod:`l3_projection`     situation transition matrix T; next-situation
                                   prediction; dynamic fairness curve

Application layers:
    :mod:`recommendation`         backbone p_B, situational score s_S,
                                   confidence-weighted harmonic combiner,
                                   list-displacement Δ_K
    :mod:`metrics`                LT, KL fairness, list displacement,
                                   next-situation F1
    :mod:`viz`                    artifacts under ``outputs/<city>/xsage/``

The orchestrator ``run_xsage_for_city`` lives here once the levels are
implemented; the CLI wrapper is :mod:`experiments.run_xsage`.

Every level module is a pure function with explicit inputs and outputs;
levels never reach into each other's internals.
"""
