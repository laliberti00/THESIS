"""experiments/run_xsage.py — X-SAGE go/no-go driver (block step02b).

Stages (see brief §5):
    A  build situations (L0 → L2)                   + §6.1 artefacts
    B  cardinal check 1: per-situation fairness     + §6.4 artefacts
    C  cardinal check 2: projection (L3) + F1       + §6.2 artefacts
    D  three-way comparison + modulation log        + §6.3 artefacts
    E  optional, time-boxed exploratory variant

CLI:
    python -m experiments.run_xsage --city NYC --stage A -v
    python -m experiments.run_xsage --city NYC --stage all -v
    python -m experiments.run_xsage --city both --stage D --kappa 0.0 0.1 0.25 0.5 1.0

All artefacts land under ``outputs/<city>/xsage/<level>/``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", choices=["NYC", "TKY", "both"], default="NYC")
    parser.add_argument("--stage", choices=["A", "B", "C", "D", "E", "all"],
                        default="A")
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--n", type=int, default=5,
                        help="recency window size for profile m (eq.4)")
    parser.add_argument("--H", type=int, default=2,
                        help="reachability horizon for intent (eq.5)")
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--eps", type=float, default=None,
                        help="rough-boundary threshold; auto-tuned to boundary "
                             "fraction ∈ [10%, 30%] if omitted")
    parser.add_argument("--kappa", type=float, nargs="+",
                        default=[0.0, 0.1, 0.25, 0.5, 1.0],
                        help="κ_S sweep for Stage D")
    parser.add_argument("--combiner", choices=["harmonic", "additive"],
                        default="harmonic",
                        help="Stage D combiner: harmonic (eq.15, original) "
                             "or additive (round-2 1.5).")
    parser.add_argument("--intent-mode", choices=["hard", "all", "soft_topr"],
                        default="hard",
                        help="Intent vector mode (round-2 1.4). 'hard' "
                             "(default) is the attractor-cutoff of eq.5; "
                             "'all' keeps every macro weighted by "
                             "reachability; 'soft_topr' keeps top-r.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(message)s")

    cities = ["NYC", "TKY"] if args.city == "both" else [args.city]
    stages = ["A", "B", "C", "D", "E"] if args.stage == "all" else [args.stage]

    for city in cities:
        # Round-2 1.4: intent_mode != "hard" goes into a sibling directory so
        # the default-mode artefacts are preserved.
        out_root = REPO_ROOT / "outputs" / city / (
            "xsage" if args.intent_mode == "hard"
            else f"xsage_intent_{args.intent_mode}"
        )
        out_root.mkdir(parents=True, exist_ok=True)
        for stage in stages:
            print(f"\n>>> [{city}] Stage {stage}")
            # The orchestrator's stage runners are imported lazily so that a
            # half-implemented stage does not import-fail other stages.
            if stage == "A":
                from pipeline.step02_models.xsage.orchestrator import run_stage_a
                run_stage_a(city, out_root, args)
            elif stage == "B":
                from pipeline.step02_models.xsage.orchestrator import run_stage_b
                run_stage_b(city, out_root, args)
            elif stage == "C":
                from pipeline.step02_models.xsage.orchestrator import run_stage_c
                run_stage_c(city, out_root, args)
            elif stage == "D":
                from pipeline.step02_models.xsage.orchestrator import run_stage_d
                run_stage_d(city, out_root, args)
            elif stage == "E":
                from pipeline.step02_models.xsage.orchestrator import run_stage_e
                run_stage_e(city, out_root, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
