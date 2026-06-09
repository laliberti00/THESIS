"""experiments/run_situational.py — step02b go/no-go driver.

Runs the proposed situation-aware FM (and its matched-pair ablations V0/V1/V2)
on NYC + TKY and emits per-user ``.npz`` files in the **same** standard schema
as the floor — so step04 statistical_validation runs on the pair without any
changes.

Examples:
    python -m experiments.run_situational --city NYC -v
    python -m experiments.run_situational --city TKY -v
    python -m experiments.run_situational --city both --seeds 42 13 2024 \
        --k-sweep 2 3 4 6 8

Outputs land under ``outputs/<city>/situational/`` (gitignored):
    V0_seed<s>.npz / V1_seed<s>.npz / V2_seed<s>.npz
    situation_report.json
    RESULTS.md (with the **VERDICT**: GREEN / YELLOW / RED)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.situational.orchestrator import (
    DEFAULT_K_SWEEP, DEFAULT_SEEDS, run_for_city,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", choices=["NYC", "TKY", "both"], default="both")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--k-sweep", nargs="+", type=int,
                        default=list(DEFAULT_K_SWEEP))
    parser.add_argument("--no-refit", action="store_true",
                        help="Skip the train+val refit step (faster).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    cities = ["NYC", "TKY"] if args.city == "both" else [args.city]
    overall = {}
    for city in cities:
        report = run_for_city(
            city,
            seeds=tuple(args.seeds),
            K_sweep=tuple(args.k_sweep),
            with_refit=not args.no_refit,
            verbose=args.verbose,
        )
        overall[city] = report

    # Final summary
    print("\n" + "=" * 70)
    print("STEP02b GO/NO-GO — summary")
    print("=" * 70)
    for city, rep in overall.items():
        v = rep["verdict"]
        print(f"  [{city}]  {v['label']:6s}  ΔR@20={v['delta_R20']:+.4f}  "
              f"ΔNDCG@20={v['delta_NDCG20']:+.4f}  "
              f"distinct={v['distinct']}  specialised={v['specialised']}")
        print(f"           reason: {v['reason']}")
        out_dir = REPO_ROOT / "outputs" / city / "situational"
        print(f"           outputs: {out_dir}")

    # Save a combined summary JSON
    summary_path = REPO_ROOT / "outputs" / "step02b_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({c: r["verdict"] for c, r in overall.items()}, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
