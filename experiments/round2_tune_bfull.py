"""Round-2 1.1 — tune B_full (light HP sweep, val R@20 early stop).

Usage:
    python -m experiments.round2_tune_bfull --city NYC -v
    python -m experiments.round2_tune_bfull --city TKY -v
    python -m experiments.round2_tune_bfull --city both -v

Overwrites ``outputs/<city>/xsage/backbone/Bfull.scores.npy`` with the tuned
test-time scores. Stage D consumers (matched-pair, three-way) automatically
pick up the new cache.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step02_models.xsage.bfull_tuning import tune_and_refit
from pipeline.step02_models.xsage.orchestrator import _load_city


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", default="NYC",
                        help="Built-in: NYC, TKY, both. Arbitrary strings ok "
                             "if data/processed/<city>/ exists (e.g. TKY_BAL).")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cities = ["NYC", "TKY"] if args.city == "both" else [args.city]
    for city in cities:
        print(f"\n>>> tuning B_full on {city}")
        out = tune_and_refit(city, _load_city,
                                max_epochs=args.max_epochs,
                                patience=args.patience,
                                seed=args.seed, verbose=args.verbose)
        print(f"   B_full tuned for {city}: winner = {out['best_cell']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
