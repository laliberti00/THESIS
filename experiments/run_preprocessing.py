"""experiments/run_preprocessing.py — CLI for step01 Foursquare preprocessing.

Examples:
    # Single city
    python -m experiments.run_preprocessing --city NYC
    python -m experiments.run_preprocessing --city TKY

    # Both cities in one shot (default)
    python -m experiments.run_preprocessing

    # Override params
    python -m experiments.run_preprocessing --city NYC --k-core 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step01_preprocessing import preprocess_tsmc2014

CITY_TO_RAW = {
    "NYC": REPO_ROOT / "data" / "raw" / "dataset_TSMC2014_NYC.txt",
    "TKY": REPO_ROOT / "data" / "raw" / "dataset_TSMC2014_TKY.txt",
}
TAXONOMY = REPO_ROOT / "config" / "foursquare_legacy_taxonomy.json"
OUT_BASE = REPO_ROOT / "data" / "processed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", choices=["NYC", "TKY", "both"], default="both")
    parser.add_argument("--k-core", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show per-iteration k-core counts and feature steps")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    cities = ["NYC", "TKY"] if args.city == "both" else [args.city]
    for city in cities:
        raw = CITY_TO_RAW[city]
        if not raw.exists():
            print(f"ERROR: raw TSV not found at {raw}", file=sys.stderr)
            return 2
        out_dir = OUT_BASE / city
        print(f"\n>>> Preprocessing {city} → {out_dir}")
        result = preprocess_tsmc2014(
            raw_tsv=raw,
            out_dir=out_dir,
            taxonomy_path=TAXONOMY,
            k_core=args.k_core,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        print(result.short_report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
