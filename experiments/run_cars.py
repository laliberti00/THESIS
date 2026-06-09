"""experiments/run_cars.py — entry point for the context-aware recommenders.

STUB. Once pipeline.step02_models.cars is implemented (CARS extending
engine.Recommenders.FactorizationMachines._BPRMFBase), this script will:
  - load Foursquare URM + per-interaction context features from
    data/processed/<city>/,
  - train each requested CARS model,
  - dump per-user metrics to outputs/cars/<model>.npz.
"""

import sys


def main() -> int:
    print("experiments/run_cars.py — STUB.")
    print("  pipeline/step02_models/cars/ is not implemented yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
