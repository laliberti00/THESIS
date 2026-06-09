"""experiments/run_statistical_validation.py — entry to the statistical pipeline.

Thin wrapper around `pipeline.step04_statistical_validation.statistical_validation`.
Forwards CLI flags. The underlying module already has its own argparse-based
``main()``; we re-use it directly so there is exactly one source of truth.

Usage:
    python -m experiments.run_statistical_validation \\
        --per-user-dir outputs/baselines/per_user \\
        --metric RECALL --cutoff 20 \\
        --out-descr outputs/stats/descriptive.tsv \\
        --out-tests outputs/stats/tests.tsv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.step04_statistical_validation.statistical_validation import main as _main


if __name__ == "__main__":
    sys.exit(_main() or 0)
