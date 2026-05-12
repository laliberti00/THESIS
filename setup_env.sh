#!/usr/bin/env bash
# Setup script for the thesis CPU environment (non-neural baselines).
#
# Creates a Python venv in ./.venv and installs requirements-cpu.txt.
# After running this, activate the env with:
#     source .venv/bin/activate
#
# For the full GPU setup (DCCF + BIGCF), see README.md "Environment / GPU".

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -d "$VENV_DIR" ]]; then
    echo ">>> Creating virtualenv in $VENV_DIR using $PYTHON_BIN"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo ">>> Re-using existing virtualenv in $VENV_DIR"
fi

echo ">>> Upgrading pip"
"$VENV_DIR/bin/pip" install --upgrade pip --quiet

echo ">>> Installing requirements-cpu.txt"
"$VENV_DIR/bin/pip" install -r requirements-cpu.txt

echo
echo ">>> Done. To activate the env, run:"
echo "    source $VENV_DIR/bin/activate"
echo
echo ">>> Quick sanity check:"
"$VENV_DIR/bin/python" -c "import numpy, scipy, pandas, sklearn; print('numpy', numpy.__version__, '/ scipy', scipy.__version__, '/ pandas', pandas.__version__, '/ sklearn', sklearn.__version__)"
