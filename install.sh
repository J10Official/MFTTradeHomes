#!/usr/bin/env bash
# install.sh — Install dependencies into the current Python environment.
#
# Usage:
#     ./install.sh                # install from requirements.txt
#     ./install.sh --dev          # also install pytest, ipython, etc. for development

set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "==> Using: $($PYTHON --version)"
echo "==> Installing dependencies from requirements.txt..."
"$PYTHON" -m pip install -r requirements.txt

if [ "${1:-}" = "--dev" ]; then
    echo "==> Installing dev dependencies (pytest, ipython)..."
    "$PYTHON" -m pip install pytest ipython
fi

echo "==> Done. Verify with: python -c 'import pandas, numpy, click, matplotlib, pyarrow; print(\"OK\")'"
