#!/usr/bin/env bash
# run_tests.sh — Run the unit test suite.
#
# Usage:
#     ./run_tests.sh                # run all tests
#     ./run_tests.sh -v             # verbose
#     ./run_tests.sh tests/test_parser.py   # specific file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"

# Verify deps
if ! "$PYTHON" -c "import pytest" >/dev/null 2>&1; then
    echo "pytest not installed. Installing..."
    "$PYTHON" -m pip install --quiet -r requirements.txt
fi

echo "==> Running unit tests..."
"$PYTHON" -m pytest "$@"
echo
echo "==> All tests passed."
