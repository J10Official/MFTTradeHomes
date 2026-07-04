#!/usr/bin/env bash
# run_tune.sh — Run hyperparameter search.
#
# Usage:
#     ./run_tune.sh                              # random search, 10 trials, default range
#     ./run_tune.sh --search-type grid           # grid search
#     ./run_tune.sh --n-trials 25                # random search, 25 trials
#     ./run_tune.sh --start 2022-11-01 --end 2022-11-15   # short range
#
# All extra args are forwarded to `python run_backtest.py tune`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
DATA_ROOT="${MFT_DATA_ROOT:-allData}"

# Verify deps
if ! "$PYTHON" -c "import pandas, click, pyarrow" >/dev/null 2>&1; then
    echo "==> Installing dependencies..."
    "$PYTHON" -m pip install --quiet -r requirements.txt
fi

# Generate synthetic data if missing
if [ ! -d "$DATA_ROOT" ] || [ -z "$(ls -A "$DATA_ROOT" 2>/dev/null)" ]; then
    echo "==> No data found at $DATA_ROOT. Generating synthetic data..."
    "$PYTHON" scripts/generate_synthetic_data.py \
        --output "$DATA_ROOT" \
        --start 2022-11-01 --end 2022-11-30 \
        --underliers NIFTY,BANKNIFTY \
        --timestep 60 --seed 42
fi

echo "==> Running hyperparameter tuner..."
exec "$PYTHON" run_backtest.py tune \
    --start 2022-11-01 --end 2022-11-30 \
    --underliers NIFTY,BANKNIFTY \
    --data-root "$DATA_ROOT" \
    "$@"
