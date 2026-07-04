#!/usr/bin/env bash
# run_basic.sh — One-command run of the ATM straddle backtest.
#
# Modes:
#   ./run_basic.sh              Full November 2022 backtest, NIFTY + BANKNIFTY.
#   ./run_basic.sh --demo       3-day quick run (Nov 1-3) at 1s cadence. ~2-4 min.
#                               All core mechanics (entry, rolling, EOD flatten,
#                               PnL, plots) are exercised identically to the full run.
#   ./run_basic.sh --no-cache   Force re-run, ignore any cached checkpoint.
#   ./run_basic.sh --timestep N Override timestep (seconds). Default: 1.
#
# Environment variables:
#   MFT_DATA_ROOT   Path to NSE data root  (default: allData/allData)
#   MFT_OUTPUT_DIR  Output directory        (default: results/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Step 1: verify Python & dependencies ────────────────────────────────────

SYS_PYTHON="${PYTHON:-python3}"
if ! command -v "$SYS_PYTHON" >/dev/null 2>&1; then
    echo "ERROR: $SYS_PYTHON not found in PATH. Install Python 3.11+ first." >&2
    exit 1
fi

# Auto-create a virtual environment to avoid PEP 668 restrictions on
# Debian/Ubuntu systems with Python 3.12+ (externally-managed-environment).
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "==> Creating virtual environment at .venv ..."
    "$SYS_PYTHON" -m venv "$VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"

PY_VERSION="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJOR="${PY_VERSION%%.*}"
PY_MINOR="${PY_VERSION#*.}"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    echo "WARNING: Python $PY_VERSION detected — recommended 3.11+. Continuing anyway." >&2
fi

echo "==> [1/4] Verifying Python dependencies..."
if ! "$PYTHON" -c "import pandas, numpy, click, matplotlib, pytest, pyarrow" >/dev/null 2>&1; then
    echo "    Missing dependencies. Installing from requirements.txt..."
    "$VENV_DIR/bin/pip" install --quiet -r requirements.txt
fi
echo "    OK"

# ─── Parse args ──────────────────────────────────────────────────────────────

DEMO_MODE=0
SKIP_DATA_GEN=0
TIMESTEP="${MFT_TIMESTEP_SECONDS:-1}"
NO_CACHE_FLAG=""
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --demo)          DEMO_MODE=1 ;;
        --skip-data-gen) SKIP_DATA_GEN=1 ;;
        --timestep=*)    TIMESTEP="${arg#*=}" ;;
        --no-cache)      NO_CACHE_FLAG="--no-cache" ;;
        *)               EXTRA_ARGS+=("$arg") ;;
    esac
done

# Handle "--timestep N" (space-separated) form
for ((i=1; i<=$#; i++)); do
    if [ "${!i}" = "--timestep" ]; then
        next=$((i+1))
        TIMESTEP="${!next:-$TIMESTEP}"
    fi
done

if [ "$DEMO_MODE" -eq 1 ]; then
    START_DATE="2022-11-01"
    END_DATE="2022-11-03"
    OUTPUT_DIR="${MFT_OUTPUT_DIR:-results_demo}"
    echo "==> Demo mode: running Nov 1-3, 2022 at ${TIMESTEP}s timestep (~2-4 min)."
else
    START_DATE="2022-11-01"
    END_DATE="2022-11-30"
    OUTPUT_DIR="${MFT_OUTPUT_DIR:-results}"
fi

# ─── Step 2: data ────────────────────────────────────────────────────────────

DATA_ROOT="${MFT_DATA_ROOT:-allData/allData}"
echo "==> [2/4] Checking for NSE data at: $DATA_ROOT"
if [ ! -d "$DATA_ROOT" ] || [ -z "$(ls -A "$DATA_ROOT" 2>/dev/null)" ]; then
    if [ "$SKIP_DATA_GEN" -eq 1 ]; then
        echo "ERROR: data not found at $DATA_ROOT and --skip-data-gen was set." >&2
        exit 1
    fi
    echo "    No data found. Generating synthetic NSE-format data (~30s)..."
    "$PYTHON" scripts/generate_synthetic_data.py \
        --output "$DATA_ROOT" \
        --start 2022-11-01 --end 2022-11-30 \
        --underliers NIFTY,BANKNIFTY \
        --timestep 60 --seed 42
fi
echo "    OK"

# ─── Step 3: run the backtest ────────────────────────────────────────────────

echo "==> [3/4] Running ATM straddle backtest (NIFTY + BANKNIFTY, ${START_DATE} → ${END_DATE}, ${TIMESTEP}s)..."
"$PYTHON" run_backtest.py run \
    --start "$START_DATE" \
    --end   "$END_DATE" \
    --underliers NIFTY,BANKNIFTY \
    --timestep "$TIMESTEP" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    $NO_CACHE_FLAG \
    "${EXTRA_ARGS[@]}"

# ─── Step 4: done ────────────────────────────────────────────────────────────

echo "==> [4/4] Done."
echo
echo "Results directory: $OUTPUT_DIR/"
echo "  - metrics_summary.json   ← all PnL/risk/trade metrics"
echo "  - event_log.parquet      ← per-second event log"
echo "  - trade_log.parquet      ← per-trade execution log"
echo "  - *.png                  ← cumulative PnL, daily bars, intraday sample,"
echo "                              roll frequency, premium decay, drawdown"
echo
echo "To run unit tests:              ./run_tests.sh"
echo "To run hyperparameter tuning:   ./run_tune.sh"
