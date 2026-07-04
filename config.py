"""
config.py — Single source of truth for all configurable parameters.

No numeric literal, string constant, or boolean flag that controls behavior
may appear anywhere else in the codebase. If a value is needed elsewhere, it
is imported from this module.
"""

import os
from datetime import time, date

# ─── UNDERLIERS ───────────────────────────────────────────────────────────────
# Which underliers to simulate. Add "FINNIFTY" to extend beyond assignment scope.
UNDERLIERS = ["NIFTY", "BANKNIFTY"]

# Lot sizes per underlier (NSE standard, November 2022)
LOT_SIZES = {
    "NIFTY":     50,
    "BANKNIFTY": 25,
    "FINNIFTY":  40,
}

# Strike price intervals on NSE (points between consecutive listed strikes)
STRIKE_INTERVALS = {
    "NIFTY":     50,
    "BANKNIFTY": 100,
    "FINNIFTY":  50,
}

# ─── SIMULATION PARAMETERS ────────────────────────────────────────────────────
# Time step between strategy evaluations, in seconds.
# At 1s this matches the assignment. Increase to 5/10/30/60 for faster runs.
TIMESTEP_SECONDS = int(os.environ.get("MFT_TIMESTEP_SECONDS", "1"))

# Trading session bounds (IST, 24h format)
MARKET_OPEN  = time(9, 15, 0)
MARKET_CLOSE = time(15, 30, 0)

# Minutes before MARKET_CLOSE to trigger end-of-day flatten.
# 0 means flatten exactly at 15:30:00. Increase to 1 or 5 to flatten early.
EOD_FLATTEN_BUFFER_MINUTES = 0

# Backtest date range (used as default; CLI flags override these)
DEFAULT_START_DATE = date(2022, 11, 1)
DEFAULT_END_DATE   = date(2022, 11, 30)

# ─── POSITION LIMITS ──────────────────────────────────────────────────────────
# Maximum number of lots held per instrument (CE and PE counted separately)
MAX_POSITION_LOTS = 1

# ─── PRICE LOOKUP ─────────────────────────────────────────────────────────────
# When no tick exists at the exact current second, look back this many seconds
# for the last known price. None = look back to market open (no limit).
# Set to e.g. 300 to treat prices stale after 5 minutes.
PRICE_STALENESS_THRESHOLD_SECONDS = None

# ─── STRIKE SELECTION ─────────────────────────────────────────────────────────
# When futures price is exactly equidistant between two strikes,
# which to prefer. "up" = higher strike, "down" = lower strike.
STRIKE_TIE_BREAK = os.environ.get("MFT_STRIKE_TIE_BREAK", "up")

# ─── TRANSACTION COSTS ────────────────────────────────────────────────────────
# Set to 0.0 for the base case (as assignment does not specify costs).
# Used in sensitivity analysis by the hyperparameter tuner.
BROKERAGE_PER_LOT_PER_LEG = float(os.environ.get("MFT_BROKERAGE", "0.0"))
STT_RATE                  = float(os.environ.get("MFT_STT_RATE", "0.0"))
EXCHANGE_CHARGES_RATE     = float(os.environ.get("MFT_EXCHANGE_CHARGES", "0.0"))

# ─── DATA ─────────────────────────────────────────────────────────────────────
# Real NSE data lives at allData/allData/NSE_YYYYMMDD/.
# Override via MFT_DATA_ROOT env var (e.g. to use synthetic data from allData_test/).
DATA_ROOT          = os.environ.get("MFT_DATA_ROOT", "allData/allData/")
DATE_FOLDER_PREFIX = "NSE_"
DATE_FOLDER_FORMAT = "%Y%m%d"          # e.g. NSE_20221101
# Real NSE data uses "Options" and "Futures (Continuous)" as subfolder names.
OPTIONS_SUBFOLDER  = "Options"
FUTURES_SUBFOLDER  = "Futures (Continuous)"
FUTURES_SERIES     = "I"               # Only near-month (-I) futures used

# ─── MEMORY OPTIMISATION ──────────────────────────────────────────────────────
# Real NSE data has 100–600 option strikes per day per underlier. Loading all
# of them would consume ~1.6 GB RAM. Instead, we only load options within
# ATM ± ATM_WINDOW_STRIKES of the intraday futures price range.
# ATM_WINDOW_STRIKES=10 covers ±500 pts (NIFTY) / ±1000 pts (BANKNIFTY) —
# safely above the 99th-percentile 1-day move.
# Set to None to disable windowing and load all strikes (matches old behaviour).
_atm_raw = os.environ.get("MFT_ATM_WINDOW", "10")
ATM_WINDOW_STRIKES: int | None = None if _atm_raw.lower() in ("none", "0", "") else int(_atm_raw)

# ─── CACHE ────────────────────────────────────────────────────────────────────
CACHE_DIR               = os.environ.get("MFT_CACHE_DIR", ".cache/")
CACHE_FORMAT            = "parquet"
CHECKPOINT_EVERY_N_DAYS = 5        # Flush partial results every N trading days

# ─── OUTPUT ───────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR    = "results/"
LOG_LEVEL             = os.environ.get("MFT_LOG_LEVEL", "INFO")
PROGRESS_UPDATE_EVERY = 100        # Print progress every N seconds of simulation

# ─── ANALYTICS ────────────────────────────────────────────────────────────────
# Date to use for intraday sample charts (must be within simulation range)
SAMPLE_INTRADAY_DATE = date(2022, 11, 3)

# ─── HYPERPARAMETER TUNER ─────────────────────────────────────────────────────
TUNER_N_WORKERS        = int(os.environ.get("MFT_TUNER_N_WORKERS", "4"))
TUNER_RANDOM_SEED      = 42
TUNER_TRAIN_FRACTION   = 0.5
TUNER_DEFAULT_N_TRIALS = int(os.environ.get("MFT_TUNER_TRIALS", "50"))
TUNER_OVERFITTING_THRESHOLD = 0.5

# ─── STRATEGY REGISTRY ────────────────────────────────────────────────────────
# Maps strategy name (CLI value) → fully qualified class path.
STRATEGY_REGISTRY = {
    "atm_straddle": "strategies.atm_straddle.ATMStraddle",
}
