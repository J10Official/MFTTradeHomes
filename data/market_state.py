"""
data/market_state.py — Convert raw tick DataFrames into per-second MarketContext.

The most algorithmically important module: implements the anti-look-ahead-bias
forward-fill via pd.merge_asof(direction='backward').
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.portfolio import PortfolioSnapshot


logger = logging.getLogger(__name__)


# ─── MarketContext (moved to top to avoid self-referential local import) ──────


@dataclass
class MarketContext:
    """
    Complete picture of the market and portfolio at one simulation timestep.
    This is the ONLY input the strategy receives. The strategy MUST NOT access
    any external data or state.
    """
    timestamp:          datetime
    underlier:          str
    futures_price:      float          # Last known near-month futures price
    atm_strike:         int            # Pre-computed: nearest strike to futures_price
    nearest_expiry:     date           # Expiry date being traded today

    # All option prices available this second (last-known-price, forward-filled).
    # Key: (strike: int, option_type: str) e.g. (18050, "CE")
    # Value: last known premium in ₹ (or None if no price available).
    option_prices: Dict[Tuple[int, str], Optional[float]]

    # Read-only portfolio view
    portfolio: PortfolioSnapshot

    # Available strikes for this underlier/expiry (sorted ascending)
    available_strikes: List[int]

    def get_price(self, strike: int, option_type: str) -> Optional[float]:
        """Convenience accessor. Returns None if no price data available."""
        return self.option_prices.get((strike, option_type))

    def get_atm_price(self, option_type: str) -> Optional[float]:
        """Price of the ATM CE or PE this second."""
        return self.get_price(self.atm_strike, option_type)


# ─── canonical index ──────────────────────────────────────────────────────────


def build_canonical_index(
    trading_date: date,
    market_open: time,
    market_close: time,
    timestep_seconds: int,
) -> pd.DatetimeIndex:
    """Generate a complete second-by-second timestamp index for the trading day."""
    start = datetime.combine(trading_date, market_open)
    end   = datetime.combine(trading_date, market_close)
    return pd.date_range(start=start, end=end, freq=f"{timestep_seconds}s")


# ─── tick → second resampling ─────────────────────────────────────────────────


def resample_to_seconds(
    tick_df: pd.DataFrame,
    trading_date: date,
    market_open: time,
    market_close: time,
    timestep_seconds: int,
    staleness_threshold_seconds: Optional[int],
) -> pd.Series:
    """
    Convert irregular tick data to a regular time series at `timestep_seconds` resolution.

    Algorithm:
      1. Build canonical timestamp index from market_open to market_close.
      2. For each canonical second T, find the most recent tick with timestamp <= T
         using pd.merge_asof(direction='backward'). This STRICTLY prevents look-ahead.
      3. If staleness_threshold_seconds is set and the most recent tick is older
         than that threshold: treat as NaN.
      4. Return pd.Series indexed by canonical timestamps.

    CRITICAL: pd.merge_asof with direction='backward' is the only safe way to fill
    forward in time. NEVER use direction='forward' or 'nearest' — those would
    leak future data into the current second.
    """
    if tick_df is None or tick_df.empty:
        idx = build_canonical_index(trading_date, market_open, market_close, timestep_seconds)
        return pd.Series(np.nan, index=idx, name="price")

    canon_idx = build_canonical_index(trading_date, market_open, market_close, timestep_seconds)
    canon_df = pd.DataFrame({"timestamp": canon_idx})

    # Make sure tick_df is sorted by index.
    ticks = tick_df[["price"]].copy()
    ticks = ticks[~ticks.index.isna()]
    ticks = ticks.sort_index()
    if ticks.empty:
        return pd.Series(np.nan, index=canon_idx, name="price")
    ticks_reset = ticks.reset_index().rename(columns={ticks.index.name or "index": "timestamp"})

    merged = pd.merge_asof(
        canon_df,
        ticks_reset,
        on="timestamp",
        direction="backward",
    )

    result = merged.set_index("timestamp")["price"]

    # Apply staleness threshold if requested.
    if staleness_threshold_seconds is not None:
        # For each canonical second, check the time gap since the last tick.
        # If gap > threshold → NaN.
        tick_times = pd.DataFrame({"timestamp": ticks.index})
        last_tick = pd.merge_asof(
            canon_df, tick_times, on="timestamp", direction="backward"
        )
        last_tick_time = last_tick["timestamp"].values
        gap_seconds = (canon_idx.values - last_tick_time).astype("timedelta64[s]").astype(np.int64)
        stale_mask = (last_tick_time == pd.NaT.value) | (gap_seconds > staleness_threshold_seconds)
        result = result.mask(stale_mask, np.nan)

    return result


# ─── ATM strike selection ─────────────────────────────────────────────────────


def find_atm_strike(
    futures_price: float,
    available_strikes: List[int],
    tie_break: str = "up",
) -> int:
    """
    Find the strike in available_strikes closest to futures_price.

    Args:
        futures_price: Current futures price (the underlying reference).
        available_strikes: List of strike prices available today.
        tie_break: "up" = prefer higher strike when equidistant.
                   "down" = prefer lower strike when equidistant.

    Raises:
        ValueError: if available_strikes is empty.
    """
    if not available_strikes:
        raise ValueError("available_strikes is empty")
    if futures_price is None or (isinstance(futures_price, float) and np.isnan(futures_price)):
        raise ValueError("futures_price is NaN")

    distances = {s: abs(s - futures_price) for s in available_strikes}
    min_dist = min(distances.values())
    candidates = [s for s, d in distances.items() if d == min_dist]

    if len(candidates) == 1:
        return candidates[0]

    candidates.sort()
    if tie_break == "up":
        return candidates[-1]
    elif tie_break == "down":
        return candidates[0]
    else:
        raise ValueError(f"Invalid tie_break value: {tie_break!r}")


# ─── MarketContext assembly (pandas version — kept for tests) ─────────────────


def build_market_context(
    timestamp: datetime,
    underlier: str,
    nearest_expiry: date,
    futures_series: pd.Series,
    options_series: Dict[Tuple[int, str], pd.Series],
    portfolio_snapshot: PortfolioSnapshot,
    tie_break: str,
    available_strikes: Optional[List[int]] = None,
) -> Optional[MarketContext]:
    """
    Assemble a MarketContext for one specific timestamp.

    Returns None if futures price is NaN at this timestamp (cannot make decisions).

    This is the original pandas-based implementation. It is kept for backward
    compatibility with tests. The production hot loop uses build_market_context_fast().
    """
    if timestamp not in futures_series.index:
        return None
    futures_price = futures_series.loc[timestamp]
    if futures_price is None or (isinstance(futures_price, float) and np.isnan(futures_price)):
        return None

    if available_strikes is None:
        available_strikes = sorted({k[0] for k in options_series.keys()})
    if not available_strikes:
        return None

    try:
        atm_strike = find_atm_strike(futures_price, available_strikes, tie_break)
    except ValueError:
        return None

    # Snapshot option prices at this exact timestamp.
    option_prices: Dict[Tuple[int, str], Optional[float]] = {}
    for (strike, otype), series in options_series.items():
        if timestamp in series.index:
            p = series.loc[timestamp]
            if p is None or (isinstance(p, float) and np.isnan(p)):
                option_prices[(strike, otype)] = None
            else:
                option_prices[(strike, otype)] = float(p)
        else:
            option_prices[(strike, otype)] = None

    return MarketContext(
        timestamp=timestamp,
        underlier=underlier,
        futures_price=float(futures_price),
        atm_strike=atm_strike,
        nearest_expiry=nearest_expiry,
        option_prices=option_prices,
        portfolio=portfolio_snapshot,
        available_strikes=available_strikes,
    )


# ─── Fast numpy-based context builder (production hot loop) ──────────────────


def build_market_context_fast(
    tick_idx: int,
    timestamp: datetime,
    underlier: str,
    nearest_expiry: date,
    futures_arr: np.ndarray,
    options_arrs: Dict[Tuple[int, str], np.ndarray],
    portfolio_snapshot: PortfolioSnapshot,
    tie_break: str,
    available_strikes: List[int],
) -> Optional[MarketContext]:
    """
    Fast MarketContext builder for the per-tick simulation loop.

    Unlike build_market_context(), this function works entirely with pre-converted
    numpy arrays and uses O(1) integer indexing instead of O(log n) pandas
    datetime index lookups. All data must be pre-aligned to the canonical index
    (i.e. produced by resample_to_seconds()) before calling this function.

    Args:
        tick_idx:           Integer index into the canonical timestamp array.
        timestamp:          The datetime at this tick (for log rows / order timestamps).
        underlier:          "NIFTY", "BANKNIFTY", etc.
        nearest_expiry:     Expiry date being traded today.
        futures_arr:        Pre-converted numpy array from futures_series.values.
        options_arrs:       Pre-converted {(strike, type): np.ndarray} from options_series.
        portfolio_snapshot: Read-only portfolio snapshot.
        tie_break:          Strike tie-break rule ("up" or "down").
        available_strikes:  Pre-sorted list of available strikes.

    Returns:
        MarketContext or None if futures price is NaN at this tick.
    """
    futures_price = futures_arr[tick_idx]
    if np.isnan(futures_price):
        return None
    if not available_strikes:
        return None

    try:
        atm_strike = find_atm_strike(float(futures_price), available_strikes, tie_break)
    except ValueError:
        return None

    # Build option_prices dict using O(1) numpy array reads.
    option_prices: Dict[Tuple[int, str], Optional[float]] = {}
    for (strike, otype), arr in options_arrs.items():
        v = arr[tick_idx]
        option_prices[(strike, otype)] = None if np.isnan(v) else float(v)

    return MarketContext(
        timestamp=timestamp,
        underlier=underlier,
        futures_price=float(futures_price),
        atm_strike=atm_strike,
        nearest_expiry=nearest_expiry,
        option_prices=option_prices,
        portfolio=portfolio_snapshot,
        available_strikes=available_strikes,
    )
