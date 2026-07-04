#!/usr/bin/env python3
"""
scripts/generate_synthetic_data.py — Generate realistic synthetic NSE-format data.

Why: The assignment data lives on Google Drive (one-month NSE options/futures
CSVs). Reviewers of THIS submission need to be able to run the backtest
end-to-end without that data. This script generates realistic synthetic data
that follows the EXACT same format as the assignment data:

  allData/NSE_YYYYMMDD/options/{UNDERLIER}{YYMMDD}{STRIKE}{CE|PE}.csv
  allData/NSE_YYYYMMDD/futures/{UNDERLIER}-I.csv

Each CSV has columns: Date, Time, Price, Volume, Open Interest
  - Date format: DD-MM-YYYY (Indian NSE convention)
  - Time format: HH:MM:SS
  - Price: simulated futures price for futures file; simulated option premium
           (Black-Scholes-like) for options files.
  - Volume: random integer per tick
  - Open Interest: slowly varying integer

Synthetic data IS NOT real market data — but it follows the same statistical
shape (geometric brownian motion for the underlying, theta-decaying premiums
for options, plausible strike grid around the futures price).

Usage:
    python scripts/generate_synthetic_data.py \\
        --output allData --start 2022-11-01 --end 2022-11-30 \\
        --underliers NIFTY,BANKNIFTY --timestep 30
"""

from __future__ import annotations

import argparse
import math
import os
import random
from datetime import date, datetime, time, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd


# ─── underlier config ────────────────────────────────────────────────────────

UNDERLIER_CONFIG = {
    "NIFTY": {
        "strike_interval": 50,
        "lot_size":        50,
        "start_price":     18000.0,
        "vol_daily":       0.011,    # ~1.1% daily vol
        "n_strikes_each_side": 10,
        "expiries": [
            date(2022, 11, 3), date(2022, 11, 10), date(2022, 11, 17),
            date(2022, 11, 24), date(2022, 12, 1),
        ],
    },
    "BANKNIFTY": {
        "strike_interval": 100,
        "lot_size":        25,
        "start_price":     41500.0,
        "vol_daily":       0.015,
        "n_strikes_each_side": 8,
        "expiries": [
            date(2022, 11, 3), date(2022, 11, 10), date(2022, 11, 17),
            date(2022, 11, 24), date(2022, 12, 1),
        ],
    },
    "FINNIFTY": {
        "strike_interval": 50,
        "lot_size":        40,
        "start_price":     16000.0,
        "vol_daily":       0.012,
        "n_strikes_each_side": 8,
        "expiries": [
            date(2022, 11, 8), date(2022, 11, 15), date(2022, 11, 22),
            date(2022, 11, 29), date(2022, 12, 6),
        ],
    },
}

MARKET_OPEN  = time(9, 15, 0)
MARKET_CLOSE = time(15, 30, 0)


def _trading_dates(start: date, end: date) -> List[date]:
    out = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def _build_canonical_index(d: date, step: int) -> pd.DatetimeIndex:
    start = datetime.combine(d, MARKET_OPEN)
    end   = datetime.combine(d, MARKET_CLOSE)
    return pd.date_range(start=start, end=end, freq=f"{step}s")


def _simulate_futures_prices(
    start_price: float, vol_daily: float, dates: List[date], step: int, seed: int
) -> pd.DataFrame:
    """
    Simulate a single continuous GBM price series across all trading dates.
    Returns long-form DataFrame: [date, time, timestamp, price, volume, open_interest]
    """
    rng = np.random.default_rng(seed)
    rows = []
    price = start_price
    dt_seconds = step
    dt_year = dt_seconds / (252 * 6.25 * 3600)  # fraction of trading year
    sigma = vol_daily / math.sqrt(1/252)        # annualized vol
    mu = 0.0                                     # zero drift (we don't care about direction)
    for d in dates:
        idx = _build_canonical_index(d, step)
        for ts in idx:
            # GBM step
            z = rng.standard_normal()
            ret = (mu - 0.5 * sigma**2) * dt_year + sigma * math.sqrt(dt_year) * z
            price = price * math.exp(ret)
            volume = int(rng.integers(50, 5000))
            oi = int(rng.integers(10000, 50000))
            rows.append({
                "date": d.strftime("%d-%m-%Y"),
                "time": ts.strftime("%H:%M:%S"),
                "price": round(price, 2),
                "volume": volume,
                "open_interest": oi,
            })
    return pd.DataFrame(rows)


def _black_scholes_call(S: float, K: int, T: float, sigma: float, r: float = 0.06) -> float:
    if T <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    N = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return S * N(d1) - K * math.exp(-r * T) * N(d2)


def _black_scholes_put(S: float, K: int, T: float, sigma: float, r: float = 0.06) -> float:
    if T <= 0:
        return max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    N = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return K * math.exp(-r * T) * N(-d2) - S * N(-d1)


def _simulate_options_for_day(
    underlier: str,
    d: date,
    futures_prices: pd.DataFrame,
    step: int,
    seed: int,
) -> Tuple[List[Tuple[int, str, pd.DataFrame]], date]:
    """For one underlier on one date, simulate all option files for the nearest expiry."""
    cfg_u = UNDERLIER_CONFIG[underlier]
    rng = np.random.default_rng(seed)

    # Filter futures prices for this date.
    futs_day = futures_prices[futures_prices["date"] == d.strftime("%d-%m-%Y")].copy()
    if futs_day.empty:
        return [], None

    # Determine nearest expiry.
    expiries = cfg_u["expiries"]
    future_expiries = sorted([e for e in expiries if e >= d])
    if not future_expiries:
        return [], None
    nearest_expiry = future_expiries[0]
    yy = str(nearest_expiry.year)[2:]
    mm = f"{nearest_expiry.month:02d}"
    dd = f"{nearest_expiry.day:02d}"
    expiry_str = f"{yy}{mm}{dd}"

    # Determine strikes around current futures price.
    interval = cfg_u["strike_interval"]
    n_each = cfg_u["n_strikes_each_side"]
    # Use the day's first futures price as the center.
    first_price = float(futs_day.iloc[0]["price"])
    center_strike = int(round(first_price / interval) * interval)
    strikes = list(range(
        center_strike - n_each * interval,
        center_strike + (n_each + 1) * interval,
        interval,
    ))

    sigma = cfg_u["vol_daily"] / math.sqrt(1/252)  # annualized
    days_to_expiry = (nearest_expiry - d).days
    T_start = max(days_to_expiry, 0) / 365.0

    option_files: List[Tuple[int, str, pd.DataFrame]] = []

    for strike in strikes:
        for opt_type in ("CE", "PE"):
            rows = []
            for i, fr in futs_day.iterrows():
                S = float(fr["price"])
                # T decreases through the day by 1/(252 * 6.25 * 3600 / step) per step
                # For simplicity use T_start - small_amount.
                T = max(T_start - i * step / (252 * 6.25 * 3600), 1e-6)
                if opt_type == "CE":
                    premium = _black_scholes_call(S, strike, T, sigma)
                else:
                    premium = _black_scholes_put(S, strike, T, sigma)
                # Add small noise so premiums aren't perfectly smooth.
                premium = max(0.05, premium * (1.0 + rng.normal(0, 0.005)))
                rows.append({
                    "date":   fr["date"],
                    "time":   fr["time"],
                    "price":  round(premium, 2),
                    "volume": int(rng.integers(0, 2000)),
                    "open_interest": int(rng.integers(100, 50000)),
                })
            df = pd.DataFrame(rows)
            option_files.append((strike, opt_type, df))

    return option_files, nearest_expiry


def _write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def generate(
    output_root: str,
    start: date,
    end: date,
    underliers: List[str],
    step: int,
    seed: int = 42,
) -> None:
    """Generate the synthetic dataset on disk."""
    dates = _trading_dates(start, end)
    print(f"Generating {len(dates)} trading days from {start} to {end}")

    for u in underliers:
        cfg_u = UNDERLIER_CONFIG[u]
        # Generate futures prices across all dates as one GBM.
        futs = _simulate_futures_prices(
            start_price=cfg_u["start_price"],
            vol_daily=cfg_u["vol_daily"],
            dates=dates,
            step=step,
            seed=seed + hash(u) % 1000,
        )

        for d in dates:
            date_folder = os.path.join(
                output_root, f"NSE_{d.strftime('%Y%m%d')}"
            )
            # Write futures file.
            futs_day = futs[futs["date"] == d.strftime("%d-%m-%Y")].copy()
            _write_csv(
                futs_day,
                os.path.join(date_folder, "futures", f"{u}-I.csv"),
            )

            # Generate option files for this date.
            option_files, nearest_expiry = _simulate_options_for_day(
                u, d, futs_day, step=step, seed=seed + hash(str(d) + u) % 1000,
            )
            if not option_files:
                continue

            yy = str(nearest_expiry.year)[2:]
            mm = f"{nearest_expiry.month:02d}"
            dd = f"{nearest_expiry.day:02d}"
            expiry_str = f"{yy}{mm}{dd}"

            for strike, opt_type, df in option_files:
                fname = f"{u}{expiry_str}{strike}{opt_type}.csv"
                _write_csv(df, os.path.join(date_folder, "options", fname))

            print(f"  {u} {d}: {len(option_files)} option files, futures={len(futs_day)} rows")

    print(f"Done. Synthetic data written to {output_root}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="allData")
    p.add_argument("--start", default="2022-11-01")
    p.add_argument("--end",   default="2022-11-30")
    p.add_argument("--underliers", default="NIFTY,BANKNIFTY")
    p.add_argument("--timestep", type=int, default=30,
                   help="Seconds between synthetic ticks (30 is fine for tests; "
                        "use 1 for full second-by-second simulation matching the assignment).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end, "%Y-%m-%d").date()
    underliers = [u.strip() for u in args.underliers.split(",") if u.strip()]
    generate(args.output, start, end, underliers, args.timestep, args.seed)
