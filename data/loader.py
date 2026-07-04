"""
data/loader.py — Raw CSV loading per date per underlier.

NSE CSV format (real data): NO header row, columns are:
    date (YYYYMMDD), time (HH:MM:SS), price, volume, open_interest

NSE CSV format (synthetic / legacy): has header row Date,Time,Price,Volume,Open Interest
with DD-MM-YYYY date format.

This module auto-detects which format is present and handles both transparently.

Real-data folder layout:
    {DATA_ROOT}/NSE_{YYYYMMDD}/Options/{INSTRUMENT}.csv
    {DATA_ROOT}/NSE_{YYYYMMDD}/Futures (Continuous)/{UNDERLIER}-I.csv

Synthetic-data / legacy folder layout:
    {DATA_ROOT}/NSE_{YYYYMMDD}/options/{INSTRUMENT}.csv
    {DATA_ROOT}/NSE_{YYYYMMDD}/futures/{UNDERLIER}-I.csv
"""

from __future__ import annotations

import os
import logging
from datetime import date, datetime, time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ─── CSV parsing ─────────────────────────────────────────────────────────────


def _looks_like_header(first_field: str) -> bool:
    """Return True if the first field looks like a text header (not a date)."""
    s = str(first_field).strip().lower()
    # Headers contain letters that are not part of an ISO date
    return any(c.isalpha() for c in s)


def _read_and_clean(path: str) -> pd.DataFrame:
    """
    Read a single NSE CSV into a clean DataFrame with timestamp index and
    columns [price, volume, open_interest].

    Auto-detects format:
      - If first row first field contains letters → legacy/synthetic (has header).
      - Otherwise → real NSE data (no header, YYYYMMDD date).

    Cleaning rules:
      - Parse Date+Time into a single 'timestamp' datetime column.
      - Set timestamp as index, sort ascending.
      - Coerce Price/Volume/Open Interest to numeric.
      - Drop rows where Price is 0 or NaN (invalid ticks).
      - Drop duplicate timestamps keeping the LAST (most recent trade at that second).
    """
    # Peek at first row to detect format.
    with open(path, "r", errors="replace") as fh:
        first_line = fh.readline().strip()

    first_fields = first_line.split(",")
    has_header = _looks_like_header(first_fields[0]) if first_fields else False

    if has_header:
        # Legacy / synthetic format: header row present, DD-MM-YYYY dates.
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        col_map = {}
        for c in df.columns:
            cl = c.lower().replace(" ", "_")
            if cl in ("date", "datetime"):
                col_map[c] = "date"
            elif cl == "time":
                col_map[c] = "time"
            elif cl == "price":
                col_map[c] = "price"
            elif cl == "volume":
                col_map[c] = "volume"
            elif cl in ("open_interest", "oi", "openinterest"):
                col_map[c] = "open_interest"
        df = df.rename(columns=col_map)

        if "date" not in df.columns or "time" not in df.columns:
            raise ValueError(
                f"{path}: missing Date or Time column; got {df.columns.tolist()}"
            )
        if "price" not in df.columns:
            raise ValueError(f"{path}: missing Price column")

        # Parse date+time — try DD-MM-YYYY first (Indian format).
        ts_str = df["date"].astype(str) + " " + df["time"].astype(str)
        try:
            ts = pd.to_datetime(ts_str, dayfirst=True, errors="coerce")
        except Exception:
            ts = pd.to_datetime(ts_str, errors="coerce")
        if ts.isna().all():
            ts = pd.to_datetime(ts_str, errors="coerce")
        df["timestamp"] = ts
    else:
        # Real NSE format: no header, YYYYMMDD date.
        col_names = ["date", "time", "price", "volume", "open_interest"]
        df = pd.read_csv(
            path,
            header=None,
            names=col_names,
            dtype={"date": str, "time": str},
        )
        # Parse YYYYMMDD + HH:MM:SS
        ts_str = df["date"].astype(str) + " " + df["time"].astype(str)
        ts = pd.to_datetime(ts_str, format="%Y%m%d %H:%M:%S", errors="coerce")
        if ts.isna().all():
            # Fallback: try generic parse
            ts = pd.to_datetime(ts_str, errors="coerce")
        df["timestamp"] = ts

    df = df.dropna(subset=["timestamp"])

    # Coerce numeric columns.
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = (
            pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
        )
    else:
        df["volume"] = 0
    if "open_interest" in df.columns:
        df["open_interest"] = (
            pd.to_numeric(df["open_interest"], errors="coerce")
            .fillna(0)
            .astype("int64")
        )
    else:
        df["open_interest"] = 0

    df = df.set_index("timestamp").sort_index()

    # Drop invalid (zero or NaN) prices — these are not real trades.
    df = df[df["price"].notna() & (df["price"] > 0)]

    # Drop duplicate timestamps keeping the most recent trade.
    df = df[~df.index.duplicated(keep="last")]

    cols = ["price", "volume", "open_interest"]
    return df[[c for c in cols if c in df.columns]]


# ─── path helpers ─────────────────────────────────────────────────────────────


def _date_folder(data_root: str, trading_date: date, prefix: str, fmt: str) -> str:
    """Build NSE_YYYYMMDD folder path."""
    return os.path.join(data_root, f"{prefix}{trading_date.strftime(fmt)}")


def _find_subfolder(date_folder: str, preferred_name: str) -> Optional[str]:
    """
    Return the full path of a subfolder by name (case-sensitive).
    Returns None if not found — caller decides whether to raise.

    This allows the loader to work with both:
      - Real data:      "Options", "Futures (Continuous)"
      - Synthetic data: "options", "futures"
    """
    path = os.path.join(date_folder, preferred_name)
    if os.path.isdir(path):
        return path
    # Case-insensitive fallback: try all entries in date_folder.
    try:
        entries = os.listdir(date_folder)
    except OSError:
        return None
    lower = preferred_name.lower()
    for entry in entries:
        if entry.lower() == lower and os.path.isdir(os.path.join(date_folder, entry)):
            logger.debug(
                "Subfolder '%s' not found; using '%s' (case-insensitive match)",
                preferred_name, entry,
            )
            return os.path.join(date_folder, entry)
    return None


# ─── ATM-windowed strike filtering ───────────────────────────────────────────


def _futures_price_range(data_root: str, trading_date: date, underlier: str,
                          series: str, date_folder_prefix: str,
                          date_folder_format: str,
                          futures_subfolder: str) -> Tuple[float, float]:
    """
    Quick-load the futures CSV and return (price_min, price_max) for the day.
    Used to restrict which option strikes need to be loaded.
    """
    try:
        fut_df = load_futures_for_date(
            data_root, trading_date, underlier, series,
            date_folder_prefix, date_folder_format, futures_subfolder,
        )
        pmin = float(fut_df["price"].min())
        pmax = float(fut_df["price"].max())
        return pmin, pmax
    except Exception:
        return 0.0, float("inf")  # conservative: load everything


def _strike_window(
    price_min: float,
    price_max: float,
    strike_interval: int,
    window_strikes: int,
) -> Tuple[float, float]:
    """
    Compute the (low, high) strike range to load given a futures price range
    and a window measured in number of strikes.

    Example: NIFTY interval=50, price 18100–18200, window=10
      → load strikes 18100 - 10*50 = 17600 to 18200 + 10*50 = 18700
    """
    padding = window_strikes * strike_interval
    return price_min - padding, price_max + padding


# ─── public API ──────────────────────────────────────────────────────────────


def load_options_for_date(
    data_root: str,
    trading_date: date,
    underlier: str,
    expiry: date,
    date_folder_prefix: str = "NSE_",
    date_folder_format: str = "%Y%m%d",
    options_subfolder: str = "Options",
    # ATM-windowed loading parameters
    atm_window_strikes: Optional[int] = 10,
    strike_interval: int = 50,
    # Pre-computed futures price range (avoids loading futures CSV twice).
    # If None and atm_window_strikes is set, futures are loaded internally.
    futures_price_range: Optional[Tuple[float, float]] = None,
    # Fallback: load futures internally using these params (only used when
    # futures_price_range is not provided).
    futures_subfolder: str = "Futures (Continuous)",
    futures_series: str = "I",
) -> Dict[Tuple[int, str], pd.DataFrame]:
    """
    Load all option files for a given underlier and expiry on a trading date.

    When atm_window_strikes is set (default: 10), only options within
    ATM ± (atm_window_strikes * strike_interval) of the intraday futures
    price range are loaded. This reduces memory from ~1.6 GB to ~40 MB on
    real NSE data (which has 100–600 option files per day per underlier).

    Returns:
        Dict mapping (strike, option_type) → cleaned DataFrame with columns
        [price, volume, open_interest] and timestamp index.
    """
    # Local import to avoid circular dependency at module load.
    from data.parser import list_option_files, parse_option_filename

    folder = _date_folder(data_root, trading_date, date_folder_prefix, date_folder_format)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Date folder not found: {folder}")

    # Resolve options subfolder (case-tolerant).
    options_dir = _find_subfolder(folder, options_subfolder)
    if options_dir is None:
        # Try common alternatives.
        for alt in ("options", "Options", "OPTIONS"):
            options_dir = _find_subfolder(folder, alt)
            if options_dir is not None:
                break
    if options_dir is None:
        logger.warning("No options subfolder found in %s", folder)
        return {}

    # Determine strike filter range if windowing is enabled.
    strike_filter: Optional[Tuple[float, float]] = None
    if atm_window_strikes is not None:
        if futures_price_range is not None:
            # Use pre-computed range from caller (avoids second futures CSV read).
            pmin, pmax = futures_price_range
        else:
            # Fallback: load futures internally (slower, kept for standalone use).
            pmin, pmax = _futures_price_range(
                data_root, trading_date, underlier,
                futures_series, date_folder_prefix, date_folder_format, futures_subfolder,
            )
        if pmax > 0:
            lo, hi = _strike_window(pmin, pmax, strike_interval, atm_window_strikes)
            strike_filter = (lo, hi)
            logger.debug(
                "%s %s: loading strikes %.0f – %.0f (futures %.0f – %.0f, window ±%d)",
                underlier, trading_date, lo, hi, pmin, pmax, atm_window_strikes,
            )

    result: Dict[Tuple[int, str], pd.DataFrame] = {}
    files = list_option_files_from_dir(options_dir, underlier)

    for path in files:
        try:
            opt = parse_option_filename(os.path.basename(path))
        except ValueError:
            continue
        if opt.expiry != expiry:
            continue
        # Apply strike window filter.
        if strike_filter is not None:
            lo, hi = strike_filter
            if not (lo <= opt.strike <= hi):
                continue
        try:
            df = _read_and_clean(path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
            continue
        if df.empty:
            continue
        result[(opt.strike, opt.option_type)] = df

    if not result:
        logger.warning(
            "%s %s expiry=%s: no option data loaded (files=%d, strike_filter=%s)",
            underlier, trading_date, expiry, len(files), strike_filter,
        )
    return result


def list_option_files_from_dir(options_dir: str, underlier: str) -> List[str]:
    """
    Return sorted list of option CSV paths for an underlier from a resolved options directory.
    Handles both case variations of the directory.
    """
    from data.parser import parse_option_filename
    import glob

    pattern = os.path.join(options_dir, f"{underlier}*.csv")
    matches = glob.glob(pattern)
    result = []
    for path in matches:
        try:
            opt = parse_option_filename(os.path.basename(path))
            if opt.underlier == underlier:
                result.append(path)
        except ValueError:
            logger.debug("Skipping non-option file: %s", path)
    return sorted(result)


def load_futures_for_date(
    data_root: str,
    trading_date: date,
    underlier: str,
    series: str = "I",
    date_folder_prefix: str = "NSE_",
    date_folder_format: str = "%Y%m%d",
    futures_subfolder: str = "Futures (Continuous)",
) -> pd.DataFrame:
    """
    Load the near-month futures CSV for a given date and underlier.

    Tries the configured futures_subfolder name first, then falls back to
    common alternatives ("futures", "Futures") to handle both real and
    synthetic data layouts.

    Returns:
        DataFrame with timestamp index and columns [price, volume, open_interest].
    """
    folder = _date_folder(data_root, trading_date, date_folder_prefix, date_folder_format)

    # Try configured name and common alternatives.
    futures_dir = _find_subfolder(folder, futures_subfolder)
    if futures_dir is None:
        for alt in ("futures", "Futures", "Futures (Continuous)"):
            futures_dir = _find_subfolder(folder, alt)
            if futures_dir is not None:
                break

    if futures_dir is None:
        raise FileNotFoundError(
            f"Futures subfolder not found in {folder}. "
            f"Tried: '{futures_subfolder}', 'futures', 'Futures', 'Futures (Continuous)'"
        )

    path = os.path.join(futures_dir, f"{underlier}-{series}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Futures file not found: {path}")
    return _read_and_clean(path)
