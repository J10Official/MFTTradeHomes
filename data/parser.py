"""
data/parser.py — Filename → Option parsing for NSE options CSV files.

NSE filename format: {UNDERLIER}{YYMMDD}{STRIKE}{TYPE}.csv
    NIFTY22110314550PE.csv      → NIFTY, 2022-11-03, 14550, PE
    BANKNIFTY22112443200CE.csv  → BANKNIFTY, 2022-11-24, 43200, CE
    FINNIFTY22110719500CE.csv   → FINNIFTY, 2022-11-07, 19500, CE

This module has no dependencies on any other module in this project.
"""

from __future__ import annotations

import os
import re
import logging
from datetime import date
from glob import glob
from typing import List, Optional

from instruments.option import Option


logger = logging.getLogger(__name__)

# Longest-first to avoid "NIFTY" matching the prefix of "BANKNIFTY".
KNOWN_UNDERLIERS = ["BANKNIFTY", "FINNIFTY", "NIFTY"]

# Pure-digit body after underlier, ends with CE|PE.
_FILENAME_RE = re.compile(r"^(?P<underlier>[A-Z]+)(?P<expiry>\d{6})(?P<strike>\d+)(?P<type>CE|PE)$")


def parse_option_filename(filename: str) -> Option:
    """
    Parse an NSE options filename into an Option instrument.

    Args:
        filename: stem or full filename (extension optional).

    Returns:
        Option with id (stem), underlier, expiry, strike, option_type.

    Raises:
        ValueError if the filename does not conform to the NSE options format.
    """
    stem = os.path.basename(filename).replace(".csv", "")
    if not stem:
        raise ValueError(f"Empty filename: {filename!r}")

    # Underlier match — try known underliers longest-first.
    underlier: Optional[str] = None
    for u in KNOWN_UNDERLIERS:
        if stem.startswith(u):
            underlier = u
            break
    if underlier is None:
        raise ValueError(f"Unknown underlier in filename: {filename!r}")

    rest = stem[len(underlier):]
    # rest should look like: 22110314550PE
    m = _FILENAME_RE.match(stem)
    if m is None:
        raise ValueError(f"Filename does not match NSE option pattern: {filename!r}")

    expiry_str = m.group("expiry")
    strike_str = m.group("strike")
    option_type = m.group("type")

    # Parse YYMMDD → date. Two-digit year → 20YY (NSE didn't exist in 19YY).
    try:
        expiry = date(2000 + int(expiry_str[:2]),
                      int(expiry_str[2:4]),
                      int(expiry_str[4:6]))
    except ValueError as e:
        raise ValueError(f"Invalid expiry date in filename {filename!r}: {e}")

    try:
        strike = int(strike_str)
    except ValueError:
        raise ValueError(f"Invalid strike in filename {filename!r}")

    return Option(
        id=stem,
        underlier=underlier,
        expiry=expiry,
        strike=strike,
        option_type=option_type,
    )


def list_option_files(date_folder: str, underlier: str) -> List[str]:
    """
    Return all option file paths for a given underlier in a date folder.

    Tries "Options" (real NSE data) then "options" (synthetic/legacy) so this
    function works with both data layouts without caller changes.

    Args:
        date_folder: absolute or relative path to NSE_YYYYMMDD directory.
        underlier:   "NIFTY", "BANKNIFTY", "FINNIFTY".

    Returns:
        Sorted list of full paths matching the underlier.
    """
    # Try both capitalisation variants.
    options_dir: Optional[str] = None
    for candidate in ("Options", "options", "OPTIONS"):
        candidate_path = os.path.join(date_folder, candidate)
        if os.path.isdir(candidate_path):
            options_dir = candidate_path
            break
    if options_dir is None:
        return []

    pattern = os.path.join(options_dir, f"{underlier}*.csv")
    matches = glob(pattern)

    # Filter to those whose parsed underlier actually matches
    # (defensive: glob "NIFTY*.csv" would also match "NIFTYBANK..." if such a
    # thing existed, but the prefix approach is already safe given our known
    # underlier list. We re-confirm by parsing.)
    result = []
    for path in matches:
        try:
            opt = parse_option_filename(os.path.basename(path))
            if opt.underlier == underlier:
                result.append(path)
        except ValueError:
            logger.debug("Skipping non-option file in options dir: %s", path)
            continue
    return sorted(result)


def get_available_expiries(date_folder: str, underlier: str) -> List[date]:
    """Return sorted list of unique expiry dates found in the options folder for one underlier."""
    expiries = set()
    for path in list_option_files(date_folder, underlier):
        try:
            opt = parse_option_filename(os.path.basename(path))
            expiries.add(opt.expiry)
        except ValueError:
            continue
    return sorted(expiries)


def get_nearest_expiry(trading_date: date, available_expiries: List[date]) -> date:
    """
    Return the smallest expiry date that is >= trading_date.

    On the expiry date itself: still use that expiry (options trade until 15:30,
    settlement happens after market close).

    Raises:
        ValueError: if no future expiry found (end of dataset).
    """
    future_expiries = [e for e in sorted(available_expiries) if e >= trading_date]
    if not future_expiries:
        raise ValueError(
            f"No expiry on or after {trading_date} in available expiries "
            f"{available_expiries}"
        )
    return future_expiries[0]
