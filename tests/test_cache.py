"""
tests/test_cache.py — Test cache key determinism, save/load roundtrip.
"""

import os
import tempfile
from datetime import date, datetime

import pandas as pd
import pytest

from engine.cache import CacheManager
from strategies.atm_straddle import ATMStraddle
import config as cfg


class _MockConfig:
    TIMESTEP_SECONDS = 1
    BROKERAGE_PER_LOT_PER_LEG = 0.0
    STT_RATE = 0.0
    EXCHANGE_CHARGES_RATE = 0.0
    EOD_FLATTEN_BUFFER_MINUTES = 0
    PRICE_STALENESS_THRESHOLD_SECONDS = None
    STRIKE_TIE_BREAK = "up"
    MAX_POSITION_LOTS = 1


def test_cache_key_deterministic():
    """Same params → same hash."""
    cache = CacheManager(tempfile.mkdtemp())
    strat = ATMStraddle()
    cfg_ = _MockConfig()
    h1 = cache.compute_params_hash(strat, ["NIFTY", "BANKNIFTY"], date(2022, 11, 1), date(2022, 11, 30), cfg_)
    h2 = cache.compute_params_hash(strat, ["NIFTY", "BANKNIFTY"], date(2022, 11, 1), date(2022, 11, 30), cfg_)
    assert h1 == h2
    assert len(h1) == 16  # truncated SHA-256


def test_cache_key_changes_on_param():
    """Change one param → different hash."""
    cache = CacheManager(tempfile.mkdtemp())
    strat = ATMStraddle()
    cfg_ = _MockConfig()
    h1 = cache.compute_params_hash(strat, ["NIFTY"], date(2022, 11, 1), date(2022, 11, 30), cfg_)
    cfg2 = _MockConfig()
    cfg2.TIMESTEP_SECONDS = 5
    h2 = cache.compute_params_hash(strat, ["NIFTY"], date(2022, 11, 1), date(2022, 11, 30), cfg2)
    assert h1 != h2


def test_cache_key_changes_on_underliers_order():
    """Underlier list order shouldn't matter (sorted before hashing)."""
    cache = CacheManager(tempfile.mkdtemp())
    strat = ATMStraddle()
    cfg_ = _MockConfig()
    h1 = cache.compute_params_hash(strat, ["NIFTY", "BANKNIFTY"], date(2022, 11, 1), date(2022, 11, 30), cfg_)
    h2 = cache.compute_params_hash(strat, ["BANKNIFTY", "NIFTY"], date(2022, 11, 1), date(2022, 11, 30), cfg_)
    assert h1 == h2


def test_save_and_load():
    """Save result, load it, compare DataFrames."""
    cache = CacheManager(tempfile.mkdtemp())
    strat = ATMStraddle()
    cfg_ = _MockConfig()
    params_hash = cache.compute_params_hash(strat, ["NIFTY"], date(2022, 11, 1), date(2022, 11, 1), cfg_)

    event_log = pd.DataFrame({
        "timestamp": [datetime(2022, 11, 1, 9, 15, 0), datetime(2022, 11, 1, 9, 15, 1)],
        "underlier": ["NIFTY", "NIFTY"],
        "futures_price": [18000.0, 18001.0],
        "atm_strike": [18000, 18000],
        "nearest_expiry": ["2022-11-03", "2022-11-03"],
        "ce_held": [False, True],
        "pe_held": [False, True],
        "ce_strike": [None, 18000],
        "pe_strike": [None, 18000],
        "ce_entry_price": [None, 100.0],
        "pe_entry_price": [None, 80.0],
        "ce_current_price": [None, 100.0],
        "pe_current_price": [None, 80.0],
        "ce_mtm_pnl": [0.0, 0.0],
        "pe_mtm_pnl": [0.0, 0.0],
        "total_mtm_pnl": [0.0, 0.0],
        "realized_pnl_today": [0.0, 0.0],
        "cumulative_realized_pnl": [0.0, 0.0],
        "total_pnl": [0.0, 0.0],
        "trade_occurred": [False, True],
        "trade_type": [None, "OPEN"],
        "rolls_today": [0, 0],
    })
    trade_log = pd.DataFrame([{
        "timestamp": datetime(2022, 11, 1, 9, 15, 1),
        "underlier": "NIFTY",
        "instrument_id": "NIFTY18000CE",
        "action": "BUY",
        "quantity": 1,
        "execution_price": 100.0,
        "transaction_cost": 0.0,
        "realized_pnl": 0.0,
        "trade_type": "OPEN",
    }])

    cache.save(params_hash, "atm_straddle", ["NIFTY"],
               date(2022, 11, 1), date(2022, 11, 1),
               event_log, trade_log)

    assert cache.has_cache(params_hash)

    loaded = cache.load(params_hash)
    pd.testing.assert_frame_equal(
        loaded["event_log"].reset_index(drop=True),
        event_log.reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        loaded["trade_log"].reset_index(drop=True),
        trade_log.reset_index(drop=True),
        check_dtype=False,
    )
    assert loaded["meta"]["strategy_name"] == "atm_straddle"


def test_clear_cache_entry():
    cache = CacheManager(tempfile.mkdtemp())
    strat = ATMStraddle()
    cfg_ = _MockConfig()
    h = cache.compute_params_hash(strat, ["NIFTY"], date(2022, 11, 1), date(2022, 11, 1), cfg_)

    empty_df = pd.DataFrame()
    cache.save(h, "atm_straddle", ["NIFTY"], date(2022, 11, 1), date(2022, 11, 1), empty_df, empty_df)
    assert cache.has_cache(h)
    cache.clear(h)
    assert not cache.has_cache(h)


def test_checkpoint_save_load():
    cache = CacheManager(tempfile.mkdtemp())
    strat = ATMStraddle()
    cfg_ = _MockConfig()
    h = cache.compute_params_hash(strat, ["NIFTY"], date(2022, 11, 1), date(2022, 11, 1), cfg_)

    rows = [{"timestamp": datetime(2022, 11, 1, 9, 15, 0), "underlier": "NIFTY", "total_pnl": 0.0}]
    cache.save_checkpoint(h, [date(2022, 11, 1)], rows)
    assert cache.has_checkpoint(h)
    cp = cache.load_checkpoint(h)
    assert date(2022, 11, 1).isoformat() in cp.completed_dates
    assert len(cp.partial_log_rows) == 1
