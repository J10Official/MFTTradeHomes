"""
tests/test_strategy.py — Test the ATMStraddle strategy signal generation.
"""

from datetime import datetime, date
from typing import List

import pytest

from data.market_state import MarketContext
from engine.order import Order
from engine.portfolio import Portfolio, PortfolioSnapshot, Position
from instruments.option import Option
from strategies.atm_straddle import ATMStraddle


def _make_context(
    timestamp: datetime,
    underlier: str = "NIFTY",
    futures_price: float = 18025.0,
    atm_strike: int = 18000,
    option_prices=None,
    positions=None,
    available_strikes=None,
) -> MarketContext:
    """Build a minimal MarketContext for testing the strategy."""
    if option_prices is None:
        # Default: ATM strike has both CE and PE priced.
        option_prices = {
            (atm_strike, "CE"): 100.0,
            (atm_strike, "PE"):  80.0,
            (atm_strike + 50, "CE"): 95.0,
            (atm_strike + 50, "PE"): 85.0,
            (atm_strike - 50, "CE"): 105.0,
            (atm_strike - 50, "PE"): 75.0,
        }
    if available_strikes is None:
        available_strikes = sorted({k[0] for k in option_prices.keys()})
    snap = PortfolioSnapshot(
        positions=positions or {},
        realized_pnl_today=0.0,
        cumulative_realized_pnl=0.0,
        trade_count_today=0,
        roll_count_today=0,
    )
    return MarketContext(
        timestamp=timestamp,
        underlier=underlier,
        futures_price=futures_price,
        atm_strike=atm_strike,
        nearest_expiry=date(2022, 11, 3),
        option_prices=option_prices,
        portfolio=snap,
        available_strikes=available_strikes,
    )


def _held(strike: int, opt_type: str, underlier="NIFTY", entry=100.0):
    opt = Option(
        id=f"{underlier}{strike}{opt_type}",
        underlier=underlier,
        expiry=date(2022, 11, 3),
        strike=strike,
        option_type=opt_type,
    )
    return Position(instrument=opt, quantity=1, entry_price=entry,
                    entry_time=datetime(2022, 11, 1, 9, 15, 0))


def test_open_when_flat():
    """Flat portfolio → strategy returns BUY CE + BUY PE at ATM strike."""
    strat = ATMStraddle()
    ctx = _make_context(datetime(2022, 11, 1, 9, 15, 0),
                        atm_strike=18000, futures_price=18025.0)
    orders = strat.generate_signals(ctx)
    actions = [(o.action, o.instrument.strike, o.instrument.option_type) for o in orders]
    assert ("BUY", 18000, "CE") in actions
    assert ("BUY", 18000, "PE") in actions
    assert len(orders) == 2


def test_hold_when_atm_unchanged():
    """Holding ATM CE+PE, ATM unchanged → returns [] (hold)."""
    strat = ATMStraddle()
    positions = {
        "NIFTY18000CE": _held(18000, "CE"),
        "NIFTY18000PE": _held(18000, "PE"),
    }
    ctx = _make_context(datetime(2022, 11, 1, 9, 16, 0),
                        atm_strike=18000, futures_price=18025.0,
                        positions=positions)
    orders = strat.generate_signals(ctx)
    assert orders == []


def test_roll_when_atm_changed():
    """Holding 18000, new ATM 18050 → 4 orders (SELL CE, SELL PE, BUY CE, BUY PE)."""
    strat = ATMStraddle()
    positions = {
        "NIFTY18000CE": _held(18000, "CE"),
        "NIFTY18000PE": _held(18000, "PE"),
    }
    ctx = _make_context(datetime(2022, 11, 1, 10, 0, 0),
                        atm_strike=18050, futures_price=18075.0,
                        positions=positions)
    orders = strat.generate_signals(ctx)
    assert len(orders) == 4
    actions = [(o.action, o.instrument.strike, o.instrument.option_type) for o in orders]
    # First two should be SELLs of old strike.
    assert ("SELL", 18000, "CE") in actions
    assert ("SELL", 18000, "PE") in actions
    # Then two BUYs of new ATM strike.
    assert ("BUY", 18050, "CE") in actions
    assert ("BUY", 18050, "PE") in actions


def test_sell_before_buy_on_roll():
    """On a roll, SELL orders MUST precede BUY orders."""
    strat = ATMStraddle()
    positions = {
        "NIFTY18000CE": _held(18000, "CE"),
        "NIFTY18000PE": _held(18000, "PE"),
    }
    ctx = _make_context(datetime(2022, 11, 1, 10, 0, 0),
                        atm_strike=18050, futures_price=18075.0,
                        positions=positions)
    orders = strat.generate_signals(ctx)
    # The first 2 orders should be SELLs, last 2 should be BUYs.
    for o in orders[:2]:
        assert o.action == "SELL"
    for o in orders[2:]:
        assert o.action == "BUY"


def test_strategy_is_pure_function():
    """Same context → same orders, every time (no internal mutable state)."""
    strat = ATMStraddle()
    ctx = _make_context(datetime(2022, 11, 1, 9, 15, 0))
    o1 = strat.generate_signals(ctx)
    o2 = strat.generate_signals(ctx)
    # Compare order-by-order: same actions, strikes, types.
    a1 = [(o.action, o.instrument.strike, o.instrument.option_type) for o in o1]
    a2 = [(o.action, o.instrument.strike, o.instrument.option_type) for o in o2]
    assert a1 == a2


def test_strategy_name():
    assert ATMStraddle().name == "atm_straddle"
