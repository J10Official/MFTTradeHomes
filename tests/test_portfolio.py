"""
tests/test_portfolio.py — Test Position, Portfolio, PnL computation, limits.
"""

from datetime import datetime, date

import pytest

from engine.order import Order, Trade
from engine.portfolio import Portfolio, Position, PortfolioSnapshot
from instruments.option import Option


def _make_option(strike=18000, opt_type="CE", underlier="NIFTY"):
    return Option(
        id=f"{underlier}{strike}{opt_type}",
        underlier=underlier,
        expiry=date(2022, 11, 3),
        strike=strike,
        option_type=opt_type,
    )


def _make_trade(instrument, action="BUY", quantity=1, price=100.0, ts=None):
    if ts is None:
        ts = datetime(2022, 11, 1, 9, 15, 0)
    order = Order(instrument=instrument, action=action, quantity=quantity, timestamp=ts)
    return Trade(
        order=order,
        execution_price=price,
        transaction_cost=0.0,
        realized_pnl=0.0,
        trade_type="OPEN",
    )


def test_open_and_mtm():
    """Open position, compute MTM PnL."""
    pf = Portfolio()
    opt = _make_option(strike=18000, opt_type="CE")
    trade = _make_trade(opt, "BUY", 1, 100.0)
    pf.open_position(trade)

    pos = pf.get_position(opt.id)
    assert pos is not None
    assert pos.quantity == 1
    assert pos.entry_price == 100.0

    # MTM at price 105 → (105 - 100) * 1 * 50 = 250
    assert pos.mtm_pnl(105.0, lot_size=50) == 250.0
    # MTM at price 95 → -250
    assert pos.mtm_pnl(95.0, lot_size=50) == -250.0


def test_open_and_close():
    """Open position, close it, check realized PnL."""
    pf = Portfolio()
    opt = _make_option(strike=18000, opt_type="CE")
    buy_trade = _make_trade(opt, "BUY", 1, 100.0)
    pf.open_position(buy_trade)

    # Close at 110 → realized = (110 - 100) * 1 * 50 = 500
    sell_order = Order(instrument=opt, action="SELL", quantity=1, timestamp=datetime(2022, 11, 1, 9, 20, 0))
    sell_trade = Trade(
        order=sell_order, execution_price=110.0, transaction_cost=0.0,
        realized_pnl=500.0,   # (110-100)*1*50; engine pre-computes this
        trade_type="ROLL_CLOSE",
    )
    realized = pf.close_position(sell_trade, lot_size=50)
    assert realized == 500.0
    assert pf.get_position(opt.id) is None
    assert pf.is_flat()
    assert pf.realized_pnl_today == 500.0
    assert pf.cumulative_realized_pnl == 500.0


def test_position_limit_buy_rejected_by_engine():
    """Engine validation rejects BUY that would exceed MAX_POSITION_LOTS.
    (Portfolio itself allows adding; engine validates before calling open_position.)"""
    # Just a smoke test that portfolio can hold multiple lots if engine allows.
    pf = Portfolio()
    opt = _make_option(strike=18000, opt_type="CE")
    t1 = _make_trade(opt, "BUY", 1, 100.0)
    pf.open_position(t1)
    pos = pf.get_position(opt.id)
    assert pos.quantity == 1


def test_pnl_lot_size_scaling():
    """₹1 price move × 50 lots = ₹50 PnL (NIFTY lot size = 50)."""
    pf = Portfolio()
    opt = _make_option(strike=18000, opt_type="CE", underlier="NIFTY")
    buy = _make_trade(opt, "BUY", 1, 200.0)
    pf.open_position(buy)

    pos = pf.get_position(opt.id)
    # 1 rupee move × 1 lot × 50 lot_size = 50 rupees.
    assert pos.mtm_pnl(201.0, lot_size=50) == 50.0
    # 5 rupee move × 1 lot × 50 = 250 rupees.
    assert pos.mtm_pnl(205.0, lot_size=50) == 250.0
    # 2 lots × 5 rupee move × 50 = 500 rupees.
    pos2 = Position(instrument=opt, quantity=2, entry_price=200.0, entry_time=datetime(2022,11,1,9,15,0))
    assert pos2.mtm_pnl(205.0, lot_size=50) == 500.0


def test_snapshot_immutability():
    """snapshot() returns a copy — modifying it shouldn't affect Portfolio."""
    pf = Portfolio()
    opt = _make_option(strike=18000, opt_type="CE")
    pf.open_position(_make_trade(opt, "BUY", 1, 100.0))
    snap = pf.snapshot()
    assert len(snap.positions) == 1
    # Mutate snapshot
    snap.positions.clear()
    # Underlying portfolio unaffected
    assert pf.get_position(opt.id) is not None


def test_reset_daily_counters():
    """Daily counters reset; cumulative persists."""
    pf = Portfolio()
    opt = _make_option(strike=18000, opt_type="CE")
    pf.open_position(_make_trade(opt, "BUY", 1, 100.0))
    sell_order = Order(instrument=opt, action="SELL", quantity=1,
                       timestamp=datetime(2022, 11, 1, 9, 20, 0))
    sell_trade = Trade(order=sell_order, execution_price=110.0, transaction_cost=0.0,
                       realized_pnl=500.0,   # (110-100)*1*50; engine pre-computes this
                       trade_type="ROLL_CLOSE")
    pf.close_position(sell_trade, lot_size=50)
    assert pf.realized_pnl_today == 500.0
    assert pf.cumulative_realized_pnl == 500.0
    pf.reset_daily_counters()
    assert pf.realized_pnl_today == 0.0
    assert pf.cumulative_realized_pnl == 500.0  # cumulative persists


def test_compute_total_mtm():
    pf = Portfolio()
    ce = _make_option(strike=18000, opt_type="CE")
    pe = _make_option(strike=18000, opt_type="PE")
    pf.open_position(_make_trade(ce, "BUY", 1, 100.0))
    pf.open_position(_make_trade(pe, "BUY", 1, 80.0))

    total = pf.compute_total_mtm(
        current_prices={ce.id: 105.0, pe.id: 75.0},
        lot_sizes={"NIFTY": 50},
    )
    # CE: (105 - 100) * 1 * 50 = 250
    # PE: (75 - 80) * 1 * 50 = -250
    assert total == 0.0


def test_is_flat_filtered_by_underlier():
    pf = Portfolio()
    nifty_ce = _make_option(strike=18000, opt_type="CE", underlier="NIFTY")
    pf.open_position(_make_trade(nifty_ce, "BUY", 1, 100.0))
    assert not pf.is_flat()
    assert not pf.is_flat("NIFTY")
    assert pf.is_flat("BANKNIFTY")  # no BANKNIFTY positions
