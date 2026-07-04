"""
engine/portfolio.py — Position tracking and PnL computation.

The Portfolio class is the authoritative source of truth for what we hold and
how much PnL we have generated. The strategy never mutates it directly — only
the engine does, via open_position/close_position after a Trade is executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from engine.order import Trade
from instruments.base import Instrument


@dataclass
class Position:
    """
    A currently held open position in one instrument.
    """
    instrument:  Instrument
    quantity:    int       # In lots. Positive = long.
    entry_price: float     # Premium or price paid per unit at entry
    entry_time:  datetime

    def mtm_pnl(self, current_price: float, lot_size: int) -> float:
        """
        Mark-to-market (unrealized) PnL for this position.
        Positive = currently profitable.
        """
        return (current_price - self.entry_price) * self.quantity * lot_size


@dataclass
class PortfolioSnapshot:
    """
    Read-only view of the portfolio passed to the strategy.
    The strategy MUST NOT modify this object.
    """
    positions:                Dict[str, Position]
    realized_pnl_today:       float
    cumulative_realized_pnl:  float
    trade_count_today:        int
    roll_count_today:         int


class Portfolio:
    """
    Stateful portfolio tracker.

    One Portfolio per underlier per day (the engine resets it appropriately).
    Actually, the engine keeps ONE Portfolio across the whole backtest so that
    cumulative_realized_pnl accumulates correctly; it calls reset_daily_counters()
    at the start of each new trading day.
    """

    def __init__(self):
        self._positions: Dict[str, Position] = {}
        self._realized_pnl_today: float = 0.0
        self._cumulative_realized_pnl: float = 0.0
        self._trade_count_today: int = 0
        self._roll_count_today: int = 0

    # ─── position management ────────────────────────────────────────────────

    def open_position(self, trade: Trade) -> None:
        """Record a new long position from a BUY trade."""
        inst_id = trade.order.instrument.id
        existing = self._positions.get(inst_id)
        if existing is not None:
            # Add to existing position (we don't average here — assignment uses
            # 1-lot max, but be defensive: if same instrument bought again
            # without selling, treat as adding to position at new price).
            new_qty = existing.quantity + trade.order.quantity
            new_entry_price = (
                (existing.entry_price * existing.quantity
                 + trade.execution_price * trade.order.quantity) / new_qty
            )
            self._positions[inst_id] = Position(
                instrument=existing.instrument,
                quantity=new_qty,
                entry_price=new_entry_price,
                entry_time=trade.order.timestamp,
            )
        else:
            self._positions[inst_id] = Position(
                instrument=trade.order.instrument,
                quantity=trade.order.quantity,
                entry_price=trade.execution_price,
                entry_time=trade.order.timestamp,
            )
        self._trade_count_today += 1

    def close_position(self, trade: Trade, lot_size: int) -> float:
        """
        Remove position for a SELL trade. Record realized PnL.
        Returns realized PnL in ₹.

        The realized PnL is taken directly from trade.realized_pnl (already
        computed by the engine in _execute_order, including cost deduction).
        We do NOT recompute it here to avoid double-counting transaction costs.
        """
        inst_id = trade.order.instrument.id
        pos = self._positions.get(inst_id)
        if pos is None:
            raise KeyError(f"No open position to close for {inst_id}")

        sell_qty = trade.order.quantity
        if sell_qty > pos.quantity:
            raise ValueError(
                f"SELL quantity {sell_qty} exceeds position {pos.quantity} for {inst_id}"
            )

        # Use the already-computed realized PnL from the Trade (avoids double-counting).
        realized = trade.realized_pnl

        if sell_qty == pos.quantity:
            del self._positions[inst_id]
        else:
            pos.quantity -= sell_qty

        self._realized_pnl_today += realized
        self._cumulative_realized_pnl += realized
        self._trade_count_today += 1
        return realized

    # ─── queries ────────────────────────────────────────────────────────────

    def get_position(self, instrument_id: str) -> Optional[Position]:
        """Return open position for instrument_id, or None if flat."""
        return self._positions.get(instrument_id)

    def get_all_positions(self) -> Dict[str, Position]:
        """Return all currently open positions."""
        return dict(self._positions)

    def compute_total_mtm(
        self,
        current_prices: Dict[str, float],
        lot_sizes: Dict[str, int],
    ) -> float:
        """
        Sum MTM PnL across all open positions.

        Args:
            current_prices: instrument.id -> current_price
            lot_sizes: underlier -> lot_size
        """
        total = 0.0
        for inst_id, pos in self._positions.items():
            price = current_prices.get(inst_id)
            if price is None:
                continue
            lot_size = lot_sizes.get(pos.instrument.underlier, 1)
            total += pos.mtm_pnl(price, lot_size)
        return total

    def is_flat(self, underlier: Optional[str] = None) -> bool:
        """True if no open positions (optionally filtered by underlier)."""
        if underlier is None:
            return len(self._positions) == 0
        return not any(p.instrument.underlier == underlier
                       for p in self._positions.values())

    def snapshot(self) -> PortfolioSnapshot:
        """Return a read-only PortfolioSnapshot for the strategy.

        Positions are shallow-copied (new dict + new Position objects) so a
        strategy cannot accidentally mutate the engine's authoritative state.
        """
        import copy
        return PortfolioSnapshot(
            positions={k: copy.copy(v) for k, v in self._positions.items()},
            realized_pnl_today=self._realized_pnl_today,
            cumulative_realized_pnl=self._cumulative_realized_pnl,
            trade_count_today=self._trade_count_today,
            roll_count_today=self._roll_count_today,
        )

    # ─── daily maintenance ──────────────────────────────────────────────────

    def reset_daily_counters(self) -> None:
        """Call at start of each new trading day."""
        self._realized_pnl_today = 0.0
        self._trade_count_today = 0
        self._roll_count_today = 0

    def increment_roll_count(self) -> None:
        self._roll_count_today += 1

    # ─── accessors used by analytics ─────────────────────────────────────────

    @property
    def cumulative_realized_pnl(self) -> float:
        return self._cumulative_realized_pnl

    @property
    def realized_pnl_today(self) -> float:
        return self._realized_pnl_today

    @property
    def trade_count_today(self) -> int:
        return self._trade_count_today

    @property
    def roll_count_today(self) -> int:
        return self._roll_count_today
