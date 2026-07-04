"""
engine/order.py — Order and Trade dataclasses.

An Order is the strategy's intent (not yet executed).
A Trade is a successfully executed Order.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from instruments.base import Instrument


@dataclass
class Order:
    """
    An instruction from the strategy to the engine. Represents intent — not yet
    executed.

    Attributes:
        instrument:  The Instrument this order applies to.
        action:      "BUY" or "SELL" (must be in instrument.valid_actions).
        quantity:    In lots. Must be > 0.
        timestamp:   When the signal was generated (current sim time).
        price_limit: Optional max/min price. None = market order (execute at
                     last known price).
        metadata:    Optional dict for strategy-supplied hints.
                     e.g. {"is_roll": True} — used by the engine for trade
                     classification without heuristic position scanning.
    """
    instrument:  Instrument
    action:      str
    quantity:    int
    timestamp:   datetime
    price_limit: Optional[float] = None
    metadata:    Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.action not in self.instrument.valid_actions:
            raise ValueError(
                f"Invalid action {self.action!r} for instrument {self.instrument.id} "
                f"(valid: {self.instrument.valid_actions})"
            )
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be > 0, got {self.quantity}")



@dataclass
class Trade:
    """
    A successfully executed order. Created by the engine after validating and
    executing an Order.

    Attributes:
        order:            The original Order.
        execution_price:  Actual price at execution (last known price in base case).
        transaction_cost: In ₹. Computed from config rates.
        realized_pnl:     0 for BUY. Entry-to-exit PnL for SELL (₹, lot-size adjusted).
        trade_type:       "OPEN", "ROLL_CLOSE", "ROLL_OPEN", "EOD_CLOSE", "MANUAL_CLOSE".
    """
    order:            Order
    execution_price:  float
    transaction_cost: float
    realized_pnl:     float
    trade_type:       str
