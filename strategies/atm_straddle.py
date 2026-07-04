"""
strategies/atm_straddle.py — Assignment strategy implementation.

Buy ATM call + put simultaneously (a straddle).
Roll to new ATM strike whenever the futures price moves enough that a different
strike becomes ATM. Flatten all positions at end of day.

The strategy is a PURE function of MarketContext:
  - Same context → same orders, every time.
  - No internal mutable state.
  - All "what do we hold" knowledge comes from context.portfolio.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List

from data.market_state import MarketContext
from engine.order import Order
from instruments.option import Option
from strategies.base import BaseStrategy


class ATMStraddle(BaseStrategy):
    """
    Assignment strategy: buy ATM CE + PE at the strike nearest the futures price.
    When futures move enough that the ATM strike changes, sell the old straddle
    and buy the new one. The engine handles EOD flatten.
    """

    def __init__(self, max_position_lots: int = 1):
        self._max_position_lots = max_position_lots

    @property
    def name(self) -> str:
        return "atm_straddle"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"max_position_lots": self._max_position_lots}

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> 'ATMStraddle':
        return cls(max_position_lots=int(params.get("max_position_lots", 1)))

    def generate_signals(self, context: MarketContext) -> List[Order]:
        atm_strike = context.atm_strike
        ts = context.timestamp
        underlier = context.underlier
        nearest_expiry = context.nearest_expiry  # used for canonical instrument ID

        # Find currently held CE / PE positions for this underlier.
        held_ce: Option = None
        held_pe: Option = None
        for pos in context.portfolio.positions.values():
            inst = pos.instrument
            if not isinstance(inst, Option):
                continue
            if inst.underlier != underlier:
                continue
            if inst.option_type == "CE":
                held_ce = inst
            elif inst.option_type == "PE":
                held_pe = inst

        flat = (held_ce is None) and (held_pe is None)
        partial = (held_ce is None) != (held_pe is None)  # exactly one leg held
        holding_atm = (
            held_ce is not None and held_pe is not None
            and held_ce.strike == atm_strike and held_pe.strike == atm_strike
        )

        if flat:
            # Open a new straddle at the current ATM strike.
            ce_price = context.get_atm_price("CE")
            pe_price = context.get_atm_price("PE")
            orders: List[Order] = []
            if ce_price is not None:
                orders.append(self._make_buy(atm_strike, "CE", underlier, nearest_expiry, ts))
            if pe_price is not None:
                orders.append(self._make_buy(atm_strike, "PE", underlier, nearest_expiry, ts))
            return orders

        if partial:
            # One leg is missing (was unavailable when first opened).
            # Try to complete the straddle without disturbing the held leg.
            # Only add the missing leg if it's at the current ATM strike
            # (if the ATM has moved, fall through to the roll branch instead).
            orders: List[Order] = []
            if held_ce is None:
                # Missing CE: try to open it if ATM strike matches held PE
                if held_pe is not None and held_pe.strike == atm_strike:
                    ce_price = context.get_atm_price("CE")
                    if ce_price is not None:
                        orders.append(self._make_buy(atm_strike, "CE", underlier, nearest_expiry, ts))
                    return orders  # wait if still unavailable; don't sell PE
            elif held_pe is None:
                # Missing PE: try to open it if ATM strike matches held CE
                if held_ce is not None and held_ce.strike == atm_strike:
                    pe_price = context.get_atm_price("PE")
                    if pe_price is not None:
                        orders.append(self._make_buy(atm_strike, "PE", underlier, nearest_expiry, ts))
                    return orders  # wait if still unavailable; don't sell CE
            # ATM has moved while we held only one leg — fall through to roll.

        if holding_atm:
            return []  # hold

        # Otherwise: roll. SELL held instruments FIRST, then BUY new ATM.
        # Tag BUY orders with is_roll=True so the engine can classify them
        # without heuristic position scanning.
        orders: List[Order] = []
        if held_ce is not None:
            orders.append(self._make_sell(held_ce.strike, "CE", underlier, nearest_expiry, ts))
        if held_pe is not None:
            orders.append(self._make_sell(held_pe.strike, "PE", underlier, nearest_expiry, ts))
        ce_price = context.get_atm_price("CE")
        pe_price = context.get_atm_price("PE")
        if ce_price is not None:
            orders.append(self._make_buy(atm_strike, "CE", underlier, nearest_expiry, ts,
                                         is_roll=True))
        if pe_price is not None:
            orders.append(self._make_buy(atm_strike, "PE", underlier, nearest_expiry, ts,
                                         is_roll=True))
        return orders


    # ─── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _make_instrument(strike: int, opt_type: str, underlier: str,
                         expiry: date, ts: datetime) -> Option:
        """
        Build an Option with a canonical id matching the NSE filename convention.

        The id format is: {UNDERLIER}{YYMMDD}{STRIKE}{TYPE}
        e.g. NIFTY22110318000CE

        This must match what the parser produces from the CSV filenames so that
        portfolio.get_position(instrument.id) can find positions correctly.
        The engine uses (strike, option_type) for price lookup, but uses id for
        portfolio tracking — so the id must be stable and consistent.
        """
        expiry_str = expiry.strftime("%y%m%d")
        inst_id = f"{underlier}{expiry_str}{strike}{opt_type}"
        return Option(
            id=inst_id,
            underlier=underlier,
            expiry=expiry,
            strike=strike,
            option_type=opt_type,
        )

    @staticmethod
    def _make_buy(strike: int, opt_type: str, underlier: str,
                  expiry: date, ts: datetime, is_roll: bool = False) -> Order:
        opt = ATMStraddle._make_instrument(strike, opt_type, underlier, expiry, ts)
        return Order(
            instrument=opt,
            action="BUY",
            quantity=1,
            timestamp=ts,
            metadata={"is_roll": is_roll},
        )

    @staticmethod
    def _make_sell(strike: int, opt_type: str, underlier: str,
                   expiry: date, ts: datetime) -> Order:
        opt = ATMStraddle._make_instrument(strike, opt_type, underlier, expiry, ts)
        return Order(instrument=opt, action="SELL", quantity=1, timestamp=ts)
