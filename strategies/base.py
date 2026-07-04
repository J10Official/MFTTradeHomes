"""
strategies/base.py — Abstract BaseStrategy interface.

The strategy is a pure function of its inputs:
  same MarketContext → same Orders, every time.
This makes it trivially unit-testable and swappable without touching the engine.
"""

from abc import ABC, abstractmethod
from typing import List, ClassVar, Dict, Any, Type

from data.market_state import MarketContext
from engine.order import Order


class BaseStrategy(ABC):
    """
    Interface that all strategies must implement.

    Contract:
      - generate_signals() is called once per simulation timestep by the engine.
      - It receives a complete, immutable MarketContext.
      - It returns a list of Orders (may be empty = do nothing).
      - It MUST NOT access any external data, files, or state.
      - It MUST NOT modify the portfolio (the engine does this).
      - It MUST NOT store mutable state between calls that depends on execution
        (use context.portfolio for state — the engine keeps it authoritative).
      - It MAY store immutable configuration (parameters) as instance variables.
    """

    @abstractmethod
    def generate_signals(self, context: MarketContext) -> List[Order]:
        """
        Given complete market context, return orders to execute this timestep.
        Orders in the returned list are executed in order.
        Sell orders should precede buy orders (cannot hold > MAX_POSITION_LOTS).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier string for logging and reports. e.g. 'atm_straddle'"""
        ...

    @property
    def parameters(self) -> Dict[str, Any]:
        """Return dict of hyperparameters for the tuner. Default: empty dict."""
        return {}

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> 'BaseStrategy':
        """Instantiate strategy from a parameter dict. Default: ignore params."""
        return cls()
