"""
instruments/base.py — Abstract Instrument base class.

All tradeable (or referenceable) financial instruments inherit from this.
The two required abstract properties are:
  - valid_actions: list of strings ("BUY", "SELL", etc.)
  - display_name:  human-readable name for logging
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List


@dataclass
class Instrument(ABC):
    """
    Base class for any tradeable (or referenceable) financial instrument.
    All instruments have a unique ID and belong to an underlier.
    """
    id:          str    # Unique string identifier. For options: the filename stem.
    underlier:   str    # "NIFTY", "BANKNIFTY", "FINNIFTY", "RELIANCE", etc.
    asset_class: str    # "option", "future", "stock"

    @property
    @abstractmethod
    def valid_actions(self) -> List[str]:
        """Returns list of valid order actions for this instrument type."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for logging and reports."""
        ...
