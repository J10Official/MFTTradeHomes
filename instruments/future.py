"""
instruments/future.py — Future(Instrument) subclass.

Used only as a price reference in this system (the "current market price" of
the underlying index). Not traded.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List

from instruments.base import Instrument


@dataclass
class Future(Instrument):
    """An NSE futures contract. Used as price reference only in this system."""
    expiry: date
    series: str   # "I", "II", "III"
    asset_class: str = field(default="future", init=False, repr=False)

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.underlier}-{self.series}"

    @property
    def valid_actions(self) -> List[str]:
        return ["BUY", "SELL"]

    @property
    def display_name(self) -> str:
        return f"{self.underlier} Futures-{self.series} exp:{self.expiry}"
