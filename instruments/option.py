"""
instruments/option.py — Option(Instrument) subclass.

Represents an NSE index options contract. The id is the CSV filename stem,
e.g. "NIFTY22110314550PE".
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List

from instruments.base import Instrument


@dataclass
class Option(Instrument):
    """An NSE index options contract."""
    expiry:      date   # Contract expiry date
    strike:      int    # Strike price in points
    option_type: str    # "CE" (Call) or "PE" (Put)
    # asset_class is set automatically; allow default so callers don't pass it.
    asset_class: str = field(default="option", init=False, repr=False)

    def __post_init__(self):
        if not self.id:
            expiry_str = self.expiry.strftime("%y%m%d")
            self.id = f"{self.underlier}{expiry_str}{self.strike}{self.option_type}"
        if self.option_type not in ("CE", "PE"):
            raise ValueError(f"Invalid option_type '{self.option_type}' for {self.id}")

    @property
    def valid_actions(self) -> List[str]:
        # European-style index options: can only BUY or SELL (no exercise)
        return ["BUY", "SELL"]

    @property
    def display_name(self) -> str:
        return f"{self.underlier} {self.strike}{self.option_type} exp:{self.expiry}"
