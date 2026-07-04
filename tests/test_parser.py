"""
tests/test_parser.py — Test the NSE filename parser.
"""

import pytest
from datetime import date

from data.parser import (
    parse_option_filename,
    get_nearest_expiry,
)
from instruments.option import Option


def test_parse_nifty():
    opt = parse_option_filename("NIFTY22110314550PE.csv")
    assert opt.underlier == "NIFTY"
    assert opt.expiry == date(2022, 11, 3)
    assert opt.strike == 14550
    assert opt.option_type == "PE"
    assert opt.id == "NIFTY22110314550PE"
    assert opt.asset_class == "option"


def test_parse_banknifty():
    opt = parse_option_filename("BANKNIFTY22112443200CE.csv")
    assert opt.underlier == "BANKNIFTY"
    assert opt.expiry == date(2022, 11, 24)
    assert opt.strike == 43200
    assert opt.option_type == "CE"


def test_parse_finnifty():
    opt = parse_option_filename("FINNIFTY22110719500CE.csv")
    assert opt.underlier == "FINNIFTY"
    assert opt.expiry == date(2022, 11, 7)
    assert opt.strike == 19500
    assert opt.option_type == "CE"


def test_parse_invalid():
    with pytest.raises(ValueError):
        parse_option_filename("garbage_string.csv")


def test_parse_short_strike():
    # Strike with 4 digits (low strike, e.g. PE on a small index)
    opt = parse_option_filename("NIFTY2211039999PE.csv")
    assert opt.strike == 9999
    assert opt.option_type == "PE"


def test_parse_long_underlier_prefix():
    """BANKNIFTY must not be matched as NIFTY."""
    opt = parse_option_filename("BANKNIFTY22112443200CE.csv")
    assert opt.underlier == "BANKNIFTY"
    assert opt.underlier != "NIFTY"


def test_parse_invalid_option_type():
    with pytest.raises(ValueError):
        parse_option_filename("NIFTY22110314550XX.csv")


def test_parse_no_extension():
    """Parser should work without .csv extension too."""
    opt = parse_option_filename("NIFTY22110314550PE")
    assert opt.underlier == "NIFTY"
    assert opt.strike == 14550


def test_get_nearest_expiry_basic():
    expiries = [date(2022, 11, 3), date(2022, 11, 10), date(2022, 11, 17)]
    assert get_nearest_expiry(date(2022, 11, 1), expiries) == date(2022, 11, 3)
    assert get_nearest_expiry(date(2022, 11, 5), expiries) == date(2022, 11, 10)


def test_get_nearest_expiry_on_expiry_day():
    """On the expiry date itself, use that expiry (settlement is post-session)."""
    expiries = [date(2022, 11, 3), date(2022, 11, 10)]
    assert get_nearest_expiry(date(2022, 11, 3), expiries) == date(2022, 11, 3)


def test_get_nearest_expiry_no_future():
    with pytest.raises(ValueError):
        get_nearest_expiry(date(2022, 12, 1), [date(2022, 11, 3)])
