"""
tuner/base.py — ParameterSpace, TrialResult, TunerResult dataclasses + train/test split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple


@dataclass
class ParameterSpace:
    """
    Defines the search space for hyperparameter tuning.

    discrete: parameters with an explicit list of values to try.
    continuous: parameters with (min, max, step) for random sampling.
    """
    discrete:   Dict[str, List[Any]]                       = field(default_factory=dict)
    continuous: Dict[str, Tuple[float, float, float]]      = field(default_factory=dict)


@dataclass
class TrialResult:
    """Result of a single hyperparameter configuration backtest."""
    params:           Dict[str, Any]
    train_pnl:        float
    test_pnl:         float
    train_sharpe:     float
    test_sharpe:      float
    overfitting_flag: bool
    n_trades:         int
    runtime_seconds:  float


@dataclass
class TunerResult:
    """Complete output from a hyperparameter search."""
    search_type:          str
    n_trials_completed:   int
    trial_results:        List[TrialResult]
    best_params_by_train: Dict[str, Any]
    best_params_by_test:  Dict[str, Any]
    recommended_params:   Dict[str, Any]
    train_date_range:     Tuple[date, date]
    test_date_range:      Tuple[date, date]
    best_train_pnl:       float = 0.0
    best_test_pnl:        float = 0.0
    recommended_pnl:      float = 0.0


def compute_train_test_split(
    start_date: date,
    end_date:   date,
    train_fraction: float,
) -> Tuple[Tuple[date, date], Tuple[date, date]]:
    """
    Split [start_date, end_date] into train and test periods.
    Train = first train_fraction of trading days (skipping weekends).
    Test = remaining days.
    """
    # Enumerate weekdays
    days = []
    cur = start_date
    while cur <= end_date:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    if not days:
        raise ValueError("No trading days in date range")
    n = len(days)
    train_end_idx = max(0, int(n * train_fraction) - 1)
    train_start = days[0]
    train_end   = days[train_end_idx]
    if train_end_idx + 1 < n:
        test_start = days[train_end_idx + 1]
        test_end   = days[-1]
    else:
        # Degenerate: no test set
        test_start = train_end
        test_end = train_end
    return ((train_start, train_end), (test_start, test_end))


def flag_overfitting(train_pnl: float, test_pnl: float, threshold: float = 0.5) -> bool:
    """True if test PnL degrades by more than `threshold` relative to |train_pnl|."""
    if train_pnl == 0:
        return False
    degradation = (train_pnl - test_pnl) / abs(train_pnl)
    return degradation > threshold
