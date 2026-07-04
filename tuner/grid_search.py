"""
tuner/grid_search.py — Exhaustive grid search.

Generates the cartesian product of all parameter values and submits them to
a ProcessPoolExecutor.
"""

from __future__ import annotations

import itertools
import logging
import time as _time
from datetime import date
from typing import Any, Dict, List, Tuple

from tuner.base import (
    ParameterSpace,
    TrialResult,
    TunerResult,
    compute_train_test_split,
    flag_overfitting,
)


logger = logging.getLogger(__name__)


# ─── standalone evaluation function (must be picklable for ProcessPool) ─────


def _evaluate_one(
    params: Dict[str, Any],
    underliers: List[str],
    train_start: date,
    train_end:   date,
    test_start:  date,
    test_end:    date,
    overfit_threshold: float = 0.5,
) -> TrialResult:
    """
    Run backtest with given params on train set, then on test set.
    Compute overfitting flag.
    """
    # Heavy imports happen here so the parent doesn't need to pickle them.
    import sys
    import os
    # Mark this process as a worker so the parent's SIGTERM handler
    # is NOT installed (which would otherwise raise KeyboardInterrupt on
    # any signal and abort the trial).
    os.environ["MFT_WORKER"] = "1"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import importlib
    import config as cfg
    from engine.backtest import BacktestEngine
    from engine.cache import CacheManager
    from analytics.metrics import compute_all_metrics
    from strategies.atm_straddle import ATMStraddle

    # Apply params to config (we mutate a copy to avoid global state issues).
    # Easier: write params to the live config module.
    for k, v in params.items():
        mapped = {
            "strike_tie_break":               "STRIKE_TIE_BREAK",
            "eod_flatten_buffer_minutes":     "EOD_FLATTEN_BUFFER_MINUTES",
            "brokerage_per_lot_per_leg":      "BROKERAGE_PER_LOT_PER_LEG",
            "stt_rate":                       "STT_RATE",
            "exchange_charges_rate":          "EXCHANGE_CHARGES_RATE",
            "price_staleness_threshold_seconds": "PRICE_STALENESS_THRESHOLD_SECONDS",
            "timestep_seconds":               "TIMESTEP_SECONDS",
            "max_position_lots":              "MAX_POSITION_LOTS",
        }
        attr = mapped.get(k)
        if attr and hasattr(cfg, attr):
            setattr(cfg, attr, v)

    cache = CacheManager(cfg.CACHE_DIR)

    def _run(start: date, end: date):
        strategy = ATMStraddle(max_position_lots=cfg.MAX_POSITION_LOTS)
        eng = BacktestEngine(
            strategy=strategy, config=cfg, cache=cache,
            underliers=underliers, start_date=start, end_date=end,
            progress_to_stderr=False,
        )
        return eng.run(use_cache=True)

    t0 = _time.time()
    train_result = _run(train_start, train_end)
    train_metrics = compute_all_metrics(train_result)
    train_pnl = float(train_metrics.get("combined", {}).get("total_pnl_inr", 0.0) or 0.0)
    train_sharpe = float(train_metrics.get("combined", {}).get("sharpe_ratio", 0.0) or 0.0)

    test_result = _run(test_start, test_end)
    test_metrics = compute_all_metrics(test_result)
    test_pnl = float(test_metrics.get("combined", {}).get("total_pnl_inr", 0.0) or 0.0)
    test_sharpe = float(test_metrics.get("combined", {}).get("sharpe_ratio", 0.0) or 0.0)

    overfit = flag_overfitting(train_pnl, test_pnl, overfit_threshold)
    n_trades = int(test_metrics.get("combined", {}).get("total_trades", 0) or 0)
    runtime = _time.time() - t0

    return TrialResult(
        params=params,
        train_pnl=train_pnl,
        test_pnl=test_pnl,
        train_sharpe=train_sharpe,
        test_sharpe=test_sharpe,
        overfitting_flag=overfit,
        n_trades=n_trades,
        runtime_seconds=runtime,
    )


# ─── GridSearchTuner ─────────────────────────────────────────────────────────


class GridSearchTuner:
    """Exhaustive grid search over all combinations in the parameter space."""

    def __init__(
        self,
        parameter_space: ParameterSpace,
        underliers: List[str],
        full_start: date,
        full_end:   date,
        train_fraction: float = 0.5,
        n_workers: int = 4,
        overfit_threshold: float = 0.5,
    ):
        self.parameter_space = parameter_space
        self.underliers = underliers
        self.full_start = full_start
        self.full_end = full_end
        self.train_fraction = train_fraction
        self.n_workers = max(1, n_workers)
        self.overfit_threshold = overfit_threshold

    def _enumerate_grid(self) -> List[Dict[str, Any]]:
        discrete_keys = list(self.parameter_space.discrete.keys())
        discrete_values = list(self.parameter_space.discrete.values())
        continuous_keys = list(self.parameter_space.continuous.keys())

        # Generate continuous values from (min, max, step)
        continuous_value_lists: List[List[float]] = []
        for k in continuous_keys:
            lo, hi, step = self.parameter_space.continuous[k]
            vals = []
            v = lo
            while v <= hi + 1e-9:
                vals.append(round(v, 6))
                v += step
            continuous_value_lists.append(vals)

        all_combos: List[Dict[str, Any]] = []
        # Cartesian product of discrete × continuous
        for d_combo in itertools.product(*discrete_values) if discrete_values else [()]:
            for c_combo in itertools.product(*continuous_value_lists) if continuous_value_lists else [()]:
                params = dict(zip(discrete_keys, d_combo))
                params.update(dict(zip(continuous_keys, c_combo)))
                all_combos.append(params)
        return all_combos

    def run(self) -> TunerResult:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import sys

        (train_start, train_end), (test_start, test_end) = compute_train_test_split(
            self.full_start, self.full_end, self.train_fraction
        )

        combos = self._enumerate_grid()
        n_total = len(combos)
        if n_total > 500:
            logger.warning("Grid search has %d combinations — consider random search instead.", n_total)
        print(f"[grid_search] {n_total} combinations, {self.n_workers} workers", file=sys.stderr)

        results: List[TrialResult] = []
        if self.n_workers == 1 or n_total == 1:
            for i, params in enumerate(combos):
                print(f"[grid_search] {i+1}/{n_total}...", file=sys.stderr)
                r = _evaluate_one(params, self.underliers,
                                  train_start, train_end, test_start, test_end,
                                  self.overfit_threshold)
                results.append(r)
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
                futures = {
                    ex.submit(_evaluate_one, p, self.underliers,
                              train_start, train_end, test_start, test_end,
                              self.overfit_threshold): p
                    for p in combos
                }
                done = 0
                for fut in as_completed(futures):
                    try:
                        r = fut.result()
                        results.append(r)
                    except Exception as e:
                        logger.exception("Trial failed: %s", e)
                    done += 1
                    print(f"[grid_search] {done}/{n_total} complete", file=sys.stderr)

        return self._build_result(results, (train_start, train_end), (test_start, test_end))

    def _build_result(
        self,
        results: List[TrialResult],
        train_range: Tuple[date, date],
        test_range:  Tuple[date, date],
    ) -> TunerResult:
        if not results:
            return TunerResult(
                search_type="grid",
                n_trials_completed=0,
                trial_results=[],
                best_params_by_train={},
                best_params_by_test={},
                recommended_params={},
                train_date_range=train_range,
                test_date_range=test_range,
            )

        best_train = max(results, key=lambda r: r.train_pnl)
        best_test  = max(results, key=lambda r: r.test_pnl)
        # Recommended: best by test_pnl, but flag if overfit.
        recommended = best_test

        return TunerResult(
            search_type="grid",
            n_trials_completed=len(results),
            trial_results=results,
            best_params_by_train=best_train.params,
            best_params_by_test=best_test.params,
            recommended_params=recommended.params,
            train_date_range=train_range,
            test_date_range=test_range,
            best_train_pnl=best_train.train_pnl,
            best_test_pnl=best_test.test_pnl,
            recommended_pnl=recommended.test_pnl,
        )
