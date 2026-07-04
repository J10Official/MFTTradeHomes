"""
tuner/random_search.py — Random sampling of the parameter space.

Recommended over grid search when total grid combinations > 50.
"""

from __future__ import annotations

import logging
import random
import sys
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
from tuner.grid_search import _evaluate_one


logger = logging.getLogger(__name__)


class RandomSearchTuner:
    """Random sampling of the parameter space."""

    def __init__(
        self,
        parameter_space: ParameterSpace,
        underliers: List[str],
        full_start: date,
        full_end:   date,
        train_fraction: float = 0.5,
        n_trials: int = 50,
        n_workers: int = 4,
        seed: int = 42,
        overfit_threshold: float = 0.5,
    ):
        self.parameter_space = parameter_space
        self.underliers = underliers
        self.full_start = full_start
        self.full_end = full_end
        self.train_fraction = train_fraction
        self.n_trials = n_trials
        self.n_workers = max(1, n_workers)
        self.seed = seed
        self.overfit_threshold = overfit_threshold

    def _sample_one(self, rng: random.Random) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for k, values in self.parameter_space.discrete.items():
            params[k] = rng.choice(values)
        for k, (lo, hi, step) in self.parameter_space.continuous.items():
            # Sample uniformly then round to step.
            v = rng.uniform(lo, hi)
            if step > 0:
                v = round(round(v / step) * step, 6)
            params[k] = v
        return params

    def _sample_n(self, n: int) -> List[Dict[str, Any]]:
        rng = random.Random(self.seed)
        seen = set()
        out: List[Dict[str, Any]] = []
        attempts = 0
        while len(out) < n and attempts < n * 10:
            p = self._sample_one(rng)
            key = tuple(sorted(p.items()))
            if key not in seen:
                seen.add(key)
                out.append(p)
            attempts += 1
        return out

    def run(self) -> TunerResult:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        (train_start, train_end), (test_start, test_end) = compute_train_test_split(
            self.full_start, self.full_end, self.train_fraction
        )

        combos = self._sample_n(self.n_trials)
        n_total = len(combos)
        print(f"[random_search] {n_total} trials, {self.n_workers} workers", file=sys.stderr)

        results: List[TrialResult] = []
        if self.n_workers == 1 or n_total == 1:
            for i, params in enumerate(combos):
                print(f"[random_search] {i+1}/{n_total}...", file=sys.stderr)
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
                    print(f"[random_search] {done}/{n_total} complete", file=sys.stderr)

        return self._build_result(results, (train_start, train_end), (test_start, test_end))

    def _build_result(
        self,
        results: List[TrialResult],
        train_range: Tuple[date, date],
        test_range:  Tuple[date, date],
    ) -> TunerResult:
        if not results:
            return TunerResult(
                search_type="random",
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
        recommended = best_test

        return TunerResult(
            search_type="random",
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
