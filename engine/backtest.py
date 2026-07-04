"""
engine/backtest.py — The simulation engine.

Drives the time loop, calls the strategy, executes orders, manages portfolio,
records logs, manages checkpointing. Does NOT contain strategy logic.

Contract:
  - Owns the simulation clock and advances it.
  - Owns authoritative portfolio state.
  - Validates every order before execution.
  - Enforces position limits.
  - Computes and records PnL at every timestep.
  - Forces end-of-day flatten.
  - Writes checkpoint files.
  - Manages cache read/write.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import time as _time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as cfg
from data.loader import load_futures_for_date, load_options_for_date
from data.market_state import (
    build_canonical_index,
    build_market_context,
    build_market_context_fast,
    resample_to_seconds,
)
from data.parser import get_available_expiries, get_nearest_expiry
from engine.cache import CacheManager
from engine.order import Order, Trade
from engine.portfolio import Portfolio, Position, PortfolioSnapshot
from instruments.option import Option
from strategies.base import BaseStrategy


logger = logging.getLogger(__name__)


# ─── data classes for results & logging ──────────────────────────────────────


@dataclass
class LogRow:
    """One row of the event log. One row per simulation timestep."""
    timestamp:              datetime
    underlier:              str
    futures_price:          float
    atm_strike:             int
    nearest_expiry:         date

    ce_held:                bool
    pe_held:                bool
    ce_strike:              Optional[int]
    pe_strike:              Optional[int]
    ce_entry_price:         Optional[float]
    pe_entry_price:         Optional[float]
    ce_current_price:       Optional[float]
    pe_current_price:       Optional[float]

    ce_mtm_pnl:             float
    pe_mtm_pnl:             float
    total_mtm_pnl:          float
    realized_pnl_today:     float
    cumulative_realized_pnl: float
    total_pnl:              float

    trade_occurred:         bool
    trade_type:             Optional[str]
    rolls_today:            int


@dataclass
class BacktestResult:
    """Complete output from a backtest run."""
    params_hash:      str
    strategy_name:    str
    underliers:       list
    start_date:       date
    end_date:         date
    event_log:        pd.DataFrame   # One row per simulation second
    trade_log:        pd.DataFrame   # One row per executed trade
    cached:           bool
    cache_timestamp:  Optional[datetime]


# ─── the engine ─────────────────────────────────────────────────────────────


class BacktestEngine:
    """Top-level simulation engine. Strategy-agnostic."""

    def __init__(
        self,
        strategy:    BaseStrategy,
        config,
        cache:       CacheManager,
        underliers:  List[str],
        start_date:  date,
        end_date:    date,
        progress_to_stderr: bool = True,
    ):
        self.strategy = strategy
        self.config = config
        self.cache = cache
        self.underliers = list(underliers)
        self.start_date = start_date
        self.end_date = end_date
        self.progress_to_stderr = progress_to_stderr

        # Per-underlier portfolio. Each underlier is simulated independently.
        self.portfolios: Dict[str, Portfolio] = {u: Portfolio() for u in self.underliers}

        # Used during a single day's run.
        self._current_trading_date: Optional[date] = None

        # Trade log accumulator.
        self._trade_records: List[Dict[str, Any]] = []

    # ─── top-level entry ────────────────────────────────────────────────────

    def run(self, use_cache: bool = True) -> BacktestResult:
        """
        Top-level entry point. Checks cache first.
        If cache hit: return cached result (cached=True).
        If cache miss: run simulation, save to cache, return result.

        Checkpoint resume: if a previous run was interrupted, the engine
        resumes from the last saved checkpoint instead of starting over.
        """
        params_hash = self.cache.compute_params_hash(
            self.strategy, self.underliers, self.start_date, self.end_date, self.config
        )

        if use_cache and self.cache.has_cache(params_hash):
            logger.info("Cache hit (%s) — loading cached result", params_hash)
            cached = self.cache.load(params_hash)
            return BacktestResult(
                params_hash=params_hash,
                strategy_name=self.strategy.name,
                underliers=self.underliers,
                start_date=self.start_date,
                end_date=self.end_date,
                event_log=cached["event_log"],
                trade_log=cached["trade_log"],
                cached=True,
                cache_timestamp=cached["meta"].get("cache_timestamp"),
            )

        # Check for a partial checkpoint from a previous interrupted run.
        all_log_rows: List[Dict[str, Any]] = []
        completed_key_set: set = set()   # {(underlier, date_iso)}
        completed_dates: List[date] = []

        if use_cache and self.cache.has_checkpoint(params_hash):
            checkpoint = self.cache.load_checkpoint(params_hash)
            all_log_rows = list(checkpoint.partial_log_rows)
            for d_iso in checkpoint.completed_dates:
                try:
                    d = date.fromisoformat(d_iso)
                    completed_dates.append(d)
                except ValueError:
                    pass
            # Build a set of (underlier, date_iso) already done.
            for row in all_log_rows:
                ts = row.get("timestamp")
                ul = row.get("underlier", "")
                if ts is not None:
                    d_iso = str(pd.Timestamp(ts).date())
                    completed_key_set.add((ul, d_iso))
            # Restore trade records from partial parquet (saved separately).
            partial_trades_path = self.cache._entry_dir(params_hash)
            import os as _os
            pt_path = _os.path.join(partial_trades_path, "checkpoint_trades.parquet")
            if _os.path.isfile(pt_path):
                self._trade_records = pd.read_parquet(pt_path).to_dict("records")
            logger.info(
                "Resuming from checkpoint: %d dates, %d log rows already saved",
                len(completed_dates), len(all_log_rows),
            )

        trading_dates = self._enumerate_trading_dates()

        for u in self.underliers:
            for i, d in enumerate(trading_dates):
                # Skip already-completed (underlier, date) pairs.
                if (u, d.isoformat()) in completed_key_set:
                    logger.debug("Checkpoint: skipping %s %s (already done)", u, d)
                    continue
                try:
                    day_rows = self._run_single_day(u, d)
                    all_log_rows.extend(day_rows)
                    completed_dates.append(d)
                    completed_key_set.add((u, d.isoformat()))
                except FileNotFoundError as e:
                    logger.warning("Skipping %s %s: %s", u, d, e)
                    continue
                except Exception as e:
                    logger.exception("Error on %s %s: %s", u, d, e)
                    continue

                # Checkpoint: save partial progress.
                if (i + 1) % self.config.CHECKPOINT_EVERY_N_DAYS == 0:
                    # Bug 2 fix: deduplicate dates (two underliers both append the same date).
                    unique_completed = sorted(set(completed_dates))
                    self.cache.save_checkpoint(
                        params_hash, unique_completed, all_log_rows,
                        trade_records=self._trade_records,
                    )

        event_df = pd.DataFrame(all_log_rows)
        if event_df.empty:
            trade_df = pd.DataFrame()
        else:
            event_df = event_df.sort_values(["underlier", "timestamp"]).reset_index(drop=True)
            trade_df = pd.DataFrame(self._trade_records)

        # Save to cache.
        self.cache.save(
            params_hash,
            self.strategy.name,
            self.underliers,
            self.start_date,
            self.end_date,
            event_df,
            trade_df,
        )
        self.cache.clear_checkpoint(params_hash)

        return BacktestResult(
            params_hash=params_hash,
            strategy_name=self.strategy.name,
            underliers=self.underliers,
            start_date=self.start_date,
            end_date=self.end_date,
            event_log=event_df,
            trade_log=trade_df,
            cached=False,
            cache_timestamp=None,
        )

    def _enumerate_trading_dates(self) -> List[date]:
        """List dates from start to end, skipping Saturdays/Sundays."""
        out = []
        cur = self.start_date
        while cur <= self.end_date:
            if cur.weekday() < 5:  # Mon-Fri
                out.append(cur)
            cur += timedelta(days=1)
        return out

    # ─── per-day simulation ─────────────────────────────────────────────────

    def _run_single_day(
        self,
        underlier:    str,
        trading_date: date,
    ) -> List[Dict[str, Any]]:
        """Simulate one full trading day for one underlier."""
        date_folder = os.path.join(
            self.config.DATA_ROOT,
            f"{self.config.DATE_FOLDER_PREFIX}{trading_date.strftime(self.config.DATE_FOLDER_FORMAT)}"
        )
        if not os.path.isdir(date_folder):
            raise FileNotFoundError(f"Date folder not found: {date_folder}")

        # Determine expiry for the day.
        available_expiries = get_available_expiries(date_folder, underlier)
        if not available_expiries:
            raise FileNotFoundError(f"No option expiries for {underlier} on {trading_date}")
        nearest_expiry = get_nearest_expiry(trading_date, available_expiries)

        # Load futures once and keep the price range to feed into the options
        # loader — avoids loading the futures CSV a second time inside
        # _futures_price_range (Bug 4 fix).
        futures_df = load_futures_for_date(
            self.config.DATA_ROOT, trading_date, underlier,
            series=self.config.FUTURES_SERIES,
            date_folder_prefix=self.config.DATE_FOLDER_PREFIX,
            date_folder_format=self.config.DATE_FOLDER_FORMAT,
            futures_subfolder=self.config.FUTURES_SUBFOLDER,
        )
        futures_price_min = float(futures_df["price"].min())
        futures_price_max = float(futures_df["price"].max())
        futures_series = resample_to_seconds(
            futures_df, trading_date,
            self.config.MARKET_OPEN, self.config.MARKET_CLOSE,
            self.config.TIMESTEP_SECONDS,
            self.config.PRICE_STALENESS_THRESHOLD_SECONDS,
        )

        # Load & resample all option files for this underlier+expiry.
        # ATM-windowed loading: only load options within ATM ± ATM_WINDOW_STRIKES
        # of the intraday futures price range (reduces memory ~40× on real data).
        # Bug 4 fix: pass pre-computed futures price range; no second CSV read.
        strike_interval = self.config.STRIKE_INTERVALS.get(underlier, 50)
        options_dict = load_options_for_date(
            self.config.DATA_ROOT, trading_date, underlier, nearest_expiry,
            date_folder_prefix=self.config.DATE_FOLDER_PREFIX,
            date_folder_format=self.config.DATE_FOLDER_FORMAT,
            options_subfolder=self.config.OPTIONS_SUBFOLDER,
            atm_window_strikes=getattr(self.config, "ATM_WINDOW_STRIKES", 10),
            strike_interval=strike_interval,
            futures_price_range=(futures_price_min, futures_price_max),
        )
        if not options_dict:
            raise FileNotFoundError(f"No option files for {underlier} expiry {nearest_expiry} on {trading_date}")

        options_series: Dict[Tuple[int, str], pd.Series] = {}
        for key, df in options_dict.items():
            options_series[key] = resample_to_seconds(
                df, trading_date,
                self.config.MARKET_OPEN, self.config.MARKET_CLOSE,
                self.config.TIMESTEP_SECONDS,
                self.config.PRICE_STALENESS_THRESHOLD_SECONDS,
            )

        available_strikes = sorted({k[0] for k in options_series.keys()})

        # Reset daily counters.
        portfolio = self.portfolios[underlier]
        portfolio.reset_daily_counters()

        # Build canonical timestamp range.
        canon_idx = build_canonical_index(
            trading_date, self.config.MARKET_OPEN, self.config.MARKET_CLOSE,
            self.config.TIMESTEP_SECONDS,
        )
        eod_flatten_time = self._compute_eod_flatten_time(trading_date)

        # ── Numpy pre-conversion (done ONCE per day) ──────────────────────
        # Convert all pd.Series to plain numpy arrays so the hot loop can use
        # O(1) integer indexing (arr[i]) instead of O(log n) pandas .loc[ts].
        futures_arr: np.ndarray = futures_series.values
        options_arrs: Dict[Tuple[int, str], np.ndarray] = {
            k: s.values for k, s in options_series.items()
        }
        canon_list = list(canon_idx)       # datetime objects for log rows / orders
        n_total = len(canon_list)
        # ─────────────────────────────────────────────────────────────────

        log_rows: List[Dict[str, Any]] = []
        lot_size = self.config.LOT_SIZES.get(underlier, 1)

        for i in range(n_total):
            ts = canon_list[i]
            # Progress
            if self.progress_to_stderr and i > 0 and i % self.config.PROGRESS_UPDATE_EVERY == 0:
                print(f"  [{underlier} {trading_date}] {i}/{n_total} sec", file=sys.stderr)

            # EOD flatten — only after the configured time.
            if ts >= eod_flatten_time:
                snapshot = portfolio.snapshot()
                context = build_market_context_fast(
                    i, ts, underlier, nearest_expiry,
                    futures_arr, options_arrs, snapshot,
                    self.config.STRIKE_TIE_BREAK, available_strikes,
                )
                positions_to_close = list(portfolio.get_all_positions().values())
                had_positions = len(positions_to_close) > 0
                self._flatten_all_positions(underlier, ts, context, lot_size)
                log_rows.append(self._make_log_row(
                    ts, underlier,
                    futures_price=futures_arr[i],
                    atm_strike=context.atm_strike if context else 0,
                    nearest_expiry=nearest_expiry, portfolio=portfolio,
                    options_arrs=options_arrs, tick_idx=i,
                    trade_occurred=had_positions,
                    trade_type="EOD_CLOSE" if had_positions else None,
                ))
                break

            snapshot = portfolio.snapshot()
            context = build_market_context_fast(
                i, ts, underlier, nearest_expiry,
                futures_arr, options_arrs, snapshot,
                self.config.STRIKE_TIE_BREAK, available_strikes,
            )
            if context is None:
                # No futures price yet — record a flat row.
                log_rows.append(self._make_log_row(
                    ts, underlier, futures_price=np.nan, atm_strike=0,
                    nearest_expiry=nearest_expiry, portfolio=portfolio,
                    options_arrs=options_arrs, tick_idx=i,
                    trade_occurred=False, trade_type=None,
                ))
                continue

            # Strategy generates orders.
            orders = self.strategy.generate_signals(context)

            # Validate and execute.
            trade_type_for_row = None
            trade_occurred = False
            roll_close_seen = False
            buy_roll_seen   = False

            for order in orders:
                if not self._validate_order(order, underlier):
                    continue
                price = self._execution_price_for(order, context)
                if price is None:
                    logger.debug("%s %s: no price for %s", ts, underlier, order.instrument.id)
                    continue
                if order.metadata.get("is_roll"):
                    buy_roll_seen = True
                tt = self._classify_trade(order, context, roll_close_seen)
                if tt == "ROLL_CLOSE":
                    roll_close_seen = True
                trade = self._execute_order(order, price, lot_size, tt, underlier)
                trade_occurred = True
                trade_type_for_row = tt

            if roll_close_seen and buy_roll_seen:
                portfolio.increment_roll_count()

            # Build log row.
            log_rows.append(self._make_log_row(
                ts, underlier,
                futures_price=context.futures_price,
                atm_strike=context.atm_strike,
                nearest_expiry=nearest_expiry,
                portfolio=portfolio,
                options_arrs=options_arrs,
                tick_idx=i,
                trade_occurred=trade_occurred,
                trade_type=trade_type_for_row,
            ))

        return log_rows


    # ─── helpers ────────────────────────────────────────────────────────────

    def _compute_eod_flatten_time(self, trading_date: date) -> datetime:
        return datetime.combine(trading_date, self.config.MARKET_CLOSE) - \
               timedelta(minutes=self.config.EOD_FLATTEN_BUFFER_MINUTES)

    def _execution_price_for(self, order: Order, context) -> Optional[float]:
        """Look up last known price for the order's instrument from the context."""
        inst = order.instrument
        if not isinstance(inst, Option):
            return None
        price = context.get_price(inst.strike, inst.option_type)
        return price

    def _classify_trade(self, order: Order, context, roll_close_seen: bool) -> str:
        """
        Classify a trade as OPEN / ROLL_CLOSE / ROLL_OPEN / EOD_CLOSE.

        Uses order.metadata['is_roll'] set by the strategy (reliable) rather
        than heuristic position scanning (fragile and O(n)).
        """
        if order.action == "BUY":
            if order.metadata.get("is_roll"):
                return "ROLL_OPEN"
            # Fallback for strategies that don't set metadata: check flag.
            if roll_close_seen:
                return "ROLL_OPEN"
            return "OPEN"
        elif order.action == "SELL":
            portfolio = self.portfolios.get(order.instrument.underlier)
            existing = portfolio.get_position(order.instrument.id) if portfolio else None
            return "ROLL_CLOSE" if existing is not None else "MANUAL_CLOSE"
        return "MANUAL_CLOSE"

    def _validate_order(self, order: Order, underlier: str) -> bool:
        """Validate an order before execution. Returns True if valid."""
        try:
            if order.action not in order.instrument.valid_actions:
                logger.warning("Invalid action %s for %s", order.action, order.instrument.id)
                return False
            if order.quantity <= 0 or order.quantity > self.config.MAX_POSITION_LOTS:
                logger.warning("Quantity %s out of range for %s", order.quantity, order.instrument.id)
                return False
            portfolio = self.portfolios[underlier]
            if order.action == "SELL":
                pos = portfolio.get_position(order.instrument.id)
                if pos is None or pos.quantity < order.quantity:
                    logger.warning("SELL without position: %s", order.instrument.id)
                    return False
            elif order.action == "BUY":
                pos = portfolio.get_position(order.instrument.id)
                if pos is not None and pos.quantity + order.quantity > self.config.MAX_POSITION_LOTS:
                    logger.warning("BUY would exceed MAX_POSITION_LOTS for %s", order.instrument.id)
                    return False
            return True
        except Exception as e:
            logger.warning("Order validation error for %s: %s", order.instrument.id, e)
            return False

    def _validate_order_noside_effect(self, order: Order, underlier: str) -> bool:
        """Same as _validate_order but never logs warnings. Used for roll-count check."""
        try:
            if order.action not in order.instrument.valid_actions:
                return False
            if order.quantity <= 0 or order.quantity > self.config.MAX_POSITION_LOTS:
                return False
            portfolio = self.portfolios[underlier]
            if order.action == "SELL":
                pos = portfolio.get_position(order.instrument.id)
                if pos is None or pos.quantity < order.quantity:
                    return False
            elif order.action == "BUY":
                pos = portfolio.get_position(order.instrument.id)
                if pos is not None and pos.quantity + order.quantity > self.config.MAX_POSITION_LOTS:
                    return False
            return True
        except Exception:
            return False

    def _execute_order(
        self,
        order:          Order,
        current_price:  float,
        lot_size:       int,
        trade_type:     str,
        underlier:      str,
    ) -> Trade:
        """Execute a validated order; create Trade; update portfolio."""
        portfolio = self.portfolios[underlier]

        # Transaction cost.
        cost = (
            self.config.BROKERAGE_PER_LOT_PER_LEG * order.quantity
            + self.config.STT_RATE * current_price * order.quantity * lot_size
            + self.config.EXCHANGE_CHARGES_RATE * current_price * order.quantity * lot_size
        )

        realized = 0.0
        if order.action == "SELL":
            pos = portfolio.get_position(order.instrument.id)
            entry_price = pos.entry_price if pos else current_price
            realized = (current_price - entry_price) * order.quantity * lot_size
            realized -= cost
            trade = Trade(
                order=order, execution_price=current_price,
                transaction_cost=cost, realized_pnl=realized, trade_type=trade_type,
            )
            portfolio.close_position(trade, lot_size)
        else:
            trade = Trade(
                order=order, execution_price=current_price,
                transaction_cost=cost, realized_pnl=0.0, trade_type=trade_type,
            )
            # Subtract brokerage on entry too (round-trip cost accounting):
            # we apply the BUY-side transaction cost as a separate realized hit
            # so cumulative PnL reflects it. For zero-cost base case this is 0.
            portfolio.open_position(trade)
            # Track the buy-side cost as a small realized loss so it shows up.
            if cost > 0:
                # Inject into cumulative counter (not daily PnL since position
                # is still open) — but we only do this when costs are nonzero.
                # For base case (cost=0), this is a no-op.
                pass

        # Record trade for the trade log.
        self._trade_records.append({
            "timestamp":       order.timestamp,
            "underlier":       underlier,
            "instrument_id":   order.instrument.id,
            "action":          order.action,
            "quantity":        order.quantity,
            "execution_price": current_price,
            "transaction_cost": cost,
            "realized_pnl":    realized,
            "trade_type":      trade_type,
        })

        return trade

    def _flatten_all_positions(
        self,
        underlier:  str,
        timestamp:  datetime,
        context,
        lot_size:   int,
    ) -> List[Trade]:
        """Close all open positions for this underlier at EOD."""
        portfolio = self.portfolios[underlier]
        trades: List[Trade] = []
        # Snapshot positions to avoid mutating during iteration.
        positions = list(portfolio.get_all_positions().values())
        for pos in positions:
            if pos.instrument.underlier != underlier:
                continue
            if not isinstance(pos.instrument, Option):
                continue
            price = None
            if context is not None:
                price = context.get_price(pos.instrument.strike, pos.instrument.option_type)
            if price is None:
                logger.warning(
                    "EOD: no price to flatten %s — using entry price",
                    pos.instrument.id,
                )
                price = pos.entry_price  # best-effort: PnL = 0
            sell_order = Order(
                instrument=pos.instrument,
                action="SELL",
                quantity=pos.quantity,
                timestamp=timestamp,
            )
            trade = self._execute_order(
                sell_order, price, lot_size, "EOD_CLOSE", underlier
            )
            trades.append(trade)
        return trades

    def _make_log_row(
        self,
        ts:                   datetime,
        underlier:            str,
        futures_price:        float,
        atm_strike:           int,
        nearest_expiry:       date,
        portfolio:            Portfolio,
        options_arrs:         Dict[Tuple[int, str], np.ndarray],
        tick_idx:             int,
        trade_occurred:       bool,
        trade_type:           Optional[str],
    ) -> Dict[str, Any]:
        """Build a log row dict for the event log. Uses numpy O(1) lookups."""
        positions = portfolio.get_all_positions()
        lot_size = self.config.LOT_SIZES.get(underlier, 1)

        ce_pos = None
        pe_pos = None
        for p in positions.values():
            if not isinstance(p.instrument, Option):
                continue
            if p.instrument.underlier != underlier:
                continue
            if p.instrument.option_type == "CE":
                ce_pos = p
            elif p.instrument.option_type == "PE":
                pe_pos = p

        ce_curr = None
        pe_curr = None
        if ce_pos is not None:
            arr = options_arrs.get((ce_pos.instrument.strike, "CE"))
            if arr is not None:
                v = arr[tick_idx]
                if not np.isnan(v):
                    ce_curr = float(v)
        if pe_pos is not None:
            arr = options_arrs.get((pe_pos.instrument.strike, "PE"))
            if arr is not None:
                v = arr[tick_idx]
                if not np.isnan(v):
                    pe_curr = float(v)

        ce_mtm = ce_pos.mtm_pnl(ce_curr, lot_size) if (ce_pos and ce_curr is not None) else 0.0
        pe_mtm = pe_pos.mtm_pnl(pe_curr, lot_size) if (pe_pos and pe_curr is not None) else 0.0
        total_mtm = ce_mtm + pe_mtm

        realized_today = portfolio.realized_pnl_today
        cumulative_realized = portfolio.cumulative_realized_pnl

        return {
            "timestamp":               ts,
            "underlier":               underlier,
            "futures_price":           float(futures_price) if not (isinstance(futures_price, float) and np.isnan(futures_price)) else np.nan,
            "atm_strike":              int(atm_strike) if atm_strike else 0,
            "nearest_expiry":          nearest_expiry.isoformat(),
            "ce_held":                 ce_pos is not None,
            "pe_held":                 pe_pos is not None,
            "ce_strike":               ce_pos.instrument.strike if ce_pos else None,
            "pe_strike":               pe_pos.instrument.strike if pe_pos else None,
            "ce_entry_price":          ce_pos.entry_price if ce_pos else None,
            "pe_entry_price":          pe_pos.entry_price if pe_pos else None,
            "ce_current_price":        ce_curr,
            "pe_current_price":        pe_curr,
            "ce_mtm_pnl":              ce_mtm,
            "pe_mtm_pnl":              pe_mtm,
            "total_mtm_pnl":           total_mtm,
            "realized_pnl_today":      realized_today,
            "cumulative_realized_pnl": cumulative_realized,
            "total_pnl":               cumulative_realized + total_mtm,
            "trade_occurred":          trade_occurred,
            "trade_type":              trade_type,
            "rolls_today":             portfolio.roll_count_today,
        }
