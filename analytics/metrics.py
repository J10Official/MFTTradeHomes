"""
analytics/metrics.py — Compute all summary statistics from the event log.

Takes a BacktestResult and returns a dict of computed metrics.
No plotting here.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from engine.backtest import BacktestResult


def _safe(x: Any) -> Any:
    """Convert numpy types to native python; NaN → None."""
    if x is None:
        return None
    try:
        if isinstance(x, (np.floating, np.integer)):
            x = x.item()
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return None
    except Exception:
        pass
    return x


def _compute_block(event_log: pd.DataFrame, trade_log: pd.DataFrame, combined: bool = False) -> Dict[str, Any]:
    """Compute metrics for a single slice of the event log.

    Args:
        event_log: rows for ONE underlier (per-underlier block) or all rows (combined block).
        trade_log: trade rows matching the same scope.
        combined:  if True, aggregate across underliers per timestamp before computing
                   cumulative metrics (so the "combined" total_pnl = sum across underliers,
                   not just whichever underlier happens to be last in the dataframe).
    """
    out: Dict[str, Any] = {}

    if event_log.empty:
        return {
            "total_pnl_inr": 0.0,
            "total_realized_pnl_inr": 0.0,
            "n_timesteps": 0,
            "n_trades": 0,
            "n_rolls": 0,
        }

    event_log = event_log.copy()
    event_log["timestamp"] = pd.to_datetime(event_log["timestamp"])
    event_log["trading_date"] = event_log["timestamp"].dt.date

    if combined:
        # Aggregate per timestamp: sum across all underliers.
        # We must be careful: each per-underlier row already has cumulative_realized_pnl
        # accumulated across days FOR THAT UNDERLIER ONLY. So summing cumulative_realized_pnl
        # across underliers at the last timestamp gives the combined total realized.
        # For total_pnl per timestamp (used in daily diff and drawdown), sum total_pnl
        # across underliers at each timestamp.
        per_ts = event_log.groupby("timestamp").agg({
            "total_pnl": "sum",
            "total_mtm_pnl": "sum",
            "cumulative_realized_pnl": "sum",
            "realized_pnl_today": "sum",
            "rolls_today": "sum",
            "trade_occurred": "max",
            "trade_type": "last",
            "futures_price": "first",
            "atm_strike": "first",
            "nearest_expiry": "first",
            "ce_held": "max",
            "pe_held": "max",
            "ce_strike": "first",
            "pe_strike": "first",
            "ce_entry_price": "first",
            "pe_entry_price": "first",
            "ce_current_price": "first",
            "pe_current_price": "first",
            "ce_mtm_pnl": "sum",
            "pe_mtm_pnl": "sum",
        }).reset_index()
        per_ts["trading_date"] = per_ts["timestamp"].dt.date
        event_log = per_ts.sort_values("timestamp").reset_index(drop=True)

    # Final cumulative total PnL = last row's cumulative_realized + last total_mtm
    last_row = event_log.iloc[-1]
    final_realized = float(last_row["cumulative_realized_pnl"])
    # Total PnL at any moment = cumulative_realized + total_mtm
    # At end of run, all positions are flattened so total_mtm = 0 (or near 0).
    final_mtm = float(last_row["total_mtm_pnl"])
    final_total = final_realized + final_mtm

    out["total_pnl_inr"] = _safe(final_total)
    out["total_realized_pnl_inr"] = _safe(final_realized)
    out["final_mtm_pnl_inr"] = _safe(final_mtm)

    # Daily PnL: for each trading_date, take last row's total_pnl minus previous date's last total_pnl.
    daily_last = event_log.groupby("trading_date").tail(1).sort_values("trading_date")
    daily_total_pnl = daily_last["total_pnl"].astype(float).values
    # First day's daily PnL = first day's ending total_pnl.
    # Subsequent days: ending total_pnl - previous ending total_pnl.
    if len(daily_total_pnl) >= 2:
        daily_pnl = np.concatenate([[daily_total_pnl[0]], np.diff(daily_total_pnl)])
    else:
        daily_pnl = daily_total_pnl.copy()
    out["daily_pnl_mean"] = _safe(float(np.mean(daily_pnl))) if len(daily_pnl) else 0.0
    out["daily_pnl_std"]  = _safe(float(np.std(daily_pnl, ddof=1))) if len(daily_pnl) > 1 else 0.0
    out["daily_pnl_min"]  = _safe(float(np.min(daily_pnl))) if len(daily_pnl) else 0.0
    out["daily_pnl_max"]  = _safe(float(np.max(daily_pnl))) if len(daily_pnl) else 0.0

    # Best/worst day
    if len(daily_pnl) > 0:
        best_idx = int(np.argmax(daily_pnl))
        worst_idx = int(np.argmin(daily_pnl))
        out["best_day_date"] = str(daily_last["trading_date"].iloc[best_idx])
        out["best_day_pnl"]  = _safe(float(daily_pnl[best_idx]))
        out["worst_day_date"] = str(daily_last["trading_date"].iloc[worst_idx])
        out["worst_day_pnl"]  = _safe(float(daily_pnl[worst_idx]))
    else:
        out["best_day_date"] = out["best_day_pnl"] = None
        out["worst_day_date"] = out["worst_day_pnl"] = None

    # Max drawdown (peak-to-trough on the cumulative total PnL series).
    cum = event_log["total_pnl"].astype(float).values
    running_peak = np.maximum.accumulate(cum)
    drawdown = cum - running_peak   # negative or zero
    if len(drawdown) > 0:
        mdd = float(np.min(drawdown))
        peak_at_mdd = float(running_peak[np.argmin(drawdown)])
        out["max_drawdown_inr"] = _safe(mdd)
        out["max_drawdown_pct"] = _safe((mdd / peak_at_mdd) if peak_at_mdd != 0 else 0.0)
    else:
        out["max_drawdown_inr"] = 0.0
        out["max_drawdown_pct"] = 0.0

    # Sharpe ratio — computed on daily RETURNS (PnL / capital deployed), not raw ₹.
    # Capital deployed per day = total premium paid on BUY trades that day.
    # This produces a dimensionless ratio comparable to standard industry Sharpe.
    sharpe_computed = False
    if trade_log is not None and not trade_log.empty and "execution_price" in trade_log.columns:
        try:
            tl = trade_log.copy()
            tl["timestamp"] = pd.to_datetime(tl["timestamp"])
            tl["trading_date"] = tl["timestamp"].dt.date
            buys = tl[tl["action"] == "BUY"].copy()
            # Capital per day = sum(execution_price * quantity * lot_size_implied)
            # We don't have lot_size per row, but realized_pnl on buys = 0 and
            # execution_price is premium per unit. Use the absolute sum as proxy.
            # Premium paid = execution_price * quantity (lots). Lot size is absorbed
            # into realized_pnl scaling; here we just need a consistent denominator.
            if "quantity" in tl.columns:
                daily_capital = buys.groupby("trading_date").apply(
                    lambda g: (g["execution_price"] * g["quantity"]).sum()
                )
            else:
                daily_capital = buys.groupby("trading_date")["execution_price"].sum()

            # Align with daily_pnl dates
            daily_last_dates = [str(d) for d in daily_last["trading_date"].values]
            capital_arr = np.array([
                float(daily_capital.get(d, np.nan)) for d in daily_last_dates
            ])
            # Only use days where capital > 0 (avoid div/0)
            valid_mask = (capital_arr > 0) & np.isfinite(capital_arr)
            if valid_mask.sum() > 1:
                daily_returns = np.where(valid_mask, daily_pnl / capital_arr, np.nan)
                daily_returns = daily_returns[~np.isnan(daily_returns)]
                if len(daily_returns) > 1 and np.std(daily_returns, ddof=1) > 0:
                    out["sharpe_ratio"] = _safe(
                        float(np.mean(daily_returns) / np.std(daily_returns, ddof=1) * np.sqrt(252))
                    )
                    out["sharpe_note"] = "return-based (PnL / premium deployed)"
                    sharpe_computed = True
        except Exception:
            pass  # fallback below

    if not sharpe_computed:
        # Fallback: ₹-denominated Sharpe. Clearly labelled so it's not
        # confused with a standard industry Sharpe.
        if len(daily_pnl) > 1 and np.std(daily_pnl, ddof=1) > 0:
            out["sharpe_ratio"] = _safe(
                float(np.mean(daily_pnl) / np.std(daily_pnl, ddof=1) * np.sqrt(252))
            )
            out["sharpe_note"] = "₹-denominated (not return-based; non-comparable)"
        else:
            out["sharpe_ratio"] = 0.0
            out["sharpe_note"] = "insufficient data"


    # Trading activity
    n_trades = int(len(trade_log)) if trade_log is not None else 0
    n_rolls  = int((daily_last["rolls_today"].astype(int).sum()) if not daily_last.empty else 0)
    out["total_trades"] = n_trades
    out["total_rolls"]  = n_rolls
    n_days = len(daily_last)
    out["n_trading_days"] = n_days
    out["avg_rolls_per_day"] = _safe(float(n_rolls / n_days)) if n_days > 0 else 0.0
    out["max_rolls_single_day"] = _safe(int(daily_last["rolls_today"].astype(int).max())) if not daily_last.empty else 0

    # Average holding time between rolls (seconds)
    if trade_log is not None and not trade_log.empty and "trade_type" in trade_log.columns:
        roll_opens = trade_log[trade_log["trade_type"].isin(["ROLL_OPEN", "OPEN"])].copy()
        roll_opens["timestamp"] = pd.to_datetime(roll_opens["timestamp"])
        if len(roll_opens) > 1:
            diffs = roll_opens["timestamp"].sort_values().diff().dt.total_seconds().dropna()
            out["avg_holding_time_seconds"] = _safe(float(diffs.mean())) if len(diffs) else None
        else:
            out["avg_holding_time_seconds"] = None
    else:
        out["avg_holding_time_seconds"] = None

    # Win/loss
    if len(daily_pnl) > 0:
        out["daily_win_rate"] = _safe(float(np.mean(daily_pnl > 0)))
    else:
        out["daily_win_rate"] = 0.0

    # Roll-to-roll PnL win rate
    if trade_log is not None and not trade_log.empty and "realized_pnl" in trade_log.columns:
        closes = trade_log[trade_log["trade_type"] == "ROLL_CLOSE"]
        if len(closes) > 0:
            out["roll_win_rate"] = _safe(float((closes["realized_pnl"].astype(float) > 0).mean()))
        else:
            out["roll_win_rate"] = 0.0
    else:
        out["roll_win_rate"] = 0.0

    # Transaction cost
    if trade_log is not None and not trade_log.empty and "transaction_cost" in trade_log.columns:
        out["total_transaction_cost_inr"] = _safe(float(trade_log["transaction_cost"].astype(float).sum()))
    else:
        out["total_transaction_cost_inr"] = 0.0
    out["pnl_before_costs_inr"] = _safe(final_total + out.get("total_transaction_cost_inr", 0.0))

    return out


def compute_all_metrics(result: BacktestResult) -> Dict[str, Any]:
    """
    Compute all summary statistics from the event log.

    Per-underlier AND combined.
    """
    event_log = result.event_log
    trade_log = result.trade_log

    metrics: Dict[str, Any] = {
        "strategy_name":   result.strategy_name,
        "underliers":      list(result.underliers),
        "start_date":      str(result.start_date),
        "end_date":        str(result.end_date),
        "cached":          result.cached,
        "params_hash":     result.params_hash,
    }

    # Per-underlier blocks
    per_underlier: Dict[str, Any] = {}
    for u in result.underliers:
        el_u = event_log[event_log["underlier"] == u] if not event_log.empty else event_log
        tl_u = trade_log[trade_log["underlier"] == u] if (trade_log is not None and not trade_log.empty) else trade_log
        per_underlier[u] = _compute_block(el_u, tl_u)
    metrics["per_underlier"] = per_underlier

    # Combined: aggregate across underliers per timestamp before computing.
    metrics["combined"] = _compute_block(event_log, trade_log, combined=True)

    return metrics
