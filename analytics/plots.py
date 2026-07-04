"""
analytics/plots.py — All chart generation from the event log.

All charts saved to {output_dir}/. All functions take output_dir: str and
return the saved filepath.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine.backtest import BacktestResult


logger = logging.getLogger(__name__)

# ─── font setup (handles Latin + symbol fallback) ───────────────────────────
try:
    for fpath in [
        "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.isfile(fpath):
            fm.fontManager.addfont(fpath)
    plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

# Color palette: one per underlier + combined.
UNDERLIER_COLORS = {
    "NIFTY":     "#1f77b4",
    "BANKNIFTY": "#ff7f0e",
    "FINNIFTY":  "#2ca02c",
    "COMBINED":  "#9467bd",
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _trading_date_column(event_log: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(event_log["timestamp"])
    return ts.dt.date


def plot_cumulative_pnl(result: BacktestResult, output_dir: str) -> str:
    """
    Line chart. One line per underlier + one combined line.
    X axis: timestamp. Y axis: cumulative total PnL in ₹.
    """
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    if not result.event_log.empty:
        df = result.event_log.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for u in result.underliers:
            sub = df[df["underlier"] == u].sort_values("timestamp")
            if sub.empty:
                continue
            # total_pnl column already includes cumulative_realized + total_mtm,
            # so it's effectively the running equity curve per underlier.
            ax.plot(sub["timestamp"], sub["total_pnl"].astype(float),
                    label=u, color=UNDERLIER_COLORS.get(u), linewidth=1.5, alpha=0.85)
        # Combined: sum total_pnl across underliers at each timestamp.
        comb_grouped = df.groupby("timestamp")["total_pnl"].apply(lambda s: s.astype(float).sum()).sort_index()
        ax.plot(comb_grouped.index, comb_grouped.values,
                label="COMBINED", color=UNDERLIER_COLORS["COMBINED"],
                linewidth=2.0, alpha=0.95)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Cumulative Total PnL (₹)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative PnL (₹)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    path = os.path.join(output_dir, "cumulative_pnl.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_daily_pnl_bars(result: BacktestResult, output_dir: str) -> str:
    """
    Bar chart. One bar per trading day per underlier, side by side.
    Bars above 0 = green. Bars below 0 = red.
    """
    _ensure_dir(output_dir)
    if result.event_log.empty:
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        path = os.path.join(output_dir, "daily_pnl_bars.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    df = result.event_log.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["trading_date"] = df["timestamp"].dt.date

    # For each (underlier, trading_date), take the last row's total_pnl.
    last_rows = df.groupby(["underlier", "trading_date"]).tail(1).sort_values(["underlier", "trading_date"])
    # Compute daily PnL per underlier.
    daily_pnl_records = []
    for u in result.underliers:
        sub = last_rows[last_rows["underlier"] == u].copy()
        sub["daily_pnl"] = sub["total_pnl"].astype(float).diff().fillna(sub["total_pnl"].astype(float))
        for _, r in sub.iterrows():
            daily_pnl_records.append({"trading_date": r["trading_date"], "underlier": u, "daily_pnl": r["daily_pnl"]})

    daily_df = pd.DataFrame(daily_pnl_records)
    pivot = daily_df.pivot(index="trading_date", columns="underlier", values="daily_pnl").fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    width = 0.35
    x = np.arange(len(pivot.index))
    for i, u in enumerate(result.underliers):
        if u not in pivot.columns:
            continue
        vals = pivot[u].values
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
        offset = (i - len(result.underliers)/2 + 0.5) * width
        ax.bar(x + offset, vals, width=width, color=colors, label=u, edgecolor="black", linewidth=0.4)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in pivot.index], rotation=45, ha="right", fontsize=9)
    ax.set_title("Daily PnL by Underlier (₹)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Trading Date")
    ax.set_ylabel("Daily PnL (₹)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    path = os.path.join(output_dir, "daily_pnl_bars.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_intraday_sample(result: BacktestResult, sample_date: date, output_dir: str) -> str:
    """
    Three-panel intraday chart for one date:
      Panel 1: Futures price + ATM strike (step line).
      Panel 2: MTM PnL second-by-second.
      Panel 3: Vertical lines at every roll event.
    Shows ONE underlier (the first one) to keep the chart readable.
    """
    _ensure_dir(output_dir)
    if result.event_log.empty:
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        path = os.path.join(output_dir, f"intraday_sample_{sample_date}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    df = result.event_log.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    sample_dt = pd.Timestamp(sample_date)
    sample = df[df["timestamp"].dt.date == sample_dt.date()].copy()
    if sample.empty:
        # try string match
        sample = df[df["timestamp"].dt.strftime("%Y-%m-%d") == str(sample_date)].copy()
    if sample.empty:
        logger.warning("No data for sample date %s", sample_date)
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax.text(0.5, 0.5, f"No data for {sample_date}", ha="center", va="center")
        path = os.path.join(output_dir, f"intraday_sample_{sample_date}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    underlier = sample["underlier"].iloc[0]
    sample = sample.sort_values("timestamp")

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)

    # Panel 1: futures + ATM strike
    ax1 = axes[0]
    ax1.plot(sample["timestamp"], sample["futures_price"].astype(float),
             label="Futures price", color="#1f77b4", linewidth=1.2)
    ax1.step(sample["timestamp"], sample["atm_strike"].astype(int),
             label="ATM strike", color="#ff7f0e", where="post", linewidth=1.0, alpha=0.8)
    ax1.set_ylabel("Price / Strike")
    ax1.set_title(f"Intraday {underlier} on {sample_date}", fontsize=12, fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: MTM PnL
    ax2 = axes[1]
    ax2.plot(sample["timestamp"], sample["total_mtm_pnl"].astype(float),
             label="Total MTM PnL", color="#2ca02c", linewidth=1.0)
    ax2.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax2.set_ylabel("MTM PnL (₹)")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: roll events (vertical lines)
    ax3 = axes[2]
    rolls = sample[sample["trade_type"] == "ROLL_CLOSE"]
    ax3.plot(sample["timestamp"], sample["cumulative_realized_pnl"].astype(float),
             label="Cumulative realized PnL", color="#9467bd", linewidth=1.0)
    for ts in rolls["timestamp"]:
        ax3.axvline(ts, color="#d62728", linewidth=0.5, alpha=0.7)
    ax3.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax3.set_ylabel("Realized PnL (₹)")
    ax3.set_xlabel("Time")
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)
    # Legend entry for roll markers
    from matplotlib.lines import Line2D
    roll_line = Line2D([0], [0], color="#d62728", linewidth=2, label="Roll events")
    handles, labels = ax3.get_legend_handles_labels()
    handles.append(roll_line)
    ax3.legend(handles=handles, loc="best", fontsize=9)

    for label in ax3.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    path = os.path.join(output_dir, f"intraday_sample_{sample_date}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_roll_frequency(result: BacktestResult, output_dir: str) -> str:
    """
    Histogram: x = rolls per day, y = count of days with that many rolls.
    One per underlier + combined.
    """
    _ensure_dir(output_dir)
    if result.event_log.empty:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        path = os.path.join(output_dir, "roll_frequency.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    df = result.event_log.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["trading_date"] = df["timestamp"].dt.date
    daily_rolls = df.groupby(["underlier", "trading_date"])["rolls_today"].max().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    width = 0.3
    all_rolls = sorted(daily_rolls["rolls_today"].unique())
    if not all_rolls:
        all_rolls = [0]
    x = np.arange(len(all_rolls))
    for i, u in enumerate(result.underliers):
        sub = daily_rolls[daily_rolls["underlier"] == u]
        counts = [(sub["rolls_today"] == r).sum() for r in all_rolls]
        offset = (i - len(result.underliers)/2 + 0.5) * width
        ax.bar(x + offset, counts, width=width, label=u,
               color=UNDERLIER_COLORS.get(u, None), edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in all_rolls])
    ax.set_xlabel("Rolls per day")
    ax.set_ylabel("Number of days")
    ax.set_title("Straddle Roll Frequency Distribution", fontsize=13, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    path = os.path.join(output_dir, "roll_frequency.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_premium_decay(result: BacktestResult, output_dir: str) -> str:
    """
    Average ATM option premium by time of day (averaged across all days).
    Separate CE and PE lines.
    """
    _ensure_dir(output_dir)
    if result.event_log.empty:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        path = os.path.join(output_dir, "premium_decay.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    df = result.event_log.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["time_of_day"] = df["timestamp"].dt.strftime("%H:%M")

    # For each underlier, average CE and PE current price by time-of-day.
    fig, axes = plt.subplots(len(result.underliers), 1,
                              figsize=(12, 4 * len(result.underliers)),
                              sharex=True, constrained_layout=True)
    if len(result.underliers) == 1:
        axes = [axes]

    for ax, u in zip(axes, result.underliers):
        sub = df[df["underlier"] == u].copy()
        # Only rows where we hold the position (so ce_current_price is meaningful).
        sub_ce = sub[sub["ce_held"]].copy()
        sub_pe = sub[sub["pe_held"]].copy()
        if not sub_ce.empty:
            avg_ce = sub_ce.groupby("time_of_day")["ce_current_price"].mean()
            ax.plot(range(len(avg_ce)), avg_ce.values, label="CE premium", color="#1f77b4", linewidth=1.2)
        if not sub_pe.empty:
            avg_pe = sub_pe.groupby("time_of_day")["pe_current_price"].mean()
            ax.plot(range(len(avg_pe)), avg_pe.values, label="PE premium", color="#ff7f0e", linewidth=1.2)
        # Show ~10 x-ticks
        if not sub_ce.empty:
            n = len(avg_ce)
            tick_positions = list(range(0, n, max(1, n // 10)))
            tick_labels = [avg_ce.index[i] for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{u} — Average ATM Premium by Time of Day", fontsize=11, fontweight="bold")
        ax.set_ylabel("Premium (₹)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time of day")
    path = os.path.join(output_dir, "premium_decay.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_drawdown_curve(result: BacktestResult, output_dir: str) -> str:
    """Running drawdown from equity peak over time."""
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    if not result.event_log.empty:
        df = result.event_log.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Combined across underliers: sum total_pnl per timestamp
        comb = df.groupby("timestamp")["total_pnl"].apply(lambda s: s.astype(float).sum()).sort_index()
        running_peak = comb.cummax()
        drawdown = comb - running_peak
        ax.plot(drawdown.index, drawdown.values, color="#d62728", linewidth=1.0)
        ax.fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title("Drawdown from Equity Peak (₹)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown (₹)")
    ax.grid(True, alpha=0.3)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    path = os.path.join(output_dir, "drawdown_curve.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_transaction_cost_sensitivity(
    results_by_cost: Dict[float, BacktestResult],
    output_dir: str,
) -> str:
    """
    Bar chart: total PnL for each transaction cost level tested.
    """
    _ensure_dir(output_dir)
    costs = sorted(results_by_cost.keys())
    pnls = []
    for c in costs:
        r = results_by_cost[c]
        if r.event_log.empty:
            pnls.append(0.0)
        else:
            last = r.event_log.iloc[-1]
            pnls.append(float(last["cumulative_realized_pnl"]) + float(last["total_mtm_pnl"]))

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = np.arange(len(costs))
    colors = ["#2ca02c" if p >= 0 else "#d62728" for p in pnls]
    ax.bar(x, pnls, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"₹{c}/lot" for c in costs])
    ax.set_xlabel("Brokerage per lot per leg")
    ax.set_ylabel("Total PnL (₹)")
    ax.set_title("Transaction Cost Sensitivity", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    path = os.path.join(output_dir, "transaction_cost_sensitivity.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def generate_all_plots(
    result: BacktestResult,
    output_dir: str,
    sample_date: date,
) -> Dict[str, str]:
    """Calls all plot functions and returns dict of chart_name -> filepath."""
    _ensure_dir(output_dir)
    return {
        "cumulative_pnl":      plot_cumulative_pnl(result, output_dir),
        "daily_pnl_bars":      plot_daily_pnl_bars(result, output_dir),
        "intraday_sample":     plot_intraday_sample(result, sample_date, output_dir),
        "roll_frequency":      plot_roll_frequency(result, output_dir),
        "premium_decay":       plot_premium_decay(result, output_dir),
        "drawdown_curve":      plot_drawdown_curve(result, output_dir),
    }
