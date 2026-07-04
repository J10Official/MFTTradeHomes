#!/usr/bin/env python3
"""
run_backtest.py — MFT TradeHomes Backtesting Engine CLI.

Usage:
    python run_backtest.py run --start 2022-11-01 --end 2022-11-30
    python run_backtest.py tune --search-type random --n-trials 20
    python run_backtest.py clear-cache --all
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import signal
import sys
from datetime import date, datetime
from typing import Any, Dict, List

import click

# Make sure local packages are importable when run from any directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from engine.backtest import BacktestEngine
from engine.cache import CacheManager
from analytics.metrics import compute_all_metrics
from analytics.plots import generate_all_plots


logger = logging.getLogger(__name__)


# ─── signal handling ────────────────────────────────────────────────────────

_shutdown_requested = {"value": False}


def handle_sigterm(signum, frame):
    """
    On SIGTERM: set the shutdown flag and raise KeyboardInterrupt to break
    out of any running Python loop. The engine checks the flag and will
    checkpoint before exiting. If we're in a worker subprocess we re-raise
    to let the parent handle it.
    """
    print("[SIGTERM] received — requesting graceful shutdown...", file=sys.stderr)
    _shutdown_requested["value"] = True
    raise KeyboardInterrupt("SIGTERM received")


# Only install the handler in the parent process. Worker subprocesses
# (launched by ProcessPoolExecutor) inherit a fresh default disposition.
if os.environ.get("MFT_WORKER") != "1":
    signal.signal(signal.SIGTERM, handle_sigterm)


# ─── helpers ────────────────────────────────────────────────────────────────


def _setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _load_strategy(name: str):
    """Load a strategy class by name from config.STRATEGY_REGISTRY."""
    fq_path = cfg.STRATEGY_REGISTRY.get(name)
    if not fq_path:
        raise click.ClickException(f"Unknown strategy: {name!r}. Available: {list(cfg.STRATEGY_REGISTRY)}")
    module_path, class_name = fq_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except Exception as e:
        raise click.ClickException(f"Failed to load strategy {name!r}: {e}")
    return cls


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _trial_result_to_dict(r):
    return {
        "params": r.params,
        "train_pnl": r.train_pnl,
        "test_pnl": r.test_pnl,
        "train_sharpe": r.train_sharpe,
        "test_sharpe": r.test_sharpe,
        "overfitting_flag": r.overfitting_flag,
        "n_trades": r.n_trades,
        "runtime_seconds": r.runtime_seconds,
    }


# ─── CLI ────────────────────────────────────────────────────────────────────


@click.group()
def cli():
    """MFT TradeHomes Backtesting Engine — production-grade NSE options simulator."""
    pass


@cli.command()
@click.option("--underliers", "-u", envvar="MFT_UNDERLIERS",
              default=",".join(cfg.UNDERLIERS), show_default=True,
              help="Comma-separated underliers to simulate.")
@click.option("--start", "-s", envvar="MFT_START_DATE",
              default=cfg.DEFAULT_START_DATE.isoformat(), show_default=True,
              help="Backtest start date YYYY-MM-DD.")
@click.option("--end", "-e", envvar="MFT_END_DATE",
              default=cfg.DEFAULT_END_DATE.isoformat(), show_default=True,
              help="Backtest end date YYYY-MM-DD.")
@click.option("--strategy", envvar="MFT_STRATEGY",
              default="atm_straddle", show_default=True,
              help="Strategy name. Must match a class in strategies/.")
@click.option("--timestep", "-t", envvar="MFT_TIMESTEP",
              default=cfg.TIMESTEP_SECONDS, show_default=True, type=int,
              help="Simulation timestep in seconds.")
@click.option("--data-root", envvar="MFT_DATA_ROOT",
              default=cfg.DATA_ROOT, show_default=True,
              help="Path to root of NSE data folder.")
@click.option("--output-dir", "-o", envvar="MFT_OUTPUT_DIR",
              default=cfg.DEFAULT_OUTPUT_DIR, show_default=True,
              help="Directory for charts and result files.")
@click.option("--output-format", envvar="MFT_OUTPUT_FORMAT",
              default="human", type=click.Choice(["human", "json"]),
              help="Output format. Use 'json' for machine-readable / piping.")
@click.option("--no-cache", is_flag=True, default=False,
              help="Force re-run, ignore any cached results.")
@click.option("--log-level", envvar="MFT_LOG_LEVEL",
              default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              help="Logging verbosity.")
@click.option("--no-plots", is_flag=True, default=False,
              help="Skip generating plot PNGs (faster).")
def run(underliers, start, end, strategy, timestep, data_root,
        output_dir, output_format, no_cache, log_level, no_plots):
    """Run a backtest for the specified configuration."""
    _setup_logging(log_level)

    # Apply runtime config overrides.
    cfg.DATA_ROOT = data_root
    cfg.TIMESTEP_SECONDS = timestep
    cfg.LOG_LEVEL = log_level
    cfg.DEFAULT_OUTPUT_DIR = output_dir

    underlier_list = [u.strip() for u in underliers.split(",") if u.strip()]
    start_d = _parse_date(start)
    end_d = _parse_date(end)

    strategy_cls = _load_strategy(strategy)
    strategy_inst = strategy_cls()

    cache = CacheManager(cfg.CACHE_DIR)
    engine = BacktestEngine(
        strategy=strategy_inst, config=cfg, cache=cache,
        underliers=underlier_list, start_date=start_d, end_date=end_d,
    )

    try:
        result = engine.run(use_cache=not no_cache)
    except FileNotFoundError as e:
        click.echo(f"Data error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("Backtest failed")
        click.echo(f"Runtime error: {e}", err=True)
        sys.exit(2)

    metrics = compute_all_metrics(result)

    # Always dump metrics JSON.
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Dump event & trade logs as parquet for reproducibility.
    result.event_log.to_parquet(os.path.join(output_dir, "event_log.parquet"), index=False)
    if not result.trade_log.empty:
        result.trade_log.to_parquet(os.path.join(output_dir, "trade_log.parquet"), index=False)
    else:
        # write empty parquet for shape consistency
        pd = __import__("pandas")
        pd.DataFrame().to_parquet(os.path.join(output_dir, "trade_log.parquet"))

    # Plots
    plot_paths: Dict[str, str] = {}
    if not no_plots:
        sample_d = cfg.SAMPLE_INTRADAY_DATE
        if not (start_d <= sample_d <= end_d):
            # fall back to start date if sample date out of range
            sample_d = start_d
        try:
            plot_paths = generate_all_plots(result, output_dir, sample_d)
        except Exception as e:
            logger.warning("Plot generation failed: %s", e)

    # Output
    if output_format == "json":
        out = {
            "metrics": metrics,
            "params_hash": result.params_hash,
            "cached": result.cached,
            "plots": plot_paths,
        }
        click.echo(json.dumps(out, indent=2, default=str))
    else:
        click.echo("\n" + "=" * 70)
        click.echo("  MFT TRADEHOMES BACKTEST — RESULTS SUMMARY")
        click.echo("=" * 70)
        click.echo(f"  Strategy:   {result.strategy_name}")
        click.echo(f"  Underliers: {', '.join(result.underliers)}")
        click.echo(f"  Date range: {result.start_date} → {result.end_date}")
        click.echo(f"  Cached:     {result.cached}")
        click.echo(f"  Timesteps:  {len(result.event_log)}")
        click.echo(f"  Trades:     {len(result.trade_log)}")
        click.echo("-" * 70)
        for u in result.underliers:
            m = metrics.get("per_underlier", {}).get(u, {})
            click.echo(f"  [{u}]")
            click.echo(f"     Total PnL:          ₹{m.get('total_pnl_inr', 0):>14,.2f}")
            click.echo(f"     Realized PnL:       ₹{m.get('total_realized_pnl_inr', 0):>14,.2f}")
            click.echo(f"     Total trades:       {m.get('total_trades', 0):>14d}")
            click.echo(f"     Total rolls:        {m.get('total_rolls', 0):>14d}")
            click.echo(f"     Avg rolls/day:      {m.get('avg_rolls_per_day', 0):>14.2f}")
            click.echo(f"     Max DD:             ₹{m.get('max_drawdown_inr', 0):>14,.2f}")
            click.echo(f"     Sharpe (annualized):{m.get('sharpe_ratio', 0):>14.3f}")
            click.echo(f"     Daily win rate:     {m.get('daily_win_rate', 0) * 100:>13.1f}%")
        m = metrics.get("combined", {})
        click.echo("-" * 70)
        click.echo("  [COMBINED]")
        click.echo(f"     Total PnL:          ₹{m.get('total_pnl_inr', 0):>14,.2f}")
        click.echo(f"     Realized PnL:       ₹{m.get('total_realized_pnl_inr', 0):>14,.2f}")
        click.echo(f"     Total trades:       {m.get('total_trades', 0):>14d}")
        click.echo(f"     Total rolls:        {m.get('total_rolls', 0):>14d}")
        click.echo(f"     Max DD:             ₹{m.get('max_drawdown_inr', 0):>14,.2f}")
        click.echo(f"     Sharpe (annualized):{m.get('sharpe_ratio', 0):>14.3f}")
        click.echo("=" * 70)
        click.echo(f"  Metrics JSON:  {metrics_path}")
        if plot_paths:
            click.echo(f"  Plots:         {output_dir}/")
            for name, p in plot_paths.items():
                click.echo(f"     - {name}: {os.path.basename(p)}")
        click.echo("=" * 70 + "\n")

    sys.exit(0)


@cli.command()
@click.option("--underliers", "-u", envvar="MFT_UNDERLIERS",
              default=",".join(cfg.UNDERLIERS))
@click.option("--start", "-s", envvar="MFT_START_DATE",
              default=cfg.DEFAULT_START_DATE.isoformat())
@click.option("--end", "-e", envvar="MFT_END_DATE",
              default=cfg.DEFAULT_END_DATE.isoformat())
@click.option("--strategy", default="atm_straddle")
@click.option("--search-type", default="random",
              type=click.Choice(["grid", "random"]))
@click.option("--n-trials", default=cfg.TUNER_DEFAULT_N_TRIALS, type=int,
              help="Number of trials (random search only).")
@click.option("--n-workers", default=cfg.TUNER_N_WORKERS, type=int,
              help="Parallel worker processes.")
@click.option("--output-dir", "-o", default=cfg.DEFAULT_OUTPUT_DIR)
@click.option("--output-format", default="human",
              type=click.Choice(["human", "json"]))
@click.option("--data-root", envvar="MFT_DATA_ROOT", default=cfg.DATA_ROOT)
@click.option("--log-level", envvar="MFT_LOG_LEVEL", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def tune(underliers, start, end, strategy, search_type, n_trials,
         n_workers, output_dir, output_format, data_root, log_level):
    """Run hyperparameter search for a strategy."""
    _setup_logging(log_level)
    cfg.DATA_ROOT = data_root

    from tuner.base import ParameterSpace
    from tuner.grid_search import GridSearchTuner
    from tuner.random_search import RandomSearchTuner

    underlier_list = [u.strip() for u in underliers.split(",") if u.strip()]
    start_d = _parse_date(start)
    end_d = _parse_date(end)

    # Default search space (sensitivity sweep).
    ps = ParameterSpace(
        discrete={
            "strike_tie_break":             ["up", "down"],
            "eod_flatten_buffer_minutes":   [0, 1, 5],
        },
        continuous={
            "brokerage_per_lot_per_leg":    (0.0, 50.0, 10.0),
            "timestep_seconds":             (1, 5, 1),
        },
    )

    if search_type == "grid":
        tuner = GridSearchTuner(
            ps, underlier_list, start_d, end_d,
            train_fraction=cfg.TUNER_TRAIN_FRACTION,
            n_workers=n_workers,
            overfit_threshold=cfg.TUNER_OVERFITTING_THRESHOLD,
        )
    else:
        tuner = RandomSearchTuner(
            ps, underlier_list, start_d, end_d,
            train_fraction=cfg.TUNER_TRAIN_FRACTION,
            n_trials=n_trials, n_workers=n_workers,
            seed=cfg.TUNER_RANDOM_SEED,
            overfit_threshold=cfg.TUNER_OVERFITTING_THRESHOLD,
        )

    result = tuner.run()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "tuner_results.json")
    payload = {
        "search_type":         result.search_type,
        "n_trials_completed":  result.n_trials_completed,
        "best_params_by_train": result.best_params_by_train,
        "best_params_by_test":  result.best_params_by_test,
        "recommended_params":   result.recommended_params,
        "train_date_range":     [str(d) for d in result.train_date_range],
        "test_date_range":      [str(d) for d in result.test_date_range],
        "best_train_pnl":       result.best_train_pnl,
        "best_test_pnl":        result.best_test_pnl,
        "recommended_pnl":      result.recommended_pnl,
        "trial_results":        [_trial_result_to_dict(r) for r in result.trial_results],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, default=str))
    else:
        click.echo("\n" + "=" * 70)
        click.echo("  MFT TRADEHOMES BACKTEST — TUNER RESULTS")
        click.echo("=" * 70)
        click.echo(f"  Search type:      {result.search_type}")
        click.echo(f"  Trials completed: {result.n_trials_completed}")
        click.echo(f"  Train range:      {result.train_date_range[0]} → {result.train_date_range[1]}")
        click.echo(f"  Test range:       {result.test_date_range[0]} → {result.test_date_range[1]}")
        click.echo("-" * 70)
        click.echo(f"  Best by TRAIN PnL: ₹{result.best_train_pnl:>12,.2f}  params={result.best_params_by_train}")
        click.echo(f"  Best by TEST PnL:  ₹{result.best_test_pnl:>12,.2f}  params={result.best_params_by_test}")
        click.echo(f"  Recommended:       ₹{result.recommended_pnl:>12,.2f}  params={result.recommended_params}")
        click.echo("=" * 70)
        click.echo(f"  Results JSON: {out_path}")
        click.echo("=" * 70 + "\n")

    sys.exit(0)


@cli.command(name="clear-cache")
@click.option("--params-hash", required=False, help="Clear specific cache entry.")
@click.option("--all", "clear_all", is_flag=True, help="Clear entire cache directory.")
def clear_cache(params_hash, clear_all):
    """Clear cached backtest results."""
    cache = CacheManager(cfg.CACHE_DIR)
    if clear_all:
        cache.clear()
        click.echo(f"Cleared entire cache at {cfg.CACHE_DIR}")
    elif params_hash:
        cache.clear(params_hash)
        click.echo(f"Cleared cache entry {params_hash}")
    else:
        click.echo("Specify --all or --params-hash HASH. Aborting.", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
