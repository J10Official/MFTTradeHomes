"""
engine/cache.py — Parquet caching + checkpoint management.

Cache key = SHA-256 hash of all parameters that affect simulation output.
Any change to any of these invalidates the cache.

Storage: {CACHE_DIR}/{params_hash}/
    result_event_log.parquet
    result_trade_log.parquet
    result_meta.json
    checkpoint.json
    checkpoint_partial.parquet
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _utcnow_iso() -> str:
    """Timezone-aware UTC timestamp in ISO format (replaces deprecated utcnow)."""
    return datetime.now(timezone.utc).isoformat()

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class CheckpointData:
    """Partial simulation progress saved to disk."""
    params_hash:        str
    completed_dates:    List[str]   # ISO date strings
    partial_log_rows:   List[Dict[str, Any]] = field(default_factory=list)


def _date_to_iso(d: Any) -> str:
    if isinstance(d, date):
        return d.isoformat()
    return str(d)


def _serialize(obj: Any) -> Any:
    """Recursively convert dates/datetimes to ISO strings for JSON."""
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


class CacheManager:
    """Manages caching of backtest results to avoid re-running identical simulations."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    # ─── key generation ────────────────────────────────────────────────────

    def compute_params_hash(
        self,
        strategy,
        underliers: List[str],
        start_date: date,
        end_date: date,
        config,
    ) -> str:
        """SHA-256 hash of all parameters that affect simulation output."""
        params = {
            "strategy_name":     strategy.name,
            "strategy_params":   _serialize(strategy.parameters),
            "underliers":        sorted(underliers),
            "start_date":        str(start_date),
            "end_date":          str(end_date),
            "timestep_seconds":  config.TIMESTEP_SECONDS,
            "brokerage":         config.BROKERAGE_PER_LOT_PER_LEG,
            "stt_rate":          config.STT_RATE,
            "exchange_charges":  config.EXCHANGE_CHARGES_RATE,
            "eod_buffer_min":    config.EOD_FLATTEN_BUFFER_MINUTES,
            "staleness_sec":     config.PRICE_STALENESS_THRESHOLD_SECONDS,
            "tie_break":         config.STRIKE_TIE_BREAK,
            "max_position":      config.MAX_POSITION_LOTS,
        }
        canonical = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    # ─── cache dir ──────────────────────────────────────────────────────────

    def _entry_dir(self, params_hash: str) -> str:
        return os.path.join(self.cache_dir, params_hash)

    # ─── full-result cache ─────────────────────────────────────────────────

    def has_cache(self, params_hash: str) -> bool:
        d = self._entry_dir(params_hash)
        return (os.path.isfile(os.path.join(d, "result_event_log.parquet"))
                and os.path.isfile(os.path.join(d, "result_trade_log.parquet"))
                and os.path.isfile(os.path.join(d, "result_meta.json")))

    def load(self, params_hash: str) -> Dict[str, Any]:
        """Load cached backtest result. Returns dict with keys:
        event_log (DataFrame), trade_log (DataFrame), meta (dict)."""
        d = self._entry_dir(params_hash)
        event_log = pd.read_parquet(os.path.join(d, "result_event_log.parquet"))
        trade_log = pd.read_parquet(os.path.join(d, "result_trade_log.parquet"))
        with open(os.path.join(d, "result_meta.json")) as f:
            meta = json.load(f)
        return {"event_log": event_log, "trade_log": trade_log, "meta": meta}

    def save(
        self,
        params_hash: str,
        strategy_name: str,
        underliers: List[str],
        start_date: date,
        end_date: date,
        event_log: pd.DataFrame,
        trade_log: pd.DataFrame,
    ) -> None:
        d = self._entry_dir(params_hash)
        os.makedirs(d, exist_ok=True)
        event_log.to_parquet(os.path.join(d, "result_event_log.parquet"), index=False)
        trade_log.to_parquet(os.path.join(d, "result_trade_log.parquet"), index=False)
        meta = {
            "strategy_name":   strategy_name,
            "underliers":      list(underliers),
            "start_date":      _date_to_iso(start_date),
            "end_date":        _date_to_iso(end_date),
            "cache_timestamp": _utcnow_iso(),
            "params_hash":     params_hash,
        }
        with open(os.path.join(d, "result_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

    # ─── checkpoint (partial) cache ─────────────────────────────────────────

    def has_checkpoint(self, params_hash: str) -> bool:
        d = self._entry_dir(params_hash)
        return os.path.isfile(os.path.join(d, "checkpoint.json"))

    def load_checkpoint(self, params_hash: str) -> CheckpointData:
        d = self._entry_dir(params_hash)
        with open(os.path.join(d, "checkpoint.json")) as f:
            payload = json.load(f)
        partial_rows: List[Dict[str, Any]] = []
        partial_path = os.path.join(d, "checkpoint_partial.parquet")
        if os.path.isfile(partial_path):
            df = pd.read_parquet(partial_path)
            partial_rows = df.to_dict("records")
        return CheckpointData(
            params_hash=params_hash,
            completed_dates=payload.get("completed_dates", []),
            partial_log_rows=partial_rows,
        )

    def save_checkpoint(
        self,
        params_hash: str,
        completed_dates: List[date],
        partial_log_rows: List[Dict[str, Any]],
        trade_records: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        d = self._entry_dir(params_hash)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "checkpoint.json"), "w") as f:
            json.dump({
                "completed_dates": [_date_to_iso(d_) for d_ in completed_dates],
                "saved_at":        _utcnow_iso(),
            }, f, indent=2)
        if partial_log_rows:
            pd.DataFrame(partial_log_rows).to_parquet(
                os.path.join(d, "checkpoint_partial.parquet"), index=False
            )
        # Also persist trade records for checkpoint resume.
        if trade_records:
            pd.DataFrame(trade_records).to_parquet(
                os.path.join(d, "checkpoint_trades.parquet"), index=False
            )

    def clear_checkpoint(self, params_hash: str) -> None:
        d = self._entry_dir(params_hash)
        for fn in ("checkpoint.json", "checkpoint_partial.parquet"):
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                os.remove(p)

    # ─── bulk clear ─────────────────────────────────────────────────────────

    def clear(self, params_hash: Optional[str] = None) -> None:
        """Clear a specific cache entry, or the entire cache if params_hash is None."""
        if params_hash is None:
            if os.path.isdir(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
            return
        d = self._entry_dir(params_hash)
        if os.path.isdir(d):
            shutil.rmtree(d)
