"""
Consolidated Data Governance Utilities.
Combines data contracts, quality checks, and lineage tracking.
"""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from pandas.api import types as pdt

# --- Data Contracts ---
def max_ohlcv_cache_bar_date(cache_dir: Path) -> pd.Timestamp | None:
    if not cache_dir.is_dir(): return None
    mx: pd.Timestamp | None = None
    for p in sorted(cache_dir.glob("*.parquet")):
        try: df = pd.read_parquet(p)
        except Exception: continue
        if df.empty: continue
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx, errors="coerce")
            if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
        try: m = pd.Timestamp(idx.max())
        except Exception: continue
        if mx is None or m > mx: mx = m
    return mx

def required_latest_cache_date() -> pd.Timestamp:
    return (pd.Timestamp.now().normalize() - pd.tseries.offsets.BDay(1)).normalize()

def cache_covers_session(mx: pd.Timestamp | None, need: pd.Timestamp) -> bool:
    return mx.normalize() >= need.normalize() if mx is not None else False

def ohlcv_cache_dir(cfg: dict[str, Any], root: Path) -> Path:
    bt = cfg.get("backtest", cfg)
    return (root / bt.get("cache_dir", "data/cache/ohlcv")).resolve()

# --- Data Quality Checks ---
class DataQualityChecker:
    REQUIRED_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
    REQUIRED_DTYPES = {"date": "datetime", "ticker": "string", "open": "numeric", "high": "numeric", "low": "numeric", "close": "numeric", "volume": "numeric"}

    def __init__(self, data: pd.DataFrame, null_threshold: float = 0.05, drift_threshold: float = 2.0) -> None:
        self.data, self.null_threshold, self.drift_threshold = data, null_threshold, drift_threshold

    def validate_schema(self) -> dict[str, Any]:
        res = {"status": "PASS", "column_check": "PASS", "dtype_check": "PASS", "missing_columns": [], "dtype_mismatches": []}
        if self.data.empty: return {**res, "status": "FAIL", "column_check": "FAIL", "reason": "empty_dataframe"}
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.data.columns]
        if missing: return {**res, "status": "FAIL", "column_check": "FAIL", "missing_columns": missing}
        for col, expected in self.REQUIRED_DTYPES.items():
            if col in self.data.columns and not self._matches_expected_dtype(self.data[col], expected):
                res["dtype_check"] = "FAIL"; res["status"] = "FAIL"
                res["dtype_mismatches"].append({"column": col, "expected": expected, "got": str(self.data[col].dtype)})
        return res

    @staticmethod
    def _matches_expected_dtype(series: pd.Series, expected: str) -> bool:
        if expected == "datetime": return bool(pdt.is_datetime64_any_dtype(series))
        if expected == "string": return bool(pdt.is_string_dtype(series) or pdt.is_object_dtype(series))
        if expected == "numeric": return bool(pdt.is_numeric_dtype(series))
        return False

    def detect_nulls(self) -> dict[str, Any]:
        res = {"status": "PASS", "total_nulls": 0, "null_fraction": 0.0, "null_by_column": {}}
        if self.data.empty: return res
        num_cols = ["open", "high", "low", "close", "volume"]
        total_cells, total_nulls = 0, 0
        for col in num_cols:
            if col in self.data.columns:
                c_nulls, c_total = int(self.data[col].isna().sum()), len(self.data)
                total_nulls += c_nulls; total_cells += c_total
                if c_nulls > 0: res["null_by_column"][col] = {"count": c_nulls, "fraction": float(c_nulls/c_total) if c_total > 0 else 0.0}
        res["total_nulls"], res["null_fraction"] = total_nulls, float(total_nulls)/float(total_cells) if total_cells > 0 else 0.0
        if res["null_fraction"] > self.null_threshold: res["status"] = "FAIL"
        return res

    def detect_drift(self, baseline: pd.DataFrame | None = None) -> dict[str, Any]:
        res = {"status": "PASS", "reason": "no_drift"}
        if baseline is None or baseline.empty: return {**res, "reason": "no_baseline" if baseline is None else "baseline_empty"}
        if self.data.empty: return {**res, "reason": "current_empty"}
        try:
            if "high" not in self.data.columns or "low" not in self.data.columns: return {**res, "reason": "missing_price_columns"}
            b_spr, c_spr = (baseline["high"]-baseline["low"]).mean(), (self.data["high"]-self.data["low"]).mean()
            if b_spr <= 0 or np.isnan(b_spr): return {**res, "reason": "baseline_no_variance"}
            z = abs(c_spr/b_spr - 1.0)
            res.update({"baseline_spread": float(b_spr), "current_spread": float(c_spr), "spread_ratio": float(c_spr/b_spr), "z_score": float(z)})
            if z > self.drift_threshold: res["status"], res["reason"] = "FAIL", "high_drift"
        except Exception as exc: res["status"], res["reason"] = "FAIL", f"drift_calc_error: {str(exc)}"
        return res

# --- Data Quality Lineage ---
class LineageTracker:
    def __init__(self, ticker: str) -> None:
        self.ticker, self.run_at_utc, self.checks, self.data_sources, self.upstream_slo, self.anomalies = ticker, datetime.now(timezone.utc).isoformat(), [], [], None, []

    def add_check(self, name: str, status: str, details: dict[str, Any] | None = None) -> None:
        check = {"name": name, "status": status}
        if details:
            check["details"] = details
            if "anomalies" in details and isinstance(details["anomalies"], list): self.anomalies.extend(details["anomalies"])
        self.checks.append(check)

    def add_data_source(self, path: str, rows: int | None = None, format: str = "parquet") -> None:
        source = {"path": path, "format": format}
        if rows is not None: source["rows"] = rows
        self.data_sources.append(source)

    def set_upstream_slo(self, status: str, failures: list[str] | None = None) -> None:
        self.upstream_slo = {"status": status, "failures": failures or []}

    def build_lineage(self) -> dict[str, Any]:
        failed = [c for c in self.checks if c.get("status") != "PASS"]
        lineage = {"ticker": self.ticker, "run_at_utc": self.run_at_utc, "overall_status": "FAIL" if failed else "PASS", "checks": self.checks, "data_sources": self.data_sources, "anomalies": self.anomalies}
        if self.upstream_slo: lineage["upstream_slo"] = self.upstream_slo
        return lineage
