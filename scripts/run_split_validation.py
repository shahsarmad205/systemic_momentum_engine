#!/usr/bin/env python3
"""Build split-stack rollout validation report from backtest outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _read_config(path: str) -> dict[str, Any]:
    try:
        import yaml

        with open(ROOT / path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _detect_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_float(v: float | int | None) -> float | None:
    if v is None:
        return None
    x = float(v)
    return x if np.isfinite(x) else None


def _compute_crisis_sharpe(equity_df: pd.DataFrame) -> tuple[float | None, int]:
    if equity_df.empty:
        return None, 0
    date_col = _detect_col(equity_df, ["date", "Date"])
    equity_col = _detect_col(equity_df, ["equity", "portfolio_value", "Equity"])
    regime_col = _detect_col(equity_df, ["regime", "Regime"])
    if date_col is None or equity_col is None or regime_col is None:
        return None, 0

    work = equity_df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[equity_col] = pd.to_numeric(work[equity_col], errors="coerce")
    work = work.dropna(subset=[date_col, equity_col]).sort_values(date_col)
    work["ret"] = work[equity_col].pct_change()
    crisis = work[work[regime_col].astype(str).str.lower() == "crisis"].dropna(subset=["ret"]) 
    n = int(len(crisis))
    if n <= 1:
        return None, n
    vol = float(crisis["ret"].std(ddof=0) or 0.0)
    if vol <= 0:
        return None, n
    sharpe = float((crisis["ret"].mean() / vol) * np.sqrt(252.0))
    return sharpe, n


def main() -> int:
    parser = argparse.ArgumentParser(description="Run split rollout validation checks")
    parser.add_argument("--config", type=str, default="backtest_config.yaml")
    parser.add_argument("--trades-path", type=str, default="output/backtests/trades.csv")
    parser.add_argument("--equity-path", type=str, default="output/backtests/daily_equity.csv")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when validation fails")
    args = parser.parse_args()

    cfg = _read_config(args.config)
    ms_cfg = (cfg.get("model_selection", {}) or {})
    gates = (ms_cfg.get("split_rollout_gates", {}) or {})

    min_short_pnl_total = float(gates.get("min_short_pnl_total", 0.0) or 0.0)
    min_short_pnl_bear = float(gates.get("min_short_pnl_bear", 0.0) or 0.0)
    max_bull_short_damage_abs = float(gates.get("max_bull_short_damage_abs", 0.0) or 0.0)
    max_short_stoploss_rate_bull = float(gates.get("max_short_stoploss_rate_bull", 0.50) or 0.50)
    min_crisis_sharpe = float(gates.get("min_crisis_sharpe", -0.50) or -0.50)
    min_short_trades_bull = int(gates.get("min_short_trades_bull", 5) or 5)
    min_crisis_days = int(gates.get("min_crisis_days", 20) or 20)
    fail_on_insufficient = bool(gates.get("fail_on_insufficient_data", True))

    trades_path = ROOT / args.trades_path
    equity_path = ROOT / args.equity_path

    failures: list[str] = []
    metrics: dict[str, Any] = {}

    if not trades_path.exists():
        failures.append("missing_trades")
        trades = pd.DataFrame()
    else:
        trades = pd.read_csv(trades_path)

    if not equity_path.exists():
        failures.append("missing_equity")
        equity = pd.DataFrame()
    else:
        equity = pd.read_csv(equity_path)

    if not trades.empty:
        direction_col = _detect_col(trades, ["direction", "position_direction"])
        pnl_col = _detect_col(trades, ["pnl", "realized_pnl"])
        regime_col = _detect_col(trades, ["regime", "Regime"])
        exit_col = _detect_col(trades, ["exit_reason", "exit"])

        if direction_col is None or pnl_col is None:
            failures.append("invalid_trades_schema")
        else:
            t = trades.copy()
            t[direction_col] = t[direction_col].astype(str).str.lower().str.strip()
            t[pnl_col] = pd.to_numeric(t[pnl_col], errors="coerce").fillna(0.0)
            short = t[t[direction_col] == "short"].copy()
            metrics["short_trade_count"] = int(len(short))
            short_pnl_total = float(short[pnl_col].sum()) if not short.empty else 0.0
            metrics["short_pnl_total"] = short_pnl_total
            if short_pnl_total < min_short_pnl_total:
                failures.append("short_pnl_total_below_min")

            if regime_col is None:
                failures.append("missing_trade_regime")
            else:
                short[regime_col] = short[regime_col].astype(str)
                bear = short[short[regime_col].str.lower() == "bear"]
                bull = short[short[regime_col].str.lower() == "bull"]
                short_pnl_bear = float(bear[pnl_col].sum()) if not bear.empty else 0.0
                short_pnl_bull = float(bull[pnl_col].sum()) if not bull.empty else 0.0
                metrics["short_pnl_bear"] = short_pnl_bear
                metrics["short_pnl_bull"] = short_pnl_bull
                metrics["short_trade_count_bull"] = int(len(bull))
                if short_pnl_bear < min_short_pnl_bear:
                    failures.append("short_pnl_bear_below_min")
                if short_pnl_bull < -abs(max_bull_short_damage_abs):
                    failures.append("bull_short_damage_cap_breached")
                if int(len(bull)) < min_short_trades_bull and fail_on_insufficient:
                    failures.append("insufficient_bull_short_trades")

                if exit_col is not None:
                    stoploss_rate_bull = 0.0
                    if len(bull) > 0:
                        exit_norm = bull[exit_col].astype(str).str.strip().str.lower()
                        stoploss_rate_bull = float((exit_norm == "stop_loss").mean())
                    metrics["short_stoploss_rate_bull"] = stoploss_rate_bull
                    if len(bull) > 0 and stoploss_rate_bull > max_short_stoploss_rate_bull:
                        failures.append("bull_short_stoploss_rate_too_high")
                else:
                    failures.append("missing_exit_reason")

    crisis_sharpe, crisis_days = _compute_crisis_sharpe(equity)
    metrics["crisis_sharpe"] = _safe_float(crisis_sharpe)
    metrics["crisis_days"] = int(crisis_days)
    if crisis_days < min_crisis_days:
        if fail_on_insufficient:
            failures.append("insufficient_crisis_days")
    else:
        if crisis_sharpe is None or crisis_sharpe < min_crisis_sharpe:
            failures.append("crisis_sharpe_below_min")

    now = datetime.now(timezone.utc)
    report = {
        "generated_at_utc": now.isoformat(),
        "run_at_utc": now.isoformat(),
        "inputs": {
            "trades_path": str(trades_path),
            "equity_path": str(equity_path),
        },
        "gates": {
            "min_short_pnl_total": min_short_pnl_total,
            "min_short_pnl_bear": min_short_pnl_bear,
            "max_bull_short_damage_abs": max_bull_short_damage_abs,
            "max_short_stoploss_rate_bull": max_short_stoploss_rate_bull,
            "min_crisis_sharpe": min_crisis_sharpe,
            "min_short_trades_bull": min_short_trades_bull,
            "min_crisis_days": min_crisis_days,
            "fail_on_insufficient_data": fail_on_insufficient,
        },
        "metrics": metrics,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }

    out_dir = ROOT / "output" / "models" / "split_validation_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"split_validation_{ts}.json"
    latest_path = ROOT / "output" / "models" / "split_validation_latest.json"
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    out_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")

    print(f"Saved split validation report: {out_path}")
    print("Status:", report["status"])
    if failures:
        print("Failing gates:", ", ".join(failures))
        if args.strict:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
