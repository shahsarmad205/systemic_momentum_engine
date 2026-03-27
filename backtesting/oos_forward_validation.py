"""
Out-of-sample (forward) validation
===================================
Runs the backtester twice using the *already learned* weights:

1) In-sample: 2013-01-01 → 2024-01-01
2) OOS (forward): 2024-01-01 → 2025-12-31

Uses `signals.mode=learned` and the existing `output/learned_weights.json`
configured via `backtest_config.yaml`. No retraining is performed.

Prints:
  - CAGR, Sharpe, max drawdown
  - Monthly returns stats for OOS (and a monthly table)
  - In-sample vs OOS comparison
  - Pass/fail against: OOS Sharpe > 0.5 and OOS CAGR > 5%
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure `trend_signal_engine/` is on sys.path when this file is executed
# as a script from within `trend_signal_engine/backtesting/`.
_pkg_root = Path(__file__).resolve().parents[1]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from backtest.engine import BacktestEngine
from backtesting.config import load_config


@dataclass(frozen=True)
class PeriodReport:
    start_date: str
    end_date: str
    cagr: float
    sharpe: float
    max_drawdown: float
    calmar: float
    monthly_returns: pd.Series


def _compute_cagr(equity: pd.Series, years: float) -> float:
    equity = equity.dropna().astype(float)
    if equity.empty or years <= 0:
        return 0.0
    e0 = float(equity.iloc[0])
    ef = float(equity.iloc[-1])
    if e0 <= 0:
        return 0.0
    if ef <= 0:
        return -1.0
    return (ef / e0) ** (1.0 / years) - 1.0


def _compute_sharpe(daily_returns: pd.Series, annualization: int = 252) -> float:
    r = daily_returns.dropna().astype(float)
    if len(r) < 2:
        return 0.0
    std = float(r.std(ddof=0))
    if std <= 1e-12:
        return 0.0
    return float((r.mean() / std) * np.sqrt(annualization))


def _compute_max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna().astype(float)
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def _equity_to_report(daily_equity: pd.DataFrame, start_date: str, end_date: str) -> PeriodReport:
    if daily_equity is None or daily_equity.empty:
        return PeriodReport(
            start_date=start_date,
            end_date=end_date,
            cagr=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            calmar=float("inf"),
            monthly_returns=pd.Series(dtype=float),
        )

    df = daily_equity.copy()
    if "date" not in df.columns or "equity" not in df.columns:
        raise ValueError("daily_equity must have `date` and `equity` columns")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    df = df.dropna(subset=["date", "equity"]).sort_values("date")
    if df.empty:
        raise ValueError("daily_equity has no valid rows after cleaning")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()
    if df.empty:
        raise ValueError(f"daily_equity has no rows within reporting window {start_date} → {end_date}")

    years = max((datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days / 365.25, 1e-6)
    equity = df.set_index("date")["equity"].astype(float)
    daily_ret = equity.pct_change().dropna()
    cagr = _compute_cagr(equity, years)
    sharpe = _compute_sharpe(daily_ret)
    max_dd = _compute_max_drawdown(equity)
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("inf")

    # Monthly returns: month-end equity pct change
    m_eq = equity.resample("ME").last()
    m_ret = m_eq.pct_change().dropna()

    return PeriodReport(
        start_date=start_date,
        end_date=end_date,
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calmar=calmar,
        monthly_returns=m_ret,
    )


def run_period(config, tickers: list[str], start_date: str, end_date: str) -> PeriodReport:
    # Mutate a copy-ish (BacktestEngine holds the reference, so we create a fresh one)
    config.start_date = start_date
    config.end_date = end_date

    # Ensure we only use the already-learned weights (no retraining).
    config.signal_mode = "learned"

    engine = BacktestEngine(config=config, config_path=config.__dict__.get("_config_path", "backtest_config.yaml"))
    result = engine.run_backtest(tickers)
    return _equity_to_report(result.daily_equity, start_date, end_date)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="backtest_config.yaml", help="Path to backtest_config.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    config = load_config(str(cfg_path))
    # Store path for BacktestEngine snapshots if needed.
    config._config_path = str(cfg_path)

    # Use the config tickers (standard backtest set).
    tickers = list(getattr(config, "tickers", []) or [])
    if not tickers:
        raise ValueError("No tickers found in config; populate `tickers:` in backtest_config.yaml or pass CLI tickers.")

    periods = [
        ("2013-01-01", "2024-01-01", "In-sample"),
        ("2024-01-01", "2025-12-31", "OOS (forward)"),
    ]

    reports: dict[str, PeriodReport] = {}
    for start_date, end_date, label in periods:
        print(f"\nRunning {label}: {start_date} → {end_date}")
        rep = run_period(config, tickers, start_date, end_date)
        reports[label] = rep

    in_rep = reports["In-sample"]
    oos_rep = reports["OOS (forward)"]

    print("\n================ Forward Validation Summary ================")
    print("In-sample:")
    print(f"  CAGR         : {in_rep.cagr:.2%}")
    print(f"  Sharpe       : {in_rep.sharpe:.2f}")
    print(f"  Max drawdown : {in_rep.max_drawdown:.2%}")
    print(f"  Calmar       : {in_rep.calmar:.2f}")

    print("\nOOS (forward):")
    print(f"  CAGR         : {oos_rep.cagr:.2%}")
    print(f"  Sharpe       : {oos_rep.sharpe:.2f}")
    print(f"  Max drawdown : {oos_rep.max_drawdown:.2%}")
    print(f"  Calmar       : {oos_rep.calmar:.2f}")

    # OOS monthly returns distribution
    m = oos_rep.monthly_returns.dropna().astype(float)
    print("\nOOS Monthly returns (month-end equity pct change):")
    if m.empty:
        print("  (no monthly data produced)")
    else:
        print(f"  mean={m.mean():.4f}, std={m.std(ddof=0):.4f}, skew={m.skew():.3f}")
        # Print a simple month table
        m_table = (m * 100).round(2)
        print("\n  Month-end returns (%):")
        for dt, val in m_table.items():
            print(f"    {dt.strftime('%Y-%m')}: {val:+.2f}%")

    pass_criteria = (oos_rep.sharpe > 0.5) and (oos_rep.cagr > 0.05)
    print("\n================ Pass/Fail ================")
    print("Criteria: OOS Sharpe > 0.5 AND OOS CAGR > 5%")
    print(f"Result  : {'PASS' if pass_criteria else 'FAIL'}")


if __name__ == "__main__":
    main()

