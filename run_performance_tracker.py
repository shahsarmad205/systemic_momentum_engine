#!/usr/bin/env python3
"""
Daily Performance Tracker
Tracks paper-trading performance against expected backtest metrics.

Usage:
    python run_performance_tracker.py
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


EXPECTED_SHARPE = 1.505
EXPECTED_CAGR = 0.1235
EXPECTED_WIN_RATE = 0.483


def _safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _compute_position_return(row: pd.Series) -> float:
    entry = _safe_float(row.get("entry_price"), np.nan)
    current = _safe_float(row.get("current_price"), np.nan)
    direction = str(row.get("direction", "LONG")).upper()
    if not np.isfinite(entry) or entry == 0 or not np.isfinite(current):
        return np.nan
    raw = (current - entry) / entry
    if direction in {"SHORT", "SELL"}:
        return -raw
    return raw


def _compute_signal_hit(row: pd.Series) -> float:
    ret = _safe_float(row.get("position_return"), np.nan)
    score = _safe_float(row.get("signal_score"), np.nan)
    if not np.isfinite(ret) or not np.isfinite(score):
        return np.nan
    pred = 1 if score >= 0 else -1
    actual = 1 if ret >= 0 else -1
    return float(pred == actual)


def _write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_tracker() -> None:
    portfolio_dir = Path("output/portfolio")
    portfolio_dir.mkdir(parents=True, exist_ok=True)

    positions_file = portfolio_dir / "paper_positions.csv"
    history_file = portfolio_dir / "performance_history.csv"
    report_file = portfolio_dir / "daily_report.txt"

    if not positions_file.exists():
        msg = [
            "No positions to track yet.",
            "Run run_daily_signals.py first, then add positions",
            "to output/portfolio/paper_positions.csv",
        ]
        print("\n".join(msg))
        _write_report(report_file, msg)
        return

    pos = pd.read_csv(positions_file)
    if pos.empty:
        msg = [
            "No positions to track yet.",
            "Run run_daily_signals.py first, then add positions",
            "to output/portfolio/paper_positions.csv",
        ]
        print("\n".join(msg))
        _write_report(report_file, msg)
        return

    required = [
        "date",
        "ticker",
        "entry_price",
        "current_price",
        "shares",
        "direction",
        "entry_date",
        "signal_score",
    ]
    missing = [c for c in required if c not in pos.columns]
    if missing:
        msg = [f"ERROR: paper_positions.csv missing columns: {missing}"]
        print(msg[0])
        _write_report(report_file, msg)
        return

    # Normalize / compute per-position metrics.
    pos["date"] = pd.to_datetime(pos["date"], errors="coerce")
    pos["entry_date"] = pd.to_datetime(pos["entry_date"], errors="coerce")
    pos["shares"] = pd.to_numeric(pos["shares"], errors="coerce").fillna(0.0)
    pos["entry_price"] = pd.to_numeric(pos["entry_price"], errors="coerce")
    pos["current_price"] = pd.to_numeric(pos["current_price"], errors="coerce")
    pos["position_return"] = pos.apply(_compute_position_return, axis=1)
    pos["signal_hit"] = pos.apply(_compute_signal_hit, axis=1)

    pos["entry_value"] = (pos["shares"].abs() * pos["entry_price"]).replace([np.inf, -np.inf], np.nan)
    pos["pnl"] = (
        (pos["current_price"] - pos["entry_price"])
        * pos["shares"]
        * np.where(pos["direction"].astype(str).str.upper().isin(["SHORT", "SELL"]), -1.0, 1.0)
    )
    pos["daily_ret_pos"] = pos["pnl"] / pos["entry_value"].replace(0, np.nan)

    # Aggregate today's book snapshot into one row.
    latest_date = pos["date"].dropna().max()
    if pd.isna(latest_date):
        latest_date = pd.Timestamp(datetime.today().date())
    today = pos[pos["date"] == latest_date].copy()
    if today.empty:
        today = pos.copy()

    total_entry_val = today["entry_value"].sum(skipna=True)
    total_pnl = today["pnl"].sum(skipna=True)
    daily_return = total_pnl / total_entry_val if total_entry_val > 0 else 0.0

    # Closed-position win rate proxy:
    # If current snapshot date > entry_date we treat return as realized-to-date.
    closed_mask = today["entry_date"].notna() & today["date"].notna() & (today["date"] > today["entry_date"])
    closed = today[closed_mask].copy()
    win_rate_closed = float((closed["position_return"] > 0).mean()) if len(closed) > 0 else np.nan

    # Signal accuracy on today's positions.
    signal_accuracy = float(today["signal_hit"].mean()) if today["signal_hit"].notna().any() else np.nan

    # Load / update history.
    hist_cols = [
        "date",
        "daily_return",
        "book_value",
        "pnl",
        "running_sharpe",
        "running_max_drawdown",
        "win_rate_closed",
        "signal_accuracy",
        "n_positions",
    ]
    if history_file.exists():
        hist = pd.read_csv(history_file)
    else:
        hist = pd.DataFrame(columns=hist_cols)

    new_row = {
        "date": latest_date.strftime("%Y-%m-%d"),
        "daily_return": float(daily_return),
        "book_value": float(total_entry_val),
        "pnl": float(total_pnl),
        "running_sharpe": np.nan,
        "running_max_drawdown": np.nan,
        "win_rate_closed": float(win_rate_closed) if np.isfinite(win_rate_closed) else np.nan,
        "signal_accuracy": float(signal_accuracy) if np.isfinite(signal_accuracy) else np.nan,
        "n_positions": int(len(today)),
    }

    # Replace same-date row if present, else append.
    if "date" in hist.columns and (hist["date"] == new_row["date"]).any():
        hist.loc[hist["date"] == new_row["date"], list(new_row.keys())] = list(new_row.values())
    else:
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.sort_values("date").reset_index(drop=True)

    # Running metrics.
    rets = pd.to_numeric(hist["daily_return"], errors="coerce").fillna(0.0).values
    n_days = int(np.isfinite(rets).sum())
    if n_days >= 2 and np.std(rets, ddof=1) > 1e-12:
        running_sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(252))
    else:
        running_sharpe = np.nan
    hist.loc[hist.index[-1], "running_sharpe"] = running_sharpe

    cum = (1.0 + pd.Series(rets)).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max.replace(0, np.nan)
    running_max_dd = float(dd.min()) if len(dd) > 0 else np.nan
    hist.loc[hist.index[-1], "running_max_drawdown"] = running_max_dd

    hist["date"] = hist["date"].dt.strftime("%Y-%m-%d")
    hist.to_csv(history_file, index=False)

    # Build text report.
    lines = []
    lines.append("=" * 60)
    lines.append(f"Daily Performance Report — {new_row['date']}")
    lines.append("=" * 60)
    lines.append(f"Open positions:           {int(new_row['n_positions'])}")
    lines.append(f"Book value:               {new_row['book_value']:.2f}")
    lines.append(f"Daily P&L:                {new_row['pnl']:.2f}")
    lines.append(f"Daily return:             {new_row['daily_return']:.4%}")
    lines.append("")
    lines.append("Running metrics:")
    rs = "N/A (need 20+ days)" if (not np.isfinite(running_sharpe) or n_days < 20) else f"{running_sharpe:.3f}"
    lines.append(f"  Running Sharpe:         {rs}")
    lines.append(
        f"  Running Max Drawdown:   {running_max_dd:.2%}" if np.isfinite(running_max_dd) else "  Running Max Drawdown:   N/A"
    )
    if np.isfinite(win_rate_closed):
        lines.append(f"  Win rate (closed):      {win_rate_closed:.2%}")
    else:
        lines.append("  Win rate (closed):      N/A")
    if np.isfinite(signal_accuracy):
        lines.append(f"  Signal accuracy:        {signal_accuracy:.2%}")
    else:
        lines.append("  Signal accuracy:        N/A")
    lines.append("")
    lines.append("Expected (backtest):")
    lines.append(f"  Sharpe:                 {EXPECTED_SHARPE:.3f}")
    lines.append(f"  CAGR:                   {EXPECTED_CAGR:.2%}")
    lines.append(f"  Win rate:               {EXPECTED_WIN_RATE:.2%}")
    lines.append("")
    lines.append("Alerts:")
    any_alert = False
    if np.isfinite(running_sharpe) and n_days >= 20 and running_sharpe < 0.5:
        lines.append("⚠️ ALERT: Live Sharpe below 0.5 — review signal")
        any_alert = True
    if np.isfinite(running_max_dd) and running_max_dd < -0.15:
        lines.append("🔴 ALERT: Drawdown > 15% — consider halting")
        any_alert = True
    if not any_alert:
        lines.append("No critical alerts.")
    lines.append("=" * 60)

    _write_report(report_file, lines)
    print("\n".join(lines))
    print(f"Saved report: {report_file}")
    print(f"Updated history: {history_file}")


if __name__ == "__main__":
    run_tracker()
