#!/usr/bin/env python3
"""
Diagnose Sideways regime degradation and suggest one config-only fix.
Console report only: no plots, no file writes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def diagnose_sideways(
    trades_csv: str = "output/backtests/trades.csv",
    equity_csv: str = "output/backtests/daily_equity.csv",
) -> dict:
    # Load inputs
    try:
        t = pd.read_csv(trades_csv)
    except FileNotFoundError:
        print(f"ERROR: Missing trades file: {trades_csv}")
        return {
            "n_trades": 0,
            "win_rate": np.nan,
            "avg_score": np.nan,
            "score_std": np.nan,
            "suggested_fix": "missing_trades_file",
        }
    try:
        e = pd.read_csv(equity_csv)
    except FileNotFoundError:
        print(f"ERROR: Missing equity file: {equity_csv}")
        return {
            "n_trades": 0,
            "win_rate": np.nan,
            "avg_score": np.nan,
            "score_std": np.nan,
            "suggested_fix": "missing_equity_file",
        }

    # Ensure expected columns exist
    req_trade_cols = {"regime", "return", "adjusted_score", "ticker", "entry_date"}
    req_equity_cols = {"regime", "equity"}
    if not req_trade_cols.issubset(set(t.columns)):
        missing = sorted(req_trade_cols - set(t.columns))
        print(f"ERROR: trades.csv missing columns: {missing}")
        return {
            "n_trades": 0,
            "win_rate": np.nan,
            "avg_score": np.nan,
            "score_std": np.nan,
            "suggested_fix": "invalid_trades_schema",
        }
    if not req_equity_cols.issubset(set(e.columns)):
        missing = sorted(req_equity_cols - set(e.columns))
        print(f"ERROR: daily_equity.csv missing columns: {missing}")
        return {
            "n_trades": 0,
            "win_rate": np.nan,
            "avg_score": np.nan,
            "score_std": np.nan,
            "suggested_fix": "invalid_equity_schema",
        }

    # Normalize numerics
    t["return"] = pd.to_numeric(t["return"], errors="coerce")
    t["adjusted_score"] = pd.to_numeric(t["adjusted_score"], errors="coerce")
    e["equity"] = pd.to_numeric(e["equity"], errors="coerce")
    e["ret"] = e["equity"].pct_change()

    # 1) Sideways diagnostics
    sw = t[t["regime"] == "Sideways"].copy()
    n_trades = int(len(sw))
    win_rate = float((sw["return"] > 0).mean()) if n_trades > 0 else np.nan
    avg_return = float(sw["return"].mean()) if n_trades > 0 else np.nan
    avg_score = float(sw["adjusted_score"].mean()) if n_trades > 0 else np.nan
    score_std = float(sw["adjusted_score"].std()) if n_trades > 1 else np.nan

    print("=== Sideways Regime Diagnostics ===")
    print(f"Total Sideways trades: {n_trades}")
    print(f"Win rate: {win_rate:.3f}" if np.isfinite(win_rate) else "Win rate: N/A")
    print(f"Avg return: {avg_return:.4f}" if np.isfinite(avg_return) else "Avg return: N/A")
    print(f"Avg score: {avg_score:.4f}" if np.isfinite(avg_score) else "Avg score: N/A")
    print(f"Score std: {score_std:.4f}" if np.isfinite(score_std) else "Score std: N/A")
    print()
    print("Return distribution:")
    if n_trades > 0:
        print(sw["return"].describe())
    else:
        print("No Sideways trades found.")
    print()
    print("Top 5 losing trades:")
    if n_trades > 0:
        print(sw.nsmallest(5, "return")[["ticker", "entry_date", "return", "adjusted_score"]])
    else:
        print("No Sideways trades found.")
    print()

    # 2) Compare score thresholds across regimes
    print("=== Regime Comparison (Trades) ===")
    for reg in ["Bull", "Bear", "Sideways", "Crisis"]:
        r = t[t["regime"] == reg]
        if len(r) == 0:
            print(f"{reg}: n=0")
            continue
        print(
            f"{reg}: n={len(r)}, "
            f"avg_score={r['adjusted_score'].mean():.4f}, "
            f"score_std={r['adjusted_score'].std():.4f}, "
            f"win_rate={(r['return'] > 0).mean():.3f}"
        )
    print()

    # 3) Regime-specific Sharpe from equity
    print("=== Regime Sharpe (Daily Equity) ===")
    for reg in ["Bull", "Bear", "Sideways", "Crisis"]:
        r = e[e["regime"] == reg]["ret"].dropna()
        s = r.mean() / r.std() * np.sqrt(252) if len(r) > 1 and r.std() > 0 else 0.0
        print(f"{reg:10s}: Sharpe={s:.3f}, n={len(r)}")
    print()

    # 4) Suggest exactly one fix
    # Priority follows user rule order.
    suggested_fix = ""
    if np.isfinite(avg_score) and abs(avg_score) < 0.10:
        suggested_fix = "signal_confidence_multiplier_sideways: 0.3"
        reason = "Sideways avg score is near zero."
    elif n_trades < 50:
        suggested_fix = "sideways_min_positions: 3"
        reason = "Sideways trade count is low (< 50)."
    else:
        bull = t[t["regime"] == "Bull"]
        bull_avg = float(bull["adjusted_score"].mean()) if len(bull) > 0 else np.nan
        if np.isfinite(avg_score) and np.isfinite(bull_avg) and avg_score < 0 and bull_avg > 0:
            suggested_fix = "regime_score_direction: check Sideways direction inversion"
            reason = "Sideways avg score is negative while Bull avg score is positive."
        else:
            suggested_fix = "signal_confidence_multiplier_sideways: 0.3"
            reason = "No stronger trigger; default to threshold tuning for Sideways."

    print("=== Suggested Config-Only Fix ===")
    print(f"Suggestion: {suggested_fix}")
    print(f"Reason: {reason}")

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_score": avg_score,
        "score_std": score_std,
        "suggested_fix": suggested_fix,
    }


if __name__ == "__main__":
    diagnose_sideways()
