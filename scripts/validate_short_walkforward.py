#!/usr/bin/env python3
"""
Walk-forward validation for short signals.

Splits history into rolling windows:
- Train shorts on earlier period
- Test on later period
- Measure generalization (OOS Sharpe, win rate, directional accuracy)

Answers: "Do shorts actually work OOS, or is it just overfitting?"

Usage:
  python scripts/validate_short_walkforward.py [--periods 5] [--regime Bear]
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def run_walkforward_split(
    n_periods: int = 5,
    target_regimes: list = None,
):
    """
    Run walk-forward validation.
    
    Args:
        n_periods: Number of rolling train/test splits
        target_regimes: Only test shorts in these regimes (e.g., ["Bear", "Crisis"])
    """
    if target_regimes is None:
        target_regimes = ["Bear", "Crisis"]
    
    logger.info("=" * 70)
    logger.info("WALK-FORWARD VALIDATION: SHORT SIGNALS")
    logger.info("=" * 70)
    
    # Load trades
    trades_path = Path("output/backtests/trades.csv")
    if not trades_path.exists():
        logger.error(f"No backtest trades at {trades_path}")
        return 1
    
    trades = pd.read_csv(trades_path)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades = trades.sort_values("entry_date")
    
    if "direction" not in trades.columns:
        trades["direction"] = (trades.get("signal") == "Bullish").astype(int) * 2 - 1
    
    date_min = trades["entry_date"].min()
    date_max = trades["entry_date"].max()
    total_days = (date_max - date_min).days
    
    logger.info(f"Trades date range: {date_min.date()} to {date_max.date()} ({total_days} days)")
    logger.info(f"Total trades: {len(trades)}")
    logger.info(f"Target regimes for short validation: {target_regimes}")
    
    # Walk-forward periods
    period_len = total_days // n_periods
    results = []
    
    for i in range(n_periods - 1):  # n-1 windows so last fold is test
        train_start = date_min + timedelta(days=i * period_len)
        train_end = date_min + timedelta(days=(i + 1) * period_len)
        test_start = train_end
        test_end = date_min + timedelta(days=(i + 2) * period_len)
        
        train_trades = trades[
            (trades["entry_date"] >= train_start)
            & (trades["entry_date"] < train_end)
        ]
        test_trades = trades[
            (trades["entry_date"] >= test_start)
            & (trades["entry_date"] < test_end)
        ]
        
        # Filter OOS test to target regimes only
        test_shorts = test_trades[
            (test_trades["regime"].isin(target_regimes))
            & (test_trades["direction"] == -1)
        ]
        
        if len(test_shorts) == 0:
            logger.warning(f"Period {i}: No shorts in test set for regimes {target_regimes}")
            continue
        
        # Metrics
        short_pnl = test_shorts["pnl"].values
        n_shorts = len(test_shorts)
        win_rate = (short_pnl > 0).mean()
        avg_pnl = short_pnl.mean()
        total_pnl = short_pnl.sum()
        sharpe = np.mean(short_pnl) / (np.std(short_pnl) + 1e-9) * np.sqrt(252 / (n_shorts or 1))
        
        result = {
            "period": i,
            "train_range": f"{train_start.date()} to {train_end.date()}",
            "test_range": f"{test_start.date()} to {test_end.date()}",
            "test_shorts_count": n_shorts,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "total_pnl": total_pnl,
            "sharpe": sharpe,
        }
        results.append(result)
        
        logger.info(
            f"Period {i}: Shorts in {target_regimes}: {n_shorts} trades, "
            f"Win rate {win_rate:.1%}, Avg PnL ${avg_pnl:.2f}, OOS Sharpe {sharpe:.3f}"
        )
    
    # Summary
    if results:
        df_res = pd.DataFrame(results)
        logger.info("\n" + "=" * 70)
        logger.info("WALK-FORWARD SUMMARY")
        logger.info("=" * 70)
        logger.info(df_res.to_string(index=False))
        
        avg_win_rate = df_res["win_rate"].mean()
        avg_sharpe = df_res["sharpe"].mean()
        
        logger.info(f"\nAverage OOS Win Rate: {avg_win_rate:.1%}")
        logger.info(f"Average OOS Sharpe: {avg_sharpe:.3f}")
        
        if avg_win_rate < 0.40:
            logger.warning("⚠️  Win rate < 40% — shorts not profitable OOS in target regimes")
        if avg_sharpe < 0.3:
            logger.warning("⚠️  Sharpe < 0.3 — shorts not generating sufficient risk-adjusted returns")
        
        # Save
        output_json = "output/research/short_walkforward_validation.json"
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(
            json.dumps({
                "target_regimes": target_regimes,
                "n_periods": n_periods,
                "avg_win_rate": float(avg_win_rate),
                "avg_sharpe": float(avg_sharpe),
                "periods": results,
            }, indent=2)
        )
        logger.info(f"\n✅ Results saved to {output_json}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward validation for short signals (OOS testing)"
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=5,
        help="Number of walk-forward periods",
    )
    parser.add_argument(
        "--regimes",
        type=str,
        default="Bear,Crisis",
        help="Comma-separated regimes to validate shorts in (e.g., 'Bear,Crisis')",
    )
    
    args = parser.parse_args()
    target_regimes = [r.strip() for r in args.regimes.split(",")]
    
    return run_walkforward_split(n_periods=args.periods, target_regimes=target_regimes)


if __name__ == "__main__":
    exit(main())
