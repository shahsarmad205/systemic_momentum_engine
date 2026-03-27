"""
BacktestEngine — coordinates historical simulation via legacy Backtester.
Produces trades, daily equity, performance summary; optional experiment snapshot.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

from backtesting.backtester import Backtester, BacktestResult
from backtesting.config import BacktestConfig
from backtesting.config import load_config as _load_config


class BacktestEngine:
    """
    Facade around Backtester.run(); run_backtest() is the single entry for scripts.
    """

    def __init__(self, config: BacktestConfig | None = None, config_path: str = "backtest_config.yaml"):
        self.config_path = config_path
        self.config = config or _load_config(config_path)
        self._backtester = Backtester(self.config)

    def run_backtest(self, tickers: list[str] | None = None) -> BacktestResult:
        return self._backtester.run(tickers)

    def run_backtest_with_experiment(
        self,
        tickers: list[str] | None = None,
        experiment_dir: str | None = None,
    ) -> BacktestResult:
        """
        Run backtest and copy outputs + config snapshot into output/experiments/<timestamp>/.
        """
        result = self.run_backtest(tickers)
        if experiment_dir is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            experiment_dir = os.path.join("output", "experiments", ts)
        os.makedirs(experiment_dir, exist_ok=True)

        if not result.trades.empty:
            result.trades.to_csv(os.path.join(experiment_dir, "trades.csv"), index=False)
        if not result.daily_equity.empty:
            result.daily_equity.to_csv(os.path.join(experiment_dir, "daily_equity.csv"), index=False)
        with open(os.path.join(experiment_dir, "metrics.json"), "w") as fh:
            json.dump({k: (float(v) if isinstance(v, float | int) and not isinstance(v, bool) else v)
                       for k, v in result.metrics.items()}, fh, indent=2, default=str)
        # Config snapshot
        cfg_dict = {
            "start_date": self.config.start_date,
            "end_date": self.config.end_date,
            "initial_capital": self.config.initial_capital,
            "max_positions": self.config.max_positions,
            "holding_period_days": self.config.holding_period_days,
            "min_signal_strength": self.config.min_signal_strength,
            "signal_mode": self.config.signal_mode,
            "slippage_bps": self.config.slippage_bps,
            "commission_per_trade": self.config.commission_per_trade,
        }
        with open(os.path.join(experiment_dir, "config_snapshot.json"), "w") as fh:
            json.dump(cfg_dict, fh, indent=2)
        # Copy YAML config used for this run, if available
        if getattr(self, "config_path", None) and os.path.isfile(self.config_path):
            dest_name = os.path.basename(self.config_path) or "config.yaml"
            shutil.copy2(self.config_path, os.path.join(experiment_dir, dest_name))
        return result

    def save_experiment_snapshot(self, result: BacktestResult, experiment_dir: str | None = None) -> str:
        """Persist an already-run result to output/experiments/<timestamp>/ without re-running."""
        if experiment_dir is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            experiment_dir = os.path.join("output", "experiments", ts)
        os.makedirs(experiment_dir, exist_ok=True)
        if not result.trades.empty:
            result.trades.to_csv(os.path.join(experiment_dir, "trades.csv"), index=False)
        if not result.daily_equity.empty:
            result.daily_equity.to_csv(os.path.join(experiment_dir, "daily_equity.csv"), index=False)
        with open(os.path.join(experiment_dir, "metrics.json"), "w") as fh:
            json.dump({k: v for k, v in result.metrics.items()}, fh, indent=2, default=str)
        # Also snapshot the YAML config used, if known
        if getattr(self, "config_path", None) and os.path.isfile(self.config_path):
            dest_name = os.path.basename(self.config_path) or "config.yaml"
            shutil.copy2(self.config_path, os.path.join(experiment_dir, dest_name))
        return experiment_dir
