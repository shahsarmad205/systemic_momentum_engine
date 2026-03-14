"""
Central configuration for the research platform.
Re-exports BacktestConfig loading; adds DEV_MODE and ticker limit helper.
"""

from __future__ import annotations

import logging

# Development mode: limit tickers, shorten date range, verbose logging
DEV_MODE = True
DEV_TICKER_LIMIT = 20
DEV_DATE_RANGE_DAYS = 730  # ~2 years for faster iteration


def get_effective_tickers(config_tickers: list | None, fallback_list: list | None) -> list:
    """
    When DEV_MODE: cap universe to DEV_TICKER_LIMIT (from config or fallback).
    Otherwise return config tickers if set, else full fallback list.
    """
    raw = list(config_tickers) if config_tickers else list(fallback_list or [])
    if DEV_MODE and raw:
        return raw[:DEV_TICKER_LIMIT]
    return raw


def apply_dev_mode(config: "BacktestConfig") -> None:
    """
    When DEV_MODE: limit ticker count only. Date range is left as in config
    (e.g. 2018-01-01 → 2024-01-01) so backtest period is not shortened.
    """
    if not DEV_MODE:
        return
    # Ticker limit is applied in get_effective_tickers(); do not override dates here.


def setup_logging(verbose: bool = False) -> None:
    # INFO by default; DEBUG only when verbose=True (avoid noisy third-party loggers)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# Re-export config loader and dataclass (single source remains backtesting.config)
from backtesting.config import BacktestConfig, load_config  # noqa: E402

__all__ = [
    "DEV_MODE",
    "DEV_TICKER_LIMIT",
    "DEV_DATE_RANGE_DAYS",
    "get_effective_tickers",
    "apply_dev_mode",
    "setup_logging",
    "BacktestConfig",
    "load_config",
]
