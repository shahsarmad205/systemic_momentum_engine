"""
Advanced Research Tools
=========================
IC decay analysis, walk-forward testing, and parameter sweeps.
All functions accept pre-computed data so they can be called
independently of the backtester.
"""

from __future__ import annotations

import copy
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# IC Decay Analysis
# ------------------------------------------------------------------

# Default forward horizons for IC decay (1d, 5d, 10d, 20d)
DEFAULT_IC_DECAY_LAGS = [1, 5, 10, 20]


def compute_ic_decay(
    price_data: dict[str, pd.DataFrame],
    signal_data: dict[str, pd.DataFrame],
    lags: list[int],
    include_all_dates: bool = True,
) -> list[float]:
    """
    For each lag, compute Pearson correlation between adjusted_score
    and the forward return at that horizon across all tickers/dates.

    If include_all_dates is True (default), use every date with valid
    score and forward return (full predictive power curve). If False,
    use only dates where signal != "Neutral".
    """
    ic_values: list[float] = []

    for lag in lags:
        scores: list[float] = []
        returns: list[float] = []

        for ticker, sig_df in signal_data.items():
            if ticker not in price_data:
                continue
            prices = price_data[ticker]["Close"]
            pos_index = prices.index

            for date, row in sig_df.iterrows():
                if not include_all_dates and row["signal"] == "Neutral":
                    continue
                idx = pos_index.get_indexer([date])[0]
                if idx < 0:
                    continue
                future_idx = idx + lag
                if future_idx >= len(pos_index):
                    continue
                cur = float(prices.iloc[idx])
                fut = float(prices.iloc[future_idx])
                if cur <= 0:
                    continue
                scores.append(float(row["adjusted_score"]))
                returns.append((fut - cur) / cur)

        if len(scores) >= 10:
            ic = float(pd.Series(scores).corr(pd.Series(returns)))
            ic_values.append(ic if not np.isnan(ic) else 0.0)
        else:
            ic_values.append(0.0)

    return ic_values


def best_ic_horizon(lags: list[int], ic_values: list[float]) -> tuple[int, float]:
    """
    Return (lag, ic) for the horizon where |IC| is strongest.
    If all ICs are zero or empty, returns (lags[0], 0.0).
    """
    if not lags or not ic_values or len(lags) != len(ic_values):
        return (lags[0] if lags else 1, 0.0)
    best_idx = int(np.argmax(np.abs(ic_values)))
    return (lags[best_idx], float(ic_values[best_idx]))


def compute_rank_ic_decay(
    price_data: dict[str, pd.DataFrame],
    signal_data: dict[str, pd.DataFrame],
    lags: list[int],
    include_all_dates: bool = True,
) -> list[float]:
    """Same as compute_ic_decay but using Spearman (rank) correlation."""
    ic_values: list[float] = []

    for lag in lags:
        scores: list[float] = []
        returns: list[float] = []

        for ticker, sig_df in signal_data.items():
            if ticker not in price_data:
                continue
            prices = price_data[ticker]["Close"]
            pos_index = prices.index

            for date, row in sig_df.iterrows():
                if not include_all_dates and row["signal"] == "Neutral":
                    continue
                idx = pos_index.get_indexer([date])[0]
                if idx < 0:
                    continue
                future_idx = idx + lag
                if future_idx >= len(pos_index):
                    continue
                cur = float(prices.iloc[idx])
                fut = float(prices.iloc[future_idx])
                if cur <= 0:
                    continue
                scores.append(float(row["adjusted_score"]))
                returns.append((fut - cur) / cur)

        if len(scores) >= 10:
            ic = float(pd.Series(scores).corr(pd.Series(returns), method="spearman"))
            ic_values.append(ic if not np.isnan(ic) else 0.0)
        else:
            ic_values.append(0.0)

    return ic_values


# ------------------------------------------------------------------
# Walk-forward splits
# ------------------------------------------------------------------

def walk_forward_splits(
    start_date: str,
    end_date: str,
    n_windows: int = 4,
    train_ratio: float = 0.7,
) -> list[dict]:
    """
    Generate rolling walk-forward train/test date splits.

    Returns list of dicts:
        {"train_start", "train_end", "test_start", "test_end"}
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    total_days = (end - start).days
    window_size = total_days // n_windows

    splits = []
    for i in range(n_windows):
        w_start = start + pd.Timedelta(days=i * window_size)
        w_end = min(w_start + pd.Timedelta(days=window_size), end)

        train_end = w_start + pd.Timedelta(days=int(window_size * train_ratio))
        test_start = train_end + pd.Timedelta(days=1)

        if test_start >= w_end:
            continue

        splits.append({
            "train_start": w_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": w_end.strftime("%Y-%m-%d"),
        })

    return splits


def run_walk_forward(
    config,
    tickers: list[str] | None = None,
    *,
    train_weights: bool | None = None,
    report_path: str | None = None,
) -> tuple[list, pd.DataFrame]:
    """
    Walk-forward validation framework:

    - Splits history into sequential train/test windows.
    - When train_weights is True and signal_mode is learned (or train_weights forced),
      trains the weight model on the training window only, saves weights, then
      runs the backtester on the out-of-sample test window with those weights.
    - Otherwise runs backtest on each test window with the current config (no retrain).

    Returns:
        (list of BacktestResult, summary DataFrame with OOS Sharpe, drawdown,
         directional accuracy, information coefficient per window.)
    """
    import os
    from .backtester import Backtester

    splits = walk_forward_splits(
        config.start_date,
        config.end_date,
        config.walk_forward_windows,
        config.walk_forward_train_ratio,
    )

    if train_weights is None:
        train_weights = getattr(config, "walk_forward_train_weights", True)
    report_path = report_path or getattr(
        config, "walk_forward_report_path", "output/backtests/walk_forward_validation_report.csv"
    )

    results = []
    summary_rows: list[dict] = []

    for i, split in enumerate(splits, 1):
        print(f"\n{'='*55}")
        print(f"  Walk-forward window {i}/{len(splits)}")
        print(f"  Train: {split['train_start']} → {split['train_end']}")
        print(f"  OOS  : {split['test_start']} → {split['test_end']}")
        print(f"{'='*55}\n")

        window_cfg = copy.deepcopy(config)
        window_cfg.start_date = split["test_start"]
        window_cfg.end_date = split["test_end"]

        weights_path = None
        if train_weights and getattr(config, "signal_mode", "") == "learned":
            try:
                from agents.weight_learning_agent import build_feature_matrix, WeightLearner

                # Build features from train start through test end (rows in train range used for fit only)
                features_df = build_feature_matrix(
                    tickers=tickers or [],
                    start_date=split["train_start"],
                    end_date=split["test_end"],
                    holding_period=getattr(config, "holding_period_days", 5),
                )
                train_ts_start = pd.Timestamp(split["train_start"])
                train_ts_end = pd.Timestamp(split["train_end"])
                train_df = features_df[
                    (features_df["date"] >= train_ts_start)
                    & (features_df["date"] <= train_ts_end)
                ]
                if len(train_df) >= 100:
                    learner = WeightLearner(
                        model_type="ridge",
                        alpha=0.01,
                        time_decay_lambda=0.001,
                    )
                    learner.fit(train_df)
                    weights_path = os.path.join(
                        os.path.dirname(report_path) or ".",
                        f"wf_window_{i}_learned_weights.json",
                    )
                    weights = learner.get_weights(train_df)
                    weights.save(weights_path)
                    window_cfg.learned_weights_path = weights_path
                    window_cfg.signal_mode = "learned"
                    print(f"  Trained weight model on {len(train_df):,} samples → {weights_path}")
                else:
                    print(f"  [WARN] Train window too few samples ({len(train_df)}); skipping retrain.")
            except Exception as exc:
                print(f"  [WARN] Weight training failed: {exc}; OOS backtest uses existing config.")

        bt = Backtester(window_cfg)
        result = bt.run(tickers)
        results.append(result)

        m = result.metrics
        summary_rows.append({
            "window": i,
            "train_start": split["train_start"],
            "train_end": split["train_end"],
            "test_start": split["test_start"],
            "test_end": split["test_end"],
            "oos_sharpe": m.get("sharpe_ratio", 0.0),
            "oos_max_drawdown": m.get("max_drawdown", 0.0),
            "oos_directional_accuracy": m.get("signal_accuracy", 0.0),
            "oos_information_coefficient": m.get("information_coefficient", 0.0),
            "oos_total_return": m.get("total_return", 0.0),
            "oos_total_trades": m.get("total_trades", 0),
            "oos_rank_ic": m.get("rank_ic", 0.0),
            "weights_path": weights_path or "",
        })

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        summary_df.to_csv(report_path, index=False)
        print(f"\n  Walk-forward validation report saved → {report_path}")

    return results, summary_df


# ------------------------------------------------------------------
# Parameter grid / sweep
# ------------------------------------------------------------------

def parameter_grid(param_ranges: dict[str, list]) -> list[dict]:
    """
    Cartesian product of parameter ranges.

    Example:
        parameter_grid({
            "holding_period_days": [3, 5, 10],
            "min_signal_strength": [0.2, 0.3, 0.5],
        })
    """
    keys = list(param_ranges.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*param_ranges.values())]


def run_parameter_sweep(
    base_config,
    param_ranges: dict[str, list],
    tickers: list[str] | None = None,
    target_metric: str = "sharpe_ratio",
) -> pd.DataFrame:
    """
    Run a grid search over the parameter space and return a DataFrame
    of (param_combo, metric_values) sorted by *target_metric*.
    """
    from .backtester import Backtester

    grid = parameter_grid(param_ranges)
    rows: list[dict] = []

    for idx, combo in enumerate(grid, 1):
        print(f"\n  Sweep {idx}/{len(grid)}: {combo}")
        cfg = copy.deepcopy(base_config)
        for key, val in combo.items():
            setattr(cfg, key, val)

        bt = Backtester(cfg)
        result = bt.run(tickers)
        row = dict(combo)
        row.update(result.metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    if target_metric in df.columns:
        df.sort_values(target_metric, ascending=False, inplace=True)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------
# Transaction cost sensitivity
# ------------------------------------------------------------------

# Default cost scenarios: (slippage_bps, commission_per_trade) for sensitivity analysis
DEFAULT_COST_SCENARIOS = [
    {"slippage_bps": 0, "commission_per_trade": 0},
    {"slippage_bps": 2, "commission_per_trade": 0.5},
    {"slippage_bps": 5, "commission_per_trade": 1.0},
    {"slippage_bps": 10, "commission_per_trade": 2.0},
    {"slippage_bps": 15, "commission_per_trade": 3.0},
    {"slippage_bps": 20, "commission_per_trade": 5.0},
    {"slippage_bps": 30, "commission_per_trade": 10.0},
]


def run_transaction_cost_sensitivity(
    config,
    tickers: list[str] | None = None,
    scenarios: list[dict] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run backtests across multiple transaction-cost assumptions and return
    a DataFrame of performance metrics per scenario.

    Each scenario is a dict with at least:
        slippage_bps: float
        commission_per_trade: float

    If scenarios is None, uses DEFAULT_COST_SCENARIOS.
    """
    from .backtester import Backtester

    scenarios = scenarios or DEFAULT_COST_SCENARIOS
    rows: list[dict] = []

    for idx, scenario in enumerate(scenarios, 1):
        slippage_bps = scenario["slippage_bps"]
        commission_per_trade = scenario["commission_per_trade"]
        if verbose:
            print(f"  Cost scenario {idx}/{len(scenarios)}: "
                  f"slippage={slippage_bps} bps, commission=${commission_per_trade:.2f}")

        cfg = copy.deepcopy(config)
        cfg.slippage_bps = float(slippage_bps)
        cfg.commission_per_trade = float(commission_per_trade)
        # Use legacy ExecutionEngine path so scenario slippage/commission are actually applied.
        # When execution_costs_enabled is True, the backtester uses TransactionCostModel and
        # commission=0, so sensitivity would show identical results.
        cfg.execution_costs_enabled = False

        bt = Backtester(cfg)
        result = bt.run(tickers)

        m = result.metrics
        row = {
            "slippage_bps": slippage_bps,
            "commission_per_trade": commission_per_trade,
            "total_return": m["total_return"],
            "sharpe_ratio": m["sharpe_ratio"],
            "max_drawdown": m["max_drawdown"],
            "win_rate": m["win_rate"],
            "total_trades": m["total_trades"],
            "total_pnl": m["total_pnl"],
            "profit_factor": m["profit_factor"],
            "sortino_ratio": m["sortino_ratio"],
            "calmar_ratio": m["calmar_ratio"],
            "final_capital": m["final_capital"],
            "total_transaction_costs": m.get("total_transaction_costs", 0.0),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Execution costs sensitivity (total bps per leg: 5, 10, 20, etc.)
# ------------------------------------------------------------------

def run_execution_costs_sensitivity(
    config,
    tickers: list[str] | None = None,
    scenarios: list[float] | None = None,
    report_path: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run backtests with execution_costs enabled at different total cost levels (bps).
    Saves results to output/research/cost_sensitivity.csv by default.

    scenarios: list of total bps per leg (e.g. [5, 10, 20]); cost is split as commission/spread/slippage.
    """
    import copy
    from .backtester import Backtester

    scenarios = scenarios or getattr(config, "execution_costs_scenarios", [5.0, 10.0, 20.0])
    report_path = report_path or getattr(
        config, "execution_costs_sensitivity_report_path", "output/research/cost_sensitivity.csv"
    )
    rows: list[dict] = []

    for idx, total_bps in enumerate(scenarios, 1):
        if verbose:
            print(f"  Execution cost scenario {idx}/{len(scenarios)}: total={total_bps} bps per leg")

        cfg = copy.deepcopy(config)
        cfg.execution_costs_enabled = True
        # Split total bps evenly across commission, spread, slippage
        third = total_bps / 3.0
        cfg.execution_costs_commission_bps = third
        cfg.execution_costs_spread_bps = third
        cfg.execution_costs_slippage_bps = third

        bt = Backtester(cfg)
        result = bt.run(tickers)
        m = result.metrics

        row = {
            "total_bps": total_bps,
            "total_return": m.get("total_return", 0.0),
            "gross_return": m.get("gross_return", m.get("total_return", 0.0)),
            "net_return": m.get("net_return", m.get("total_return", 0.0)),
            "sharpe_ratio": m.get("sharpe_ratio", 0.0),
            "max_drawdown": m.get("max_drawdown", 0.0),
            "win_rate": m.get("win_rate", 0.0),
            "total_trades": m.get("total_trades", 0),
            "total_transaction_costs": m.get("total_transaction_costs", 0.0),
            "average_cost_per_trade": m.get("average_cost_per_trade", 0.0),
            "final_capital": m.get("final_capital", config.initial_capital),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if report_path and not df.empty:
        import os
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        df.to_csv(report_path, index=False)
        if verbose:
            print(f"  Saved → {report_path}")
    return df
