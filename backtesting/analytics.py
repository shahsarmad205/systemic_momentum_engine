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

def _date_add_years(ts: pd.Timestamp, years: float) -> pd.Timestamp:
    months = max(int(round(years * 12)), 1)
    return ts + pd.DateOffset(months=months)


def walk_forward_splits(
    start_date: str,
    end_date: str,
    n_windows: int = 4,
    train_ratio: float = 0.7,
    embargo_days: int = 21,
    *,
    train_years: float | None = None,
    test_years: float | None = None,
    step_years: float | None = None,
) -> list[dict]:
    """
    Generate rolling walk-forward train/test date splits with embargo period.

    Preferred institutional path:
      - calendar-based train/test/step windows (e.g. 5y / 1y / 1y)
      - training set purged by embargo_days before the OOS boundary

    Legacy fallback path:
      - equal-sized block splits controlled by n_windows/train_ratio

    Parameters
    ----------
    start_date, end_date : str
        ISO date strings for the full backtest period.
    n_windows : int
        Number of walk-forward windows.
    train_ratio : float
        Fraction of each window used for training (before embargo).
    embargo_days : int
        Calendar-day purge between train_end and test_start.
    train_years, test_years, step_years : float | None
        When all are positive, use the institutional calendar roll:
        train on [cursor, cursor + train_years) and test on
        [cursor + train_years, cursor + train_years + test_years),
        then advance by step_years.

    Returns
    -------
    list of dicts:
        {"train_start", "train_end", "test_start", "test_end", "embargo_days"}
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if (
        train_years is not None
        and test_years is not None
        and step_years is not None
        and train_years > 0
        and test_years > 0
        and step_years > 0
    ):
        splits: list[dict] = []
        cursor = start
        embargo_delta = pd.Timedelta(days=max(int(embargo_days), 0))
        while True:
            train_start = cursor
            test_start = _date_add_years(train_start, float(train_years))
            if test_start >= end:
                break
            test_end = min(_date_add_years(test_start, float(test_years)), end)
            train_end = test_start - embargo_delta
            if train_end > train_start:
                splits.append({
                    "train_start": train_start.strftime("%Y-%m-%d"),
                    "train_end": train_end.strftime("%Y-%m-%d"),
                    "test_start": test_start.strftime("%Y-%m-%d"),
                    "test_end": test_end.strftime("%Y-%m-%d"),
                    "embargo_days": int(embargo_days),
                })
            cursor = _date_add_years(cursor, float(step_years))
        return splits

    total_days = (end - start).days
    window_size = total_days // n_windows

    splits = []
    for i in range(n_windows):
        w_start = start + pd.Timedelta(days=i * window_size)
        w_end = min(w_start + pd.Timedelta(days=window_size), end)

        train_end = w_start + pd.Timedelta(days=int(window_size * train_ratio))
        test_start = train_end + pd.Timedelta(days=max(embargo_days, 1))

        if test_start >= w_end:
            continue

        splits.append({
            "train_start": w_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": w_end.strftime("%Y-%m-%d"),
            "embargo_days": embargo_days,
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

    embargo_days = int(getattr(config, "walk_forward_embargo_days", 21))
    splits = walk_forward_splits(
        config.start_date,
        config.end_date,
        config.walk_forward_windows,
        config.walk_forward_train_ratio,
        embargo_days=embargo_days,
        train_years=float(getattr(config, "train_years", 0) or 0),
        test_years=float(getattr(config, "test_years", 0) or 0),
        step_years=float(getattr(config, "step_years", 0) or 0),
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
                from agents.weight_learning_agent import WeightLearner, build_feature_matrix

                # Build features from train start through test end (rows in train range used for fit only)
                features_df = build_feature_matrix(
                    tickers=tickers or [],
                    start_date=split["train_start"],
                    end_date=split["test_end"],
                    holding_period=getattr(config, "holding_period_days", 5),
                    data_provider=getattr(config, "data_provider", None),
                    cache_dir=getattr(config, "cache_dir", None),
                    cache_ttl_days=getattr(config, "cache_ttl_days", 1),
                    wrds_username=os.environ.get("WRDS_USERNAME"),
                    wrds_ticker_to_permno=getattr(config, "wrds_ticker_to_permno", {}) or {},
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

    # Statistical significance report on OOS results
    oos_sharpes = summary_df["oos_sharpe"].dropna().tolist() if "oos_sharpe" in summary_df.columns else []
    if oos_sharpes:
        try:
            from .metrics import walk_forward_significance_report
            n_days = max(
                int((pd.Timestamp(config.end_date) - pd.Timestamp(config.start_date)).days
                    / max(len(splits), 1) * (1 - getattr(config, "walk_forward_train_ratio", 0.7))),
                20,
            )
            sig_report = walk_forward_significance_report(
                oos_sharpes=oos_sharpes,
                n_days_per_window=n_days,
                n_total_trials=len(splits),
            )
            print(f"\n{'='*65}")
            print("  WALK-FORWARD STATISTICAL SIGNIFICANCE REPORT")
            print(f"{'='*65}")
            print(f"  {sig_report.get('summary', '')}")
            print(f"  Mean OOS Sharpe   : {sig_report.get('mean_oos_sharpe', 0):.4f}")
            print(f"  Positive fraction : {sig_report.get('positive_fraction', 0):.1%}")
            print(f"  Deflated SR (DSR) : {sig_report.get('deflated_sharpe_ratio', 0):.4f}  "
                  f"({'CREDIBLE ✓' if sig_report.get('dsr_is_credible') else 'SUSPECT ✗'})")
            print(f"  BHY significant   : {sig_report.get('bhy_n_significant', 0)}/{len(oos_sharpes)} windows")
            print(f"  Bonferroni α      : {sig_report.get('bonferroni_corrected_alpha', 0):.5f}")
            print(f"{'='*65}")
        except Exception as exc:
            print(f"\n  [WARN] Statistical significance report failed: {exc}")

    return results, summary_df


# ------------------------------------------------------------------
# Walk-forward OOS equity stitch
# ------------------------------------------------------------------


def run_oos_stitch(
    config,
    tickers: list[str] | None = None,
    *,
    report_path: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Honest OOS backtest: stitch equity from non-overlapping test windows.

    The standard single-run backtest (2008–2022) is in-sample because every
    parameter choice was informed by that period's results.  This function
    evaluates the SAME strategy on held-out test slices:

    Walk-forward protocol:
      • Uses research.train_years / test_years / step_years from config
        (default: train=7y, test=1y, step=1y → 8 windows, 2015–2022)
      • Each window: run backtester on test period only (train split not used
        for tuning here — strategy is rules-based, no ML retrain)
      • Chain equity curves: window i+1 starts from where window i ended
      • Report Sharpe/CAGR from the stitched curve

    Returns
    -------
    (stitched_daily_equity : pd.DataFrame, metrics_dict : dict)
        stitched_daily_equity has columns ["equity", "daily_return"]
        metrics_dict has "sharpe", "cagr", "max_drawdown", "n_windows"
    """
    import copy
    import os

    from .backtester import Backtester

    embargo_days = int(getattr(config, "walk_forward_embargo_days", 21))
    train_y = float(getattr(config, "train_years", 7) or 7)
    test_y = float(getattr(config, "test_years", 1) or 1)
    step_y = float(getattr(config, "step_years", 1) or 1)
    n_windows = int(getattr(config, "walk_forward_windows", 8) or 8)
    train_ratio = float(getattr(config, "walk_forward_train_ratio", 0.7) or 0.7)

    splits = walk_forward_splits(
        config.start_date,
        config.end_date,
        n_windows=n_windows,
        train_ratio=train_ratio,
        embargo_days=embargo_days,
        train_years=train_y,
        test_years=test_y,
        step_years=step_y,
    )

    if not splits:
        print("[OOS Stitch] No walk-forward windows generated — check config dates.")
        return pd.DataFrame(), {}

    report_path = report_path or getattr(
        config, "walk_forward_report_path",
        "output/backtests/oos_stitch_report.csv",
    )

    all_equity_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    chain_start_equity = float(getattr(config, "initial_capital", 100_000.0))

    print("\n" + "=" * 65)
    print("  WALK-FORWARD OOS STITCH  (honest out-of-sample evaluation)")
    print("=" * 65)

    for i, split in enumerate(splits, 1):
        print(f"\n  Window {i}/{len(splits)}: "
              f"train {split['train_start']} → {split['train_end']} | "
              f"OOS  {split['test_start']} → {split['test_end']}")

        window_cfg = copy.deepcopy(config)
        window_cfg.start_date = split["test_start"]
        window_cfg.end_date = split["test_end"]
        # Scale initial capital to where previous window ended (continuous curve)
        window_cfg.initial_capital = chain_start_equity

        bt = Backtester(window_cfg)
        result = bt.run(tickers)

        eq_df = getattr(result, "daily_equity", None)
        if eq_df is None or eq_df.empty:
            print(f"    [WARN] No equity data for window {i} — skipping.")
            continue

        if "equity" not in eq_df.columns:
            # Try common alternative names
            for col in ("portfolio_value", "total_equity", "value"):
                if col in eq_df.columns:
                    eq_df = eq_df.rename(columns={col: "equity"})
                    break

        if "equity" not in eq_df.columns:
            print(f"    [WARN] Window {i}: equity column not found in daily_equity.")
            continue

        # Scale equity to chain properly
        first_equity = float(eq_df["equity"].iloc[0])
        if first_equity > 0:
            eq_df = eq_df.copy()
            eq_df["equity"] = eq_df["equity"] * (chain_start_equity / first_equity)

        chain_start_equity = float(eq_df["equity"].iloc[-1])
        all_equity_frames.append(eq_df[["equity"]])

        m = result.metrics
        win_sharpe = m.get("sharpe_ratio", 0.0)
        win_cagr = m.get("cagr", m.get("annualized_return", 0.0))
        win_dd = m.get("max_drawdown", 0.0)
        print(f"    OOS Sharpe={win_sharpe:.3f}  CAGR={win_cagr:.1%}  MaxDD={win_dd:.1%}")

        summary_rows.append({
            "window": i,
            "train_start": split["train_start"],
            "train_end": split["train_end"],
            "test_start": split["test_start"],
            "test_end": split["test_end"],
            "oos_sharpe": win_sharpe,
            "oos_cagr": win_cagr,
            "oos_max_drawdown": win_dd,
            "oos_total_trades": m.get("total_trades", 0),
        })

    if not all_equity_frames:
        print("[OOS Stitch] No windows completed successfully.")
        return pd.DataFrame(), {}

    # ── Stitch equity curves ─────────────────────────────────────────
    stitched = pd.concat(all_equity_frames, axis=0)
    stitched = stitched[~stitched.index.duplicated(keep="last")].sort_index()
    stitched["daily_return"] = stitched["equity"].pct_change(fill_method=None)

    # ── Compute aggregate metrics ────────────────────────────────────
    daily_rets = stitched["daily_return"].dropna()
    n_days = len(daily_rets)

    sharpe = 0.0
    cagr = 0.0
    max_dd = 0.0

    if n_days >= 20:
        ann_ret = float(daily_rets.mean()) * 252
        ann_vol = float(daily_rets.std()) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 1e-9 else 0.0
        total_return = float(stitched["equity"].iloc[-1] / stitched["equity"].iloc[0]) - 1.0
        years = n_days / 252.0
        cagr = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0
        rolling_max = stitched["equity"].cummax()
        dd_series = (stitched["equity"] - rolling_max) / rolling_max
        max_dd = float(dd_series.min())

    metrics = {
        "sharpe": round(sharpe, 4),
        "cagr": round(cagr, 4),
        "max_drawdown": round(max_dd, 4),
        "n_windows": len(summary_rows),
        "n_oos_days": n_days,
    }

    print("\n" + "=" * 65)
    print("  OOS STITCH RESULTS  (honest out-of-sample)")
    print("=" * 65)
    print(f"  Windows         : {metrics['n_windows']}")
    print(f"  OOS Days        : {metrics['n_oos_days']}")
    print(f"  Stitched Sharpe : {metrics['sharpe']:.4f}")
    print(f"  Stitched CAGR   : {metrics['cagr']:.2%}")
    print(f"  Max Drawdown    : {metrics['max_drawdown']:.2%}")
    print("=" * 65)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        summary_df.to_csv(report_path, index=False)
        print(f"\n  OOS report saved → {report_path}")

    # Save stitched equity curve
    stitched_path = os.path.join(
        os.path.dirname(report_path) or "output/backtests",
        "oos_stitched_equity.csv",
    )
    stitched.to_csv(stitched_path)
    print(f"  Stitched equity  → {stitched_path}\n")

    return stitched, metrics


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
    return [dict(zip(keys, combo, strict=False)) for combo in itertools.product(*param_ranges.values())]


def run_parameter_sweep(
    base_config,
    param_ranges: dict[str, list],
    tickers: list[str] | None = None,
    target_metric: str = "sharpe_ratio",
    price_data: dict[str, pd.DataFrame] | None = None,
    signal_data: dict[str, pd.DataFrame] | None = None,
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
        result = bt.run(tickers, price_data=price_data, signal_data=signal_data)
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
    price_data: dict[str, pd.DataFrame] | None = None,
    signal_data: dict[str, pd.DataFrame] | None = None,
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
        result = bt.run(tickers, price_data=price_data, signal_data=signal_data)

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
    price_data: dict[str, pd.DataFrame] | None = None,
    signal_data: dict[str, pd.DataFrame] | None = None,
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
        result = bt.run(tickers, price_data=price_data, signal_data=signal_data)
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


# ------------------------------------------------------------------
# Returns matrix / vol helpers (extracted from Backtester)
# ------------------------------------------------------------------

def build_returns_matrix(
    price_data: dict,
    as_of_date: "pd.Timestamp",
    lookback_days: int,
) -> "pd.DataFrame | None":
    """
    Build DataFrame of daily returns for all tickers up to *as_of_date*.

    Uses only past data (no lookahead). Returns None if insufficient data.
    Extracted from ``Backtester._build_returns_matrix`` so it can be used
    independently (e.g. in tests and Monte Carlo runners).
    """
    series_list = []
    for ticker, df in price_data.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        price_col = "Close" if "Close" in df.columns else "AdjClose"
        past = df[price_col].loc[df.index <= as_of_date].tail(lookback_days + 1)
        if len(past) < 2:
            continue
        rets = past.pct_change().dropna()
        if len(rets) < 2:
            continue
        rets.name = ticker
        series_list.append(rets)
    if not series_list:
        return None
    out = pd.concat(series_list, axis=1, join="inner")
    if out.empty or out.shape[0] < 2 or out.shape[1] < 1:
        return None
    out = out.tail(lookback_days)
    if len(out) < 2:
        return None
    return out


def annualized_vol(
    price_data: dict,
    ticker: str,
    as_of_date: "pd.Timestamp",
    lookback_days: int = 20,
) -> "float | None":
    """
    Return annualized volatility (std of daily returns) for *ticker* as of
    *as_of_date*, or None if insufficient data.

    Extracted from ``Backtester._annualized_vol``.
    """
    if ticker not in price_data:
        return None
    df = price_data[ticker]
    if "Close" not in df.columns or df.empty:
        return None
    price_col = "Close" if "Close" in df.columns else "AdjClose"
    series = df[price_col].loc[df.index <= as_of_date].tail(lookback_days + 1)
    if len(series) < 2:
        return None
    rets = series.pct_change().dropna()
    if len(rets) < 2:
        return None
    return float(rets.std() * (252 ** 0.5))
