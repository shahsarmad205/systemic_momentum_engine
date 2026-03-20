"""
Backtest Performance Metrics
==============================
All standard quantitative performance measures for evaluating the
backtest, plus an aggregation helper.

Expected trade DataFrame columns:
    ticker, signal, direction, signal_date, entry_date, exit_date,
    entry_price, exit_price, position_size, shares, return, pnl,
    adjusted_score, confidence, regime, entry_cost, exit_cost, holding_days
"""

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Win rate
# ------------------------------------------------------------------

def compute_win_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    return float((trades["return"] > 0).sum() / len(trades))


# ------------------------------------------------------------------
# Average return (per-trade %)
# ------------------------------------------------------------------

def compute_average_return(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    return float(trades["return"].mean())


# ------------------------------------------------------------------
# Profit factor (dollar-weighted)
# ------------------------------------------------------------------

def compute_profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    col = "pnl" if "pnl" in trades.columns else "return"
    profits = trades.loc[trades[col] > 0, col].sum()
    losses = trades.loc[trades[col] < 0, col].abs().sum()
    if losses == 0:
        return float("inf") if profits > 0 else 0.0
    return float(profits / losses)


# ------------------------------------------------------------------
# Sharpe ratio (annualised, from trade-level % returns)
# ------------------------------------------------------------------

def compute_sharpe_ratio(trades: pd.DataFrame, holding_period_days: int = 5) -> float:
    if len(trades) < 2:
        return 0.0
    mean_r = trades["return"].mean()
    std_r = trades["return"].std()
    if std_r == 0:
        return 0.0
    periods = 252.0 / holding_period_days
    return float((mean_r / std_r) * np.sqrt(periods))


def compute_equity_sharpe_ratio(daily_equity: pd.DataFrame, *, ddof: int = 0) -> float:
    """
    Net Sharpe based on daily equity curve.

    Unlike `compute_sharpe_ratio()` (which uses trade-level % returns and
    excludes explicit transaction costs from the Sharpe input), this uses
    mark-to-market equity changes which already include execution costs.
    """
    if daily_equity is None or daily_equity.empty or "equity" not in daily_equity.columns:
        return 0.0

    eq = daily_equity
    if "date" in eq.columns:
        eq = eq.sort_values("date")

    equity_series = pd.to_numeric(eq["equity"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(equity_series) < 3:
        return 0.0

    rets = equity_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(rets) < 10:
        return 0.0

    mean_r = float(rets.mean())
    std_r = float(rets.std(ddof=ddof))
    if std_r == 0.0 or not np.isfinite(std_r) or not np.isfinite(mean_r):
        return 0.0

    sharpe = (mean_r / std_r) * np.sqrt(252.0)
    return float(sharpe) if np.isfinite(sharpe) else 0.0


# ------------------------------------------------------------------
# Sortino ratio (annualised, downside deviation)
# ------------------------------------------------------------------

def compute_sortino_ratio(trades: pd.DataFrame, holding_period_days: int = 5) -> float:
    if len(trades) < 2:
        return 0.0
    mean_r = trades["return"].mean()
    downside = trades.loc[trades["return"] < 0, "return"]
    if downside.empty:
        return float("inf") if mean_r > 0 else 0.0
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std == 0:
        return 0.0
    periods = 252.0 / holding_period_days
    return float((mean_r / downside_std) * np.sqrt(periods))


# ------------------------------------------------------------------
# Max drawdown (from daily equity series)
# ------------------------------------------------------------------

def compute_max_drawdown(daily_equity: pd.DataFrame) -> float:
    """Largest peak-to-trough decline.  Returns a negative number."""
    if daily_equity.empty or "equity" not in daily_equity.columns:
        return 0.0
    equity = daily_equity["equity"]
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    return float(dd.min())


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    """
    Helper: compute max drawdown from a series of returns (no dates needed).
    Returns a negative number.
    """
    if returns.empty:
        return 0.0
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def compute_drawdown_stats(daily_equity: pd.DataFrame) -> dict:
    """
    Compute drawdown depth and duration statistics from the equity curve.

    Returns
    -------
    dict with keys:
        max_drawdown_duration : longest peak-to-recovery period in days
        avg_drawdown          : average depth of individual drawdowns (negative number)
        avg_drawdown_duration : average length of drawdown periods in days
    """
    if daily_equity.empty or "equity" not in daily_equity.columns or "date" not in daily_equity.columns:
        return {
            "max_drawdown_duration": 0,
            "avg_drawdown": 0.0,
            "avg_drawdown_duration": 0.0,
        }

    equity = daily_equity["equity"].reset_index(drop=True)
    dates = pd.to_datetime(daily_equity["date"]).reset_index(drop=True)

    peak = equity.cummax()
    dd = (equity - peak) / peak  # <= 0 during drawdowns

    in_dd = False
    start_idx = None
    depths = []
    durations = []

    for i in range(len(dd)):
        if not in_dd:
            # Start of a new drawdown
            if dd.iloc[i] < 0:
                in_dd = True
                start_idx = i
        else:
            # End of drawdown when we recover to a new high (dd back to 0)
            if dd.iloc[i] == 0:
                end_idx = i
                segment = dd.iloc[start_idx:end_idx]
                if not segment.empty:
                    depths.append(float(segment.min()))
                    durations.append((dates.iloc[end_idx - 1] - dates.iloc[start_idx]).days)
                in_dd = False
                start_idx = None

    # If we finish still in drawdown, treat the last date as end
    if in_dd and start_idx is not None and start_idx < len(dd):
        segment = dd.iloc[start_idx:]
        if not segment.empty:
            depths.append(float(segment.min()))
            durations.append((dates.iloc[len(dd) - 1] - dates.iloc[start_idx]).days)

    if not depths or not durations:
        return {
            "max_drawdown_duration": 0,
            "avg_drawdown": 0.0,
            "avg_drawdown_duration": 0.0,
        }

    max_dd_duration = int(max(durations))
    avg_dd = float(np.mean(depths))  # negative on average
    avg_dd_duration = float(np.mean(durations))

    return {
        "max_drawdown_duration": max_dd_duration,
        "avg_drawdown": avg_dd,
        "avg_drawdown_duration": avg_dd_duration,
    }


def bootstrap_performance_cis(
    daily_returns: pd.Series,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Bootstrap confidence intervals for Sharpe, Sortino, and Calmar ratios
    using daily returns.

    Parameters
    ----------
    daily_returns : pd.Series
        Series of daily portfolio returns (e.g., from equity pct_change()).
    n_bootstrap : int
        Number of bootstrap resamples (default: 1000).

    Returns
    -------
    dict with keys:
        sharpe_ci_low, sharpe_ci_high,
        sortino_ci_low, sortino_ci_high,
        calmar_ci_low, calmar_ci_high
    """
    daily_returns = daily_returns.dropna()
    n = len(daily_returns)
    if n < 5:
        return {
            "sharpe_ci_low": 0.0,
            "sharpe_ci_high": 0.0,
            "sortino_ci_low": 0.0,
            "sortino_ci_high": 0.0,
            "calmar_ci_low": 0.0,
            "calmar_ci_high": 0.0,
        }

    sharpe_vals = []
    sortino_vals = []
    calmar_vals = []

    r = daily_returns.values

    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        sample = r[idx]
        sample_s = pd.Series(sample)

        mean_r = float(sample_s.mean())
        std_r = float(sample_s.std(ddof=0))
        if std_r > 0:
            sharpe = (mean_r / std_r) * np.sqrt(252.0)
        else:
            sharpe = 0.0
        sharpe_vals.append(sharpe)

        downside = sample_s[sample_s < 0]
        if not downside.empty:
            downside_std = float(np.sqrt((downside ** 2).mean()))
            sortino = (mean_r / downside_std) * np.sqrt(252.0) if downside_std > 0 else 0.0
        else:
            sortino = 0.0
        sortino_vals.append(sortino)

        # Calmar from bootstrap equity curve
        max_dd = abs(_max_drawdown_from_returns(sample_s))
        if n > 0:
            equity_end = float((1.0 + sample_s).prod())
            # Annualise over number of days represented by the sample
            ann_return = equity_end ** (252.0 / n) - 1.0
        else:
            ann_return = 0.0
        if max_dd > 0:
            calmar = ann_return / max_dd
        else:
            calmar = 0.0
        calmar_vals.append(calmar)

    def _ci(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return 0.0, 0.0
        low, high = np.percentile(vals, [2.5, 97.5])
        return float(low), float(high)

    s_low, s_high = _ci(sharpe_vals)
    so_low, so_high = _ci(sortino_vals)
    c_low, c_high = _ci(calmar_vals)

    return {
        "sharpe_ci_low": s_low,
        "sharpe_ci_high": s_high,
        "sortino_ci_low": so_low,
        "sortino_ci_high": so_high,
        "calmar_ci_low": c_low,
        "calmar_ci_high": c_high,
    }


# ------------------------------------------------------------------
# Calmar ratio (annualised return / |max drawdown|)
# ------------------------------------------------------------------

def compute_calmar_ratio(
    daily_equity: pd.DataFrame,
    initial_capital: float = 100_000,
) -> float:
    if daily_equity.empty:
        return 0.0
    equity = daily_equity["equity"]
    final = equity.iloc[-1]
    n_days = (daily_equity["date"].iloc[-1] - daily_equity["date"].iloc[0]).days
    if n_days <= 0:
        return 0.0

    ann_return = (final / initial_capital) ** (365.25 / n_days) - 1

    peak = equity.expanding().max()
    max_dd = abs(((equity - peak) / peak).min())
    if max_dd == 0:
        return float("inf") if ann_return > 0 else 0.0
    return float(ann_return / max_dd)


# ------------------------------------------------------------------
# Signal accuracy (directional)
# ------------------------------------------------------------------

def compute_signal_accuracy(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    correct = (
        ((trades["signal"] == "Bullish") & (trades["exit_price"] > trades["entry_price"]))
        | ((trades["signal"] == "Bearish") & (trades["exit_price"] < trades["entry_price"]))
    ).sum()
    return float(correct / len(trades))


# ------------------------------------------------------------------
# Information Coefficient (Pearson)
# ------------------------------------------------------------------

def compute_information_coefficient(trades: pd.DataFrame) -> float:
    if len(trades) < 3 or "adjusted_score" not in trades.columns:
        return 0.0
    ic = trades["adjusted_score"].corr(trades["return"])
    return 0.0 if pd.isna(ic) else float(ic)


# ------------------------------------------------------------------
# Rank IC (Spearman)
# ------------------------------------------------------------------

def compute_rank_ic(trades: pd.DataFrame) -> float:
    if len(trades) < 3 or "adjusted_score" not in trades.columns:
        return 0.0
    ic = trades["adjusted_score"].corr(trades["return"], method="spearman")
    return 0.0 if pd.isna(ic) else float(ic)


# ------------------------------------------------------------------
# Trade duration statistics
# ------------------------------------------------------------------

def compute_trade_duration_stats(trades: pd.DataFrame) -> dict:
    if trades.empty or "holding_days" not in trades.columns:
        return {"mean": 0, "median": 0, "min": 0, "max": 0}
    hd = trades["holding_days"]
    return {
        "mean": round(float(hd.mean()), 1),
        "median": round(float(hd.median()), 1),
        "min": int(hd.min()),
        "max": int(hd.max()),
    }


def compute_threshold_efficiency(signal_df: pd.DataFrame) -> dict:
    """
    Measure how often signals clear the dynamic threshold vs the static 0.5.

    Returns
    -------
    dict with keys:
        signals_fired_dynamic : int   — signals generated with dynamic threshold
        signals_fired_static  : int   — signals that would fire at static 0.5
        threshold_tightening  : float — mean increase in threshold vs 0.5 baseline
        estimated_cost_saving_bps: float — trades avoided × avg round trip cost
    """
    if signal_df is None or signal_df.empty:
        return {
            "signals_fired_dynamic": 0,
            "signals_fired_static": 0,
            "threshold_tightening": 0.0,
            "estimated_cost_saving_bps": 0.0,
        }

    score_col = "smoothed_score" if "smoothed_score" in signal_df.columns else "adjusted_score"

    dynamic_mask = signal_df["signal"].isin(["Bullish", "Bearish"])
    signals_fired_dynamic = int(dynamic_mask.sum())

    static_bull = signal_df[score_col] > 0.5
    static_bear = signal_df[score_col] < -0.5
    signals_fired_static = int((static_bull | static_bear).sum())

    if "bull_threshold" in signal_df.columns:
        bt = signal_df["bull_threshold"]
        threshold_tightening = float((bt - 0.5).mean())
    else:
        threshold_tightening = 0.0

    avoided = max(0, signals_fired_static - signals_fired_dynamic)
    avg_round_trip_cost_bps = float(signal_df.attrs.get("avg_round_trip_cost_bps", 0.0))
    estimated_cost_saving_bps = float(avoided * avg_round_trip_cost_bps)

    return {
        "signals_fired_dynamic": signals_fired_dynamic,
        "signals_fired_static": signals_fired_static,
        "threshold_tightening": threshold_tightening,
        "estimated_cost_saving_bps": estimated_cost_saving_bps,
    }


def compute_turnover(trades: pd.DataFrame, initial_capital: float, avg_cost_bps: float) -> dict:
    """
    Compute portfolio turnover and its approximate cost impact.

    Definitions
    -----------
    - Daily traded notional = total dollar value traded on that day (buys + sells),
      approximated from entries and exits in `trades`.
    - Daily turnover       = daily_traded_notional / average AUM
                             (we approximate AUM with initial_capital here).
    - Annualised turnover  = avg_daily_turnover × 252.
    - Cost impact          = annualised_turnover × avg_cost_per_trade (given in bps).
    """
    if trades.empty or "position_size" not in trades.columns or initial_capital <= 0:
        return {
            "avg_daily_turnover": 0.0,
            "annualised_turnover": 0.0,
            "turnover_cost_drag_bps": 0.0,
        }

    df = trades.copy()

    # Approximate total value traded per day:
    #   buys  = sum(position_size) on entry_date
    #   sells = sum(position_size) on exit_date
    entry_notional = df.groupby("entry_date")["position_size"].sum().abs()
    exit_notional = df.groupby("exit_date")["position_size"].sum().abs()
    daily_traded = entry_notional.add(exit_notional, fill_value=0.0)

    if daily_traded.empty:
        avg_daily_turnover = 0.0
    else:
        avg_daily_turnover = float((daily_traded / float(initial_capital)).mean())

    annualised_turnover = avg_daily_turnover * 252.0
    turnover_cost_drag_bps = annualised_turnover * float(avg_cost_bps)

    return {
        "avg_daily_turnover": avg_daily_turnover,
        "annualised_turnover": annualised_turnover,
        "turnover_cost_drag_bps": turnover_cost_drag_bps,
    }


def compute_turnover_corrected(
    trades: pd.DataFrame,
    daily_equity: pd.DataFrame,
    config,
) -> dict:
    """
    Corrected turnover estimate based on weight changes:
      turnover ≈ sum(abs(weight_changes)) / 2 per rebalance, annualised.

    We don't have explicit per-rebalance weight vectors, so we approximate the
    weight changes for each trade using the implied entry/exit weights from
    the daily equity curve:
      w_entry = abs(position_size) / equity(entry_date)
      w_exit  = abs(position_size * (1 + return)) / equity(exit_date)
    Then each trade contributes:
      turnover_trade = (abs(w_entry) + abs(w_exit)) / 2
    and annualised turnover is sum(turnover_trade) / years.
    """
    out = {
        "annualised_turnover_corrected": 0.0,
        "avg_daily_turnover_corrected": 0.0,
        "turnover_cost_drag_bps_corrected": 0.0,
        # Debug / cross-checks requested by user
        "turnover_total_shares_traded": 0.0,
        "turnover_avg_trade_price": 0.0,
        "turnover_avg_portfolio_value": 0.0,
        "turnover_shares_based": 0.0,
        "turnover_position_changes_total": 0.0,
        "turnover_avg_positions_held": 0.0,
        "turnover_changes_over_avg_positions": 0.0,
        "turnover_trades_per_year_x_avg_position_size_based": 0.0,
    }

    if trades is None or trades.empty or daily_equity is None or daily_equity.empty:
        return out

    if "position_size" not in trades.columns or "entry_date" not in trades.columns or "exit_date" not in trades.columns:
        return out

    # Years in the backtest window (calendar-day approximation)
    try:
        start = pd.to_datetime(getattr(config, "start_date", None) or getattr(config, "start", "1970-01-01"))
        end = pd.to_datetime(getattr(config, "end_date", None) or getattr(config, "end", "1970-01-02"))
        years = max(float((end - start).days) / 365.25, 1e-6)
    except Exception:
        years = 1.0

    eq = daily_equity.copy()
    if "date" in eq.columns:
        eq = eq.sort_values("date").set_index("date")
    elif eq.index.name is not None:
        eq = eq.sort_index()
    else:
        eq = eq.sort_index()

    if "equity" not in eq.columns:
        return out

    eq_series = eq["equity"]
    # Average portfolio value for the share-based cross-check
    avg_portfolio_value = float(pd.to_numeric(eq_series, errors="coerce").dropna().mean())
    out["turnover_avg_portfolio_value"] = avg_portfolio_value

    if avg_portfolio_value <= 0:
        return out

    # Entry/exit weights
    # - trades['return'] is per-trade return fraction; for equity-at-exit we use (1 + return)
    if "return" not in trades.columns:
        return out

    # Ensure keys are timestamp-like to match daily_equity['date']
    entry_dates = pd.to_datetime(trades["entry_date"], errors="coerce")
    exit_dates = pd.to_datetime(trades["exit_date"], errors="coerce")
    returns = pd.to_numeric(trades["return"], errors="coerce").fillna(0.0)
    pos_sizes = pd.to_numeric(trades["position_size"], errors="coerce").abs().fillna(0.0)

    valid = (pos_sizes > 0) & entry_dates.notna() & exit_dates.notna()
    if not bool(valid.any()):
        return out

    trades_valid_idx = valid[valid].index
    entry_eq = eq_series.reindex(entry_dates.loc[trades_valid_idx]).values
    exit_eq = eq_series.reindex(exit_dates.loc[trades_valid_idx]).values

    # Drop trades where the daily equity at entry/exit isn't available
    w_entry = []
    w_exit = []
    for idx, ee, xe, ps, r in zip(
        trades_valid_idx,
        entry_eq,
        exit_eq,
        pos_sizes.loc[trades_valid_idx].values,
        returns.loc[trades_valid_idx].values,
    ):
        if ee is None or xe is None:
            continue
        if not (pd.notna(ee) and pd.notna(xe)):
            continue
        ee_f = float(ee)
        xe_f = float(xe)
        if ee_f <= 0 or xe_f <= 0:
            continue
        we = float(ps) / ee_f
        # Market value at exit (approx): position_size grows/shrinks by (1+return)
        mv_exit_abs = float(ps) * (1.0 + float(r))
        if mv_exit_abs < 0:
            mv_exit_abs = abs(mv_exit_abs)
        wx = mv_exit_abs / xe_f
        w_entry.append(we)
        w_exit.append(wx)

    if not w_entry:
        return out

    turnover_trade = (np.array(w_entry) + np.array(w_exit)) / 2.0
    annualised_turnover_corrected = float(turnover_trade.sum() / years)

    out["annualised_turnover_corrected"] = annualised_turnover_corrected
    out["avg_daily_turnover_corrected"] = annualised_turnover_corrected / 252.0

    # Cost drag: infer *effective* total bps/leg from observed total costs.
    total_transaction_costs = 0.0
    if "total_cost" in trades.columns:
        total_transaction_costs = float(pd.to_numeric(trades["total_cost"], errors="coerce").fillna(0.0).sum())

    sum_abs_pos = float(pos_sizes.sum())
    if sum_abs_pos > 0 and total_transaction_costs > 0:
        # total_transaction_costs = 2 * sum_abs_pos * (effective_total_bps_per_leg / 10_000)
        effective_total_bps_per_leg = total_transaction_costs * 10_000.0 / (2.0 * sum_abs_pos)
        # round-trip cost fraction = 2*effective_bps/10_000; convert to bps by *10_000
        out["turnover_cost_drag_bps_corrected"] = annualised_turnover_corrected * (2.0 * effective_total_bps_per_leg)

    # --- Cross-checks requested by user ---
    # 1) total_shares_traded * avg_price per year / avg_portfolio_value
    if "shares" in trades.columns:
        total_shares_traded = float(pd.to_numeric(trades["shares"], errors="coerce").abs().fillna(0.0).sum())
        out["turnover_total_shares_traded"] = total_shares_traded
        avg_trade_price = float(pd.to_numeric(trades["entry_price"], errors="coerce").fillna(0.0).mean()) if "entry_price" in trades.columns else 0.0
        out["turnover_avg_trade_price"] = avg_trade_price
        if years > 0 and avg_trade_price > 0 and avg_portfolio_value > 0:
            traded_notional_per_year = total_shares_traded * avg_trade_price / years
            out["turnover_shares_based"] = traded_notional_per_year / avg_portfolio_value

    # 2) number_of_position_changes / avg_positions_held
    if "n_positions" in daily_equity.columns:
        avg_positions_held = float(pd.to_numeric(daily_equity["n_positions"], errors="coerce").fillna(0.0).mean())
        out["turnover_avg_positions_held"] = avg_positions_held
    else:
        avg_positions_held = float(len(w_entry))  # fallback; shouldn't happen
        out["turnover_avg_positions_held"] = avg_positions_held

    number_of_position_changes = float(len(trades) * 2)  # entry + exit per trade
    out["turnover_position_changes_total"] = number_of_position_changes
    if avg_positions_held > 0:
        out["turnover_changes_over_avg_positions"] = number_of_position_changes / avg_positions_held

    # 3) trades_per_year * avg_position_size (cross-check)
    trades_per_year = float(len(trades) / years) if years > 0 else 0.0
    avg_position_size = float(pos_sizes.mean()) if len(pos_sizes) else 0.0
    out["turnover_trades_per_year_x_avg_position_size_based"] = (
        trades_per_year * avg_position_size / avg_portfolio_value if avg_portfolio_value > 0 else 0.0
    )

    return out


# ------------------------------------------------------------------
# CAPM metrics (portfolio vs SPY)
# ------------------------------------------------------------------

def compute_capm_metrics(
    daily_equity: pd.DataFrame,
    spy_returns: pd.Series,
    risk_free_rate: float = 0.04,
) -> dict:
    """
    Portfolio beta, Jensen's alpha (annualised), Treynor ratio, Information ratio.

    Parameters
    ----------
    daily_equity : pd.DataFrame
        Must have 'date' and 'equity' columns (or index as date).
    spy_returns : pd.Series
        Daily returns of SPY, index = date.
    risk_free_rate : float
        Annual risk-free rate (default 0.04).

    Returns
    -------
    dict with portfolio_beta, portfolio_alpha_annual, treynor_ratio, information_ratio.
    """
    out = {
        "portfolio_beta": np.nan,
        "portfolio_alpha_annual": np.nan,
        "treynor_ratio": np.nan,
        "information_ratio": np.nan,
    }
    if daily_equity.empty or spy_returns.empty:
        return out
    eq = daily_equity.sort_values("date") if "date" in daily_equity.columns else daily_equity
    if "equity" not in eq.columns:
        return out
    port_ret = eq.set_index("date")["equity"].pct_change().dropna()
    spy_aligned = spy_returns.reindex(port_ret.index).ffill().bfill().fillna(0.0)
    common = port_ret.index.intersection(spy_aligned.index)
    if len(common) < 20:
        return out
    r_p = port_ret.loc[common].astype(float)
    r_m = spy_aligned.loc[common].astype(float)
    var_m = r_m.var()
    if var_m <= 1e-12:
        return out
    beta = float(r_p.cov(r_m) / var_m)
    alpha_daily = float(r_p.mean() - beta * r_m.mean())
    alpha_annual = alpha_daily * 252
    out["portfolio_beta"] = beta
    out["portfolio_alpha_annual"] = alpha_annual
    # Treynor: (portfolio_return_annual - rf) / beta
    port_ret_annual = float(r_p.mean() * 252)
    out["treynor_ratio"] = (port_ret_annual - risk_free_rate) / beta if beta != 0 else np.nan
    # Information ratio: alpha / tracking_error (annualised)
    tracking_err_daily = (r_p - r_m).std()
    tracking_err_annual = float(tracking_err_daily * np.sqrt(252)) if pd.notna(tracking_err_daily) and tracking_err_daily > 1e-12 else np.nan
    out["information_ratio"] = alpha_annual / tracking_err_annual if tracking_err_annual and np.isfinite(tracking_err_annual) else np.nan
    return out


# ------------------------------------------------------------------
# Aggregate helper
# ------------------------------------------------------------------

def compute_all_metrics(
    trades: pd.DataFrame,
    daily_equity: pd.DataFrame,
    config,
) -> dict:
    """Compute every metric and return a flat dict."""
    m: dict = {}

    m["total_trades"] = len(trades)
    m["bullish_trades"] = int((trades["signal"] == "Bullish").sum()) if not trades.empty else 0
    m["bearish_trades"] = int((trades["signal"] == "Bearish").sum()) if not trades.empty else 0

    m["win_rate"] = compute_win_rate(trades)
    m["average_return"] = compute_average_return(trades)
    m["profit_factor"] = compute_profit_factor(trades)
    # Use actual realised holding days if available; fall back to config.holding_period_days.
    if not trades.empty and "holding_days" in trades.columns:
        eff_holding_days = max(1.0, float(trades["holding_days"].mean()))
    else:
        eff_holding_days = float(getattr(config, "holding_period_days", 5))

    m["sharpe_ratio"] = compute_sharpe_ratio(trades, eff_holding_days)
    # Net Sharpe: computed from daily equity curve (already net of costs).
    m["net_sharpe_ratio"] = compute_equity_sharpe_ratio(daily_equity)
    m["sortino_ratio"] = compute_sortino_ratio(trades, eff_holding_days)
    m["calmar_ratio"] = compute_calmar_ratio(daily_equity, config.initial_capital)
    m["max_drawdown"] = compute_max_drawdown(daily_equity)
    dd_stats = compute_drawdown_stats(daily_equity)
    m["max_drawdown_duration"] = dd_stats["max_drawdown_duration"]
    m["avg_drawdown"] = dd_stats["avg_drawdown"]
    m["avg_drawdown_duration"] = dd_stats["avg_drawdown_duration"]
    m["signal_accuracy"] = compute_signal_accuracy(trades)
    m["information_coefficient"] = compute_information_coefficient(trades)
    m["rank_ic"] = compute_rank_ic(trades)
    m["duration_stats"] = compute_trade_duration_stats(trades)
    to = compute_turnover(trades, config.initial_capital, config.execution_costs_commission_bps)
    # Keep old turnover for comparison/debugging.
    m["avg_daily_turnover_old"] = float(to.get("avg_daily_turnover", 0.0) or 0.0)
    m["annualised_turnover_old"] = float(to.get("annualised_turnover", 0.0) or 0.0)
    m["turnover_cost_drag_bps_old"] = float(to.get("turnover_cost_drag_bps", 0.0) or 0.0)
    m.update(to)

    # Corrected turnover: based on implied weight changes from trade entry/exit weights.
    tc = compute_turnover_corrected(trades, daily_equity, config)
    m.update(tc)
    m["annualised_turnover"] = float(tc.get("annualised_turnover_corrected", 0.0) or 0.0)
    m["avg_daily_turnover"] = float(tc.get("avg_daily_turnover_corrected", 0.0) or 0.0)
    # Replace turnover cost drag estimate with one consistent with observed total costs.
    m["turnover_cost_drag_bps"] = float(tc.get("turnover_cost_drag_bps_corrected", 0.0) or 0.0)

    # Bootstrap confidence intervals for performance metrics (daily returns).
    if not daily_equity.empty and "equity" in daily_equity.columns:
        eq_sorted = daily_equity.sort_values("date") if "date" in daily_equity.columns else daily_equity
        equity_series = eq_sorted["equity"]
        daily_ret = equity_series.pct_change().dropna()
        ci = bootstrap_performance_cis(daily_ret)
        m.update(ci)

    # VaR metrics (Historical 95%/99%, CVaR 95%, 5d scaled, breach count/rate)
    try:
        from risk.var import portfolio_var_report
        var_report = portfolio_var_report(
            daily_equity,
            confidence_levels=[0.95, 0.99],
            holding_period_days=5,
            window=252,
        )
        m["var_95_1d"] = var_report.get("var_95_1d", np.nan)
        m["var_99_1d"] = var_report.get("var_99_1d", np.nan)
        m["cvar_95"] = var_report.get("cvar_95", np.nan)
        m["var_95_5d"] = var_report.get("var_95_5d", np.nan)
        m["var_breach_rate_95"] = var_report.get("var_breach_rate_95", np.nan)
        m["var_breach_count_95"] = var_report.get("var_breach_count_95", 0)
    except ImportError:
        m["var_95_1d"] = np.nan
        m["var_99_1d"] = np.nan
        m["cvar_95"] = np.nan
        m["var_95_5d"] = np.nan
        m["var_breach_rate_95"] = np.nan
        m["var_breach_count_95"] = 0

    if not trades.empty and not daily_equity.empty:
        m["starting_capital"] = config.initial_capital
        m["final_capital"] = round(float(daily_equity["equity"].iloc[-1]), 2)
        m["total_pnl"] = round(float(trades["pnl"].sum()), 2)
        m["total_return"] = (m["final_capital"] - m["starting_capital"]) / m["starting_capital"]
        # Cost-aware metrics (use net_return column if present)
        if "total_cost" in trades.columns:
            m["total_transaction_costs"] = round(float(trades["total_cost"].sum()), 2)
            m["average_cost_per_trade"] = round(
                m["total_transaction_costs"] / len(trades), 2
            ) if len(trades) > 0 else 0.0
        else:
            m["total_transaction_costs"] = 0.0
            m["average_cost_per_trade"] = 0.0
        if "gross_return" in trades.columns and "position_size" in trades.columns:
            gross_pnl = (trades["position_size"] * trades["gross_return"]).sum()
            m["gross_return"] = gross_pnl / m["starting_capital"]
            m["net_return"] = m["total_return"]
        else:
            m["gross_return"] = m["total_return"]
            m["net_return"] = m["total_return"]
    else:
        m["starting_capital"] = config.initial_capital
        m["final_capital"] = config.initial_capital
        m["total_pnl"] = 0.0
        m["total_return"] = 0.0
        m["gross_return"] = 0.0
        m["net_return"] = 0.0
        m["total_transaction_costs"] = 0.0
        m["average_cost_per_trade"] = 0.0

    return m
