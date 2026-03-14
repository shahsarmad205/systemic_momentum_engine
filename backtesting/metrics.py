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
    m.update(to)

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
