"""
Strategy vs Benchmark Report
============================
Institutional-style comparison of the current strategy run vs SPY buy-and-hold
over the same date range.

Outputs:
  - output/backtests/strategy_report.png
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logging.getLogger("matplotlib").setLevel(logging.WARNING)

from matplotlib import pyplot as plt
from trend_signal_engine.utils.market_data import get_ohlcv


@dataclass(frozen=True)
class PerfMetrics:
    cagr: float
    sharpe: float
    max_drawdown: float
    calmar: float
    beta: float | None = None
    alpha_annual: float | None = None


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def compute_cagr(equity: pd.Series, years: float) -> float:
    equity = equity.dropna()
    if equity.empty or years <= 0:
        return 0.0
    e0 = float(equity.iloc[0])
    ef = float(equity.iloc[-1])
    if e0 <= 0:
        return 0.0
    if ef <= 0:
        # If equity can go to zero/negative (shouldn't happen here),
        # clamp CAGR to -100%.
        return -1.0
    return (ef / e0) ** (1.0 / years) - 1.0


def compute_sharpe(daily_returns: pd.Series, annualization: int = 252) -> float:
    r = daily_returns.dropna().astype(float)
    if len(r) < 2:
        return 0.0
    std = float(r.std(ddof=0))
    if std <= 1e-12:
        return 0.0
    return float((r.mean() / std) * np.sqrt(annualization))


def compute_max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna().astype(float)
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    drawdown = eq / peak - 1.0
    return float(drawdown.min())


def compute_capm_beta_alpha(port_daily_ret: pd.Series, spy_daily_ret: pd.Series) -> tuple[float, float]:
    """
    CAPM-like regression metrics (no risk-free adjustment for alpha).
    Matches backtesting.metrics.compute_capm_metrics logic.
    Returns:
      beta, alpha_annual
    """
    common = port_daily_ret.dropna().index.intersection(spy_daily_ret.dropna().index)
    if len(common) < 20:
        return float("nan"), float("nan")
    r_p = port_daily_ret.loc[common].astype(float)
    r_m = spy_daily_ret.loc[common].astype(float)
    var_m = float(r_m.var(ddof=0))
    if var_m <= 1e-12:
        return float("nan"), float("nan")
    beta = float(r_p.cov(r_m) / var_m)
    alpha_daily = float(r_p.mean() - beta * r_m.mean())
    alpha_annual = alpha_daily * 252.0
    return beta, alpha_annual


def compute_yearly_returns_from_equity(equity: pd.Series) -> pd.Series:
    eq = equity.dropna().astype(float)
    if eq.empty:
        return pd.Series(dtype=float)
    by_year = eq.groupby(eq.index.year)
    # Total return per year = last / first - 1
    yearly = by_year.apply(lambda s: float(s.iloc[-1]) / float(s.iloc[0]) - 1.0)
    yearly.index = yearly.index.astype(int)
    return yearly


def compute_monthly_returns_from_equity(equity: pd.Series) -> pd.Series:
    """
    Monthly return based on month-end equity values.
    """
    eq = equity.dropna().astype(float)
    if eq.empty:
        return pd.Series(dtype=float)
    m_eq = eq.resample("ME").last()
    m_ret = m_eq.pct_change().dropna()
    return m_ret


def plot_heatmap(ax, monthly_ret: pd.Series, title: str, outlier_mask: pd.DataFrame | None = None):
    if monthly_ret.empty:
        ax.set_title(title)
        ax.axis("off")
        return

    df = monthly_ret.copy()
    df.index = pd.to_datetime(df.index)
    grid = df.to_frame("ret")
    grid["year"] = grid.index.year
    grid["month"] = grid.index.month
    pivot = grid.pivot(index="year", columns="month", values="ret")

    # Ensure months are 1..12 in order
    pivot = pivot.reindex(columns=list(range(1, 13)))
    years = pivot.index.tolist()
    data = pivot.values.astype(float)

    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 0.0
    if vmax <= 1e-12:
        vmax = 1.0
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="RdYlGn",
        vmin=-vmax,
        vmax=vmax,
        origin="lower",
    )

    ax.set_title(title)
    ax.set_ylabel("Year")
    ax.set_xlabel("Month")
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years])
    ax.set_xticks(range(12))
    ax.set_xticklabels([pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)])

    # Annotate values (and optionally flag outliers)
    for i in range(len(years)):
        for j in range(12):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val*100:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
                if outlier_mask is not None:
                    try:
                        if bool(outlier_mask.iloc[i, j]):
                            # Draw a subtle border around the outlier cell.
                            rect = plt.Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False, edgecolor="red", linewidth=2.0)
                            ax.add_patch(rect)
                    except Exception:
                        pass

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Monthly return")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy-equity-csv", default="output/backtests/daily_equity.csv")
    ap.add_argument("--benchmark-ticker", default="SPY")
    ap.add_argument("--output", default="output/backtests/strategy_report.png")
    ap.add_argument("--risk-free-rate", type=float, default=0.04)
    args = ap.parse_args()

    equity_csv = Path(args.strategy_equity_csv)
    if not equity_csv.exists():
        raise FileNotFoundError(f"Missing strategy equity CSV: {equity_csv}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    strat_eq = pd.read_csv(equity_csv)
    if "date" not in strat_eq.columns or "equity" not in strat_eq.columns:
        raise ValueError("daily_equity.csv must contain `date` and `equity` columns")

    strat_eq["date"] = pd.to_datetime(strat_eq["date"])
    strat_eq["equity"] = pd.to_numeric(strat_eq["equity"], errors="coerce")
    strat_eq = strat_eq.dropna(subset=["equity"]).sort_values("date")
    strat_eq_series = strat_eq.set_index("date")["equity"].astype(float)

    start_dt = strat_eq_series.index.min()
    end_dt = strat_eq_series.index.max()
    years = max((end_dt - start_dt).days / 365.25, 1e-6)

    # Strategy daily returns
    strat_daily_ret = strat_eq_series.pct_change().dropna()

    # Benchmark: SPY buy-and-hold over same date range
    # Add small buffer so Yahoo slicing doesn't miss first/last trading days.
    spy_start = (start_dt - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    spy_end = (end_dt + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    spy_df = get_ohlcv(
        args.benchmark_ticker,
        spy_start,
        spy_end,
        provider="yahoo",
        cache_dir="data/cache/ohlcv",
        use_cache=True,
        cache_ttl_days=0,
    )
    if spy_df.empty or "Close" not in spy_df.columns:
        raise RuntimeError("Failed to load SPY benchmark data")

    spy_df = spy_df.dropna(subset=["Close"]).sort_index()
    spy_df["Close"] = pd.to_numeric(spy_df["Close"], errors="coerce")
    spy_df = spy_df.dropna(subset=["Close"])
    spy_df = spy_df.loc[(spy_df.index >= start_dt) & (spy_df.index <= end_dt)]

    spy_close = spy_df["Close"].astype(float)
    spy_daily_ret = spy_close.pct_change().dropna()

    # Normalize both equity curves to $100k at strategy start
    strat_eq0 = float(strat_eq_series.iloc[0])
    if strat_eq0 <= 0:
        strat_eq0 = 100_000.0
    strat_norm = 100_000.0 * (strat_eq_series / strat_eq0)

    spy_close0 = float(spy_close.iloc[0])
    if spy_close0 <= 0:
        spy_close0 = 1.0
    spy_norm = 100_000.0 * (spy_close / spy_close0)

    # Metrics: strategy
    strat_cagr = compute_cagr(strat_eq_series, years)
    strat_sharpe = compute_sharpe(strat_daily_ret)
    strat_max_dd = compute_max_drawdown(strat_eq_series)
    strat_calmar = strat_cagr / abs(strat_max_dd) if strat_max_dd < 0 else float("inf")

    # Metrics: SPY
    spy_years = years
    spy_daily_ret_aligned = spy_daily_ret
    spy_cagr = compute_cagr(spy_close, spy_years)  # treating close as equity
    spy_sharpe = compute_sharpe(spy_daily_ret_aligned)
    spy_max_dd = compute_max_drawdown(spy_close)  # treating close as equity
    spy_calmar = spy_cagr / abs(spy_max_dd) if spy_max_dd < 0 else float("inf")

    # CAPM beta/alpha (strategy vs SPY only)
    beta, alpha_annual = compute_capm_beta_alpha(strat_daily_ret, spy_daily_ret)

    # Best/Worst year
    strat_yearly = compute_yearly_returns_from_equity(strat_eq_series)
    spy_yearly = compute_yearly_returns_from_equity(spy_close)

    strat_best_year = float(strat_yearly.max()) if not strat_yearly.empty else float("nan")
    strat_worst_year = float(strat_yearly.min()) if not strat_yearly.empty else float("nan")
    spy_best_year = float(spy_yearly.max()) if not spy_yearly.empty else float("nan")
    spy_worst_year = float(spy_yearly.min()) if not spy_yearly.empty else float("nan")

    # Monthly return distribution stats
    strat_monthly = compute_monthly_returns_from_equity(strat_eq_series)
    spy_monthly = compute_monthly_returns_from_equity(spy_close)

    # Outlier month detection (used to flag heatmap cells)
    # Strategy-only: flag unusually large positive months.
    strat_monthly_clean = strat_monthly.dropna().astype(float)
    strat_monthly_mean = float(strat_monthly_clean.mean()) if not strat_monthly_clean.empty else 0.0
    strat_monthly_std = float(strat_monthly_clean.std(ddof=0)) if not strat_monthly_clean.empty else 1.0
    z = (strat_monthly_clean - strat_monthly_mean) / (strat_monthly_std if strat_monthly_std > 1e-12 else 1.0)
    outlier_months = strat_monthly_clean.index[(z >= 3.0) | (strat_monthly_clean >= 0.10)]
    outlier_set = {(d.year, d.month) for d in outlier_months.to_pydatetime().tolist()} if len(outlier_months) else set()

    strat_m_mean = float(strat_monthly.mean()) if not strat_monthly.empty else float("nan")
    strat_m_std = float(strat_monthly.std(ddof=0)) if not strat_monthly.empty else float("nan")
    strat_m_skew = float(strat_monthly.skew()) if not strat_monthly.empty else float("nan")

    spy_m_mean = float(spy_monthly.mean()) if not spy_monthly.empty else float("nan")
    spy_m_std = float(spy_monthly.std(ddof=0)) if not spy_monthly.empty else float("nan")
    spy_m_skew = float(spy_monthly.skew()) if not spy_monthly.empty else float("nan")

    # Rolling 252d Sharpe
    window = 252
    strat_roll_mean = strat_daily_ret.rolling(window).mean()
    strat_roll_std = strat_daily_ret.rolling(window).std(ddof=0)
    strat_roll_sharpe = (strat_roll_mean / strat_roll_std) * np.sqrt(252.0)

    spy_roll_mean = spy_daily_ret.rolling(window).mean()
    spy_roll_std = spy_daily_ret.rolling(window).std(ddof=0)
    spy_roll_sharpe = (spy_roll_mean / spy_roll_std) * np.sqrt(252.0)

    # Plot
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.plot(strat_norm.index, strat_norm.values, label="Strategy", linewidth=2.0)
    ax1.plot(spy_norm.index, spy_norm.values, label=f"{args.benchmark_ticker} Buy&Hold", linewidth=2.0, alpha=0.8)
    ax1.set_title("Equity Curves (Normalized to $100k)")
    ax1.set_ylabel("Normalized Equity ($)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2.plot(strat_roll_sharpe.index, strat_roll_sharpe.values, label="Strategy", linewidth=2.0)
    ax2.plot(spy_roll_sharpe.index, spy_roll_sharpe.values, label=f"{args.benchmark_ticker} (Sharpe)", linewidth=2.0, alpha=0.8)
    ax2.set_title("Rolling 252-day Sharpe Ratio")
    ax2.set_ylabel("Sharpe")
    ax2.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    # Build outlier mask aligned to heatmap grid.
    outlier_mask = None
    if outlier_set:
        # Convert monthly_ret to pivot like plot_heatmap does.
        df = strat_monthly.copy()
        df.index = pd.to_datetime(df.index)
        grid = df.to_frame("ret")
        grid["year"] = grid.index.year
        grid["month"] = grid.index.month
        pivot = grid.pivot(index="year", columns="month", values="ret")
        pivot = pivot.reindex(columns=list(range(1, 13)))
        year_labels = pivot.index.tolist()
        mask_data = np.zeros((len(year_labels), 12), dtype=bool)
        for i, y in enumerate(year_labels):
            for j, m in enumerate(range(1, 13)):
                if (int(y), int(m)) in outlier_set:
                    mask_data[i, j] = True
        outlier_mask = pd.DataFrame(mask_data, index=year_labels, columns=list(range(1, 13)))

    plot_heatmap(ax3, strat_monthly, "Monthly Returns Heatmap — Strategy", outlier_mask=outlier_mask)
    plot_heatmap(ax4, spy_monthly, f"Monthly Returns Heatmap — {args.benchmark_ticker}")

    fig.suptitle(
        f"Strategy vs Benchmark Report | {start_dt.date()} → {end_dt.date()}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    # Print metrics
    print("\n================ Strategy vs SPY Comparison ================")
    print(f"Period used: {start_dt.date()} → {end_dt.date()} ({years:.2f} years)")
    print()
    print("Strategy:")
    print(f"  CAGR            : {strat_cagr:.2%}")
    print(f"  Sharpe (ann.)  : {strat_sharpe:.2f}")
    print(f"  Max Drawdown   : {strat_max_dd:.2%}")
    print(f"  Calmar         : {strat_calmar:.2f}")
    print(f"  Beta vs SPY    : {beta:.3f}" if beta is not None else "  Beta vs SPY    : n/a")
    print(f"  Alpha (ann.)   : {alpha_annual:.2%}")
    print(f"  Best year      : {strat_yearly.idxmax()} ({strat_best_year:.2%})" if not strat_yearly.empty else "  Best year      : n/a")
    print(f"  Worst year     : {strat_yearly.idxmin()} ({strat_worst_year:.2%})" if not strat_yearly.empty else "  Worst year     : n/a")
    print("  Monthly returns:")
    print(f"    mean={strat_m_mean:.4f}, std={strat_m_std:.4f}, skew={strat_m_skew:.3f}")
    if outlier_set:
        outlier_list = sorted(list(outlier_set))
        print(f"  Outlier months (flagged): {outlier_list}")
    print()
    print("SPY Buy-and-Hold:")
    print(f"  CAGR            : {spy_cagr:.2%}")
    print(f"  Sharpe (ann.)  : {spy_sharpe:.2f}")
    print(f"  Max Drawdown   : {spy_max_dd:.2%}")
    print(f"  Calmar         : {spy_calmar:.2f}")
    print(f"  Best year      : {spy_yearly.idxmax()} ({spy_best_year:.2%})" if not spy_yearly.empty else "  Best year      : n/a")
    print(f"  Worst year     : {spy_yearly.idxmin()} ({spy_worst_year:.2%})" if not spy_yearly.empty else "  Worst year     : n/a")
    print("  Monthly returns:")
    print(f"    mean={spy_m_mean:.4f}, std={spy_m_std:.4f}, skew={spy_m_skew:.3f}")
    print("==========================================================\n")

    print(f"Saved report figure → {out_path}")


if __name__ == "__main__":
    main()

