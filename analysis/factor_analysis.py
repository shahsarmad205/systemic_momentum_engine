#!/usr/bin/env python3
"""
Fama–French 5-factor attribution for portfolio daily returns (analysis only).

Uses Ken French daily factors via pandas_datareader and backtest daily equity.

Dependencies (pip):
    pip install pandas-datareader statsmodels matplotlib

Run from trend_signal_engine root:
    python analysis/factor_analysis.py
"""

from __future__ import annotations

import warnings
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

START = "2018-01-01"
END = "2024-01-01"
FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
ROLL_WIN = 252


def _load_ff5() -> pd.DataFrame:
    try:
        import pandas_datareader as pdr
    except ImportError as e:
        raise SystemExit(
            "Install pandas-datareader: pip install pandas-datareader"
        ) from e

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        ff5 = pdr.get_data_famafrench(
            "F-F_Research_Data_5_Factors_2x3_daily",
            start=START,
            end=END,
        )[0]
    ff5 = ff5 / 100.0
    ff5.index = pd.to_datetime(ff5.index.astype(str))
    ff5 = ff5.sort_index()
    return ff5


def _load_portfolio_returns() -> pd.Series:
    path = ROOT / "output" / "backtests" / "daily_equity.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run a backtest first.")
    eq = pd.read_csv(path)
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date").set_index("date")
    ret = pd.to_numeric(eq["equity"], errors="coerce").pct_change()
    ret.name = "port_ret"
    return ret


def main() -> None:
    try:
        import statsmodels.api as sm
        from statsmodels.regression.rolling import RollingOLS
    except ImportError as e:
        raise SystemExit("Install statsmodels: pip install statsmodels") from e

    print("Loading Fama–French 5 factors (Ken French data library)…")
    ff = _load_ff5()
    print(f"  FF5 shape: {ff.shape}  ({ff.index.min().date()} → {ff.index.max().date()})")

    print("Loading portfolio returns from output/backtests/daily_equity.csv…")
    port_ret = _load_portfolio_returns()

    merged = port_ret.to_frame().join(ff, how="inner")
    merged = merged.dropna(subset=["port_ret", "RF"] + FACTOR_COLS)
    if len(merged) < 60:
        raise SystemExit("Not enough overlapping days for factor regression.")

    merged["port_excess"] = merged["port_ret"] - merged["RF"]
    y = merged["port_excess"]
    X = merged[FACTOR_COLS]

    print("\nFF factor map (not momentum/mean-reversion — use for style tilts):")
    print("  Mkt-RF: market; SMB: size; HML: value; RMW: profitability/quality; CMA: investment.")

    print("\n" + "=" * 72)
    print("OLS: portfolio_excess ~ 1 + Mkt-RF + SMB + HML + RMW + CMA")
    print("=" * 72)
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    print(model.summary())

    const = float(model.params.get("const", np.nan))
    t_alpha = float(model.tvalues.get("const", np.nan))
    alpha_daily = const
    alpha_ann = alpha_daily * 252.0
    r2 = float(model.rsquared)

    print("\n" + "-" * 72)
    print("KEY METRICS (daily regression; alpha scaled to annual)")
    print("-" * 72)
    print(f"  Alpha (intercept, daily):     {alpha_daily:.6f}")
    print(f"  Alpha (annualized ~ ×252):    {alpha_ann:.4f}")
    print(f"  Alpha t-stat:                 {t_alpha:.3f}")
    print(f"  R-squared:                    {r2:.4f}")
    for c in FACTOR_COLS:
        b = float(model.params.get(c, np.nan))
        t = float(model.tvalues.get(c, np.nan))
        print(f"  Beta {c:8s}: {b:+.4f}  (t={t:+.2f})")

    # Rolling exposures
    print("\nRolling 252-trading-day factor exposures…")
    y_roll = y
    X_roll = sm.add_constant(X)
    roll = RollingOLS(y_roll, X_roll, window=ROLL_WIN, min_nobs=max(60, ROLL_WIN // 3))
    rres = roll.fit()
    params = rres.params.copy()

    out_dir = ROOT / "output" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "factor_exposures.png"

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()
    const_series = params["const"] * 252.0  # annualized rolling alpha
    axes[0].plot(params.index, const_series, color="black", linewidth=1.2)
    axes[0].set_ylabel("Annualized α (rolling)")
    axes[0].set_title("Rolling intercept (×252)")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    for i, col in enumerate(FACTOR_COLS):
        ax = axes[i + 1]
        ax.plot(params.index, params[col], linewidth=1.0)
        ax.set_ylabel(col)
        ax.set_title(f"Rolling β: {col}")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved rolling betas plot → {plot_path}")

    # Strategy classification
    print("\n" + "=" * 72)
    print("STRATEGY CLASSIFICATION")
    print("=" * 72)
    if t_alpha > 2 and r2 < 0.3:
        print("GENUINE ALPHA: low factor exposure, sig alpha")
    elif t_alpha > 2 and r2 > 0.5:
        print("FACTOR TILTS: alpha partly explained by factors")
    else:
        print("UNCERTAIN: alpha not statistically significant")

    smb_b = float(model.params.get("SMB", np.nan))
    print("\nInterpretation notes:")
    print("  • High R² (>0.5): returns largely explained by factors — diversify feature set.")
    print("  • Alpha t < 2: limited statistical evidence of skill in this window.")
    if smb_b > 0.02:
        print("  • Positive SMB β: small-cap tilt; consider cap-neutral sizing if unintended.")
    elif smb_b < -0.02:
        print("  • Negative SMB β: large-cap tilt vs small-cap factor.")
    else:
        print("  • SMB β near zero: little size tilt vs FF factor.")


if __name__ == "__main__":
    main()
