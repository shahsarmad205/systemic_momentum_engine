from __future__ import annotations

"""
Holding-Period Sweep
====================

Runs the full weight-learning + walk-forward + backtest pipeline
for multiple forward-return / holding windows and compares:

  - Mean IC across walk-forward windows (vs raw forward_return)
  - IC t-stat across walk-forward windows
  - Sharpe ratio of the resulting strategy in a backtest

Defaults:
  - Holding windows: [1, 3, 5, 10] trading days
  - Dates: 2018-01-01 → 2024-01-01
  - Model: ridge regression, regression task, raw forward_return target

Usage (from project root):

    python -m analysis.holding_period_sweep
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from agents.weight_learning_agent import WeightLearner, build_feature_matrix
from backtesting import Backtester, load_config
from config import apply_dev_mode, get_effective_tickers

START_DATE = "2013-01-01"
END_DATE = "2024-01-01"
# Focus on shorter horizons where price-based signals are typically stronger.
HOLDING_WINDOWS = [1, 3, 5]
N_SPLITS = 5


def _resolve_tickers() -> list[str]:
    """Mirror run_weight_learning's ticker resolution."""
    try:
        from main import TICKERS

        fallback = list(TICKERS)
    except Exception:
        fallback = [
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOG",
            "META",
            "TSLA",
            "JPM",
            "V",
            "JNJ",
            "WMT",
            "PG",
            "XOM",
            "UNH",
            "BAC",
            "ABBV",
            "PFE",
            "KO",
            "PEP",
            "MRK",
            "AVGO",
            "COST",
            "TMO",
            "CSCO",
            "MCD",
            "NKE",
            "ADBE",
            "CRM",
            "AMD",
            "INTC",
            "ORCL",
            "IBM",
            "GS",
            "CAT",
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            "XLK",
            "VTI",
        ]
    # Use config helper to potentially downsize universe in DEV mode.
    return get_effective_tickers([], fallback)


@dataclass
class SweepResult:
    holding: int
    mean_ic: float
    ic_tstat: float
    mean_dir: float
    sharpe: float


def _compute_ic_stats(wf_results: list[dict]) -> tuple[float, float]:
    """Return (mean_ic, t_stat) from per-split ICs."""
    ics = [r.get("ic") for r in wf_results if r.get("ic") is not None]
    ics = [v for v in ics if not (isinstance(v, float) and (math.isnan(v)))]
    if not ics:
        return float("nan"), float("nan")
    arr = np.asarray(ics, dtype=float)
    mean_ic = float(arr.mean())
    if arr.size <= 1 or float(arr.std(ddof=1)) == 0.0:
        return mean_ic, float("nan")
    se = float(arr.std(ddof=1)) / math.sqrt(arr.size)
    t_stat = mean_ic / se
    return mean_ic, t_stat


def run_for_holding(holding_days: int, tickers: list[str]) -> SweepResult:
    print(f"\n=== Holding period: {holding_days} days ===\n")

    # 1) Build features for this forward window
    print("  Building feature matrix…")
    features_df = build_feature_matrix(
        tickers=tickers,
        start_date=START_DATE,
        end_date=END_DATE,
        holding_period=holding_days,
    )
    if features_df.empty:
        print("  [WARN] No features produced; skipping.")
        return SweepResult(holding_days, float("nan"), float("nan"), float("nan"), float("nan"))

    print(
        f"    features: {len(features_df):,} rows × {features_df.shape[1]} cols "
        f"({features_df['ticker'].nunique()} tickers)"
    )

    # 2) Train weight learner and run walk-forward validation
    print("  Training weight model + walk-forward validation…")
    learner = WeightLearner(
        model_type="ridge",
        alpha=0.01,
        time_decay_lambda=0.001,
        target_type="regression",
        return_target_type="raw",
    )
    wf_results = learner.walk_forward_validate(features_df, n_splits=N_SPLITS)
    mean_ic, ic_tstat = _compute_ic_stats(wf_results)
    # Average directional accuracy across folds (OOS)
    dir_vals = [r.get("directional_accuracy") for r in wf_results if r.get("directional_accuracy") is not None]
    mean_dir = float(np.mean(dir_vals)) if dir_vals else float("nan")
    print(f"    Mean IC across splits   : {mean_ic:+.4f}")
    print(f"    IC t-stat (splits)      : {ic_tstat:+.2f}")
    if not np.isnan(mean_dir):
        print(f"    Mean Dir. Acc. (OOS)    : {mean_dir:.4f}")
    else:
        print("    Mean Dir. Acc. (OOS)    : N/A")

    # Fit on full sample to get weights for backtest
    weights = learner.fit(features_df)
    weights_path = f"output/holding_sweep/learned_weights_hp{holding_days}.json"
    scaler_path = weights_path.replace(".json", "_scaler.json")
    import os

    os.makedirs("output/holding_sweep", exist_ok=True)
    weights.save(weights_path)
    learner.save_scaler(scaler_path)

    # 3) Backtest with learned weights to get Sharpe
    print("  Backtesting learned strategy…")
    cfg = load_config("backtest_config.yaml")
    apply_dev_mode(cfg)
    cfg.tickers = tickers
    cfg.start_date = START_DATE
    cfg.end_date = END_DATE
    cfg.holding_period_days = holding_days
    cfg.signal_mode = "learned"
    cfg.learned_weights_path = weights_path

    bt = Backtester(cfg)
    result = bt.run(tickers)
    sharpe = float(result.metrics.get("sharpe_ratio", float("nan")))
    print(f"    Sharpe ratio             : {sharpe:+.3f}")

    return SweepResult(holding_days, mean_ic, ic_tstat, mean_dir, sharpe)


def main() -> None:
    tickers = _resolve_tickers()
    results: list[SweepResult] = []

    for h in HOLDING_WINDOWS:
        res = run_for_holding(h, tickers)
        results.append(res)

    # Summary table
    df = pd.DataFrame(
        [
            {
                "holding_days": r.holding,
                "mean_ic": r.mean_ic,
                "ic_tstat": r.ic_tstat,
                "mean_dir": r.mean_dir,
                "sharpe": r.sharpe,
            }
            for r in results
        ]
    )
    print("\n=== Holding-period comparison ===\n")
    disp = df.copy()
    disp["mean_ic"] = disp["mean_ic"].map(lambda x: "N/A" if pd.isna(x) else f"{x:+.4f}")
    disp["ic_tstat"] = disp["ic_tstat"].map(lambda x: "N/A" if pd.isna(x) else f"{x:+.2f}")
    disp["mean_dir"] = disp["mean_dir"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.4f}")
    disp["sharpe"] = disp["sharpe"].map(lambda x: "N/A" if pd.isna(x) else f"{x:+.3f}")
    print(disp.to_string(index=False))
    print()


if __name__ == "__main__":
    main()

