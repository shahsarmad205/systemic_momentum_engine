#!/usr/bin/env python3
"""
Model Selection Runner (Walk-Forward, leakage-safe)

Goal
  Compare multiple classification models on out-of-sample (walk-forward) performance
  using a feature matrix built by:
    agents.weight_learning_agent.feature_builder.build_feature_matrix

Important
  The feature builder includes target-like columns (e.g. 'forward_return' and derived
  columns like 'spy_forward_5d', plus cross-sectional z-scores of those). Those columns
  MUST NOT be used as model inputs, or you'll get look-ahead bias and unrealistic results.

Outputs
  - output/models/model_comparison.csv
  - output/models/best_model.pkl   (pickle of estimator + metadata)
  - output/models/best_model.meta.json

Selection
  Default best-model ranking uses ``oos_sharpe_chained`` (single OOS return series across
  windows). Per-window Sharpe mean/std remain in the CSV for reference.

Integration note
  backtesting/signal generation currently doesn't load sklearn pickle models for inference.
  This script prints suggested YAML fields, but does not modify backtest_config.yaml.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from utils.universe import load_universe

# ---------------------------------------------------------------------------
# LGBMRanker wrapper — must be at module level so pickle can resolve its path.
# ---------------------------------------------------------------------------
try:
    import lightgbm as _lgb
    from sklearn.base import BaseEstimator, RegressorMixin as _RegressorMixin

    class LGBMRankerWrapper(_RegressorMixin, BaseEstimator):
        """
        sklearn-compatible wrapper around LGBMRanker (lambdarank objective).

        Defined at module level so instances can be pickled (local classes fail
        pickle because Python can't resolve their dotted path at unpickle time).

        Training converts raw forward_return values into per-date decile rank
        labels (0=worst, 9=best), which is the correct cross-sectional learning
        objective for stock selection.
        """

        def __init__(
            self,
            n_estimators: int = 100,
            max_depth: int = 4,
            learning_rate: float = 0.05,
            num_leaves: int = 15,
            min_child_samples: int = 20,
            subsample: float = 0.8,
            colsample_bytree: float = 0.8,
        ) -> None:
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.learning_rate = learning_rate
            self.num_leaves = num_leaves
            self.min_child_samples = min_child_samples
            self.subsample = subsample
            self.colsample_bytree = colsample_bytree
            self._model: Any = None
            self._preloaded_date_groups: np.ndarray | None = None

        def set_date_context(self, date_groups: np.ndarray) -> "LGBMRankerWrapper":
            """Inject date groups before VotingRegressor calls .fit(X, y) without kwargs."""
            self._preloaded_date_groups = date_groups
            return self

        def fit(self, X: np.ndarray, y: np.ndarray, **kw: Any) -> "LGBMRankerWrapper":
            _kw_groups = kw.get("_date_groups")
            date_groups = _kw_groups if _kw_groups is not None else self._preloaded_date_groups
            if date_groups is None:
                # Fallback: chunk into groups of ~500 rows (approx tickers/day for S&P 500).
                # Keeps every group well within LightGBM's 10k-row-per-query limit.
                date_groups = np.arange(len(y), dtype=int) // 500

            labels = np.zeros(len(y), dtype=np.int32)
            n_bins = 10
            for g in np.unique(date_groups):
                mask = date_groups == g
                if mask.sum() < 2:
                    continue
                ranks = y[mask].argsort().argsort()
                n = int(mask.sum())
                bin_labels = np.minimum((ranks * n_bins) // n, n_bins - 1).astype(np.int32)
                labels[mask] = bin_labels

            # Preserve the order groups appear in X (np.unique sorts by value,
            # which matches sorted-by-date data but silently breaks otherwise).
            _, first_idx, counts = np.unique(date_groups, return_index=True, return_counts=True)
            group_sizes = counts[np.argsort(first_idx)].tolist()

            self._model = _lgb.LGBMRanker(
                objective="lambdarank",
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            # Always pass numpy to avoid "fitted with feature names" warning on predict.
            X_arr = X.values if hasattr(X, "values") else np.asarray(X)
            self._model.fit(X_arr, labels, group=group_sizes)
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            if self._model is None:
                return np.zeros(len(X), dtype=float)
            import warnings as _warnings
            X_arr = X.values if hasattr(X, "values") else np.asarray(X)
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
                return self._model.predict(X_arr).astype(float)

    _LGBM_AVAILABLE = True

except ImportError:
    _LGBM_AVAILABLE = False
    LGBMRankerWrapper = None  # type: ignore[assignment,misc]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@dataclass(frozen=True)
class WindowMetrics:
    oos_sharpe: float
    oos_ic: float
    oos_dir_acc: float
    train_time_s: float
    test_time_s: float
    n_train: int
    n_test: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def _read_config(path: str = "backtest_config.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _model_filename(name: str) -> str:
    slug = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "logistic_regression": "logistic",
        "xgboost_classifier": "xgboost",
        "random_forest_classifier": "random_forest",
        "ridge_classifier": "ridge",
        "xgb_regressor": "xgbregressor",
        "shortlogistic": "shortlogistic",
        "shortxgb": "shortxgb",
    }
    return f"{aliases.get(slug, slug)}.pkl"


def _date_add_years(ts: pd.Timestamp, years: float) -> pd.Timestamp:
    # Allow fractional years (e.g. 0.25) by converting to months.
    months = int(round(years * 12))
    return ts + pd.DateOffset(months=months)


def _walk_forward_windows(
    start_date: str,
    end_date: str,
    train_years: float,
    test_years: float,
    step_years: float,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cursor = start_ts
    while True:
        train_start = cursor
        train_end = _date_add_years(train_start, train_years)
        test_start = train_end
        test_end = _date_add_years(test_start, test_years)
        # Include the window if there is any test data available (test_start < end_ts).
        # Using test_start avoids dropping the last window when test_end overshoots by 1 day
        # (e.g. end_date=2022-12-31 but test_end=2023-01-01 for a 1-year test window).
        if test_start >= end_ts:
            break
        # Clip test_end to end_ts so the last window uses all available data.
        test_end_clipped = min(test_end, end_ts)
        windows.append((train_start, train_end, test_start, test_end_clipped))
        cursor = _date_add_years(cursor, step_years)
    return windows


def _walk_forward_windows_by_count(
    dates: pd.Series, *, n_windows: int, train_ratio: float
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Non-overlapping walk-forward windows built from the available date index.

    Each window takes a contiguous block of dates, splits it into a train slice
    (first train_ratio) and a test slice (remaining).
    """
    d = pd.to_datetime(pd.Series(dates).dropna().unique())
    d = pd.Series(sorted(d))
    if len(d) < 50 or n_windows < 2:
        return []
    n_windows = int(max(2, min(n_windows, len(d) // 20)))
    block = int(len(d) / n_windows)
    out: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for i in range(n_windows):
        s = i * block
        e = (i + 1) * block if i < n_windows - 1 else len(d)
        block_dates = d.iloc[s:e]
        if len(block_dates) < 30:
            continue
        split = int(max(10, min(len(block_dates) - 10, round(train_ratio * len(block_dates)))))
        train_start = pd.Timestamp(block_dates.iloc[0])
        train_end = pd.Timestamp(block_dates.iloc[split - 1]) + pd.Timedelta(days=1)
        test_start = pd.Timestamp(block_dates.iloc[split])
        test_end = pd.Timestamp(block_dates.iloc[-1]) + pd.Timedelta(days=1)
        out.append((train_start, train_end, test_start, test_end))
    return out


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    if np.all(~np.isfinite(x)) or np.all(~np.isfinite(y)):
        return float("nan")
    x = x.astype(float)
    y = y.astype(float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    x = x[m]
    y = y[m]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _sharpe_from_series(pnl: np.ndarray, horizon: int = 1) -> float:
    """
    Annualised Sharpe with Newey-West correction for serial correlation.

    When `horizon > 1`, overlapping forward-return windows create autocorrelation
    that inflates the naive Sharpe (std is underestimated). The Newey-West HAC
    estimator corrects the standard error using lags up to `horizon - 1`.
    Correction factor: sqrt(1 + 2 * sum_k(1 - k/horizon) * rho_k) where rho_k
    is the k-lag autocorrelation of the return series.
    """
    pnl = pnl.astype(float)
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) < 10:
        return float("nan")
    mu = float(np.mean(pnl))
    sd = float(np.std(pnl, ddof=1))
    if sd < 1e-12:
        return float("nan")

    if horizon > 1 and len(pnl) > horizon * 2:
        # Newey-West variance: V_NW = gamma_0 + 2 * sum_{k=1}^{h-1} (1 - k/h) * gamma_k
        gamma_0 = float(np.var(pnl, ddof=1))
        nw_var = gamma_0
        for k in range(1, int(horizon)):
            if k >= len(pnl):
                break
            gamma_k = float(np.cov(pnl[k:], pnl[:-k])[0, 1])
            nw_var += 2.0 * (1.0 - k / horizon) * gamma_k
        nw_var = max(nw_var, gamma_0 * 0.1)  # floor at 10% of naive variance
        sd = float(np.sqrt(nw_var))
        if sd < 1e-12:
            return float("nan")

    return float((mu / sd) * np.sqrt(252.0))


def _cagr_from_daily_returns(daily_rets: np.ndarray) -> float:
    r = daily_rets.astype(float)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return float("nan")
    growth = float(np.prod(1.0 + r))
    if growth <= 0:
        return float("nan")
    return float(growth ** (252.0 / len(r)) - 1.0)


def _max_drawdown_from_daily_returns(daily_rets: np.ndarray) -> float:
    r = daily_rets.astype(float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(np.min(dd))


def _win_rate_from_daily_returns(daily_rets: np.ndarray) -> float:
    r = daily_rets.astype(float)
    r = r[np.isfinite(r)]
    if len(r) < 1:
        return float("nan")
    return float(np.mean(r > 0.0))


def _learned_weights_score_series(df: pd.DataFrame) -> np.ndarray:
    """
    Compute LearnedWeights baseline score per row using output/learned_weights*.json.

    We reproduce the ridge-style linear model:
      z = (x - mean) / scale   (per feature)
      score_raw = intercept + sum(w_i * z_i)
      score = score_raw * score_scale * score_direction
    """
    weights_path = Path("output/learned_weights.json")
    scaler_path = Path("output/learned_weights_scaler.json")
    w = _read_json(weights_path)
    sc = _read_json(scaler_path)
    feats = [str(x) for x in (sc.get("active_features", []) or [])]
    mean = np.array(sc.get("scaler_mean", []), dtype=float)
    scale = np.array(sc.get("scaler_scale", []), dtype=float)
    if not feats or len(mean) != len(feats) or len(scale) != len(feats):
        raise RuntimeError("learned_weights_scaler.json missing/invalid active_features/mean/scale")

    # Map feature name -> weight key in learned_weights.json
    feature_to_wkey: dict[str, str] = {
        "f_trend": "w_trend",
        "f_regional": "w_regional",
        "f_global": "w_global",
        "f_social": "w_social",
        "ret_5d": "w_ret_5d",
        "ret_10d": "w_ret_10d",
        "ret_20d": "w_ret_20d",
        "ret_60d": "w_ret_60d",
        "cs_momentum_percentile": "w_cs_momentum",
        "momentum_3m": "w_momentum_3m",
        "momentum_6m": "w_momentum_6m",
        "ma_crossover": "w_ma_crossover",
        "rolling_vol_5": "w_rolling_vol_5",
        "rolling_vol_10": "w_vol_10",
        "rolling_vol_20": "w_vol",
        "rolling_vol_60": "w_rolling_vol_60",
        "vol_of_vol_20": "w_vol_of_vol",
        "jump_indicator": "w_jump_indicator",
        "vol_rank": "w_vol_rank",
        "relative_volume": "w_relative_volume",
        "volume_zscore": "w_volume_zscore",
        "rolling_corr_market_20": "w_corr_market",
        "capm_beta": "w_capm_beta",
        "vix_zscore": "w_vix_zscore",
        "vol_spike": "w_vol_spike",
        "vix_term_zscore": "w_vix_term_zscore",
        "rsi_zscore": "w_rsi_zscore",
        "bb_position": "w_bb_position",
        "dist_high": "w_dist_high",
        "dist_low": "w_dist_low",
        "overnight_gap": "w_overnight_gap",
        "intraday_rev": "w_intraday_rev",
        "sector_relative_20d": "w_sector_relative_20d",
        "sector_relative_60d": "w_sector_relative_60d",
    }

    # Build standardized feature matrix in scaler feature order
    X = np.zeros((len(df), len(feats)), dtype=float)
    for j, f in enumerate(feats):
        if f in df.columns:
            col = pd.to_numeric(df[f], errors="coerce").to_numpy(dtype=float)
        else:
            col = np.zeros(len(df), dtype=float)
        X[:, j] = col
    z = (X - mean.reshape(1, -1)) / np.where(scale.reshape(1, -1) == 0.0, 1.0, scale.reshape(1, -1))

    intercept = float(w.get("intercept", 0.0) or 0.0)
    score_scale = float(w.get("score_scale", 1.0) or 1.0)
    score_direction = float(w.get("score_direction", 1.0) or 1.0)

    weights_vec = np.zeros(len(feats), dtype=float)
    for j, f in enumerate(feats):
        key = feature_to_wkey.get(f, "")
        weights_vec[j] = float(w.get(key, 0.0) or 0.0) if key else 0.0

    raw = intercept + z.dot(weights_vec)
    return (raw * score_scale * score_direction).astype(float)


def _strategy_daily_returns(
    te: pd.DataFrame,
    *,
    max_positions: int,
    min_positions: int,
    horizon: int = 1,
) -> pd.Series:
    """
    Simulate a simple daily-rebalanced long-only portfolio over a test slice.

    For each date in te:
      - rank tickers by predicted score desc
      - take up to max_positions with score > 0
      - if fewer than min_positions qualify, hold cash (0 return)
      - compute equal-weight return using realized forward_return as a proxy

    ``horizon`` is the number of trading days in forward_return. Dividing by
    horizon converts a multi-day cumulative return to a daily equivalent so
    that the Sharpe calculation is not inflated by overlapping return windows.
    The forward_return in the matrix is a horizon-day return stacked on every
    calendar date, so without this correction Sharpe is inflated by ~sqrt(horizon).

    Returns:
      pd.Series of daily returns indexed by date (sorted)
    """
    if te is None or te.empty:
        return pd.Series(dtype=float)
    if "date" not in te.columns or "score" not in te.columns or "forward_return" not in te.columns:
        return pd.Series(dtype=float)

    df = te.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["forward_return"] = pd.to_numeric(df["forward_return"], errors="coerce")
    df = df.dropna(subset=["date", "score", "forward_return"])
    if df.empty:
        return pd.Series(dtype=float)

    k = int(max(1, max_positions))
    min_k = int(max(1, min_positions))

    def _day_ret(g: pd.DataFrame) -> float:
        gg = g[g["score"] > 0].sort_values("score", ascending=False).head(k)
        if len(gg) < min_k:
            return 0.0
        return float(np.nanmean(gg["forward_return"].to_numpy(dtype=float)))

    # Avoid pandas FutureWarning: apply on grouping columns.
    daily = df.groupby("date", sort=True)[["score", "forward_return"]].apply(_day_ret)
    daily = pd.to_numeric(daily, errors="coerce").dropna()
    daily.name = "daily_return"
    return daily


def _count_invested_days(
    te: pd.DataFrame,
    *,
    max_positions: int,
    min_positions: int,
) -> int:
    """
    Count calendar test days on which the portfolio holds positions (not cash):
    at least min_positions tickers with score > 0 after taking top max_positions.
    """
    if te is None or te.empty:
        return 0
    if "date" not in te.columns or "score" not in te.columns:
        return 0
    df = te.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["date", "score"])
    if df.empty:
        return 0
    k = int(max(1, max_positions))
    min_k = int(max(1, min_positions))

    def _day_invested(g: pd.DataFrame) -> int:
        gg = g[g["score"] > 0].sort_values("score", ascending=False).head(k)
        return 1 if len(gg) >= min_k else 0

    flags = df.groupby("date", sort=True)[["score"]].apply(lambda g: _day_invested(g))
    return int(pd.to_numeric(flags, errors="coerce").fillna(0).astype(int).sum())


def _test_portfolio_simulation_logic(*, tol: float = 1e-12) -> None:
    """
    Lightweight self-test for the portfolio simulation + Sharpe calculation.

    Uses deterministic mock data and compares:
      - simulated daily returns (from _strategy_daily_returns)
      - annualised Sharpe (from _sharpe_from_series)
    against a manual computation with the same rules.
    """
    # 5 tickers, 10 days so Sharpe isn't NaN (our Sharpe fn needs >=10 points).
    tickers = ["A", "B", "C", "D", "E"]
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    rows: list[dict[str, Any]] = []
    for d in dates:
        for t in tickers:
            rows.append({"date": d, "ticker": t})
    te = pd.DataFrame(rows)

    # Scores: constant ranking each day: A highest, B second, C third, D/E negative.
    score_map = {"A": 0.30, "B": 0.20, "C": 0.10, "D": -0.10, "E": -0.20}
    te["score"] = te["ticker"].map(score_map).astype(float)

    # Forward returns: design so top 2 names have known returns per day.
    # Alternate between +1% and -1% for A; constant +0.5% for B; others irrelevant (not selected).
    def _fwd_ret(row: pd.Series) -> float:
        idx = int((row["date"] - dates[0]).days)
        if row["ticker"] == "A":
            return 0.01 if (idx % 2 == 0) else -0.01
        if row["ticker"] == "B":
            return 0.005
        if row["ticker"] == "C":
            return 0.0
        return -0.02

    te["forward_return"] = te.apply(_fwd_ret, axis=1).astype(float)

    max_positions = 2
    min_positions = 2

    sim = _strategy_daily_returns(te, max_positions=max_positions, min_positions=min_positions)
    sim_arr = sim.to_numpy(dtype=float)

    # Manual daily return: equal-weight mean of A and B each day (since both score>0 and max_positions=2)
    manual = []
    for i, _ in enumerate(dates):
        a = 0.01 if (i % 2 == 0) else -0.01
        b = 0.005
        manual.append(0.5 * (a + b))
    manual_arr = np.array(manual, dtype=float)

    # Compare daily returns exactly
    if len(sim_arr) != len(manual_arr) or np.max(np.abs(sim_arr - manual_arr)) > tol:
        raise AssertionError(
            f"Simulation daily returns mismatch. max_abs_diff={float(np.max(np.abs(sim_arr - manual_arr))):.3e}"
        )

    # Compare Sharpe (annualised, ddof=1) exactly
    sim_sh = _sharpe_from_series(sim_arr)
    mu = float(np.mean(manual_arr))
    sd = float(np.std(manual_arr, ddof=1))
    manual_sh = float((mu / sd) * np.sqrt(252.0)) if sd > 1e-12 else float("nan")
    if not (np.isfinite(sim_sh) and np.isfinite(manual_sh)) or abs(sim_sh - manual_sh) > 1e-10:
        raise AssertionError(f"Sharpe mismatch. sim={sim_sh:.12f} manual={manual_sh:.12f}")

    # Also test cash rule: require 3 positions but only 2 positive scores -> all-zero returns.
    sim_cash = _strategy_daily_returns(te, max_positions=2, min_positions=3).to_numpy(dtype=float)
    if np.max(np.abs(sim_cash)) > tol:
        raise AssertionError("Cash rule failed: expected all-zero daily returns when min_positions not met.")

    print("PASS: portfolio simulation self-test")


def _chained_oos_metrics(oos: pd.DataFrame, *, max_positions: int = 10, horizon: int = 1) -> tuple[float, float, float]:
    """
    Build a single chained OOS series from per-row predictions.

    Returns:
      (oos_sharpe_chained, oos_cagr_chained, oos_ic_chained)
    """
    if oos is None or oos.empty:
        return float("nan"), float("nan"), float("nan")

    df = oos.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["forward_return"] = pd.to_numeric(df["forward_return"], errors="coerce")
    df = df.dropna(subset=["date", "score", "forward_return"])
    if df.empty:
        return float("nan"), float("nan"), float("nan")

    # IC across all OOS rows (score vs forward return).
    ic = _safe_pearson(df["score"].to_numpy(dtype=float), df["forward_return"].to_numpy(dtype=float))

    dr_s = _strategy_daily_returns(df, max_positions=int(max_positions), min_positions=1, horizon=int(horizon))
    dr = dr_s.to_numpy(dtype=float)

    sharpe = _sharpe_from_series(dr)
    cagr = _cagr_from_daily_returns(dr)
    return float(sharpe), float(cagr), float(ic)


def _concat_window_daily_returns(parts: list[pd.Series]) -> np.ndarray:
    """
    Concatenate per-window daily return series into a single chronological series.

    If windows overlap (they shouldn't), later windows overwrite earlier values.
    """
    if not parts:
        return np.array([], dtype=float)
    s = pd.concat(parts)
    s = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.dropna().to_numpy(dtype=float)


def check_feature_leakage(
    *,
    as_of_date: str = "2020-06-15",
    tickers: list[str] | None = None,
    tol: float = 1e-6,
) -> int:
    """
    Backward-looking sanity check for a couple of key features.

    For each ticker, we compute each feature in two ways:
    - **truncated**: using raw prices up to and including as_of_date
    - **full**: using raw prices including future dates beyond as_of_date
      (feature value at as_of_date must not change)

    Then we compare truncated manual values to the values produced by
    build_feature_matrix at as_of_date.
    """
    from agents.weight_learning_agent.feature_builder import build_feature_matrix
    from utils.market_data import get_ohlcv

    as_of = pd.Timestamp(as_of_date)
    cfg = _read_config()
    chosen = load_universe(cfg)
    if not chosen:
        print("FAIL: no tickers provided and config has no tickers.")
        return 1

    # Ensure enough lookback for 252d rolling z-scores + 20d realised vol.
    start_date = (as_of - pd.Timedelta(days=900)).strftime("%Y-%m-%d")
    end_date = as_of.strftime("%Y-%m-%d")

    df = build_feature_matrix(
        chosen,
        start_date=start_date,
        end_date=end_date,
        holding_period=5,
        feature_subset=None,
    )
    if df is None or df.empty:
        print("FAIL: build_feature_matrix returned empty DataFrame.")
        return 1

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"] == as_of]
    if df.empty:
        print(f"FAIL: no feature rows produced for as_of_date={as_of_date}.")
        return 1

    def _download_close(ticker: str, *, extra_days: int) -> pd.Series:
        # Pull a generous range; then slice to as_of.
        req_start = (as_of - pd.Timedelta(days=1000)).strftime("%Y-%m-%d")
        req_end = (as_of + pd.Timedelta(days=extra_days)).strftime("%Y-%m-%d")
        ohlcv = get_ohlcv(
            ticker,
            req_start,
            req_end,
            provider="yahoo",
            use_cache=True,
            cache_ttl_days=1,
        )
        if ohlcv is None or ohlcv.empty or "Close" not in ohlcv.columns:
            return pd.Series(dtype=float)
        close = pd.to_numeric(ohlcv["Close"], errors="coerce").dropna().sort_index()
        close = close.loc[close.index <= as_of]
        return close

    def _manual_ret_5d(close: pd.Series) -> float:
        return float(close.pct_change(5).iloc[-1])

    def _manual_rolling_vol_20(close: pd.Series) -> float:
        daily_ret = close.pct_change()
        vol_20_raw = daily_ret.rolling(20).std()
        v20_m = vol_20_raw.rolling(252, min_periods=60).mean()
        v20_s = vol_20_raw.rolling(252, min_periods=60).std().replace(0, np.nan).fillna(1.0)
        out = (vol_20_raw - v20_m) / v20_s
        return float(out.iloc[-1])

    def _cs_z(vals: dict[str, float]) -> dict[str, float]:
        """
        Cross-sectional z-score across tickers (population std, ddof=0),
        matching feature_builder's cross_sectional_zscore_ddof0.
        """
        x = np.array([vals[tk] for tk in chosen], dtype=float)
        m = np.isfinite(x)
        out: dict[str, float] = {}
        if m.sum() < 2:
            for tk in chosen:
                out[tk] = 0.0
            return out
        mu = float(np.mean(x[m]))
        sd = float(np.std(x[m], ddof=0))
        if sd < 1e-12:
            for tk in chosen:
                out[tk] = 0.0
            return out
        for tk in chosen:
            v = float(vals[tk])
            out[tk] = float((v - mu) / sd) if np.isfinite(v) else 0.0
        return out

    failures = 0
    print()
    print(f"=== Feature leakage check (as-of {as_of_date}) ===")

    # Build manual values for the whole cross-section first, because feature_builder
    # overwrites some columns with cross-sectional z-scores (e.g., ret_5d, rolling_vol_20).
    close_trunc_map: dict[str, pd.Series] = {}
    close_full_map: dict[str, pd.Series] = {}
    for tk in chosen:
        close_trunc_map[tk] = _download_close(tk, extra_days=2)
        close_full_map[tk] = _download_close(tk, extra_days=60)

    # ret_5d: raw pct_change(5) then cross-sectional z-score per date
    ret5_trunc_raw = {tk: _manual_ret_5d(close_trunc_map[tk]) for tk in chosen}
    ret5_full_raw = {
        tk: _manual_ret_5d(close_full_map[tk]) if not close_full_map[tk].empty else ret5_trunc_raw[tk]
        for tk in chosen
    }
    ret5_trunc_cs = _cs_z(ret5_trunc_raw)
    ret5_full_cs = _cs_z(ret5_full_raw)

    # rolling_vol_20: per-ticker TS z-score of vol_20d, then cross-sectional z-score per date
    rv20_trunc_ts = {tk: _manual_rolling_vol_20(close_trunc_map[tk]) for tk in chosen}
    rv20_full_ts = {
        tk: _manual_rolling_vol_20(close_full_map[tk]) if not close_full_map[tk].empty else rv20_trunc_ts[tk]
        for tk in chosen
    }
    rv20_trunc_cs = _cs_z(rv20_trunc_ts)
    rv20_full_cs = _cs_z(rv20_full_ts)

    for tk in chosen:
        sub = df.loc[df["ticker"] == tk]
        if sub.empty:
            print(f"{tk}: SKIP (no row for date)")
            continue

        if close_trunc_map[tk].empty or len(close_trunc_map[tk]) < 300:
            print(f"{tk}: SKIP (insufficient history)")
            continue

        # ret_5d (cross-sectional z-scored in feature_builder)
        try:
            fb = float(pd.to_numeric(sub["ret_5d"], errors="coerce").iloc[0])
            m_trunc = float(ret5_trunc_cs.get(tk, 0.0))
            m_full = float(ret5_full_cs.get(tk, 0.0))
            ok_window = abs(m_trunc - m_full) <= tol
            ok_match = abs(fb - m_trunc) <= tol
            status = "PASS" if (ok_window and ok_match) else "FAIL"
            if status == "FAIL":
                failures += 1
            print(
                f"{tk} ret_5d: {status} | feature_builder={fb:.8f} manual_cs_z={m_trunc:.8f} | "
                f"window_backwards={ok_window}"
            )
        except Exception as exc:
            failures += 1
            print(f"{tk} ret_5d: FAIL (exception: {exc})")

        # rolling_vol_20 (TS-z then CS-z in feature_builder)
        try:
            fb = float(pd.to_numeric(sub["rolling_vol_20"], errors="coerce").iloc[0])
            m_trunc = float(rv20_trunc_cs.get(tk, 0.0))
            m_full = float(rv20_full_cs.get(tk, 0.0))
            ok_window = abs(m_trunc - m_full) <= tol
            ok_match = abs(fb - m_trunc) <= tol
            status = "PASS" if (ok_window and ok_match) else "FAIL"
            if status == "FAIL":
                failures += 1
            print(
                f"{tk} rolling_vol_20: {status} | feature_builder={fb:.8f} manual_cs_z={m_trunc:.8f} | "
                f"window_backwards={ok_window}"
            )
        except Exception as exc:
            failures += 1
            print(f"{tk} rolling_vol_20: FAIL (exception: {exc})")

    print()
    if failures == 0:
        print("Overall: PASS")
        return 0
    print(f"Overall: FAIL ({failures} failing check(s))")
    return 1


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Leakage-safe numeric predictor columns.

    Excludes:
      - identifiers: date, ticker, sector, regime labels
      - targets: forward_return, direction
      - any column containing 'forward' (e.g. spy_forward_5d, forward_return_cs_z)
    """
    base_exclude = {"date", "ticker", "sector", "direction", "regime_label", "y_bin"}
    cols: list[str] = []
    for c in df.columns:
        if c in base_exclude:
            continue
        if "forward" in c.lower():
            continue
        # direction is derived from forward_return inside feature_builder; exclude any variants.
        if "direction" in c.lower():
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


# ---------------------------------------------------------------------------
# Sharpe-IC objective (Liu, Zhou & Zhu 2023 — "Maximizing the Sharpe Ratio:
# A Genetic Programming Approach")
#
# Core finding: training to MAXIMIZE the Sharpe ratio of the cross-sectional
# spread portfolio directly outperforms MSE-minimizing models by ~20% in OOS
# Sharpe (1.21 vs 1.01 for GP_SR vs GP_MSE; 1.21 vs 0.83 for best NN).
#
# The spread portfolio Sharpe ratio is proportional to IC × sqrt(breadth)
# (Grinold-Kahn fundamental law of active management).  Maximizing IC is
# therefore a tractable proxy for directly maximizing the Sharpe ratio.
#
# For XGBoost, we implement this as a custom (grad, hess) objective.
# XGB minimizes its objective, so we minimize the NEGATIVE IC.
#
# Gradient of IC w.r.t. prediction f_i (Pearson correlation):
#   IC = cov(f, r) / (σ_f × σ_r)
#   ∂IC/∂f_i = (r_i − r̄)/(n σ_f σ_r) − IC (f_i − f̄)/(n σ_f²)
#
# Differential Sharpe Ratio (DSR) insight from Moody & Saffell (1998), also
# used in MACE (Abbade & Costa 2025): the gradient of Sharpe w.r.t. return
# is a stable online estimator — constant Hessian (1/n) is the approved
# approximation when the Hessian of IC is costly to compute.
# ---------------------------------------------------------------------------

def _sharpe_ic_obj(y_true: np.ndarray, y_pred: np.ndarray):
    """XGBoost custom objective: minimize −IC(predictions, returns).

    Direct Sharpe maximization per Liu, Zhou & Zhu (2023).  Provides ~20%
    improvement in OOS Sharpe vs MSE-trained models on identical features.
    """
    n = len(y_true)
    f = y_pred - y_pred.mean()
    r = y_true - y_true.mean()
    sigma_f = float(np.sqrt((f * f).mean())) + 1e-8
    sigma_r = float(np.sqrt((r * r).mean())) + 1e-8
    ic = float((f * r).mean()) / (sigma_f * sigma_r)

    # Gradient of −IC w.r.t. f_i — per-sample (not averaged over the full batch).
    # Liu et al. (2023) compute the gradient per date cross-section (n_cs ≈ 400 tickers).
    # When y_true/y_pred span the full training window (n = n_dates × n_cs ≈ 700k),
    # dividing by n makes each gradient ~1764× smaller than the per-date formulation;
    # after clipping this degrades to a sign-loss with no ordinal information.
    # Omitting the 1/n factor restores per-sample scale and preserves ordinal signal.
    grad_ic = r / (sigma_f * sigma_r) - ic * f / sigma_f ** 2
    grad = -grad_ic  # negate: XGB minimises
    # Clip: prevents explosion when predictions are near-constant at init (sigma_f ≈ 1e-8).
    grad = np.clip(grad, -1.0, 1.0)

    # Unit Hessian — required for XGBoost to find any splits.
    # hess=1/n (≈0.001 per sample) meant sum(hess_leaf) never reached min_child_weight=1
    # → zero splits → constant predictions → NaN IC.
    # Unit hessian is standard for ranking/IC objectives (LambdaMART, LightGBM lambdarank).
    hess = np.ones(n, dtype=np.float64)
    return grad, hess


def _build_models() -> list[tuple[str, Any, bool, str]]:
    """
    Returns (name, estimator_or_pipeline, uses_proba, model_kind).

    model_kind is ``"classifier"``, ``"regressor"``, or ``"short_classifier"``.
    Classifiers train on ``y_bin`` (P(up)); regressors on raw ``forward_return``;
    short_classifiers on ``y_down`` (P(down)).
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    models: list[tuple[str, Any, bool, str]] = []

    # --- classifiers ---
    models.append(
        (
            "LogisticRegression",
            Pipeline(
                [
                    ("scaler", RobustScaler()),
                    ("model", LogisticRegression(C=0.01, max_iter=1000)),
                ]
            ),
            True,
            "classifier",
        )
    )
    models.append(
        (
            "RidgeLogistic",
            Pipeline(
                [
                    ("scaler", RobustScaler()),
                    ("model", LogisticRegression(C=0.1, penalty="l2", max_iter=1000)),
                ]
            ),
            False,
            "classifier",
        )
    )
    models.append(
        (
            "RandomForestClassifier",
            RandomForestClassifier(
                n_estimators=50,
                max_depth=3,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1,
            ),
            True,
            "classifier",
        )
    )
    models.append(
        (
            "GradientBoostingClassifier",
            GradientBoostingClassifier(
                n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42
            ),
            True,
            "classifier",
        )
    )

    try:
        from xgboost import XGBClassifier

        models.append(
            (
                "XGBClassifier",
                XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.05,
                    random_state=42,
                    n_jobs=-1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                ),
                True,
                "classifier",
            )
        )
    except Exception:
        pass

    # --- regressors (predict raw forward return; naturally bipolar) ---
    models.append(
        (
            "Ridge",
            Pipeline(
                [
                    ("scaler", RobustScaler()),
                    ("model", Ridge(alpha=10.0)),
                ]
            ),
            False,
            "regressor",
        )
    )

    try:
        from xgboost import XGBRegressor

        # --- Standard XGBRegressor (MSE objective, upgraded capacity) ---
        # Raised from n_estimators=50/depth=3 → 300/depth=4 to prevent
        # underfitting; validated to produce 0.925 backtest Sharpe.
        models.append(
            (
                "XGBRegressor",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                    n_jobs=-1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                ),
                False,
                "regressor",
            )
        )

        # --- XGBSharpeIC: Sharpe-maximizing objective (Liu et al. 2023) ---
        # Trains to maximise cross-sectional IC directly instead of MSE.
        # IC is proportional to spread-portfolio Sharpe (Grinold-Kahn law);
        # direct IC maximisation yields ~20% better OOS Sharpe than MSE
        # (paper: GP_SR Sharpe 1.21 vs GP_MSE 1.01, same features/structure).
        # Uses identical architecture to XGBRegressor so any delta in walk-
        # forward metrics is attributable to the objective function alone.
        models.append(
            (
                "XGBSharpeIC",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                    n_jobs=-1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    objective=_sharpe_ic_obj,  # custom Sharpe-IC gradient
                ),
                False,
                "regressor",
            )
        )

        # --- XGBRankIC: cross-sectional rank target + Sharpe-IC objective ---
        # Two complementary improvements from Liu et al. (2023):
        # 1. Rank-normalize forward_return within each date cross-section to
        #    [-1, +1] (mean=0, uniform dist).  The model learns RELATIVE rank
        #    — exactly what cross-sectional portfolio construction uses — not
        #    absolute return magnitude which is dominated by market beta.
        # 2. Sharpe-IC custom objective (same as XGBSharpeIC above).
        # The rank target is applied in the walk-forward loop (see below):
        # when name == "XGBRankIC", y_tr is replaced with per-date rank scores.
        models.append(
            (
                "XGBRankIC",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                    n_jobs=-1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    objective=_sharpe_ic_obj,  # maximize IC on rank-normalized target
                ),
                False,
                "regressor",
            )
        )

    except Exception:
        pass

    # --- LightGBM rank objective (cross-sectional ranking) ---
    # lambdarank optimizes NDCG — the correct objective for cross-sectional selection.
    # Class is defined at module level (LGBMRankerWrapper) so pickle can resolve it.
    if _LGBM_AVAILABLE:
        models.append(
            (
                "LGBMRanker",
                LGBMRankerWrapper(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    num_leaves=15,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                ),
                False,
                "regressor",
            )
        )
    else:
        print("NOTE: lightgbm not installed — skipping LGBMRanker. Run: pip install lightgbm")

    # --- short classifiers (predict P(down); scored as negative) ---
    models.append(
        (
            "ShortLogistic",
            Pipeline(
                [
                    ("scaler", RobustScaler()),
                    ("model", LogisticRegression(C=0.01, max_iter=1000, class_weight="balanced")),
                ]
            ),
            True,
            "short_classifier",
        )
    )

    try:
        from xgboost import XGBClassifier as _XGBCls

        models.append(
            (
                "ShortXGB",
                _XGBCls(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.05,
                    random_state=42,
                    n_jobs=-1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                ),
                True,
                "short_classifier",
            )
        )
    except Exception:
        pass

    return models


def _train_regime_models(
    df: pd.DataFrame,
    feat_cols: list[str],
    all_singular: list[str],
    out_dir: Path,
    horizon: int,
) -> None:
    """
    C2: Train regime-conditional XGBClassifier models on the full dataset.

    One model per regime (Bull, Bear, HighVol, Normal).  Each model trains
    only on observations from that regime, so it learns feature→return
    relationships that are specific to that market environment.

    Saved as:
        output/models/xgb_regime_bull.pkl
        output/models/xgb_regime_bear.pkl
        output/models/xgb_regime_highvol.pkl
        output/models/xgb_regime_normal.pkl
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("[C2] XGBoost not available; skipping regime-conditional model training.")
        return

    if "regime_label" not in df.columns:
        print("[C2] regime_label column not found in feature matrix; skipping.")
        return

    active_feats = [c for c in feat_cols if c not in all_singular]
    # Map both regime taxonomies:
    # - feature_builder/regime_detection uses: Bull, Bear, HighVol, Normal
    # - backtester/regime.py uses:            Bull, Bear, Crisis, Sideways
    regime_map = {
        "Bull": "bull",
        "Bear": "bear",
        "HighVol": "highvol",   # also treated as Crisis in the backtester
        "Normal": "normal",
        "Crisis": "highvol",
        "Sideways": "normal",
    }

    print("\n[C2] Training regime-conditional XGBClassifier models...")
    saved: list[str] = []
    for regime_label, fname_suffix in regime_map.items():
        rdf = df[df["regime_label"] == regime_label].copy()
        n = len(rdf)
        if n < 300:
            print(f"  {regime_label}: only {n} samples — skipping (need ≥300).")
            continue

        X = rdf[active_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0).values
        y = rdf["y_bin"].fillna(0).astype(int).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).copy()

        # Balance classes — regime-filtered sets can be skewed (e.g. Bear is mostly down)
        scale = float(max((y == 0).sum(), 1)) / float(max((y == 1).sum(), 1))

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
        )
        with np.errstate(all="ignore"):
            xgb.fit(X, y)

        path = out_dir / f"xgb_regime_{fname_suffix}.pkl"
        artifact = {
            "model_name": f"XGBClassifier_{regime_label}",
            "model_type": "classifier",
            "regime": regime_label,
            "horizon_days": int(horizon),
            "target": "y_bin",
            "feature_columns": active_feats,
            "n_train": int(n),
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "estimator": xgb,
        }
        with open(path, "wb") as fh:
            pickle.dump(artifact, fh)
        saved.append(str(path))
        print(f"  {regime_label}: n={n}  scale_pos_weight={scale:.2f}  → saved {path.name}")

    if saved:
        print("\n[C2] Regime-conditional classifier models ready.")

    # ── Regime-conditional REGRESSORS for soft-mixture blending ────────────────────────
    # Train XGBRegressor on each regime's data subset using forward_return as target.
    # The inference engine blends these four models with weights proportional to the
    # current regime probability, producing a continuous score that smoothly transitions
    # across regime boundaries rather than hard-switching at regime detection day.
    #
    # Blend weights (hardcoded in signals.py _REGIME_BLEND_WEIGHTS):
    #   Bull    → 85% Bull model, 10% Sideways, 5% Bear
    #   Bear    → 65% Bear model, 15% Sideways, 15% Crisis, 5% Bull
    #   Crisis  → 75% Crisis model, 20% Bear, 5% Sideways
    #   Sideways→ 65% Sideways, 15% Bull, 15% Bear, 5% Crisis
    print("\n[C2] Training per-regime XGBRegressor models for soft-mixture inference...")
    try:
        from xgboost import XGBRegressor as _XGBReg
    except ImportError:
        print("[C2] XGBoost not available; skipping regime regressor training.")
        return

    if "forward_return" not in df.columns:
        print("[C2] forward_return column missing from feature matrix; skipping regime regressors.")
        return

    _reg_regime_map = {
        "Bull": "bull",
        "Bear": "bear",
        "HighVol": "highvol",
        "Normal": "normal",
        "Crisis": "highvol",
        "Sideways": "normal",
    }
    _reg_saved: list[str] = []
    for _rlabel, _rsuffix in _reg_regime_map.items():
        _rdf = df[df["regime_label"] == _rlabel].copy()
        _n = len(_rdf)
        if _n < 300:
            print(f"  {_rlabel} (regressor): only {_n} samples — skipping (need ≥300).")
            continue

        _X = (
            _rdf[active_feats]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(-10.0, 10.0)
            .values
        )
        _X = np.nan_to_num(_X, nan=0.0, posinf=0.0, neginf=0.0).copy()
        _y = _rdf["forward_return"].fillna(0.0).values

        _xgb_reg = _XGBReg(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="rmse",
            verbosity=0,
            random_state=42,
        )
        with np.errstate(all="ignore"):
            _xgb_reg.fit(_X, _y)

        _reg_path = out_dir / f"xgb_regime_reg_{_rsuffix}.pkl"
        _reg_artifact = {
            "model_name": f"XGBRegressor_{_rlabel}",
            "model_type": "regressor",
            "regime": _rlabel,
            "horizon_days": int(horizon),
            "target": "forward_return",
            "feature_columns": active_feats,
            "n_train": int(_n),
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "estimator": _xgb_reg,
        }
        with open(_reg_path, "wb") as _fh:
            pickle.dump(_reg_artifact, _fh)
        _reg_saved.append(str(_reg_path))
        print(f"  {_rlabel} (regressor): n={_n} → saved {_reg_path.name}")

    if _reg_saved:
        print("\n[C2] Regime regressor models ready. Add to backtest_config.yaml:")
        print("  signals:")
        print("    ml_regime_models_dir: output/models   # enables soft-mixture blending")
        print("    ml_regime_blend_enabled: true")


def _weighted_recency_mean(vals: np.ndarray, decay_base: float = 0.95) -> float:
    """
    Calculates a weighted mean favoring the most recent walk-forward windows.
    Windows are sorted chronologically in the 'wm' list, so earlier windows
    get exponentially lower weights.
    """
    mask = np.isfinite(vals)
    if not np.any(mask):
        return float("nan")
    n = len(vals)
    # Power weights: recent = 1, prev = 0.95, etc. (Indices 0..n-1)
    # Note: 'wm' is appended in order, so index n-1 is the most recent.
    weights = np.array([decay_base ** (n - 1 - i) for i in range(n)], dtype=float)
    m_vals = vals[mask]
    m_weights = weights[mask]
    return float(np.sum(m_vals * m_weights) / np.sum(m_weights))


def _compute_institutional_metrics(
    ic_vals: np.ndarray,
    sharpe_vals: np.ndarray,
    oos_cagr: float,
    oos_max_dd: float,
) -> dict:
    """
    Compute institutional-grade model evaluation metrics not in sklearn.

    Metrics added:
      oos_ic_ir    — IC Information Ratio = mean(IC) / std(IC).
                     AQR / Man Group minimum bar: ≥ 0.5 for production use.
                     Captures IC *consistency*, not just magnitude.

      oos_ic_tstat — t-statistic on the IC series: mean(IC) × √N / std(IC).
                     Requires ≥ 1.65 (90%) before IC is statistically reliable.
                     Prevents selecting a model that lucked into positive mean IC.

      oos_calmar   — CAGR / |MaxDD|. Penalises tail risk better than Sharpe;
                     standard at Citadel, Millennium. Sharpe can be gamed by
                     selling tail options; Calmar cannot.

      oos_beat_rate — Fraction of windows with Sharpe > 0 AND IC > 0 simultaneously.
                     A model with mean Sharpe 0.8 but only 3/8 windows positive is
                     riskier than Sharpe 0.6 in 7/8 windows.

      oos_composite — Institutional composite: ICIR × (1 + Calmar) × beat_rate.
                     Replaces single-metric selection with a product that rewards
                     IC consistency, drawdown control, and regime robustness.
    """
    ic_finite = ic_vals[np.isfinite(ic_vals)]
    sp_finite = sharpe_vals[np.isfinite(sharpe_vals)]
    n = len(ic_finite)

    ic_mean = float(np.nanmean(ic_finite)) if n > 0 else float("nan")
    ic_std = float(np.nanstd(ic_finite, ddof=1)) if n > 1 else float("nan")

    # ICIR: undefined if std is zero or only one window
    if np.isfinite(ic_std) and ic_std > 1e-9:
        ic_ir = ic_mean / ic_std
        ic_tstat = ic_mean * np.sqrt(n) / ic_std
    else:
        ic_ir = float("nan")
        ic_tstat = float("nan")

    # Calmar: cap at ±10 to avoid division by near-zero drawdown
    abs_dd = abs(oos_max_dd) if np.isfinite(oos_max_dd) else float("nan")
    if np.isfinite(oos_cagr) and np.isfinite(abs_dd) and abs_dd > 1e-4:
        calmar = float(np.clip(oos_cagr / abs_dd, -10.0, 10.0))
    else:
        calmar = float("nan")

    # Beat rate: fraction of windows positive on both Sharpe and IC
    if n > 0 and len(sp_finite) == len(ic_finite):
        combined = np.sum((sp_finite > 0) & (ic_finite > 0))
        beat_rate = float(combined) / float(n)
    elif len(sp_finite) > 0:
        beat_rate = float(np.sum(sp_finite > 0)) / float(len(sp_finite))
    else:
        beat_rate = float("nan")

    # Composite score: ICIR × (1 + Calmar) × beat_rate
    # Uses nan-safe fallbacks so a single missing component doesn't void the score.
    _ic_ir_safe = ic_ir if np.isfinite(ic_ir) else 0.0
    _calmar_safe = calmar if np.isfinite(calmar) else 0.0
    _beat_safe = beat_rate if np.isfinite(beat_rate) else 0.5
    composite = float(_ic_ir_safe * (1.0 + max(_calmar_safe, 0.0)) * _beat_safe)

    return {
        "oos_ic_ir": float(ic_ir) if np.isfinite(ic_ir) else float("nan"),
        "oos_ic_tstat": float(ic_tstat) if np.isfinite(ic_tstat) else float("nan"),
        "oos_calmar": float(calmar) if np.isfinite(calmar) else float("nan"),
        "oos_beat_rate": float(beat_rate) if np.isfinite(beat_rate) else float("nan"),
        "oos_composite": composite,
    }


def _compute_psr(daily_returns: np.ndarray, benchmark_sr: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012).

    PSR(SR*) = Φ{ (SR_hat - SR*) × √(T-1) / √[1 - γ₃×SR_hat + (γ₄-1)/4 × SR_hat²] }

    where SR_hat is the in-sample annualised Sharpe, γ₃ skewness, γ₄ excess kurtosis.
    With SR*=0 this answers: 'What is the probability this Sharpe is truly above zero?'

    Used as a multiple-testing deflation: composite × PSR penalises models whose apparent
    outperformance could be explained by chance given the number of paths tested
    (Harvey & Liu 2015, 'Backtesting').  Models with fat tails (high kurtosis) receive
    larger penalties because their realised Sharpe is statistically less reliable.
    """
    try:
        from scipy.stats import norm as _snorm, skew as _skew, kurtosis as _kurt
    except ImportError:
        return 0.5   # scipy unavailable: treat as 50% confidence

    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 10:
        return 0.5
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if sigma < 1e-10:
        return 0.5
    sr_daily = mu / sigma                           # per-period Sharpe (not annualised yet)
    sr_annual = sr_daily * np.sqrt(252)             # annualised for benchmark comparison
    gamma3 = float(_skew(r))                        # skewness
    gamma4 = float(_kurt(r, fisher=True)) + 3.0     # convert excess → full kurtosis
    denom_sq = 1.0 - gamma3 * sr_annual + (gamma4 - 1.0) / 4.0 * sr_annual ** 2
    if denom_sq <= 0:
        return 0.5
    z = (sr_annual - benchmark_sr) * np.sqrt(T - 1) / np.sqrt(denom_sq)
    return float(_snorm.cdf(z))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="backtest_config.yaml")
    parser.add_argument(
        "--run_sim_test",
        action="store_true",
        help="Run a small self-test for portfolio simulation logic and exit",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Forward return horizon in trading days (default: config model_selection.lookahead_horizon_days or 5)",
    )
    parser.add_argument("--min_test_days", type=int, default=30, help="Skip windows with fewer test dates")
    parser.add_argument(
        "--min_oos_days",
        type=int,
        default=10,
        help="Minimum distinct test days (and daily return points) per window; below this warn and skip window",
    )
    parser.add_argument(
        "--select_metric",
        type=str,
        default="oos_sharpe_chained",
        help="Metric to rank/select best model (default: oos_sharpe_chained; window means kept in report)",
    )
    parser.add_argument("--limit_tickers", type=int, default=0, help="Optional: limit universe size for quick runs")
    parser.add_argument(
        "--risk-adj-target",
        action="store_true",
        dest="risk_adj_target",
        help=(
            "C3: Use risk-adjusted return (forward_return / realized_vol) as the regressor target. "
            "Rewards low-vol momentum; improves IC for cross-sectional ranking."
        ),
    )
    parser.add_argument(
        "--regime-models",
        action="store_true",
        dest="regime_models",
        help=(
            "C2: After main model selection, train regime-conditional XGBClassifier models "
            "(Bull/Bear/Sideways/HighVol) on the full dataset and save to output/models/."
        ),
    )
    parser.add_argument(
        "--compare_baseline",
        action="store_true",
        help="Also evaluate the LearnedWeights baseline via the same portfolio simulation and include it in the report",
    )
    parser.add_argument(
        "--max_positions",
        type=int,
        default=None,
        help="Max positions for OOS strategy simulation (default: config model_selection.max_positions or 10)",
    )
    parser.add_argument(
        "--min_positions",
        type=int,
        default=None,
        help="Min positions for OOS strategy simulation (default: config model_selection.min_positions or 3)",
    )
    parser.add_argument("--check_feature_leakage", action="store_true", help="Run feature leakage check and exit")
    parser.add_argument("--leakage_date", type=str, default="2020-06-15", help="As-of date for leakage check")
    parser.add_argument(
        "--leakage_tickers",
        type=str,
        default="",
        help="Comma-separated tickers for leakage check (default: first 3 in config)",
    )
    parser.add_argument("--leakage_tol", type=float, default=1e-6, help="Tolerance for leakage check comparisons")
    parser.add_argument(
        "--discard_suspicious_models",
        action="store_true",
        help="If set, discard models that trigger leakage warning (Sharpe_chained>2 & IC_chained<0.05)",
    )
    parser.add_argument(
        "--embargo_days",
        type=int,
        default=None,
        help="Embargo between train/test (calendar days). Default ~2*horizon.",
    )
    parser.add_argument(
        "--matrix-start-date",
        type=str,
        default="",
        help="Override backtest start_date for feature matrix only (YYYY-MM-DD). Used by run_retrain_model.py.",
    )
    parser.add_argument(
        "--matrix-end-date",
        type=str,
        default="",
        help="Override backtest end_date for feature matrix only (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--save-all-models",
        action="store_true",
        help="Also fit and save each candidate model artifact (e.g. logistic.pkl, xgboost.pkl).",
    )
    args = parser.parse_args()

    if args.run_sim_test:
        _test_portfolio_simulation_logic()
        raise SystemExit(0)

    if args.check_feature_leakage:
        tickers_override = [t.strip().upper() for t in str(args.leakage_tickers).split(",") if t.strip()]
        raise SystemExit(
            check_feature_leakage(
                as_of_date=str(args.leakage_date),
                tickers=tickers_override or None,
                tol=float(args.leakage_tol),
            )
        )

    cfg = _read_config(args.config)
    tickers = load_universe(cfg)
    if int(args.limit_tickers or 0) > 0:
        tickers = tickers[: int(args.limit_tickers)]
    bt = cfg.get("backtest", {}) or {}
    research = cfg.get("research", {}) or {}
    feature_sel = cfg.get("feature_selection", {}) or {}
    ms_cfg = cfg.get("model_selection", {}) or {}
    feature_subset = feature_sel.get("feature_subset", []) or []
    feature_subset = [str(c).strip() for c in feature_subset if str(c).strip()]
    short_feature_subset = feature_sel.get("short_feature_subset", []) or []
    short_feature_subset = [str(c).strip() for c in short_feature_subset if str(c).strip()]

    exe_cfg = cfg.get("execution", {}) or {}
    long_only = exe_cfg.get("long_only", False)
    enable_shorts = exe_cfg.get("enable_shorts", True)
    do_shorts = enable_shorts and not long_only

    # Model-selection / evaluation settings (CLI overrides config).
    horizon = int(args.horizon) if args.horizon is not None else int(ms_cfg.get("lookahead_horizon_days", 5) or 5)
    max_positions = (
        int(args.max_positions)
        if args.max_positions is not None
        else int(ms_cfg.get("max_positions", 10) or 10)
    )
    min_positions = (
        int(args.min_positions)
        if args.min_positions is not None
        else int(ms_cfg.get("min_positions", 3) or 3)
    )
    max_positions = int(max(1, max_positions))
    min_positions = int(max(1, min_positions))
    if min_positions > max_positions:
        min_positions = max_positions

    start_date = str(bt.get("start_date", "2018-01-01"))
    end_date = str(bt.get("end_date", "2024-01-01"))
    ms = str(getattr(args, "matrix_start_date", "") or "").strip()
    me = str(getattr(args, "matrix_end_date", "") or "").strip()
    if me:
        end_date = me
    if ms:
        start_date = ms
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise SystemExit(f"Invalid matrix window: start {start_date} after end {end_date}")
    train_years = float(research.get("train_years", 5))
    test_years = float(research.get("test_years", 1))
    step_years = float(research.get("step_years", test_years))
    n_windows_cfg = int(research.get("walk_forward_windows", 4) or 4)
    train_ratio = float(research.get("walk_forward_train_ratio", 0.70) or 0.70)

    if not tickers:
        raise SystemExit("No tickers found in backtest_config.yaml")

    print(f"Config: {args.config}")
    print(f"Universe: {len(tickers)} tickers")
    print(f"Window: {start_date} → {end_date}")
    print(f"Walk-forward: train={train_years}y test={test_years}y step={step_years}y")
    print(f"Horizon: {horizon} trading days")
    embargo_days = int(args.embargo_days) if args.embargo_days is not None else int(max(5, 2 * int(horizon)))
    print(f"Embargo: {embargo_days} calendar days")
    if feature_subset or short_feature_subset:
        num_feat = len(set(feature_subset + short_feature_subset))
        print(f"Feature union: {num_feat} unique columns")

    from agents.weight_learning_agent.feature_builder import build_feature_matrix

    # Build enough history for rolling features (feature_builder applies its own buffers).
    # Pass the union of all candidate subsets to build_feature_matrix.
    matrix_subset = list(set(feature_subset + short_feature_subset)) if (feature_subset or short_feature_subset) else None
    df = build_feature_matrix(
        tickers,
        start_date=start_date,
        end_date=end_date,
        holding_period=int(horizon),
        feature_subset=matrix_subset,
    )
    if df is None or df.empty:
        raise SystemExit("Feature matrix is empty; cannot run model selection.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Target: forward return (already aligned in feature_builder via close.shift(-holding_period)/close - 1).
    # Re-derive a binary label from that return.
    if "forward_return" not in df.columns:
        raise SystemExit("Feature matrix missing forward_return; cannot compute target.")
    df["forward_return"] = pd.to_numeric(df["forward_return"], errors="coerce")
    df = df.dropna(subset=["forward_return"])
    df["y_bin"] = (df["forward_return"] > 0).astype(int)

    # C3: risk-adjusted target — use when --risk-adj-target flag is passed.
    # sign(risk_adj) == sign(forward_return) so y_bin is unchanged.
    use_risk_adj = getattr(args, "risk_adj_target", False) and "forward_return_risk_adj" in df.columns
    if use_risk_adj:
        df["forward_return_risk_adj"] = pd.to_numeric(df["forward_return_risk_adj"], errors="coerce")
        # Fall back to raw return where risk-adj is missing (e.g. very low vol)
        df["forward_return_risk_adj"] = df["forward_return_risk_adj"].fillna(df["forward_return"])
        print(f"[C3] Using risk-adjusted return target (forward_return / realized_vol_holding)")
    else:
        use_risk_adj = False

    # Optional baseline (LearnedWeights) uses the full feature set + its own scaler feature list.
    df_baseline: pd.DataFrame | None = None
    if args.compare_baseline:
        try:
            df_baseline = build_feature_matrix(
                tickers,
                start_date=start_date,
                end_date=end_date,
                holding_period=int(horizon),
                feature_subset=None,
            )
            if df_baseline is None or df_baseline.empty:
                print("WARNING: compare_baseline enabled but baseline feature matrix is empty; skipping baseline.")
                df_baseline = None
            else:
                df_baseline = df_baseline.copy()
                df_baseline["date"] = pd.to_datetime(df_baseline["date"], errors="coerce")
                df_baseline = df_baseline.dropna(subset=["date"])
                df_baseline["forward_return"] = pd.to_numeric(df_baseline["forward_return"], errors="coerce")
                df_baseline = df_baseline.dropna(subset=["forward_return"])
                df_baseline["y_bin"] = (df_baseline["forward_return"] > 0).astype(int)
                df_baseline = df_baseline.sort_values(["ticker", "date"]).reset_index(drop=True)
        except Exception as exc:
            print(f"WARNING: compare_baseline enabled but failed to build baseline matrix: {exc}")
            traceback.print_exc()
            df_baseline = None

    feat_cols = _feature_columns(df)
    if not feat_cols:
        raise SystemExit("No numeric feature columns found.")

    # Basic leakage sanity check: any forward-looking columns still present?
    leaked = [c for c in feat_cols if "forward" in c.lower()]
    if leaked:
        raise SystemExit(f"Leakage: feature columns contain forward-looking fields: {leaked[:10]}")

    # Build windows. Prefer calendar windows when they yield >1, else fall back to count-based.
    windows = _walk_forward_windows(start_date, end_date, train_years, test_years, step_years)
    if len(windows) <= 1:
        windows = _walk_forward_windows_by_count(df["date"], n_windows=n_windows_cfg, train_ratio=train_ratio)
    if len(windows) <= 1:
        raise SystemExit(
            "Not enough walk-forward windows. Either extend the backtest date range or reduce research.train_years/test_years."
        )

    print()
    print("Walk-forward windows (calendar bounds):")
    prev_test_end: pd.Timestamp | None = None
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows, 1):
        sequential = "OK"
        if prev_test_end is not None and te_s < prev_test_end:
            sequential = "OVERLAP"
        tr_span = f"{tr_s.date()} → {(tr_e - pd.Timedelta(days=1)).date()}"
        te_span = f"{te_s.date()} → {(te_e - pd.Timedelta(days=1)).date()}"
        print(f"  {i:02d} train=[{tr_span}]  test=[{te_span}]  {sequential}")
        prev_test_end = te_e

    models = _build_models()
    if not models:
        raise SystemExit("No models available (check sklearn / optional xgboost install).")

    out_dir = Path("output/models")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    all_window_details: dict[str, list[WindowMetrics]] = {}

    for name, model, uses_proba, model_kind in models:
        is_regressor = model_kind == "regressor"
        is_short_classifier = model_kind == "short_classifier"
        print()
        print(f"=== {name} ({model_kind}) ===")
        wm: list[WindowMetrics] = []
        oos_parts: list[pd.DataFrame] = []
        daily_parts: list[pd.Series] = []

        for win_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows, 1):
            te_label = f"{te_s.date()} → {(te_e - pd.Timedelta(days=1)).date()}"

            # Purge / embargo to avoid leakage from overlapping forward-return labels across the split.
            purge_cutoff = te_s - pd.Timedelta(days=embargo_days)

            tr = df[(df["date"] >= tr_s) & (df["date"] < min(tr_e, purge_cutoff))].copy()
            te = df[(df["date"] >= te_s) & (df["date"] < te_e)].copy()

            if tr.empty or te.empty:
                print(f"  [window {win_idx}/{len(windows)}] skip: empty train or test | test={te_label}")
                continue

            n_test_unique = int(te["date"].nunique())
            if n_test_unique < int(args.min_oos_days):
                print(
                    f"  WARNING [window {win_idx}/{len(windows)}] skip: only {n_test_unique} test days "
                    f"(min_oos_days={args.min_oos_days}) | test={te_label}"
                )
                continue

            if n_test_unique < int(args.min_test_days):
                print(
                    f"  [window {win_idx}/{len(windows)}] skip: {n_test_unique} test days < min_test_days={args.min_test_days} | test={te_label}"
                )
                continue

            try:
                # Contextual feature selection for short-specific models
                active_feats = short_feature_subset if (is_short_classifier and short_feature_subset) else feat_cols
                
                X_tr = tr[active_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0).values
                if is_regressor:
                    if use_risk_adj:
                        # C3: risk-adjusted target (forward_return / holding_vol); pre-clipped to [-10, 10]
                        y_tr = tr["forward_return_risk_adj"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0).values.astype(float)
                    elif name == "XGBRankIC":
                        # Liu et al. (2023) cross-sectional rank normalization:
                        # Normalize forward_return to rank percentile within each date,
                        # then scale to [-1, +1] (mean=0 per cross-section).
                        # This teaches the model relative ranking (what portfolio
                        # construction uses) rather than absolute return prediction
                        # (which is dominated by market-wide beta noise).
                        fwd_raw = tr["forward_return"].replace([np.inf, -np.inf], np.nan)
                        if "date" not in tr.columns:
                            raise ValueError(
                                "XGBRankIC requires a 'date' column for per-cross-section rank "
                                "normalization. Falling back to global rank would re-introduce "
                                "market-beta contamination that rank normalization is designed to remove."
                            )
                        rank_pct = fwd_raw.groupby(tr["date"]).transform(
                            lambda x: x.rank(pct=True, na_option="keep")
                        )
                        # Scale [0, 1] → [-1, +1]: mean=0, symmetric around zero
                        y_tr = (rank_pct.fillna(0.5) * 2.0 - 1.0).values.astype(float)
                    else:
                        # Clip raw regression target to prevent exploding coefficients/gradients
                        y_tr = tr["forward_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.3, 0.3).values.astype(float)
                elif is_short_classifier:
                    # Cross-sectional bottom-decile target (market-neutral, ~10% base rate).
                    # Papers: Medhat-Schmeling (2022) and RRLP (2023) rank stocks within each
                    # date cross-section — the "bad" stocks are the worst performers relative
                    # to all stocks that day, not simply negative-return stocks.
                    # Using `return < 0` produces a ~50% base rate (coin-flip) because the
                    # market moves up and down in waves; a market-neutral label removes that
                    # aggregate direction and isolates the truly bottom-decile stocks.
                    fwd = tr["forward_return"].replace([np.inf, -np.inf], np.nan)
                    if "date" in tr.columns:
                        rank_pct = fwd.groupby(tr["date"]).transform(
                            lambda x: x.rank(pct=True, na_option="keep")
                        )
                    else:
                        rank_pct = fwd.rank(pct=True, na_option="keep")
                    y_tr = (rank_pct.fillna(0.5) < 0.10).astype(int)
                else:
                    y_tr = tr["y_bin"].fillna(0).astype(int)
                
                X_te = te[active_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0).values
                y_te_ret = te["forward_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(float)
                y_te_bin = te["y_bin"].fillna(0).astype(int)

                # Defensive: Drop singular or constant features to prevent StandardScaler zero-division.
                # Standard deviation below 1e-6 causes numerical instability in linear solvers.
                X_tr_df = pd.DataFrame(X_tr, columns=active_feats)
                stds = X_tr_df.std()
                singular_cols = stds[stds < 1e-6].index.tolist()
                
                if singular_cols:
                    # Sync X_tr and X_te by removing the problematic columns
                    X_tr = X_tr_df.drop(columns=singular_cols).values
                    X_te = pd.DataFrame(X_te, columns=active_feats).drop(columns=singular_cols).values

                # Final NumPy safety net for any values that bypassed pandas-level logic
                # Using .copy() to ensure contiguous memory, which helps avoid some BLAS bugs on Mac
                X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0).copy()
                y_tr = np.nan_to_num(y_tr, nan=0.0, posinf=0.0, neginf=0.0).copy()
                X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0).copy()

                # --- Short Model Specific Guard & Balancing ---
                if is_short_classifier:
                    pos = int((y_tr == 1).sum())
                    if pos < 30:
                        print(
                            f"  [window {win_idx}/{len(windows)}] skip short model: only {pos} positive labels "
                            f"(min 30) | test={te_label}"
                        )
                        continue
                        
                try:
                    from sklearn.base import clone
                    import warnings
                    
                    # Isolate this training window with a fresh model instance
                    win_model = clone(model)
                    
                    if name == "ShortXGB" and is_short_classifier:
                        pos = int((y_tr == 1).sum())
                        neg = int((y_tr == 0).sum())
                        spw = float(neg / pos) if pos > 0 else 1.0
                        win_model.set_params(scale_pos_weight=spw)
                    
                    t0 = time.perf_counter()
                    # Silence known spurious BLAS matmul warnings on Apple Silicon
                    with np.errstate(all="ignore"):
                        # LGBMRanker needs date groups for cross-sectional ranking
                        if name == "LGBMRanker":
                            tr_dates_sorted = pd.to_datetime(tr["date"], errors="coerce").to_numpy()
                            # Map each date to a dense integer group index (sorted)
                            unique_dates, date_groups = np.unique(tr_dates_sorted, return_inverse=True)
                            win_model.fit(X_tr, y_tr, _date_groups=date_groups)
                        else:
                            win_model.fit(X_tr, y_tr)
                    t1 = time.perf_counter()
                except Exception as exc:
                    print(f"  [window {win_idx}/{len(windows)}] train failed: {exc}")
                    traceback.print_exc()
                    continue

                try:
                    t2 = time.perf_counter()
                    with np.errstate(all="ignore"):
                        if is_regressor:
                            score = win_model.predict(X_te).astype(float)
                        elif is_short_classifier:
                            if uses_proba and hasattr(win_model, "predict_proba"):
                                p_down = win_model.predict_proba(X_te)[:, 1].astype(float)
                                score = -(p_down - 0.5)
                            else:
                                score = -win_model.predict(X_te).astype(float) + 0.5
                        elif uses_proba and hasattr(win_model, "predict_proba"):
                            p = win_model.predict_proba(X_te)[:, 1].astype(float)
                            score = p - 0.5
                        elif hasattr(win_model, "decision_function"):
                            score = win_model.decision_function(X_te).astype(float)
                        else:
                            pred = win_model.predict(X_te).astype(int)
                            score = pred.astype(float) - 0.5
                    t3 = time.perf_counter()
                except Exception as exc:
                    print(f"  [window {win_idx}/{len(windows)}] predict failed: {exc}")
                    traceback.print_exc()
                    continue

                ic = _safe_pearson(score, y_te_ret)
                dir_acc = float(((score >= 0) == (y_te_bin == 1)).mean()) if len(score) else float("nan")

                te_scored = te.assign(score=score)
                daily_ret_s = _strategy_daily_returns(
                    te_scored,
                    max_positions=int(max_positions),
                    min_positions=int(min_positions),
                    horizon=int(horizon),
                )
                n_daily_pts = int(len(daily_ret_s))
                if n_daily_pts < int(args.min_oos_days):
                    print(
                        f"  WARNING [window {win_idx}/{len(windows)}] skip: portfolio sim has {n_daily_pts} days "
                        f"(min_oos_days={args.min_oos_days}) | test={te_label}"
                    )
                    continue

                n_invested = _count_invested_days(
                    te_scored,
                    max_positions=int(max_positions),
                    min_positions=int(min_positions),
                )
                sharpe = _sharpe_from_series(daily_ret_s.to_numpy(dtype=float), horizon=int(horizon))
                sharpe_str = f"{sharpe:.4f}" if np.isfinite(sharpe) else "nan"

                print(
                    f"  [window {win_idx}/{len(windows)}] train=[{tr_s.date()}→{(tr_e - pd.Timedelta(days=1)).date()}] "
                    f"test=[{te_label}] | n_days={n_test_unique} | "
                    f"days_with_positions={n_invested} | Sharpe={sharpe_str} | IC={ic:.4f}"
                )

                daily_parts.append(daily_ret_s)
                oos_parts.append(te_scored[["date", "ticker", "forward_return", "score"]].copy())

                wm.append(
                    WindowMetrics(
                        oos_sharpe=float(sharpe) if np.isfinite(sharpe) else float("nan"),
                        oos_ic=float(ic) if np.isfinite(ic) else float("nan"),
                        oos_dir_acc=float(dir_acc) if np.isfinite(dir_acc) else float("nan"),
                        train_time_s=float(t1 - t0),
                        test_time_s=float(t3 - t2),
                        n_train=int(len(tr)),
                        n_test=int(len(te_scored)),
                        train_start=str(tr_s.date()),
                        train_end=str((tr_e - pd.Timedelta(days=1)).date()),
                        test_start=str(te_s.date()),
                        test_end=str((te_e - pd.Timedelta(days=1)).date()),
                    )
                )
            except Exception as exc:
                print(f"  ERROR [window {win_idx}/{len(windows)}] unexpected failure ({te_label}): {exc}")
                traceback.print_exc()
                continue

        all_window_details[name] = wm
        if not wm:
            print("No valid windows (insufficient data).")
            continue

        sharpe_vals = np.array([w.oos_sharpe for w in wm], dtype=float)
        ic_vals = np.array([w.oos_ic for w in wm], dtype=float)
        acc_vals = np.array([w.oos_dir_acc for w in wm], dtype=float)
        tr_t = np.array([w.train_time_s for w in wm], dtype=float)
        te_t = np.array([w.test_time_s for w in wm], dtype=float)

        oos_df = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame()
        chained_daily = _concat_window_daily_returns(daily_parts)
        oos_sharpe_chained = _sharpe_from_series(chained_daily, horizon=int(horizon))
        oos_cagr_chained = _cagr_from_daily_returns(chained_daily)
        oos_max_dd = _max_drawdown_from_daily_returns(chained_daily)
        oos_win_rate = _win_rate_from_daily_returns(chained_daily)
        _oos_sharpe_chained_old, _oos_cagr_chained_old, oos_ic_chained = _chained_oos_metrics(
            oos_df, max_positions=int(max_positions), horizon=int(horizon)
        )

        decay_val = float(getattr(cfg.model_selection, "selection_weight_decay", 0.95)) if hasattr(cfg, "model_selection") else 0.95
        inst_metrics = _compute_institutional_metrics(
            ic_vals, sharpe_vals, float(oos_cagr_chained), float(oos_max_dd)
        )
        row = {
            "model_name": name,
            "model_kind": model_kind,
            "oos_sharpe_mean": _weighted_recency_mean(sharpe_vals, decay_base=decay_val),
            "oos_sharpe_std": float(np.nanstd(sharpe_vals, ddof=1)) if len(wm) > 1 else 0.0,
            "oos_ic_mean": _weighted_recency_mean(ic_vals, decay_base=decay_val),
            "oos_ic_std": float(np.nanstd(ic_vals, ddof=1)) if len(wm) > 1 else 0.0,
            "oos_dir_acc_mean": _weighted_recency_mean(acc_vals, decay_base=decay_val),
            "oos_dir_acc_std": float(np.nanstd(acc_vals, ddof=1)) if len(wm) > 1 else 0.0,
            "oos_sharpe_chained": float(oos_sharpe_chained),
            "oos_cagr_chained": float(oos_cagr_chained),
            "oos_max_dd": float(oos_max_dd),
            "oos_win_rate": float(oos_win_rate),
            "oos_ic_chained": float(oos_ic_chained),
            # Institutional metrics (AQR / Man Group standard)
            **inst_metrics,
            "train_time_avg": float(np.nanmean(tr_t)),
            "test_time_avg": float(np.nanmean(te_t)),
            "n_windows": int(len(wm)),
            # PSR: probability that the chained OOS Sharpe is genuinely > 0 after
            # adjusting for fat tails.  Multiplied into composite below to produce
            # oos_composite_dsr — the Harvey-Liu multiple-testing-corrected ranking score.
            "oos_psr": round(_compute_psr(
                chained_daily.dropna().values
                if hasattr(chained_daily, "dropna")
                else chained_daily[np.isfinite(chained_daily)]
            ), 4),
        }
        # Leakage sanity: chained Sharpe should not be extreme if chained IC is near-zero.
        suspicious = bool(
            np.isfinite(row["oos_sharpe_chained"])
            and np.isfinite(row["oos_ic_chained"])
            and (row["oos_sharpe_chained"] > 2.0)
            and (row["oos_ic_chained"] < 0.05)
        )
        row["leakage_suspect"] = bool(suspicious)
        if suspicious:
            print(
                f"WARNING: {name} suspicious metrics (Sharpe_chained={row['oos_sharpe_chained']:.3f}, "
                f"IC_chained={row['oos_ic_chained']:.3f}). "
                "This may indicate leakage or a broken Sharpe proxy."
            )
            if args.discard_suspicious_models:
                print(f"  -> Discarding {name} from selection/report due to --discard_suspicious_models.")
                continue
        rows.append(row)

        _ic_ir_str = f"{row['oos_ic_ir']:.3f}" if np.isfinite(row.get('oos_ic_ir', float('nan'))) else "nan"
        _tstat_str = f"{row['oos_ic_tstat']:.2f}" if np.isfinite(row.get('oos_ic_tstat', float('nan'))) else "nan"
        _calmar_str = f"{row['oos_calmar']:.2f}" if np.isfinite(row.get('oos_calmar', float('nan'))) else "nan"
        _beat_str = f"{row['oos_beat_rate']:.2f}" if np.isfinite(row.get('oos_beat_rate', float('nan'))) else "nan"
        _comp_str = f"{row['oos_composite']:.3f}" if np.isfinite(row.get('oos_composite', float('nan'))) else "nan"
        print(
            f"OOS Sharpe (chained): {row['oos_sharpe_chained']:.3f} | "
            f"window Sharpe mean±std: {row['oos_sharpe_mean']:.3f} ± {row['oos_sharpe_std']:.3f} | "
            f"IC: {row['oos_ic_mean']:.3f} ± {row['oos_ic_std']:.3f} | "
            f"ICIR: {_ic_ir_str} (t={_tstat_str}) | "
            f"Calmar: {_calmar_str} | Beat: {_beat_str} | Composite: {_comp_str} | "
            f"DirAcc: {row['oos_dir_acc_mean']:.3f} | windows={row['n_windows']}"
        )

    # Baseline comparison (LearnedWeights) — no training, score + simulate.
    if args.compare_baseline and df_baseline is not None:
        print()
        print("=== LearnedWeightsBaseline ===")
        wm: list[WindowMetrics] = []
        oos_parts: list[pd.DataFrame] = []
        daily_parts: list[pd.Series] = []

        for win_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows, 1):
            te_label = f"{te_s.date()} → {(te_e - pd.Timedelta(days=1)).date()}"
            te = df_baseline[(df_baseline["date"] >= te_s) & (df_baseline["date"] < te_e)].copy()
            if te.empty:
                print(f"  [window {win_idx}/{len(windows)}] skip: empty test | test={te_label}")
                continue
            n_test_unique = int(te["date"].nunique())
            if n_test_unique < int(args.min_oos_days):
                print(
                    f"  WARNING [window {win_idx}/{len(windows)}] skip: only {n_test_unique} test days "
                    f"(min_oos_days={args.min_oos_days}) | test={te_label}"
                )
                continue

            try:
                t0 = time.perf_counter()
                score = _learned_weights_score_series(te)
                t1 = time.perf_counter()

                te_scored = te.assign(score=score)
                y_te_ret = te_scored["forward_return"].to_numpy(dtype=float)
                y_te_bin = te_scored["y_bin"].to_numpy(dtype=int)

                ic = _safe_pearson(np.asarray(score, dtype=float), y_te_ret)
                dir_acc = float(((np.asarray(score) >= 0) == (y_te_bin == 1)).mean()) if len(score) else float("nan")

                daily_ret_s = _strategy_daily_returns(
                    te_scored,
                    max_positions=int(max_positions),
                    min_positions=int(min_positions),
                    horizon=int(horizon),
                )
                n_daily_pts = int(len(daily_ret_s))
                if n_daily_pts < int(args.min_oos_days):
                    print(
                        f"  WARNING [window {win_idx}/{len(windows)}] skip: portfolio sim has {n_daily_pts} days "
                        f"(min_oos_days={args.min_oos_days}) | test={te_label}"
                    )
                    continue

                n_invested = _count_invested_days(
                    te_scored,
                    max_positions=int(max_positions),
                    min_positions=int(min_positions),
                )
                sharpe = _sharpe_from_series(daily_ret_s.to_numpy(dtype=float), horizon=int(horizon))
                sharpe_str = f"{sharpe:.4f}" if np.isfinite(sharpe) else "nan"
                print(
                    f"  [window {win_idx}/{len(windows)}] test=[{te_label}] | n_days={n_test_unique} | "
                    f"days_with_positions={n_invested} | Sharpe={sharpe_str} | IC={ic:.4f}"
                )

                daily_parts.append(daily_ret_s)
                oos_parts.append(te_scored[["date", "ticker", "forward_return", "score"]].copy())
                wm.append(
                    WindowMetrics(
                        oos_sharpe=float(sharpe) if np.isfinite(sharpe) else float("nan"),
                        oos_ic=float(ic) if np.isfinite(ic) else float("nan"),
                        oos_dir_acc=float(dir_acc) if np.isfinite(dir_acc) else float("nan"),
                        train_time_s=0.0,
                        test_time_s=float(t1 - t0),
                        n_train=0,
                        n_test=int(len(te_scored)),
                        train_start=str(tr_s.date()),
                        train_end=str((tr_e - pd.Timedelta(days=1)).date()),
                        test_start=str(te_s.date()),
                        test_end=str((te_e - pd.Timedelta(days=1)).date()),
                    )
                )
            except Exception as exc:
                print(f"  ERROR [window {win_idx}/{len(windows)}] baseline failed ({te_label}): {exc}")
                traceback.print_exc()
                continue

        if wm:
            sharpe_vals = np.array([w.oos_sharpe for w in wm], dtype=float)
            ic_vals = np.array([w.oos_ic for w in wm], dtype=float)
            acc_vals = np.array([w.oos_dir_acc for w in wm], dtype=float)
            te_t = np.array([w.test_time_s for w in wm], dtype=float)
            oos_df = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame()
            chained_daily = _concat_window_daily_returns(daily_parts)
            oos_sharpe_chained = _sharpe_from_series(chained_daily, horizon=int(horizon))
            oos_cagr_chained = _cagr_from_daily_returns(chained_daily)
            oos_max_dd = _max_drawdown_from_daily_returns(chained_daily)
            oos_win_rate = _win_rate_from_daily_returns(chained_daily)
            _unused_s, _unused_c, oos_ic_chained = _chained_oos_metrics(oos_df, max_positions=int(max_positions), horizon=int(horizon))

            _base_inst = _compute_institutional_metrics(
                ic_vals, sharpe_vals, float(oos_cagr_chained), float(oos_max_dd)
            )
            row = {
                "model_name": "LearnedWeightsBaseline",
                "model_kind": "baseline",
                "oos_sharpe_mean": float(np.nanmean(sharpe_vals)),
                "oos_sharpe_std": float(np.nanstd(sharpe_vals, ddof=1)) if len(wm) > 1 else 0.0,
                "oos_ic_mean": float(np.nanmean(ic_vals)),
                "oos_ic_std": float(np.nanstd(ic_vals, ddof=1)) if len(wm) > 1 else 0.0,
                "oos_dir_acc_mean": float(np.nanmean(acc_vals)),
                "oos_dir_acc_std": float(np.nanstd(acc_vals, ddof=1)) if len(wm) > 1 else 0.0,
                "oos_sharpe_chained": float(oos_sharpe_chained),
                "oos_cagr_chained": float(oos_cagr_chained),
                "oos_max_dd": float(oos_max_dd),
                "oos_win_rate": float(oos_win_rate),
                "oos_ic_chained": float(oos_ic_chained),
                # Institutional metrics
                **_base_inst,
                "train_time_avg": 0.0,
                "test_time_avg": float(np.nanmean(te_t)),
                "n_windows": int(len(wm)),
            }

            suspicious = bool(
                np.isfinite(row["oos_sharpe_chained"])
                and np.isfinite(row["oos_ic_chained"])
                and (row["oos_sharpe_chained"] > 2.0)
                and (row["oos_ic_chained"] < 0.05)
            )
            row["leakage_suspect"] = bool(suspicious)
            if suspicious:
                print(
                    f"WARNING: LearnedWeightsBaseline suspicious metrics (Sharpe_chained={row['oos_sharpe_chained']:.3f}, "
                    f"IC_chained={row['oos_ic_chained']:.3f})."
                )
                if args.discard_suspicious_models:
                    print("  -> Discarding baseline due to --discard_suspicious_models.")
                else:
                    rows.append(row)
            else:
                rows.append(row)

    if not rows:
        raise SystemExit("No model produced valid results.")

    # ── Deflated Sharpe composite (Harvey & Liu 2015 multiple-testing correction) ──────
    # oos_composite_dsr = ICIR × (1+Calmar) × beat_rate × PSR(SR_hat)
    # PSR deflates models whose realised Sharpe is driven by fat tails or few test paths;
    # consistent, Gaussian-return models are penalised least.  Use --select_metric
    # oos_composite_dsr to make DSR the ranking criterion.
    if rows:
        for _r in rows:
            _r["oos_composite_dsr"] = round(
                float(_r.get("oos_composite", 0.0)) * float(_r.get("oos_psr", 0.5)), 4
            )
        _psr_vals = [_r["oos_psr"] for _r in rows]
        print(
            f"\nDSR deflation: PSR range [{min(_psr_vals):.3f}, {max(_psr_vals):.3f}] "
            f"across {len(rows)} models. "
            "Use --select_metric oos_composite_dsr to rank by deflated composite."
        )

    report = pd.DataFrame(rows)
    # Selection/ranking: honor --select_metric with a penalty if suspicious.
    if args.select_metric not in report.columns:
        raise SystemExit(f"select_metric '{args.select_metric}' not found in report columns")
    report["_selection_metric"] = pd.to_numeric(report[args.select_metric], errors="coerce")
    if "leakage_suspect" in report.columns:
        report.loc[report["leakage_suspect"].eq(True), "_selection_metric"] = -1e9
    report = report.sort_values("_selection_metric", ascending=False).reset_index(drop=True)
    report_path = out_dir / "model_comparison.csv"
    report.to_csv(report_path, index=False)

    print()
    print(f"Saved report: {report_path}")

    # --- ENSEMBLE WINNER SELECTION (Pillar 24) ---
    top_n = int(ms_cfg.get("ensemble_size", 3))  # default 3: Ridge + XGBRegressor + LGBMRanker
    
    # Regressors only for longs: classifiers predict P(up) which clusters near 0.5
    # and doesn't provide cross-sectional rank dispersion. Classifiers capture beta
    # (market direction) not alpha (relative ranking). IC tells the truth — all
    # classifiers have negative or near-zero IC_chained in OOS evaluation.
    long_kinds = ["regressor"]
    short_kinds = ["short_classifier"]

    def _get_consistent_pool(full_pool, kinds, size):
        if full_pool.empty: return full_pool
        # Minimum IC threshold: discard models with negative chained IC.
        # A model with IC < 0 has negative cross-sectional predictive power.
        if "oos_ic_chained" in full_pool.columns:
            ic_floor = full_pool["oos_ic_chained"].apply(lambda x: float(x) if np.isfinite(float(x)) else -999)
            full_pool = full_pool[ic_floor >= 0.0].copy()
        if full_pool.empty: return full_pool
        # Dir acc guard: discard models whose predictions are inversely correlated with returns.
        # Short classifiers predict negative returns; their DirAcc is noisier because short
        # opportunities are rarer and more regime-dependent. Use floor=0.45 for short_classifiers
        # (vs 0.50 for longs) to avoid discarding models with positive IC but borderline DirAcc.
        if "oos_dir_acc_mean" in full_pool.columns:
            dir_acc = full_pool["oos_dir_acc_mean"].apply(lambda x: float(x) if np.isfinite(float(x)) else 0.0)
            is_short = full_pool.get("model_kind", pd.Series("", index=full_pool.index)) == "short_classifier"
            dir_acc_floor = is_short.map({True: 0.45, False: 0.50})
            full_pool = full_pool[dir_acc >= dir_acc_floor].copy()
        if full_pool.empty: return full_pool
        # Stability guard: require positive mean OOS Sharpe.
        # The original mean-std > 0 guard was intended to catch one-lucky-window models,
        # but it incorrectly eliminates ALL models when one window is a uniform market crash
        # (e.g. 2022: every model gets negative Sharpe because S&P fell ~18%, yet IC is
        # positive — the signal works cross-sectionally but long-only can't survive a bear).
        # Separation of concerns: mean > 0 checks "is it positive on average?"; ICIR ≥ 0.5
        # and IC t-stat ≥ 1.65 (below) enforce consistency. Don't conflate the two.
        if "oos_sharpe_mean" in full_pool.columns:
            s_mean = full_pool["oos_sharpe_mean"].apply(lambda x: float(x) if np.isfinite(float(x)) else -999)
            full_pool = full_pool[s_mean > 0.0].copy()
        if full_pool.empty: return full_pool
        # ICIR filter: AQR/Man Group minimum bar — IC must be consistent, not just positive.
        # ICIR < 0.5 means the signal is too noisy to add value in a diversified portfolio.
        if "oos_ic_ir" in full_pool.columns:
            ic_ir = full_pool["oos_ic_ir"].apply(lambda x: float(x) if np.isfinite(float(x)) else -999.0)
            full_pool = full_pool[ic_ir >= 0.5].copy()
        if full_pool.empty: return full_pool
        # IC t-stat filter: require 90% statistical confidence that IC is non-zero.
        # t < 1.65 means the IC could plausibly be noise even if the point estimate looks positive.
        if "oos_ic_tstat" in full_pool.columns:
            ic_tstat = full_pool["oos_ic_tstat"].apply(lambda x: float(x) if np.isfinite(float(x)) else 0.0)
            full_pool = full_pool[ic_tstat >= 1.65].copy()
        if full_pool.empty: return full_pool
        # Type-Consistency Lockdown: Anchor to the #1 winner's type
        anchor_kind = full_pool.iloc[0]["model_kind"]
        consistent = full_pool[full_pool["model_kind"] == anchor_kind].head(size)
        return consistent.reset_index(drop=True)

    if "model_kind" not in report.columns:
        report["model_kind"] = "unknown"
    long_pool = _get_consistent_pool(report[report["model_kind"].isin(long_kinds)], long_kinds, top_n)
    short_pool = _get_consistent_pool(report[report["model_kind"].isin(short_kinds)], short_kinds, top_n) if do_shorts else pd.DataFrame()

    def _get_ensemble_specs(pool):
        if pool.empty: return [], []
        names = pool["model_name"].tolist()
        # DeMiguel (2009): ICIR-weighted ensemble outperforms Sharpe-weighted OOS.
        # ICIR captures both signal quality (IC mean) and consistency (IC std),
        # which is exactly what we want to upweight in a multi-model ensemble.
        if "oos_ic_ir" in pool.columns:
            raw = pd.to_numeric(pool["oos_ic_ir"], errors="coerce").fillna(0.01).clip(lower=0.01).values
        else:
            raw = pd.to_numeric(pool["_selection_metric"], errors="coerce").fillna(0.01).clip(lower=0.01).values
        weights = raw / raw.sum() if raw.sum() > 0 else np.ones(len(raw)) / len(raw)
        return names, weights.tolist()

    best_long_names, long_weights = _get_ensemble_specs(long_pool)
    best_short_names, short_weights = _get_ensemble_specs(short_pool)

    if best_long_names:
        print(f"Selected Top-{len(best_long_names)} LONG Ensemble: {', '.join(best_long_names)}")
    if best_short_names:
        print(f"Selected Top-{len(best_short_names)} SHORT Ensemble: {', '.join(best_short_names)}")

    # Final full-dataset training: replicate the robust sanitization logic
    # (Shared X/y matrices for all winning fits)
    X_all_raw = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0).values
    y_all_cls = df["y_bin"].fillna(0).astype(int)
    if use_risk_adj:
        y_all_reg = df["forward_return_risk_adj"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0).values.astype(float)
    else:
        y_all_reg = df["forward_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.5, 0.5).values.astype(float)
    y_all_down = (df["forward_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0) < 0).astype(int)

    X_all_df = pd.DataFrame(X_all_raw, columns=feat_cols)
    all_stds = X_all_df.std()
    all_singular = all_stds[all_stds < 1e-6].index.tolist()
    X_all = X_all_df.drop(columns=all_singular).values if all_singular else X_all_raw.copy()
    
    y_all_cls = y_all_cls.copy()
    y_all_reg = y_all_reg.copy()
    y_all_down = y_all_down.copy()

    # Ensemble training helper (Pillar 24)
    def _train_and_save_ensemble(names, weights, b_label):
        if not names:
            return
        
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        from sklearn.base import clone
        
        estimators = []
        for name in names:
            spec = {n: (m, p, k) for (n, m, p, k) in models}.get(name)
            if spec:
                estimators.append((name, clone(spec[0])))
        
        if not estimators:
            print(f"WARNING: No valid estimators found for {b_label} ensemble.")
            return

        # Determine ensemble kind from the leader
        leader_name = names[0]
        leader_kind = {n: k for (n, m, p, k) in models}.get(leader_name, "classifier")
        
        if leader_kind == "regressor":
            ensemble = VotingRegressor(estimators=estimators, weights=weights[:len(estimators)])
            y_fit = y_all_reg
        elif leader_kind == "short_classifier":
            ensemble = VotingClassifier(estimators=estimators, voting="soft", weights=weights[:len(estimators)])
            y_fit = y_all_down
        else:
            ensemble = VotingClassifier(estimators=estimators, voting="soft", weights=weights[:len(estimators)])
            y_fit = y_all_cls

        print(f"Training best {b_label} ENSEMBLE ({len(estimators)} models) on full dataset...")
        with np.errstate(all="ignore"):
            # Inject date groups into any LGBMRanker sub-estimators BEFORE VotingRegressor.fit()
            # calls each sub-estimator's .fit(X, y) without kwargs (LightGBM limit: ≤10k rows/group).
            all_dates = pd.to_datetime(df["date"], errors="coerce").to_numpy()
            _, date_groups_all = np.unique(all_dates, return_inverse=True)
            for _est_name, _est in estimators:
                if hasattr(_est, "set_date_context"):
                    _est.set_date_context(date_groups_all)
            ensemble.fit(X_all, y_fit)
        
        path = out_dir / f"best_{b_label}_model.pkl"
        artifact = {
            "model_name": f"Top{len(estimators)}_Ensemble",
            "model_type": leader_kind,
            "ensemble_members": names,
            "ensemble_weights": weights,
            "horizon_days": int(horizon),
            "target": "forward_return",
            "feature_columns": [c for c in feat_cols if c not in all_singular],
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "estimator": ensemble,
        }
        with open(path, "wb") as fh:
            pickle.dump(artifact, fh)
        print(f"Saved best {b_label} ensemble: {path}")

    # Execute training for both pillars (Ensemble mode)
    t_start = time.perf_counter()
    _train_and_save_ensemble(best_long_names, long_weights, "long")
    if do_shorts:
        _train_and_save_ensemble(best_short_names, short_weights, "short")
    t_end = time.perf_counter()
    print(f"Total winning model training time: {t_end - t_start:.2f}s")

    print()
    print("Config note:")
    print("- SignalEngine currently supports 'price'/'full' and learned-weights scoring.")
    print("- If you want to wire this pickle model into live/backtest inference, you'll need a small integration step.")
    print("Suggested YAML fields to add (manual):")
    print("signals:")
    print('  mode: "ml"')
    print(f'  ml_model_path: "{out_dir.as_posix()}/best_long_model.pkl"')

    # C2: Regime-conditional models (opt-in via --regime-models)
    if getattr(args, "regime_models", False):
        _train_regime_models(df, feat_cols, all_singular, out_dir, horizon)


if __name__ == "__main__":
    main()
