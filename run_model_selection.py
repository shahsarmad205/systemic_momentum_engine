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
        if test_end > end_ts:
            break
        windows.append((train_start, train_end, test_start, test_end))
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


def _sharpe_from_series(pnl: np.ndarray) -> float:
    pnl = pnl.astype(float)
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) < 10:
        return float("nan")
    mu = float(np.mean(pnl))
    sd = float(np.std(pnl, ddof=1))
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
) -> pd.Series:
    """
    Simulate a simple daily-rebalanced long-only portfolio over a test slice.

    For each date in te:
      - rank tickers by predicted score desc
      - take up to max_positions with score > 0
      - if fewer than min_positions qualify, hold cash (0 return)
      - compute equal-weight return using realized forward_return as a proxy

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


def _chained_oos_metrics(oos: pd.DataFrame, *, max_positions: int = 10) -> tuple[float, float, float]:
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

    dr_s = _strategy_daily_returns(df, max_positions=int(max_positions), min_positions=1)
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
    cfg_tickers = cfg.get("tickers", []) if isinstance(cfg, dict) else []
    chosen = list(tickers or cfg_tickers[:3])
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
    from sklearn.preprocessing import StandardScaler

    models: list[tuple[str, Any, bool, str]] = []

    # --- classifiers ---
    models.append(
        (
            "LogisticRegression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(C=0.01, max_iter=1000)),
                ]
            ),
            True,
            "classifier",
        )
    )
    models.append(
        (
            "RidgeClassifier",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", RidgeClassifier(alpha=1.0)),
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
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
            False,
            "regressor",
        )
    )

    try:
        from xgboost import XGBRegressor

        models.append(
            (
                "XGBRegressor",
                XGBRegressor(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.05,
                    random_state=42,
                    n_jobs=-1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                ),
                False,
                "regressor",
            )
        )
    except Exception:
        pass

    # --- short classifiers (predict P(down); scored as negative) ---
    models.append(
        (
            "ShortLogistic",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(C=0.01, max_iter=1000)),
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
    tickers = list(cfg.get("tickers", []) or [])
    if int(args.limit_tickers or 0) > 0:
        tickers = tickers[: int(args.limit_tickers)]
    bt = cfg.get("backtest", {}) or {}
    research = cfg.get("research", {}) or {}
    feature_sel = cfg.get("feature_selection", {}) or {}
    ms_cfg = cfg.get("model_selection", {}) or {}
    feature_subset = feature_sel.get("feature_subset", []) or []
    feature_subset = [str(c).strip() for c in feature_subset if str(c).strip()]

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
    if feature_subset:
        print(f"Feature subset: {len(feature_subset)} columns")

    from agents.weight_learning_agent.feature_builder import build_feature_matrix

    # Build enough history for rolling features (feature_builder applies its own buffers).
    df = build_feature_matrix(
        tickers,
        start_date=start_date,
        end_date=end_date,
        holding_period=int(horizon),
        feature_subset=feature_subset if feature_subset else None,
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
                X_tr = tr[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
                if is_regressor:
                    y_tr = tr["forward_return"].values.astype(float)
                elif is_short_classifier:
                    y_tr = (tr["forward_return"].values < 0).astype(int)
                else:
                    y_tr = tr["y_bin"].values.astype(int)
                X_te = te[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
                y_te_ret = te["forward_return"].values.astype(float)
                y_te_bin = te["y_bin"].values.astype(int)

                try:
                    t0 = time.perf_counter()
                    model.fit(X_tr, y_tr)
                    t1 = time.perf_counter()
                except Exception as exc:
                    print(f"  [window {win_idx}/{len(windows)}] train failed: {exc}")
                    traceback.print_exc()
                    continue

                try:
                    t2 = time.perf_counter()
                    if is_regressor:
                        score = model.predict(X_te).astype(float)
                    elif is_short_classifier:
                        if uses_proba and hasattr(model, "predict_proba"):
                            p_down = model.predict_proba(X_te)[:, 1].astype(float)
                            score = -(p_down - 0.5)
                        else:
                            score = -model.predict(X_te).astype(float) + 0.5
                    elif uses_proba and hasattr(model, "predict_proba"):
                        p = model.predict_proba(X_te)[:, 1].astype(float)
                        score = p - 0.5
                    elif hasattr(model, "decision_function"):
                        score = model.decision_function(X_te).astype(float)
                    else:
                        pred = model.predict(X_te).astype(int)
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
                sharpe = _sharpe_from_series(daily_ret_s.to_numpy(dtype=float))
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
        oos_sharpe_chained = _sharpe_from_series(chained_daily)
        oos_cagr_chained = _cagr_from_daily_returns(chained_daily)
        oos_max_dd = _max_drawdown_from_daily_returns(chained_daily)
        oos_win_rate = _win_rate_from_daily_returns(chained_daily)
        _oos_sharpe_chained_old, _oos_cagr_chained_old, oos_ic_chained = _chained_oos_metrics(
            oos_df, max_positions=int(max_positions)
        )

        row = {
            "model_name": name,
            "model_kind": model_kind,
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
            "train_time_avg": float(np.nanmean(tr_t)),
            "test_time_avg": float(np.nanmean(te_t)),
            "n_windows": int(len(wm)),
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

        print(
            f"OOS Sharpe (chained): {row['oos_sharpe_chained']:.3f} | "
            f"window Sharpe mean±std: {row['oos_sharpe_mean']:.3f} ± {row['oos_sharpe_std']:.3f} | "
            f"IC: {row['oos_ic_mean']:.3f} ± {row['oos_ic_std']:.3f} | "
            f"DirAcc: {row['oos_dir_acc_mean']:.3f} ± {row['oos_dir_acc_std']:.3f} | "
            f"windows={row['n_windows']}"
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
                sharpe = _sharpe_from_series(daily_ret_s.to_numpy(dtype=float))
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
            oos_sharpe_chained = _sharpe_from_series(chained_daily)
            oos_cagr_chained = _cagr_from_daily_returns(chained_daily)
            oos_max_dd = _max_drawdown_from_daily_returns(chained_daily)
            oos_win_rate = _win_rate_from_daily_returns(chained_daily)
            _unused_s, _unused_c, oos_ic_chained = _chained_oos_metrics(oos_df, max_positions=int(max_positions))

            row = {
                "model_name": "LearnedWeightsBaseline",
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

    best_name = str(report.loc[0, "model_name"])
    print()
    print(f"Saved report: {report_path}")
    print(f"Selected best model by {args.select_metric}: {best_name}")

    # Recreate and train best model on full dataset
    best_spec = {n: (m, p, k) for (n, m, p, k) in models}.get(best_name)
    if best_spec is None:
        # Allow selecting the baseline as "best" when --compare_baseline is enabled.
        if best_name == "LearnedWeightsBaseline" and args.compare_baseline:
            artifact = {
                "model_name": best_name,
                "model_type": "learned_weights_baseline",
                "horizon_days": int(horizon),
                "target": "forward_return",
                "trained_at": pd.Timestamp.utcnow().isoformat(),
                "weights_path": "output/learned_weights.json",
                "scaler_path": "output/learned_weights_scaler.json",
                "estimator": None,
            }
            best_path = out_dir / "best_model.pkl"
            with open(best_path, "wb") as fh:
                pickle.dump(artifact, fh)
            print(f"Saved best model (baseline artifact): {best_path}")

            meta_path = out_dir / "best_model.meta.json"
            meta = {
                "model_name": best_name,
                "model_type": "learned_weights_baseline",
                "horizon_days": int(horizon),
                "target": "forward_return",
                "n_rows": int(len(df)),
                "n_tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else None,
                "feature_columns": _read_json(Path("output/learned_weights_scaler.json")).get("active_features", []),
                "selected_by": args.select_metric,
                "windows": [],
            }
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            print(f"Saved metadata: {meta_path}")

            print()
            print("Config note:")
            print("- Selected baseline artifact (no sklearn estimator).")
            print("- If you want to wire this baseline into inference, reuse your existing learned-weights scorer.")
            return

        raise SystemExit(f"Best model {best_name} not found in model list (was it skipped due to missing deps?).")
    best_model, _best_uses_proba, best_kind = best_spec

    t0 = time.perf_counter()
    X_all = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y_all_cls = df["y_bin"].values.astype(int)
    y_all_reg = df["forward_return"].values.astype(float)
    y_all_down = (df["forward_return"].values < 0).astype(int)
    if args.save_all_models:
        for model_name, model_obj, _uses_proba, mkind in models:
            try:
                if mkind == "regressor":
                    y_fit = y_all_reg
                elif mkind == "short_classifier":
                    y_fit = y_all_down
                else:
                    y_fit = y_all_cls
                model_obj.fit(X_all, y_fit)
                model_artifact = {
                    "model_name": model_name,
                    "model_type": mkind,
                    "horizon_days": int(horizon),
                    "target": "forward_return",
                    "feature_columns": feat_cols,
                    "trained_at": pd.Timestamp.utcnow().isoformat(),
                    "estimator": model_obj,
                }
                model_path = out_dir / _model_filename(model_name)
                with open(model_path, "wb") as fh:
                    pickle.dump(model_artifact, fh)
                print(f"Saved model copy: {model_path}")
            except Exception as exc:
                print(f"WARNING: failed to save model copy for {model_name}: {exc}")
    if best_kind == "regressor":
        y_best = y_all_reg
    elif best_kind == "short_classifier":
        y_best = y_all_down
    else:
        y_best = y_all_cls
    best_model.fit(X_all, y_best)
    t1 = time.perf_counter()
    print(f"Trained best model on full dataset in {t1 - t0:.2f}s")

    artifact = {
        "model_name": best_name,
        "model_type": best_kind,
        "horizon_days": int(horizon),
        "target": "forward_return",
        "feature_columns": feat_cols,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "estimator": best_model,
    }
    best_path = out_dir / "best_model.pkl"
    with open(best_path, "wb") as fh:
        pickle.dump(artifact, fh)
    print(f"Saved best model: {best_path}")

    meta_path = out_dir / "best_model.meta.json"
    meta = {
        "model_name": best_name,
        "horizon_days": int(horizon),
        "target": "forward_return",
        "n_rows": int(len(df)),
        "n_tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else None,
        "feature_columns": feat_cols,
        "selected_by": args.select_metric,
        "windows": [
            {
                "train_start": w.train_start,
                "train_end": w.train_end,
                "test_start": w.test_start,
                "test_end": w.test_end,
            }
            for w in all_window_details.get(best_name, [])
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metadata: {meta_path}")

    print()
    print("Config note:")
    print("- SignalEngine currently supports 'price'/'full' and learned-weights scoring.")
    print("- If you want to wire this pickle model into live/backtest inference, you'll need a small integration step.")
    print("Suggested YAML fields to add (manual):")
    print("signals:")
    print('  mode: "ml"')
    print(f'  ml_model_path: "{best_path.as_posix()}"')


if __name__ == "__main__":
    main()

