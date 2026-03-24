#!/usr/bin/env python3
"""
Alpha Research Engine — feature search via Information Coefficient (IC).

Tests candidate predictive features on 2013–2024 panel data before any
model integration. Does not modify feature_builder or other modules.

Run from trend_signal_engine root:
    python analysis/feature_search.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Project root (parent of analysis/)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.market_data import get_ohlcv  # noqa: E402

START_DATE = "2013-01-01"
END_DATE = "2024-01-01"
IC_LAGS = (1, 3, 5, 10)


def _load_config() -> dict:
    cfg_path = ROOT / "backtest_config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = get_ohlcv(
        ticker,
        start,
        end,
        provider="yahoo",
        use_cache=True,
        cache_ttl_days=0,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    for col in ("Close", "Volume"):
        if col not in df.columns:
            return pd.DataFrame()
    return df.sort_index()


def _build_long_panel(tickers: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for tk in tickers:
        px = _download_ohlcv(tk, START_DATE, END_DATE)
        if px.empty or len(px) < 300:
            continue
        close = pd.to_numeric(px["Close"], errors="coerce")
        volume = pd.to_numeric(px["Volume"], errors="coerce").clip(lower=0)
        daily_return = close.pct_change()
        ret_5d = close.pct_change(5)

        rolling_vol_5 = daily_return.rolling(5).std()
        rolling_vol_20 = daily_return.rolling(20).std()

        roll_max_252 = close.rolling(252, min_periods=126).max()
        dist_52w_high = (close - roll_max_252) / roll_max_252.replace(0, np.nan)

        ret_5d_change = ret_5d - ret_5d.shift(5)
        vol_ratio_5_20 = rolling_vol_5 / rolling_vol_20.replace(0, np.nan)

        vol_ma5 = volume.rolling(5).mean()
        vol_ma20 = volume.rolling(20).mean()
        vol_trend = vol_ma5 / vol_ma20.replace(0, np.nan)

        rel_vol_raw = volume / vol_ma20.replace(0, np.nan)
        pv_divergence = ret_5d * (1.0 - rel_vol_raw)

        ma20 = close.rolling(20).mean()
        sd20 = close.rolling(20).std()
        price_zscore = (close - ma20) / sd20.replace(0, np.nan)

        streak = np.sign(daily_return).rolling(5).sum()

        chunk = pd.DataFrame(
            {
                "date": pd.to_datetime(close.index),
                "ticker": tk,
                "close": close.values,
                "ret_5d": ret_5d.values,
                "rolling_vol_20": rolling_vol_20.values,
                "dist_52w_high": dist_52w_high.values,
                "ret_5d_change": ret_5d_change.values,
                "vol_ratio_5_20": vol_ratio_5_20.values,
                "vol_trend": vol_trend.values,
                "pv_divergence": pv_divergence.values,
                "price_zscore": price_zscore.values,
                "streak": streak.values,
            }
        )
        rows.append(chunk)

    if not rows:
        return pd.DataFrame()

    long_df = pd.concat(rows, ignore_index=True)
    long_df = long_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Cross-sectional ranks (same date, universe)
    long_df["return_rank_5d"] = long_df.groupby("date")["ret_5d"].rank(pct=True)
    long_df["vol_rank"] = long_df.groupby("date")["rolling_vol_20"].rank(pct=True)

    return long_df


def _forward_returns(long_df: pd.DataFrame) -> pd.DataFrame:
    out = long_df.copy()
    for L in IC_LAGS:
        out[f"fwd_{L}"] = out.groupby("ticker", group_keys=False)["close"].transform(
            lambda s, lag=L: s.shift(-lag) / s - 1.0
        )
    return out


def _ic_one_day(x: pd.Series, y: pd.Series) -> float:
    m = x.notna() & y.notna()
    if m.sum() < 4:
        return np.nan
    xc = x[m]
    yc = y[m]
    if xc.std(ddof=0) < 1e-12 or yc.std(ddof=0) < 1e-12:
        return np.nan
    return float(xc.corr(yc, method="pearson"))


def _daily_ic(panel: pd.DataFrame, feat_col: str, y_col: str) -> pd.Series:
    """Cross-sectional Pearson IC by date."""
    ic_by_date: dict[pd.Timestamp, float] = {}
    for dt, sub in panel.groupby("date", sort=True):
        x = pd.to_numeric(sub[feat_col], errors="coerce")
        y = pd.to_numeric(sub[y_col], errors="coerce")
        ic_by_date[dt] = _ic_one_day(x, y)
    return pd.Series(ic_by_date).sort_index()


def _ic_t_stat(ic_series: pd.Series) -> float:
    ic = pd.to_numeric(ic_series, errors="coerce").dropna()
    if len(ic) < 10:
        return float("nan")
    mu = float(ic.mean())
    sd = float(ic.std(ddof=1))
    if sd < 1e-12:
        return float("nan")
    return mu / sd * np.sqrt(len(ic))


def _ic_stable(ic_series: pd.Series, n_buckets: int = 4) -> bool:
    """IC stable across time: majority of sub-period mean ICs share overall sign."""
    ic = pd.to_numeric(ic_series, errors="coerce").dropna()
    ic = ic.sort_index()
    if len(ic) < n_buckets * 8:
        return False
    overall = np.sign(ic.mean())
    if overall == 0:
        return False
    parts = np.array_split(ic.values, n_buckets)
    means = [np.nanmean(p) for p in parts if len(p) > 0]
    if len(means) < n_buckets // 2:
        return False
    signs = [np.sign(m) for m in means if not np.isnan(m)]
    same = sum(1 for s in signs if s == overall)
    return same >= max(3, int(0.75 * len(signs)))


def _shift_feature(panel: pd.DataFrame, col: str) -> pd.Series:
    return panel.groupby("ticker")[col].shift(1)


def _evaluate_feature(
    base: pd.DataFrame,
    feature_name: str,
    feat_shifted: pd.Series,
) -> dict:
    work = base.copy()
    work["_feat"] = feat_shifted

    row: dict = {"feature_name": feature_name}
    ic_lag5 = pd.Series(dtype=float)
    for L in IC_LAGS:
        ycol = f"fwd_{L}"
        ic_s = _daily_ic(work, "_feat", ycol)
        row[f"IC_lag{L}"] = float(np.nanmean(ic_s.values))
        if L == 5:
            ic_lag5 = ic_s

    t_stat = _ic_t_stat(ic_lag5)
    stable = _ic_stable(ic_lag5)
    row["t_stat"] = t_stat
    row["stable"] = stable

    ic5 = row["IC_lag5"]
    abs_ic5 = abs(ic5) if np.isfinite(ic5) else 0.0
    ts = t_stat if np.isfinite(t_stat) else 0.0

    # Spec: ADD / MAYBE from IC magnitude + t-stat; stable is informational.
    if abs_ic5 > 0.02 and ts > 2.0:
        verdict = "ADD"
    elif abs_ic5 > 0.01 and ts > 1.5:
        verdict = "MAYBE"
    else:
        verdict = "SKIP"

    row["verdict"] = verdict
    return row


def main() -> None:
    os.makedirs(ROOT / "output" / "research", exist_ok=True)

    cfg = _load_config()
    tickers = list(cfg.get("tickers") or [])
    if not tickers:
        raise SystemExit("No tickers in backtest_config.yaml")

    print(f"Building panel: {len(tickers)} tickers, {START_DATE} → {END_DATE} …")
    long_df = _build_long_panel(tickers)
    if long_df.empty:
        raise SystemExit("No price data — check cache / tickers.")

    base = _forward_returns(long_df)

    feature_specs = [
        "dist_52w_high",
        "ret_5d_change",
        "vol_ratio_5_20",
        "vol_trend",
        "pv_divergence",
        "price_zscore",
        "streak",
        "return_rank_5d",
        "vol_rank",
    ]

    results: list[dict] = []
    for fname in feature_specs:
        if fname not in base.columns:
            continue
        f_shifted = _shift_feature(base, fname)
        results.append(_evaluate_feature(base, fname, f_shifted))

    out_df = pd.DataFrame(results)
    # Column order for CSV
    cols = (
        ["feature_name"]
        + [f"IC_lag{L}" for L in IC_LAGS]
        + ["t_stat", "stable", "verdict"]
    )
    for c in cols:
        if c not in out_df.columns:
            out_df[c] = np.nan
    out_df = out_df[cols]
    out_path = ROOT / "output" / "research" / "feature_search_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    pass_add = out_df[out_df["verdict"] == "ADD"]
    not_ready = out_df[out_df["verdict"] != "ADD"]

    print()
    print("=== FEATURES THAT PASS IC THRESHOLD ===")
    if not pass_add.empty:
        print(pass_add[["feature_name", "IC_lag5", "t_stat", "stable"]].to_string(index=False))
    else:
        print("(none met ADD criteria this run)")
    print()
    print("These should be added to feature_builder.py")
    print()
    print("=== FEATURES THAT FAILED ===")
    if not not_ready.empty:
        print(not_ready[["feature_name", "verdict", "IC_lag5", "t_stat"]].to_string(index=False))
    else:
        print("(none)")
    print()
    print("Don't add these — they don't predict returns")


if __name__ == "__main__":
    main()
