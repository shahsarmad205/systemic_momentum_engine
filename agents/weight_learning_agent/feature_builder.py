"""
Feature Matrix Builder
========================
Constructs the historical feature matrix used to learn optimal signal
weights.  All features use only past data (no look-ahead bias).

    Compound: f_trend, f_regional, f_global, f_social
    Momentum: ret_5d, ret_10d
    Volatility: rolling_vol_10, rolling_vol_20
    Volume: relative_volume, volume_zscore
    Cross-ticker: rolling_corr_market_20 (correlation with SPY over 20d)
    Crisis / macro:
        vix_zscore: z-scored VIX level (shifted by 1)
        vol_spike: SPY vol(5d)/vol(60d) ratio (shifted by 1)
        vix_term: VIX spot / VIX3M term structure (shifted by 1)
    Target: forward_return, direction
"""

from __future__ import annotations

from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from agents.trend_agent.feature_engineering import build_features
from agents.volatility_agent.volatility_model import (
    compute_rolling_confidence,
    compute_vol_term_structure,
)
from agents.weight_learning_agent.regime_detection import get_regime_series_for_dates
from execution.cost_model import TransactionCostModel
from main import CONFIDENCE_MULTIPLIER, compute_rolling_trend_scores
from utils.sectors import get_sector
from utils.market_data import get_ohlcv

HISTORY_BUFFER_DAYS = 400
MARKET_TICKER = "SPY"  # for rolling correlation feature


def _download(ticker: str, start, end) -> pd.DataFrame:
    """
    Download OHLCV for a single ticker using the shared market_data layer with caching.
    """
    df = get_ohlcv(
        ticker,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        provider="yahoo",
        use_cache=True,
        cache_ttl_days=1,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    keep = ["Open", "High", "Low", "Close", "Volume"]
    # get_ohlcv already enforces OHLCV_COLUMNS, but keep explicit selection for safety
    return df[keep].dropna()


def _build_features_for_ticker(
    ticker: str,
    dl_start: pd.Timestamp,
    dl_end: pd.Timestamp,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    holding_period: int,
    market_ret: pd.Series | None,
) -> pd.DataFrame:
    """
    Worker helper: build feature chunk for a single ticker.
    Returns an empty DataFrame on failure so that the caller can skip it.
    """
    try:
        data = _download(ticker, dl_start, dl_end)
        if data.empty or len(data) < 210:
            return pd.DataFrame()

        features = build_features(data)
        if features.empty:
            return pd.DataFrame()

        trend_scores = compute_rolling_trend_scores(features)
        daily_ret = features["daily_return"]
        rolling_conf = compute_rolling_confidence(daily_ret, window=20)
        conf_mult = rolling_conf.map(CONFIDENCE_MULTIPLIER).fillna(0.5)

        f_trend = trend_scores * conf_mult

        close = data["Close"].reindex(features.index)
        volume = data["Volume"].reindex(features.index)
        open_px = data["Open"].reindex(features.index)

        # Momentum: 5-day, 10-day, 20-day, and 60-day returns (historical only)
        ret_5d = close.pct_change(5)
        ret_10d = close.pct_change(10)
        ret_20d = close.pct_change(20)
        ret_60d = close.pct_change(60)
        # Cross-sectional momentum signal: 6m return excluding most recent ~1 month
        if "momentum_6m" in features:
            cs_mom_raw = features["momentum_6m"].shift(21)
        else:
            cs_mom_raw = pd.Series(0.0, index=features.index)

        # Volatility: realised volatility term-structure (daily decimal), each z-scored on its own history
        vol_struct = compute_vol_term_structure(daily_ret)
        vol_5_raw = vol_struct["vol_5d"]
        vol_10_raw = vol_struct["vol_10d"]
        vol_20_raw = vol_struct["vol_20d"]
        vol_60_raw = vol_struct["vol_60d"]
        v5_m = vol_5_raw.rolling(252, min_periods=60).mean()
        v5_s = vol_5_raw.rolling(252, min_periods=60).std().replace(0, np.nan).fillna(1.0)
        rolling_vol_5 = (vol_5_raw - v5_m) / v5_s
        v10_m = vol_10_raw.rolling(252, min_periods=60).mean()
        v10_s = vol_10_raw.rolling(252, min_periods=60).std().replace(0, np.nan).fillna(1.0)
        rolling_vol_10 = (vol_10_raw - v10_m) / v10_s
        v20_m = vol_20_raw.rolling(252, min_periods=60).mean()
        v20_s = vol_20_raw.rolling(252, min_periods=60).std().replace(0, np.nan).fillna(1.0)
        rolling_vol_20 = (vol_20_raw - v20_m) / v20_s
        v60_m = vol_60_raw.rolling(252, min_periods=60).mean()
        v60_s = vol_60_raw.rolling(252, min_periods=60).std().replace(0, np.nan).fillna(1.0)
        rolling_vol_60 = (vol_60_raw - v60_m) / v60_s
        vol_of_vol_20_raw = vol_struct["vol_of_vol_20"]
        vov_mean = vol_of_vol_20_raw.rolling(252, min_periods=60).mean()
        vov_std = vol_of_vol_20_raw.rolling(252, min_periods=60).std().replace(0, np.nan).fillna(1.0)
        vol_of_vol_20 = (vol_of_vol_20_raw - vov_mean) / vov_std
        jump_indicator = vol_struct["jump_indicator"]

        # Per-ticker volatility regime: 20d realised vol / 252d realised vol.
        # Values > 1.2 indicate a high-volatility stock-specific regime.
        vol20 = daily_ret.rolling(20).std()
        vol252 = daily_ret.rolling(252).std()
        vol_regime_ratio = (vol20 / vol252).replace([np.inf, -np.inf], np.nan)
        is_high_vol_stock_regime = (vol_regime_ratio > 1.2).astype(float)

        # Relative volume: volume / 20-day mean volume, then z-score for scale
        vol_ma20 = volume.rolling(20).mean()
        relative_volume_raw = (volume / vol_ma20).replace([np.inf, -np.inf], np.nan)
        rv_mean = relative_volume_raw.rolling(252, min_periods=60).mean()
        rv_std = relative_volume_raw.rolling(252, min_periods=60).std().replace(0, np.nan).fillna(1.0)
        relative_volume = (relative_volume_raw - rv_mean) / rv_std
        # Volume z-score: (volume - mean) / std over 20d (no look-ahead).
        # This also serves as a "volume surprise" indicator.
        vol_std20 = volume.rolling(20).std()
        volume_zscore = (volume - vol_ma20) / vol_std20.replace(0, np.nan)
        volume_zscore = volume_zscore.replace([np.inf, -np.inf], np.nan)

        # Cross-ticker: rolling 20d correlation with market (SPY)
        rolling_corr_market_20 = pd.Series(np.nan, index=features.index)
        capm_alpha = pd.Series(np.nan, index=features.index)
        capm_beta = pd.Series(1.0, index=features.index)
        capm_residual_vol = pd.Series(np.nan, index=features.index)
        if market_ret is not None:
            market_aligned = market_ret.reindex(features.index).ffill().bfill().fillna(0.0)
            rolling_corr_market_20 = daily_ret.rolling(20).corr(market_aligned)
            # CAPM: rolling 60d beta, alpha, residual vol; alpha z-scored over 252d
            try:
                from features.capm_features import compute_capm_features
                stock_ret = daily_ret.astype(float).fillna(0.0)
                capm_df = compute_capm_features(stock_ret, market_aligned, window=60, zscore_window=252)
                capm_alpha = capm_df["capm_alpha"]
                capm_beta = capm_df["capm_beta"]
                capm_residual_vol = capm_df["capm_residual_vol"]
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Short-horizon / mean-reversion features (1–5 day horizon)
        # ------------------------------------------------------------------
        # 1) RSI(14) on close
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_14 = 100 - (100 / (1 + rs))

        # 2) Distance from 20-day high/low
        rolling_max_20 = close.rolling(20, min_periods=10).max()
        rolling_min_20 = close.rolling(20, min_periods=10).min()
        dist_20d_high = (close - rolling_max_20) / rolling_max_20.replace(0, np.nan)
        dist_20d_low = (close - rolling_min_20) / rolling_min_20.replace(0, np.nan)

        # 3) Overnight gap: (open - prev_close) / prev_close
        prev_close = close.shift(1)
        overnight_gap = (open_px - prev_close) / prev_close.replace(0, np.nan)

        # 4) Intraday reversal: (close - open) / open
        intraday_reversal = (close - open_px) / open_px.replace(0, np.nan)

        # 5) Bollinger Band position (20-day)
        bb_mid = close.rolling(20, min_periods=20).mean()
        bb_std = close.rolling(20, min_periods=20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = (bb_upper - bb_lower).replace(0, np.nan)
        bb_pos_20 = (close - bb_lower) / bb_width

        forward_ret = close.shift(-holding_period) / close - 1

        chunk = pd.DataFrame(
            {
                "ticker": ticker,
                "f_trend": f_trend,
                "f_regional": 0.0,
                "f_global": 0.0,
                "f_social": 0.0,
                "ret_5d": ret_5d,
                "ret_10d": ret_10d,
                "ret_20d": ret_20d,
                "ret_60d": ret_60d,
                "rolling_vol_5": rolling_vol_5,
                "rolling_vol_10": rolling_vol_10,
                "rolling_vol_20": rolling_vol_20,
                "rolling_vol_60": rolling_vol_60,
                "vol_of_vol_20": vol_of_vol_20,
                "jump_indicator": jump_indicator,
                "realised_vol_20d": vol_20_raw,
                "vol_regime_ratio": vol_regime_ratio,
                "is_high_vol_stock_regime": is_high_vol_stock_regime,
                "relative_volume": relative_volume,
                "volume_zscore": volume_zscore,
                "volume_surprise": volume_zscore,
                "rolling_corr_market_20": rolling_corr_market_20,
                "capm_alpha": capm_alpha,
                "capm_beta": capm_beta,
                "capm_residual_vol": capm_residual_vol,
                # Mean-reversion / short-horizon signals
                "rsi_14": rsi_14,
                "dist_20d_high": dist_20d_high,
                "dist_20d_low": dist_20d_low,
                "overnight_gap": overnight_gap,
                "intraday_reversal": intraday_reversal,
                "bb_pos_20": bb_pos_20,
                "trend_score": trend_scores,
                "confidence_mult": conf_mult,
                "momentum_3m": features["momentum_3m"] if "momentum_3m" in features else 0.0,
                "momentum_6m": features["momentum_6m"] if "momentum_6m" in features else 0.0,
                "ma_crossover": features["ma_crossover_signal"] if "ma_crossover_signal" in features else 0.0,
                "cs_momentum_raw": cs_mom_raw,
                "daily_return": daily_ret,
                "forward_return": forward_ret,
            },
            index=features.index,
        )

        chunk.index.name = "date"
        chunk = chunk.reset_index()

        chunk["volume_zscore"] = chunk["volume_zscore"].fillna(0)
        chunk["rolling_corr_market_20"] = chunk["rolling_corr_market_20"].fillna(0)

        mask = (chunk["date"] >= start_ts) & (chunk["date"] <= end_ts)
        chunk = chunk[mask].dropna(
            subset=[
                "forward_return",
                "ret_5d",
                "ret_10d",
                "rolling_vol_10",
                "rolling_vol_20",
                "relative_volume",
            ]
        )

        if chunk.empty:
            return pd.DataFrame()

        chunk["direction"] = np.sign(chunk["forward_return"]).astype(int)
        chunk["sector"] = get_sector(ticker)
        return chunk
    except Exception:
        # Let caller handle logging; here we just return empty on failure
        return pd.DataFrame()


def build_feature_matrix(
    tickers: list[str],
    start_date: str,
    end_date: str,
    holding_period: int = 5,
) -> pd.DataFrame:
    """
    Build a DataFrame of (ticker, date, features, target) rows suitable
    for training weight-learning models.

    Parameters:
        tickers         : symbols to include
        start_date      : first observation date  (str YYYY-MM-DD)
        end_date        : last  observation date
        holding_period  : trading days for forward return

    Returns:
        DataFrame with columns listed in the module docstring.
    """
    dl_start = pd.Timestamp(start_date) - timedelta(days=HISTORY_BUFFER_DAYS)
    dl_end = pd.Timestamp(end_date) + timedelta(days=holding_period * 2)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Market series for cross-ticker feature (no look-ahead: historical SPY returns only)
    market_ret = None
    spy = None
    vix_zscore = pd.Series(dtype=float)
    vol_spike = pd.Series(dtype=float)
    vix_term = pd.Series(dtype=float)
    try:
        spy = _download(MARKET_TICKER, dl_start, dl_end)
        if not spy.empty and len(spy) >= 25:
            market_ret = spy["Close"].pct_change()
    except Exception:
        spy = None
        market_ret = None

    # Macro inputs (VIX + SPY realised vol ratios) for learned-weight GBR.
    # These are shifted by 1 to avoid look-ahead (values for date t
    # only use information available after day t-1).
    try:
        if spy is not None and not spy.empty and "Close" in spy.columns:
            spy_close = pd.to_numeric(spy["Close"], errors="coerce").dropna().sort_index()
            spy_ret = spy_close.pct_change()
            vol5 = spy_ret.rolling(5).std() * np.sqrt(252.0)
            vol60 = spy_ret.rolling(60).std() * np.sqrt(252.0)
            vol_spike = (vol5 / vol60).replace([np.inf, -np.inf], np.nan).shift(1)

        vix_raw = get_ohlcv(
            "^VIX",
            dl_start.strftime("%Y-%m-%d"),
            dl_end.strftime("%Y-%m-%d"),
            provider="yahoo",
            use_cache=True,
            cache_ttl_days=1,
        )
        if vix_raw is not None and not vix_raw.empty and "Close" in vix_raw.columns:
            vix_close = pd.to_numeric(vix_raw["Close"], errors="coerce").dropna().sort_index()
            vix_mean_252 = vix_close.rolling(252).mean()
            vix_std_252 = vix_close.rolling(252).std(ddof=0).replace(0, np.nan)
            vix_zscore = ((vix_close - vix_mean_252) / vix_std_252).shift(1)

            vix3m_raw = get_ohlcv(
                "^VIX3M",
                dl_start.strftime("%Y-%m-%d"),
                dl_end.strftime("%Y-%m-%d"),
                provider="yahoo",
                use_cache=True,
                cache_ttl_days=1,
            )
            if vix3m_raw is not None and not vix3m_raw.empty and "Close" in vix3m_raw.columns:
                vix3m_close = pd.to_numeric(vix3m_raw["Close"], errors="coerce").dropna().sort_index()
                vix3m_aligned = vix3m_close.reindex(vix_close.index).ffill()
                vix_term = (vix_close / vix3m_aligned.replace(0.0, np.nan)).replace(
                    [np.inf, -np.inf], np.nan
                ).shift(1)
    except Exception:
        # If downloads/derivations fail, keep defaults (features will be 0.0 later).
        pass

    chunks: list[pd.DataFrame] = []

    # Parallelise per-ticker feature construction.
    with ProcessPoolExecutor() as executor:
        future_to_ticker = {
            executor.submit(
                _build_features_for_ticker,
                ticker,
                dl_start,
                dl_end,
                start_ts,
                end_ts,
                holding_period,
                market_ret,
            ): (i, ticker)
            for i, ticker in enumerate(tickers, 1)
        }

        for fut in as_completed(future_to_ticker):
            i, ticker = future_to_ticker[fut]
            print(f"  [{i}/{len(tickers)}] {ticker}…", end=" ")
            try:
                chunk = fut.result()
                if chunk is None or chunk.empty:
                    print("no valid rows")
                    continue
                chunks.append(chunk)
                print(f"{len(chunk)} rows")
            except Exception as exc:
                print(f"ERROR: {exc}")

    if not chunks:
        return pd.DataFrame()

    result = pd.concat(chunks, ignore_index=True)
    result.sort_values(["date", "ticker"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    # Attach VIX / macro features to every (ticker, date) row.
    # We fill NaNs with 0.0 so WeightLearner won't drop rows purely due to
    # early rolling-window warmup.
    if "date" in result.columns:
        if not vix_zscore.empty:
            # `result["date"]` contains duplicates across tickers; use map()
            # instead of reindex() to avoid "duplicate labels" errors.
            result["vix_zscore"] = result["date"].map(vix_zscore.to_dict()).astype(float).fillna(0.0)
        else:
            result["vix_zscore"] = 0.0
        if not vol_spike.empty:
            result["vol_spike"] = result["date"].map(vol_spike.to_dict()).astype(float).fillna(0.0)
        else:
            result["vol_spike"] = 0.0
        if not vix_term.empty:
            result["vix_term"] = result["date"].map(vix_term.to_dict()).astype(float).fillna(0.0)
        else:
            result["vix_term"] = 0.0

    # Sector-relative momentum: stock 20d/60d return minus sector-median 20d/60d return (no look-ahead).
    if {"ret_20d", "sector"}.issubset(result.columns):
        sector_median_20 = (
            result.groupby(["date", "sector"])["ret_20d"]
            .transform("median")
        )
        result["sector_relative_mom_20d"] = result["ret_20d"] - sector_median_20
        # Backwards-compatible name used earlier in the project.
        result["sector_relative_strength"] = result["sector_relative_mom_20d"]

    if {"ret_60d", "sector"}.issubset(result.columns):
        sector_median_60 = (
            result.groupby(["date", "sector"])["ret_60d"]
            .transform("median")
        )
        result["sector_relative_mom_60d"] = result["ret_60d"] - sector_median_60

    # Cross-sectional momentum percentile (0–1) based on 6m return excluding last month.
    if "cs_momentum_raw" in result.columns:
        result["cs_momentum_percentile"] = (
            result.groupby("date")["cs_momentum_raw"]
            .rank(pct=True, method="average")
        )

    # SPY-based volatility regime features: map each date to a SPY-based regime and flag high-volatility days.
    try:
        unique_dates = pd.to_datetime(result["date"].unique())
        regime_series = get_regime_series_for_dates(unique_dates, start_date, end_date)
        regime_map = regime_series.to_dict()
        result["regime_label"] = result["date"].map(regime_map).fillna("Normal")
        result["is_high_vol_regime"] = (
            result["regime_label"].isin(["HighVol"])
        ).astype(float)
    except Exception:
        # If regime detection fails (e.g. data/download issues), skip these features.
        result["regime_label"] = "Normal"
        result["is_high_vol_regime"] = 0.0

    # Expected round-trip execution cost as a constant fraction of notional, based on TransactionCostModel.
    try:
        cost_model = TransactionCostModel()
        leg_cost_frac = cost_model.cost_fraction()
        round_trip_frac = 2.0 * leg_cost_frac
        result["expected_round_trip_cost_frac"] = float(round_trip_frac)
    except Exception:
        # If the cost model is unavailable, omit the column (WeightLearner will fall back to time-decay only).
        pass

    # Benchmark 5-day forward return for SPY so we can build excess-return targets.
    if spy is not None and not spy.empty and "forward_return" in result.columns:
        try:
            spy_fwd = spy["Close"].shift(-holding_period) / spy["Close"] - 1
            spy_fwd = spy_fwd.rename("spy_forward_5d")
            # Ensure the date column is named 'date' regardless of the original index name.
            idx_name = spy_fwd.index.name or "index"
            spy_fwd_df = spy_fwd.reset_index().rename(columns={idx_name: "date"})
            result = result.merge(spy_fwd_df, on="date", how="left")
            # Target variants:
            result["forward_return_excess"] = result["forward_return"] - result["spy_forward_5d"]
        except Exception:
            # If anything goes wrong, fall back to raw forward_return only for excess.
            result["spy_forward_5d"] = np.nan
            result["forward_return_excess"] = result["forward_return"]

    # Cross-sectional z-scores: for each numeric feature, normalise across tickers per date.
    num_cols = result.select_dtypes(include=["number"]).columns
    for col in num_cols:
        cs = result.groupby("date")[col].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) not in (0, 0.0) else 1.0)
        )
        result[f"{col}_cs_z"] = cs.fillna(0.0)

    return result
