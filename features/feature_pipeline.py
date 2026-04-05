# region imports
from __future__ import annotations
try:
    from AlgorithmImports import *
except ImportError:
    pass
# endregion
"""
Feature pipeline for single-name OHLCV time series.

This module combines the existing trend-agent feature builder with
additional momentum, volatility, liquidity, and regime features into
one enriched per-ticker feature matrix.
"""


import logging
import warnings

import numpy as np
import pandas as pd

# Financial intuition based features
from .capm_features import compute_capm_features
from .alternative_features import (
    compute_sue_score,
    compute_short_interest_features,
    compute_analyst_revision_momentum,
)

_HAS_TREND_AGENT = True

# Columns that duplicate existing SignalEngine features under different names.
# These are dropped from the final feature matrix to avoid feeding redundant
# signals to the weight learner.
KNOWN_REDUNDANCIES = {
    "volatility_20": "rolling_vol_20",   # same calculation, different name
    "volume_spike": "relative_volume",   # equivalent ratio
}


import pandas_ta as ta

def robust_zscore(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """Rolling z-score with zero-std protection and clipping to [-10, 10]."""
    roll = series.rolling(window=window, min_periods=min_periods)
    mu = roll.mean()
    sigma = roll.std().replace(0.0, np.nan).fillna(1.0)
    # Floor the standard deviation to prevent extremely large z-scores (near-overflow)
    sigma = sigma.clip(lower=1e-6)
    z = (series - mu) / sigma
    return z.clip(-10.0, 10.0).fillna(0.0)

def sanitize_dataframe(df: pd.DataFrame, clip_val: float = 10.0) -> pd.DataFrame:
    """Replaces infs, fills NaNs, and clips all numeric columns."""
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(-clip_val, clip_val)
    return df


def calculate_core_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    close = df["AdjClose"] if "AdjClose" in df.columns else (df["Close"] if "Close" in df.columns else df["close"])
    df["daily_return"] = close.pct_change()
    df["momentum_3m"] = close / close.shift(63) - 1
    df["momentum_6m"] = close / close.shift(126) - 1
    
    ma50 = ta.sma(close, length=50)
    ma200 = ta.sma(close, length=200)
    df["ma_50_ratio"] = ((ma50 / close.replace(0.0, np.nan)) - 1.0).fillna(0.0)
    df["ma_200_ratio"] = ((ma200 / close.replace(0.0, np.nan)) - 1.0).fillna(0.0)
    
    df["ma_crossover_signal"] = np.where(df["ma_50_ratio"] > df["ma_200_ratio"], 1.0, -1.0)
    
    core_cols = ["daily_return", "momentum_3m", "momentum_6m", "ma_50_ratio", "ma_200_ratio", "ma_crossover_signal"]
    df.dropna(subset=["daily_return"], inplace=True)
    
    # Clip extreme returns and momentum to prevent exploding gradients
    df["daily_return"] = df["daily_return"].clip(-0.5, 0.5)
    df["momentum_3m"] = df["momentum_3m"].clip(-1.0, 5.0)
    df["momentum_6m"] = df["momentum_6m"].clip(-1.0, 5.0)
    
    df[core_cols] = df[core_cols].fillna(0)
    return df

def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    df["momentum_5"] = close.pct_change(5).clip(-1.0, 3.0)
    df["momentum_20"] = close.pct_change(20).clip(-1.0, 4.0)
    df["momentum_60"] = close.pct_change(60).clip(-1.0, 5.0)
    df["momentum_acceleration"] = (df["momentum_5"] - df["momentum_20"]).clip(-2.0, 2.0)
    
    ratio = df["momentum_20"] / df["momentum_60"].replace(0.0, np.nan)
    df["trend_strength"] = ratio.rolling(63, min_periods=10).rank(pct=True).fillna(0.5) * 2.0 - 1.0
    return df

def calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    rets = df["daily_return"] if "daily_return" in df.columns else np.log(df["close"]).diff()
    df["volatility_20"] = (rets.rolling(20, min_periods=10).std() * np.sqrt(252.0)).clip(0.0, 5.0)
    
    med_60 = df["volatility_20"].rolling(60, min_periods=20).median().replace(0.0, np.nan)
    df["volatility_spike"] = (df["volatility_20"] / (med_60 + 1e-9)).clip(0.0, 10.0)
    df["volatility_percentile"] = df["volatility_20"].rolling(252, min_periods=20).rank(pct=True).clip(0, 1)
    df["volatility_trend"] = np.sign(df["volatility_20"] - df["volatility_20"].shift(5)).fillna(0.0)
    return df

def calculate_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    vol = pd.to_numeric(df["volume"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    
    vol_mean_20 = vol.rolling(20, min_periods=5).mean()
    df["volume_mean"] = robust_zscore(vol_mean_20)
    
    spike_raw = (vol / vol_mean_20.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    df["volume_spike"] = robust_zscore(spike_raw)
    
    dollar_vol = close * vol
    df["dollar_volume"] = robust_zscore(dollar_vol)
    
    dv_20 = dollar_vol.rolling(20, min_periods=5).mean()
    dv_mean_roll = dollar_vol.rolling(252, min_periods=60).mean().replace(0.0, np.nan)
    ratio_raw = (dv_20 / dv_mean_roll).replace([np.inf, -np.inf], np.nan)
    df["turnover_ratio"] = robust_zscore(ratio_raw)
    return df

def detect_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(df["close"], errors="coerce")
    m50_r = df["ma_50_ratio"] if "ma_50_ratio" in df.columns else (ta.sma(close, 50) / close - 1)
    m200_r = df["ma_200_ratio"] if "ma_200_ratio" in df.columns else (ta.sma(close, 200) / close - 1)
    
    df["trend_regime"] = "sideways"
    df.loc[(m50_r > 0) & (m200_r > 0) & (m50_r > m200_r), "trend_regime"] = "bull_trend"
    df.loc[(m50_r < 0) & (m200_r < 0) & (m50_r < m200_r), "trend_regime"] = "bear_trend"
    
    vol_pct = df["volatility_percentile"] if "volatility_percentile" in df.columns else df["volatility_20"].rolling(252, min_periods=20).rank(pct=True)
    df["volatility_regime"] = "normal_vol"
    df.loc[vol_pct < 0.33, "volatility_regime"] = "low_vol"
    df.loc[vol_pct > 0.67, "volatility_regime"] = "high_vol"
    
    t_score = np.where(df["trend_regime"] == "bull_trend", 1.0, np.where(df["trend_regime"] == "bear_trend", -1.0, 0.0))
    v_mult = np.where(df["volatility_regime"] == "low_vol", 1.0, np.where(df["volatility_regime"] == "high_vol", 0.0, 0.5))
    df["regime_score"] = (t_score * v_mult).clip(-1.0, 1.0)
    return df



def build_feature_matrix(df: pd.DataFrame, config=None) -> pd.DataFrame:
    """
    Construct an enriched per-ticker feature matrix from raw OHLCV.

    Financial intuition
    -------------------
    - Starts from the existing trend-agent features (momentum_3m/6m,
      moving averages, crossover, daily returns).
    - Adds more granular momentum, volatility structure, and liquidity
      metrics to capture the shape of price moves and trading activity.
    - Derives simple per-name regime labels and scores to reflect
      whether a stock is in a favourable trend/risk environment.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data from the market data loader. Must contain the
        usual price and volume columns with either lower-case
        (``open``, ``high``, ``low``, ``close``, ``volume``) or
        title-case (``Open``, ``High``, ``Low``, ``Close``, ``Volume``)
        names.

    Returns
    -------
    pd.DataFrame
        Enriched feature matrix containing all original columns plus
        the additional features described above.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    work = df.copy()
    if "close" not in work.columns and "Close" in work.columns:
        work["close"] = work["Close"]
    if "volume" not in work.columns and "Volume" in work.columns:
        work["volume"] = work["Volume"]

    base = calculate_core_trend_features(work)
    if "close" not in base.columns and "close" in work.columns:
        base["close"] = work["close"]
    if "volume" not in base.columns and "volume" in work.columns:
        base["volume"] = work["volume"]

    try:
        base = calculate_momentum_features(base)
    except Exception as exc:
        warnings.warn(
            f"calculate_momentum_features failed with {type(exc).__name__}: {exc}",
            UserWarning,
        )

    try:
        base = calculate_volatility_features(base)
    except Exception as exc:
        warnings.warn(
            f"calculate_volatility_features failed with {type(exc).__name__}: {exc}",
            UserWarning,
        )

    try:
        base = calculate_liquidity_features(base)
    except Exception as exc:
        warnings.warn(
            f"calculate_liquidity_features failed with {type(exc).__name__}: {exc}",
            UserWarning,
        )

    try:
        base = detect_market_regime(base)
    except Exception as exc:
        warnings.warn(
            f"detect_market_regime failed with {type(exc).__name__}: {exc}",
            UserWarning,
        )

    # CAPM: Jensen's alpha, beta, residual vol (rolling 60d vs SPY; alpha z-scored over 252d)
    try:
        if "daily_return" in base.columns and not base.empty:
            try:
                from utils.market_data import get_ohlcv
                ix = base.index
                start = ix.min().strftime("%Y-%m-%d") if hasattr(ix.min(), "strftime") else str(ix.min())[:10]
                end = ix.max().strftime("%Y-%m-%d") if hasattr(ix.max(), "strftime") else str(ix.max())[:10]
                spy = get_ohlcv("SPY", start, end, use_cache=True, cache_ttl_days=0)
                if spy is not None and not spy.empty and "Close" in spy.columns:
                    spy_ret = spy["Close"].pct_change()
                    stock_ret = base["daily_return"]
                    capm_df = compute_capm_features(stock_ret, spy_ret)
                    for col in ("capm_alpha", "capm_beta", "capm_residual_vol"):
                        if col in capm_df.columns:
                            base[col] = capm_df[col]
            except Exception:
                base["capm_alpha"] = np.nan
                base["capm_beta"] = 1.0
                base["capm_residual_vol"] = np.nan
        else:
            base["capm_alpha"] = np.nan
            base["capm_beta"] = 1.0
            base["capm_residual_vol"] = np.nan
    except Exception as exc:
        warnings.warn(
            f"CAPM features failed with {type(exc).__name__}: {exc}",
            UserWarning,
        )
        if "capm_alpha" not in base.columns:
            base["capm_alpha"] = np.nan
            base["capm_beta"] = 1.0
            base["capm_residual_vol"] = np.nan

    # GBM-derived features (optional; only when gbm_enabled in config to save time)
    if config is not None and getattr(config, "gbm_enabled", False):
        try:
            from simulation.gbm import gbm_price_targets
            price_col = "Close" if "Close" in base.columns else "close"
            if price_col in base.columns and not base.empty:
                holding = int(getattr(config, "holding_period_days", 5))
                gbm_df = gbm_price_targets(
                    base[price_col],
                    horizon_days=holding,
                    n_paths=500,
                    seed=42,
                    window=252,
                )
                if not gbm_df.empty:
                    for src, dst in [
                        ("prob_positive", "gbm_prob_positive"),
                        ("expected_return", "gbm_expected_return"),
                        ("gbm_var_95", "gbm_var_95"),
                    ]:
                        if src in gbm_df.columns:
                            base[dst] = gbm_df[src].reindex(base.index)
        except Exception as exc:
            warnings.warn(
                f"GBM features failed with {type(exc).__name__}: {exc}",
                UserWarning,
            )

    if "gbm_prob_positive" not in base.columns:
        base["gbm_prob_positive"] = np.nan
        base["gbm_expected_return"] = np.nan
        base["gbm_var_95"] = np.nan

    # Support for fundamental/alternative signals (SUE, Short Interest, Analyst Revisions)
    try:
        base["f_sue_score"] = compute_sue_score(base)
        shorts = compute_short_interest_features(base)
        for col in shorts.columns:
            base[col] = shorts[col]
        base["f_analyst_revisions"] = compute_analyst_revision_momentum(base)
    except Exception as exc:
        warnings.warn(
            f"Alternative features failed with {type(exc).__name__}: {exc}",
            UserWarning,
        )

    # Drop known redundant columns when the canonical version is present.
    enriched = base
    for redundant_col, canonical_col in KNOWN_REDUNDANCIES.items():
        if redundant_col in enriched.columns and canonical_col in enriched.columns:
            enriched.drop(columns=[redundant_col], inplace=True)

    logger = logging.getLogger(__name__)
    if logger.isEnabledFor(logging.DEBUG):
        try:
            from utils.feature_audit import compute_feature_correlation_report
            report = compute_feature_correlation_report(enriched)
            high_corr = report[report["recommendation"].isin(["drop_b", "drop_a"])]
            if not high_corr.empty:
                logger.debug(
                    "High feature correlation detected:\n%s",
                    high_corr[["feature_a", "feature_b", "correlation"]].to_string(),
                )
        except Exception:
            pass

    return sanitize_dataframe(enriched)

