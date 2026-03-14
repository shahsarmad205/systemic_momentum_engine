"""
Feature pipeline for single-name OHLCV time series.

This module combines the existing trend-agent feature builder with
additional momentum, volatility, liquidity, and regime features into
one enriched per-ticker feature matrix.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from .momentum_features import calculate_momentum_features
from .volatility_features import calculate_volatility_features
from .liquidity_features import calculate_liquidity_features
from .regime_features import detect_market_regime
from .capm_features import compute_capm_features

_HAS_TREND_AGENT = True

# Columns that duplicate existing SignalEngine features under different names.
# These are dropped from the final feature matrix to avoid feeding redundant
# signals to the weight learner.
KNOWN_REDUNDANCIES = {
    "volatility_20": "rolling_vol_20",   # same calculation, different name
    "volume_spike": "relative_volume",   # equivalent ratio
}


def _call_trend_agent(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Safely call the existing trend-agent build_features, if available.

    Parameters
    ----------
    stock_data : pd.DataFrame
        OHLCV DataFrame as returned by the data layer.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame from the trend agent, or a shallow copy of
        ``stock_data`` if the agent is unavailable or fails.
    """
    global _HAS_TREND_AGENT
    if not _HAS_TREND_AGENT:
        return stock_data.copy()

    try:
        from agents.trend_agent.feature_engineering import build_features as _build_features
    except ImportError:
        _HAS_TREND_AGENT = False
        return stock_data.copy()

    try:
        return _build_features(stock_data)
    except Exception:
        warnings.warn(
            "feature_pipeline.build_feature_matrix: trend_agent.build_features failed; "
            "falling back to raw OHLCV.",
            UserWarning,
        )
        return stock_data.copy()


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

    base = _call_trend_agent(work)
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

    enriched = base

    # Drop known redundant columns when the canonical version is present.
    for redundant_col, canonical_col in KNOWN_REDUNDANCIES.items():
        if redundant_col in enriched.columns and canonical_col in enriched.columns:
            enriched = enriched.drop(columns=[redundant_col])

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
            # Correlation audit is best-effort and must never break the pipeline.
            pass

    return enriched

