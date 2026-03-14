"""
Signal Engine
===============
Supports two modes:

    "price"  — Vectorised trend + volatility signals from historical OHLCV.
               Fast, suitable for long historical backtests.

    "full"   — Enriches price-based signals with live multi-agent sentiment
               (regional news, global news, social) fetched once per ticker.
               Suited for recent-window backtests and paper-trading analysis.

Agent weights are configurable and control how much each source
contributes to the final adjusted score.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from agents.trend_agent.feature_engineering import build_features
from agents.volatility_agent.volatility_model import compute_rolling_confidence
from main import CONFIDENCE_MULTIPLIER, compute_rolling_trend_scores, classify_final_signal

# Calendar-day buffers for downloading enough data around the backtest window
HISTORY_BUFFER_DAYS = 400   # 200-day MA + 126-day momentum warm-up
EXIT_BUFFER_DAYS = 30       # room for last trade's exit

_DEFAULT_WEIGHTS = {
    "trend": 1.0,
    "regional_news": 0.5,
    "global_news": 0.4,
    "social": 0.3,
}

logger = logging.getLogger(__name__)


def _normalise_regime_label(label: str) -> str:
    """
    Map any regime label to the canonical set used by regime models.

    Handles both MarketRegimeAgent labels (Bull/Bear/HighVol/Sideways/Crisis)
    and feature_pipeline labels (bull_trend/bear_trend/sideways/high_vol etc.).
    Returns one of: 'Bull', 'Bear', 'HighVol', 'Sideways'.
    """
    mapping = {
        "bull_trend": "Bull",
        "Bull": "Bull",
        "bear_trend": "Bear",
        "Bear": "Bear",
        "sideways": "Sideways",
        "Sideways": "Sideways",
        "high_vol": "HighVol",
        "HighVol": "HighVol",
        "Crisis": "Bear",
    }
    return mapping.get(str(label), "Sideways")


class SignalEngine:
    """
    Multi-agent signal generator.

    Parameters:
        weights         : dict mapping agent names to score multipliers
        learned_weights : LearnedWeights object (from weight_learning_agent).
                          When set, overrides rule-based weight computation.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        learned_weights=None,
        regime_weights: dict | None = None,
        regime_series: pd.Series | None = None,
        signal_smoothing_enabled: bool = True,
        signal_smoothing_span: int = 5,
    ):
        self.weights = dict(_DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.learned_weights = learned_weights
        self.regime_weights = regime_weights or None
        self.regime_series = regime_series
        self.signal_smoothing_enabled = bool(signal_smoothing_enabled)
        self.signal_smoothing_span = int(signal_smoothing_span)

    # ==============================================================
    # Core: vectorised price-based signals (trend + volatility)
    # ==============================================================

    def generate_signals(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trend_score, confidence, adjusted_score, signal for every bar.
        Uses only OHLCV price data — no external APIs.

        When learned_weights is set, the adjusted_score uses learned
        coefficients instead of the rule-based formula.
        """
        features = build_features(stock_data)
        if features.empty:
            logger.warning("SignalEngine: build_features returned empty frame; no signals.")
            return pd.DataFrame()

        logger.debug(
            "SignalEngine: features built: shape=%s, dates=%s→%s, cols=%s",
            features.shape,
            features.index.min(),
            features.index.max(),
            list(features.columns),
        )
        if features.isna().all().any():
            bad_cols = [c for c in features.columns if features[c].isna().all()]
            logger.warning("SignalEngine: all-NaN feature columns detected: %s", bad_cols)

        trend_scores = compute_rolling_trend_scores(features)
        if trend_scores.isna().all():
            logger.warning("SignalEngine: trend_scores are all NaN.")
        else:
            ts_desc = trend_scores.describe()
            logger.debug(
                "SignalEngine: trend_scores stats: min=%.4f max=%.4f mean=%.4f std=%.4f",
                float(ts_desc["min"]),
                float(ts_desc["max"]),
                float(ts_desc["mean"]),
                float(ts_desc["std"]),
            )

        daily_ret = features["daily_return"]
        rolling_conf = compute_rolling_confidence(daily_ret, window=20)
        conf_mult = rolling_conf.map(CONFIDENCE_MULTIPLIER).fillna(0.5)

        f_trend = trend_scores * conf_mult

        price_col = "AdjClose" if "AdjClose" in stock_data.columns else "Close"
        close = stock_data[price_col].reindex(features.index)
        volume = stock_data["Volume"].reindex(features.index)
        ret_5d = close.pct_change(5)
        ret_10d = close.pct_change(10)
        rolling_vol_10 = daily_ret.rolling(10).std()
        rolling_vol_20 = daily_ret.rolling(20).std()
        vol_ma20 = volume.rolling(20).mean()
        vol_std20 = volume.rolling(20).std().replace(0, np.nan)
        relative_volume = (volume / vol_ma20).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        volume_zscore = ((volume - vol_ma20) / vol_std20).replace([np.inf, -np.inf], np.nan).fillna(0)
        rolling_corr_market_20 = pd.Series(0.0, index=features.index)

        regime_label_for_log = "Sideways"
        if self.regime_series is not None and not self.regime_series.empty:
            raw_label = self.regime_series.reindex(features.index).iloc[-1]
            regime_label_for_log = _normalise_regime_label(raw_label)
        elif "trend_regime" in features.columns:
            raw_label = features["trend_regime"].iloc[-1]
            regime_label_for_log = _normalise_regime_label(raw_label)

        if self.regime_weights is not None:
            if self.regime_series is not None:
                base_regimes = self.regime_series.reindex(features.index)
            elif "trend_regime" in features.columns:
                base_regimes = features["trend_regime"]
            else:
                base_regimes = pd.Series("Sideways", index=features.index)

            regimes = base_regimes.astype(str).map(_normalise_regime_label).fillna("Sideways")
            default_lw = self.regime_weights.get("Sideways") or next(iter(self.regime_weights.values()))
            adjusted = pd.Series(np.nan, index=features.index, dtype=float)
            for reg_key, lw in self.regime_weights.items():
                canon = _normalise_regime_label(reg_key)
                mask = regimes == canon
                if not mask.any():
                    continue
                raw = (
                    lw.intercept
                    + lw.w_trend * f_trend.loc[mask]
                    + getattr(lw, "w_ret_5d", 0) * ret_5d.loc[mask].fillna(0)
                    + getattr(lw, "w_ret_10d", 0) * ret_10d.loc[mask].fillna(0)
                    + getattr(lw, "w_vol_10", 0) * rolling_vol_10.loc[mask].fillna(0)
                    + getattr(lw, "w_vol", 0) * rolling_vol_20.loc[mask].fillna(0)
                    + getattr(lw, "w_rel_vol", 0) * relative_volume.loc[mask]
                    + getattr(lw, "w_vol_zscore", 0) * volume_zscore.loc[mask]
                    + getattr(lw, "w_corr_market", 0) * rolling_corr_market_20.loc[mask]
                )
                adjusted.loc[mask] = raw * getattr(lw, "score_scale", 1.0)
            still_missing = adjusted.isna()
            if still_missing.any():
                raw_def = (
                    default_lw.intercept
                    + default_lw.w_trend * f_trend[still_missing]
                    + getattr(default_lw, "w_ret_5d", 0) * ret_5d[still_missing].fillna(0)
                    + getattr(default_lw, "w_ret_10d", 0) * ret_10d[still_missing].fillna(0)
                    + getattr(default_lw, "w_vol_10", 0) * rolling_vol_10[still_missing].fillna(0)
                    + getattr(default_lw, "w_vol", 0) * rolling_vol_20[still_missing].fillna(0)
                    + getattr(default_lw, "w_rel_vol", 0) * relative_volume[still_missing]
                    + getattr(default_lw, "w_vol_zscore", 0) * volume_zscore[still_missing]
                    + getattr(default_lw, "w_corr_market", 0) * rolling_corr_market_20[still_missing]
                )
                adjusted.loc[still_missing] = raw_def * getattr(default_lw, "score_scale", 1.0)
            adjusted = adjusted.fillna(0)
            logger.debug(
                "SignalEngine: ticker=%s regime=%s weights_source=%s",
                "N/A",
                regime_label_for_log,
                "regime_specific",
            )
        elif self.learned_weights is not None:
            lw = self.learned_weights
            raw = (
                lw.intercept
                + lw.w_trend * f_trend
                + getattr(lw, "w_ret_5d", 0) * ret_5d.fillna(0)
                + getattr(lw, "w_ret_10d", 0) * ret_10d.fillna(0)
                + getattr(lw, "w_vol_10", 0) * rolling_vol_10.fillna(0)
                + getattr(lw, "w_vol", 0) * rolling_vol_20.fillna(0)
                + getattr(lw, "w_rel_vol", 0) * relative_volume
                + getattr(lw, "w_vol_zscore", 0) * volume_zscore
                + getattr(lw, "w_corr_market", 0) * rolling_corr_market_20
            )
            scale = getattr(lw, "score_scale", 1.0)
            direction = getattr(lw, "score_direction", 1)
            adjusted = (raw * scale) * direction
            if adjusted.isna().all():
                logger.warning("SignalEngine: adjusted_score (learned) is all NaN.")
            else:
                adj_desc = adjusted.describe()
                logger.debug(
                    "SignalEngine: adjusted_score (learned) stats: min=%.4f max=%.4f mean=%.4f std=%.4f",
                    float(adj_desc["min"]),
                    float(adj_desc["max"]),
                    float(adj_desc["mean"]),
                    float(adj_desc["std"]),
                )
            logger.debug(
                "SignalEngine: ticker=%s regime=%s weights_source=%s",
                "N/A",
                regime_label_for_log,
                "default",
            )
        else:
            adjusted = f_trend * self.weights.get("trend", 1.0)
            if adjusted.isna().all():
                logger.warning("SignalEngine: adjusted_score (price-only) is all NaN.")
            else:
                adj_desc = adjusted.describe()
                logger.debug(
                    "SignalEngine: adjusted_score (price-only) stats: min=%.4f max=%.4f mean=%.4f std=%.4f",
                    float(adj_desc["min"]),
                    float(adj_desc["max"]),
                    float(adj_desc["mean"]),
                    float(adj_desc["std"]),
                )
        signal_df = pd.DataFrame(
            {
                "trend_score": trend_scores,
                "confidence": rolling_conf,
                "adjusted_score": adjusted,
            },
            index=features.index,
        )

        if self.signal_smoothing_enabled:
            span = max(1, int(self.signal_smoothing_span))
            signal_df["smoothed_score"] = (
                signal_df["adjusted_score"]
                .ewm(span=span, min_periods=min(3, span))
                .mean()
            )
        score_col = "smoothed_score" if "smoothed_score" in signal_df.columns else "adjusted_score"

        # Dynamic, cost-aware thresholds when config is available; base on
        # config.min_signal_strength so behaviour is user-tunable instead of
        # hard-coded.
        base = 0.3
        if hasattr(self, "config") and getattr(self, "config") is not None:
            base = float(getattr(self.config, "min_signal_strength", base))
            try:
                from utils.signal_threshold import compute_dynamic_thresholds

                if "volatility_percentile" in features.columns:
                    vol_val = features["volatility_percentile"].iloc[-1]
                    vol_pct = 0.5 if pd.isna(vol_val) else float(vol_val)
                else:
                    vol_pct = 0.5
                bull_thresh, bear_thresh = compute_dynamic_thresholds(self.config, vol_pct)
            except Exception as exc:
                logger.warning(
                    "SignalEngine: compute_dynamic_thresholds failed (%s); falling back to ±%.4f",
                    exc,
                    base,
                )
                bull_thresh, bear_thresh = base, -base
        else:
            bull_thresh, bear_thresh = base, -base

        # Convention: high score → Bullish (long); low score → Bearish (short).
        signal_df["signal"] = "Neutral"
        signal_df.loc[signal_df[score_col] > bull_thresh, "signal"] = "Bullish"
        signal_df.loc[signal_df[score_col] < bear_thresh, "signal"] = "Bearish"
        signal_df["bull_threshold"] = bull_thresh
        signal_df["bear_threshold"] = bear_thresh

        counts = signal_df["signal"].value_counts(dropna=False).to_dict()
        logger.debug(
            "SignalEngine: classification using score_col=%s, bull_thresh=%.4f, bear_thresh=%.4f, counts=%s",
            score_col,
            float(bull_thresh),
            float(bear_thresh),
            counts,
        )

        abs_adj = adjusted.clip(-1.0, 1.0).abs()
        confidence_numeric = abs_adj.clip(lower=0.3, upper=0.95)
        signal_df["confidence_numeric"] = confidence_numeric

        # CAPM: Jensen's alpha (z-scored), beta, residual vol (rolling 60d vs SPY)
        try:
            from features.capm_features import compute_capm_features
            from utils.market_data import get_ohlcv as _get_ohlcv
            stock_ret = daily_ret
            ix = features.index
            start = ix.min().strftime("%Y-%m-%d") if hasattr(ix.min(), "strftime") else str(ix.min())[:10]
            end = ix.max().strftime("%Y-%m-%d") if hasattr(ix.max(), "strftime") else str(ix.max())[:10]
            spy_df = _get_ohlcv("SPY", start, end, use_cache=True, cache_ttl_days=0)
            if spy_df is not None and not spy_df.empty and "Close" in spy_df.columns:
                spy_ret = spy_df["Close"].pct_change()
                capm_df = compute_capm_features(stock_ret, spy_ret)
                for col in ("capm_alpha", "capm_beta", "capm_residual_vol"):
                    if col in capm_df.columns:
                        signal_df[col] = capm_df[col].reindex(signal_df.index)
            else:
                signal_df["capm_alpha"] = np.nan
                signal_df["capm_beta"] = 1.0
                signal_df["capm_residual_vol"] = np.nan
        except Exception:
            signal_df["capm_alpha"] = np.nan
            signal_df["capm_beta"] = 1.0
            signal_df["capm_residual_vol"] = np.nan

        return signal_df

    # ==============================================================
    # Multi-agent: fetch live sentiments for one ticker
    # ==============================================================

    @staticmethod
    def fetch_ticker_sentiments(ticker: str) -> dict[str, float]:
        """
        Call all news / social agents for *ticker* at the current time.

        Returns a dict with six keys:
            regional_sentiment, regional_impact,
            global_sentiment,   global_impact,
            social_sentiment,   social_impact

        Any agent that fails silently returns zeros for its fields.
        """
        result = {
            "regional_sentiment": 0.0, "regional_impact": 0.0,
            "global_sentiment": 0.0,   "global_impact": 0.0,
            "social_sentiment": 0.0,   "social_impact": 0.0,
        }

        # --- Regional News Agent ---
        try:
            from agents.regional_news_agent import run_regional_news_model
            out = run_regional_news_model(ticker)
            result["regional_sentiment"] = float(out.get("sentiment_score", 0.0))
            result["regional_impact"] = float(out.get("impact_factor", 0.0))
        except Exception:
            pass

        # --- Global News Agent ---
        try:
            from agents.global_news_agent import run_global_news_model
            out = run_global_news_model()
            result["global_sentiment"] = float(out.get("sentiment_score", 0.0))
            result["global_impact"] = float(out.get("impact_factor", 0.0))
        except Exception:
            pass

        # --- Social Sentiment Agent ---
        try:
            from agents.social_sentiment_agent import run_social_sentiment_model
            out = run_social_sentiment_model(ticker)
            result["social_sentiment"] = float(out.get("social_sentiment_score", 0.0))
            result["social_impact"] = float(out.get("impact_factor", 0.0))
        except Exception:
            pass

        return result

    # ==============================================================
    # Multi-agent: overlay sentiments onto price-based signals
    # ==============================================================

    def apply_sentiment_overlay(
        self,
        signal_df: pd.DataFrame,
        sentiments: dict[str, float],
    ) -> pd.DataFrame:
        """
        Add news + social sentiment contributions to the adjusted score,
        then reclassify all signals.

        Uses learned weights when available, otherwise rule-based weights.
        """
        f_regional = sentiments["regional_sentiment"] * sentiments["regional_impact"]
        f_global = sentiments["global_sentiment"] * sentiments["global_impact"]
        f_social = sentiments["social_sentiment"] * sentiments["social_impact"]

        if self.learned_weights is not None:
            lw = self.learned_weights
            overlay = (
                lw.w_regional * f_regional
                + lw.w_global * f_global
                + lw.w_social * f_social
            )
        else:
            overlay = (
                f_regional * self.weights.get("regional_news", 0.0)
                + f_global * self.weights.get("global_news", 0.0)
                + f_social * self.weights.get("social", 0.0)
            )

        enriched = signal_df.copy()
        enriched["adjusted_score"] = enriched["adjusted_score"] + overlay
        enriched["signal"] = enriched["adjusted_score"].apply(classify_final_signal)
        # Recompute numeric confidence from final adjusted_score
        if "confidence_numeric" in enriched.columns:
            abs_adj = enriched["adjusted_score"].clip(-1.0, 1.0).abs()
            enriched["confidence_numeric"] = abs_adj.clip(lower=0.3, upper=0.95)
        return enriched
