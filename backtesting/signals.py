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
from agents.volatility_agent.volatility_model import compute_rolling_confidence, compute_vol_term_structure
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


def _ts_zscore_shifted(s: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """Per-ticker rolling z-score + shift(1). Inference proxy for panel CS z-score used in training."""
    m = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    z = (s - m) / std
    return z.shift(1).fillna(0.0)


def _mean_reversion_features_ts_zscore(close: pd.Series, open_px: pd.Series) -> dict[str, pd.Series]:
    """Raw MR inputs then time-series z (training uses cross-sectional z + shift(1) in feature_builder)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_raw = 100 - (100 / (1 + rs))

    rm20 = close.rolling(20, min_periods=10).max()
    rmi20 = close.rolling(20, min_periods=10).min()
    dist_high_raw = (close - rm20) / rm20.replace(0, np.nan)
    dist_low_raw = (close - rmi20) / rmi20.replace(0, np.nan)
    prev_c = close.shift(1)
    overnight_raw = (open_px - prev_c) / prev_c.replace(0, np.nan)
    intraday_raw = (close - open_px) / open_px.replace(0, np.nan)

    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_w = (bb_upper - bb_lower).replace(0, np.nan)
    bb_pos_raw = (close - bb_lower) / bb_w

    return {
        "rsi_zscore": _ts_zscore_shifted(rsi_raw),
        "bb_position": _ts_zscore_shifted(bb_pos_raw),
        "dist_high": _ts_zscore_shifted(dist_high_raw),
        "dist_low": _ts_zscore_shifted(dist_low_raw),
        "overnight_gap": _ts_zscore_shifted(overnight_raw),
        "intraday_rev": _ts_zscore_shifted(intraday_raw),
    }


def _learned_vix_term_coef(lw) -> float:
    """Prefer w_vix_term_zscore; fall back to legacy w_vix_term for old JSON."""
    z = float(getattr(lw, "w_vix_term_zscore", 0.0) or 0.0)
    if abs(z) > 1e-12:
        return z
    return float(getattr(lw, "w_vix_term", 0.0) or 0.0)


def _learned_rel_vol_coef(lw) -> float:
    z = float(getattr(lw, "w_relative_volume", 0.0) or 0.0)
    if abs(z) > 1e-12:
        return z
    return float(getattr(lw, "w_rel_vol", 0.0) or 0.0)


def _learned_volume_zscore_coef(lw) -> float:
    z = float(getattr(lw, "w_volume_zscore", 0.0) or 0.0)
    if abs(z) > 1e-12:
        return z
    return float(getattr(lw, "w_vol_zscore", 0.0) or 0.0)


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

    def generate_signals(
        self,
        stock_data: pd.DataFrame,
        sector_relative_20d: pd.Series | None = None,
        sector_relative_60d: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Compute trend_score, confidence, adjusted_score, signal for every bar.
        Uses only OHLCV price data — no external APIs.

        When learned_weights is set, the adjusted_score uses learned
        coefficients instead of the rule-based formula.

        Optional sector_relative_* series (indexed like OHLCV dates) should match
        training panel CS z-scores (from feature_builder.sector_relative_features_by_ticker).
        """
        features = build_features(stock_data)
        if features.empty:
            logger.warning("SignalEngine: build_features returned empty frame; no signals.")
            return pd.DataFrame()

        ix = features.index
        if sector_relative_20d is not None:
            sr20 = pd.to_numeric(sector_relative_20d, errors="coerce").reindex(ix).fillna(0.0).astype(float)
        else:
            sr20 = pd.Series(0.0, index=ix, dtype=float)
        if sector_relative_60d is not None:
            sr60 = pd.to_numeric(sector_relative_60d, errors="coerce").reindex(ix).fillna(0.0).astype(float)
        else:
            sr60 = pd.Series(0.0, index=ix, dtype=float)

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
        ret_20d = close.pct_change(20)
        ret_60d = close.pct_change(60)
        momentum_3m = (
            pd.to_numeric(features["momentum_3m"], errors="coerce")
            if "momentum_3m" in features.columns
            else pd.Series(0.0, index=features.index, dtype=float)
        )
        momentum_6m = (
            pd.to_numeric(features["momentum_6m"], errors="coerce")
            if "momentum_6m" in features.columns
            else pd.Series(0.0, index=features.index, dtype=float)
        )
        ma_crossover = (
            pd.to_numeric(features["ma_crossover_signal"], errors="coerce")
            if "ma_crossover_signal" in features.columns
            else pd.Series(0.0, index=features.index, dtype=float)
        )
        cs_momentum_series = (
            momentum_6m.shift(21).rolling(252, min_periods=60).rank(pct=True).fillna(0.0)
        )
        vol_struct = compute_vol_term_structure(daily_ret)
        rolling_vol_5 = pd.to_numeric(vol_struct["vol_5d"], errors="coerce").fillna(0.0)
        rolling_vol_10 = daily_ret.rolling(10).std()
        rolling_vol_20 = daily_ret.rolling(20).std()
        vol_of_vol_20 = pd.to_numeric(vol_struct["vol_of_vol_20"], errors="coerce").fillna(0.0)
        jump_indicator = pd.to_numeric(vol_struct["jump_indicator"], errors="coerce").fillna(0.0)
        vol_ma20 = volume.rolling(20).mean()
        vol_std20 = volume.rolling(20).std().replace(0, np.nan)
        relative_volume = (volume / vol_ma20).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        volume_zscore = ((volume - vol_ma20) / vol_std20).replace([np.inf, -np.inf], np.nan).fillna(0)
        rolling_corr_market_20 = pd.Series(0.0, index=features.index)
        capm_beta_series = pd.Series(0.0, index=features.index, dtype=float)

        open_series = (
            stock_data["Open"].reindex(features.index)
            if "Open" in stock_data.columns
            else close.copy()
        )
        mr_feat = _mean_reversion_features_ts_zscore(close, open_series)

        # Optional macro features (only needed for learned-weight models).
        vix_zscore_series = pd.Series(0.0, index=features.index, dtype=float)
        vol_spike_series = pd.Series(0.0, index=features.index, dtype=float)
        vix_term_zscore_series = pd.Series(0.0, index=features.index, dtype=float)
        need_macro = False
        if self.learned_weights is not None:
            need_macro = (
                float(getattr(self.learned_weights, "w_vix_zscore", 0.0) or 0.0) != 0.0
                or float(getattr(self.learned_weights, "w_vol_spike", 0.0) or 0.0) != 0.0
                or abs(_learned_vix_term_coef(self.learned_weights)) > 1e-12
            )
        if self.regime_weights is not None and not need_macro:
            try:
                for _lw in self.regime_weights.values():
                    if (
                        float(getattr(_lw, "w_vix_zscore", 0.0) or 0.0) != 0.0
                        or float(getattr(_lw, "w_vol_spike", 0.0) or 0.0) != 0.0
                        or abs(_learned_vix_term_coef(_lw)) > 1e-12
                    ):
                        need_macro = True
                        break
            except Exception:
                need_macro = False

        if need_macro:
            try:
                from utils.market_data import get_ohlcv as _get_ohlcv
                ix = features.index
                start = ix.min().strftime("%Y-%m-%d") if hasattr(ix.min(), "strftime") else str(ix.min())[:10]
                end = ix.max().strftime("%Y-%m-%d") if hasattr(ix.max(), "strftime") else str(ix.max())[:10]

                # Vol spike = SPY realised vol (5d) / realised vol (60d)
                spy_df = _get_ohlcv("SPY", start, end, use_cache=True, cache_ttl_days=0)
                if spy_df is not None and not spy_df.empty and "Close" in spy_df.columns:
                    spy_close = pd.to_numeric(spy_df["Close"], errors="coerce").dropna().sort_index()
                    spy_ret = spy_close.pct_change()
                    vol5 = spy_ret.rolling(5).std(ddof=0) * np.sqrt(252.0)
                    vol60 = spy_ret.rolling(60).std(ddof=0) * np.sqrt(252.0)
                    vol_spike = (vol5 / vol60).replace([np.inf, -np.inf], np.nan).shift(1)
                    vol_spike_series = vol_spike.reindex(features.index).fillna(0.0).astype(float)

                # VIX z-score = (VIX - rolling_252d_mean) / rolling_252d_std, shifted by 1
                vix_df = _get_ohlcv("^VIX", start, end, use_cache=True, cache_ttl_days=0)
                if vix_df is not None and not vix_df.empty and "Close" in vix_df.columns:
                    vix_close = pd.to_numeric(vix_df["Close"], errors="coerce").dropna().sort_index()
                    vix_mean_252 = vix_close.rolling(252).mean()
                    vix_std_252 = vix_close.rolling(252).std(ddof=0).replace(0, np.nan)
                    vix_z = ((vix_close - vix_mean_252) / vix_std_252).shift(1)
                    vix_zscore_series = vix_z.reindex(features.index).fillna(0.0).astype(float)

                    vix3m_df = _get_ohlcv("^VIX3M", start, end, use_cache=True, cache_ttl_days=0)
                    if vix3m_df is not None and not vix3m_df.empty and "Close" in vix3m_df.columns:
                        vix3m_close = pd.to_numeric(vix3m_df["Close"], errors="coerce").dropna().sort_index()
                        vix3m_aligned = vix3m_close.reindex(vix_close.index).ffill()
                        vix_ratio_lag = (vix_close / vix3m_aligned.replace(0.0, np.nan)).replace(
                            [np.inf, -np.inf], np.nan
                        ).shift(1)
                        vt_m = vix_ratio_lag.rolling(252, min_periods=60).mean()
                        vt_s = vix_ratio_lag.rolling(252, min_periods=60).std(ddof=0).replace(0, np.nan)
                        vix_tz = (
                            ((vix_ratio_lag - vt_m) / vt_s)
                            .replace([np.inf, -np.inf], np.nan)
                            .shift(1)
                        )
                        vix_term_zscore_series = vix_tz.reindex(features.index).fillna(0.0).astype(float)
            except Exception:
                # If macro downloads fail, fall back to zeros.
                pass

        # CAPM beta (rolling 60d) if any learned/regime model uses it.
        need_capm_beta = False
        if self.learned_weights is not None:
            need_capm_beta = float(getattr(self.learned_weights, "w_capm_beta", 0.0) or 0.0) != 0.0
        if self.regime_weights is not None and not need_capm_beta:
            try:
                need_capm_beta = any(
                    float(getattr(_lw, "w_capm_beta", 0.0) or 0.0) != 0.0
                    for _lw in self.regime_weights.values()
                )
            except Exception:
                need_capm_beta = False
        if need_capm_beta:
            try:
                from features.capm_features import compute_capm_features
                from utils.market_data import get_ohlcv as _get_ohlcv

                ix = features.index
                start = ix.min().strftime("%Y-%m-%d") if hasattr(ix.min(), "strftime") else str(ix.min())[:10]
                end = ix.max().strftime("%Y-%m-%d") if hasattr(ix.max(), "strftime") else str(ix.max())[:10]
                spy_df = _get_ohlcv("SPY", start, end, use_cache=True, cache_ttl_days=0)
                if spy_df is not None and not spy_df.empty and "Close" in spy_df.columns:
                    spy_ret = pd.to_numeric(spy_df["Close"], errors="coerce").pct_change()
                    capm_df = compute_capm_features(daily_ret.astype(float), spy_ret.astype(float), window=60)
                    if "capm_beta" in capm_df.columns:
                        capm_beta_series = (
                            pd.to_numeric(capm_df["capm_beta"], errors="coerce")
                            .reindex(features.index)
                            .fillna(0.0)
                            .astype(float)
                        )
            except Exception:
                pass

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
                    + getattr(lw, "w_cs_momentum", 0) * cs_momentum_series.loc[mask].fillna(0)
                    + getattr(lw, "w_ma_crossover", 0) * ma_crossover.loc[mask].fillna(0)
                    + getattr(lw, "w_rolling_vol_5", 0) * rolling_vol_5.loc[mask].fillna(0)
                    + getattr(lw, "w_vol_10", 0) * rolling_vol_10.loc[mask].fillna(0)
                    + getattr(lw, "w_vol", 0) * rolling_vol_20.loc[mask].fillna(0)
                    + getattr(lw, "w_vol_of_vol", 0) * vol_of_vol_20.loc[mask].fillna(0)
                    + getattr(lw, "w_jump_indicator", 0) * jump_indicator.loc[mask].fillna(0)
                    + _learned_rel_vol_coef(lw) * relative_volume.loc[mask]
                    + _learned_volume_zscore_coef(lw) * volume_zscore.loc[mask]
                    + getattr(lw, "w_corr_market", 0) * rolling_corr_market_20.loc[mask]
                    + getattr(lw, "w_vix_zscore", 0) * vix_zscore_series.loc[mask]
                    + getattr(lw, "w_vol_spike", 0) * vol_spike_series.loc[mask]
                    + _learned_vix_term_coef(lw) * vix_term_zscore_series.loc[mask]
                    + getattr(lw, "w_rsi_zscore", 0) * mr_feat["rsi_zscore"].loc[mask]
                    + getattr(lw, "w_bb_position", 0) * mr_feat["bb_position"].loc[mask]
                    + getattr(lw, "w_dist_high", 0) * mr_feat["dist_high"].loc[mask]
                    + getattr(lw, "w_dist_low", 0) * mr_feat["dist_low"].loc[mask]
                    + getattr(lw, "w_overnight_gap", 0) * mr_feat["overnight_gap"].loc[mask]
                    + getattr(lw, "w_intraday_rev", 0) * mr_feat["intraday_rev"].loc[mask]
                    + getattr(lw, "w_sector_relative_20d", 0) * sr20.loc[mask]
                    + getattr(lw, "w_sector_relative_60d", 0) * sr60.loc[mask]
                )
                adjusted.loc[mask] = raw * getattr(lw, "score_scale", 1.0)
            still_missing = adjusted.isna()
            if still_missing.any():
                raw_def = (
                    default_lw.intercept
                    + default_lw.w_trend * f_trend[still_missing]
                    + getattr(default_lw, "w_ret_5d", 0) * ret_5d[still_missing].fillna(0)
                    + getattr(default_lw, "w_ret_10d", 0) * ret_10d[still_missing].fillna(0)
                    + getattr(default_lw, "w_cs_momentum", 0) * cs_momentum_series[still_missing].fillna(0)
                    + getattr(default_lw, "w_ma_crossover", 0) * ma_crossover[still_missing].fillna(0)
                    + getattr(default_lw, "w_rolling_vol_5", 0) * rolling_vol_5[still_missing].fillna(0)
                    + getattr(default_lw, "w_vol_10", 0) * rolling_vol_10[still_missing].fillna(0)
                    + getattr(default_lw, "w_vol", 0) * rolling_vol_20[still_missing].fillna(0)
                    + getattr(default_lw, "w_vol_of_vol", 0) * vol_of_vol_20[still_missing].fillna(0)
                    + getattr(default_lw, "w_jump_indicator", 0) * jump_indicator[still_missing].fillna(0)
                    + _learned_rel_vol_coef(default_lw) * relative_volume[still_missing]
                    + _learned_volume_zscore_coef(default_lw) * volume_zscore[still_missing]
                    + getattr(default_lw, "w_corr_market", 0) * rolling_corr_market_20[still_missing]
                    + getattr(default_lw, "w_vix_zscore", 0) * vix_zscore_series[still_missing]
                    + getattr(default_lw, "w_vol_spike", 0) * vol_spike_series[still_missing]
                    + _learned_vix_term_coef(default_lw) * vix_term_zscore_series[still_missing]
                    + getattr(default_lw, "w_rsi_zscore", 0) * mr_feat["rsi_zscore"][still_missing]
                    + getattr(default_lw, "w_bb_position", 0) * mr_feat["bb_position"][still_missing]
                    + getattr(default_lw, "w_dist_high", 0) * mr_feat["dist_high"][still_missing]
                    + getattr(default_lw, "w_dist_low", 0) * mr_feat["dist_low"][still_missing]
                    + getattr(default_lw, "w_overnight_gap", 0) * mr_feat["overnight_gap"][still_missing]
                    + getattr(default_lw, "w_intraday_rev", 0) * mr_feat["intraday_rev"][still_missing]
                    + getattr(default_lw, "w_sector_relative_20d", 0) * sr20[still_missing]
                    + getattr(default_lw, "w_sector_relative_60d", 0) * sr60[still_missing]
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
                + getattr(lw, "w_cs_momentum", 0) * cs_momentum_series.fillna(0)
                + getattr(lw, "w_ma_crossover", 0) * ma_crossover.fillna(0)
                + getattr(lw, "w_rolling_vol_5", 0) * rolling_vol_5.fillna(0)
                + getattr(lw, "w_vol_10", 0) * rolling_vol_10.fillna(0)
                + getattr(lw, "w_vol", 0) * rolling_vol_20.fillna(0)
                + getattr(lw, "w_vol_of_vol", 0) * vol_of_vol_20.fillna(0)
                + getattr(lw, "w_jump_indicator", 0) * jump_indicator.fillna(0)
                + _learned_rel_vol_coef(lw) * relative_volume
                + _learned_volume_zscore_coef(lw) * volume_zscore
                + getattr(lw, "w_corr_market", 0) * rolling_corr_market_20
                + getattr(lw, "w_vix_zscore", 0) * vix_zscore_series
                + getattr(lw, "w_vol_spike", 0) * vol_spike_series
                + _learned_vix_term_coef(lw) * vix_term_zscore_series
                + getattr(lw, "w_rsi_zscore", 0) * mr_feat["rsi_zscore"]
                + getattr(lw, "w_bb_position", 0) * mr_feat["bb_position"]
                + getattr(lw, "w_dist_high", 0) * mr_feat["dist_high"]
                + getattr(lw, "w_dist_low", 0) * mr_feat["dist_low"]
                + getattr(lw, "w_overnight_gap", 0) * mr_feat["overnight_gap"]
                + getattr(lw, "w_intraday_rev", 0) * mr_feat["intraday_rev"]
                + getattr(lw, "w_sector_relative_20d", 0) * sr20
                + getattr(lw, "w_sector_relative_60d", 0) * sr60
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
                "sector_relative_20d": sr20,
                "sector_relative_60d": sr60,
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
