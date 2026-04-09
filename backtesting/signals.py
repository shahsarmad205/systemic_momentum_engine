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
from typing import Any

import numpy as np
import pandas as pd

from features.feature_pipeline import calculate_core_trend_features as build_features
from agents.volatility_agent.volatility_model import (
    compute_rolling_confidence,
    compute_vol_term_structure,
)
CONFIDENCE_MULTIPLIER = {
    "High": 1.0,
    "Medium": 0.6,
    "Low": 0.3,
}

def classify_final_signal(adjusted_trend: float) -> str:
    if adjusted_trend > 0.5:
        return "Bullish"
    elif adjusted_trend < -0.5:
        return "Bearish"
    else:
        return "Neutral"

def compute_rolling_trend_scores(features: pd.DataFrame) -> pd.Series:
    scale = 10.0
    scores = (
        0.30 * features["momentum_3m"] * scale
        + 0.25 * features["momentum_6m"] * scale
        + 0.25 * features["ma_crossover_signal"]
        + 0.20 * features["daily_return"] * scale
    )
    return scores

from utils.ensemble_scoring import compute_ensemble_score, load_ensemble_models

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
    """
    Per-ticker rolling (time-series) z-score + shift(1).

    IMPORTANT — distribution mismatch warning
    ------------------------------------------
    During *training* (feature_builder.py), mean-reversion features are
    normalised using **cross-sectional** z-scores: for each date, we subtract
    the cross-sectional mean and divide by the cross-sectional std across all
    tickers in the panel.  This produces a zero-mean, unit-variance signal
    *relative to the peer group*.

    During *inference* (this function), we use a **time-series** z-score:
    we subtract the ticker's own rolling mean and divide by its rolling std
    over the past *window* days.  This captures whether the signal is extreme
    *relative to its own history*, not relative to the current peer group.

    The two approaches are NOT equivalent:
      - CS z-score: captures relative rank within cross-section (better for
        long-short, less sensitive to market-wide level shifts)
      - TS z-score: captures absolute deviation from own history (better for
        single-ticker signals, more sensitive to regime changes)

    The mismatch introduces a systematic distribution shift between training
    and inference.  Estimated impact: 10–30 bps per trade in overconfident
    signals (the ML model was calibrated on CS z-scores but sees TS z-scores).

    Fix options:
      Option A (preferred): At inference time, pass the full cross-sectional
        panel and use CS z-scores.  This requires all tickers to be processed
        together, which the backtester already does in ``prepare_data``.
        Use ``panel_features`` injection (Backtester._build_panel_features).
      Option B (current): Accept the mismatch but document it clearly.
        TS z-scores are still valid normalisation; they just have a different
        distributional interpretation than the training features.

    Until Option A is fully wired, this function is the operational path.
    The flag ``USE_CS_ZSCORE_AT_INFERENCE`` (set via environment or config)
    controls which path is taken when the panel is available.
    """
    m = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    z = (s - m) / std
    return z.shift(1).fillna(0.0)


def _cs_zscore_from_panel(
    raw_series: pd.Series,
    panel: pd.DataFrame,
    ticker: str,
    clip: float = 3.0,
) -> pd.Series:
    """
    Cross-sectional z-score for a single ticker's raw series.

    Given ``panel`` (shape: dates × tickers) containing pre-computed raw
    values, compute the CS z-score: (x_i - mean_cs) / std_cs per date,
    then shift(1) to avoid lookahead.

    This produces the same normalisation as feature_builder training.

    Parameters
    ----------
    raw_series : per-ticker raw values (dates index)
    panel : full cross-section of the same raw values (dates × tickers)
    ticker : name of the current ticker (used to extract from panel if needed)
    clip : winsorise at ±clip std deviations
    """
    cs_mean = panel.mean(axis=1)
    cs_std = panel.std(axis=1).replace(0, np.nan)

    # Align to raw_series index
    cs_mean = cs_mean.reindex(raw_series.index)
    cs_std = cs_std.reindex(raw_series.index)

    z = (raw_series - cs_mean) / cs_std
    z = z.clip(-clip, clip).shift(1).fillna(0.0)
    return z


def _mean_reversion_features_ts_zscore(close: pd.Series, open_px: pd.Series) -> dict[str, pd.Series]:
    """
    Compute mean-reversion features using time-series z-score normalisation.

    NOTE: Training used cross-sectional z-scores (see _ts_zscore_shifted
    docstring for the full explanation of the mismatch and fix options).
    """
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
        "rsi_14_raw": rsi_raw,
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
        self._ensemble_models_cache = None
        self._warned_all_nan_cols = False
        self._warned_ensemble_norm_ignored = False
        # C2: regime-conditional model routing
        self._current_regime: str = "Normal"
        self._regime_model_cache: dict[str, Any] = {}  # regime_label → LoadedEnsembleModel

    def set_regime(self, regime: str) -> None:
        """C2: Called by the backtester each day to route predictions to the regime-specific model."""
        self._current_regime = str(regime) if regime else "Normal"

    def _get_fundamental_features(self, ticker: str, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return Piotroski F-score + accruals DataFrame aligned to index.
        Caches per ticker per backtest run. Falls back to zeros on any error.
        """
        _FUND_COLS = ["f_score", "accruals_ratio", "roa", "delta_roa", "delta_leverage"]
        zeros = pd.DataFrame(0.0, index=index, columns=_FUND_COLS)
        if not ticker:
            return zeros
        if not hasattr(self, "_fundamental_cache"):
            self._fundamental_cache: dict[str, pd.DataFrame] = {}
        if ticker not in self._fundamental_cache:
            try:
                from features.fundamental_builder import fetch_fundamental_features
                df = fetch_fundamental_features(ticker, index)
                self._fundamental_cache[ticker] = df.reindex(index).fillna(0.0)
            except Exception:
                self._fundamental_cache[ticker] = zeros.copy()
        cached = self._fundamental_cache[ticker]
        return cached.reindex(index).fillna(0.0)

    # ==============================================================
    # Core: vectorised price-based signals (trend + volatility)
    # ==============================================================

    def generate_signals(
        self,
        stock_data: pd.DataFrame,
        sector_relative_20d: pd.Series | None = None,
        sector_relative_60d: pd.Series | None = None,
        vol_rank: pd.Series | None = None,
        spy_df: pd.DataFrame | None = None,
        vix_df: pd.DataFrame | None = None,
        vix3m_df: pd.DataFrame | None = None,
        panel_features: pd.DataFrame | None = None, # Pillar 29: Joint-Panel Feature Injection
        ticker: str = "",  # C1: used to fetch earnings_surprise per ticker
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

        # Diagnostic features for ML/Ensemble analysis (defined later if in ML mode)
        model_features = pd.DataFrame(index=features.index)

        ix = features.index
        if sector_relative_20d is not None:
            sr20 = pd.to_numeric(sector_relative_20d, errors="coerce").reindex(ix).fillna(0.0).astype(float)
        else:
            sr20 = pd.Series(0.0, index=ix, dtype=float)
        if sector_relative_60d is not None:
            sr60 = pd.to_numeric(sector_relative_60d, errors="coerce").reindex(ix).fillna(0.0).astype(float)
        else:
            sr60 = pd.Series(0.0, index=ix, dtype=float)

        if vol_rank is not None:
            vr_series = pd.to_numeric(vol_rank, errors="coerce").reindex(ix).fillna(0.5).astype(float)
        else:
            vr_series = pd.Series(0.5, index=ix, dtype=float)

        logger.debug(
            "SignalEngine: features built: shape=%s, dates=%s→%s, cols=%s",
            features.shape,
            features.index.min(),
            features.index.max(),
            list(features.columns),
        )
        if features.isna().all().any():
            bad_cols = [c for c in features.columns if features[c].isna().all()]
            # Suppress warning for expected/benign empty columns like delisted_date or trailing missing
            if bad_cols and bad_cols != ["delisted_date"]:
                if not self._warned_all_nan_cols:
                    logger.warning("SignalEngine: all-NaN feature columns detected: %s", bad_cols)
                    self._warned_all_nan_cols = True
                else:
                    logger.debug("SignalEngine: all-NaN feature columns detected: %s", bad_cols)

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
        rolling_vol_60 = daily_ret.rolling(60).std()
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
                # Institutional Loop-Safety: Do NOT fetch macro data inside the loop.
                # If spy_df/vix_df weren't passed in, macro features stay zero.
                ix = features.index

                # 1) SPY Macro: realised vol spike
                if spy_df is not None and not spy_df.empty and "Close" in spy_df.columns:
                    spy_close = pd.to_numeric(spy_df["Close"], errors="coerce").dropna().sort_index()
                    spy_ret = spy_close.pct_change()
                    vol5 = spy_ret.rolling(5).std(ddof=0) * np.sqrt(252.0)
                    vol60 = spy_ret.rolling(60).std(ddof=0) * np.sqrt(252.0)
                    vol_spike = (vol5 / vol60).replace([np.inf, -np.inf], np.nan).shift(1)
                    vol_spike_series = vol_spike.reindex(features.index).fillna(0.0).astype(float)
                elif not getattr(self, "_spy_missing_warned", False):
                    logger.warning("SignalEngine: spy_df not provided; vol_spike features disabled for this run.")
                    self._spy_missing_warned = True

                # 2) VIX Macro
                if vix_df is not None and not vix_df.empty and "Close" in vix_df.columns:
                    vix_close = pd.to_numeric(vix_df["Close"], errors="coerce").dropna().sort_index()
                    vix_mean_252 = vix_close.rolling(252).mean()
                    vix_std_252 = vix_close.rolling(252).std(ddof=0).replace(0, np.nan)
                    vix_z = ((vix_close - vix_mean_252) / vix_std_252).shift(1)
                    vix_zscore_series = vix_z.reindex(features.index).fillna(0.0).astype(float)

                    # Thermal VIX (VIX/VIX3M)
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
                    elif not getattr(self, "_vix3m_missing_warned", False):
                        logger.warning("SignalEngine: vix3m_df not provided; VIX term structure features disabled.")
                        self._vix3m_missing_warned = True
                elif not getattr(self, "_vix_missing_warned", False):
                    logger.warning("SignalEngine: vix_df not provided; VIX-based features disabled for this run.")
                    self._vix_missing_warned = True
            except Exception as e:
                logger.debug("SignalEngine: Macro data processing failed; falling back to zeros. (%s)", e)
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
                # 3) CAPM Beta
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
                    + getattr(lw, "w_vol_rank", 0) * vr_series.loc[mask].fillna(0.5)
                    + _learned_rel_vol_coef(lw) * relative_volume.loc[mask].fillna(1.0)
                    + _learned_volume_zscore_coef(lw) * volume_zscore.loc[mask].fillna(0.0)
                    + getattr(lw, "w_corr_market", 0) * rolling_corr_market_20.loc[mask].fillna(0.0)
                    + getattr(lw, "w_vix_zscore", 0) * vix_zscore_series.loc[mask].fillna(0.0)
                    + getattr(lw, "w_vol_spike", 0) * vol_spike_series.loc[mask].fillna(0.0)
                    + _learned_vix_term_coef(lw) * vix_term_zscore_series.loc[mask].fillna(0.0)
                    + getattr(lw, "w_rsi_zscore", 0) * mr_feat["rsi_zscore"].loc[mask].fillna(0.0)
                    + getattr(lw, "w_bb_position", 0) * mr_feat["bb_position"].loc[mask].fillna(0.0)
                    + getattr(lw, "w_dist_high", 0) * mr_feat["dist_high"].loc[mask].fillna(0.0)
                    + getattr(lw, "w_dist_low", 0) * mr_feat["dist_low"].loc[mask].fillna(0.0)
                    + getattr(lw, "w_overnight_gap", 0) * mr_feat["overnight_gap"].loc[mask].fillna(0.0)
                    + getattr(lw, "w_intraday_rev", 0) * mr_feat["intraday_rev"].loc[mask].fillna(0.0)
                    + getattr(lw, "w_sector_relative_20d", 0) * sr20.loc[mask].fillna(0.0)
                    + getattr(lw, "w_sector_relative_60d", 0) * sr60.loc[mask].fillna(0.0)
                    + getattr(lw, "w_capm_beta", 0) * capm_beta_series.loc[mask].fillna(0.0)
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
                    + getattr(default_lw, "w_vol_rank", 0) * vr_series[still_missing].fillna(0.5)
                    + _learned_rel_vol_coef(default_lw) * relative_volume[still_missing].fillna(1.0)
                    + _learned_volume_zscore_coef(default_lw) * volume_zscore[still_missing].fillna(0.0)
                    + getattr(default_lw, "w_corr_market", 0) * rolling_corr_market_20[still_missing].fillna(0.0)
                    + getattr(default_lw, "w_vix_zscore", 0) * vix_zscore_series[still_missing].fillna(0.0)
                    + getattr(default_lw, "w_vol_spike", 0) * vol_spike_series[still_missing].fillna(0.0)
                    + _learned_vix_term_coef(default_lw) * vix_term_zscore_series[still_missing].fillna(0.0)
                    + getattr(default_lw, "w_rsi_zscore", 0) * mr_feat["rsi_zscore"][still_missing].fillna(0.0)
                    + getattr(default_lw, "w_bb_position", 0) * mr_feat["bb_position"][still_missing].fillna(0.0)
                    + getattr(default_lw, "w_dist_high", 0) * mr_feat["dist_high"][still_missing].fillna(0.0)
                    + getattr(default_lw, "w_dist_low", 0) * mr_feat["dist_low"][still_missing].fillna(0.0)
                    + getattr(default_lw, "w_overnight_gap", 0) * mr_feat["overnight_gap"][still_missing].fillna(0.0)
                    + getattr(default_lw, "w_intraday_rev", 0) * mr_feat["intraday_rev"][still_missing].fillna(0.0)
                    + getattr(default_lw, "w_sector_relative_20d", 0) * sr20[still_missing].fillna(0.0)
                    + getattr(default_lw, "w_sector_relative_60d", 0) * sr60[still_missing].fillna(0.0)
                    + getattr(default_lw, "w_capm_beta", 0) * capm_beta_series[still_missing].fillna(0.0)
                )
                adjusted.loc[still_missing] = raw_def * getattr(default_lw, "score_scale", 1.0)
            adjusted = adjusted.fillna(0)
            logger.debug(
                "SignalEngine: ticker=%s regime=%s weights_source=%s",
                "N/A",
                regime_label_for_log,
                "regime_specific",
            )
        elif (
            hasattr(self, "config")
            and self.config is not None
            and str(getattr(self.config, "signal_mode", "price")).lower() in {"ensemble", "ml"}
        ):
            mode = str(getattr(self.config, "signal_mode", "price")).lower()
            if mode == "ensemble":
                ens_cfg = {
                    "models": getattr(self.config, "ensemble_models", []) or [],
                    "normalize": bool(getattr(self.config, "ensemble_normalize", True)),
                    "clip": bool(getattr(self.config, "ensemble_clip", False)),
                }
            else:
                # ML Mode: Dual Ensemble (Long + Short specialised models)
                ens_cfg = {
                    "models": [
                        {
                            "path": str(getattr(self.config, "ml_long_model_path", "") or "").strip(),
                            "weight": float(getattr(self.config, "ml_long_weight", 0.5)),
                            "type": "regressor",   # best_long_model.pkl is a VotingRegressor ensemble
                        },
                        {
                            "path": str(getattr(self.config, "ml_short_model_path", "") or "").strip(),
                            "weight": float(getattr(self.config, "ml_short_weight", 0.5)),
                            "type": "short_regressor",   # regressor on forward_return; lower score = stronger short
                        }
                    ],
                    "normalize": False,
                    "standardize": bool(getattr(self.config, "ml_standardize", False)),
                    "clip": bool(getattr(self.config, "ml_clip", False)),
                }
            if self._ensemble_models_cache is None:
                self._ensemble_models_cache = load_ensemble_models(ens_cfg)

            # Pillar 29: Institutional Re-sequencing — Compute CAPM before ML model call
            signal_df_tmp = pd.DataFrame(index=features.index)
            try:
                from features.capm_features import compute_capm_features
                if spy_df is not None and not spy_df.empty and "Close" in spy_df.columns:
                    spy_ret = spy_df["Close"].reindex(stock_data.index).ffill().pct_change()
                    stock_ret = stock_data["Close"].pct_change()
                    capm_df = compute_capm_features(stock_ret, spy_ret)
                    for col in ("capm_alpha", "capm_beta", "capm_residual_vol"):
                        if col in capm_df.columns:
                            signal_df_tmp[col] = capm_df[col].reindex(features.index).ffill().bfill()
                else:
                    for col in ("capm_alpha", "capm_beta", "capm_residual_vol"):
                        signal_df_tmp[col] = 0.0 if col != "capm_beta" else 1.0
            except Exception:
                for col in ("capm_alpha", "capm_beta", "capm_residual_vol"):
                    signal_df_tmp[col] = 0.0 if col != "capm_beta" else 1.0
            if panel_features is not None:
                # Pillar 29: Override isolation features with Joint-Panel Normalised features
                pf = panel_features.reindex(features.index).ffill().fillna(0.0)
                ret_5d = pf.get("ret_5d", ret_5d)
                ret_10d = pf.get("ret_10d", ret_10d)
                rolling_vol_20 = pf.get("rolling_vol_20", rolling_vol_20)
                rolling_vol_60 = pf.get("rolling_vol_60", daily_ret.rolling(60).std().fillna(0.0))
                volume_zscore = pf.get("volume_zscore", volume_zscore)
                vix_zscore_series = pf.get("vix_zscore", vix_zscore_series)
                vol_spike_series = pf.get("vol_spike", vol_spike_series)
                # A1 fix: override time-series percentile with cross-sectional percentile
                # (training used CS rank; inference was using per-ticker TS rolling rank)
                if "cs_momentum_percentile" in pf.columns:
                    cs_momentum_series = pf["cs_momentum_percentile"]

            ret_20d = close.pct_change(20)
            ret_60d = close.pct_change(60)
            model_features = pd.DataFrame(
                {
                    "f_trend": f_trend,
                    "ret_5d": ret_5d.fillna(0.0),
                    "ret_10d": ret_10d.fillna(0.0),
                    "ret_20d": ret_20d.fillna(0.0),
                    "ret_60d": ret_60d.fillna(0.0),
                    "cs_momentum_percentile": cs_momentum_series.fillna(0.5),
                    "momentum_3m": pd.to_numeric(features.get("momentum_3m", 0.0), errors="coerce").fillna(0.0),
                    "momentum_6m": momentum_6m.fillna(0.0),
                    "ma_crossover": ma_crossover.fillna(0.0),
                    "rolling_vol_5": rolling_vol_5.fillna(0.0),
                    "rolling_vol_10": rolling_vol_10.fillna(0.0),
                    "rolling_vol_20": rolling_vol_20.fillna(0.0),
                    "rolling_vol_60": rolling_vol_60.fillna(0.0),
                    "vol_of_vol_20": vol_of_vol_20.fillna(0.0),
                    "jump_indicator": jump_indicator.fillna(0.0),
                    "vol_rank": vr_series.fillna(0.5),
                    "relative_volume": relative_volume.fillna(1.0),
                    "volume_zscore": volume_zscore.fillna(0.0),
                    "rolling_corr_market_20": rolling_corr_market_20.fillna(0.0),
                    "capm_beta": signal_df_tmp["capm_beta"].fillna(0.0),
                    "vix_zscore": vix_zscore_series.fillna(0.0),
                    "vol_spike": vol_spike_series.fillna(0.0),
                    "vix_term_zscore": vix_term_zscore_series.fillna(0.0),
                    "rsi_zscore": mr_feat["rsi_zscore"].fillna(0.0),
                    "bb_position": mr_feat["bb_position"].fillna(0.0),
                    "dist_high": mr_feat["dist_high"].fillna(0.0),
                    "dist_low": mr_feat["dist_low"].fillna(0.0),
                    "overnight_gap": mr_feat["overnight_gap"].fillna(0.0),
                    "intraday_rev": mr_feat["intraday_rev"].fillna(0.0),
                    "sector_relative_20d": sr20.fillna(0.0),
                    "sector_relative_60d": sr60.fillna(0.0),
                    "rsi_14": mr_feat["rsi_14_raw"].shift(1).fillna(50.0),
                    "ret_1d": daily_ret.shift(1).fillna(0.0),
                    "vol_ratio_5_20": (
                        pd.to_numeric(vol_struct["vol_5d"], errors="coerce")
                        / pd.to_numeric(vol_struct["vol_20d"], errors="coerce").replace(0, np.nan)
                    ).replace([np.inf, -np.inf], np.nan).shift(1).fillna(1.0),
                    # NEW SHORT SYNC FEATURES (RAW SCALE)
                    "dist_from_52w_high": (
                        (close / close.rolling(252).max() - 1.0)
                    ).fillna(0.0),
                    "rsi_overbought": (
                        mr_feat["rsi_14_raw"].shift(1) > 70.0
                    ).astype(float).fillna(0.0),
                    "vol_expansion": (
                        pd.to_numeric(vol_struct["vol_5d"], errors="coerce") 
                        / pd.to_numeric(vol_struct["vol_20d"], errors="coerce").replace(0, np.nan)
                    ).fillna(1.0).shift(1).fillna(1.0),
                    "momentum_acceleration": (
                        ret_5d - ret_10d
                    ).fillna(0.0),
                    "down_up_vol_ratio": (
                        (stock_data["Volume"].where(daily_ret.shift(1) < 0, 0).rolling(20).sum() 
                         / stock_data["Volume"].where(daily_ret.shift(1) > 0, 0).rolling(20).sum().replace(0, np.nan))
                        .fillna(1.0)
                    ).fillna(1.0),
                    "capm_residual_vol": signal_df_tmp["capm_residual_vol"].fillna(0.0),
                    "capm_alpha": signal_df_tmp["capm_alpha"].fillna(0.0),
                    # All-weather features (in training feature_subset but missing from inference)
                    "short_term_reversal": (-ret_5d.shift(1)).clip(-0.5, 0.5).fillna(0.0),
                    "nearness_52w_low": (
                        1.0 / (1.0 + ((close / close.rolling(252, min_periods=60).min().replace(0, np.nan).clip(lower=1e-6)) - 1.0).clip(0, 10))
                    ).fillna(0.5),
                    "nearness_52w_high": (
                        (close / close.rolling(252, min_periods=100).max().replace(0, np.nan)).clip(0.0, 1.0)
                    ).fillna(0.5),
                    "low_vol_score": (
                        1.0 - daily_ret.rolling(20, min_periods=10).std().rolling(252, min_periods=60).rank(pct=True)
                    ).fillna(0.5),
                    "quality_score": (
                        (daily_ret.rolling(60, min_periods=20).mean()
                         / daily_ret.rolling(60, min_periods=20).std().replace(0, np.nan).fillna(1e-6)
                         * np.sqrt(252)).clip(-5.0, 5.0)
                    ).fillna(0.0),
                    "rel_ret_5d": (
                        ret_5d - (
                            spy_df["Close"].pct_change(5).shift(1).reindex(features.index).fillna(0.0)
                            if spy_df is not None and not spy_df.empty and "Close" in spy_df.columns
                            else 0.0
                        )
                    ).fillna(0.0),
                    "dollar_volume_20d": (
                        (stock_data["Close"] * stock_data["Volume"]).rolling(20).mean().shift(1)
                    ).fillna(0.0),
                    # Fundamental features: Piotroski F-score + accruals (point-in-time, 45d lag)
                    **{
                        col: self._get_fundamental_features(ticker, features.index)[col]
                        for col in ["f_score", "accruals_ratio", "roa", "delta_roa", "delta_leverage"]
                    },
                },
                index=features.index,
            )
            if not self._ensemble_models_cache:
                logger.warning("SignalEngine: no %s models loaded; falling back to learned/price mode.", mode)
                adjusted = f_trend * self.weights.get("trend", 1.0)
            else:
                # 1) Calculate Long Score and Short Score independently to prevent dilution
                long_model = next(
                    (m for m in self._ensemble_models_cache if m.model_type in ("classifier", "regressor")), None
                )
                short_model = next((m for m in self._ensemble_models_cache if m.model_type in ("short_classifier", "short_regressor")), None)
                
                long_score = pd.Series(0.0, index=features.index)
                short_score = pd.Series(0.0, index=features.index)
                short_score_raw = pd.Series(0.0, index=features.index)  # ungated — used for CS short ranking

                from utils.ensemble_scoring import _predict_model, LoadedEnsembleModel

                if long_model:
                    long_score = _predict_model(long_model, model_features, clip=bool(ens_cfg.get("clip", False)))

                if short_model:
                    raw_short = _predict_model(short_model, model_features, clip=bool(ens_cfg.get("clip", False)))
                    # Store ungated raw short score for cross-sectional short-leg selection
                    short_score_raw = raw_short.copy()

                    # Regime Gating: only allow short model to contribute to combined adjusted_score
                    # in certain regimes (keeps long-side clean in Bull regime)
                    if self.regime_series is not None:
                        panel_regimes = self.regime_series.reindex(features.index).astype(str).map(_normalise_regime_label).fillna("Sideways")
                    elif "trend_regime" in features.columns:
                        panel_regimes = features["trend_regime"].astype(str).map(_normalise_regime_label).fillna("Sideways")
                    else:
                        panel_regimes = pd.Series("Sideways", index=features.index)

                    allowed = getattr(self.config, "ml_short_allowed_regimes", ["Bear", "Crisis", "Sideways"])
                    gate_mask = panel_regimes.isin(allowed)
                    short_score = raw_short.where(gate_mask, 0.0)
                
                # Composite: Additive with multipliers to preserve signal strength
                w_long = float(getattr(self.config, "ml_long_weight", 1.0))
                w_short = float(getattr(self.config, "ml_short_weight", 1.0))
                adjusted = (w_long * long_score) + (w_short * short_score)
                
                if adjusted.empty or adjusted.isna().all():
                    logger.warning("SignalEngine: %s combined predictions empty; falling back to price trend.", mode)
                    adjusted = f_trend * self.weights.get("trend", 1.0)
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
                + getattr(lw, "w_vol_rank", 0) * vr_series.fillna(0.5)
                + _learned_rel_vol_coef(lw) * relative_volume.fillna(1.0)
                + _learned_volume_zscore_coef(lw) * volume_zscore.fillna(0.0)
                + getattr(lw, "w_corr_market", 0) * rolling_corr_market_20.fillna(0.0)
                + getattr(lw, "w_vix_zscore", 0) * vix_zscore_series.fillna(0.0)
                + getattr(lw, "w_vol_spike", 0) * vol_spike_series.fillna(0.0)
                + _learned_vix_term_coef(lw) * vix_term_zscore_series.fillna(0.0)
                + getattr(lw, "w_rsi_zscore", 0) * mr_feat["rsi_zscore"].fillna(0.0)
                + getattr(lw, "w_bb_position", 0) * mr_feat["bb_position"].fillna(0.0)
                + getattr(lw, "w_dist_high", 0) * mr_feat["dist_high"].fillna(0.0)
                + getattr(lw, "w_dist_low", 0) * mr_feat["dist_low"].fillna(0.0)
                + getattr(lw, "w_overnight_gap", 0) * mr_feat["overnight_gap"].fillna(0.0)
                + getattr(lw, "w_intraday_rev", 0) * mr_feat["intraday_rev"].fillna(0.0)
                + getattr(lw, "w_sector_relative_20d", 0) * sr20.fillna(0.0)
                + getattr(lw, "w_sector_relative_60d", 0) * sr60.fillna(0.0)
                + getattr(lw, "w_capm_beta", 0) * capm_beta_series.fillna(0.0)
            )
            scale = getattr(lw, "score_scale", 1.0)
            # score_direction is handled globally at the end of the function to avoid double-inversion
            adjusted = (raw * scale)
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
        # Apply global score direction / scaling from config if present
        if hasattr(self, "config") and self.config is not None:
            direction = float(getattr(self.config, "score_direction", 1.0))
            scale = float(getattr(self.config, "score_scale", 1.0))
            adjusted = (adjusted * scale) * direction

        # short_score_raw is only set in mode==ml; default to zeros in other modes
        if "short_score_raw" not in dir():
            short_score_raw = pd.Series(0.0, index=features.index)

        signal_df = pd.DataFrame(
            {
                "trend_score": trend_scores,
                "confidence": rolling_conf,
                "adjusted_score": adjusted,
                "short_score_raw": short_score_raw,
                "sector_relative_20d": sr20,
                "sector_relative_60d": sr60,
                "rel_ret_5d": model_features["rel_ret_5d"] if "rel_ret_5d" in model_features.columns else 0.0,
                "avg_dollar_volume_20d": model_features["dollar_volume_20d"] if "dollar_volume_20d" in model_features.columns else 0.0,
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
        if hasattr(self, "config") and self.config is not None:
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

        # Pillar 28: Institutional Hysteresis (High-Bar for Entry, Low-Bar for Exit)
        # Require 'bull_thresh' to enter, but only exit if we drop below 'exit_thresh'.
        exit_mult = 0.6
        if hasattr(self, "config") and self.config:
            exit_mult = float(getattr(self.config, "hysteresis_exit_multiplier", 0.6))
        bull_exit = bull_thresh * exit_mult
        bear_exit = bear_thresh * exit_mult if bear_thresh < 0 else bear_thresh * (1.0 + (1.0 - exit_mult))

        signals = []
        last_sig = "Neutral"
        scores = signal_df[score_col].values
        
        for val in scores:
            new_sig = "Neutral"
            if last_sig == "Bullish":
                if val > bull_exit: new_sig = "Bullish"
                elif val < bear_thresh: new_sig = "Bearish"
            elif last_sig == "Bearish":
                if val < bear_exit: new_sig = "Bearish"
                elif val > bull_thresh: new_sig = "Bullish"
            else: # Neutral
                if val > bull_thresh: new_sig = "Bullish"
                elif val < bear_thresh: new_sig = "Bearish"
            
            signals.append(new_sig)
            last_sig = new_sig

        signal_df["signal"] = signals
        if len(scores) > 0:
            logger.info("DIAGNOSTIC: %s raw scores: max=%.5f, mean=%.5f, [threshold=%.4f]", signal_df.index[0], float(np.max(scores)), float(np.mean(scores)), float(bull_thresh))
        signal_df["bull_threshold"] = bull_thresh
        signal_df["bear_threshold"] = bear_thresh
        signal_df["bull_exit_threshold"] = bull_exit
        signal_df["bear_exit_threshold"] = bear_exit

        counts = signal_df["signal"].value_counts(dropna=False).to_dict()
        logger.debug(
            "SignalEngine: hysteresis classification using score_col=%s, bull_thresh=%.4f (exit=%.4f), bear_thresh=%.4f (exit=%.4f), counts=%s",
            score_col, float(bull_thresh), float(bull_exit), float(bear_thresh), float(bear_exit), counts,
        )

        abs_adj = adjusted.clip(-1.0, 1.0).abs()
        confidence_numeric = abs_adj.clip(lower=0.3, upper=0.95)
        signal_df["confidence_numeric"] = confidence_numeric

        # Pillar 29: Assign CAPM features to signal_df after model call
        for col in ("capm_alpha", "capm_beta", "capm_residual_vol"):
            if col in signal_df_tmp.columns:
                signal_df[col] = signal_df_tmp[col]
            else:
                signal_df[col] = 0.0 if col != "capm_beta" else 1.0

        # IPO Warm-up Mask: do not trade during the first HISTORY_BUFFER_DAYS of data.
        # This ensures that lagging indicators (like 200-day MAs) have stabilized.
        if len(signal_df) > 0:
            warmup_cutoff = signal_df.index.min() + pd.Timedelta(days=HISTORY_BUFFER_DAYS)
            mask = signal_df.index < warmup_cutoff
            if mask.any():
                signal_df.loc[mask, "signal"] = "Neutral"
                if score_col in signal_df.columns:
                    signal_df.loc[mask, score_col] = 0.0

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
