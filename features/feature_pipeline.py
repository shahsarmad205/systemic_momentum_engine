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


def flag_feature_anomalies(
    df: pd.DataFrame,
    zscore_threshold: float = 5.0,
    repeated_run_threshold: int = 10,
    window: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """
    BIS ML validation step 2 — anomaly detection for financial time series.

    Erdem & Park (2021) BIS Working Paper 964, §3.2:
      "Before feeding features to any ML model, detect two classes of anomalies:
       (a) extreme outliers — rolling z-score exceedances beyond ±5σ indicate
           data errors, corporate events, or fat-tail shocks that will distort
           model training if uncorrected.
       (b) repeated-value runs — N consecutive identical values signal a stale
           feed, halted trading, or a data vendor fill that will cause spurious
           feature stability and leak-free look-ahead."

    This function adds a boolean ``anomaly_flag`` column: True when either
    condition is detected on ANY numeric feature for that row.  Callers should
    winsorise or drop flagged rows before passing data to the weight learner.

    Parameters
    ----------
    df : feature matrix (output of build_feature_matrix)
    zscore_threshold : flag row if |rolling_z| > threshold on any feature
    repeated_run_threshold : flag row if same value repeats ≥ N consecutive times
    window / min_periods : rolling window for z-score computation

    Returns
    -------
    df with ``anomaly_flag`` (bool) and ``anomaly_reasons`` (str) columns added.
    """
    _log = logging.getLogger(__name__)
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in ("anomaly_flag",)
    ]

    n = len(df)
    outlier_flag = np.zeros(n, dtype=bool)
    stale_flag = np.zeros(n, dtype=bool)
    reason_parts: list[list[str]] = [[] for _ in range(n)]

    for col in numeric_cols:
        series = df[col].astype(float)

        # (a) Rolling z-score outlier detection
        roll = series.rolling(window=window, min_periods=min_periods)
        mu = roll.mean()
        sigma = roll.std().replace(0.0, np.nan).fillna(1.0).clip(lower=1e-8)
        z = ((series - mu) / sigma).abs()
        mask_out = z > zscore_threshold
        if mask_out.any():
            idxs = np.where(mask_out.to_numpy())[0]
            for idx in idxs:
                outlier_flag[idx] = True
                reason_parts[idx].append(f"z({col})={z.iloc[idx]:.1f}")

        # (b) Repeated-value stale run detection
        # Count how many consecutive rows have the same value
        shifted = series != series.shift(1)
        run_id = shifted.cumsum()
        run_lengths = run_id.map(run_id.value_counts())
        mask_stale = run_lengths >= repeated_run_threshold
        if mask_stale.any():
            idxs = np.where(mask_stale.to_numpy())[0]
            for idx in idxs:
                stale_flag[idx] = True
                reason_parts[idx].append(f"stale({col})")

    combined = outlier_flag | stale_flag
    df = df.copy()
    df["anomaly_flag"] = combined
    df["anomaly_reasons"] = ["| ".join(r) if r else "" for r in reason_parts]

    n_flagged = int(combined.sum())
    if n_flagged > 0:
        pct = 100.0 * n_flagged / max(n, 1)
        _log.warning(
            "BIS anomaly detection: %d/%d rows flagged (%.1f%%) — "
            "consider winsorising before model training.",
            n_flagged, n, pct,
        )

    return df


def calculate_factor_momentum_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-horizon momentum consistency score — stock-level proxy for Factor Momentum.

    Gupta & Kelly (AQR/JPM 2019) "Factor Momentum Everywhere":
      Individual factors exhibit strong serial correlation (AR(1) avg 0.11, positive
      for 59 of 65 factors).  Scaling factor exposure by its own past 1-month return
      (TSFM) earns SR 0.84 standalone, outperforming stock-level UMD (SR 0.56).
      Key: factor persistence is robust across 1m, 12m, and even 5-year lookbacks.

    Stock-level proxy: rather than timing an abstract factor exposure, we check
    whether a stock's price momentum is directionally consistent across multiple
    horizons (1m, 3m, 6m, 12m).  Horizon agreement is the observable analogue of
    "the factor showed positive returns over multiple look-back periods", the core
    TSFM signal.

    Formula:
        momentum_consistency_score = mean(
            sign(mom_1m_skip_eom),  # 1-month (end-of-month adjusted)
            sign(mom_3m),           # 3-month
            sign(mom_6m),           # 6-month
            sign(mom_12m_skip1),    # 12-month ex last month
        )

    Range: [-1, +1]
      +1 = all 4 horizons agree → "factor momentum everywhere" → use as confirming signal
       0 = mixed → normal
      -1 = all horizons point down → strong bear signal

    Used in cross_sectional.py as a tertiary tiebreaker for long selection (after
    adjusted_score and information_discreteness).
    """
    required = {"momentum_3m", "momentum_6m", "momentum_12m_skip1"}
    if not required.issubset(df.columns) or len(df) < 252:
        df["momentum_consistency_score"] = 0.0
        return df

    # 1-month momentum: use skip-EOM version if available, else raw pct-change
    if "momentum_1m_skip_eom" in df.columns:
        mom_1m = df["momentum_1m_skip_eom"]
    else:
        close_col = "Close" if "Close" in df.columns else "close"
        if close_col in df.columns:
            mom_1m = df[close_col].pct_change(21)
        else:
            mom_1m = pd.Series(0.0, index=df.index)

    signs = (
        np.sign(mom_1m)
        + np.sign(df["momentum_3m"])
        + np.sign(df["momentum_6m"])
        + np.sign(df["momentum_12m_skip1"])
    ) / 4.0  # normalised to [-1, 1]

    df["momentum_consistency_score"] = signs.clip(-1.0, 1.0).fillna(0.0)
    return df


def calculate_information_discreteness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Information Discreteness (Da, Gurun & Warachka 2014, RFS; FIP hypothesis).

    Da et al. (2014) / Skardhamar & Polykarpou (2024):
      Momentum signal quality filter that separates CONTINUOUS momentum
      (gradual daily appreciation = limited investor attention = persistent)
      from DISCRETE momentum (a few large jumps = crowds notice = mean-reverts).

    Formula (Da et al. eq.1):
        ID = sign(PRET) × [%neg − %pos]

    where:
      PRET  = 12-month cumulative return excluding the most recent month
              (= momentum_12m_skip1, the Bird et al. formation period).
      %neg  = fraction of trading days with negative daily returns in the
              231-day formation window (252 - 21 = skipping last month).
      %pos  = fraction of trading days with positive daily returns in the
              same window.

    Interpretation:
      LOW  ID → CONTINUOUS winner: many small up-days, market didn't notice.
                → Under-reaction anomaly: alpha persists LONGER. PREFER for longs.
      HIGH ID → DISCRETE winner: few big jump days, crowds chased the event.
                → Information already fully incorporated. Momentum weakens fast.
                → AVOID or down-weight for longs.

    Result column:
      information_discreteness : float in approx [-1, 1]
        Lower = more continuous (better long quality)
        Higher = more discrete (weaker, mean-reverts sooner)
    """
    close_col = "Close" if "Close" in df.columns else "close"
    if close_col not in df.columns or len(df) < 252:
        df["information_discreteness"] = 0.0
        return df

    close = df[close_col]
    # Formation period: 231 trading days, skipping last 21 (= 12m ex 1m, Jegadeesh 1990)
    formation = 231

    # Sign of cumulative 12m-ex-1m return
    pret = close.shift(21) / close.shift(252).replace(0, np.nan) - 1
    pret_sign = np.sign(pret)

    # Daily returns for %neg / %pos computation
    daily_ret = close.pct_change(fill_method=None)
    neg_frac = (daily_ret < 0).rolling(formation, min_periods=100).mean()
    pos_frac = (daily_ret > 0).rolling(formation, min_periods=100).mean()

    id_raw = pret_sign * (neg_frac - pos_frac)
    df["information_discreteness"] = id_raw.clip(-1.0, 1.0).fillna(0.0)

    return df


def calculate_short_logic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Short-logic features derived from the ShortLogic research paper corpus.

    Paper 1 — Medhat & Schmeling (2022) "Short-term Momentum" (RFS):
      KEY FINDING: Turnover FLIPS the sign of expected returns at the 1-month horizon.
      - High-turnover LOSERS exhibit short-term MOMENTUM (continuation DOWN) → SHORT
      - Low-turnover LOSERS exhibit short-term REVERSAL (bounce UP) → do NOT short
      - Formula: short_term_momentum_score = momentum_20 × turnover_percentile_rank
        Positive = high-turnover winner → long momentum
        Negative = high-turnover loser  → short momentum (best short candidates)
        Near zero = low-turnover stock  → reversal expected, avoid shorting

      REFINEMENT: Skip last 5 trading days of formation period.
        Standard 1-month return uses close[t-1]/close[t-21]-1.
        Skipping the last ~5 days (end-of-month liquidity window) improves alpha
        from +16.4% to +22.0% annualized (Medhat & Schmeling, Table III).
        momentum_1m_skip_eom = close[t-5] / close[t-26] - 1

    Paper 2 — Dai, Medhat, Rizova & Novy-Marx (2023) "Reversals and Returns to Liquidity":
      KEY FINDING: Standard reversals are contaminated by PEAD and industry momentum.
        REV  = 31 bps/month (t=1.68, barely significant)
        IRR  = 74 bps/month (t=5.40) — remove industry momentum
        IRRX = 108 bps/month (t=9.35) — also remove PEAD noise
      IMPLEMENTATION: industry_relative_reversal = -(momentum_20 - sector_relative_20d)
        This is the clean liquidity-driven reversal signal, stripped of sector effects.
        For longs (reversal bounce): high positive irrx = fell vs sector → buy bounce
        For shorts: do NOT use standard reversal as short signal — it shorts PEAD
          instead use short_term_momentum_score (turnover-conditioned above)

      PERSISTENCE RULE: Volatility → faster reversals; Turnover → persistent reversals.
        High-volatility stocks: reversal completes in 1-2 weeks (RRLP Figure 1 left)
        Low-turnover stocks: reversal persists up to 3 months (RRLP Figure 1 right)
        high_vol_reversal_flag: marks stocks where reversal will be fast (use short HPs)
    """
    close = df["close"].astype(float) if "close" in df.columns else (
        df["Close"].astype(float) if "Close" in df.columns else None
    )
    if close is None:
        return df

    # ------------------------------------------------------------------
    # 1. End-of-month-adjusted 1-month momentum (Medhat & Schmeling, §2.1)
    #    Skip last 5 trading days: close[t-5] / close[t-26] - 1
    #    Avoids end-of-month liquidity window effects that contaminate
    #    the standard t-1/t-21 formation period.
    # ------------------------------------------------------------------
    momentum_1m_raw = (close.shift(5) / close.shift(26).replace(0, np.nan) - 1)
    df["momentum_1m_skip_eom"] = momentum_1m_raw.clip(-1.0, 4.0).fillna(0.0)

    # ------------------------------------------------------------------
    # 2. Short-term momentum score: return × turnover percentile
    #    (Medhat & Schmeling, Table I — double sort construction)
    #    turnover_pct_rank: rolling 63-day percentile of 20d avg share turnover
    #    Rescaled to [-1, 1]: rank=0 → -1 (lowest TO), rank=1 → +1 (highest TO)
    # ------------------------------------------------------------------
    vol = pd.to_numeric(df.get("volume", df.get("Volume", pd.Series(dtype=float))), errors="coerce")
    shrout_proxy = close * vol  # dollar volume as proxy for share turnover
    to_20d = (vol / vol.rolling(252, min_periods=60).mean().replace(0, np.nan)).clip(0.0, 20.0)
    to_pct_rank = to_20d.rolling(63, min_periods=21).rank(pct=True).fillna(0.5)
    to_centered = (to_pct_rank * 2.0 - 1.0)  # rescale to [-1, 1]

    # short_term_momentum_score = last-month-return (skip-EOM) × turnover-rank
    # Negative value = high-TO loser = best short candidate (momentum continues down)
    df["short_term_momentum_score"] = (
        df["momentum_1m_skip_eom"] * to_centered
    ).clip(-2.0, 2.0).fillna(0.0)

    # turnover_percentile_rank stored for use in cross-sectional short gate
    df["turnover_pct_rank"] = to_pct_rank.clip(0.0, 1.0).fillna(0.5)

    # ------------------------------------------------------------------
    # 3. Industry-relative reversal — clean liquidity signal (RRLP §2)
    #    industry_relative_reversal = -(momentum_20 - sector_relative_20d)
    #    Removes sector-level price movements (industry momentum + PEAD) so
    #    the remaining signal reflects pure liquidity-driven price pressure.
    #    High positive value = stock fell vs its sector → bounce expected (LONG)
    #    Do NOT use negative values as short signal (use short_term_momentum_score).
    # ------------------------------------------------------------------
    mom_20 = close.pct_change(20).clip(-1.0, 4.0).fillna(0.0)
    sector_rel_20d = df.get("sector_relative_20d", pd.Series(0.0, index=df.index))
    if not isinstance(sector_rel_20d, pd.Series):
        sector_rel_20d = pd.Series(0.0, index=df.index)
    df["industry_relative_reversal"] = (
        -(mom_20 - sector_rel_20d.reindex(df.index).fillna(0.0))
    ).clip(-2.0, 2.0).fillna(0.0)

    # ------------------------------------------------------------------
    # 4. High-volatility reversal flag (RRLP §3.1, Figure 1 left panel)
    #    High-vol stocks: reversal completes in 1-2 weeks, not 1 month.
    #    Flag to shorten holding period when using reversal signals on these.
    # ------------------------------------------------------------------
    if "volatility_percentile" in df.columns:
        df["high_vol_reversal_flag"] = (df["volatility_percentile"] > 0.67).astype(float)
    else:
        rets = close.pct_change(fill_method=None)
        vol_20 = rets.rolling(20, min_periods=10).std()
        vol_pct = vol_20.rolling(252, min_periods=60).rank(pct=True)
        df["high_vol_reversal_flag"] = (vol_pct > 0.67).astype(float).fillna(0.0)

    return df


def calculate_core_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    close = df["Close"] if "Close" in df.columns else df["close"]
    # Use CRSP total return directly when available (WRDS path); fall back to
    # pct_change for Yahoo/legacy data.  For adj_price-based Close the two are
    # numerically identical on a daily basis, but ret is the authoritative source.
    if "ret" in df.columns and "daily_return" not in df.columns:
        df["daily_return"] = pd.to_numeric(df["ret"], errors="coerce")
    else:
        df["daily_return"] = close.pct_change(fill_method=None)
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

    # 12M-1M momentum (Jegadeesh & Titman 1993 standard formation window).
    # Return from 252 days ago to 21 days ago — skips most recent month to avoid
    # short-term reversal contamination (Jegadeesh 1990, Lehmann 1990).
    # Bird et al. (2015) confirm this is the core TS momentum signal.
    df["momentum_12m_skip1"] = close.shift(21).pct_change(231).clip(-0.95, 10.0).fillna(0.0)

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

def calculate_value_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    All-weather signals: mean-reversion, value proximity, low-vol anomaly, quality.

    These complement pure price momentum and provide signal in Bear/Crisis regimes
    where momentum breaks down:
      - short_term_reversal : contrarian bounce signal (stocks that fell tend to mean-revert)
      - nearness_52w_low    : value proxy — the closer to 52w low the "cheaper" the stock
      - low_vol_score       : low-volatility anomaly (low-vol stocks outperform risk-adjusted)
      - quality_score       : rolling 60d Sharpe ratio proxy (consistent earners beat volatile)
    """
    close = df["Close"] if "Close" in df.columns else df["close"]
    # Prefer pre-computed daily_return (sourced from CRSP ret on WRDS path)
    if "daily_return" in df.columns:
        daily_ret = pd.to_numeric(df["daily_return"], errors="coerce")
    else:
        daily_ret = close.pct_change(fill_method=None)

    # 1. Short-term reversal (1-week contrarian; lagged 1d for look-ahead safety)
    ret_5d = close.pct_change(5).shift(1)
    df["short_term_reversal"] = (-ret_5d).clip(-0.5, 0.5).fillna(0.0)

    # 2. 52-week low proximity (value proxy)
    min_52w = close.rolling(252, min_periods=60).min().replace(0, np.nan)
    dist_low = ((close - min_52w) / min_52w.clip(lower=1e-6)).clip(0, 10)
    df["nearness_52w_low"] = (1.0 / (1.0 + dist_low)).fillna(0.5)  # 1.0 = at low, ~0 = far above

    # 3. Low-volatility score (negated percentile rank; low vol → high score)
    vol_20 = daily_ret.rolling(20, min_periods=10).std()
    vol_pct = vol_20.rolling(252, min_periods=60).rank(pct=True).fillna(0.5)
    df["low_vol_score"] = (1.0 - vol_pct).fillna(0.5)  # 1.0 = lowest vol, 0.0 = highest

    # 4. Quality score: rolling 60d Sharpe ratio (risk-adjusted consistency)
    roll_mean = daily_ret.rolling(60, min_periods=20).mean()
    roll_std = daily_ret.rolling(60, min_periods=20).std().replace(0, np.nan).fillna(1e-6)
    df["quality_score"] = (roll_mean / roll_std * np.sqrt(252)).clip(-5.0, 5.0).fillna(0.0)

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
        base = calculate_value_quality_features(base)
    except Exception as exc:
        warnings.warn(
            f"calculate_value_quality_features failed with {type(exc).__name__}: {exc}",
            UserWarning,
        )

    # Short-logic features: turnover-conditioned short-term momentum, industry-relative
    # reversal, and end-of-month-adjusted formation return.
    # Sources: Medhat & Schmeling (2022) RFS; Dai, Medhat, Rizova & Novy-Marx (2023).
    try:
        base = calculate_short_logic_features(base)
    except Exception as exc:
        warnings.warn(
            f"calculate_short_logic_features failed with {type(exc).__name__}: {exc}",
            UserWarning,
        )

    # Factor Momentum consistency: Gupta & Kelly (AQR/JPM 2019) stock-level proxy.
    # Multi-horizon momentum sign agreement — all horizons pointing same direction
    # signals "factor momentum everywhere"; used as confirming tiebreaker for longs.
    try:
        base = calculate_factor_momentum_consistency(base)
    except Exception as exc:
        warnings.warn(
            f"calculate_factor_momentum_consistency failed with {type(exc).__name__}: {exc}",
            UserWarning,
        )

    # Information Discreteness: Da, Gurun & Warachka (2014) / FIP hypothesis.
    # Continuous momentum (low ID) persists longer; discrete momentum (high ID) mean-reverts.
    # Used downstream in cross_sectional.py to quality-filter long candidates.
    try:
        base = calculate_information_discreteness(base)
    except Exception as exc:
        warnings.warn(
            f"calculate_information_discreteness failed with {type(exc).__name__}: {exc}",
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
                    spy_ret = spy["Close"].pct_change(fill_method=None)
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

    # BIS ML validation — anomaly detection (Erdem & Park 2021, §3.2).
    # Adds anomaly_flag (bool) and anomaly_reasons (str) columns so callers
    # can winsorise or drop rows before passing features to the weight learner.
    try:
        enriched = flag_feature_anomalies(enriched)
    except Exception as exc:
        warnings.warn(
            f"BIS anomaly detection failed with {type(exc).__name__}: {exc}",
            UserWarning,
        )

    return sanitize_dataframe(enriched)

