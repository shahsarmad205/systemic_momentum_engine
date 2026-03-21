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
        vix_term_zscore: rolling 252d z-score of lagged VIX/VIX3M ratio (shifted; scale-aligned with other features)
    Mean-reversion (panel):
        Raw per-ticker series are built in workers (mr_*_raw), then in
        build_feature_matrix: cross_sectional_zscore (dates × tickers) then
        shift(1) → rsi_zscore, bb_position, dist_high, dist_low, overnight_gap,
        intraday_rev. Live SignalEngine uses per-ticker rolling z-score as proxy.
    Sector-relative momentum (panel):
        ret_20d/ret_60d minus sector median (same date), then shift(1), then
        cross_sectional z-score → sector_relative_20d, sector_relative_60d.
        SECTOR_MAP / get_sector from utils.sectors. Backtests inject panel values
        via inject_sector_relative_panel_into_signals.
    Cross-sectional ranking (panel):
        ret_5d, ret_10d, rolling_vol_20, rolling_vol_60, volume_zscore, vix_zscore,
        vol_spike → per-date z-score across tickers (population std, ddof=0).
        is_high_vol_regime is not cross-sectionally normalized; vix_term_zscore is time-series z-scored (not CS).
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
from agents.weight_learning_agent.feature_flags import feature_columns_to_zero_for_ablation
from execution.cost_model import TransactionCostModel
from main import CONFIDENCE_MULTIPLIER, compute_rolling_trend_scores
from utils.sectors import SECTOR_MAP, get_sector
from utils.market_data import get_ohlcv

HISTORY_BUFFER_DAYS = 400
MARKET_TICKER = "SPY"  # for rolling correlation feature

# Re-export: canonical ticker→sector lives in utils.sectors (covers full learning universe).
# Use get_sector(ticker) in workers; SECTOR_MAP is the static mapping dict.

# Mean-reversion raw columns (per ticker) → cross-sectional z-score + shift(1) in build_feature_matrix
_MR_RAW_TO_OUT = {
    "mr_rsi_raw": "rsi_zscore",
    "mr_bb_raw": "bb_position",
    "mr_dist_high_raw": "dist_high",
    "mr_dist_low_raw": "dist_low",
    "mr_overnight_raw": "overnight_gap",
    "mr_intraday_raw": "intraday_rev",
}

# Panel CS z-score applied in-place; skip redundant ``{col}_cs_z`` suffix pass.
_CS_Z_PANEL_INPLACE_COLS = frozenset(
    {
        "ret_5d",
        "ret_10d",
        "rolling_vol_20",
        "rolling_vol_60",
        "volume_zscore",
        "vix_zscore",
        "vol_spike",
    }
)


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional z-score: tickers as columns, dates as index.
    (x - row_mean) / row_std; std=0 → NaN.
    """
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1).replace(0, np.nan)
    return df.sub(row_mean, axis=0).div(row_std, axis=0)


def cross_sectional_zscore_ddof0(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same as cross_sectional z-score but population std (ddof=0), matching
    ``groupby('date')[col].transform(lambda x: (x - x.mean()) / x.std(ddof=0))``.
    """
    row_mean = df.mean(axis=1, skipna=True)
    row_std = df.std(axis=1, ddof=0, skipna=True).replace(0, np.nan)
    return df.sub(row_mean, axis=0).div(row_std, axis=0)


def _apply_cross_sectional_zscore_columns(
    result: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Replace each column with cross-sectional z-scores across tickers on each date.
    Missing pivot cells become NaN then 0 after merge.
    """
    out = result
    for col in columns:
        if col not in out.columns:
            continue
        pivot = out.pivot(index="date", columns="ticker", values=col)
        z = cross_sectional_zscore_ddof0(pivot)
        z_long = z.stack(future_stack=True).reset_index()
        z_long.columns = ["date", "ticker", col]
        out = out.drop(columns=[col], errors="ignore")
        out = out.merge(z_long, on=["date", "ticker"], how="left")
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def compute_sector_relative_shifted_cs_long(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sector-relative momentum: for each (date, ticker),
      raw = ret_Nd - median(ret_Nd | same date & sector),
    then pivot → shift(1) along time → cross-sectional z-score across tickers.

    Input columns: date, ticker, ret_20d, ret_60d, sector.
    Output adds: sector_relative_20d, sector_relative_60d (CS z-scored, lagged 1 day).
    """
    need = {"date", "ticker", "ret_20d", "ret_60d", "sector"}
    if not need.issubset(long_df.columns):
        out = long_df.copy()
        out["sector_relative_20d"] = 0.0
        out["sector_relative_60d"] = 0.0
        return out
    out = long_df.copy()
    out["date"] = pd.to_datetime(out["date"])
    m20 = out.groupby(["date", "sector"])["ret_20d"].transform("median")
    m60 = out.groupby(["date", "sector"])["ret_60d"].transform("median")
    out["_sr20_raw"] = out["ret_20d"] - m20
    out["_sr60_raw"] = out["ret_60d"] - m60
    for raw_col, out_col in [("_sr20_raw", "sector_relative_20d"), ("_sr60_raw", "sector_relative_60d")]:
        pivot = out.pivot(index="date", columns="ticker", values=raw_col)
        z = cross_sectional_zscore(pivot).shift(1)
        z_long = z.stack().reset_index()
        z_long.columns = ["date", "ticker", out_col]
        out = out.drop(columns=[out_col], errors="ignore")
        out = out.merge(z_long, on=["date", "ticker"], how="left")
        out[out_col] = pd.to_numeric(out[out_col], errors="coerce").fillna(0.0)
    out = out.drop(columns=["_sr20_raw", "_sr60_raw"], errors="ignore")
    return out


def sector_relative_features_by_ticker(
    price_data: dict[str, pd.DataFrame],
    *,
    exclude_tickers: frozenset[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build sector_relative_20d / sector_relative_60d (CS z-scored, lagged 1d) per ticker,
    using the same panel logic as training. Benchmarks (e.g. SPY) should be excluded from
    the cross-section via exclude_tickers.

    Returns:
        ticker -> DataFrame indexed by date with columns sector_relative_20d, sector_relative_60d.
    """
    ex = frozenset(exclude_tickers or ())
    tickers = [tk for tk in price_data.keys() if tk not in ex]
    if not tickers:
        return {}
    rows: list[dict] = []
    for tk in tickers:
        df = price_data[tk]
        price_col = "AdjClose" if "AdjClose" in df.columns else "Close"
        if price_col not in df.columns:
            continue
        close = pd.to_numeric(df[price_col], errors="coerce")
        r20 = close.pct_change(20)
        r60 = close.pct_change(60)
        sector = get_sector(tk)
        for dt in df.index:
            rows.append(
                {
                    "date": pd.Timestamp(dt),
                    "ticker": tk,
                    "ret_20d": float(r20.loc[dt]) if pd.notna(r20.loc[dt]) else np.nan,
                    "ret_60d": float(r60.loc[dt]) if pd.notna(r60.loc[dt]) else np.nan,
                    "sector": sector,
                }
            )
    if not rows:
        return {}
    long = pd.DataFrame(rows)
    long = compute_sector_relative_shifted_cs_long(long)
    out: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        sub = long.loc[long["ticker"] == tk, ["date", "sector_relative_20d", "sector_relative_60d"]]
        if sub.empty:
            continue
        out[tk] = sub.set_index("date").sort_index()
    return out


def inject_sector_relative_panel_into_signals(
    price_data: dict[str, pd.DataFrame],
    signal_data: dict[str, pd.DataFrame],
) -> None:
    """
    After per-ticker signals exist, add sector_relative_20d / 60d using the same
    panel logic as training (mutates signal_data in place).
    """
    sr_map = sector_relative_features_by_ticker(
        price_data,
        exclude_tickers=frozenset({"SPY"}),
    )
    for tk, sig in signal_data.items():
        if tk not in sr_map:
            continue
        sub = sr_map[tk]
        sig_idx = pd.to_datetime(sig.index)
        s20 = sub["sector_relative_20d"].reindex(sig_idx)
        s60 = sub["sector_relative_60d"].reindex(sig_idx)
        sig["sector_relative_20d"] = s20.fillna(0.0).to_numpy(dtype=float)
        sig["sector_relative_60d"] = s60.fillna(0.0).to_numpy(dtype=float)


def _attach_mr_cross_sectional_zscore_shifted(result: pd.DataFrame) -> pd.DataFrame:
    """Attach rsi_zscore, bb_position, … from raw MR columns; CS z-score then shift(1)."""
    for raw_col, out_col in _MR_RAW_TO_OUT.items():
        if raw_col not in result.columns:
            result[out_col] = 0.0
            continue
        pivot = result.pivot(index="date", columns="ticker", values=raw_col)
        z = cross_sectional_zscore(pivot).shift(1)
        z_long = z.stack().reset_index()
        z_long.columns = ["date", "ticker", out_col]
        result = result.drop(columns=[out_col], errors="ignore")
        result = result.merge(z_long, on=["date", "ticker"], how="left")
        result[out_col] = result[out_col].fillna(0.0)
    for raw_col in _MR_RAW_TO_OUT:
        if raw_col in result.columns:
            result = result.drop(columns=[raw_col])
    return result


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
        # Mean-reversion raw inputs (CS z-score + shift(1) applied in build_feature_matrix)
        # ------------------------------------------------------------------
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        mr_rsi_raw = 100 - (100 / (1 + rs))

        rolling_max_20 = close.rolling(20, min_periods=10).max()
        rolling_min_20 = close.rolling(20, min_periods=10).min()
        mr_dist_high_raw = (close - rolling_max_20) / rolling_max_20.replace(0, np.nan)
        mr_dist_low_raw = (close - rolling_min_20) / rolling_min_20.replace(0, np.nan)

        prev_close = close.shift(1)
        mr_overnight_raw = (open_px - prev_close) / prev_close.replace(0, np.nan)
        mr_intraday_raw = (close - open_px) / open_px.replace(0, np.nan)

        bb_mid = close.rolling(20, min_periods=20).mean()
        bb_std = close.rolling(20, min_periods=20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = (bb_upper - bb_lower).replace(0, np.nan)
        mr_bb_raw = (close - bb_lower) / bb_width

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
                # Mean-reversion raw (panel CS z + shift(1) applied in build_feature_matrix)
                "mr_rsi_raw": mr_rsi_raw,
                "mr_bb_raw": mr_bb_raw,
                "mr_dist_high_raw": mr_dist_high_raw,
                "mr_dist_low_raw": mr_dist_low_raw,
                "mr_overnight_raw": mr_overnight_raw,
                "mr_intraday_raw": mr_intraday_raw,
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
    vix_term_zscore_series = pd.Series(dtype=float)
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
                # Lagged spot/3M ratio (no same-day look-ahead), then TS z-score vs 252d history, lagged 1d.
                vix_ratio_lag = (vix_close / vix3m_aligned.replace(0.0, np.nan)).replace(
                    [np.inf, -np.inf], np.nan
                ).shift(1)
                vt_m = vix_ratio_lag.rolling(252, min_periods=60).mean()
                vt_s = vix_ratio_lag.rolling(252, min_periods=60).std(ddof=0).replace(0, np.nan)
                vix_term_zscore_series = (
                    ((vix_ratio_lag - vt_m) / vt_s).replace([np.inf, -np.inf], np.nan).shift(1)
                )
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

    # Rank tickers vs each other on the same date (raw time-series features first).
    result = _apply_cross_sectional_zscore_columns(
        result,
        ["ret_5d", "ret_10d", "rolling_vol_20", "rolling_vol_60", "volume_zscore"],
    )
    if "volume_surprise" in result.columns:
        result["volume_surprise"] = result["volume_zscore"]

    # Mean-reversion: cross-sectional z-score per date, then shift(1) (no lookahead).
    result = _attach_mr_cross_sectional_zscore_shifted(result)

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
        if not vix_term_zscore_series.empty:
            result["vix_term_zscore"] = (
                result["date"].map(vix_term_zscore_series.to_dict()).astype(float).fillna(0.0)
            )
        else:
            result["vix_term_zscore"] = 0.0

        # Macro levels are identical across tickers per date; CS z collapses to ~0 — kept for spec / symmetry.
        result = _apply_cross_sectional_zscore_columns(result, ["vix_zscore", "vol_spike"])

    # Sector-relative momentum: (ret - sector median) → shift(1) → CS z-score (panel).
    if {"ret_20d", "ret_60d", "sector", "ticker", "date"}.issubset(result.columns):
        for c in (
            "sector_relative_mom_20d",
            "sector_relative_strength",
            "sector_relative_mom_60d",
            "sector_relative_20d",
            "sector_relative_60d",
        ):
            result = result.drop(columns=[c], errors="ignore")
        sector_long = result[["date", "ticker", "ret_20d", "ret_60d", "sector"]].copy()
        sector_fe = compute_sector_relative_shifted_cs_long(sector_long)
        result = result.merge(
            sector_fe[["date", "ticker", "sector_relative_20d", "sector_relative_60d"]],
            on=["date", "ticker"],
            how="left",
        )
        result["sector_relative_20d"] = result["sector_relative_20d"].fillna(0.0)
        result["sector_relative_60d"] = result["sector_relative_60d"].fillna(0.0)
    else:
        result["sector_relative_20d"] = 0.0
        result["sector_relative_60d"] = 0.0

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
    # MR + sector-relative + panel-inplace columns are already CS-treated; do not duplicate.
    _precomputed_cs_z = (
        set(_MR_RAW_TO_OUT.values())
        | {"sector_relative_20d", "sector_relative_60d"}
        | set(_CS_Z_PANEL_INPLACE_COLS)
        | {"vix_term_zscore"}  # already TS z-scored; identical across tickers per date
    )
    num_cols = result.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if col in _precomputed_cs_z:
            continue
        cs = result.groupby("date")[col].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) not in (0, 0.0) else 1.0)
        )
        result[f"{col}_cs_z"] = cs.fillna(0.0)

    # Phase 1 / Phase 2 ablation: zero disabled COMPOUND columns (TSE_ABLATION_STEP env).
    _zero_cols = feature_columns_to_zero_for_ablation()
    for col in _zero_cols:
        if col in result.columns:
            result[col] = 0.0

    return result
