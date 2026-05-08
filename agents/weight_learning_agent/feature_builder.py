from __future__ import annotations
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
        SECTOR_MAP / get_sector from utils.quant_utils. Backtests inject panel values
        via inject_sector_relative_panel_into_signals.
    Cross-sectional ranking (panel):
        ret_5d, ret_10d, rolling_vol_20, rolling_vol_60, volume_zscore, vix_zscore,
        vol_spike → per-date z-score across tickers (population std, ddof=0).
    Volatility rank (panel, ARE 1.3):
        vol_rank — cross-sectional percentile rank of 20d realised vol (std of daily
        returns) across tickers per date, then shift(1). Matches IC screen in feature_search.
        is_high_vol_regime is not cross-sectionally normalized; vix_term_zscore is time-series z-scored (not CS).
    Target: forward_return, direction
"""


from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
import logging

import numpy as np
import pandas as pd

from features.cross_sectional import (
    apply_cross_sectional_zscore_columns,
    attach_cross_sectional_zscore_suffix_block,
    compute_sector_relative_shifted_cs_long,
    cross_sectional_zscore,
)
from features.feature_pipeline import calculate_core_trend_features as build_features
from agents.volatility_agent.volatility_model import (
    compute_rolling_confidence,
    compute_vol_term_structure,
)
from agents.weight_learning_agent.feature_flags import feature_columns_to_zero_for_ablation
from agents.weight_learning_agent.regime_detection import get_regime_series_for_dates
from execution.cost_model import TransactionCostModel
from backtesting.signals import CONFIDENCE_MULTIPLIER, compute_rolling_trend_scores
from utils.market_data import get_ohlcv
from utils.quant_utils import get_sector
from utils.wrds_data import load_wrds_price_panel, resolve_data_provider

HISTORY_BUFFER_DAYS = 400
MARKET_TICKER = "SPY"  # for rolling correlation feature
logger = logging.getLogger(__name__)


def _timeseries_zscore(s: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std(ddof=0).replace(0, np.nan)
    return ((s - mean) / std).replace([np.inf, -np.inf], np.nan)


def _fetch_fundamental_cols(ticker: str, dates: pd.DatetimeIndex, *, strict: bool = False) -> dict:
    """
    Return a dict of fundamental feature Series for the given ticker and dates.
    Keys include legacy Piotroski fields and WRDS/Compustat short-book
    deterioration fields when available.
    Falls back to 0.0 on any error — never blocks training.
    """
    zero_cols = [
        "f_score",
        "accruals_ratio",
        "roa",
        "delta_roa",
        "delta_leverage",
        "gross_margin",
        "delta_gross_margin",
        "operating_margin",
        "delta_operating_margin",
        "margin_deterioration",
        "debt_to_assets",
        "total_debt_to_assets",
        "weak_profitability",
        "share_issuance_growth",
        "dilution_pressure",
        "filing_delay_days",
        "late_filing_flag",
        "restatement_like_flag",
        "fundamental_deterioration_score",
        "short_interest_ratio",
        "days_to_cover",
        "borrow_crowding_risk",
    ]
    try:
        from features.fundamental_router import fetch_fundamental_features
        fund_df = fetch_fundamental_features(ticker, dates, strict=strict)
        fund_df = fund_df.reindex(columns=zero_cols, fill_value=0.0)
        return {col: fund_df[col] for col in fund_df.columns}
    except Exception:
        if strict:
            raise
        return {col: pd.Series(0.0, index=dates) for col in zero_cols}


def _fetch_earnings_surprise(ticker: str, dates: pd.DatetimeIndex) -> pd.Series:
    """
    Legacy placeholder retained for compatibility only.

    The production research stack no longer uses Yahoo earnings-surprise data.
    Returning NaNs here prevents accidental reintroduction of a non-WRDS
    historical dependency into the training pipeline.
    """
    return pd.Series(np.nan, index=dates)

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

# Compatibility aliases: older tests and internal monkeypatch hooks still target
# the private helper names while the implementation now lives in features.cross_sectional.
_apply_cross_sectional_zscore_columns = apply_cross_sectional_zscore_columns
_attach_cross_sectional_zscore_suffix_block = attach_cross_sectional_zscore_suffix_block


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
        price_col = "Close" if "Close" in df.columns else "AdjClose"
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


def vol_rank_features_by_ticker(
    price_data: dict[str, pd.DataFrame],
    *,
    exclude_tickers: frozenset[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Cross-sectional volatility rank (0–1) per date, lagged 1 trading day per ticker.
    Matches training: rank of 20d realised vol (std of daily returns) across the
    universe, then shift(1). Used at inference so learned w_vol_rank aligns with Ridge.
    """
    ex = frozenset(exclude_tickers or ())
    tickers = [tk for tk in price_data.keys() if tk not in ex]
    if len(tickers) < 2:
        return {}
    rows: list[dict] = []
    for tk in tickers:
        df = price_data[tk]
        price_col = "Close" if "Close" in df.columns else "AdjClose"
        if price_col not in df.columns:
            continue
        close = pd.to_numeric(df[price_col], errors="coerce")
        daily_ret = close.pct_change()
        vol_20_simple = daily_ret.rolling(20).std()
        for dt in vol_20_simple.index:
            v = vol_20_simple.loc[dt]
            if pd.isna(v):
                continue
            rows.append(
                {
                    "date": pd.Timestamp(dt),
                    "ticker": tk,
                    "vol_20_simple": float(v),
                }
            )
    if not rows:
        return {}
    long = pd.DataFrame(rows)
    long["vol_rank"] = long.groupby("date")["vol_20_simple"].rank(pct=True, method="average")
    long["vol_rank"] = long.groupby("ticker")["vol_rank"].shift(1)
    out: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        sub = long.loc[long["ticker"] == tk, ["date", "vol_rank"]]
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


def _download(
    ticker: str,
    start,
    end,
    *,
    provider: str | None = None,
    cache_dir: str | None = None,
    cache_ttl_days: int = 1,
    wrds_username: str | None = None,
    wrds_ticker_to_permno: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Download OHLCV for a single ticker using the shared market_data layer with caching.
    """
    provider = resolve_data_provider(provider)
    ticker = str(ticker).upper()
    if provider == "wrds":
        panel = load_wrds_price_panel(
            [ticker],
            start_date=start,
            end_date=end,
            username=wrds_username,
            cache_dir=cache_dir or "data/cache/wrds",
            cache_ttl_days=cache_ttl_days,
            ticker_to_permno=wrds_ticker_to_permno,
            as_of_date=end,
        )
        df = panel.get(ticker, pd.DataFrame())
        if df is None or df.empty:
            return pd.DataFrame()
        keep = ["Open", "High", "Low", "Close", "Volume"]
        return df[keep].dropna()

    df = get_ohlcv(
        ticker,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        provider=provider,
        use_cache=True,
        cache_dir=cache_dir,
        cache_ttl_days=cache_ttl_days,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    keep = ["Open", "High", "Low", "Close", "Volume"]
    # get_ohlcv already enforces OHLCV_COLUMNS, but keep explicit selection for safety
    return df[keep].dropna()


def _build_features_from_data(
    ticker: str,
    data: pd.DataFrame,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    holding_period: int,
    market_ret: pd.Series | None,
    strict_fundamentals: bool = False,
) -> pd.DataFrame:
    """
    Build a single ticker's feature chunk from a preloaded OHLCV frame.
    """
    try:
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
        vol_std20 = volume.rolling(20).std()
        volume_zscore = (volume - vol_ma20) / vol_std20.replace(0, np.nan)
        volume_zscore = volume_zscore.replace([np.inf, -np.inf], np.nan)

        rolling_corr_market_20 = pd.Series(np.nan, index=features.index)
        capm_alpha = pd.Series(np.nan, index=features.index)
        capm_beta = pd.Series(1.0, index=features.index)
        capm_residual_vol = pd.Series(np.nan, index=features.index)
        idio_momentum_20d = pd.Series(0.0, index=features.index)
        vol_20_simple = daily_ret.rolling(20).std()
        if market_ret is not None:
            market_aligned = market_ret.reindex(features.index).ffill().fillna(0.0)
            rolling_corr_market_20 = daily_ret.rolling(20).corr(market_aligned)
            try:
                from features.capm_features import compute_capm_features
                stock_ret = daily_ret.astype(float).fillna(0.0)
                capm_df = compute_capm_features(stock_ret, market_aligned, window=60, zscore_window=252)
                capm_alpha = capm_df["capm_alpha"]
                capm_beta = capm_df["capm_beta"]
                capm_residual_vol = capm_df["capm_residual_vol"]
            except Exception:
                pass
            spy_ret_20d = (1.0 + market_aligned).rolling(20, min_periods=10).apply(
                np.prod, raw=True
            ) - 1.0
            idio_momentum_20d = (ret_20d - capm_beta * spy_ret_20d).clip(-0.5, 0.5).fillna(0.0)

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
        _holding_period_vol = vol_20_raw * np.sqrt(holding_period)
        forward_ret_risk_adj = (
            forward_ret / _holding_period_vol.clip(lower=0.001)
        ).clip(-10.0, 10.0)

        rsi_14 = mr_rsi_raw.shift(1)
        ret_1d = daily_ret.shift(1)
        vol_ratio_5_20 = (vol_5_raw / vol_20_raw.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        ).shift(1)

        max_52w = close.rolling(252, min_periods=100).max()
        dist_from_52w_high = (close - max_52w) / max_52w.replace(0, np.nan)
        rsi_overbought = (mr_rsi_raw > 70).astype(float)
        vol_expansion = (vol_5_raw / vol_20_raw.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        momentum_acceleration = ret_5d - ret_10d
        d_ret_lag = daily_ret.shift(1)
        down_v = volume.where(d_ret_lag < 0, 0).rolling(20).sum()
        up_v = volume.where(d_ret_lag > 0, 0).rolling(20).sum()
        down_up_vol_ratio = (down_v / up_v.replace(0, np.nan)).fillna(1.0)

        short_term_reversal = (-ret_5d.shift(1)).clip(-0.5, 0.5)
        min_52w = close.rolling(252, min_periods=60).min().replace(0, np.nan)
        dist_from_52w_low = ((close - min_52w) / min_52w.clip(lower=1e-6)).clip(0, 10).shift(1)
        nearness_52w_low = 1.0 / (1.0 + dist_from_52w_low)

        high = data["High"].reindex(features.index)
        low = data["Low"].reindex(features.index)
        parkinson_raw = ((high - low) / close.replace(0, np.nan)).clip(0, 0.25)
        pk_mean = parkinson_raw.rolling(252, min_periods=60).mean()
        pk_std = parkinson_raw.rolling(252, min_periods=60).std(ddof=0).replace(0, np.nan).fillna(1e-6)
        liquidity_stress = ((parkinson_raw - pk_mean) / pk_std).clip(-4, 4).shift(1)

        beta_safe = capm_beta.clip(lower=0.3, upper=3.0)
        beta_adj_momentum = (ret_20d / beta_safe).clip(-2, 2).shift(1)
        nearness_52w_high = (close / max_52w.replace(0, np.nan)).clip(0.0, 1.0).fillna(0.5)
        vol_20_lv = daily_ret.rolling(20, min_periods=10).std()
        low_vol_score = (1.0 - vol_20_lv.rolling(252, min_periods=60).rank(pct=True)).fillna(0.5)
        roll_mean_60 = daily_ret.rolling(60, min_periods=20).mean()
        roll_std_60 = daily_ret.rolling(60, min_periods=20).std().replace(0, np.nan).fillna(1e-6)
        quality_score = (roll_mean_60 / roll_std_60 * np.sqrt(252)).clip(-5.0, 5.0)

        momentum_1m_skip_eom = (close.shift(5) / close.shift(26).replace(0, np.nan) - 1).clip(-1.0, 4.0).fillna(0.0)
        adv_dollar_20 = (close * volume).rolling(20, min_periods=5).mean().shift(1)
        to_20d = (volume / volume.rolling(252, min_periods=60).mean().replace(0, np.nan)).clip(0.0, 20.0)
        to_pct_rank = to_20d.rolling(63, min_periods=21).rank(pct=True).fillna(0.5)
        turnover_pct_rank = to_pct_rank.clip(0.0, 1.0)
        to_centered = (to_pct_rank * 2.0 - 1.0)
        short_term_momentum_score = (momentum_1m_skip_eom * to_centered).clip(-2.0, 2.0).fillna(0.0)
        _vol_pct = vol_20_raw.rolling(252, min_periods=60).rank(pct=True)
        high_vol_reversal_flag = (_vol_pct > 0.67).astype(float).fillna(0.0)
        momentum_12m_skip1 = close.shift(21).pct_change(231).clip(-0.95, 10.0).fillna(0.0)
        fundamental_cols = _fetch_fundamental_cols(ticker, features.index, strict=bool(strict_fundamentals))

        near_high_risk = nearness_52w_high.shift(1).fillna(0.5).clip(0.0, 1.0)
        positive_momentum_risk = (ret_20d.shift(1).clip(lower=0.0, upper=0.50) / 0.50).fillna(0.0)
        vol_expansion_risk = ((vol_expansion.shift(1) - 1.0) / 3.0).clip(0.0, 1.0).fillna(0.0)
        volume_pressure_risk = ((relative_volume.shift(1).fillna(0.0) + 4.0) / 8.0).clip(0.0, 1.0)
        crowding_risk = pd.to_numeric(
            fundamental_cols.get("borrow_crowding_risk", pd.Series(0.0, index=features.index)),
            errors="coerce",
        ).reindex(features.index).fillna(0.0).clip(0.0, 1.0)
        short_interest_risk = pd.to_numeric(
            fundamental_cols.get("short_interest_ratio", pd.Series(0.0, index=features.index)),
            errors="coerce",
        ).reindex(features.index).fillna(0.0).clip(0.0, 1.0)
        short_squeeze_risk = (
            0.25 * near_high_risk
            + 0.20 * positive_momentum_risk
            + 0.20 * vol_expansion_risk
            + 0.15 * volume_pressure_risk
            + 0.10 * crowding_risk
            + 0.10 * short_interest_risk
        ).clip(0.0, 1.0)
        hard_short_squeeze_filter = (short_squeeze_risk > 0.75).astype(float)

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
                "adv_dollar_20": adv_dollar_20,
                "vol_20_simple": vol_20_simple,
                "rsi_14": rsi_14,
                "ret_1d": ret_1d,
                "vol_ratio_5_20": vol_ratio_5_20,
                "dist_from_52w_high": dist_from_52w_high,
                "rsi_overbought": rsi_overbought,
                "vol_expansion": vol_expansion,
                "momentum_acceleration": momentum_acceleration,
                "down_up_vol_ratio": down_up_vol_ratio,
                **fundamental_cols,
                "short_term_reversal": short_term_reversal,
                "nearness_52w_low": nearness_52w_low,
                "dist_from_52w_low": dist_from_52w_low,
                "nearness_52w_high": nearness_52w_high,
                "idio_momentum_20d": idio_momentum_20d,
                "low_vol_score": low_vol_score,
                "quality_score": quality_score,
                "liquidity_stress": liquidity_stress,
                "beta_adj_momentum": beta_adj_momentum,
                "momentum_1m_skip_eom": momentum_1m_skip_eom,
                "turnover_pct_rank": turnover_pct_rank,
                "short_term_momentum_score": short_term_momentum_score,
                "high_vol_reversal_flag": high_vol_reversal_flag,
                "momentum_12m_skip1": momentum_12m_skip1,
                "short_squeeze_risk": short_squeeze_risk,
                "hard_short_squeeze_filter": hard_short_squeeze_filter,
                "forward_return": forward_ret,
                "forward_return_risk_adj": forward_ret_risk_adj,
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
        return pd.DataFrame()


def _build_features_for_ticker(
    ticker: str,
    dl_start: pd.Timestamp,
    dl_end: pd.Timestamp,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    holding_period: int,
    market_ret: pd.Series | None,
    provider: str | None = None,
    cache_dir: str | None = None,
    cache_ttl_days: int = 1,
    wrds_username: str | None = None,
    wrds_ticker_to_permno: dict[str, int] | None = None,
    strict_fundamentals: bool = False,
) -> pd.DataFrame:
    """
    Worker helper: build feature chunk for a single ticker.
    Returns an empty DataFrame on failure so that the caller can skip it.
    """
    try:
        data = _download(
            ticker,
            dl_start,
            dl_end,
            provider=provider,
            cache_dir=cache_dir,
            cache_ttl_days=cache_ttl_days,
            wrds_username=wrds_username,
            wrds_ticker_to_permno=wrds_ticker_to_permno,
        )
        build_kwargs = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "holding_period": holding_period,
            "market_ret": market_ret,
        }
        if strict_fundamentals:
            build_kwargs["strict_fundamentals"] = True
        return _build_features_from_data(ticker, data, **build_kwargs)
    except Exception:
        return pd.DataFrame()


def build_feature_matrix(
    tickers: list[str],
    start_date: str,
    end_date: str,
    holding_period: int = 5,
    feature_subset: list[str] | None = None,
    data_provider: str | None = None,
    cache_dir: str | None = None,
    cache_ttl_days: int = 1,
    wrds_username: str | None = None,
    wrds_ticker_to_permno: dict[str, int] | None = None,
    strict_fundamentals: bool = False,
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
    provider = resolve_data_provider(data_provider)
    if provider == "wrds":
        from features.wrds_panel_engine import build_wrds_feature_matrix_batched

        return build_wrds_feature_matrix_batched(
            tickers,
            start_date=start_date,
            end_date=end_date,
            holding_period=holding_period,
            feature_subset=feature_subset,
            cache_dir=cache_dir,
            cache_ttl_days=cache_ttl_days,
            wrds_username=wrds_username,
            wrds_ticker_to_permno=wrds_ticker_to_permno,
            strict_fundamentals=bool(strict_fundamentals),
            market_ticker=MARKET_TICKER,
        )

    dl_start = pd.Timestamp(start_date) - timedelta(days=HISTORY_BUFFER_DAYS)
    dl_end = pd.Timestamp(end_date) + timedelta(days=holding_period * 2)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    wrds_ticker_to_permno = {
        str(t).upper(): int(p)
        for t, p in (wrds_ticker_to_permno or {}).items()
        if t is not None and p is not None
    }
    cache_dir = cache_dir or ("data/cache/wrds" if provider == "wrds" else None)

    # Market series for cross-ticker feature (no look-ahead: historical SPY returns only)
    market_ret = None
    spy = None
    vix_zscore = pd.Series(dtype=float)
    vol_spike = pd.Series(dtype=float)
    vix_term_zscore_series = pd.Series(dtype=float)
    vol_risk_premium = pd.Series(dtype=float)  # P4: VRP = implied vol - realized vol

    credit_spread_proxy = pd.Series(dtype=float)   # HYG vs IEF: widening = risk-off
    yield_curve_slope   = pd.Series(dtype=float)   # 10y - 3m yield: inverted = recession
    wrds_panel: dict[str, pd.DataFrame] = {}
    if provider == "wrds":
        try:
            wrds_panel = load_wrds_price_panel(
                list(dict.fromkeys([*(str(t).upper() for t in tickers), MARKET_TICKER, "HYG", "IEF", "SHY"])),
                start_date=dl_start,
                end_date=dl_end,
                username=wrds_username,
                cache_dir=cache_dir or "data/cache/wrds",
                cache_ttl_days=cache_ttl_days,
                ticker_to_permno=wrds_ticker_to_permno,
                as_of_date=dl_end,
            )
        except Exception:
            logger.exception("WRDS preload failed in feature_builder")
            wrds_panel = {}

    try:
        if provider == "wrds":
            spy = wrds_panel.get(MARKET_TICKER, pd.DataFrame())
        else:
            spy = _download(
                MARKET_TICKER,
                dl_start,
                dl_end,
                provider=provider,
                cache_dir=cache_dir,
                cache_ttl_days=cache_ttl_days,
                wrds_username=wrds_username,
                wrds_ticker_to_permno=wrds_ticker_to_permno,
            )
        if not spy.empty and len(spy) >= 25:
            market_ret = pd.to_numeric(spy["Close"], errors="coerce").pct_change()
    except Exception:
        spy = None
        market_ret = None

    try:
        if provider == "wrds":
            hyg = wrds_panel.get("HYG", pd.DataFrame())
            ief = wrds_panel.get("IEF", pd.DataFrame())
        else:
            hyg = _download("HYG", dl_start, dl_end, provider=provider, cache_dir=cache_dir, cache_ttl_days=cache_ttl_days)
            ief = _download("IEF", dl_start, dl_end, provider=provider, cache_dir=cache_dir, cache_ttl_days=cache_ttl_days)
        if not hyg.empty and not ief.empty:
            hyg_ret = pd.to_numeric(hyg["Close"], errors="coerce").pct_change()
            ief_ret = pd.to_numeric(ief["Close"], errors="coerce").pct_change()
            spread = (ief_ret - hyg_ret).reindex(hyg_ret.index.union(ief_ret.index)).ffill()
            cs_m = spread.rolling(60, min_periods=20).mean()
            cs_s = spread.rolling(60, min_periods=20).std(ddof=0).replace(0, np.nan).fillna(1e-6)
            credit_spread_proxy = ((spread - cs_m) / cs_s).clip(-4, 4).shift(1)
    except Exception:
        pass

    try:
        if provider == "wrds":
            shy = wrds_panel.get("SHY", pd.DataFrame())
            ief = wrds_panel.get("IEF", pd.DataFrame())
            if not shy.empty and not ief.empty:
                shy_ret = pd.to_numeric(shy["Close"], errors="coerce").pct_change()
                ief_ret = pd.to_numeric(ief["Close"], errors="coerce").pct_change()
                slope_raw = (ief_ret - shy_ret).reindex(ief_ret.index.union(shy_ret.index)).ffill()
                yield_curve_slope = _timeseries_zscore(slope_raw).clip(-4, 4).shift(1)
        else:
            tnx_raw = get_ohlcv(
                "^TNX",
                dl_start.strftime("%Y-%m-%d"),
                dl_end.strftime("%Y-%m-%d"),
                provider=provider,
                use_cache=True,
                cache_dir=cache_dir,
                cache_ttl_days=cache_ttl_days,
            )
            irx_raw = get_ohlcv(
                "^IRX",
                dl_start.strftime("%Y-%m-%d"),
                dl_end.strftime("%Y-%m-%d"),
                provider=provider,
                use_cache=True,
                cache_dir=cache_dir,
                cache_ttl_days=cache_ttl_days,
            )
            if tnx_raw is not None and irx_raw is not None and not tnx_raw.empty and not irx_raw.empty:
                tnx = pd.to_numeric(tnx_raw["Close"], errors="coerce").dropna()
                irx = pd.to_numeric(irx_raw["Close"], errors="coerce").dropna()
                slope_raw = (tnx - irx.reindex(tnx.index).ffill())
                yield_curve_slope = _timeseries_zscore(slope_raw).clip(-4, 4).shift(1)
    except Exception:
        pass

    try:
        if spy is not None and not spy.empty and "Close" in spy.columns:
            spy_close = pd.to_numeric(spy["Close"], errors="coerce").dropna().sort_index()
            spy_ret = spy_close.pct_change()
            vol5 = spy_ret.rolling(5).std() * np.sqrt(252.0)
            vol20 = spy_ret.rolling(20).std() * np.sqrt(252.0)
            vol60 = spy_ret.rolling(60).std() * np.sqrt(252.0)
            vol_spike = (vol5 / vol60).replace([np.inf, -np.inf], np.nan).shift(1)
            if provider == "wrds":
                vix_zscore = _timeseries_zscore(vol20 * 100.0).shift(1)
                realized_term = (vol20 / vol60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
                vix_term_zscore_series = _timeseries_zscore(realized_term).shift(1)
            else:
                vix_raw = get_ohlcv(
                    "^VIX",
                    dl_start.strftime("%Y-%m-%d"),
                    dl_end.strftime("%Y-%m-%d"),
                    provider=provider,
                    use_cache=True,
                    cache_dir=cache_dir,
                    cache_ttl_days=cache_ttl_days,
                )
                if vix_raw is not None and not vix_raw.empty and "Close" in vix_raw.columns:
                    vix_close = pd.to_numeric(vix_raw["Close"], errors="coerce").dropna().sort_index()
                    vix_zscore = _timeseries_zscore(vix_close).shift(1)
                    vix3m_raw = get_ohlcv(
                        "^VIX3M",
                        dl_start.strftime("%Y-%m-%d"),
                        dl_end.strftime("%Y-%m-%d"),
                        provider=provider,
                        use_cache=True,
                        cache_dir=cache_dir,
                        cache_ttl_days=cache_ttl_days,
                    )
                    if vix3m_raw is not None and not vix3m_raw.empty and "Close" in vix3m_raw.columns:
                        vix3m_close = pd.to_numeric(vix3m_raw["Close"], errors="coerce").dropna().sort_index()
                        vix3m_aligned = vix3m_close.reindex(vix_close.index).ffill()
                        vix_ratio_lag = (vix_close / vix3m_aligned.replace(0.0, np.nan)).replace(
                            [np.inf, -np.inf], np.nan
                        ).shift(1)
                        vix_term_zscore_series = _timeseries_zscore(vix_ratio_lag).shift(1)
                    # P4: Volatility risk premium = implied vol (VIX close) - realized vol (SPY 20d annualized)
                    # High VRP signals risk aversion; low/negative VRP signals complacency.
                    if "vix_close" in locals() and not vix_close.empty:
                        vix_level = vix_close.reindex(vol20.index)
                        vol_risk_premium = (vix_level - vol20 * 100.0).shift(1)
                    else:
                        vol_risk_premium = _timeseries_zscore(vol20 * 100.0 - vol20.rolling(60).mean() * 100.0).shift(1)
    except Exception:
        pass

    chunks: list[pd.DataFrame] = []

    if provider == "wrds":
        for i, ticker in enumerate(tickers, 1):
            ticker = str(ticker).upper()
            print(f"  [{i}/{len(tickers)}] {ticker}…", end=" ")
            try:
                build_kwargs = {
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "holding_period": holding_period,
                    "market_ret": market_ret,
                }
                if strict_fundamentals:
                    build_kwargs["strict_fundamentals"] = True
                chunk = _build_features_from_data(
                    ticker,
                    wrds_panel.get(ticker, pd.DataFrame()),
                    **build_kwargs,
                )
                if chunk is None or chunk.empty:
                    print("no valid rows")
                    continue
                chunks.append(chunk)
                print(f"{len(chunk)} rows")
            except Exception as exc:
                print(f"ERROR: {exc}")
    else:
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
                    provider,
                    cache_dir,
                    cache_ttl_days,
                    wrds_username,
                    wrds_ticker_to_permno,
                    strict_fundamentals,
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
    # Pillar 26: Duplicate Firewall (Sanitization)
    # Expanded universes often contain data-overlaps in the cache; we must ensure
    # uniqueness before cross-sectional pivoting (Pillar 24 Ensemble stability).
    result.drop_duplicates(subset=["date", "ticker"], keep="last", inplace=True)
    result.sort_values(["date", "ticker"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    # Cross-sectional volatility rank (0–1); lag 1d per ticker (no lookahead).
    if "vol_20_simple" in result.columns:
        result["vol_rank"] = result.groupby("date")["vol_20_simple"].rank(
            pct=True,
            method="average",
        )
        result["vol_rank"] = result.groupby("ticker")["vol_rank"].shift(1).fillna(0.5)
    else:
        result["vol_rank"] = 0.5

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

        # Bear/Crisis regime macro features — date-level, same value for all tickers on a date.
        if not credit_spread_proxy.empty:
            result["credit_spread_proxy"] = (
                result["date"].map(credit_spread_proxy.to_dict()).astype(float).fillna(0.0)
            )
        else:
            result["credit_spread_proxy"] = 0.0

        if not yield_curve_slope.empty:
            result["yield_curve_slope"] = (
                result["date"].map(yield_curve_slope.to_dict()).astype(float).fillna(0.0)
            )
        else:
            result["yield_curve_slope"] = 0.0

        # P4: Volatility risk premium (VRP) — attached per date, same across tickers.
        if not vol_risk_premium.empty:
            result["vol_risk_premium"] = (
                result["date"].map(vol_risk_premium.to_dict()).astype(float).fillna(0.0)
            )
        else:
            result["vol_risk_premium"] = 0.0

        # Macro levels are identical across tickers per date; CS z collapses to ~0 — kept for spec / symmetry.
        # vol_risk_premium is already a level (implied - realized); CS z-score preserves relative differences.
        result = _apply_cross_sectional_zscore_columns(result, ["vix_zscore", "vol_spike", "vol_risk_premium"])

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

    # Industry-relative reversal (RRLP 2023, IRRX): -(ret_20d - sector_relative_20d)
    # Computed at panel level because it requires the cross-sectional sector_relative_20d.
    # Positive = stock fell vs sector (liquidity-driven) → bounce expected (LONG signal)
    # Negative = stock rose vs sector without news → reversal expected (SHORT signal)
    if "ret_20d" in result.columns:
        result["industry_relative_reversal"] = (
            -(result["ret_20d"] - result["sector_relative_20d"])
        ).clip(-2.0, 2.0).fillna(0.0)
    else:
        result["industry_relative_reversal"] = 0.0

    # Cross-sectional momentum percentile (0–1) based on 6m return excluding last month.
    if "cs_momentum_raw" in result.columns:
        result["cs_momentum_percentile"] = (
            result.groupby("date")["cs_momentum_raw"]
            .rank(pct=True, method="average")
        )

    # SPY-based volatility regime features: map each date to a SPY-based regime and flag high-volatility days.
    try:
        unique_dates = pd.to_datetime(result["date"].unique())
        regime_series = get_regime_series_for_dates(
            unique_dates,
            start_date,
            end_date,
            data_provider=provider,
            cache_dir=cache_dir,
            cache_ttl_days=cache_ttl_days,
            wrds_username=wrds_username,
            wrds_ticker_to_permno=wrds_ticker_to_permno,
        )
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

    # Cross-sectional z-scores: compute the generic ``*_cs_z`` block in one pass.
    # MR + sector-relative + panel-inplace columns are already CS-treated; do not duplicate.
    _precomputed_cs_z = (
        set(_MR_RAW_TO_OUT.values())
        | {"sector_relative_20d", "sector_relative_60d"}
        | set(_CS_Z_PANEL_INPLACE_COLS)
        | {"vix_term_zscore"}  # already TS z-scored; identical across tickers per date
        | {"vol_rank", "vol_20_simple"}  # vol_rank is percentile; vol_20_simple is raw input only
        | {"credit_spread_proxy", "yield_curve_slope", "vol_risk_premium"}  # P4: macro features, TS z-scored
    )
    result = _attach_cross_sectional_zscore_suffix_block(
        result,
        exclude_columns=frozenset(_precomputed_cs_z),
    )

    # Phase 1 / Phase 2 ablation: zero disabled COMPOUND columns (TSE_ABLATION_STEP env).
    _zero_cols = feature_columns_to_zero_for_ablation()
    for col in _zero_cols:
        if col in result.columns:
            result[col] = 0.0

    # Optional feature selection: keep only requested feature columns, but always
    # preserve identifiers and targets for downstream training/backtesting.
    subset = [str(c) for c in (feature_subset or []) if str(c).strip()]
    if subset:
        always_keep = [
            "date",
            "ticker",
            "sector",
            "regime_label",
            "daily_return",
            "adv_dollar_20",
            "realised_vol_20d",
            "capm_beta",
            "expected_round_trip_cost_frac",
            "forward_return",
            "forward_return_risk_adj",  # C3: risk-adjusted target
            "forward_return_excess",
            "direction",
        ]
        keep = []
        for c in always_keep + subset:
            if c in result.columns and c not in keep:
                keep.append(c)
        result = result[keep].copy()

    return result
