from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agents.weight_learning_agent.feature_flags import feature_columns_to_zero_for_ablation
from agents.weight_learning_agent.regime_detection import get_regime_series_for_dates
from agents.volatility_agent.volatility_model import HIGH_VOL_THRESHOLD, LOW_VOL_THRESHOLD
from execution.cost_model import TransactionCostModel
from features.capm_features import compute_capm_features
from features.cross_sectional import (
    apply_cross_sectional_zscore_columns,
    attach_cross_sectional_zscore_suffix_block,
    compute_sector_relative_shifted_cs_long,
    cross_sectional_zscore,
)
from features.wrds_fundamental_builder import fetch_fundamental_features_batch
from utils.quant_utils import get_sector
from utils.wrds_data import load_wrds_price_panel
from utils.wrds_universe import connect_wrds


_MR_RAW_TO_OUT = {
    "mr_rsi_raw": "rsi_zscore",
    "mr_bb_raw": "bb_position",
    "mr_dist_high_raw": "dist_high",
    "mr_dist_low_raw": "dist_low",
    "mr_overnight_raw": "overnight_gap",
    "mr_intraday_raw": "intraday_rev",
}

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


def _group_rolling_stat(
    series: pd.Series,
    tickers: pd.Series,
    *,
    window: int,
    min_periods: int,
    stat: str,
    ddof: int = 1,
) -> pd.Series:
    grouped = series.groupby(tickers, sort=False)
    if stat == "mean":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    if stat == "max":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).max())
    if stat == "min":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).min())
    if stat == "std":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).std(ddof=ddof))
    raise ValueError(f"unsupported rolling stat: {stat}")


def _group_shift(series: pd.Series, tickers: pd.Series, periods: int = 1) -> pd.Series:
    return series.groupby(tickers, sort=False).shift(periods)


def _group_pct_change(series: pd.Series, tickers: pd.Series, periods: int) -> pd.Series:
    return series.groupby(tickers, sort=False).pct_change(periods)


def _timeseries_zscore(series: pd.Series, tickers: pd.Series, *, window: int = 252, min_periods: int = 60) -> pd.Series:
    mean = _group_rolling_stat(series, tickers, window=window, min_periods=min_periods, stat="mean")
    std = _group_rolling_stat(series, tickers, window=window, min_periods=min_periods, stat="std", ddof=0)
    std = std.replace(0.0, np.nan).fillna(1e-6)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan)


def _group_rolling_rank_pct(series: pd.Series, tickers: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    grouped = series.groupby(tickers, sort=False)
    return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).rank(pct=True))


def _attach_mr_cross_sectional_zscore_shifted(result: pd.DataFrame) -> pd.DataFrame:
    for raw_col, out_col in _MR_RAW_TO_OUT.items():
        if raw_col not in result.columns:
            result[out_col] = 0.0
            continue
        pivot = result.pivot(index="date", columns="ticker", values=raw_col)
        z = cross_sectional_zscore(pivot).shift(1)
        z_long = z.stack(future_stack=True).reset_index()
        z_long.columns = ["date", "ticker", out_col]
        result = result.drop(columns=[out_col], errors="ignore")
        result = result.merge(z_long, on=["date", "ticker"], how="left")
        result[out_col] = pd.to_numeric(result[out_col], errors="coerce").fillna(0.0)
    return result.drop(columns=[c for c in _MR_RAW_TO_OUT if c in result.columns], errors="ignore")


def _build_macro_series(wrds_panel: dict[str, pd.DataFrame], *, holding_period: int) -> dict[str, pd.Series]:
    market_ret = pd.Series(dtype=float)
    vix_zscore = pd.Series(dtype=float)
    vol_spike = pd.Series(dtype=float)
    vix_term_zscore = pd.Series(dtype=float)
    credit_spread_proxy = pd.Series(dtype=float)
    yield_curve_slope = pd.Series(dtype=float)
    spy_forward = pd.Series(dtype=float)

    spy = wrds_panel.get("SPY", pd.DataFrame())
    if not spy.empty and "Close" in spy.columns:
        spy_close = pd.to_numeric(spy["Close"], errors="coerce").dropna().sort_index()
        market_ret = pd.to_numeric(spy.get("ret"), errors="coerce").reindex(spy_close.index)
        if market_ret.isna().all():
            market_ret = spy_close.pct_change()
        vol20 = market_ret.rolling(20, min_periods=10).std() * np.sqrt(252.0)
        vol60 = market_ret.rolling(60, min_periods=20).std() * np.sqrt(252.0)
        vol5 = market_ret.rolling(5, min_periods=5).std() * np.sqrt(252.0)
        vol_spike = (vol5 / vol60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).shift(1)
        vix_zscore = ((vol20 * 100.0 - vol20.rolling(252, min_periods=60).mean() * 100.0) /
                      (vol20.rolling(252, min_periods=60).std(ddof=0) * 100.0).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).shift(1)
        realized_term = (vol20 / vol60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        vix_term_zscore = (
            (realized_term - realized_term.rolling(252, min_periods=60).mean()) /
            realized_term.rolling(252, min_periods=60).std(ddof=0).replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).shift(1)
        spy_forward = (spy_close.shift(-holding_period) / spy_close - 1.0).rename("spy_forward_5d")

    hyg = wrds_panel.get("HYG", pd.DataFrame())
    ief = wrds_panel.get("IEF", pd.DataFrame())
    shy = wrds_panel.get("SHY", pd.DataFrame())
    if not hyg.empty and not ief.empty:
        hyg_ret = pd.to_numeric(hyg.get("ret"), errors="coerce")
        if hyg_ret.isna().all():
            hyg_ret = pd.to_numeric(hyg["Close"], errors="coerce").pct_change()
        ief_ret = pd.to_numeric(ief.get("ret"), errors="coerce")
        if ief_ret.isna().all():
            ief_ret = pd.to_numeric(ief["Close"], errors="coerce").pct_change()
        spread = (ief_ret - hyg_ret).reindex(hyg_ret.index.union(ief_ret.index)).ffill()
        cs_mean = spread.rolling(60, min_periods=20).mean()
        cs_std = spread.rolling(60, min_periods=20).std(ddof=0).replace(0, np.nan).fillna(1e-6)
        credit_spread_proxy = ((spread - cs_mean) / cs_std).clip(-4.0, 4.0).shift(1)
    if not shy.empty and not ief.empty:
        shy_ret = pd.to_numeric(shy.get("ret"), errors="coerce")
        if shy_ret.isna().all():
            shy_ret = pd.to_numeric(shy["Close"], errors="coerce").pct_change()
        ief_ret = pd.to_numeric(ief.get("ret"), errors="coerce")
        if ief_ret.isna().all():
            ief_ret = pd.to_numeric(ief["Close"], errors="coerce").pct_change()
        slope_raw = (ief_ret - shy_ret).reindex(ief_ret.index.union(shy_ret.index)).ffill()
        yc_mean = slope_raw.rolling(252, min_periods=60).mean()
        yc_std = slope_raw.rolling(252, min_periods=60).std(ddof=0).replace(0, np.nan).fillna(1e-6)
        yield_curve_slope = ((slope_raw - yc_mean) / yc_std).clip(-4.0, 4.0).shift(1)

    return {
        "market_ret": market_ret,
        "vix_zscore": vix_zscore,
        "vol_spike": vol_spike,
        "vix_term_zscore": vix_term_zscore,
        "credit_spread_proxy": credit_spread_proxy,
        "yield_curve_slope": yield_curve_slope,
        "spy_forward": spy_forward,
    }


def _canonical_long_panel(
    tickers: list[str],
    wrds_panel: dict[str, pd.DataFrame],
    *,
    ticker_to_permno: dict[str, int] | None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for ticker in tickers:
        ticker_u = str(ticker).upper()
        data = wrds_panel.get(ticker_u, pd.DataFrame())
        if data is None or data.empty or len(data) < 210:
            continue
        tmp = data.copy()
        tmp.index = pd.to_datetime(tmp.index, errors="coerce")
        idx_name = tmp.index.name or "index"
        tmp = tmp.reset_index().rename(columns={idx_name: "date"})
        tmp["ticker"] = ticker_u
        tmp["permno"] = int((ticker_to_permno or {}).get(ticker_u, 0) or 0)
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def _load_batched_fundamentals(
    panel: pd.DataFrame,
    *,
    wrds_username: str | None,
    cache_dir: str | None,
    cache_ttl_days: int,
    strict_fundamentals: bool,
) -> pd.DataFrame:
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
    if panel.empty or "permno" not in panel.columns:
        return pd.DataFrame(columns=["date", "ticker", *zero_cols])

    valid = panel.loc[pd.to_numeric(panel["permno"], errors="coerce").fillna(0).astype(int) > 0, ["ticker", "permno", "date"]].copy()
    if valid.empty:
        return pd.DataFrame(columns=["date", "ticker", *zero_cols])

    permno_dates_map = {
        int(permno): pd.DatetimeIndex(sorted(pd.to_datetime(group["date"], errors="coerce").dropna().unique()))
        for permno, group in valid.groupby("permno", sort=False)
    }
    if not permno_dates_map:
        return pd.DataFrame(columns=["date", "ticker", *zero_cols])

    try:
        db = connect_wrds(wrds_username)
        cache_root = Path(cache_dir or "data/cache/wrds") / "fundamentals"
        batch = fetch_fundamental_features_batch(
            db,
            permno_dates_map,
            cache_dir=cache_root,
            cache_ttl_days=int(cache_ttl_days),
        )
    except Exception:
        if strict_fundamentals:
            raise
        batch = {permno: pd.DataFrame(0.0, index=dates, columns=zero_cols) for permno, dates in permno_dates_map.items()}

    rows: list[pd.DataFrame] = []
    permno_to_ticker = (
        valid[["permno", "ticker"]]
        .drop_duplicates(subset=["permno"], keep="first")
        .set_index("permno")["ticker"]
        .astype(str)
        .to_dict()
    )
    for permno, frame in batch.items():
        tmp = frame.copy().reindex(columns=zero_cols, fill_value=0.0)
        tmp.index = pd.DatetimeIndex(tmp.index)
        tmp = tmp.reset_index().rename(columns={"index": "date"})
        tmp["ticker"] = permno_to_ticker.get(int(permno), "")
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", *zero_cols])
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def _compute_capm_block(panel: pd.DataFrame, market_ret: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    out["rolling_corr_market_20"] = 0.0
    out["capm_alpha"] = 0.0
    out["capm_beta"] = 1.0
    out["capm_residual_vol"] = 0.0
    out["idio_momentum_20d"] = 0.0
    if market_ret is None or market_ret.empty:
        return out
    for _, g in panel.groupby("ticker", sort=False):
        idx = g.index
        stock_ret = pd.to_numeric(g["daily_return"], errors="coerce").fillna(0.0)
        market_values = (
            market_ret.reindex(pd.to_datetime(g["date"], errors="coerce")).ffill().fillna(0.0).to_numpy(dtype=float)
        )
        aligned_market = pd.Series(market_values, index=idx, dtype=float)
        rolling_corr = stock_ret.rolling(20, min_periods=10).corr(aligned_market).fillna(0.0)
        out.loc[idx, "rolling_corr_market_20"] = rolling_corr.reindex(idx).to_numpy(dtype=float)
        try:
            capm_df = compute_capm_features(stock_ret, aligned_market, window=60, zscore_window=252)
            out.loc[idx, "capm_alpha"] = (
                pd.to_numeric(capm_df["capm_alpha"], errors="coerce").fillna(0.0).reindex(idx).to_numpy(dtype=float)
            )
            out.loc[idx, "capm_beta"] = (
                pd.to_numeric(capm_df["capm_beta"], errors="coerce").fillna(1.0).reindex(idx).to_numpy(dtype=float)
            )
            out.loc[idx, "capm_residual_vol"] = (
                pd.to_numeric(capm_df["capm_residual_vol"], errors="coerce").fillna(0.0).reindex(idx).to_numpy(dtype=float)
            )
        except Exception:
            out.loc[idx, "capm_beta"] = 1.0
        spy_ret_20d = (1.0 + aligned_market).rolling(20, min_periods=10).apply(np.prod, raw=True) - 1.0
        ret_20d = pd.to_numeric(panel.loc[idx, "ret_20d"], errors="coerce").fillna(0.0)
        beta = pd.to_numeric(out.loc[idx, "capm_beta"], errors="coerce").fillna(1.0)
        idio = (ret_20d - beta * spy_ret_20d).clip(-0.5, 0.5).fillna(0.0)
        out.loc[idx, "idio_momentum_20d"] = idio.reindex(idx).to_numpy(dtype=float)
    return out


def build_wrds_feature_matrix_batched(
    tickers: list[str],
    *,
    start_date: str,
    end_date: str,
    holding_period: int,
    feature_subset: list[str] | None,
    cache_dir: str | None,
    cache_ttl_days: int,
    wrds_username: str | None,
    wrds_ticker_to_permno: dict[str, int] | None,
    strict_fundamentals: bool,
    market_ticker: str = "SPY",
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    dl_start = start_ts - pd.Timedelta(days=400)
    dl_end = end_ts + pd.Timedelta(days=holding_period * 2)
    helper_tickers = [market_ticker, "HYG", "IEF", "SHY"]
    requested = list(dict.fromkeys([*(str(t).upper() for t in tickers), *helper_tickers]))
    wrds_panel = load_wrds_price_panel(
        requested,
        start_date=dl_start,
        end_date=dl_end,
        username=wrds_username,
        cache_dir=cache_dir or "data/cache/wrds",
        cache_ttl_days=int(cache_ttl_days),
        ticker_to_permno=wrds_ticker_to_permno,
        as_of_date=dl_end,
    )

    panel = _canonical_long_panel([str(t).upper() for t in tickers], wrds_panel, ticker_to_permno=wrds_ticker_to_permno)
    if panel.empty:
        return pd.DataFrame()

    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["sector"] = panel["ticker"].map(get_sector)
    panel["f_regional"] = 0.0
    panel["f_global"] = 0.0
    panel["f_social"] = 0.0

    tick = panel["ticker"]
    close = pd.to_numeric(panel["Close"], errors="coerce")
    open_px = pd.to_numeric(panel["Open"], errors="coerce")
    high_px = pd.to_numeric(panel["High"], errors="coerce")
    low_px = pd.to_numeric(panel["Low"], errors="coerce")
    volume = pd.to_numeric(panel["Volume"], errors="coerce")

    daily_return = pd.to_numeric(panel.get("ret"), errors="coerce")
    if daily_return.isna().all():
        daily_return = _group_pct_change(close, tick, 1)
    daily_return = daily_return.replace([np.inf, -np.inf], np.nan)
    panel["daily_return"] = daily_return

    panel["ret_5d"] = _group_pct_change(close, tick, 5)
    panel["ret_10d"] = _group_pct_change(close, tick, 10)
    panel["ret_20d"] = _group_pct_change(close, tick, 20)
    panel["ret_60d"] = _group_pct_change(close, tick, 60)
    panel["momentum_3m"] = _group_pct_change(close, tick, 63)
    panel["momentum_6m"] = _group_pct_change(close, tick, 126)
    panel["momentum_12m_skip1"] = close.groupby(tick, sort=False).transform(lambda s: s.shift(21).pct_change(231))
    ma50 = _group_rolling_stat(close, tick, window=50, min_periods=50, stat="mean")
    ma200 = _group_rolling_stat(close, tick, window=200, min_periods=200, stat="mean")
    panel["ma_crossover"] = np.where((ma50 / close.replace(0.0, np.nan) - 1.0) > (ma200 / close.replace(0.0, np.nan) - 1.0), 1.0, -1.0)
    trend_scores = (
        0.25 * panel["momentum_3m"].fillna(0.0) * 10.0
        + 0.25 * panel["momentum_6m"].fillna(0.0) * 10.0
        + 0.25 * panel["ma_crossover"]
        + 0.25 * panel["momentum_12m_skip1"].fillna(0.0) * 10.0
    )

    vol_5_raw = _group_rolling_stat(daily_return, tick, window=5, min_periods=5, stat="std")
    vol_10_raw = _group_rolling_stat(daily_return, tick, window=10, min_periods=10, stat="std")
    vol_20_raw = _group_rolling_stat(daily_return, tick, window=20, min_periods=10, stat="std")
    vol_60_raw = _group_rolling_stat(daily_return, tick, window=60, min_periods=20, stat="std")
    panel["realised_vol_20d"] = vol_20_raw
    panel["vol_of_vol_20"] = _group_rolling_stat(vol_20_raw, tick, window=20, min_periods=20, stat="std")
    panel["jump_indicator"] = (daily_return.abs() > (2.0 * vol_20_raw.replace(0.0, np.nan))).astype(float).fillna(0.0)
    panel["rolling_vol_5"] = _timeseries_zscore(vol_5_raw, tick).fillna(0.0)
    panel["rolling_vol_10"] = _timeseries_zscore(vol_10_raw, tick).fillna(0.0)
    panel["rolling_vol_20"] = _timeseries_zscore(vol_20_raw, tick).fillna(0.0)
    panel["rolling_vol_60"] = _timeseries_zscore(vol_60_raw, tick).fillna(0.0)
    panel["vol_20_simple"] = vol_20_raw
    vol252 = _group_rolling_stat(daily_return, tick, window=252, min_periods=60, stat="std")
    panel["vol_regime_ratio"] = (vol_20_raw / vol252.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    panel["is_high_vol_stock_regime"] = (panel["vol_regime_ratio"] > 1.2).astype(float)

    ann_vol20 = vol_20_raw * np.sqrt(252.0)
    confidence_mult = np.where(
        ann_vol20 < LOW_VOL_THRESHOLD,
        1.0,
        np.where(ann_vol20 <= HIGH_VOL_THRESHOLD, 0.6, 0.3),
    )
    panel["confidence_mult"] = pd.Series(confidence_mult, index=panel.index, dtype=float)
    panel["f_trend"] = trend_scores.fillna(0.0) * panel["confidence_mult"]

    vol_ma20 = _group_rolling_stat(volume, tick, window=20, min_periods=5, stat="mean")
    rel_volume_raw = (volume / vol_ma20.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    panel["relative_volume"] = _timeseries_zscore(rel_volume_raw, tick).fillna(0.0)
    vol_std20 = _group_rolling_stat(volume, tick, window=20, min_periods=5, stat="std")
    panel["volume_zscore"] = ((volume - vol_ma20) / vol_std20.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    panel["volume_surprise"] = panel["volume_zscore"]
    panel["adv_dollar_20"] = _group_rolling_stat(close * volume, tick, window=20, min_periods=5, stat="mean")

    capm_block = _compute_capm_block(panel, _build_macro_series(wrds_panel, holding_period=holding_period)["market_ret"])
    for col in capm_block.columns:
        panel[col] = capm_block[col]

    delta = close.groupby(tick, sort=False).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.groupby(tick, sort=False).transform(lambda s: s.rolling(14, min_periods=14).mean())
    avg_loss = loss.groupby(tick, sort=False).transform(lambda s: s.rolling(14, min_periods=14).mean())
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    panel["mr_rsi_raw"] = 100.0 - (100.0 / (1.0 + rs))
    rolling_max_20 = _group_rolling_stat(close, tick, window=20, min_periods=10, stat="max")
    rolling_min_20 = _group_rolling_stat(close, tick, window=20, min_periods=10, stat="min")
    panel["mr_dist_high_raw"] = (close - rolling_max_20) / rolling_max_20.replace(0, np.nan)
    panel["mr_dist_low_raw"] = (close - rolling_min_20) / rolling_min_20.replace(0, np.nan)
    prev_close = _group_shift(close, tick, 1)
    panel["mr_overnight_raw"] = (open_px - prev_close) / prev_close.replace(0, np.nan)
    panel["mr_intraday_raw"] = (close - open_px) / open_px.replace(0, np.nan)
    bb_mid = _group_rolling_stat(close, tick, window=20, min_periods=20, stat="mean")
    bb_std = _group_rolling_stat(close, tick, window=20, min_periods=20, stat="std")
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower).replace(0, np.nan)
    panel["mr_bb_raw"] = (close - bb_lower) / bb_width

    panel["forward_return"] = close.groupby(tick, sort=False).shift(-holding_period) / close - 1.0
    holding_vol = vol_20_raw * np.sqrt(max(1, int(holding_period)))
    panel["forward_return_risk_adj"] = (panel["forward_return"] / holding_vol.clip(lower=0.001)).clip(-10.0, 10.0)
    panel["rsi_14"] = _group_shift(panel["mr_rsi_raw"], tick, 1)
    panel["ret_1d"] = _group_shift(daily_return, tick, 1)
    panel["vol_ratio_5_20"] = (vol_5_raw / vol_20_raw.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    panel["vol_ratio_5_20"] = _group_shift(panel["vol_ratio_5_20"], tick, 1)

    max_52w = _group_rolling_stat(close, tick, window=252, min_periods=100, stat="max")
    min_52w = _group_rolling_stat(close, tick, window=252, min_periods=60, stat="min")
    panel["dist_from_52w_high"] = (close - max_52w) / max_52w.replace(0, np.nan)
    panel["rsi_overbought"] = (panel["mr_rsi_raw"] > 70.0).astype(float)
    vol_expansion = (vol_5_raw / vol_20_raw.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    panel["vol_expansion"] = vol_expansion
    panel["momentum_acceleration"] = panel["ret_5d"] - panel["ret_10d"]
    d_ret_lag = _group_shift(daily_return, tick, 1)
    down_v = volume.where(d_ret_lag < 0.0, 0.0).groupby(tick, sort=False).transform(lambda s: s.rolling(20, min_periods=5).sum())
    up_v = volume.where(d_ret_lag > 0.0, 0.0).groupby(tick, sort=False).transform(lambda s: s.rolling(20, min_periods=5).sum())
    panel["down_up_vol_ratio"] = (down_v / up_v.replace(0, np.nan)).fillna(1.0)
    panel["short_term_reversal"] = (-_group_shift(panel["ret_5d"], tick, 1)).clip(-0.5, 0.5)
    panel["dist_from_52w_low"] = ((close - min_52w) / min_52w.clip(lower=1e-6)).clip(0.0, 10.0)
    panel["nearness_52w_low"] = 1.0 / (1.0 + panel["dist_from_52w_low"])
    panel["liquidity_stress"] = _timeseries_zscore(((high_px - low_px) / close.replace(0, np.nan)).clip(0.0, 0.25), tick).fillna(0.0)
    panel["beta_adj_momentum"] = (panel["ret_20d"] / panel["capm_beta"].clip(lower=0.3, upper=3.0)).clip(-2.0, 2.0)
    panel["nearness_52w_high"] = (close / max_52w.replace(0, np.nan)).clip(0.0, 1.0).fillna(0.5)
    vol_pct = _group_rolling_rank_pct(vol_20_raw, tick, window=252, min_periods=60)
    panel["low_vol_score"] = (1.0 - vol_pct).fillna(0.5)
    roll_mean_60 = _group_rolling_stat(daily_return, tick, window=60, min_periods=20, stat="mean")
    roll_std_60 = _group_rolling_stat(daily_return, tick, window=60, min_periods=20, stat="std").replace(0, np.nan).fillna(1e-6)
    panel["quality_score"] = (roll_mean_60 / roll_std_60 * np.sqrt(252.0)).clip(-5.0, 5.0)
    panel["momentum_1m_skip_eom"] = (close.groupby(tick, sort=False).shift(5) / close.groupby(tick, sort=False).shift(26).replace(0, np.nan) - 1.0).clip(-1.0, 4.0).fillna(0.0)
    turnover_raw = (volume / _group_rolling_stat(volume, tick, window=252, min_periods=60, stat="mean").replace(0, np.nan)).clip(0.0, 20.0)
    panel["turnover_pct_rank"] = _group_rolling_rank_pct(turnover_raw, tick, window=63, min_periods=21).fillna(0.5).clip(0.0, 1.0)
    to_centered = panel["turnover_pct_rank"] * 2.0 - 1.0
    panel["short_term_momentum_score"] = (panel["momentum_1m_skip_eom"] * to_centered).clip(-2.0, 2.0).fillna(0.0)
    panel["high_vol_reversal_flag"] = (vol_pct > 0.67).astype(float).fillna(0.0)

    fundamentals = _load_batched_fundamentals(
        panel,
        wrds_username=wrds_username,
        cache_dir=cache_dir,
        cache_ttl_days=cache_ttl_days,
        strict_fundamentals=bool(strict_fundamentals),
    )
    if not fundamentals.empty:
        panel = panel.merge(fundamentals, on=["date", "ticker"], how="left")
        fund_cols = [c for c in fundamentals.columns if c not in {"date", "ticker"}]
        # P10 Institutional fix: missing fundamentals ≠ zero.
        # Use cross-sectional median fill instead of zero-fill to avoid structural
        # bias against small caps and new IPOs.  Compute fundamental_coverage
        # BEFORE filling — a single 0-1 score per stock-date that tells the model
        # how much fundamental data is available.  This replaces the naive approach
        # of 22 individual {col}_missing indicator columns, which introduce
        # spurious per-column correlations and inflate feature dimensionality.
        missing_flags = []
        for col in fund_cols:
            if col in panel.columns:
                panel[col] = pd.to_numeric(panel[col], errors="coerce")
                missing_flags.append(panel[col].isna())

        if missing_flags:
            missing_df = pd.concat(missing_flags, axis=1)
            panel["fundamental_coverage"] = 1.0 - missing_df.mean(axis=1)

        for col in fund_cols:
            if col in panel.columns:
                cs_median = panel.groupby("date")[col].transform("median")
                panel[col] = panel[col].fillna(cs_median)

    near_high_risk = _group_shift(panel["nearness_52w_high"], tick, 1).fillna(0.5).clip(0.0, 1.0)
    positive_momentum_risk = _group_shift(panel["ret_20d"], tick, 1).clip(lower=0.0, upper=0.50).fillna(0.0) / 0.50
    vol_expansion_risk = ((_group_shift(panel["vol_expansion"], tick, 1) - 1.0) / 3.0).clip(0.0, 1.0).fillna(0.0)
    volume_pressure_risk = ((_group_shift(panel["relative_volume"], tick, 1).fillna(0.0) + 4.0) / 8.0).clip(0.0, 1.0)
    crowding_col = panel["borrow_crowding_risk"] if "borrow_crowding_risk" in panel.columns else pd.Series(0.0, index=panel.index)
    short_int_col = panel["short_interest_ratio"] if "short_interest_ratio" in panel.columns else pd.Series(0.0, index=panel.index)
    crowding_risk = pd.to_numeric(crowding_col, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    short_interest_risk = pd.to_numeric(short_int_col, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    panel["short_squeeze_risk"] = (
        0.25 * near_high_risk
        + 0.20 * positive_momentum_risk
        + 0.20 * vol_expansion_risk
        + 0.15 * volume_pressure_risk
        + 0.10 * crowding_risk
        + 0.10 * short_interest_risk
    ).clip(0.0, 1.0)
    panel["hard_short_squeeze_filter"] = (panel["short_squeeze_risk"] > 0.75).astype(float)

    panel["cs_momentum_raw"] = _group_shift(panel["momentum_6m"], tick, 21).fillna(0.0)
    panel = panel[(panel["date"] >= start_ts) & (panel["date"] <= end_ts)].copy()
    if panel.empty:
        return pd.DataFrame()

    panel = panel.dropna(subset=["forward_return", "ret_5d", "ret_10d", "rolling_vol_10", "rolling_vol_20", "relative_volume"])
    if panel.empty:
        return pd.DataFrame()

    panel["vol_rank"] = panel.groupby("date")["vol_20_simple"].rank(pct=True, method="average")
    panel["vol_rank"] = panel.groupby("ticker", sort=False)["vol_rank"].shift(1).fillna(0.5)
    panel = apply_cross_sectional_zscore_columns(panel, ["ret_5d", "ret_10d", "rolling_vol_20", "rolling_vol_60", "volume_zscore"])
    panel = _attach_mr_cross_sectional_zscore_shifted(panel)

    macro = _build_macro_series(wrds_panel, holding_period=holding_period)
    for col in ("vix_zscore", "vol_spike", "vix_term_zscore", "credit_spread_proxy", "yield_curve_slope"):
        ser = macro[col]
        panel[col] = panel["date"].map(ser.to_dict()).astype(float).fillna(0.0) if not ser.empty else 0.0
    panel = apply_cross_sectional_zscore_columns(panel, ["vix_zscore", "vol_spike"])

    panel = compute_sector_relative_shifted_cs_long(panel)
    panel["sector_relative_20d"] = pd.to_numeric(panel.get("sector_relative_20d"), errors="coerce").fillna(0.0)
    panel["sector_relative_60d"] = pd.to_numeric(panel.get("sector_relative_60d"), errors="coerce").fillna(0.0)
    panel["industry_relative_reversal"] = (-(panel["ret_20d"] - panel["sector_relative_20d"])).clip(-2.0, 2.0).fillna(0.0)
    panel["cs_momentum_percentile"] = panel.groupby("date")["cs_momentum_raw"].rank(pct=True, method="average").fillna(0.5)

    unique_dates = pd.to_datetime(panel["date"].unique())
    regime_series = get_regime_series_for_dates(
        unique_dates,
        start_date,
        end_date,
        data_provider="wrds",
        cache_dir=cache_dir,
        cache_ttl_days=cache_ttl_days,
        wrds_username=wrds_username,
        wrds_ticker_to_permno=wrds_ticker_to_permno,
    )
    regime_map = regime_series.to_dict()
    panel["regime_label"] = panel["date"].map(regime_map).fillna("Normal")
    panel["is_high_vol_regime"] = panel["regime_label"].isin(["HighVol"]).astype(float)

    try:
        cost_model = TransactionCostModel()
        round_trip_frac = 2.0 * float(cost_model.cost_fraction())
        panel["expected_round_trip_cost_frac"] = round_trip_frac
    except Exception:
        pass

    if not macro["spy_forward"].empty:
        panel["spy_forward_5d"] = panel["date"].map(macro["spy_forward"].to_dict()).astype(float)
        panel["forward_return_excess"] = panel["forward_return"] - panel["spy_forward_5d"]
    else:
        panel["spy_forward_5d"] = np.nan
        panel["forward_return_excess"] = panel["forward_return"]

    precomputed = (
        set(_MR_RAW_TO_OUT.values())
        | {"sector_relative_20d", "sector_relative_60d"}
        | set(_CS_Z_PANEL_INPLACE_COLS)
        | {"vix_term_zscore", "vol_rank", "vol_20_simple"}
    )
    panel = attach_cross_sectional_zscore_suffix_block(panel, exclude_columns=frozenset(precomputed))

    # P10 Institutional fix: fundamental features must enter models as CS-zscored
    # values, not raw. Raw fundamentals live on incomparable scales (f_score 0-5,
    # accruals -1 to 1, gross_margin 0-100%) and would dominate gradient-based
    # learners. CS-zscored variants are appended with a _cs_z suffix by
    # attach_cross_sectional_zscore_suffix_block above.
    #
    # Strategy: drop the raw column, then rename the _cs_z column to the canonical
    # name. Downstream FEATURE_SPECS/feature_admission/feat_cols filtering all
    # reference canonical names (e.g. "f_score"), so the rename keeps the entire
    # pipeline unaware that a normalization swap happened.
    _FUNDAMENTAL_RAW_COLS = frozenset({
        "f_score", "accruals_ratio", "roa", "delta_roa", "delta_leverage",
        "gross_margin", "delta_gross_margin", "operating_margin",
        "delta_operating_margin", "margin_deterioration", "debt_to_assets",
        "total_debt_to_assets", "weak_profitability", "share_issuance_growth",
        "dilution_pressure", "filing_delay_days", "late_filing_flag",
        "restatement_like_flag", "fundamental_deterioration_score",
        "short_interest_ratio", "days_to_cover", "borrow_crowding_risk",
    })
    _rename_map: dict[str, str] = {}
    for col in _FUNDAMENTAL_RAW_COLS:
        cs_col = f"{col}_cs_z"
        if col in panel.columns and cs_col in panel.columns:
            _rename_map[cs_col] = col
    if _rename_map:
        # Step 1: drop raw fundamental columns (keep only _cs_z versions)
        panel = panel.drop(columns=[c for c in _FUNDAMENTAL_RAW_COLS if c in panel.columns], errors="ignore")
        # Step 2: rename _cs_z → canonical name so FEATURE_SPECS/admission match
        panel = panel.rename(columns=_rename_map)

    for col in feature_columns_to_zero_for_ablation():
        if col in panel.columns:
            panel[col] = 0.0

    panel["direction"] = np.sign(pd.to_numeric(panel["forward_return"], errors="coerce").fillna(0.0)).astype(int)
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
            "forward_return_risk_adj",
            "forward_return_excess",
            "direction",
        ]
        keep: list[str] = []
        for col in always_keep + subset:
            if col in panel.columns and col not in keep:
                keep.append(col)
        panel = panel[keep].copy()

    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)
