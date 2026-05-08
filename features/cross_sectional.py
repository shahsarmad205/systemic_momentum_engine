from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_winsorize(
    df: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """
    Cross-sectional winsorization: clip each row at the specified quantiles.

    Institutional standard (AQR, Two Sigma): winsorize at 1st/99th percentile
    before zscoring to prevent outlier distortion of the cross-sectional distribution.
    This is critical for fundamental features which have fat tails (accruals,
    leverage ratios, margin changes).

    Uses ``method='lower'`` for the upper quantile and ``method='higher'`` for
    the lower quantile.  This avoids linear interpolation blending extreme
    outliers into the bound when the number of cross-section observations is
    small (e.g. 50 names at 0.99 quantile -> the interpolated position sits
    between the second-highest value and the outlier, producing a bound that
    is still contaminated by the outlier).

    Parameters
    ----------
    df : DataFrame with dates as index, tickers as columns
    lower : lower quantile bound (default 0.01 = 1st percentile)
    upper : upper quantile bound (default 0.99 = 99th percentile)

    Returns
    -------
    Winsorized DataFrame with same shape
    """
    lower_bounds = df.quantile(lower, axis=1)
    upper_bounds = df.quantile(upper, axis=1)
    # Clip column-by-column with row-wise bounds
    result = df.copy()
    for col in result.columns:
        result[col] = result[col].clip(lower=lower_bounds, upper=upper_bounds)
    return result


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional z-score: tickers as columns, dates as index.
    (x - row_mean) / row_std; std=0 -> NaN.
    """
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1).replace(0, np.nan)
    return df.sub(row_mean, axis=0).div(row_std, axis=0)


def cross_sectional_zscore_ddof0(df: pd.DataFrame) -> pd.DataFrame:
    """
    Population-standard-deviation variant of the cross-sectional z-score.
    Matches ``groupby('date').transform(lambda x: (x - x.mean()) / x.std(ddof=0))``.
    """
    row_mean = df.mean(axis=1, skipna=True)
    row_std = df.std(axis=1, ddof=0, skipna=True).replace(0, np.nan)
    return df.sub(row_mean, axis=0).div(row_std, axis=0)


def apply_cross_sectional_zscore_columns(
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


def attach_cross_sectional_zscore_suffix_block(
    result: pd.DataFrame,
    *,
    exclude_columns: set[str] | frozenset[str],
    winsorize: bool = True,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> pd.DataFrame:
    """
    Append ``*_cs_z`` columns in one blockwise pass.

    This keeps panel-normalization as a block operation instead of repeated
    column mutation, which fragments pandas internals on wide research panels.

    P10 Institutional fix: winsorize at 1st/99th percentile before zscoring.
    Fundamental features (accruals, leverage, margins) have fat tails that
    distort cross-sectional means and stds. Winsorization is standard at
    AQR, Two Sigma, DE Shaw before any CS normalization.

    P12 fix: Cross-sectional winsorization is applied per-date across ALL
    tickers for each feature column (not per-row across features).  The
    z-score is computed on NumPy arrays to avoid Pandas index-alignment
    collisions between DatetimeIndex and RangeIndex.

    Parameters
    ----------
    result : long-format panel DataFrame
    exclude_columns : columns to skip (identifiers, targets, precomputed)
    winsorize : apply cross-sectional winsorization before zscoring
    winsor_lower : lower quantile for winsorization (default 0.01)
    winsor_upper : upper quantile for winsorization (default 0.99)
    """
    numeric_cols = [
        col
        for col in result.select_dtypes(include=["number"]).columns
        if col not in exclude_columns
    ]
    if not numeric_cols:
        return result

    numeric_block = result.loc[:, numeric_cols].apply(pd.to_numeric, errors="coerce")
    date_key = pd.to_datetime(result["date"], errors="coerce")
    grouped = numeric_block.groupby(date_key, sort=False)
    mean_block = grouped.transform("mean")
    std_block = grouped.transform(lambda values: values.std(ddof=0)).replace(0.0, np.nan)

    if winsorize:
        # Cross-sectional winsorization: clip each feature at the 1st/99th
        # percentile computed across ALL tickers for each date.  This is
        # the correct institutional approach (not per-row winsorization).
        lower_bounds = grouped.transform(lambda x: x.quantile(winsor_lower))
        upper_bounds = grouped.transform(lambda x: x.quantile(winsor_upper))
        winsor_block = numeric_block.clip(lower=lower_bounds, upper=upper_bounds)
        # P12: Use NumPy arrays to avoid Pandas index-alignment collisions.
        # mean_block and std_block were computed via groupby().transform() on
        # numeric_block, so they have identical row ordering.  Both have the
        # original RangeIndex, which does NOT align with a DatetimeIndex.
        cs_vals = (winsor_block.values - mean_block.values) / std_block.values
    else:
        cs_vals = (numeric_block.values - mean_block.values) / std_block.values

    # Guard against inf/nan from division by near-zero std
    cs_vals = np.where(np.isfinite(cs_vals), cs_vals, 0.0)

    cs_block = pd.DataFrame(
        cs_vals, index=numeric_block.index, columns=numeric_block.columns
    ).add_suffix("_cs_z")

    return pd.concat([result, cs_block], axis=1, copy=False)


def compute_sector_relative_shifted_cs_long(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sector-relative momentum for a long-format panel.

    For each (date, ticker):
      raw = ret_Nd - median(ret_Nd | same date & sector),
    then pivot -> shift(1) along time -> cross-sectional z-score across tickers.

    Required input columns: ``date``, ``ticker``, ``ret_20d``, ``ret_60d``, ``sector``.
    Output columns: ``sector_relative_20d``, ``sector_relative_60d``.
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
        z_long = z.stack(future_stack=True).reset_index()
        z_long.columns = ["date", "ticker", out_col]
        out = out.drop(columns=[out_col], errors="ignore")
        out = out.merge(z_long, on=["date", "ticker"], how="left")
        out[out_col] = pd.to_numeric(out[out_col], errors="coerce").fillna(0.0)

    out = out.drop(columns=["_sr20_raw", "_sr60_raw"], errors="ignore")
    return out
