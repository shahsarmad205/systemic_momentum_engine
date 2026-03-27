"""
Factor Exposure Neutralization
================================
Removes market beta, sector, and size exposures from signals via cross-sectional
regression so that the residual represents idiosyncratic alpha.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Avoid importing main; replicate signal classification
def _classify_signal(adjusted_score: float) -> str:
    if adjusted_score > 0:
        return "Bullish"
    if adjusted_score < 0:
        return "Bearish"
    return "Neutral"


def _row_to_dict(row: Any) -> dict:
    if hasattr(row, "to_dict"):
        return row.to_dict()
    return dict(row)


class FactorNeutralizer:
    """
    Remove factor exposures from signals using cross-sectional regression per day.

    For each date: signal = X @ beta + residual; neutralized_signal = residual.
    Factors: market beta (rolling vs market index), sector (one-hot), size (log(price*volume)).
    """

    def __init__(
        self,
        neutralize_market_beta: bool = True,
        neutralize_sector: bool = True,
        neutralize_size: bool = True,
        market_index: str = "SPY",
        rolling_window: int = 60,
        min_observations: int = 10,
        dev_mode_limit: int | None = None,
    ):
        self.neutralize_market_beta = neutralize_market_beta
        self.neutralize_sector = neutralize_sector
        self.neutralize_size = neutralize_size
        self.market_index = market_index
        self.rolling_window = int(rolling_window)
        self.min_observations = int(min_observations)
        self.dev_mode_limit = dev_mode_limit  # if set, use only first N tickers for faster testing

    def _compute_market_beta(
        self,
        price_data: dict[str, pd.DataFrame],
        as_of_date: pd.Timestamp,
        tickers: list[str],
    ) -> dict[str, float]:
        """Rolling beta vs market index (e.g. SPY) over rolling_window days."""
        out: dict[str, float] = {}
        if self.market_index not in price_data or not self.neutralize_market_beta:
            return {t: 0.0 for t in tickers}
        market_df = price_data[self.market_index]
        if "Close" not in market_df.columns or as_of_date not in market_df.index:
            return {t: 0.0 for t in tickers}
        market = market_df["Close"].loc[market_df.index <= as_of_date].tail(self.rolling_window + 1)
        market_ret = market.pct_change().dropna()
        if len(market_ret) < 2:
            return {t: 0.0 for t in tickers}
        var_m = float(market_ret.var())
        if var_m < 1e-12:
            return {t: 0.0 for t in tickers}
        for ticker in tickers:
            if ticker not in price_data or ticker == self.market_index:
                out[ticker] = 0.0
                continue
            df = price_data[ticker]
            if "Close" not in df.columns or as_of_date not in df.index:
                out[ticker] = 0.0
                continue
            close = df["Close"].loc[df.index <= as_of_date].tail(self.rolling_window + 1)
            ret = close.pct_change().dropna()
            common = ret.index.intersection(market_ret.index)
            if len(common) < 2:
                out[ticker] = 0.0
                continue
            cov = float(ret.reindex(common).fillna(0).cov(market_ret.reindex(common).fillna(0)))
            out[ticker] = cov / var_m
        return out

    def _compute_sector_dummies(self, tickers: list[str]) -> pd.DataFrame:
        """One-hot sector encoding; drop first to avoid collinearity."""
        try:
            from utils.sectors import SECTOR_MAP
        except ImportError:
            return pd.DataFrame(index=tickers)
        sectors = [SECTOR_MAP.get(t, "Other") for t in tickers]
        df = pd.get_dummies(pd.Series(sectors, index=tickers), prefix="sector", drop_first=True)
        return df.reindex(tickers).fillna(0)

    def _compute_size_factor(
        self,
        price_data: dict[str, pd.DataFrame],
        as_of_date: pd.Timestamp,
        tickers: list[str],
    ) -> dict[str, float]:
        """Size proxy: log(price * average_volume) or log(price) if no volume."""
        out: dict[str, float] = {}
        for ticker in tickers:
            if ticker not in price_data:
                out[ticker] = 0.0
                continue
            df = price_data[ticker]
            if as_of_date not in df.index:
                out[ticker] = 0.0
                continue
            row = df.loc[as_of_date]
            price = float(row.get("Close", 1.0) or 1.0)
            if "Volume" in df.columns and df["Volume"].dtype in (np.float64, np.int64, float, int):
                vol = float(row.get("Volume", 1.0) or 1.0)
                # Optional: use rolling mean volume for stability
                vol_series = df["Volume"].loc[df.index <= as_of_date].tail(21)
                if len(vol_series) >= 5:
                    vol = float(vol_series.mean())
                size = np.log(max(price * vol, 1.0))
            else:
                size = np.log(max(price, 1e-6))
            out[ticker] = size
        return out

    def neutralize(
        self,
        date: pd.Timestamp,
        daily_signals: list[tuple[str, Any]],
        price_data: dict[str, pd.DataFrame],
        collect_diagnostics: bool = True,
    ) -> tuple[list[tuple[str, dict]], dict[str, Any] | None]:
        """
        Run cross-sectional regression: signal = X*beta + residual; return (ticker, row) with
        adjusted_score = residual and signal reclassified. No look-ahead: factors use data up to date.

        Returns
        -------
        neutralized_list : list of (ticker, row_dict) with adjusted_score = residual, signal updated
        diagnostics : dict with correlations before/after (if collect_diagnostics), else None
        """
        if not daily_signals:
            return [], None
        if self.dev_mode_limit is not None and len(daily_signals) > self.dev_mode_limit:
            daily_signals = daily_signals[: self.dev_mode_limit]
        if len(daily_signals) < self.min_observations:
            return list(daily_signals), None
        tickers = [t for t, _ in daily_signals]
        y = np.array([float(_row_to_dict(r).get("adjusted_score", 0.0)) for _, r in daily_signals])
        if len(tickers) != len(y):
            return list(daily_signals), None
        X_list: list[np.ndarray] = []
        col_names: list[str] = []
        if self.neutralize_market_beta:
            beta_map = self._compute_market_beta(price_data, date, tickers)
            X_list.append(np.array([beta_map.get(t, 0.0) for t in tickers]))
            col_names.append("market_beta")
        if self.neutralize_sector:
            sector_df = self._compute_sector_dummies(tickers)
            if not sector_df.empty:
                for c in sector_df.columns:
                    X_list.append(sector_df[c].values)
                    col_names.append(c)
        if self.neutralize_size:
            size_map = self._compute_size_factor(price_data, date, tickers)
            X_list.append(np.array([size_map.get(t, 0.0) for t in tickers]))
            col_names.append("size_factor")
        if not X_list:
            return list(daily_signals), None
        X = np.column_stack(X_list)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        n, k = X.shape
        if n < self.min_observations or k >= n:
            return list(daily_signals), None
        # Add intercept
        X_with_const = np.column_stack([np.ones(n), X])
        try:
            beta, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
            if beta.size == 0:
                return list(daily_signals), None
            fitted = X_with_const @ beta
            residuals = y - fitted
        except Exception:
            return list(daily_signals), None
        diagnostics = None
        if collect_diagnostics and X_list:
            corr_before = {}
            for i, name in enumerate(col_names):
                if i < len(X_list):
                    xi = X_list[i]
                    if np.std(xi) > 1e-10 and np.std(y) > 1e-10:
                        corr_before[name] = float(np.corrcoef(xi, y)[0, 1])
                    else:
                        corr_before[name] = 0.0
            corr_after = {}
            for i, name in enumerate(col_names):
                if i < len(X_list):
                    xi = X_list[i]
                    if np.std(xi) > 1e-10 and np.std(residuals) > 1e-10:
                        corr_after[name] = float(np.corrcoef(xi, residuals)[0, 1])
                    else:
                        corr_after[name] = 0.0
            diagnostics = {"date": str(date), "corr_before": corr_before, "corr_after": corr_after}
        out: list[tuple[str, dict]] = []
        for idx, (ticker, row) in enumerate(daily_signals):
            row_dict = _row_to_dict(row)
            if idx < len(residuals):
                res = float(residuals[idx])
            else:
                res = float(row_dict.get("adjusted_score", 0.0))
            row_dict["adjusted_score"] = round(res, 6)
            row_dict["signal"] = _classify_signal(res)
            out.append((ticker, row_dict))
        return out, diagnostics


def write_exposure_diagnostics(
    diagnostics_list: list[dict],
    output_path: str = "output/research/factor_exposures.csv",
) -> None:
    """Flatten per-date diagnostics into a CSV: date, factor, corr_before, corr_after."""
    if not diagnostics_list:
        return
    rows: list[dict] = []
    for d in diagnostics_list:
        date = d.get("date", "")
        for factor, corr_b in (d.get("corr_before") or {}).items():
            corr_a = (d.get("corr_after") or {}).get(factor, None)
            rows.append({"date": date, "factor": factor, "corr_before": corr_b, "corr_after": corr_a})
    if not rows:
        return
    df = pd.DataFrame(rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_neutralization_report(
    diagnostics_list: list[dict],
    output_path: str = "output/research/factor_neutralization_report.json",
    ic_before: float | None = None,
    ic_after: float | None = None,
) -> None:
    """Summarize factor correlations before/after and optional IC; write JSON."""
    report: dict[str, Any] = {
        "n_dates": len(diagnostics_list),
        "factor_correlations_before": {},
        "factor_correlations_after": {},
    }
    if ic_before is not None:
        report["signal_ic_before_neutralization"] = ic_before
    if ic_after is not None:
        report["signal_ic_after_neutralization"] = ic_after
    if not diagnostics_list:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        return
    factors = set()
    for d in diagnostics_list:
        factors.update((d.get("corr_before") or {}).keys())
    for f in factors:
        before_vals = [
            (d.get("corr_before") or {}).get(f)
            for d in diagnostics_list
            if (d.get("corr_before") or {}).get(f) is not None
        ]
        after_vals = [
            (d.get("corr_after") or {}).get(f)
            for d in diagnostics_list
            if (d.get("corr_after") or {}).get(f) is not None
        ]
        report["factor_correlations_before"][f] = float(np.mean(before_vals)) if before_vals else None
        report["factor_correlations_after"][f] = float(np.mean(after_vals)) if after_vals else None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


__all__ = ["FactorNeutralizer", "write_exposure_diagnostics", "write_neutralization_report"]
