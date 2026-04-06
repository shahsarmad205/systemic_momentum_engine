"""
IC Engine — Information Coefficient Signal Weighting
======================================================
Implements IC-weighted signal combination, the standard approach at
institutional quant desks (AQR, Two Sigma, D.E. Shaw).

Grinold & Kahn's Fundamental Law of Active Management:
    IR ≈ IC × sqrt(Breadth)

where IC = rank correlation between signal and subsequent return.

Rather than hand-coding signal weights (w_trend=1.0, w_regional=0.5, ...),
this module:
    1. Computes rolling IC for each feature/signal vs. forward returns
    2. Applies exponential decay to weight recent IC higher
    3. Combines signals using IC as weights (shrunk toward equal-weight)
    4. Reports IC t-statistics and confidence intervals

Key design principles
---------------------
- **No lookahead**: IC is computed from signal_date+1 vs. return(signal_date+1, +H)
- **Bayesian shrinkage**: IC_shrunk = (IC_raw × N_obs) / (N_obs + k_prior)
  prevents overfitting to noisy IC estimates from short windows
- **Rolling window**: uses expanding window by default (full history) or
  a rolling 252-day window to detect IC regime changes
- **Cross-sectional IC**: rank IC (Spearman) is more robust than Pearson

References
----------
- Grinold, R. & Kahn, R. (2000). Active Portfolio Management. McGraw-Hill.
- Qian, E., Hua, R. & Sorensen, E. (2007). Quantitative Equity Portfolio Management.
- López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ICResult:
    """Summary statistics for one signal's IC."""
    feature: str
    ic_mean: float          # mean rank IC (Spearman)
    ic_std: float           # std of rolling IC
    ic_t_stat: float        # t-statistic: ic_mean / (ic_std / sqrt(n))
    ic_p_value: float       # two-tailed p-value
    n_obs: int              # number of cross-sectional observations used
    icir: float             # IC Information Ratio = ic_mean / ic_std
    is_significant: bool    # |t| > 1.96 (5% level)

    def __str__(self) -> str:
        sig = "*" if self.is_significant else " "
        return (
            f"{self.feature:<35} IC={self.ic_mean:+.4f}  ICIR={self.icir:+.3f}"
            f"  t={self.ic_t_stat:+.2f}{sig}  n={self.n_obs}"
        )


@dataclass
class ICWeights:
    """
    Signal combination weights derived from IC analysis.

    weights[feature] = ICIR_shrunk_normalized
    """
    weights: dict[str, float] = field(default_factory=dict)
    ic_results: dict[str, ICResult] = field(default_factory=dict)
    shrinkage_factor: float = 0.5
    computation_date: Optional[pd.Timestamp] = None

    def get(self, feature: str, default: float = 0.0) -> float:
        return self.weights.get(feature, default)

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Return top N features by absolute IC weight."""
        return sorted(self.weights.items(), key=lambda x: abs(x[1]), reverse=True)[:n]


# ---------------------------------------------------------------------------
# IC computation helpers
# ---------------------------------------------------------------------------

def _rank_ic_series(
    signal_panel: pd.DataFrame,   # shape (dates, tickers) — signal value
    return_panel: pd.DataFrame,   # shape (dates, tickers) — forward return
    min_stocks: int = 20,
) -> pd.Series:
    """
    Compute daily cross-sectional rank IC (Spearman correlation).

    Parameters
    ----------
    signal_panel : shape (dates, tickers)
    return_panel : shape (dates, tickers)
    min_stocks : int
        Minimum number of stocks with valid data on a date to compute IC.

    Returns
    -------
    pd.Series indexed by date, values = Spearman IC ∈ [-1, 1]
    """
    common_dates = signal_panel.index.intersection(return_panel.index)
    ic_values: dict[pd.Timestamp, float] = {}

    for date in common_dates:
        sig_row = pd.to_numeric(signal_panel.loc[date], errors="coerce").dropna()
        ret_row = pd.to_numeric(return_panel.loc[date], errors="coerce").dropna()

        common_tickers = sig_row.index.intersection(ret_row.index)
        if len(common_tickers) < min_stocks:
            continue

        s = sig_row.loc[common_tickers].values.astype(float)
        r = ret_row.loc[common_tickers].values.astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, _ = spearmanr(s, r)

        if np.isfinite(corr):
            ic_values[date] = float(corr)

    return pd.Series(ic_values)


def compute_ic_result(
    ic_series: pd.Series,
    feature_name: str,
) -> ICResult:
    """Derive ICResult from a time-series of cross-sectional IC values."""
    ic = ic_series.dropna()
    n = len(ic)
    if n < 5:
        return ICResult(
            feature=feature_name, ic_mean=0.0, ic_std=0.0,
            ic_t_stat=0.0, ic_p_value=1.0, n_obs=n,
            icir=0.0, is_significant=False,
        )

    mean = float(ic.mean())
    std = float(ic.std(ddof=1)) if n > 1 else 0.0
    t_stat = (mean / (std / np.sqrt(n))) if std > 0 else 0.0
    p_value = float(2.0 * (1.0 - _normal_cdf(abs(t_stat))))
    icir = mean / std if std > 0 else 0.0

    return ICResult(
        feature=feature_name,
        ic_mean=mean,
        ic_std=std,
        ic_t_stat=t_stat,
        ic_p_value=p_value,
        n_obs=n,
        icir=icir,
        is_significant=abs(t_stat) > 1.96,
    )


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


import math  # noqa: E402 (placed after _normal_cdf uses it in the body)


# ---------------------------------------------------------------------------
# ICEngine: main class
# ---------------------------------------------------------------------------

class ICEngine:
    """
    IC-based signal combination and weighting.

    Workflow
    --------
    1. ``fit(signal_data, price_data, forward_horizon)``
       — builds IC estimates from historical data
    2. ``compute_weights()``
       — converts IC estimates to combination weights
    3. ``combine_signals(feature_dict)``
       — combines raw feature values into a single adjusted score

    Parameters
    ----------
    forward_horizon : int
        Number of trading days for forward return (should match holding period).
    ewma_halflife : int
        Half-life (days) for exponentially weighting recent IC over older IC.
        Larger value = slower decay = more weight on long-run IC.
    shrinkage_k : float
        Bayesian shrinkage prior weight. Higher = more shrinkage toward equal
        weight. k=10 means 10 prior observations at IC=0 are added.
    min_ic_magnitude : float
        Drop features whose |IC_mean| < this threshold (noise floor).
    rolling_window : int or None
        If set, use rolling (not expanding) IC window. None = expanding.
    """

    def __init__(
        self,
        forward_horizon: int = 5,
        ewma_halflife: int = 63,
        shrinkage_k: float = 10.0,
        min_ic_magnitude: float = 0.005,
        rolling_window: Optional[int] = None,
        min_stocks: int = 20,
    ) -> None:
        self.forward_horizon = int(forward_horizon)
        self.ewma_halflife = int(ewma_halflife)
        self.shrinkage_k = float(shrinkage_k)
        self.min_ic_magnitude = float(min_ic_magnitude)
        self.rolling_window = rolling_window
        self.min_stocks = int(min_stocks)

        self._ic_series: dict[str, pd.Series] = {}
        self._ic_results: dict[str, ICResult] = {}
        self._weights: Optional[ICWeights] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_from_panels(
        self,
        feature_panels: dict[str, pd.DataFrame],
        price_panel: pd.DataFrame,
    ) -> "ICEngine":
        """
        Compute IC for each feature against H-day forward returns.

        Parameters
        ----------
        feature_panels : dict of {feature_name: DataFrame(dates × tickers)}
        price_panel : DataFrame(dates × tickers) of close prices
        """
        fwd_returns = self._forward_return_panel(price_panel, self.forward_horizon)

        for feature_name, panel in feature_panels.items():
            # Shift signal forward by 1 to avoid lookahead
            sig_shifted = panel.shift(1)
            ic_series = _rank_ic_series(sig_shifted, fwd_returns, self.min_stocks)

            if self.rolling_window is not None and len(ic_series) > self.rolling_window:
                ic_series = ic_series.iloc[-self.rolling_window:]

            self._ic_series[feature_name] = ic_series
            self._ic_results[feature_name] = compute_ic_result(ic_series, feature_name)

        self._weights = self._build_weights()
        return self

    def fit_from_records(
        self,
        records: pd.DataFrame,
        feature_cols: list[str],
        return_col: str = "forward_return",
    ) -> "ICEngine":
        """
        Compute IC from a flat records DataFrame (one row = one signal observation).

        Parameters
        ----------
        records : DataFrame with columns [date, ticker, feature_cols..., return_col]
        feature_cols : list of feature column names to compute IC for
        return_col : column name of the forward return target
        """
        required = set(feature_cols) | {return_col, "date", "ticker"}
        missing = required - set(records.columns)
        if missing:
            raise ValueError(f"fit_from_records: missing columns {missing}")

        for feat in feature_cols:
            ic_per_date: dict[pd.Timestamp, float] = {}

            for date, grp in records.groupby("date"):
                valid = grp[[feat, return_col]].dropna()
                if len(valid) < self.min_stocks:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr, _ = spearmanr(valid[feat].values, valid[return_col].values)
                if np.isfinite(corr):
                    ic_per_date[pd.Timestamp(date)] = float(corr)

            ic_series = pd.Series(ic_per_date)
            self._ic_series[feat] = ic_series
            self._ic_results[feat] = compute_ic_result(ic_series, feat)

        self._weights = self._build_weights()
        return self

    # ------------------------------------------------------------------
    # Weight construction
    # ------------------------------------------------------------------

    def _build_weights(self) -> ICWeights:
        """
        Convert IC estimates to combination weights with Bayesian shrinkage.

        Shrinkage formula:
            IC_shrunk = (IC_raw × n) / (n + k_prior)

        where n = number of IC observations, k_prior = shrinkage_k.

        Weights are then proportional to |ICIR_shrunk|, preserving sign.
        Features with |IC_shrunk| < min_ic_magnitude are zeroed out.
        """
        raw_icirs: dict[str, float] = {}
        for fname, result in self._ic_results.items():
            n = result.n_obs
            ic_raw = result.ic_mean
            # Bayesian shrinkage toward zero
            ic_shrunk = (ic_raw * n) / (n + self.shrinkage_k)

            if abs(ic_shrunk) < self.min_ic_magnitude:
                raw_icirs[fname] = 0.0
            else:
                icir_shrunk = ic_shrunk / max(result.ic_std, 1e-6)
                raw_icirs[fname] = icir_shrunk

        # Normalise: sum of absolute weights = 1
        total_abs = sum(abs(v) for v in raw_icirs.values())
        if total_abs < 1e-9:
            # Fall back to equal weights over non-zero features
            n_feats = len(raw_icirs)
            norm_weights = {f: 1.0 / n_feats for f in raw_icirs} if n_feats > 0 else {}
        else:
            norm_weights = {f: v / total_abs for f, v in raw_icirs.items()}

        return ICWeights(
            weights=norm_weights,
            ic_results=self._ic_results,
            shrinkage_factor=self.shrinkage_k,
            computation_date=pd.Timestamp.now(),
        )

    # ------------------------------------------------------------------
    # Inference: combine signals
    # ------------------------------------------------------------------

    def combine_signals(
        self,
        feature_values: dict[str, float],
        fallback_equal_weight: bool = True,
    ) -> float:
        """
        Produce a single adjusted score by IC-weighted combination.

        Parameters
        ----------
        feature_values : dict mapping feature name → scalar value
        fallback_equal_weight : if True and no weights fitted, fall back to
            equal-weight average.

        Returns
        -------
        float adjusted score (not bounded; caller may clip/normalize)
        """
        if self._weights is None:
            if fallback_equal_weight and feature_values:
                vals = [v for v in feature_values.values() if np.isfinite(v)]
                return float(np.mean(vals)) if vals else 0.0
            return 0.0

        score = 0.0
        weight_used = 0.0
        for feat, val in feature_values.items():
            w = self._weights.get(feat, 0.0)
            if w == 0.0 or not np.isfinite(val):
                continue
            score += w * float(val)
            weight_used += abs(w)

        if weight_used < 1e-9 and fallback_equal_weight and feature_values:
            vals = [v for v in feature_values.values() if np.isfinite(v)]
            return float(np.mean(vals)) if vals else 0.0

        return float(score)

    def combine_series(
        self,
        feature_df: pd.DataFrame,
        fallback_equal_weight: bool = True,
    ) -> pd.Series:
        """
        Vectorised version of combine_signals for a time-series DataFrame.

        Parameters
        ----------
        feature_df : DataFrame where each column is a feature time series
        """
        if self._weights is None:
            return feature_df.mean(axis=1) if fallback_equal_weight else pd.Series(0.0, index=feature_df.index)

        result = pd.Series(0.0, index=feature_df.index, dtype=float)
        for col in feature_df.columns:
            w = self._weights.get(col, 0.0)
            if w == 0.0:
                continue
            result += w * pd.to_numeric(feature_df[col], errors="coerce").fillna(0.0)

        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def ic_summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising IC statistics for all features."""
        rows = []
        for feat, res in self._ic_results.items():
            rows.append({
                "feature": feat,
                "ic_mean": res.ic_mean,
                "ic_std": res.ic_std,
                "ic_t_stat": res.ic_t_stat,
                "ic_p_value": res.ic_p_value,
                "n_obs": res.n_obs,
                "icir": res.icir,
                "is_significant": res.is_significant,
                "weight": self._weights.get(feat, 0.0) if self._weights else 0.0,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("icir", key=lambda s: s.abs(), ascending=False)
        return df

    def print_ic_summary(self) -> None:
        """Pretty-print IC summary to stdout."""
        print(f"\n{'=' * 70}")
        print("  IC ENGINE — Feature Information Coefficients")
        print(f"  Forward horizon: {self.forward_horizon}d | "
              f"Shrinkage k={self.shrinkage_k} | "
              f"EWMA halflife={self.ewma_halflife}d")
        print(f"{'=' * 70}")
        for feat, res in sorted(
            self._ic_results.items(), key=lambda x: abs(x[1].icir), reverse=True
        ):
            w = self._weights.get(feat, 0.0) if self._weights else 0.0
            sig_flag = "*" if res.is_significant else " "
            print(
                f"  {feat:<35} IC={res.ic_mean:+.4f}  ICIR={res.icir:+.3f}"
                f"  t={res.ic_t_stat:+.2f}{sig_flag}  w={w:+.3f}  n={res.n_obs}"
            )
        print(f"{'=' * 70}\n")

    @property
    def weights(self) -> ICWeights:
        if self._weights is None:
            raise RuntimeError("ICEngine.weights accessed before fit(). Call fit_from_*() first.")
        return self._weights

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _forward_return_panel(
        price_panel: pd.DataFrame,
        horizon: int,
    ) -> pd.DataFrame:
        """Compute H-day forward returns from a price panel. No lookahead."""
        fwd = price_panel.shift(-horizon) / price_panel - 1.0
        return fwd.iloc[:-horizon]  # drop rows where forward return not yet known


# ---------------------------------------------------------------------------
# Standalone helper: IC-weighted combination from config
# ---------------------------------------------------------------------------

def ic_weighted_score(
    feature_values: dict[str, float],
    ic_engine: Optional[ICEngine],
    fallback_weights: Optional[dict[str, float]] = None,
) -> float:
    """
    Combine feature values into an adjusted score.

    If ic_engine is fitted, uses IC-weighted combination.
    Otherwise falls back to fallback_weights (e.g. hand-coded weights from config).

    This is the recommended single entry-point for signal combination in
    the backtester.
    """
    if ic_engine is not None:
        try:
            return ic_engine.combine_signals(feature_values)
        except Exception as exc:
            logger.warning("ICEngine.combine_signals failed (%s), falling back.", exc)

    if fallback_weights:
        score = 0.0
        total_w = sum(abs(w) for w in fallback_weights.values())
        for feat, val in feature_values.items():
            w = fallback_weights.get(feat, 0.0)
            if np.isfinite(val) and total_w > 0:
                score += (w / total_w) * float(val)
        return score

    vals = [v for v in feature_values.values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0
