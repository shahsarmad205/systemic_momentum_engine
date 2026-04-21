"""
Risk Model
===========
Estimates the covariance matrix used inside the portfolio optimizer.

Supports:
  1. Sample covariance (rolling window)
  2. Ledoit-Wolf shrinkage (preferred — stable when T/N is small)

Also computes the eigenvalue-based effective N (participation ratio),
replacing the hardcoded avg_corr=0.3 assumption in metrics.py.

Reference:
  Ledoit & Wolf (2004) "A well-conditioned estimator for large-dimensional
  covariance matrices". Journal of Multivariate Analysis.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# RiskModel
# ------------------------------------------------------------------

class RiskModel:
    """
    Rolling covariance estimator for a dynamic universe of assets.

    Parameters
    ----------
    window : int
        Number of trading days to use for covariance estimation.
        Must be > N (number of assets) for full-rank matrix.
        Default 60d is the minimum practical window for ~20 assets.
    min_periods : int
        Minimum observations required; returns identity if below this.
    method : str
        'ledoit_wolf' (default, recommended) or 'sample'.
    annualize : bool
        If True, multiply daily cov by 252 to get annualised covariance.
        Default True so that lambda_risk is in annualised return units.
    """

    def __init__(
        self,
        window: int = 60,
        min_periods: int = 20,
        method: str = "ledoit_wolf",
        annualize: bool = True,
    ):
        self.window = int(window)
        self.min_periods = int(min_periods)
        self.method = str(method)
        self.annualize = bool(annualize)

    # ------------------------------------------------------------------
    # Core: fit covariance for a given returns matrix
    # ------------------------------------------------------------------

    def fit(self, returns_df: pd.DataFrame) -> np.ndarray:
        """
        Estimate covariance matrix from a (T × N) returns DataFrame.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Columns = tickers, rows = daily returns (already aligned + filled).
            Caller should pass tail(window) before calling.

        Returns
        -------
        cov : np.ndarray, shape (N, N), annualised if self.annualize.
        """
        N = len(returns_df.columns)
        T = len(returns_df)

        if T < self.min_periods or N < 2:
            # Fallback: identity with 20% annual vol
            return np.eye(N) * (0.20 ** 2)

        # ── Step 1: sanitise extreme and missing values ───────────────────
        # Factor neutralization on bad price data can produce Inf or returns
        # of ±100%+.  Clip to ±50% daily (>10σ for any normal equity) before
        # LedoitWolf — extreme finite values blow up the sample covariance and
        # produce an ill-conditioned matrix that overflows SLSQP's lstsq.
        X_full = (
            returns_df
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(-0.5, 0.5)   # ±50% daily return hard cap
            .values
        )

        # ── Step 2: drop near-zero-variance columns ───────────────────────
        # Columns with daily std < 1e-6 are either constant (data gap filled
        # with 0) or have negligible signal.  LedoitWolf passes them to scipy's
        # pseudoinverse which divides by ~0 singular value → overflow.
        # Threshold 1e-6 ≈ 0.1bp daily std; well below any real equity.
        col_std = X_full.std(axis=0)
        active_mask = col_std > 1e-6
        n_active = int(active_mask.sum())

        if n_active < 2:
            return np.eye(N) * (0.15 ** 2)

        X = X_full[:, active_mask]

        if self.method == "ledoit_wolf":
            cov_active = self._ledoit_wolf(X)
        else:
            cov_active = np.cov(X, rowvar=False)

        if self.annualize:
            cov_active = cov_active * 252.0

        # ── Step 3: reconstruct full N×N ─────────────────────────────────
        _fallback_var = (0.15 ** 2) * (252.0 if self.annualize else 1.0)
        cov = np.eye(N) * _fallback_var
        active_idx = np.where(active_mask)[0]
        cov[np.ix_(active_idx, active_idx)] = cov_active

        # ── Step 4: symmetry + PD ─────────────────────────────────────────
        cov = (cov + cov.T) / 2.0
        min_eig = float(np.min(np.linalg.eigvalsh(cov)))
        if min_eig < 1e-8:
            cov += np.eye(N) * (1e-8 - min_eig + 1e-8)

        # ── Step 5: bound condition number ───────────────────────────────
        # This is the structural fix for SLSQP overflow.
        # SLSQP's internal QP subproblem uses scipy.linalg.lstsq on the
        # augmented KKT matrix.  When cond(Σ) > ~1e8, the pseudoinverse SVD
        # overflows float64.  Cap cond(Σ) at MAX_COND by adding a ridge:
        #   ridge = (max_eig - MAX_COND * min_eig) / (MAX_COND - 1)
        # This shifts all eigenvalues up proportionally, keeping the
        # relative structure of Σ while making it numerically tractable.
        MAX_COND = 1000.0
        eigvals = np.linalg.eigvalsh(cov)
        lam_max = float(eigvals[-1])
        lam_min = float(eigvals[0])
        if lam_min > 0 and (lam_max / lam_min) > MAX_COND:
            ridge = (lam_max - MAX_COND * lam_min) / (MAX_COND - 1.0)
            cov += np.eye(N) * ridge

        return cov

    # ------------------------------------------------------------------
    # Build rolling covariance snapshot for a given date
    # ------------------------------------------------------------------

    def fit_at_date(
        self,
        price_data: dict[str, pd.DataFrame],
        tickers: list[str],
        date: pd.Timestamp,
        sector_id_map: dict | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build the covariance matrix using `window` days of history ending at `date`.
        Returns (cov, active_tickers) where active_tickers may be a subset of `tickers`
        if some have insufficient history.

        When sector_id_map is provided, the covariance is estimated from
        idiosyncratic returns (r_idio = r_raw − r_market − r_sector).  This
        makes Σ consistent with the alpha signals in MultiAlphaEngine, which
        also operate on r_idio.  Without this, the optimizer minimises w'Σ_raw·w
        while the alpha μ lives in idio-space — the risk and return forecasts
        use different units, making λ uncalibrated.

        When sector_id_map is None or empty, raw-return covariance is returned
        (backward-compatible behaviour).
        """
        close_dict: dict[str, pd.Series] = {}
        for t in tickers:
            df = price_data.get(t)
            if df is None or df.empty:
                continue
            # Prefer AdjClose to avoid artificial return spikes from stock splits/dividends.
            for col in ("Close", "AdjClose", "close"):
                if col in df.columns:
                    close_col = col
                    break
            else:
                continue
            px = pd.to_numeric(df.loc[df.index <= date, close_col], errors="coerce").dropna()
            if len(px) < max(self.min_periods + 1, 5):
                continue
            close_dict[t] = px

        if len(close_dict) < 2:
            active = [t for t in tickers if t in close_dict]
            if not active:
                active = tickers
            return np.eye(len(active)) * (0.20 ** 2), active

        ret_df = (
            pd.DataFrame(close_dict)
            .pct_change(fill_method=None)
            .dropna(how="all")
            .tail(self.window)
            .dropna(axis=1, thresh=self.min_periods)
        )
        active_tickers = list(ret_df.columns)
        if len(active_tickers) < 2:
            return np.eye(len(tickers)) * (0.20 ** 2), tickers

        # ── Factor neutralization: estimate Σ_idio, not Σ_raw ────────────
        # Consistent return space: alphas are in r_idio, covariance must be too.
        # r_idio = r_raw − market_mean − sector_mean (Frisch-Waugh residualization)
        if sector_id_map:
            try:
                from backtesting.multi_alpha import _factor_neutralize
                ret_df = _factor_neutralize(ret_df, sector_id_map, active_tickers)
            except Exception as exc:
                logger.warning(
                    "RiskModel: factor neutralization failed (%s); "
                    "falling back to raw-return covariance.", exc
                )

        cov = self.fit(ret_df)
        return cov, active_tickers

    # ------------------------------------------------------------------
    # Factor model: market beta estimation
    # ------------------------------------------------------------------

    def compute_betas_at_date(
        self,
        price_data: dict[str, pd.DataFrame],
        tickers: list[str],
        date: pd.Timestamp,
        market_ticker: str = "SPY",
    ) -> np.ndarray:
        """
        Estimate rolling market betas via OLS: ret_i = α + β_i × ret_mkt + ε.

        Returns a float array of length len(tickers) aligned to the tickers list.
        Tickers with insufficient history fall back to beta=1.0.

        Parameters
        ----------
        price_data : dict[ticker, DataFrame]
        tickers : list[str]  — universe to estimate (must match optimizer order)
        date : pd.Timestamp  — end of estimation window (exclusive of future data)
        market_ticker : str  — proxy for the market (default 'SPY')
        """
        # ── Market returns ───────────────────────────────────────────────────
        df_mkt = price_data.get(market_ticker)
        if df_mkt is None or df_mkt.empty:
            return np.ones(len(tickers))

        for col in ("AdjClose", "adjclose", "Adj Close", "Close", "close"):
            if col in df_mkt.columns:
                mkt_col = col
                break
        else:
            return np.ones(len(tickers))

        mkt_px = pd.to_numeric(
            df_mkt.loc[df_mkt.index <= date, mkt_col], errors="coerce"
        ).dropna()
        mkt_ret = mkt_px.pct_change(fill_method=None).dropna().tail(self.window)

        if len(mkt_ret) < self.min_periods:
            return np.ones(len(tickers))

        mkt_var = float(np.var(mkt_ret.values))
        if mkt_var < 1e-12:
            return np.ones(len(tickers))

        # ── Per-ticker beta via OLS ──────────────────────────────────────────
        betas: list[float] = []
        for t in tickers:
            df = price_data.get(t)
            if df is None or df.empty:
                betas.append(1.0)
                continue

            for col in ("Close", "AdjClose", "close"):
                if col in df.columns:
                    t_col = col
                    break
            else:
                betas.append(1.0)
                continue

            px = pd.to_numeric(
                df.loc[df.index <= date, t_col], errors="coerce"
            ).dropna()
            ret = px.pct_change(fill_method=None).dropna()

            common = ret.index.intersection(mkt_ret.index)
            if len(common) < self.min_periods:
                betas.append(1.0)
                continue

            y = ret.loc[common].values
            x = mkt_ret.loc[common].values
            # OLS: β = cov(y, x) / var(x)
            cov_yx = float(np.cov(y, x, ddof=1)[0, 1])
            var_x = float(np.var(x, ddof=1))
            beta = cov_yx / max(var_x, 1e-12)
            betas.append(float(np.clip(beta, -3.0, 5.0)))  # winsorise extremes

        return np.array(betas, dtype=float)

    # ------------------------------------------------------------------
    # Effective N: eigenvalue participation ratio
    # ------------------------------------------------------------------

    @staticmethod
    def effective_n(cov: np.ndarray) -> float:
        """
        Participation ratio of eigenvalues = (sum λ)² / sum λ².

        Replaces the hardcoded avg_corr=0.3 approximation in Grinold-Kahn.
        Interpretation: equivalent number of independent bets in the portfolio.

        Reference: Effective rank (Roy & Vetterli 2007); used in portfolio
        breadth estimation by Qian (2006) and Menchero et al. (2011).
        """
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-10]  # drop numerical noise
        if len(eigvals) == 0:
            return 1.0
        s1 = float(eigvals.sum())
        s2 = float((eigvals ** 2).sum())
        if s2 < 1e-12:
            return 1.0
        return (s1 ** 2) / s2

    # ------------------------------------------------------------------
    # Private: Ledoit-Wolf via sklearn
    # ------------------------------------------------------------------

    @staticmethod
    def _ledoit_wolf(X: np.ndarray) -> np.ndarray:
        """
        Compute Ledoit-Wolf shrunk covariance using the sklearn FUNCTION
        interface, not the class.

        WHY NOT LedoitWolf() CLASS:
          LedoitWolf.fit() unconditionally computes the precision matrix
          (covariance inverse) via scipy.linalg.pinvh() after fitting:
              self.precision_ = linalg.pinvh(covariance, check_finite=False)
          When the covariance has near-zero eigenvalues (T/N ratio ≈ 2, or
          sparse idio returns after factor neutralization), pinvh computes
          1/λ for near-zero λ, which overflows float64 inside the SVD
          at scipy.linalg._basic.py: B = (u * psigma_diag) @ u.conj().T.
          We never use precision_ — we only need covariance_ — so computing
          it is both wasteful and numerically unsafe.

        WHY ledoit_wolf() FUNCTION:
          The function interface returns (covariance, shrinkage) and does NOT
          compute the precision matrix.  The same shrinkage coefficient is
          computed via the Oracle Approximating Shrinkage Estimator (OAS)
          algorithm, but the inverse is never taken.  Identical covariance
          output, no pinvh call, no overflow.
        """
        try:
            from sklearn.covariance import ledoit_wolf
            covariance, _ = ledoit_wolf(X, assume_centered=False)
            return covariance
        except Exception as exc:
            logger.warning("ledoit_wolf failed (%s); falling back to sample cov.", exc)
            return np.cov(X, rowvar=False)
