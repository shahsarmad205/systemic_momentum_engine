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

    def compute_factor_exposures_at_date(
        self,
        price_data: dict[str, pd.DataFrame],
        tickers: list[str],
        date: pd.Timestamp,
        *,
        market_ticker: str = "SPY",
        sector_id_map: dict[str, int] | None = None,
        sector_labels: dict[int, str] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Build a lightweight explicit factor exposure panel aligned to ``tickers``.

        Available today:
        - ``market_beta`` from rolling OLS beta vs market_ticker
        - ``sector:<name>`` one-hot net exposure controls
        - ``size`` from log market cap when available
        - ``momentum`` from trailing 12m return excluding the most recent month
        - ``value`` / ``quality`` only when the underlying columns exist
        """
        exposures: dict[str, np.ndarray] = {}
        if not tickers:
            return exposures

        market_beta = self.compute_betas_at_date(
            price_data,
            tickers,
            date,
            market_ticker=market_ticker,
        )
        if np.allclose(market_beta, 1.0):
            fallback_beta: list[float] = []
            beta_available = False
            for t in tickers:
                df = price_data.get(t)
                if df is None or df.empty:
                    fallback_beta.append(1.0)
                    continue
                hist = df.loc[df.index <= date]
                beta_col = next((c for c in ("market_beta", "capm_beta", "beta") if c in hist.columns), None)
                if beta_col is None or hist.empty:
                    fallback_beta.append(1.0)
                    continue
                beta_available = True
                raw = pd.to_numeric(hist[beta_col], errors="coerce").dropna()
                fallback_beta.append(float(raw.iloc[-1]) if not raw.empty else 1.0)
            if beta_available:
                market_beta = np.clip(np.asarray(fallback_beta, dtype=float), -3.0, 5.0)
        exposures["market_beta"] = market_beta

        def _zscore(arr: np.ndarray) -> np.ndarray:
            arr = np.array(arr, dtype=float)
            mask = np.isfinite(arr)
            if mask.sum() < 2:
                return np.zeros_like(arr)
            vals = arr[mask]
            mean = float(vals.mean())
            std = float(vals.std(ddof=0))
            if std <= 1e-12:
                return np.zeros_like(arr)
            out = np.zeros_like(arr)
            out[mask] = (vals - mean) / std
            return out

        size_proxy: list[float] = []
        momentum_proxy: list[float] = []
        value_proxy: list[float] = []
        quality_proxy: list[float] = []
        value_available = False
        quality_available = False

        for t in tickers:
            df = price_data.get(t)
            if df is None or df.empty:
                size_proxy.append(np.nan)
                momentum_proxy.append(np.nan)
                value_proxy.append(np.nan)
                quality_proxy.append(np.nan)
                continue

            hist = df.loc[df.index <= date]
            if hist.empty:
                size_proxy.append(np.nan)
                momentum_proxy.append(np.nan)
                value_proxy.append(np.nan)
                quality_proxy.append(np.nan)
                continue

            direct_size_col = next((c for c in ("size", "size_exposure", "log_market_cap") if c in hist.columns), None)
            if direct_size_col is not None:
                size_proxy.append(float(pd.to_numeric(hist[direct_size_col].iloc[-1], errors="coerce")))
                mcap_col = None
            else:
                mcap_col = "market_cap" if "market_cap" in hist.columns else None
            if mcap_col is not None:
                last_mcap = float(pd.to_numeric(hist[mcap_col].iloc[-1], errors="coerce"))
                size_proxy.append(np.log(max(last_mcap, 1.0)))
            elif direct_size_col is None:
                close_col = "Close" if "Close" in hist.columns else "close" if "close" in hist.columns else None
                if close_col and "Volume" in hist.columns and len(hist) >= 20:
                    approx_size = float(
                        pd.to_numeric(hist[close_col].iloc[-1], errors="coerce")
                    ) * float(pd.to_numeric(hist["Volume"].tail(20), errors="coerce").mean() or 0.0)
                    size_proxy.append(np.log(max(approx_size, 1.0)))
                else:
                    size_proxy.append(np.nan)

            direct_mom_col = next((c for c in ("momentum", "momentum_exposure", "momentum_12m_skip1") if c in hist.columns), None)
            close_col = next((c for c in ("AdjClose", "Adj Close", "Close", "close") if c in hist.columns), None)
            if direct_mom_col is not None:
                momentum_proxy.append(float(pd.to_numeric(hist[direct_mom_col].iloc[-1], errors="coerce")))
            elif close_col and len(hist) >= 252:
                px = pd.to_numeric(hist[close_col], errors="coerce").dropna()
                if len(px) >= 252:
                    ref = float(px.iloc[-252])
                    recent = float(px.iloc[-21]) if len(px) >= 21 else float(px.iloc[-1])
                    momentum_proxy.append((recent / max(ref, 1e-12)) - 1.0)
                else:
                    momentum_proxy.append(np.nan)
            else:
                momentum_proxy.append(np.nan)

            val_col = next((c for c in ("book_to_market", "bm", "value_score") if c in hist.columns), None)
            if val_col is not None:
                value_available = True
                value_proxy.append(float(pd.to_numeric(hist[val_col].iloc[-1], errors="coerce")))
            else:
                value_proxy.append(np.nan)

            qual_col = next((c for c in ("quality_score", "roa", "profitability") if c in hist.columns), None)
            if qual_col is not None:
                quality_available = True
                quality_proxy.append(float(pd.to_numeric(hist[qual_col].iloc[-1], errors="coerce")))
            else:
                quality_proxy.append(np.nan)

        exposures["size"] = _zscore(np.array(size_proxy, dtype=float))
        exposures["momentum"] = _zscore(np.array(momentum_proxy, dtype=float))
        if value_available:
            exposures["value"] = _zscore(np.array(value_proxy, dtype=float))
        if quality_available:
            exposures["quality"] = _zscore(np.array(quality_proxy, dtype=float))

        if sector_id_map:
            sector_names: dict[int, str] = sector_labels or {}
            sector_ids = [sector_id_map.get(t) for t in tickers]
            unique_sector_ids = sorted({sid for sid in sector_ids if sid is not None})
            for sid in unique_sector_ids:
                label = sector_names.get(sid, str(sid))
                exposures[f"sector:{label}"] = np.array(
                    [1.0 if sector_id_map.get(t) == sid else 0.0 for t in tickers],
                    dtype=float,
                )

        return exposures

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


# ------------------------------------------------------------------
# PCARiskModel: Factor-based risk model (Σ = BFB' + D)
# ------------------------------------------------------------------

class PCARiskModel:
    """
    PCA-based factor risk model: Σ = B F B' + D

    Decomposes asset returns into:
      - B: factor loading matrix (N × k) from PCA
      - F: factor covariance (k × k)
      - D: diagonal specific risk matrix (N × N)

    This reduces parameter estimation from O(N²) to O(N*k + k² + N),
    dramatically improving conditioning when T << N.

    For N=500, k=10:
      Raw covariance: ~125,000 parameters from T=60 observations
      PCA factor model: 500*10 + 10*10 + 500 = 5,600 parameters

    The number of factors k is chosen adaptively using the cumulative
    variance explained criterion (default: 80% of total variance).
    """

    def __init__(
        self,
        window: int = 60,
        min_periods: int = 20,
        max_factors: int = 20,
        variance_threshold: float = 0.80,
        min_specific_risk_weight: float = 0.10,
        annualize: bool = True,
    ):
        self.window = int(window)
        self.min_periods = int(min_periods)
        self.max_factors = int(max_factors)
        self.variance_threshold = float(variance_threshold)
        self.min_specific_risk_weight = float(min_specific_risk_weight)
        self.annualize = bool(annualize)

    def fit(self, returns_df: pd.DataFrame) -> np.ndarray:
        """
        Estimate factor risk model from a (T × N) returns DataFrame.

        Returns full covariance matrix Σ = B F B' + D.
        """
        N = len(returns_df.columns)
        T = len(returns_df)

        if T < self.min_periods or N < 2:
            return np.eye(N) * (0.20 ** 2)

        X_full = (
            returns_df
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(-0.5, 0.5)
            .values
        )

        col_std = X_full.std(axis=0)
        active_mask = col_std > 1e-6
        n_active = int(active_mask.sum())

        if n_active < 2:
            return np.eye(N) * (0.15 ** 2)

        X = X_full[:, active_mask]
        T_active, N_active = X.shape

        # Step 1: Standardize returns (zero mean, unit variance)
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True)
        X_std = np.where(X_std < 1e-12, 1.0, X_std)
        X_z = (X - X_mean) / X_std

        # Step 2: PCA on correlation matrix
        try:
            from sklearn.decomposition import PCA
            n_components = min(self.max_factors, N_active, T_active - 1)
            pca = PCA(n_components=n_components)
            pca.fit(X_z)
            explained_var = np.cumsum(pca.explained_variance_ratio_)
            k = int(np.searchsorted(explained_var, self.variance_threshold) + 1)
            k = max(1, min(k, n_components, N_active))
        except Exception:
            k = min(self.max_factors, N_active // 4, T_active // 6)
            k = max(1, min(k, N_active))
            pca = None

        # Step 3: Extract factor loadings B (N × k)
        if pca is not None:
            loadings = pca.components_[:k].T  # shape (N_active, k)
            # Scale loadings by sqrt of eigenvalues to get proper factor exposures
            loadings = loadings * np.sqrt(pca.explained_variance_[:k])
            factor_cov = np.diag(pca.explained_variance_[:k])  # F matrix
        else:
            # Manual PCA via SVD
            U, S, Vt = np.linalg.svd(X_z, full_matrices=False)
            loadings = Vt[:k].T * S[:k] / np.sqrt(T_active - 1)
            factor_cov = np.diag((S[:k] ** 2) / (T_active - 1))

        # Step 4: Compute specific risks D = diag(var(r) - B F B')
        total_var = np.var(X, axis=0)
        common_var = np.sum(loadings @ factor_cov @ loadings.T * np.eye(N_active), axis=1)
        # More efficient: common variance per asset = sum of squared loadings * factor variance
        common_var = np.sum(loadings**2 * np.diag(factor_cov)[None, :], axis=1)
        specific_var = total_var - common_var
        # Ensure specific risk is positive and at least a fraction of total risk
        min_specific = total_var * self.min_specific_risk_weight
        specific_var = np.maximum(specific_var, min_specific)

        # Step 5: Reconstruct full covariance Σ = B F B' + D
        cov_common = loadings @ factor_cov @ loadings.T
        cov = cov_common + np.diag(specific_var)

        # Step 6: De-standardize back to original scale
        cov = cov * (X_std.T @ X_std)

        # Step 7: Reconstruct to full N × N with inactive assets
        _fallback_var = (0.15 ** 2) * (252.0 if self.annualize else 1.0)
        full_cov = np.eye(N) * _fallback_var
        active_idx = np.where(active_mask)[0]
        full_cov[np.ix_(active_idx, active_idx)] = cov

        if self.annualize:
            full_cov = full_cov * 252.0

        # Step 8: Symmetry + PD enforcement
        full_cov = (full_cov + full_cov.T) / 2.0
        min_eig = float(np.min(np.linalg.eigvalsh(full_cov)))
        if min_eig < 1e-8:
            full_cov += np.eye(N) * (1e-8 - min_eig + 1e-8)

        # Step 9: Condition number bound
        MAX_COND = 1000.0
        eigvals = np.linalg.eigvalsh(full_cov)
        lam_max = float(eigvals[-1])
        lam_min = float(eigvals[0])
        if lam_min > 0 and (lam_max / lam_min) > MAX_COND:
            ridge = (lam_max - MAX_COND * lam_min) / (MAX_COND - 1.0)
            full_cov += np.eye(N) * ridge

        return full_cov

    def fit_at_date(
        self,
        price_data: dict[str, pd.DataFrame],
        tickers: list[str],
        date: pd.Timestamp,
        sector_id_map: dict | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build the PCA factor risk model covariance using `window` days of history.
        """
        close_dict: dict[str, pd.Series] = {}
        for t in tickers:
            df = price_data.get(t)
            if df is None or df.empty:
                continue
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

        if sector_id_map:
            try:
                from backtesting.multi_alpha import _factor_neutralize
                ret_df = _factor_neutralize(ret_df, sector_id_map, active_tickers)
            except Exception as exc:
                logger.warning(
                    "PCARiskModel: factor neutralization failed (%s); "
                    "falling back to raw-return covariance.", exc
                )

        cov = self.fit(ret_df)
        return cov, active_tickers

    @staticmethod
    def factor_summary(cov: np.ndarray, n_factors: int = 10) -> dict[str, float]:
        """
        Return diagnostic summary of the factor risk model decomposition.
        """
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-10]
        if len(eigvals) == 0:
            return {"n_factors": 0, "variance_explained": 0.0, "specific_risk_ratio": 1.0}
        total_var = float(eigvals.sum())
        top_k = min(n_factors, len(eigvals))
        common_var = float(np.sort(eigvals)[-top_k:].sum())
        return {
            "n_eigenvalues": float(len(eigvals)),
            "top_k_factors": float(top_k),
            "common_variance_ratio": common_var / total_var if total_var > 0 else 0.0,
            "specific_risk_ratio": 1.0 - (common_var / total_var) if total_var > 0 else 1.0,
            "effective_n": RiskModel.effective_n(cov),
        }


# ------------------------------------------------------------------
# VolatilityTargeting: Integrated vol forecast and position scaling
# ------------------------------------------------------------------

class VolatilityTargeting:
    """
    Institutional volatility targeting integrated with the risk model.

    Unlike ad-hoc scalar multipliers, this produces a forward-looking
    volatility forecast that feeds directly into position sizing:

      1. Forecast portfolio volatility from Σ and current weights:
         σ_p = sqrt(w' Σ w)

      2. Compute scaling factor to hit target vol:
         s = target_vol / σ_p

      3. Apply bounds to prevent excessive leverage or deleveraging:
         s_clipped = clip(s, min_scale, max_scale)

      4. Scale positions: w_targeted = s_clipped × w

    This ensures the portfolio maintains consistent risk exposure across
    regimes, automatically deleveraging in high-vol environments and
        levering up when vol is low (within bounds).

    References:
      - Moreiras (2004) "Understanding Risk and Return"
      - Barroso & Santa-Clara (2015) "Momentum Has Its Moments"
      - AQR "Volatility-Managed Portfolios" (2014)
    """

    def __init__(
        self,
        target_vol: float = 0.15,
        vol_forecast_window: int = 60,
        min_scale: float = 0.2,
        max_scale: float = 1.5,
        ewm_halflife: int = 20,
        use_garch_approx: bool = True,
    ):
        self.target_vol = float(target_vol)
        self.vol_forecast_window = int(vol_forecast_window)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.ewm_halflife = int(ewm_halflife)
        self.use_garch_approx = bool(use_garch_approx)

    def forecast_portfolio_vol(
        self,
        weights: np.ndarray,
        cov: np.ndarray,
        realized_vols: np.ndarray | None = None,
    ) -> float:
        """
        Forecast portfolio volatility using covariance matrix.

        Uses the risk model's covariance estimate with optional
        GARCH-style volatility adjustment for recent market conditions.

        Parameters
        ----------
        weights : np.ndarray
            Current portfolio weights (N,)
        cov : np.ndarray
            Covariance matrix (N, N), annualized
        realized_vols : np.ndarray, optional
            Recent realized volatilities per asset for GARCH adjustment

        Returns
        -------
        float
            Forecast annualized portfolio volatility
        """
        w = np.asarray(weights, dtype=float)
        cov_matrix = np.asarray(cov, dtype=float)

        # Base forecast: w' Σ w
        port_var = float(np.dot(w, np.dot(cov_matrix, w)))
        port_vol = np.sqrt(max(port_var, 1e-12))

        # GARCH-style adjustment: if recent realized vols differ from
        # cov-implied vols, adjust the forecast
        if realized_vols is not None and self.use_garch_approx:
            rv = np.asarray(realized_vols, dtype=float)
            implied_vols = np.sqrt(np.diag(cov_matrix))
            mask = implied_vols > 1e-12
            if np.any(mask):
                ratio = rv[mask] / implied_vols[mask]
                ratio = np.clip(ratio, 0.5, 2.0)
                avg_ratio = float(np.nanmean(ratio))
                port_vol = port_vol * avg_ratio

        return float(port_vol)

    def compute_scale_factor(
        self,
        weights: np.ndarray,
        cov: np.ndarray,
        realized_vols: np.ndarray | None = None,
    ) -> float:
        """
        Compute the scaling factor to achieve target volatility.

        Returns clipped scale factor s such that:
          scale(σ_forecast) → target_vol

        where scale() = clip(target_vol / σ_forecast, min_scale, max_scale)
        """
        forecast_vol = self.forecast_portfolio_vol(weights, cov, realized_vols)
        if forecast_vol < 1e-8:
            return self.max_scale
        raw_scale = self.target_vol / forecast_vol
        return float(np.clip(raw_scale, self.min_scale, self.max_scale))

    def apply_vol_targeting(
        self,
        weights: np.ndarray,
        cov: np.ndarray,
        realized_vols: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """
        Apply volatility targeting to portfolio weights.

        Parameters
        ----------
        weights : np.ndarray
            Current portfolio weights (N,)
        cov : np.ndarray
            Covariance matrix (N, N)
        realized_vols : np.ndarray, optional
            Recent realized volatilities for GARCH adjustment

        Returns
        -------
        tuple[np.ndarray, float, float]
            (targeted_weights, scale_factor, forecast_vol)
        """
        forecast_vol = self.forecast_portfolio_vol(weights, cov, realized_vols)
        scale = self.compute_scale_factor(weights, cov, realized_vols)
        targeted = weights * scale
        return targeted, float(scale), float(forecast_vol)

    @staticmethod
    def compute_realized_vols(
        returns: np.ndarray,
        window: int = 20,
        annualize: bool = True,
    ) -> np.ndarray:
        """
        Compute realized volatilities from a returns matrix (T × N).

        Uses rolling window with optional annualization.
        """
        ret = np.asarray(returns, dtype=float)
        if ret.ndim == 1:
            ret = ret[:, None]
        if ret.shape[0] < window:
            return np.full(ret.shape[1], np.nan)
        vols = np.nanstd(ret[-window:], axis=0, ddof=1)
        if annualize:
            vols = vols * np.sqrt(252)
        return vols


# ------------------------------------------------------------------
# KellyCriterion: Optimal position sizing
# ------------------------------------------------------------------

class KellyCriterion:
    """
    Kelly criterion position sizing for portfolio allocation.

    The Kelly formula maximizes the expected geometric growth rate:
      f* = (p × b - q) / b
    where:
      p = win probability
      q = 1 - p = loss probability
      b = win/loss ratio (avg_win / avg_loss)

    For continuous returns with known mean μ and variance σ²:
      f* = μ / σ²

    Institutional implementations use fractional Kelly (typically 0.25-0.5)
    to account for parameter estimation error and drawdown aversion.

    References:
      - Kelly (1956) "A New Interpretation of Information Rate"
      - Thorpe (2008) "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
      - MacLean, Thorpe, Ziemba (2011) "The Kelly Capital Growth Investment Criterion"
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        max_position: float = 0.10,
        min_position: float = 0.001,
        use_full_covariance: bool = True,
        risk_free_rate: float = 0.02,
    ):
        self.kelly_fraction = float(kelly_fraction)
        self.max_position = float(max_position)
        self.min_position = float(min_position)
        self.use_full_covariance = bool(use_full_covariance)
        self.risk_free_rate = float(risk_free_rate)

    def compute_kelly_weights(
        self,
        expected_returns: np.ndarray,
        cov: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Kelly-optimal weights: w* = (μ - r_f) / σ² for diagonal,
        or w* = Σ⁻¹(μ - r_f) for full covariance.

        Parameters
        ----------
        expected_returns : np.ndarray
            Expected excess returns per asset (N,)
        cov : np.ndarray
            Covariance matrix (N, N), annualized

        Returns
        -------
        np.ndarray
            Kelly-optimal weights, clipped and scaled
        """
        mu = np.asarray(expected_returns, dtype=float) - self.risk_free_rate
        cov_matrix = np.asarray(cov, dtype=float)
        N = len(mu)

        if self.use_full_covariance and N > 1:
            # Full Kelly: w* = Σ⁻¹(μ - r_f)
            try:
                kelly_weights = np.linalg.solve(cov_matrix, mu)
            except np.linalg.LinAlgError:
                # Fallback to diagonal if singular
                diag_var = np.diag(cov_matrix)
                mask = diag_var > 1e-12
                kelly_weights = np.zeros(N)
                kelly_weights[mask] = mu[mask] / diag_var[mask]
        else:
            # Diagonal Kelly: w*_i = (μ_i - r_f) / σ²_i
            diag_var = np.diag(cov_matrix)
            mask = diag_var > 1e-12
            kelly_weights = np.zeros(N)
            kelly_weights[mask] = mu[mask] / diag_var[mask]

        # Apply fractional Kelly
        kelly_weights = kelly_weights * self.kelly_fraction

        # Clip to position bounds
        kelly_weights = np.clip(kelly_weights, -self.max_position, self.max_position)

        # Zero out sub-threshold positions
        kelly_weights[np.abs(kelly_weights) < self.min_position] = 0.0

        return kelly_weights

    def compute_kelly_from_edge(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Compute Kelly fraction from discrete trade statistics.

        Parameters
        ----------
        win_rate : float
            Fraction of trades that are profitable
        avg_win : float
            Average return on winning trades
        avg_loss : float
            Average absolute return on losing trades

        Returns
        -------
        float
            Kelly-optimal position size (fractional)
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        b = avg_win / avg_loss  # win/loss ratio
        p = win_rate
        q = 1.0 - p
        kelly = (p * b - q) / b
        return float(max(0.0, kelly * self.kelly_fraction))
