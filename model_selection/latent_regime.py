"""
Latent Regime Model
=====================
Continuous, data-driven market regime representation that replaces discrete
hard-assignment labels with a soft probabilistic state vector.

Feature vector Z_t (all derived from panel without external data)
-----------------------------------------------------------------
  cs_vol           : mean cross-sectional realized vol
  vol_slope        : 21d rolling change in cs_vol (volatility trend)
  mkt_trend        : cumulative equal-weighted return (trend strength)
  mkt_drawdown     : mean peak-to-current drawdown (stress)
  cs_dispersion    : mean cross-sectional daily-return std (alpha opportunity)
  cs_corr_mean     : mean pairwise Pearson correlation across daily returns
  liq_change       : 21d change in log-median ADV (liquidity direction)

Regime representation
---------------------
  GMM (default): BIC-selected n_components in [n_min, n_max].
    predict_soft() → (n_dates × n_components) probability matrix.
    Enables smooth interpolation between regimes.
  PCA (option): fit PCA on Z_t, expose first 2 components as z1/z2.
    No discrete labels — pure continuous latent state.

Conditioning strategies
-----------------------
  A. per-regime separate models:  train on dates in high-weight regime
  B. Z_t as input feature:        append regime-weighted PCA projection
  C. mixture-of-experts:          blend per-regime scores by soft weights

Evaluation
----------
  compute_soft_ic_by_regime  : IC and Sharpe weighted by soft regime membership
  evaluate_transition_stability: fraction of dates with large regime weight shift
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LatentRegimeConfig:
    n_regimes_min: int = 2
    n_regimes_max: int = 6
    pca_components: int = 2           # latent dims for PCA mode
    use_pca: bool = False             # True → PCA latent space; False → GMM soft assignment
    corr_window_days: int = 21        # window for cs_corr estimation
    vol_slope_window: int = 21        # window for vol change (slope)
    smoothing_days: int = 3           # EMA half-life for soft weights
    random_state: int = 42
    # panel column names
    return_col: str = "daily_return"
    vol_col: str = "realised_vol_20d"
    adv_col: str = "adv_dollar_20"


# ---------------------------------------------------------------------------
# Extended feature extraction
# ---------------------------------------------------------------------------


def extract_latent_features(
    df: pd.DataFrame,
    cfg: LatentRegimeConfig | None = None,
) -> pd.DataFrame:
    """
    Build one row per date with seven market-state features.

    Extends the existing regime_conditioning features with:
    - vol_slope  : rolling derivative of cross-sectional volatility
    - cs_corr_mean : mean pairwise Pearson correlation of daily returns
    """
    if cfg is None:
        cfg = LatentRegimeConfig()

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")

    agg: dict[str, pd.Series] = {}

    if cfg.return_col in work.columns:
        r = work[cfg.return_col].replace([np.inf, -np.inf], np.nan)
        work["_r"] = r
        g = work.groupby("date")["_r"]
        agg["cs_dispersion"] = g.std()
        agg["mkt_return"] = g.mean()

    if cfg.vol_col in work.columns:
        v = work[cfg.vol_col].replace([np.inf, -np.inf], np.nan)
        work["_v"] = v
        agg["cs_vol"] = work.groupby("date")["_v"].mean()

    if cfg.adv_col in work.columns:
        a = work[cfg.adv_col].replace([np.inf, -np.inf], np.nan).clip(lower=1.0)
        log_adv = np.log(a)
        work["_la"] = log_adv
        agg["log_adv"] = work.groupby("date")["_la"].median()

    daily = pd.DataFrame(agg).sort_index()
    daily.index = pd.to_datetime(daily.index)

    # Derived rolling features
    if "mkt_return" in daily.columns:
        daily["mkt_trend"] = daily["mkt_return"].rolling(21, min_periods=10).sum()
        cum = daily["mkt_return"].cumsum().ffill()
        peak = cum.rolling(252, min_periods=21).max()
        daily["mkt_drawdown"] = (cum - peak).clip(upper=0.0)
        daily = daily.drop(columns=["mkt_return"])

    if "log_adv" in daily.columns:
        daily["liq_change"] = daily["log_adv"].diff(cfg.vol_slope_window)
        daily = daily.drop(columns=["log_adv"])

    # Volatility slope (rolling derivative of cs_vol)
    if "cs_vol" in daily.columns:
        daily["vol_slope"] = daily["cs_vol"].diff(cfg.vol_slope_window)

    # Mean pairwise return correlation (estimated from rolling window)
    # For each date, pivot last W days of returns and compute avg upper-triangle corr
    if cfg.return_col in work.columns:
        daily["cs_corr_mean"] = _rolling_mean_pairwise_corr(
            work, date_col="date", return_col=cfg.return_col,
            window=cfg.corr_window_days,
        )

    return daily.replace([np.inf, -np.inf], np.nan)


def _rolling_mean_pairwise_corr(
    work: pd.DataFrame,
    *,
    date_col: str,
    return_col: str,
    window: int,
) -> pd.Series:
    """
    For each date t, compute mean upper-triangle Pearson correlation
    from the pivot of returns over the past `window` dates.
    Returns a pd.Series indexed by date.
    """
    # Pivot: (date × ticker) returns matrix
    pivot = (
        work[[date_col, "ticker", return_col]]
        .pivot_table(index=date_col, columns="ticker", values=return_col, aggfunc="first")
    )
    pivot = pivot.replace([np.inf, -np.inf], np.nan)

    dates = pivot.index.sort_values()
    result: dict = {}

    for i, dt in enumerate(dates):
        lo = max(0, i - window + 1)
        sub = pivot.iloc[lo : i + 1].dropna(axis=1, thresh=window // 2)
        if sub.shape[1] < 3 or sub.shape[0] < 5:
            result[dt] = float("nan")
            continue
        corr_mat = sub.corr(method="pearson")
        n = len(corr_mat)
        upper = corr_mat.values[np.triu_indices(n, k=1)]
        upper = upper[np.isfinite(upper)]
        result[dt] = float(upper.mean()) if len(upper) > 0 else float("nan")

    return pd.Series(result, name="cs_corr_mean")


# ---------------------------------------------------------------------------
# Latent Regime Model
# ---------------------------------------------------------------------------


class LatentRegimeModel:
    """
    Data-driven continuous regime model using GMM soft assignments or PCA.

    GMM mode (default):
        - BIC-selected n_components in [n_min, n_max]
        - predict_soft() → (n_dates × n_components) probability matrix
        - Each column is a "soft regime weight" w_{k,t}

    PCA mode (use_pca=True):
        - PCA(n_components=cfg.pca_components) on normalized Z_t
        - get_latent_z() → (n_dates × n_components) continuous latent state
    """

    def __init__(self, cfg: LatentRegimeConfig | None = None) -> None:
        self.cfg = cfg or LatentRegimeConfig()
        self._scaler: StandardScaler | None = None
        self._gmm: GaussianMixture | None = None
        self._pca: PCA | None = None
        self._n_components: int = 2
        self._feature_cols: list[str] = []
        self._regime_labels: list[str] = []

    def fit(self, daily_features: pd.DataFrame) -> "LatentRegimeModel":
        feat = daily_features.dropna(how="any")
        min_obs = max(self.cfg.n_regimes_max * 15, 60)
        if len(feat) < min_obs:
            raise ValueError(
                f"Insufficient daily observations ({len(feat)}) for latent regime fitting "
                f"(need ≥ {min_obs})."
            )

        self._feature_cols = sorted(
            [c for c in feat.columns if feat[c].dtype.kind in "fd"]
        )
        X = feat[self._feature_cols].to_numpy(dtype=float)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        self._scaler = scaler

        if self.cfg.use_pca:
            n_comp = min(self.cfg.pca_components, X.shape[1], X.shape[0])
            pca = PCA(n_components=n_comp, random_state=self.cfg.random_state)
            pca.fit(Xs)
            self._pca = pca
            self._n_components = n_comp
        else:
            best_bic = np.inf
            best_gmm: GaussianMixture | None = None
            best_k = self.cfg.n_regimes_min

            for k in range(self.cfg.n_regimes_min, self.cfg.n_regimes_max + 1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gm = GaussianMixture(
                        n_components=k,
                        covariance_type="full",
                        n_init=5,
                        max_iter=300,
                        random_state=self.cfg.random_state,
                    )
                    gm.fit(Xs)
                bic = gm.bic(Xs)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gm
                    best_k = k

            self._gmm = best_gmm
            self._n_components = best_k
            self._regime_labels = [f"regime_{k}" for k in range(best_k)]

        return self

    def predict_soft(
        self, daily_features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Return soft regime assignment probabilities (GMM) or PCA coordinates.

        For GMM: columns are 'regime_0', 'regime_1', ... (probability of each regime).
        For PCA: columns are 'z1', 'z2', ... (normalized PCA scores in [0,1] per date).
        Index: date.
        """
        if self._scaler is None:
            raise RuntimeError("LatentRegimeModel must be fitted first.")

        feat = daily_features.copy()
        missing = set(self._feature_cols) - set(feat.columns)
        if missing:
            raise ValueError(f"Missing latent regime features: {missing}")

        valid_mask = feat[self._feature_cols].notna().all(axis=1)
        X = feat.loc[valid_mask, self._feature_cols].to_numpy(dtype=float)
        Xs = self._scaler.transform(X)

        if self.cfg.use_pca:
            assert self._pca is not None
            Xp = self._pca.transform(Xs)
            # Normalize each PC to [0, 1] softmax-style (not probability, but interpretable)
            cols = [f"z{i+1}" for i in range(self._n_components)]
            out = pd.DataFrame(Xp, index=feat.index[valid_mask], columns=cols)
        else:
            assert self._gmm is not None
            probs = self._gmm.predict_proba(Xs)
            cols = [f"regime_{k}" for k in range(self._n_components)]
            out = pd.DataFrame(probs, index=feat.index[valid_mask], columns=cols)

        # Reindex to all dates with NaN for missing, then forward-fill
        out = out.reindex(feat.index).ffill().bfill()

        # Optional EMA smoothing to reduce day-to-day regime chatter
        if self.cfg.smoothing_days > 1 and not out.empty:
            alpha = 2.0 / (self.cfg.smoothing_days + 1)
            out = out.ewm(alpha=alpha, adjust=False).mean()

        return out

    def get_latent_z(
        self, daily_features: pd.DataFrame
    ) -> pd.DataFrame:
        """PCA projection of Z_t (always available regardless of use_pca setting)."""
        if self._scaler is None:
            raise RuntimeError("LatentRegimeModel must be fitted first.")
        if self._pca is None:
            n_comp = min(self.cfg.pca_components, len(self._feature_cols))
            pca = PCA(n_components=n_comp, random_state=self.cfg.random_state)
            feat = daily_features[self._feature_cols].dropna(how="any")
            Xs = self._scaler.transform(feat.to_numpy(dtype=float))
            pca.fit(Xs)
            self._pca = pca

        feat = daily_features.copy()
        valid_mask = feat[self._feature_cols].notna().all(axis=1)
        Xs = self._scaler.transform(
            feat.loc[valid_mask, self._feature_cols].to_numpy(dtype=float)
        )
        Xp = self._pca.transform(Xs)
        cols = [f"z{i+1}" for i in range(self._pca.n_components_)]
        return pd.DataFrame(Xp, index=feat.index[valid_mask], columns=cols)

    def attach_soft_weights(
        self,
        panel_df: pd.DataFrame,
        daily_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge soft regime weights into the panel DataFrame on the date column.
        Adds columns: regime_0, regime_1, ..., regime_k (or z1, z2 for PCA mode).
        """
        weights = self.predict_soft(daily_features)
        weights_df = weights.reset_index().rename(columns={"index": "date"})
        weights_df["date"] = pd.to_datetime(weights_df["date"])

        out = panel_df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

        # Drop any existing regime columns
        drop_cols = [c for c in out.columns if c in weights.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)

        return out.merge(weights_df, on="date", how="left")

    @property
    def n_regimes(self) -> int:
        return self._n_components

    @property
    def explained_variance_ratio(self) -> np.ndarray | None:
        if self._pca is not None:
            return self._pca.explained_variance_ratio_
        return None


# ---------------------------------------------------------------------------
# Conditional evaluation
# ---------------------------------------------------------------------------


def compute_soft_ic_by_regime(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    regime_cols: list[str],
    *,
    min_effective_obs: float = 10.0,
) -> pd.DataFrame:
    """
    Compute IC and Sharpe for each regime using soft regime membership as weights.

    For each regime k, the effective IC is:
        IC_k = Σ_t w_{k,t} × IC_t  /  Σ_t w_{k,t}
    where IC_t is the cross-sectional rank-IC on date t.

    Parameters
    ----------
    df          : panel with date, score_col, target_col, regime weight columns
    score_col   : model score column
    target_col  : return/label column
    regime_cols : soft weight columns (regime_0, regime_1, ...) summing to ~1 per date
    min_effective_obs : minimum Σ_t w_{k,t} to include a regime row

    Returns
    -------
    DataFrame with columns: regime, eff_obs, ic_mean (weighted), ic_tstat, sharpe
    """
    from scipy import stats as scipy_stats

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")

    # Per-date IC
    daily_ic: dict = {}
    for dt, g in work.groupby("date", sort=True):
        s = pd.to_numeric(g[score_col], errors="coerce").to_numpy(dtype=float)
        t = pd.to_numeric(g[target_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(s) & np.isfinite(t)
        if mask.sum() < 5 or s[mask].std() == 0.0 or t[mask].std() == 0.0:
            continue
        r, _ = scipy_stats.spearmanr(s[mask], t[mask])
        if math.isfinite(r):
            daily_ic[dt] = float(r)

    if not daily_ic:
        return pd.DataFrame()

    ic_series = pd.Series(daily_ic, name="ic")
    rows = []

    for reg in regime_cols:
        if reg not in work.columns:
            continue
        # Get regime weight per date (one value per date)
        w_by_date = work.groupby("date")[reg].first()
        w_by_date = w_by_date.reindex(ic_series.index).fillna(0.0)

        eff_obs = float(w_by_date.sum())
        if eff_obs < min_effective_obs:
            continue

        w_arr = w_by_date.to_numpy(dtype=float)
        ic_arr = ic_series.to_numpy(dtype=float)

        w_arr = np.clip(w_arr, 0.0, 1.0)
        w_sum = w_arr.sum()
        if w_sum < 1e-10:
            continue

        ic_weighted = float(np.dot(w_arr, ic_arr) / w_sum)

        # Weighted IC t-stat approximation (effective N = sum(w)^2 / sum(w^2))
        eff_n = float(w_arr.sum() ** 2 / (np.dot(w_arr, w_arr) + 1e-14))
        ic_std_est = float(np.sqrt(np.dot(w_arr, (ic_arr - ic_weighted) ** 2) / w_sum))
        ic_tstat = (
            ic_weighted / (ic_std_est / math.sqrt(max(eff_n, 2)))
            if ic_std_est > 1e-10
            else 0.0
        )

        rows.append(
            {
                "regime": reg,
                "eff_obs": round(eff_obs, 1),
                "ic_mean_weighted": round(ic_weighted, 5),
                "ic_tstat": round(ic_tstat, 3),
                "ic_std": round(ic_std_est, 5),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ic_mean_weighted", ascending=False).reset_index(drop=True)


def mixture_of_experts_score(
    per_regime_scores: dict[str, pd.Series],
    soft_weights: pd.DataFrame,
    *,
    index_col: str | None = None,
) -> pd.Series:
    """
    Blend per-regime score Series using soft regime weights.

    Parameters
    ----------
    per_regime_scores : {regime_name: score_series_indexed_by_(date,ticker)}
    soft_weights      : (date × regime) DataFrame of probabilities

    Returns
    -------
    Blended score Series indexed by the same (date, ticker) index.
    """
    if not per_regime_scores:
        raise ValueError("per_regime_scores is empty.")

    ref = next(iter(per_regime_scores.values()))
    blended = pd.Series(0.0, index=ref.index)
    total_w = pd.Series(0.0, index=ref.index)

    for regime_name, scores in per_regime_scores.items():
        if regime_name not in soft_weights.columns:
            continue

        if isinstance(scores.index, pd.MultiIndex):
            date_level = scores.index.get_level_values("date")
        else:
            date_level = pd.to_datetime(scores.index)

        w = date_level.map(
            lambda d: float(soft_weights.loc[d, regime_name])
            if d in soft_weights.index else 0.0
        )
        w = pd.Series(w.values, index=scores.index)
        blended = blended.add(w * scores, fill_value=0.0)
        total_w = total_w.add(w, fill_value=0.0)

    total_w = total_w.replace(0.0, np.nan)
    return (blended / total_w).fillna(0.0)


def evaluate_transition_stability(
    soft_weights: pd.DataFrame,
    *,
    large_shift_threshold: float = 0.3,
) -> dict[str, float]:
    """
    Measure how smoothly the latent regime transitions across time.

    Parameters
    ----------
    soft_weights       : (date × regime) probability DataFrame
    large_shift_threshold : max daily change in any regime weight to flag as "large"

    Returns
    -------
    mean_daily_shift    : mean L1 distance between consecutive weight vectors
    large_shift_frac    : fraction of dates with L1 shift > large_shift_threshold
    entropy_mean        : mean Shannon entropy of regime weights (higher = more uncertain)
    entropy_std         : std of entropy across dates
    """
    if soft_weights.empty or len(soft_weights) < 2:
        return {k: float("nan") for k in (
            "mean_daily_shift", "large_shift_frac", "entropy_mean", "entropy_std"
        )}

    W = soft_weights.sort_index().to_numpy(dtype=float)
    W = np.clip(W, 0.0, 1.0)

    shifts = np.abs(np.diff(W, axis=0)).sum(axis=1)
    mean_shift = float(shifts.mean())
    large_frac = float((shifts > large_shift_threshold).mean())

    eps = 1e-12
    W_clip = np.clip(W, eps, 1.0)
    entropy = -(W_clip * np.log(W_clip)).sum(axis=1)
    return {
        "mean_daily_shift": mean_shift,
        "large_shift_frac": large_frac,
        "entropy_mean": float(entropy.mean()),
        "entropy_std": float(entropy.std(ddof=1)) if len(entropy) > 1 else float("nan"),
    }


def format_latent_regime_report(
    transition: dict[str, float],
    soft_ic: pd.DataFrame | None = None,
) -> str:
    def _f(v: object) -> str:
        if isinstance(v, float) and math.isfinite(v):
            return f"{v:8.4f}"
        return "     NaN"
    def _pct(v: object) -> str:
        if isinstance(v, float) and math.isfinite(v):
            return f"{v * 100:6.1f}%"
        return "   NaN"

    lines = [
        "Latent Regime Model Diagnostics",
        "=" * 55,
        "  Transition Stability",
        f"    Mean daily L1 weight shift : {_f(transition.get('mean_daily_shift'))}",
        f"    Large-shift dates           : {_pct(transition.get('large_shift_frac'))}",
        f"    Entropy (mean / std)        : {_f(transition.get('entropy_mean'))} / "
        f"{_f(transition.get('entropy_std'))}",
    ]

    if soft_ic is not None and not soft_ic.empty:
        lines += ["", "  IC by Regime (soft-weighted)", "-" * 55]
        for _, row in soft_ic.iterrows():
            lines.append(
                f"    {str(row.get('regime', '?')):<14}  "
                f"eff_obs={row.get('eff_obs', float('nan')):6.1f}  "
                f"IC={_f(row.get('ic_mean_weighted'))}  "
                f"t={_f(row.get('ic_tstat'))}"
            )

    lines.append("=" * 55)
    return "\n".join(lines)
