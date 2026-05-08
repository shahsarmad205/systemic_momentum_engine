"""
P6/P9: Simple Baseline Scoring Methods
========================================
Equal-weight and IC-weighted feature combination baselines that compete
with ML models in the walk-forward evaluation pipeline.

These baselines are evaluated OUTSIDE the model fitting loop (P9) to
establish the "no-ML" floor that learned models must beat. If ML models
produce negative subsumption alpha, they are not adding value beyond what
a simple linear combination of features provides.

Architecture:
- EqualWeightBaseline: z-scores each feature on training data, then
  averages. This is invariant to feature scale differences.
- ICWeightedBaseline: weights features by their trailing out-of-sample
  IC (Spearman rank correlation with target), then averages.

Both are sklearn-compatible estimators (fit/predict) and are invoked
by `_evaluate_baseline_scorers()` directly on the prepared feature matrix.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator, RegressorMixin

import numpy as np
import pandas as pd


class EqualWeightBaseline(BaseEstimator, RegressorMixin):
    """
    Equal-weight average of feature z-scores.

    During fit: compute per-feature mean and std on training data.
    During predict: z-score each feature, then average.

    This is scale-invariant — features with larger ranges don't dominate.
    """

    def __init__(self):
        self.feature_means_ = None
        self.feature_stds_ = None
        self.n_features_in_ = 0

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_means_ = np.nanmean(X, axis=0)
        self.feature_stds_ = np.nanstd(X, axis=0).clip(min=1e-8)
        return self

    def predict(self, X) -> np.ndarray:
        """Return equal-weight composite of z-scored features."""
        X = np.asarray(X, dtype=float)
        if self.feature_means_ is None or self.feature_stds_ is None:
            return np.zeros(X.shape[0])
        X_z = (X - self.feature_means_) / self.feature_stds_
        valid = np.isfinite(X_z)
        X_safe = np.where(valid, X_z, 0.0)
        count = valid.sum(axis=1).clip(min=1)
        return X_safe.sum(axis=1) / count


class ICWeightedBaseline(BaseEstimator, RegressorMixin):
    """
    IC-weighted feature combination.

    During fit: compute per-feature IC (Spearman correlation with target)
    on the training data. Features with higher |IC| get higher weight.
    During predict: z-score features, weight by IC, then average.

    Parameters
    ----------
    ic_window : int
        Rolling window (number of dates) for IC computation.
    min_ic_obs : int
        Minimum observations before IC weights are used.
    """

    def __init__(
        self,
        ic_window: int = 126,
        min_ic_obs: int = 30,
    ):
        self.ic_window = int(ic_window)
        self.min_ic_obs = int(min_ic_obs)
        self.ic_weights_ = None
        self.feature_means_ = None
        self.feature_stds_ = None
        self.n_features_in_ = 0

    def fit(self, X, y=None, *, dates=None):
        """Compute IC weights and feature standardization from training data."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float) if y is not None else None
        self.n_features_in_ = X.shape[1]
        n_features = X.shape[1]

        # Feature standardization
        self.feature_means_ = np.nanmean(X, axis=0)
        self.feature_stds_ = np.nanstd(X, axis=0).clip(min=1e-8)

        if y is None or n_features == 0:
            self.ic_weights_ = np.ones(n_features) / max(n_features, 1)
            return self

        ic_means = np.zeros(n_features)

        if dates is not None and len(dates) == len(y):
            dates = np.asarray(dates)
            unique_dates = pd.unique(dates)
            if len(unique_dates) >= self.min_ic_obs:
                recent_dates = unique_dates[-self.ic_window:]
                mask = np.isin(dates, recent_dates)
                if mask.sum() >= self.min_ic_obs:
                    X_z = (X[mask] - self.feature_means_) / self.feature_stds_
                    for j in range(n_features):
                        x_col = X_z[:, j]
                        y_col = y[mask]
                        valid = np.isfinite(x_col) & np.isfinite(y_col)
                        if valid.sum() >= self.min_ic_obs:
                            ic_means[j] = _spearman_ic(x_col[valid], y_col[valid])
        else:
            # Fallback: compute IC on full training set
            X_z = (X - self.feature_means_) / self.feature_stds_
            for j in range(n_features):
                valid = np.isfinite(X_z[:, j]) & np.isfinite(y)
                if valid.sum() >= self.min_ic_obs:
                    ic_means[j] = _spearman_ic(X_z[valid, j], y[valid])

        # Absolute IC weighting: weight ∝ |IC| × sign(IC)
        abs_ic = np.abs(ic_means)
        total = abs_ic.sum()
        if total > 1e-12:
            self.ic_weights_ = ic_means / total
        else:
            self.ic_weights_ = np.ones(n_features) / max(n_features, 1)

        return self

    def predict(self, X) -> np.ndarray:
        """Return IC-weighted composite of z-scored features."""
        X = np.asarray(X, dtype=float)
        if self.ic_weights_ is None or len(self.ic_weights_) == 0:
            return np.zeros(X.shape[0])
        X_z = (X - self.feature_means_) / self.feature_stds_
        X_z = np.where(np.isfinite(X_z), X_z, 0.0)
        n_features = min(X_z.shape[1], len(self.ic_weights_))
        return X_z[:, :n_features] @ self.ic_weights_[:n_features]


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation between x and y."""
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5:
        return 0.0
    rho, _ = spearmanr(x[valid], y[valid])
    return float(rho) if np.isfinite(rho) else 0.0
