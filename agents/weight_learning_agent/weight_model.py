"""
Weight Learning Model
=======================
Learns optimal weights for the adjusted_score formula from historical
return data using regression.

Supports:
    - Ridge regression  (linear, interpretable — default)
    - Gradient Boosting (non-linear, higher capacity)
    - Logistic Regression (directional classification)

Walk-forward validation prevents lookahead bias.
Time-decay weighting gives more importance to recent observations.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import spearmanr
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score

try:
    from xgboost import XGBRegressor
except ImportError:  # optional dependency
    XGBRegressor = None  # type: ignore[misc,assignment]


# ------------------------------------------------------------------
# Feature names used in training (order matches LearnedWeights mapping)
# ------------------------------------------------------------------

COMPOUND_AND_PRICE_FEATURES = [
    "f_trend",
    "f_regional",
    "f_global",
    "f_social",
    "ret_5d",
    "ret_10d",
    "sector_relative_strength",
    "cs_momentum_percentile",
    # Volatility term-structure
    "rolling_vol_5",
    "rolling_vol_10",
    "rolling_vol_20",
    "rolling_vol_60",
    "vol_of_vol_20",
    "jump_indicator",
    # Volume / liquidity proxies
    "relative_volume",
    "volume_zscore",
    # Cross-ticker
    "rolling_corr_market_20",
    # Regime / volatility context
    "is_high_vol_regime",
]


# ------------------------------------------------------------------
# Learned weights container (serialisable)
# ------------------------------------------------------------------

@dataclass
class LearnedWeights:
    w_trend: float = 1.0
    w_regional: float = 0.0
    w_global: float = 0.0
    w_social: float = 0.0
    w_ret_5d: float = 0.0
    w_ret_10d: float = 0.0
    w_vol_10: float = 0.0
    w_vol: float = 0.0
    w_rel_vol: float = 0.0
    w_vol_zscore: float = 0.0
    w_corr_market: float = 0.0
    intercept: float = 0.0

    model_type: str = "ridge"
    train_start: str = ""
    train_end: str = ""
    n_samples: int = 0

    r2: float = 0.0
    mae: float = 0.0
    directional_accuracy: float = 0.0
    ic: float = 0.0
    score_scale: float = 1.0  # scale raw prediction to match rule-based score magnitude for thresholds
    score_direction: int = 1   # +1 = higher score => long; -1 = invert (use when train/OOS IC < 0)
    target_type: str = "regression"
    auc_score: float = 0.0  # only meaningful for classification

    def compute_adjusted_score(
        self,
        f_trend: float,
        f_regional: float = 0.0,
        f_global: float = 0.0,
        f_social: float = 0.0,
        ret_5d: float = 0.0,
        ret_10d: float = 0.0,
        rolling_vol_10: float = 0.0,
        rolling_vol: float = 0.0,
        relative_volume: float = 0.0,
        volume_zscore: float = 0.0,
        rolling_corr_market: float = 0.0,
    ) -> float:
        raw = (
            self.intercept
            + self.w_trend * f_trend
            + self.w_regional * f_regional
            + self.w_global * f_global
            + self.w_social * f_social
            + self.w_ret_5d * ret_5d
            + self.w_ret_10d * ret_10d
            + getattr(self, "w_vol_10", 0) * rolling_vol_10
            + self.w_vol * rolling_vol
            + self.w_rel_vol * relative_volume
            + getattr(self, "w_vol_zscore", 0) * volume_zscore
            + getattr(self, "w_corr_market", 0) * rolling_corr_market
        )
        raw_scaled = getattr(self, "score_scale", 1.0) * raw
        direction = getattr(self, "score_direction", 1)
        return raw_scaled * direction

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> LearnedWeights:
        with open(path) as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> LearnedWeights:
        """Load from a dict (e.g. one entry of regime_models)."""
        for key in (
            "w_ret_5d", "w_ret_10d", "w_vol_10", "w_vol", "w_rel_vol",
            "w_vol_zscore", "w_corr_market", "score_scale", "score_direction",
        ):
            if key == "score_scale":
                data.setdefault(key, 1.0)
            elif key == "score_direction":
                data.setdefault(key, 1)
            else:
                data.setdefault(key, 0.0)
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in valid})


def load_regime_weights(path: str) -> tuple[dict[str, LearnedWeights] | None, list[str] | None]:
    """
    Load weights file. If it contains regime_models, return (regime_weights, active_features).
    Otherwise return (None, None); caller should use LearnedWeights.load(path) for single weights.
    """
    with open(path) as fh:
        data = json.load(fh)
    if "regime_models" not in data:
        return (None, None)
    models = {}
    for regime, d in data["regime_models"].items():
        models[regime] = LearnedWeights.from_dict(dict(d))
    return (models, data.get("active_features", []))


def save_regime_weights(
    path: str,
    regime_models: dict[str, LearnedWeights],
    active_features: list[str],
    default_regime: str = "normal",
) -> None:
    """Save regime-specific weights in the extended format."""
    payload = {
        "regime_models": {r: w.to_dict() for r, w in regime_models.items()},
        "active_features": active_features,
        "default_regime": default_regime,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


# ------------------------------------------------------------------
# Weight Learner
# ------------------------------------------------------------------

TARGET = "forward_return"


def _make_binary_labels(forward_returns: pd.Series) -> np.ndarray:
    """
    Convert forward returns to binary classification labels.

    Convention:
        1 = forward_return > 0 (price up)
        0 = forward_return <= 0 (price flat or down)

    Using a single helper ensures the convention is identical across
    fit(), walk_forward_validate(), and any other evaluation code.
    """
    return (forward_returns.values > 0).astype(np.int64)


class WeightLearner:
    """
    Train a regression model on historical features to learn the optimal
    weights for the adjusted_score formula.

    All input features are standardized to mean=0, std=1 before fitting.
    Ridge uses low default alpha (0.01) so weights can move away from zero.
    GBR and RF are evaluated for non-linear fit; deployable weights are
    taken from a Ridge fit on the same data so we always have non-zero
    interpretable weights.
    """

    def __init__(
        self,
        model_type: str = "ridge",
        alpha: float = 0.01,
        time_decay_lambda: float = 0.001,
        target_type: str = "regression",  # or 'classification'
        return_target_type: str = "raw",  # 'raw' | 'excess' | 'sharpe_scaled'
    ):
        self.model_type = model_type
        self.alpha = alpha
        self.time_decay_lambda = time_decay_lambda
        self.target_type = target_type
        self.return_target_type = return_target_type
        # Hyperparameter tuning flag: when True, fit() may call tune_hyperparams
        self.tune: bool = False
        self.model = None
        self.ridge_for_weights = None  # used when model is gbr/rf to get deployable weights
        self.ensemble_models: dict[str, object] | None = None
        self.ensemble_weights: dict[str, float] | None = None
        self.scaler: StandardScaler | None = None
        self.active_features: list[str] = []
        self._train_metrics: dict = {}
        self.regime_models: dict[str, LearnedWeights] = {}

    # ==============================================================
    # Fit
    # ==============================================================

    def fit(self, df: pd.DataFrame, feature_cols: list[str] | None = None) -> LearnedWeights:
        """
        Train the model on *df* and return LearnedWeights.
        Only features with non-zero variance are included.
        """
        cols = feature_cols or [
            c for c in COMPOUND_AND_PRICE_FEATURES
            if c in df.columns and df[c].std() > 1e-12
        ]
        if not cols:
            raise ValueError("No features with non-zero variance found")

        # Work on a copy so we can safely drop rows with missing values for training.
        train_df = df.copy()

        # Select and attach the appropriate target variant as TARGET.
        train_df[TARGET] = self._select_target_series(train_df)

        # Drop any rows with NaNs in the active feature columns or in the target.
        feature_mask = train_df[cols].notna().all(axis=1)
        if TARGET in train_df.columns:
            feature_mask &= train_df[TARGET].notna()

        n_before = len(train_df)
        train_df = train_df.loc[feature_mask].copy()
        n_after = len(train_df)
        if n_after < n_before:
            logging.getLogger(__name__).info(
                "Dropped %d rows with NaNs in features/target (kept %d)",
                n_before - n_after,
                n_after,
            )
        if train_df.empty:
            raise ValueError("No samples left after dropping rows with NaNs in features/target")

        self.active_features = cols
        X = train_df[cols].values.astype(np.float64)

        # ISSUE 1 diagnostic: identify which feature causes variance disparity
        logger = logging.getLogger(__name__)
        col_stds = pd.Series(np.std(X, axis=0), index=cols).sort_values(ascending=False)
        logger.debug("Feature stds:\n%s", col_stds.to_string())
        worst = col_stds.index[0]
        best = col_stds.index[-1]
        print(f"  Highest std feature: {worst} = {col_stds.iloc[0]:.4f}")
        print(f"  Lowest std feature:  {best}  = {col_stds.iloc[-1]:.6f}")

        # Pre-scaling sanity check for extreme variance disparity
        col_stds_arr = np.std(X, axis=0)
        if col_stds_arr.min() > 1e-12 and col_stds_arr.max() / col_stds_arr.min() > 100:
            warnings.warn(
                "Feature matrix has extreme variance disparity "
                f"(max_std/min_std = {col_stds_arr.max() / col_stds_arr.min():.1f}x). "
                "Check that all features are on comparable scales before "
                "StandardScaler is applied.",
                UserWarning,
            )

        sample_weight = self._compute_sample_weights(train_df)

        # Standardize all inputs to mean=0, std=1. Same scaler can be used at inference
        # for scale-then-predict; we convert coefficients to original space so
        # inference uses: score = intercept + sum(w_i * x_i) with no explicit scaling.
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        vol_idx = [i for i, c in enumerate(cols) if "vol" in c.lower()]
        for i in vol_idx:
            logger.debug(
                "Scaler for %s: mean=%.4f scale=%.4f",
                cols[i],
                self.scaler.mean_[i],
                self.scaler.scale_[i],
            )

        if self.target_type == "classification":
            y = _make_binary_labels(train_df[TARGET])
            print(f"[DEBUG] Building model of type: {self.model_type} (target_type=classification)")

            if self.model_type == "ensemble":
                # Ensemble: logistic + gradient boosting classifier.
                logit = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                gbc = GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                )
                logit.fit(X_scaled, y, sample_weight=sample_weight)
                gbc.fit(X_scaled, y, sample_weight=sample_weight)

                self.ensemble_models = {"logistic": logit, "gbc": gbc}
                self.ensemble_weights = {"logistic": 0.5, "gbc": 0.5}

                # For linear deployable weights, still use a small ridge on forward_return.
                y_cont = train_df[TARGET].values.astype(np.float64)
                self.model = logit
                self.ridge_for_weights = Ridge(alpha=0.01, fit_intercept=True)
                self.ridge_for_weights.fit(X_scaled, y_cont, sample_weight=sample_weight)

                # Compute training metrics from ensemble probabilities.
                prob_logit = logit.predict_proba(X_scaled)[:, 1]
                prob_gbc = gbc.predict_proba(X_scaled)[:, 1]
                y_prob = 0.5 * prob_logit + 0.5 * prob_gbc
                y_pred_class = (y_prob >= 0.5).astype(np.int64)
                acc = float(accuracy_score(y, y_pred_class))
                try:
                    auc = float(roc_auc_score(y, y_prob))
                except ValueError:
                    auc = 0.5
                self._train_metrics = {
                    "r2": 0.0,
                    "mae": 0.0,
                    "directional_accuracy": round(acc, 4),
                    "ic": 0.0,
                    "auc_score": round(auc, 4),
                    "ensemble_model_types": ["logistic", "gbc"],
                    "ensemble_weights": self.ensemble_weights,
                }
                # Use ridge-based predictions to set score scale.
                raw_pred = self.ridge_for_weights.predict(X_scaled)
                std_pred = float(np.std(raw_pred)) + 1e-8
                self._score_scale = 0.5 / std_pred
                return self.get_weights(df)

            # Classification: branch on model_type so --model ridge/gbr/rf/logistic is respected.
            if self.model_type in ("ridge", "logistic"):
                self.model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                self.model.fit(X_scaled, y, sample_weight=sample_weight)
                classes = list(self.model.classes_)
                self._pos_class_index = classes.index(1) if 1 in classes else 0
                self.ridge_for_weights = None
                self.ensemble_models = None
                self.ensemble_weights = None
                self._compute_train_metrics(X_scaled, y, train_df)
                raw_pred = self.model.decision_function(X_scaled)
                std_pred = float(np.std(raw_pred)) + 1e-8
                self._score_scale = 0.5 / std_pred
                return self.get_weights(df)

            if self.model_type == "gbr":
                self.model = GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                )
                self.model.fit(X_scaled, y, sample_weight=sample_weight)
                classes = list(self.model.classes_)
                self._pos_class_index = int(classes.index(1)) if 1 in classes else 0
                y_cont = train_df[TARGET].values.astype(np.float64)
                self.ridge_for_weights = Ridge(alpha=0.01, fit_intercept=True)
                self.ridge_for_weights.fit(X_scaled, y_cont, sample_weight=sample_weight)
                self.ensemble_models = None
                self.ensemble_weights = None
                self._compute_train_metrics(X_scaled, y, train_df)
                raw_pred = self.ridge_for_weights.predict(X_scaled)
                std_pred = float(np.std(raw_pred)) + 1e-8
                self._score_scale = 0.5 / std_pred
                return self.get_weights(df)

            if self.model_type == "rf":
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_leaf=20,
                    random_state=42,
                )
                self.model.fit(X_scaled, y, sample_weight=sample_weight)
                classes = list(self.model.classes_)
                self._pos_class_index = int(classes.index(1)) if 1 in classes else 0
                y_cont = train_df[TARGET].values.astype(np.float64)
                self.ridge_for_weights = Ridge(alpha=0.01, fit_intercept=True)
                self.ridge_for_weights.fit(X_scaled, y_cont, sample_weight=sample_weight)
                self.ensemble_models = None
                self.ensemble_weights = None
                self._compute_train_metrics(X_scaled, y, df)
                raw_pred = self.ridge_for_weights.predict(X_scaled)
                std_pred = float(np.std(raw_pred)) + 1e-8
                self._score_scale = 0.5 / std_pred
                return self.get_weights(df)

            raise ValueError(f"Unknown model_type for classification: {self.model_type}")

        y = train_df[TARGET].values.astype(np.float64)
        print(f"[DEBUG] Building model of type: {self.model_type} (target_type=regression)")

        if self.tune and self.model_type in ("ridge", "gbr"):
            # Hyperparameter tuning for ridge / GBR using time-series CV on scaled X, forward_return y.
            tuned = self.tune_hyperparams(X_scaled, y)
            self.model = tuned
            # Optional: keep alpha in sync for ridge so downstream uses consistent regularisation
            if isinstance(tuned, Ridge):
                self.alpha = float(getattr(tuned, "alpha", self.alpha))
            self.ridge_for_weights = None
            self.ensemble_models = None
            self.ensemble_weights = None

            if self.model_type == "gbr":
                # For deployable linear weights, still fit a small ridge on the tuned predictions.
                self.ridge_for_weights = Ridge(alpha=0.01, fit_intercept=True)
                self.ridge_for_weights.fit(X_scaled, y, sample_weight=sample_weight)
        else:
            if self.model_type == "ridge":
                self.model = Ridge(alpha=self.alpha, fit_intercept=True)
                self.model.fit(X_scaled, y, sample_weight=sample_weight)
                self.ridge_for_weights = None
                self.ensemble_models = None
                self.ensemble_weights = None

            elif self.model_type == "gbr":
                self.model = GradientBoostingRegressor(
                    n_estimators=200, max_depth=3, learning_rate=0.05,
                    subsample=0.8, random_state=42,
                )
                self.model.fit(X_scaled, y, sample_weight=sample_weight)
                # Ridge on same data for deployable non-zero weights
                self.ridge_for_weights = Ridge(alpha=0.01, fit_intercept=True)
                self.ridge_for_weights.fit(X_scaled, y, sample_weight=sample_weight)
                self.ensemble_models = None
                self.ensemble_weights = None

            elif self.model_type == "rf":
                self.model = RandomForestRegressor(
                    n_estimators=100, max_depth=5, min_samples_leaf=20,
                    random_state=42,
                )
                self.model.fit(X_scaled, y, sample_weight=sample_weight)
                self.ridge_for_weights = Ridge(alpha=0.01, fit_intercept=True)
                self.ridge_for_weights.fit(X_scaled, y, sample_weight=sample_weight)
                self.ensemble_models = None
                self.ensemble_weights = None

            elif self.model_type == "logistic":
                y_dir = np.sign(y).astype(int)
                self.model = LogisticRegression(
                    C=1.0 / max(self.alpha, 0.01),
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                )
                self.model.fit(X_scaled, y_dir, sample_weight=sample_weight)
                # Deployable weights from Ridge (regression) for consistent score scale
                self.ridge_for_weights = Ridge(alpha=0.01, fit_intercept=True)
                self.ridge_for_weights.fit(X_scaled, y, sample_weight=sample_weight)
                self.ensemble_models = None
                self.ensemble_weights = None

            elif self.model_type == "ensemble":
                # Regression ensemble: ridge + gradient boosting regressor.
                ridge = Ridge(alpha=self.alpha, fit_intercept=True)
                gbr = GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                )
                ridge.fit(X_scaled, y, sample_weight=sample_weight)
                gbr.fit(X_scaled, y, sample_weight=sample_weight)

                self.ensemble_models = {"ridge": ridge, "gbr": gbr}
                self.ensemble_weights = {"ridge": 0.5, "gbr": 0.5}

                # Use ridge for deployable linear weights
                self.model = ridge
                self.ridge_for_weights = ridge

                # Ensemble predictions for metrics and score scaling
                y_pred_ridge = ridge.predict(X_scaled)
                y_pred_gbr = gbr.predict(X_scaled)
                y_pred = 0.5 * y_pred_ridge + 0.5 * y_pred_gbr

                corr = float(np.corrcoef(y_pred, y)[0, 1]) if len(y) > 2 else 0.0
                self._train_metrics = {
                    "r2": round(float(r2_score(y, y_pred)), 4),
                    "mae": round(float(mean_absolute_error(y, y_pred)), 6),
                    "directional_accuracy": round(float((np.sign(y_pred) == np.sign(y)).mean()), 4),
                    "ic": round(corr, 4),
                    "auc_score": 0.0,
                    "ensemble_model_types": ["ridge", "gbr"],
                    "ensemble_weights": self.ensemble_weights,
                }

                std_pred = float(np.std(y_pred)) + 1e-8
                self._score_scale = 0.5 / std_pred
                return self.get_weights(df)

            elif self.model_type == "xgb":
                if XGBRegressor is None:
                    raise ImportError(
                        "Model type 'xgb' requested but the 'xgboost' package is "
                        "not installed. Install it with 'pip install xgboost' "
                        "or choose a different --model."
                    )
                # Base XGBoost model; start from aggressively regularised
                # configuration and optionally update with tuned params.
                base_params = {
                    "objective": "reg:squarederror",
                    "random_state": 42,
                    "tree_method": "hist",
                    "n_jobs": -1,
                    "max_depth": 3,
                    "min_child_weight": 50,
                    "reg_alpha": 1.0,
                    "reg_lambda": 5.0,
                    "subsample": 0.6,
                    "colsample_bytree": 0.5,
                    "n_estimators": 200,
                    "eval_metric": "rmse",
                }
                best_params = getattr(self, "_best_params_", {})
                base_params.update(best_params)
                self.model = XGBRegressor(**base_params)

                # Time-ordered train/validation split for early stopping
                n = X_scaled.shape[0]
                idx = np.arange(n)
                if "date" in train_df.columns:
                    # Preserve temporal order
                    order = np.argsort(train_df["date"].values)
                    idx = idx[order]

                split = int(max(0.8 * len(idx), 1))
                train_idx = idx[:split]
                val_idx = idx[split:]
                if len(val_idx) < 10:
                    # Fallback: no dedicated validation, fit on all data
                    self.model.fit(
                        X_scaled,
                        y,
                        sample_weight=sample_weight,
                    )
                else:
                    # For xgboost>=2, early stopping is handled via callbacks; to
                    # keep the dependency surface simple we omit it here and rely
                    # on the tuned number of trees instead.
                    self.model.fit(
                        X_scaled[train_idx],
                        y[train_idx],
                        sample_weight=sample_weight[train_idx] if sample_weight is not None else None,
                        eval_set=[(X_scaled[val_idx], y[val_idx])],
                        verbose=False,
                    )

                # Linear deployable weights: small ridge on full data
                self.ridge_for_weights = Ridge(alpha=0.01, fit_intercept=True)
                self.ridge_for_weights.fit(X_scaled, y, sample_weight=sample_weight)
                self.ensemble_models = None
                self.ensemble_weights = None

        self._compute_train_metrics(X_scaled, y, train_df)

        # Scale raw predictions to match rule-based score magnitude so thresholds 0.5 / -0.5 yield signals
        deploy_model = self.ridge_for_weights if self.ridge_for_weights is not None else self.model
        raw_pred = deploy_model.predict(X_scaled)
        std_pred = float(np.std(raw_pred)) + 1e-8
        self._score_scale = 0.5 / std_pred  # target std ~0.5 so scores span ~[-0.5, 0.5] and exceed thresholds

        # For XGBoost, log feature importances ranked by gain (via booster).
        if self.model_type == "xgb":
            try:
                booster = self.model.get_booster()
                gain_imp = booster.get_score(importance_type="gain")
                # Map f{i} keys back to active feature names
                mapping = {f"f{i}": name for i, name in enumerate(self.active_features)}
                items = []
                for key, val in gain_imp.items():
                    feat = mapping.get(key, key)
                    items.append((feat, float(val)))
                if items:
                    items.sort(key=lambda x: x[1], reverse=True)
                    logger = logging.getLogger(__name__)
                    logger.info("XGBoost feature importances (gain):")
                    for feat, val in items:
                        logger.info("  %s: %.6f", feat, val)
            except Exception:
                pass

        return self.get_weights(train_df)

    def save_scaler(self, path: str) -> None:
        """Save scaler mean/scale and feature order so inference can use identical scaling if needed."""
        if self.scaler is None:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "active_features": self.active_features,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": (np.where(self.scaler.scale_ > 1e-12, self.scaler.scale_, 1.0)).tolist(),
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

    # ==============================================================
    # Extract weights in original feature space
    # ==============================================================

    def get_weights(self, df: pd.DataFrame | None = None) -> LearnedWeights:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted yet")

        # Use Ridge coefficients for deployable weights (Ridge uses its own; GBR/RF use Ridge fallback)
        # For classification: use model.coef_ if linear (logistic), else ridge_for_weights (gbr/rf).
        if self.target_type == "classification":
            if hasattr(self.model, "coef_") and self.model.coef_ is not None:
                coef = np.asarray(self.model.coef_, dtype=np.float64).ravel()
                inter = float(self.model.intercept_[0])
            elif self.ridge_for_weights is not None:
                coef = self.ridge_for_weights.coef_.ravel().astype(np.float64)
                inter = float(self.ridge_for_weights.intercept_)
            else:
                coef = np.zeros(len(self.active_features))
                inter = 0.0
        elif self.ridge_for_weights is not None:
            coef = self.ridge_for_weights.coef_.ravel().astype(np.float64)
            inter = self.ridge_for_weights.intercept_
        elif self.model_type == "ridge":
            coef = np.asarray(self.model.coef_, dtype=np.float64).ravel()
            inter = self.model.intercept_
        else:
            coef = np.zeros(len(self.active_features))
            inter = 0.0

        scale = self.scaler.scale_
        mean = self.scaler.mean_
        # Avoid division by zero for constant features
        scale = np.where(scale > 1e-12, scale, 1.0)
        w_orig = coef / scale
        intercept = float(inter) - float(np.dot(coef, mean / scale))

        weight_map = dict(zip(self.active_features, w_orig.tolist()))

        train_start = ""
        train_end = ""
        n_samples = 0
        if df is not None and "date" in df.columns:
            train_start = str(df["date"].min())[:10]
            train_end = str(df["date"].max())[:10]
            n_samples = len(df)

        return LearnedWeights(
            w_trend=weight_map.get("f_trend", 0.0),
            w_regional=weight_map.get("f_regional", 0.0),
            w_global=weight_map.get("f_global", 0.0),
            w_social=weight_map.get("f_social", 0.0),
            w_ret_5d=weight_map.get("ret_5d", 0.0),
            w_ret_10d=weight_map.get("ret_10d", 0.0),
            w_vol_10=weight_map.get("rolling_vol_10", 0.0),
            w_vol=weight_map.get("rolling_vol_20", 0.0),
            w_rel_vol=weight_map.get("relative_volume", 0.0),
            w_vol_zscore=weight_map.get("volume_zscore", 0.0),
            w_corr_market=weight_map.get("rolling_corr_market_20", 0.0),
            intercept=round(intercept, 8),
            model_type=self.model_type,
            train_start=train_start,
            train_end=train_end,
            n_samples=n_samples,
            score_scale=getattr(self, "_score_scale", 1.0),
            score_direction=(-1 if (self._train_metrics.get("ic") or 0) < 0 else 1),
            target_type=getattr(self, "target_type", "regression"),
            **self._train_metrics,
        )

    # ==============================================================
    # SHAP analysis for fitted models
    # ==============================================================

    def analyze_shap(
        self,
        df: pd.DataFrame,
        max_samples: int = 500,
        output_dir: str = "output/learning",
    ) -> None:
        """
        Compute SHAP values on a sample and generate:
          - Bar plot of mean |SHAP| per feature
          - Beeswarm plot of SHAP distribution
        Also writes a CSV with mean |SHAP| by feature and by year
        (using df['date'] when available) to detect instability.

        This is intended for research diagnostics; it assumes the
        learner has already been fitted.
        """
        if self.model is None or self.scaler is None or not self.active_features:
            print("analyze_shap: model or scaler not fitted; skipping.")
            return

        # Choose a model suitable for SHAP TreeExplainer when possible
        model_for_shap = None
        if isinstance(
            self.model,
            (
                GradientBoostingRegressor,
                GradientBoostingClassifier,
                RandomForestRegressor,
                RandomForestClassifier,
            ),
        ):
            model_for_shap = self.model
        elif self.ensemble_models is not None:
            # Prefer tree-based member of the ensemble if available
            for name in ("gbr", "gbc", "gbm"):
                if name in self.ensemble_models and isinstance(
                    self.ensemble_models[name],
                    (GradientBoostingRegressor, GradientBoostingClassifier),
                ):
                    model_for_shap = self.ensemble_models[name]
                    break

        if model_for_shap is None:
            print("analyze_shap: SHAP analysis currently implemented for tree-based models only.")
            return

        # Prepare sample
        df_sorted = df.sort_values("date") if "date" in df.columns else df.copy()
        X_full = df_sorted[self.active_features].values.astype(np.float64)
        if X_full.shape[0] == 0:
            print("analyze_shap: empty dataframe; skipping.")
            return
        n = X_full.shape[0]
        n_sample = min(max_samples, n)
        idx = np.linspace(0, n - 1, n_sample, dtype=int)
        X_sample = X_full[idx]
        X_sample_scaled = self.scaler.transform(X_sample)

        # Fit SHAP explainer and compute values
        try:
            explainer = shap.TreeExplainer(model_for_shap)
            shap_values = explainer.shap_values(X_sample_scaled)
        except Exception as exc:
            print(f"analyze_shap: error computing SHAP values: {exc}")
            return

        # For classifiers, shap_values can be a list (per class); take positive class if so
        if isinstance(shap_values, list):
            # use class 1 if available, else first
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        shap_values = np.asarray(shap_values, dtype=np.float64)
        if shap_values.ndim != 2 or shap_values.shape[1] != len(self.active_features):
            print("analyze_shap: unexpected SHAP shape; skipping plots.")
            return

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Bar plot of mean |SHAP|
        plt.figure(figsize=(8, 4))
        shap.summary_plot(
            shap_values,
            features=None,
            feature_names=self.active_features,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()
        bar_path = out_path / "shap_summary_bar.png"
        plt.savefig(bar_path)
        plt.close()

        # Beeswarm plot
        plt.figure(figsize=(8, 4))
        shap.summary_plot(
            shap_values,
            X_sample_scaled,
            feature_names=self.active_features,
            show=False,
        )
        plt.tight_layout()
        beeswarm_path = out_path / "shap_summary_beeswarm.png"
        plt.savefig(beeswarm_path)
        plt.close()

        # Also write a single main summary image path for convenience
        # (bar plot as primary summary)
        bar_path.rename(out_path / "shap_summary.png")

        # Average |SHAP| per feature and over time
        abs_shap = np.abs(shap_values)
        shap_df = pd.DataFrame(abs_shap, columns=self.active_features)
        if "date" in df_sorted.columns:
            shap_df["date"] = df_sorted["date"].iloc[idx].values
            shap_df["year"] = pd.to_datetime(shap_df["date"]).dt.year
            by_year = (
                shap_df.groupby("year")[self.active_features]
                .mean()
                .reset_index()
            )
            by_year_path = out_path / "shap_feature_stability_by_year.csv"
            by_year.to_csv(by_year_path, index=False)

        # Global mean |SHAP| per feature
        global_mean = shap_df[self.active_features].mean().reset_index()
        global_mean.columns = ["feature", "mean_abs_shap"]
        global_path = out_path / "shap_feature_importance.csv"
        global_mean.to_csv(global_path, index=False)

        print(f"SHAP summary bar/beeswarm plots written under {out_path}")

    # ==============================================================
    # Walk-forward validation
    # ==============================================================

    def walk_forward_validate(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
    ) -> list[dict]:
        """
        Expanding-window walk-forward cross-validation with an embargo gap
        between train and test windows.

        The embargo is fixed at 5 trading days to match the default
        5-day forward-return horizon, so the last 5 training dates do
        not overlap with the test labels' look-ahead window.

        Returns a list of per-split result dicts with metrics and weights.
        """
        dates = np.sort(df["date"].unique())
        split_size = len(dates) // n_splits

        # Embargo length in trading days between train and test.
        EMBARGO_DAYS = 5

        results: list[dict] = []

        for k in range(1, n_splits):
            cutoff_idx = k * split_size
            test_end_idx = min((k + 1) * split_size, len(dates) - 1)

            # First date in the test window
            test_start = dates[cutoff_idx]
            test_end = dates[test_end_idx]

            # End date for the training window with an embargo gap:
            # exclude the last EMBARGO_DAYS dates before test_start.
            train_end_idx = max(0, cutoff_idx - EMBARGO_DAYS)
            train_end = dates[train_end_idx]

            train_df = df[df["date"] < train_end]
            test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]

            if len(train_df) < 100 or len(test_df) < 20:
                continue

            # ----------------------------------------------------------
            # Fold-specific feature selection on training data only.
            # Keep features with |IC(train)| > 0.02 vs forward_return.
            # ----------------------------------------------------------
            if "forward_return" in train_df.columns:
                y_train_ic = train_df["forward_return"].astype(float).values
            else:
                # Fallback: use TARGET column if forward_return not available.
                y_train_ic = train_df[TARGET].astype(float).values

            candidate_features = [
                c
                for c in COMPOUND_AND_PRICE_FEATURES
                if c in train_df.columns and train_df[c].std() > 1e-12
            ]

            # Helper to compute IC-based selection for a given threshold.
            def _select_by_ic(threshold: float) -> list[str]:
                feats: list[str] = []
                for col in candidate_features:
                    x = train_df[col].values.astype(float)
                    mask = np.isfinite(x) & np.isfinite(y_train_ic)
                    if mask.sum() <= 2:
                        continue
                    try:
                        rho, _ = spearmanr(x[mask], y_train_ic[mask])
                        if np.isfinite(rho) and abs(float(rho)) > threshold:
                            feats.append(col)
                    except Exception:
                        continue
                return feats

            selected_features: list[str] = _select_by_ic(0.02)
            effective_threshold = 0.02

            if len(selected_features) < 5:
                # Relax threshold within this fold only.
                relaxed = _select_by_ic(0.01)
                if len(relaxed) >= 3:
                    selected_features = relaxed
                    effective_threshold = 0.01
                    print(
                        f"⚠ Fold {k}: only {len(selected_features)} features passed "
                        f"|IC(train)| > 0.02, using relaxed threshold > 0.01"
                    )
                else:
                    # Too few even with relaxed threshold — disable IC filtering
                    # and let Ridge regularisation handle noisy dimensions.
                    selected_features = candidate_features
                    effective_threshold = 0.0
                    print(
                        f"⚠ Fold {k}: only {len(relaxed)} features passed relaxed "
                        f"|IC(train)| > 0.01, disabling IC filter (using all "
                        f"{len(candidate_features)} features)"
                    )

            if effective_threshold > 0:
                print(
                    f"  Fold {k}: selected {len(selected_features)}/{len(candidate_features)} "
                    f"features with |IC(train)| > {effective_threshold:.2f}"
                )

            fold_learner = WeightLearner(
                model_type=self.model_type,
                alpha=self.alpha,
                time_decay_lambda=self.time_decay_lambda,
                target_type=self.target_type,
                return_target_type=self.return_target_type,
            )
            # Propagate tuning flag into each fold if enabled on the parent learner.
            fold_learner.tune = getattr(self, "tune", False)
            fold_weights = fold_learner.fit(train_df, feature_cols=selected_features)

            # Drop any rows in the test fold that still contain NaNs in the
            # active feature columns or in the target. This prevents downstream
            # models (e.g. Ridge) from seeing NaNs at prediction time.
            X_test_df = test_df[fold_learner.active_features].copy()
            mask_test = X_test_df.notna().all(axis=1)
            if TARGET in test_df.columns:
                mask_test &= test_df[TARGET].notna()
            test_df_clean = test_df.loc[mask_test].copy()
            X_test = X_test_df.loc[mask_test].values
            if X_test.size == 0 or len(test_df_clean) == 0:
                # No valid test rows for this split after cleaning; skip.
                continue
            X_test_scaled = fold_learner.scaler.transform(X_test)
            y_true = test_df_clean[TARGET].values

            if self.target_type == "classification":
                y_test_binary = _make_binary_labels(test_df_clean[TARGET])
                classes = list(fold_learner.model.classes_)
                pos_class_idx = classes.index(1) if 1 in classes else 0
                y_pred_proba = fold_learner.model.predict_proba(X_test_scaled)[:, pos_class_idx]
                y_pred_class = (y_pred_proba >= 0.5).astype(np.int64)
                dir_acc = float(accuracy_score(y_test_binary, y_pred_class))
                try:
                    auc = float(roc_auc_score(y_test_binary, y_pred_proba))
                except ValueError:
                    auc = 0.5
                # Information coefficient (Spearman rank correlation between scores and forward returns)
                ic = None
                if len(y_true) > 2:
                    try:
                        ic_val, _ = spearmanr(y_pred_proba, y_true)
                        ic = float(ic_val) if np.isfinite(ic_val) else 0.0
                    except Exception:
                        ic = 0.0
                effective_auc = auc if auc >= 0.5 else 1.0 - auc
                # AUC and Dir can disagree with class imbalance or small splits; warn only.
                if abs(auc - 0.5) > 0.03 and abs(dir_acc - 0.5) > 0.03:
                    if (auc >= 0.5) != (dir_acc >= 0.5):
                        _logger = logging.getLogger(__name__)
                        _logger.warning(
                            "Split %s: AUC=%.4f and Dir=%.4f on opposite sides of 0.5 "
                            "(can happen with class imbalance or small validation set).",
                            k, auc, dir_acc,
                        )
                r2 = None
                mae = None
                ic = None
            elif self.model_type == "logistic":
                y_pred = fold_learner.model.predict(X_test_scaled)
                y_true_dir = np.sign(y_true).astype(int)
                # Binary labels for AUC/IC: 1 = up, 0 = down (match _make_binary_labels)
                y_test_binary = (y_true > 0).astype(np.int64)
                classes = list(fold_learner.model.classes_)
                pos_idx = classes.index(1) if 1 in classes else 0
                y_pred_proba = fold_learner.model.predict_proba(X_test_scaled)[:, pos_idx]
                # Debug: diagnose IC=0, AUC=0 with non-constant Dir Acc
                unique_preds = np.unique(y_pred)
                print(f"  [logistic debug] y_pred unique values: {unique_preds}")
                print(
                    f"  [logistic debug] y_pred_proba range: "
                    f"[{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]"
                )
                print(
                    f"  [logistic debug] y_test distribution: "
                    f"{np.bincount(y_test_binary.astype(int))}"
                )
                if len(np.unique(y_test_binary)) < 2:
                    continue
                dir_acc = float((y_pred == y_true_dir).mean())
                r2 = 0.0
                mae = 0.0
                if len(y_true) > 2:
                    try:
                        ic_val, _ = spearmanr(y_pred_proba, y_true)
                        ic = float(ic_val) if np.isfinite(ic_val) else 0.0
                    except Exception:
                        ic = 0.0
                else:
                    ic = 0.0
                try:
                    auc = float(roc_auc_score(y_test_binary, y_pred_proba))
                except ValueError:
                    auc = 0.5
            else:
                y_pred = fold_learner.model.predict(X_test_scaled)
                r2 = float(r2_score(y_true, y_pred))
                mae = float(mean_absolute_error(y_true, y_pred))
                dir_acc = float((np.sign(y_pred) == np.sign(y_true)).mean())
                # IC vs raw forward_return where available
                if "forward_return" in test_df_clean.columns:
                    fwd = test_df_clean["forward_return"].values.astype(float)
                else:
                    fwd = y_true
                if len(fwd) > 2:
                    try:
                        ic_val, _ = spearmanr(y_pred, fwd)
                        ic = float(ic_val) if np.isfinite(ic_val) else 0.0
                    except Exception:
                        ic = 0.0
                else:
                    ic = 0.0
                y_true_bin_reg = (y_true > 0).astype(np.int64)
                try:
                    auc = float(roc_auc_score(y_true_bin_reg, y_pred))
                except ValueError:
                    auc = 0.5

            if dir_acc < 0.50 and self.target_type == "regression":
                print(
                    f"  ⚠ Split {k}: Dir={dir_acc:.1%} < 50% — "
                    "consider switching to target_type='classification'"
                )
            if self.target_type == "classification" and auc < 0.52:
                print(
                    f"  ⚠ Split {k}: AUC={auc:.4f} < 0.52 — "
                    "features may lack predictive power at this holding period. "
                    "Consider increasing holding_period_days to 20."
                )

            result_row = {
                "split": k,
                # train_end uses the last date included in the training window
                "train_end": str(train_end)[:10],
                "test_end": str(test_end)[:10],
                "n_train": len(train_df),
                "n_test": len(test_df),
                "r2": None if r2 is None else round(r2, 4),
                "mae": None if mae is None else round(mae, 6),
                "directional_accuracy": round(dir_acc, 4),
                "ic": None if ic is None else round(ic, 4),
                # Track in-sample IC from the fold's own training metrics
                "train_ic": fold_learner._train_metrics.get("ic"),
                "auc": round(auc, 4),
                "weights": fold_weights.to_dict(),
            }
            if self.target_type == "classification":
                result_row["effective_auc"] = round(effective_auc, 4)
            results.append(result_row)

        if results and self.target_type == "classification":
            aucs = [r["auc"] for r in results]
            if aucs and all(a < 0.52 for a in aucs):
                print(
                    "WARNING: model not better than random, consider increasing "
                    "training window or holding period"
                )

        # Overfit diagnostic: compare average train IC vs average OOS IC.
        if results:
            train_ics = [r.get("train_ic") for r in results if r.get("train_ic") is not None]
            oos_ics = [r.get("ic") for r in results if r.get("ic") is not None]
            if train_ics and oos_ics:
                mean_train_ic = float(np.mean(train_ics))
                mean_oos_ic = float(np.mean(oos_ics))
                gap = mean_train_ic - mean_oos_ic
                if gap > 0.3:
                    print(
                        f"⚠ Overfit detected: train IC={mean_train_ic:+.3f}, "
                        f"OOS IC={mean_oos_ic:+.3f} (gap={gap:+.3f}). "
                        "Consider increasing regularization."
                    )

        return results

    # ==============================================================
    # Internals
    # ==============================================================

    def _select_target_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Choose which numeric target to use for training, based on
        self.return_target_type:

            - 'raw'          : forward_return
            - 'excess'       : forward_return_excess
            - 'sharpe_scaled': forward_return / rolling_20d_vol (train only)
        """
        # Base raw forward return
        raw = df["forward_return"].astype(float)

        if self.return_target_type == "excess" and "forward_return_excess" in df.columns:
            return df["forward_return_excess"].astype(float)

        if self.return_target_type == "sharpe_scaled":
            # Compute rolling 20-day volatility of the raw forward return
            # on the current (train) dataframe only, then scale.
            vol20 = raw.rolling(20, min_periods=20).std()
            vol20 = vol20.replace(0, np.nan)
            sharpe_scaled = raw / vol20
            return sharpe_scaled.astype(float)

        # Default: raw forward_return
        return raw

    def _estimate_ess(self, series: pd.Series) -> int:
        """
        Estimate effective sample size accounting for serial correlation.
        Uses n / (1 + 2 * sum(|autocorr(lag)|)) for lags 1..L (L = min(20, n//2 - 1)).
        """
        s = series.dropna()
        n = len(s)
        if n < 3:
            return max(1, n)
        L = min(20, n // 2 - 1)
        if L < 1:
            return max(1, n)
        try:
            rho_sum = 0.0
            for lag in range(1, L + 1):
                r = s.autocorr(lag=lag)
                if r is not None and not np.isnan(r):
                    rho_sum += abs(float(r))
            ess = n / (1.0 + 2.0 * rho_sum)
            return max(1, int(round(ess)))
        except Exception:
            return max(1, n)

    def _compute_sample_weights(self, df: pd.DataFrame) -> np.ndarray | None:
        """
        Combine time-decay weights with an optional cost-awareness term.

        - Time decay: newer observations get higher weight (lambda controls half-life).
        - Cost-aware: observations where |forward_return| < expected_round_trip_cost_frac
          are down-weighted, since they are unlikely to overcome transaction costs.
        """
        n = len(df)
        if n == 0:
            return None

        weights = np.ones(n, dtype=np.float64)

        # Time-decay component (if dates are available and lambda > 0).
        if self.time_decay_lambda > 0 and "date" in df.columns:
            max_date = df["date"].max()
            days_ago = (max_date - df["date"]).dt.days.values.astype(np.float64)
            decay = np.exp(-self.time_decay_lambda * days_ago)
            weights *= decay

        # Cost-aware component: down-weight trades with low edge relative to expected cost.
        if (
            TARGET in df.columns
            and "expected_round_trip_cost_frac" in df.columns
        ):
            fr = df[TARGET].values.astype(np.float64)
            cost_frac = df["expected_round_trip_cost_frac"].values.astype(np.float64)
            cost_frac = np.where(cost_frac > 0, cost_frac, 1e-6)
            edge = np.abs(fr) - cost_frac
            # Observations with negative edge (|return| < cost) get a small weight;
            # others keep full weight (can be tuned later).
            edge_multiplier = np.where(edge <= 0, 0.25, 1.0)
            weights *= edge_multiplier

        return weights

    def tune_hyperparams(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Hyperparameter tuning for regression models using time-series CV.

        Currently supports:
          - model_type == 'ridge': GridSearchCV over alpha
          - model_type == 'gbr'  : RandomizedSearchCV over key tree params
        """
        tscv = TimeSeriesSplit(n_splits=3)

        if self.model_type == "ridge":
            base = Ridge(fit_intercept=True)
            param_grid = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]}
            search = GridSearchCV(
                base,
                param_grid=param_grid,
                cv=tscv,
                scoring="r2",
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            logging.getLogger(__name__).info(
                "Ridge tuning complete; best params: %s (score=%.4f)",
                search.best_params_,
                search.best_score_,
            )
            # Stash for diagnostics
            self._best_params_ = search.best_params_
            return best

        if self.model_type == "gbr":
            base = GradientBoostingRegressor(random_state=42)
            param_dist = {
                "n_estimators": [100, 200, 300],
                "max_depth": [2, 3, 4],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.6, 0.8, 1.0],
            }
            search = RandomizedSearchCV(
                base,
                param_distributions=param_dist,
                n_iter=10,
                cv=tscv,
                scoring="r2",
                random_state=42,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            logging.getLogger(__name__).info(
                "GBR tuning complete; best params: %s (score=%.4f)",
                search.best_params_,
                search.best_score_,
            )
            self._best_params_ = search.best_params_
            return best

        if self.model_type == "xgb":
            if XGBRegressor is None:
                raise ImportError(
                    "Model type 'xgb' requested for tuning but the 'xgboost' "
                    "package is not installed. Install it with 'pip install xgboost' "
                    "or choose a different --model."
                )
            # Aggressively regularised prior for XGBoost; tuning will only make
            # small adjustments around this configuration.
            base = XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                tree_method="hist",
                n_jobs=-1,
                max_depth=3,
                min_child_weight=50,
                reg_alpha=1.0,
                reg_lambda=5.0,
                subsample=0.6,
                colsample_bytree=0.5,
                n_estimators=200,
                eval_metric="rmse",
            )
            param_dist = {
                # Narrow ranges around the aggressively regularised defaults.
                "n_estimators": [150, 200, 250],
                "max_depth": [3],
                "min_child_weight": [40, 50, 60],
                "learning_rate": [0.01, 0.03, 0.05],
                "subsample": [0.5, 0.6, 0.7],
                "colsample_bytree": [0.4, 0.5, 0.6],
                "reg_alpha": [0.5, 1.0, 2.0],
                "reg_lambda": [3.0, 5.0, 8.0],
            }
            search = RandomizedSearchCV(
                base,
                param_distributions=param_dist,
                n_iter=15,
                cv=tscv,
                scoring="r2",
                random_state=42,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            logging.getLogger(__name__).info(
                "XGBoost tuning complete; best params: %s (score=%.4f)",
                search.best_params_,
                search.best_score_,
            )
            self._best_params_ = search.best_params_
            return best

        # Fallback: return a simple Ridge if tuning not supported for this model_type
        logging.getLogger(__name__).warning(
            "tune_hyperparams called for unsupported model_type=%s; skipping tuning",
            self.model_type,
        )
        return Ridge(alpha=self.alpha, fit_intercept=True).fit(X_train, y_train)

    def _compute_train_metrics(self, X_scaled, y, df):
        if self.target_type == "classification":
            pos_idx = getattr(self, "_pos_class_index", 1)
            y_prob = self.model.predict_proba(X_scaled)[:, pos_idx]
            y_pred_class = (y_prob >= 0.5).astype(np.int64)
            directional_accuracy = float(accuracy_score(y, y_pred_class))
            try:
                auc = float(roc_auc_score(y, y_prob))
            except ValueError:
                auc = 0.5
            # Information coefficient: Spearman between scores and continuous forward returns
            ic_val = np.nan
            if TARGET in df.columns:
                y_cont = df[TARGET].values.astype(np.float64)
                if len(y_cont) > 2:
                    try:
                        rho, _ = spearmanr(y_prob, y_cont)
                        if np.isfinite(rho):
                            ic_val = float(rho)
                    except Exception:
                        ic_val = np.nan
            self._train_metrics = {
                "r2": np.nan,
                "mae": np.nan,
                "directional_accuracy": round(directional_accuracy, 4),
                "ic": ic_val,
                "auc_score": round(auc, 4),
            }
        elif self.model_type == "logistic":
            y_pred_dir = self.model.predict(X_scaled)
            y_true_dir = np.sign(y).astype(int)
            dir_acc = float((y_pred_dir == y_true_dir).mean())
            # Train IC and AUC from proba vs continuous/binary target
            classes = list(self.model.classes_)
            pos_idx = classes.index(1) if 1 in classes else 0
            y_prob = self.model.predict_proba(X_scaled)[:, pos_idx]
            ic_val = 0.0
            if len(y) > 2:
                try:
                    rho, _ = spearmanr(y_prob, y)
                    ic_val = float(rho) if np.isfinite(rho) else 0.0
                except Exception:
                    ic_val = 0.0
            y_binary = (y > 0).astype(np.int64)
            try:
                auc_val = float(roc_auc_score(y_binary, y_prob))
            except ValueError:
                auc_val = 0.5
            self._train_metrics = {
                "r2": 0.0,
                "mae": 0.0,
                "directional_accuracy": round(dir_acc, 4),
                "ic": round(ic_val, 4),
                "auc_score": round(auc_val, 4),
            }
        else:
            y_pred = self.model.predict(X_scaled)
            # IC as rank correlation vs *raw* forward_return when available.
            if "forward_return" in df.columns:
                fwd = df["forward_return"].values.astype(float)
            else:
                fwd = y
            if len(fwd) > 2:
                try:
                    ic_val, _ = spearmanr(y_pred, fwd)
                    ic_val = float(ic_val) if np.isfinite(ic_val) else 0.0
                except Exception:
                    ic_val = 0.0
            else:
                ic_val = 0.0
            self._train_metrics = {
                "r2": round(float(r2_score(y, y_pred)), 4),
                "mae": round(float(mean_absolute_error(y, y_pred)), 6),
                "directional_accuracy": round(float((np.sign(y_pred) == np.sign(y)).mean()), 4),
                "ic": round(ic_val, 4),
                "auc_score": 0.0,
            }

    def fit_regime_models(
        self,
        feature_df: pd.DataFrame,
        target: pd.Series,
        regime_series: pd.Series,
    ) -> None:
        """
        Train a separate LearnedWeights set for each regime present in regime_series.

        This method is intended for offline research: given a global
        feature matrix, target returns, and a date-indexed regime
        label series, it fits one set of weights per regime and stores
        them in ``self.regime_models``.  Each regime must have at
        least 30 samples; regimes with fewer observations are skipped.
        """
        self.regime_models = {}

        def _norm(label: str) -> str:
            mapping = {
                "bull_trend": "Bull",
                "Bull": "Bull",
                "bear_trend": "Bear",
                "Bear": "Bear",
                "sideways": "Sideways",
                "Sideways": "Sideways",
                "high_vol": "HighVol",
                "HighVol": "HighVol",
                "Crisis": "Bear",
            }
            return mapping.get(str(label), "Sideways")

        regimes = regime_series.dropna()
        for raw_label in regimes.unique():
            mask = regime_series == raw_label
            if mask.sum() < 30:
                continue
            df_regime = feature_df.loc[mask].copy()
            df_regime = df_regime.assign(**{TARGET: target.loc[mask].values})

            learner = WeightLearner(
                model_type=self.model_type,
                alpha=self.alpha,
                time_decay_lambda=self.time_decay_lambda,
                target_type=self.target_type,
                return_target_type=self.return_target_type,
            )
            w = learner.fit(df_regime, feature_cols=self.active_features or None)
            canon = _norm(raw_label)
            self.regime_models[canon] = w
