"""
Stacked ensemble of Ridge (RidgeCV), GBR, and XGBoost regressors.

Blend weights are chosen to maximize Spearman IC on a held-out validation slice.
Base models are refit on train+validation before deployment predictions.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None  # type: ignore[misc,assignment]


class StackedEnsemble:
    """Equal-weight default; blend weights optimized by IC on validation."""

    def __init__(self, models=None, weights=None):
        if XGBRegressor is None:
            raise ImportError(
                "StackedEnsemble requires xgboost. Install with: pip install xgboost"
            )

        self.base_models = models or {
            "ridge": RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0]),
            "gbr": GradientBoostingRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=50,
                random_state=42,
            ),
            "xgb": XGBRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=5.0,
                min_child_weight=50,
                objective="reg:squarederror",
                random_state=42,
                tree_method="hist",
                n_jobs=-1,
                eval_metric="rmse",
            ),
        }
        self.blend_weights = weights or {"ridge": 0.4, "gbr": 0.3, "xgb": 0.3}

    def fit(self, X_train, y_train, X_val, y_val):
        predictions: dict[str, np.ndarray] = {}
        for name, model in self.base_models.items():
            model.fit(X_train, y_train)
            predictions[name] = np.asarray(model.predict(X_val), dtype=np.float64)

        best_weights = self._optimize_weights(predictions, y_val)
        self.blend_weights = best_weights

        y_val_arr = np.asarray(y_val, dtype=np.float64)
        for name, pred in predictions.items():
            ic_val, _ = spearmanr(pred, y_val_arr)
            ic_val = float(ic_val) if np.isfinite(ic_val) else float("nan")
            print(f"  {name} IC: {ic_val:.4f}")

        ensemble_pred = self._blend(predictions)
        ens_ic, _ = spearmanr(ensemble_pred, y_val_arr)
        ens_ic = float(ens_ic) if np.isfinite(ens_ic) else float("nan")
        print(f"  ensemble IC: {ens_ic:.4f}")

        # Refit all base models on full data (train + val) for deployment.
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([np.asarray(y_train, dtype=np.float64), y_val_arr])
        for _name, model in self.base_models.items():
            model.fit(X_full, y_full)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        predictions: dict[str, np.ndarray] = {}
        for name, model in self.base_models.items():
            predictions[name] = np.asarray(model.predict(X), dtype=np.float64)
        return self._blend(predictions)

    def _blend(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        names = list(predictions.keys())
        n = len(predictions[names[0]])
        result = np.zeros(n, dtype=np.float64)
        for name in names:
            w = float(self.blend_weights.get(name, 0.0))
            result += w * predictions[name]
        return result

    def _optimize_weights(
        self,
        predictions: dict[str, np.ndarray],
        y_val: np.ndarray,
    ) -> dict[str, float]:
        """Grid search over blend weights; maximize Spearman IC on validation."""
        y_val = np.asarray(y_val, dtype=np.float64)
        names = list(predictions.keys())
        best_ic = -999.0
        best_w = dict(self.blend_weights)

        for r_w, g_w in product([0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4]):
            x_w = round(1.0 - r_w - g_w, 2)
            if x_w < 0.1:
                continue
            if abs(r_w + g_w + x_w - 1.0) > 1e-6:
                continue
            w = {"ridge": r_w, "gbr": g_w, "xgb": x_w}
            pred = np.zeros_like(y_val, dtype=np.float64)
            for n in names:
                if n in w:
                    pred += w[n] * predictions[n]
            ic_val, _ = spearmanr(pred, y_val)
            ic_val = float(ic_val) if np.isfinite(ic_val) else -999.0
            if ic_val > best_ic:
                best_ic = ic_val
                best_w = dict(w)

        print(f"  Optimal blend: {best_w} (IC={best_ic:.4f})")
        return best_w
