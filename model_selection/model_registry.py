from __future__ import annotations

import copy
import warnings
from typing import Any

import numpy as np
import pandas as pd

# ── Model Tier Definitions ────────────────────────────────────────────────────
# P9: Baselines removed from CORE_MODELS — they are reference statistics, not
# hypothesis tests. They are evaluated in a separate pass (outside SignalDiscovery)
# to establish the "no-ML" floor that learned models must beat.
CORE_MODELS = {
    "Ridge",
    "XGBRegressor",
    "LGBMRanker",
    "ElasticNet",
}

BASELINE_SCORERS = {
    "EqualWeightBaseline",
    "ICWeightedBaseline",
}

EXPERIMENTAL_MODELS = {
    "LogisticRegression",
    "RidgeLogistic",
    "XGBClassifier",
    "ShortLogistic",
    "ShortXGB",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "ElasticNetL1",
    "QuantileMedian",
    "QuantileUpper",
    "XGBSharpeIC",
    "XGBRankerPairwise",
    "XGBRankerNDCG",
}

# Portfolio-utility objectives are not AlphaModels. They are intentionally not
# registered for model fitting; portfolio economics belong in validation and
# construction, downstream of pure score generation.
CLASSIFIER_STOCK_SELECTION_MODELS: frozenset[str] = frozenset(
    {
        "LogisticRegression",
        "RidgeLogistic",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "XGBClassifier",
        "ShortLogistic",
        "ShortXGB",
    }
)

DIAGNOSTIC_ONLY_MODELS: frozenset[str] = CLASSIFIER_STOCK_SELECTION_MODELS


def is_diagnostic_only(name: str) -> bool:
    """Return True if the named model must not be promoted to production."""
    return name in DIAGNOSTIC_ONLY_MODELS


def is_classifier_stock_selector(name: str, model_kind: str | None = None) -> bool:
    """Return True for binary direction classifiers unsuitable for alpha ranking."""
    kind = str(model_kind or "").lower()
    return name in CLASSIFIER_STOCK_SELECTION_MODELS or kind in {"classifier", "short_classifier"}

from model_selection.training import FeaturePreprocessor


try:
    from xgboost import XGBRanker as _XGBRanker
    from sklearn.base import BaseEstimator, RegressorMixin as _XGBRankerMixin

    class XGBRankerWrapper(_XGBRankerMixin, BaseEstimator):
        """sklearn-compatible XGBoost pairwise/NDCG ranker wrapper."""

        def __init__(
            self,
            n_estimators: int = 100,
            max_depth: int = 4,
            learning_rate: float = 0.05,
            subsample: float = 0.8,
            colsample_bytree: float = 0.8,
            reg_alpha: float = 0.1,
            reg_lambda: float = 1.0,
            objective: str = "rank:pairwise",
            n_jobs: int = -1,
            min_child_weight: int = 1,
        ) -> None:
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.learning_rate = learning_rate
            self.subsample = subsample
            self.colsample_bytree = colsample_bytree
            self.reg_alpha = reg_alpha
            self.reg_lambda = reg_lambda
            self.objective = objective
            self.n_jobs = n_jobs
            self.min_child_weight = min_child_weight
            self._model: Any = None
            self._preloaded_date_groups: np.ndarray | None = None

        def set_date_context(self, date_groups: np.ndarray) -> "XGBRankerWrapper":
            self._preloaded_date_groups = date_groups
            return self

        def fit(self, X: np.ndarray, y: np.ndarray, **kw: Any) -> "XGBRankerWrapper":
            _kw_groups = kw.get("_date_groups")
            date_groups = _kw_groups if _kw_groups is not None else self._preloaded_date_groups
            if date_groups is None:
                date_groups = np.arange(len(y), dtype=int) // 500

            import time
            t0 = time.perf_counter()

            sort_idx = np.argsort(date_groups, kind="stable")
            sorted_groups = date_groups[sort_idx]
            sorted_y = y[sort_idx]

            diffs = np.concatenate(([True], sorted_groups[1:] != sorted_groups[:-1], [True]))
            boundaries = np.where(diffs)[0]

            t1 = time.perf_counter()

            # Convert continuous targets to ordinal labels per cross-section
            labels = np.zeros(len(y), dtype=np.int32)
            sorted_labels = np.zeros(len(y), dtype=np.int32)
            n_bins = 10
            
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i+1]
                n = end - start
                if n < 2:
                    continue
                ranks = sorted_y[start:end].argsort().argsort()
                sorted_labels[start:end] = np.minimum((ranks * n_bins) // n, n_bins - 1).astype(np.int32)
                
            labels[sort_idx] = sorted_labels

            _, first_idx, counts = np.unique(date_groups, return_index=True, return_counts=True)
            group_sizes = counts[np.argsort(first_idx)].tolist()

            t2 = time.perf_counter()

            self._model = _XGBRanker(
                objective=self.objective,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=42,
                n_jobs=self.n_jobs,
                min_child_weight=self.min_child_weight,
            )
            x_arr = X.values if hasattr(X, "values") else np.asarray(X)
            self._model.fit(x_arr, labels, qid=np.asarray(date_groups, dtype=np.int32))
            t3 = time.perf_counter()
            self.fit_telemetry_ = {
                "group_build_time": t1 - t0,
                "rank_label_build_time": t2 - t1,
                "fit_time": t3 - t2,
                "total_model_runtime": t3 - t0
            }
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            if self._model is None:
                return np.zeros(len(X), dtype=float)
            x_arr = X.values if hasattr(X, "values") else np.asarray(X)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
                return self._model.predict(x_arr).astype(float)

    XGB_RANKER_AVAILABLE = True
except Exception:
    XGB_RANKER_AVAILABLE = False
    XGBRankerWrapper = None  # type: ignore[assignment,misc]


try:
    import lightgbm as _lgb
    from sklearn.base import BaseEstimator, RegressorMixin as _RegressorMixin

    class LGBMRankerWrapper(_RegressorMixin, BaseEstimator):
        """sklearn-compatible LightGBM ranker wrapper."""

        def __init__(
            self,
            n_estimators: int = 100,
            max_depth: int = 4,
            learning_rate: float = 0.05,
            num_leaves: int = 15,
            min_child_samples: int = 20,
            subsample: float = 0.8,
            colsample_bytree: float = 0.8,
            n_jobs: int = -1,
        ) -> None:
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.learning_rate = learning_rate
            self.num_leaves = num_leaves
            self.min_child_samples = min_child_samples
            self.subsample = subsample
            self.colsample_bytree = colsample_bytree
            self.n_jobs = n_jobs
            self._model: Any = None
            self._preloaded_date_groups: np.ndarray | None = None

        def set_date_context(self, date_groups: np.ndarray) -> "LGBMRankerWrapper":
            self._preloaded_date_groups = date_groups
            return self

        def fit(self, X: np.ndarray, y: np.ndarray, **kw: Any) -> "LGBMRankerWrapper":
            _kw_groups = kw.get("_date_groups")
            date_groups = _kw_groups if _kw_groups is not None else self._preloaded_date_groups
            if date_groups is None:
                date_groups = np.arange(len(y), dtype=int) // 500

            import time
            t0 = time.perf_counter()

            sort_idx = np.argsort(date_groups, kind="stable")
            sorted_groups = date_groups[sort_idx]
            sorted_y = y[sort_idx]

            diffs = np.concatenate(([True], sorted_groups[1:] != sorted_groups[:-1], [True]))
            boundaries = np.where(diffs)[0]

            t1 = time.perf_counter()

            labels = np.zeros(len(y), dtype=np.int32)
            sorted_labels = np.zeros(len(y), dtype=np.int32)
            n_bins = 10
            
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i+1]
                n = end - start
                if n < 2:
                    continue
                ranks = sorted_y[start:end].argsort().argsort()
                sorted_labels[start:end] = np.minimum((ranks * n_bins) // n, n_bins - 1).astype(np.int32)
                
            labels[sort_idx] = sorted_labels

            _, first_idx, counts = np.unique(date_groups, return_index=True, return_counts=True)
            group_sizes = counts[np.argsort(first_idx)].tolist()

            t2 = time.perf_counter()

            self._model = _lgb.LGBMRanker(
                objective="lambdarank",
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=42,
                n_jobs=self.n_jobs,
                verbose=-1,
            )
            x_arr = X.values if hasattr(X, "values") else np.asarray(X)
            self._model.fit(x_arr, labels, group=group_sizes)
            t3 = time.perf_counter()
            self.fit_telemetry_ = {
                "group_build_time": t1 - t0,
                "rank_label_build_time": t2 - t1,
                "fit_time": t3 - t2,
                "total_model_runtime": t3 - t0
            }
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            if self._model is None:
                return np.zeros(len(X), dtype=float)
            x_arr = X.values if hasattr(X, "values") else np.asarray(X)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
                return self._model.predict(x_arr).astype(float)

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    LGBMRankerWrapper = None  # type: ignore[assignment,misc]


class PrefitWeightedEnsemble:
    """Pickle-safe weighted ensemble over already-fitted heterogeneous models."""

    def __init__(
        self,
        estimators: list[tuple[str, Any, str, bool]],
        weights: list[float],
        feature_preprocessor: FeaturePreprocessor | None = None,
        estimator_preprocessors: list[FeaturePreprocessor | None] | None = None,
    ) -> None:
        self.estimators = estimators
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / self.weights.sum() if self.weights.sum() > 0 else self.weights
        self.feature_preprocessor = feature_preprocessor
        self.estimator_preprocessors = estimator_preprocessors or [feature_preprocessor] * len(estimators)

    def _score_estimator(self, est: Any, kind: str, uses_proba: bool, x: np.ndarray) -> np.ndarray:
        if kind in {"regressor", "long_alpha", "overlay_alpha", "short_alpha"}:
            return est.predict(x).astype(float)
        if kind == "short_classifier":
            if uses_proba and hasattr(est, "predict_proba"):
                p_down = est.predict_proba(x)[:, 1].astype(float)
                return p_down - 0.5
            return est.predict(x).astype(float) - 0.5
        if uses_proba and hasattr(est, "predict_proba"):
            return est.predict_proba(x)[:, 1].astype(float) - 0.5
        if hasattr(est, "decision_function"):
            return est.decision_function(x).astype(float)
        return est.predict(x).astype(float) - 0.5

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            scores = []
            for idx, (_, est, kind, uses_proba) in enumerate(self.estimators):
                prep = self.estimator_preprocessors[idx] if idx < len(self.estimator_preprocessors) else self.feature_preprocessor
                arr = prep.transform(x) if prep is not None else np.asarray(x)
                scores.append(self._score_estimator(est, kind, uses_proba, arr))
        else:
            arr = np.asarray(x)
            scores = [self._score_estimator(est, kind, uses_proba, arr) for _, est, kind, uses_proba in self.estimators]
        if not scores:
            return np.zeros(len(x), dtype=float)
        w = self.weights if len(self.weights) == len(scores) else np.ones(len(scores), dtype=float) / len(scores)
        return np.average(np.vstack(scores), axis=0, weights=w)


def sharpe_ic_objective(y_true: np.ndarray, y_pred: np.ndarray):
    n = len(y_true)
    f = y_pred - y_pred.mean()
    r = y_true - y_true.mean()
    sigma_f = float(np.sqrt((f * f).mean())) + 1e-8
    sigma_r = float(np.sqrt((r * r).mean())) + 1e-8
    ic = float((f * r).mean()) / (sigma_f * sigma_r)
    grad_ic = r / (sigma_f * sigma_r) - ic * f / sigma_f**2
    grad = -np.clip(grad_ic, -1.0, 1.0)
    hess = np.ones(n, dtype=np.float64)
    return grad, hess


def economic_objective_params(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Deprecated: economic objectives are not part of alpha-model fitting."""
    return {}


def build_models(cfg: dict[str, Any] | None = None) -> list[tuple[str, Any, bool, str]]:
    from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    ms_cfg = ((cfg or {}).get("model_selection", {}) or {}) if isinstance(cfg, dict) else {}
    # Binary classifiers are intentionally excluded from executable stock
    # selection by default. They estimate P(return > 0), not return magnitude or
    # cross-sectional rank, so their probabilities tend to cluster near 0.50
    # and create high-turnover optimizer inputs. Set include_classifiers=true
    # only for diagnostic/meta-label comparisons; they remain diagnostic-only.
    include_classifiers = bool(ms_cfg.get("include_classifiers", False))
    include_short_classifiers = bool(ms_cfg.get("include_short_classifiers", include_classifiers))
    include_slow_models = bool(ms_cfg.get("include_slow_models", False))
    core_only = bool(ms_cfg.get("core_models_only", False))

    models: list[tuple[str, Any, bool, str]] = []

    # ── Binary direction classifiers (diagnostic/meta-label only) ─────────────
    # These predict target_up (P(return > 0)) and produce probability scores.
    # They are not valid executable alpha selectors because classification
    # discards ordinal return information. Keep them out of production ranking.
    if include_classifiers:
        models.append(
            (
                "LogisticRegression",
                Pipeline([("scaler", RobustScaler()), ("model", LogisticRegression(C=0.01, max_iter=1000))]),
                True,
                "classifier",
            )
        )
        models.append(
            (
                "RidgeLogistic",
                Pipeline([("scaler", RobustScaler()), ("model", LogisticRegression(C=0.1, max_iter=500, solver="liblinear"))]),
                False,
                "classifier",
            )
        )
        models.append(("RandomForestClassifier", RandomForestClassifier(n_estimators=50, max_depth=3, min_samples_split=10, random_state=42, n_jobs=-1), True, "classifier"))
        if include_slow_models:
            from sklearn.ensemble import GradientBoostingClassifier
            models.append(("GradientBoostingClassifier", GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42), True, "classifier"))
        try:
            from xgboost import XGBClassifier
            models.append(("XGBClassifier", XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", min_child_weight=50), True, "classifier"))
        except Exception:
            pass

    # ── XGBoost pairwise ranking (rank:pairwise objective) ────────────────────
    # Directly optimizes rank ordering within each cross-section using pairwise
    # comparisons. Superior to regression when ordinal rank matters more than magnitude.
    # Placed BEFORE linear models so LPT scheduling picks them up first — rankers
    # are 5-10× more expensive per window and should start immediately rather than
    # waiting behind fast models in the FIFO worker queue.
    if not core_only and XGB_RANKER_AVAILABLE:
        models.append((
            "XGBRankerPairwise",
            XGBRankerWrapper(n_estimators=50, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=50, objective="rank:pairwise"),
            False,
            "regressor",
        ))
        models.append((
            "XGBRankerNDCG",
            XGBRankerWrapper(n_estimators=50, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=50, objective="rank:ndcg"),
            False,
            "regressor",
        ))
    elif not XGB_RANKER_AVAILABLE:
        print("NOTE: XGBRanker not available — skipping XGBRankerPairwise/NDCG. Upgrade xgboost >= 1.3")

    # ── LightGBM lambdarank ───────────────────────────────────────────────────
    # Complexity aligned with XGBRanker to prevent asymmetric wall-time blowout
    # in the parallel worker queue. Was n_estimators=100, max_depth=4, num_leaves=15.
    if LGBM_AVAILABLE:
        models.append(("LGBMRanker", LGBMRankerWrapper(n_estimators=50, max_depth=3, learning_rate=0.05, num_leaves=7, min_child_samples=20, subsample=0.8, colsample_bytree=0.8), False, "regressor"))
    else:
        print("NOTE: lightgbm not installed — skipping LGBMRanker. Run: pip install lightgbm")

    # ── Ridge regression: sparse linear signal on residualized return ─────────
    models.append(("Ridge", Pipeline([("scaler", RobustScaler()), ("model", Ridge(alpha=10.0))]), False, "regressor"))

    # ── P6: Baseline scorers — no-ML floor that learned models must beat ──────
    # EqualWeightBaseline: simple average of feature z-scores. If ML cannot
    # beat this, the model is overfitting noise (negative subsumption alpha).
    # ICWeightedBaseline: weights features by trailing IC — learns which
    # features are predictive vs anti-predictive without nonlinear fitting.
    try:
        from model_selection.baseline_scorers import EqualWeightBaseline, ICWeightedBaseline
        models.append(("EqualWeightBaseline", EqualWeightBaseline(), False, "regressor"))
        models.append(("ICWeightedBaseline", ICWeightedBaseline(ic_window=126, min_ic_obs=30), False, "regressor"))
    except Exception as _bs_exc:
        print(f"NOTE: baseline_scorers import failed ({_bs_exc}) — skipping EqualWeight/ICWeighted baselines")

    # ── ElasticNet: L1+L2 for automatic feature selection on rank target ──────
    # alpha calibrated for financial ML SNR (~0.01-0.05, not typical ML SNR ~10+).
    # alpha=0.05 zeros ALL coefficients with ~15 features at financial noise levels.
    # alpha=0.001 allows weak features to survive L1 while L2 handles collinearity.
    models.append((
        "ElasticNet",
        Pipeline([("scaler", RobustScaler()), ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=2000, tol=1e-4))]),
        False,
        "regressor",
    ))
    if not core_only:
        # More L1-sparse variant for sparser cross-sections
        models.append((
            "ElasticNetL1",
            Pipeline([("scaler", RobustScaler()), ("model", ElasticNet(alpha=0.001, l1_ratio=0.8, max_iter=2000, tol=1e-4))]),
            False,
            "regressor",
        ))

    # ── XGBoost objectives ────────────────────────────────────────────────────
    try:
        from xgboost import XGBRegressor
        # FIXED: reduced complexity to control turnover. Was n_estimators=100, max_depth=4.
        # At 15 features × 500 stocks, max_depth=4 overfits idiosyncratic patterns that
        # reverse within days → high turnover → cost drag = 100%. min_child_weight=50
        # prevents leaf nodes fitting to <50 samples, producing smoother rank surfaces.
        _xgb_common = dict(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=50)
        models.append(("XGBRegressor", XGBRegressor(**_xgb_common), False, "regressor"))
        if not core_only:
            models.append(("XGBSharpeIC", XGBRegressor(objective=sharpe_ic_objective, **_xgb_common), False, "regressor"))
    except Exception:
        pass

    # ── Quantile regression: robust to fat tails, long-biased at q>0.5 ───────
    # q=0.5 (median) is robust to outliers; q=0.7 biases predictions toward the
    # upper tail — a soft long filter without explicit binary classification.
    if not core_only:
        try:
            from sklearn.linear_model import QuantileRegressor
            models.append((
                "QuantileMedian",
                Pipeline([("scaler", RobustScaler()), ("model", QuantileRegressor(quantile=0.5, alpha=0.05, solver="highs"))]),
                False,
                "regressor",
            ))
            models.append((
                "QuantileUpper",
                Pipeline([("scaler", RobustScaler()), ("model", QuantileRegressor(quantile=0.7, alpha=0.05, solver="highs"))]),
                False,
                "regressor",
            ))
        except Exception:
            pass

    # ── Short-side binary classifiers (diagnostic/meta-label only) ────────────
    if include_short_classifiers:
        models.append(("ShortLogistic", Pipeline([("scaler", RobustScaler()), ("model", LogisticRegression(C=0.01, max_iter=1000, class_weight="balanced"))]), True, "short_classifier"))
        try:
            from xgboost import XGBClassifier as _XGBCls
            models.append(("ShortXGB", _XGBCls(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss"), True, "short_classifier"))
        except Exception:
            pass

    model_tier = ms_cfg.get("model_tier", "core").lower()
    if core_only:
        if model_tier != "core":
            print(
                f"[ModelRegistry] WARNING: core_models_only=true conflicts with "
                f"model_tier={model_tier}. Forcing model_tier='core'."
            )
        model_tier = "core"
        print(f"[ModelRegistry] core_models_only=true — evaluating {len(CORE_MODELS)} core models only: {sorted(CORE_MODELS)}")
    
    filtered_models = []
    for m in models:
        name = m[0]
        is_core = name in CORE_MODELS
        is_exp = name in EXPERIMENTAL_MODELS
        
        if model_tier == "core" and not is_core:
            continue
        if model_tier == "experimental" and not is_exp:
            continue
        if model_tier == "core_plus_experimental" and not (is_core or is_exp):
            continue
            
        filtered_models.append(m)

    return filtered_models


def constrain_model_parallelism(
    models: list[tuple[str, Any, bool, str]],
    *,
    max_jobs: int = 1,
) -> list[tuple[str, Any, bool, str]]:
    """
    Enforce a single parallelism layer across the research stack.

    When the outer model-selection loop is parallelized, inner estimators should
    not each claim the full machine. This helper clones compatible estimators and
    caps their internal worker count.
    """

    bounded: list[tuple[str, Any, bool, str]] = []
    for name, model, uses_proba, model_kind in models:
        tuned = copy.deepcopy(model)
        params: dict[str, Any] = {}
        inner_model = tuned
        if hasattr(tuned, "named_steps") and isinstance(getattr(tuned, "named_steps", None), dict):
            inner_model = tuned.named_steps.get("model", tuned)
        inner_class_name = type(inner_model).__name__
        if hasattr(tuned, "get_params"):
            try:
                available = tuned.get_params(deep=True)
            except Exception:
                available = {}
            for key in ("n_jobs", "nthread", "num_threads", "model__n_jobs", "model__nthread", "model__num_threads"):
                if "n_jobs" in key and inner_class_name == "LogisticRegression":
                    continue
                if key in available:
                    params[key] = int(max_jobs)
            if params and hasattr(tuned, "set_params"):
                try:
                    tuned = tuned.set_params(**params)
                except Exception:
                    pass
        bounded.append((name, tuned, uses_proba, model_kind))
    return bounded
