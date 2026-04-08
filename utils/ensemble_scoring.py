# region imports
from __future__ import annotations
try:
    from AlgorithmImports import *
except ImportError:
    pass
# endregion

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class EnsembleModelSpec:
    path: str
    weight: float
    model_type: str  # "classifier" | "regressor"


@dataclass
class LoadedEnsembleModel:
    path: str
    weight: float
    model_type: str
    estimator: Any
    feature_columns: list[str] | None


def _minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn = float(s.min()) if len(s) else 0.0
    mx = float(s.max()) if len(s) else 0.0
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-12:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return (s - mn) / (mx - mn)


def _rank_norm(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if len(s) == 0:
        return s
    # Center on 0: map [0, 1] ranking to [-1, 1]
    return (s.rank(pct=True, method="average").fillna(0.5).astype(float) * 2.0) - 1.0


def _zscore_standardize(s: pd.Series) -> pd.Series:
    """Expanding-window z-score: no lookahead, scales all models to comparable units."""
    s = pd.to_numeric(s, errors="coerce")
    if len(s) < 20:
        return s
    expanding_mean = s.expanding(min_periods=20).mean()
    expanding_std = s.expanding(min_periods=20).std().clip(lower=1e-8)
    return (s - expanding_mean) / expanding_std


def _load_pickle(path: Path) -> Any:
    """Load a pickled model or joblib artifact with warnings suppressed."""
    import joblib
    import sys
    import warnings

    # LGBMRankerWrapper is defined in run_model_selection.py which runs as __main__
    # when invoked directly.  Artifacts pickled there store the class as
    # "__main__.LGBMRankerWrapper".  Patch __main__ before unpickling so that
    # any context (backtester, signal engine, etc.) can deserialise the artifact.
    try:
        import run_model_selection as _rms  # noqa: PLC0415
        _main = sys.modules.get("__main__")
        if _main is not None and not hasattr(_main, "LGBMRankerWrapper"):
            _lgbmrw = getattr(_rms, "LGBMRankerWrapper", None)
            if _lgbmrw is not None:
                setattr(_main, "LGBMRankerWrapper", _lgbmrw)
    except Exception:
        pass

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Trying to unpickle estimator .*",
            category=UserWarning,
        )
        try:
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as fh:
                return pickle.load(fh)


def load_ensemble_models(ensemble_cfg: dict[str, Any]) -> list[LoadedEnsembleModel]:
    models_raw = ensemble_cfg.get("models") or []
    out: list[LoadedEnsembleModel] = []
    for m in models_raw:
        if not isinstance(m, dict):
            continue
        p = str(m.get("path") or "").strip()
        if not p:
            continue
        w = float(m.get("weight", 0.0) or 0.0)
        t = str(m.get("type") or "classifier").strip().lower()
        if t not in ("classifier", "regressor", "short_classifier"):
            t = "classifier"
        pp = Path(p)
        if not pp.is_absolute():
            pp = Path.cwd() / pp
        if not pp.is_file():
            print(f"[ensemble] WARN: missing model file, skipping: {pp}")
            continue
        try:
            obj = _load_pickle(pp)
            # run_model_selection saves artifact dict with estimator + feature_columns
            est = obj.get("estimator") if isinstance(obj, dict) else obj
            feat_cols = obj.get("feature_columns") if isinstance(obj, dict) else None
            if est is None:
                print(f"[ensemble] WARN: model file has no estimator, skipping: {pp}")
                continue
            out.append(
                LoadedEnsembleModel(
                    path=str(pp),
                    weight=float(w),
                    model_type=t,
                    estimator=est,
                    feature_columns=[str(c) for c in feat_cols] if isinstance(feat_cols, list) else None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ensemble] WARN: failed to load {pp}: {exc}")
            continue
    return out


def _predict_model(model: LoadedEnsembleModel, features_df: pd.DataFrame, clip: bool) -> pd.Series:
    cols = model.feature_columns if model.feature_columns else list(features_df.columns)
    # Aggressive sanitization similar to training pipeline
    X_df = (
        features_df.reindex(columns=cols)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-10.0, 10.0)
    )
    est_mod = getattr(model.estimator, "__module__", "") or ""
    # Use ndarray for sklearn estimators to avoid repetitive feature-name warnings.
    # Force copy(True) to ensure contiguous memory for BLAS stability on Mac ARM64
    X = X_df.to_numpy(dtype=float, copy=True) if est_mod.startswith("sklearn.") else X_df
    
    est = model.estimator
    y: np.ndarray
    
    # Silence known spurious BLAS matmul/overflow warnings on Apple Silicon during inference
    with np.errstate(all="ignore"):
        if model.model_type == "classifier":
            if hasattr(est, "predict_proba"):
                proba = est.predict_proba(X)
                y = np.asarray(proba)[:, 1] if np.asarray(proba).ndim == 2 else np.asarray(proba)
                y = 2.0 * y - 1.0
            else:
                if hasattr(est, "decision_function"):
                    y = np.asarray(est.decision_function(X), dtype=float)
                else:
                    y = np.asarray(est.predict(X), dtype=float)
            s = pd.Series(y, index=features_df.index, dtype=float)
            if clip:
                s = s.clip(-1.0, 1.0)
            return s
        
        if model.model_type == "short_classifier":
            if hasattr(est, "predict_proba"):
                proba = est.predict_proba(X)
                p_down = np.asarray(proba)[:, 1] if np.asarray(proba).ndim == 2 else np.asarray(proba)
                # High P(down) → strongly negative score (bearish)
                y = -(2.0 * p_down - 1.0)
            else:
                if hasattr(est, "decision_function"):
                    y = -np.asarray(est.decision_function(X), dtype=float)
                else:
                    y = -np.asarray(est.predict(X), dtype=float)
            s = pd.Series(y, index=features_df.index, dtype=float)
            if clip:
                s = s.clip(-1.0, 1.0)
            return s
            
        # Auto-detect classifiers passed with type="regressor".
        # RandomForestClassifier / GradientBoostingClassifier etc. expose
        # predict_proba; calling .predict() returns binary {0,1} labels which
        # discard probability information.  Use predict_proba when available.
        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(X)
            y = np.asarray(proba)[:, 1] if np.asarray(proba).ndim == 2 else np.asarray(proba)
            y = 2.0 * y - 1.0  # map [0,1] → [-1,1]
        else:
            y = np.asarray(est.predict(X), dtype=float)

    return pd.Series(y, index=features_df.index, dtype=float)


def compute_ensemble_score(
    features_df: pd.DataFrame,
    models: list[LoadedEnsembleModel],
    *,
    normalize: bool = True,
    standardize: bool = False,
    clip: bool = False,
) -> pd.Series:
    if features_df is None or features_df.empty or not models:
        return pd.Series(dtype=float)

    preds: list[pd.Series] = []
    weights: list[float] = []
    for m in models:
        try:
            s = _predict_model(m, features_df, clip=clip).astype(float)
            if normalize:
                s = _rank_norm(s)
            elif standardize:
                s = _zscore_standardize(s)
            preds.append(s)
            weights.append(max(0.0, float(m.weight)))
        except Exception as exc:  # noqa: BLE001
            print(f"[ensemble] WARN: prediction failed for {m.path}: {exc}")
            continue

    if not preds:
        return pd.Series(dtype=float)
    w_sum = float(sum(weights))
    if w_sum <= 1e-12:
        weights = [1.0] * len(preds)
        w_sum = float(len(preds))
    out = pd.Series(np.zeros(len(features_df), dtype=float), index=features_df.index)
    for s, w in zip(preds, weights):
        out = out + (float(w) / w_sum) * s
    return out

