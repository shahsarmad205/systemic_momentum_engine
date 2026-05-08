# region imports
from __future__ import annotations
try:
    from AlgorithmImports import *
except ImportError:
    pass
# endregion

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class EnsembleModelSpec:
    path: str
    weight: float
    model_type: str  # "classifier"|"regressor"|"short_classifier"|"short_regressor"|"long_alpha"|"overlay_alpha"|"short_alpha"


_VALID_MODEL_TYPES = frozenset({
    "classifier", "regressor",
    "short_classifier", "short_regressor",
    "long_alpha", "overlay_alpha", "short_alpha",
})


@dataclass
class LoadedEnsembleModel:
    path: str
    weight: float
    model_type: str
    estimator: Any
    feature_columns: list[str] | None
    # True when the estimator (PrefitWeightedEnsemble) applies FeaturePreprocessor
    # internally — skip the generic fill/clip so training preprocessing is honoured.
    preprocessor_handled: bool = False
    # M1: modal training horizon stored in the artifact (days). None if not present.
    horizon_days: int | None = None
    # H1: per-member specs from artifact["ensemble_member_specs"] — used for diagnostics.
    member_specs: list[dict] | None = None


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

    # run_model_selection.py runs as __main__ when invoked directly.
    # Pickle stores references to classes/functions defined there as "__main__.<name>".
    # Patch __main__ with every known symbol before unpickling so that any context
    # (backtester, signal engine, etc.) can deserialise these artifacts.
    _SYMBOLS_TO_PATCH = ("LGBMRankerWrapper", "_sharpe_ic_obj")
    try:
        import run_model_selection as _rms  # noqa: PLC0415
        _main = sys.modules.get("__main__")
        if _main is not None:
            for _sym in _SYMBOLS_TO_PATCH:
                if not hasattr(_main, _sym):
                    _val = getattr(_rms, _sym, None)
                    if _val is not None:
                        setattr(_main, _sym, _val)
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
        if t not in _VALID_MODEL_TYPES:
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
            # Artifact's model_type is authoritative — it reflects what model_selection
            # actually trained (e.g. "long_alpha", "overlay_alpha", "short_alpha").
            artifact_type = str(obj.get("model_type", "")).strip().lower() if isinstance(obj, dict) else ""
            if artifact_type in _VALID_MODEL_TYPES:
                t = artifact_type
            # PrefitWeightedEnsemble applies FeaturePreprocessor internally per member.
            # Skip the generic fill/clip in _predict_model so training preprocessing wins.
            preproc_handled = hasattr(est, "feature_preprocessor") or hasattr(est, "estimator_preprocessors")

            # M1: read modal training horizon from artifact.
            _h_raw = obj.get("horizon_days") if isinstance(obj, dict) else None
            _horizon_days: int | None = int(_h_raw) if isinstance(_h_raw, (int, float)) and _h_raw > 0 else None

            # H1: read per-member specs for feature_view diagnostics.
            _specs_raw = obj.get("ensemble_member_specs") if isinstance(obj, dict) else None
            _member_specs: list[dict] | None = list(_specs_raw) if isinstance(_specs_raw, list) else None
            if _member_specs:
                _views: dict[str, int] = {}
                for _s in _member_specs:
                    _v = str(_s.get("feature_view", "?") or "?")
                    _views[_v] = _views.get(_v, 0) + 1
                print(f"[ensemble] {pp.name}: {len(_member_specs)} members — views={_views} horizon={_horizon_days}d")

            out.append(
                LoadedEnsembleModel(
                    path=str(pp),
                    weight=float(w),
                    model_type=t,
                    estimator=est,
                    feature_columns=[str(c) for c in feat_cols] if isinstance(feat_cols, list) else None,
                    preprocessor_handled=bool(preproc_handled),
                    horizon_days=_horizon_days,
                    member_specs=_member_specs,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ensemble] WARN: failed to load {pp}: {exc}")
            continue
    return out


def _predict_model(model: LoadedEnsembleModel, features_df: pd.DataFrame, clip: bool) -> pd.Series:
    cols = model.feature_columns if model.feature_columns else list(features_df.columns)

    # C4: warn when artifact feature_columns diverge from the constructed feature matrix.
    _missing = [c for c in cols if c not in features_df.columns]
    if _missing:
        print(
            f"[ensemble] WARN: {len(_missing)} feature(s) in artifact not found in model_features "
            f"(first 5: {_missing[:5]}). Predictions may be degraded."
        )

    if model.preprocessor_handled:
        # H1/C2: Pass the full sanitized DataFrame to PrefitWeightedEnsemble.predict().
        # Each member's FeaturePreprocessor.transform() independently selects its own
        # active_features (full-view OR program-view) and applies the fitted median/
        # winsorization stats.  Pre-restricting to model.feature_columns here would hide
        # features that program-view members need but that weren't in the shared preprocessor.
        X_df = features_df.replace([np.inf, -np.inf], np.nan)
    else:
        X_df = (
            features_df.reindex(columns=cols)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(-10.0, 10.0)
        )

    est_mod = getattr(model.estimator, "__module__", "") or ""
    # Use ndarray for pure sklearn estimators to avoid feature-name warnings.
    # For PrefitWeightedEnsemble (preprocessor_handled) pass the DataFrame so the
    # ensemble's internal transform() receives the column names it was fitted on.
    X = X_df.to_numpy(dtype=float, copy=True) if (est_mod.startswith("sklearn.") and not model.preprocessor_handled) else X_df

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

        if model.model_type in ("short_regressor", "short_alpha"):
            # short_alpha (DateGroupedEconomicModel, objective="short_side"): the gradient
            # descent objective assigns more-negative scores to expected decliners, so
            # .predict() already has "lower = stronger short" semantics — same as
            # short_regressor.  Backward-compat: convert predict_proba if present.
            if hasattr(est, "predict_proba"):
                proba = est.predict_proba(X)
                p_down = np.asarray(proba)[:, 1] if np.asarray(proba).ndim == 2 else np.asarray(proba)
                y = -(2.0 * p_down - 1.0)
            else:
                y = np.asarray(est.predict(X), dtype=float)
            s = pd.Series(y, index=features_df.index, dtype=float)
            if clip:
                s = s.clip(-0.3, 0.3)
            return s

        if model.model_type in ("long_alpha", "overlay_alpha"):
            # C1: DateGroupedEconomicModel trained with long_short_spread / long_only_overlay
            # objective.  .predict() returns raw portfolio-utility scores; higher = better long.
            # Same direction as "regressor" — no sign flip needed.
            y = np.asarray(est.predict(X), dtype=float)
            s = pd.Series(y, index=features_df.index, dtype=float)
            if clip:
                s = s.clip(-0.3, 0.3)
            return s

        # Generic fallback for "regressor" and unknown types.
        # Auto-detect classifiers mislabelled as "regressor": use predict_proba when
        # available (predict() returns binary {0,1} and discards probability information).
        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(X)
            y = np.asarray(proba)[:, 1] if np.asarray(proba).ndim == 2 else np.asarray(proba)
            y = 2.0 * y - 1.0
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

