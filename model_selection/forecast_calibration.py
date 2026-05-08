from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreCalibrationResult:
    """Causal mapping from model score units into expected-return units."""

    slope: float
    intercept: float
    shrinkage: float
    n_obs: int
    n_dates: int
    score_std: float
    target_std: float
    residual_std: float
    slope_tstat: float
    method: str

    def transform(self, raw_score: np.ndarray) -> np.ndarray:
        arr = np.asarray(raw_score, dtype=float)
        out = self.intercept + self.slope * arr
        return out.astype(float)

    def diagnostics(self, prefix: str = "forecast_calibration") -> dict[str, float | str]:
        return {
            f"{prefix}_slope": float(self.slope),
            f"{prefix}_intercept": float(self.intercept),
            f"{prefix}_shrinkage": float(self.shrinkage),
            f"{prefix}_n_obs": float(self.n_obs),
            f"{prefix}_n_dates": float(self.n_dates),
            f"{prefix}_score_std": float(self.score_std),
            f"{prefix}_target_std": float(self.target_std),
            f"{prefix}_residual_std": float(self.residual_std),
            f"{prefix}_slope_tstat": float(self.slope_tstat),
            f"{prefix}_method": str(self.method),
        }


def fit_score_calibrator(
    panel: pd.DataFrame,
    raw_score: np.ndarray,
    *,
    target_col: str = "target_return",
    date_col: str = "date",
) -> ScoreCalibrationResult:
    """Fit a causal score-to-forward-return calibration.

    The fit uses cross-sectionally demeaned score and target observations so the
    slope is learned from stock-selection information rather than market drift.
    Shrinkage is empirical-Bayes style: weak calibration slopes are pulled
    toward zero by their own signal-to-noise instead of by a fixed threshold.
    """

    if panel is None or panel.empty or target_col not in panel.columns:
        return ScoreCalibrationResult(1.0, 0.0, 1.0, 0, 0, float("nan"), float("nan"), float("nan"), float("nan"), "identity_no_target")

    df = panel[[date_col, target_col]].copy() if date_col in panel.columns else panel[[target_col]].copy()
    scores = np.asarray(raw_score, dtype=float)
    n = min(len(df), len(scores))
    if n <= 0:
        return ScoreCalibrationResult(1.0, 0.0, 1.0, 0, 0, float("nan"), float("nan"), float("nan"), float("nan"), "identity_empty")

    df = df.iloc[:n].copy()
    df["_score"] = scores[:n]
    df["_target"] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["_score", "_target"])
    if df.empty:
        return ScoreCalibrationResult(1.0, 0.0, 1.0, 0, 0, float("nan"), float("nan"), float("nan"), float("nan"), "identity_no_finite")

    if date_col in df.columns:
        date_key = pd.to_datetime(df[date_col], errors="coerce")
        score_center = df["_score"] - df.groupby(date_key, sort=False)["_score"].transform("mean")
        target_center = df["_target"] - df.groupby(date_key, sort=False)["_target"].transform("mean")
        n_dates = int(date_key.nunique())
    else:
        score_center = df["_score"] - float(df["_score"].mean())
        target_center = df["_target"] - float(df["_target"].mean())
        n_dates = 1

    y_raw = target_center.to_numpy(dtype=float)
    mask = np.isfinite(score_center.to_numpy(dtype=float)) & np.isfinite(y_raw)
    x = score_center.to_numpy(dtype=float)[mask]
    y = y_raw[mask]
    n_obs = int(len(x))
    if n_obs < 3:
        return ScoreCalibrationResult(1.0, 0.0, 1.0, n_obs, n_dates, float("nan"), float("nan"), float("nan"), float("nan"), "identity_insufficient")

    ssx = float(np.dot(x, x))
    score_std = float(np.std(x, ddof=1)) if n_obs > 1 else float("nan")
    target_std = float(np.std(y, ddof=1)) if n_obs > 1 else float("nan")
    if not np.isfinite(ssx) or ssx <= 1e-18:
        return ScoreCalibrationResult(1.0, 0.0, 1.0, n_obs, n_dates, score_std, target_std, float("nan"), float("nan"), "identity_zero_score_var")

    raw_slope = float(np.dot(x, y) / ssx)
    residual = y - raw_slope * x
    dof = max(n_obs - 1, 1)
    residual_var = float(np.dot(residual, residual) / dof)
    residual_std = float(np.sqrt(max(residual_var, 0.0)))
    slope_se = float(np.sqrt(residual_var / ssx)) if ssx > 0 else float("nan")
    if np.isfinite(slope_se) and slope_se > 0:
        slope_tstat = float(raw_slope / slope_se)
    elif residual_var <= 1e-24 and abs(raw_slope) > 0:
        slope_tstat = float(np.sign(raw_slope) * np.inf)
    else:
        slope_tstat = float("nan")

    if np.isinf(slope_tstat):
        shrinkage = 1.0
    elif np.isfinite(slope_tstat):
        snr2 = slope_tstat * slope_tstat
        shrinkage = float(snr2 / (snr2 + 1.0))
    else:
        shrinkage = 0.0
    slope = float(raw_slope * shrinkage)

    # Preserve the unconditional mean only after learning the cross-sectional
    # slope. The optimizer mostly consumes ranks/relative alpha, but this keeps
    # score units economically interpretable for diagnostics and calibration logs.
    score_mean = float(pd.to_numeric(df["_score"], errors="coerce").mean())
    target_mean = float(pd.to_numeric(df["_target"], errors="coerce").mean())
    intercept = float(target_mean - slope * score_mean)

    return ScoreCalibrationResult(
        slope=slope,
        intercept=intercept,
        shrinkage=shrinkage,
        n_obs=n_obs,
        n_dates=n_dates,
        score_std=score_std,
        target_std=target_std,
        residual_std=residual_std,
        slope_tstat=slope_tstat,
        method="cross_sectional_linear_shrunk",
    )


def calibrate_scores(
    calibration_panel: pd.DataFrame,
    calibration_raw_score: np.ndarray,
    evaluation_raw_score: np.ndarray,
    *,
    target_col: str = "target_return",
    date_col: str = "date",
) -> tuple[np.ndarray, ScoreCalibrationResult]:
    result = fit_score_calibrator(
        calibration_panel,
        calibration_raw_score,
        target_col=target_col,
        date_col=date_col,
    )
    return result.transform(np.asarray(evaluation_raw_score, dtype=float)), result
