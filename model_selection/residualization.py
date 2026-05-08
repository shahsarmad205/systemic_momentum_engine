from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np


@dataclass(frozen=True)
class ResidualizationResult:
    residual: np.ndarray
    n_controls_in: int
    n_controls_used: int
    effective_rank: int
    used_projection: bool
    fallback_reason: str | None = None
    condition_number: float = float("nan")
    max_abs_coef: float = float("nan")


def safe_center(values: np.ndarray, *, cap: float = 5.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if cap > 0.0:
        clean = np.clip(clean, -cap, cap)
    if clean.size == 0:
        return clean
    centered = clean - np.mean(clean, axis=0, keepdims=True)
    if cap > 0.0:
        centered = np.clip(centered, -cap, cap)
    return np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)


def residualize_against_controls(
    values: np.ndarray,
    controls: np.ndarray | None,
    *,
    ridge: float,
    value_cap: float = 5.0,
    control_cap: float = 10.0,
    min_control_std: float = 1e-10,
    max_condition_number: float = 1e8,
    max_abs_coef: float = 1e4,
) -> ResidualizationResult:
    """
    Stable ridge residualization via augmented least squares.

    This avoids the fragile normal-equation form ``(Z'Z)^-1 Z'Y`` which squares
    the condition number and is prone to overflow on wide/collinear control
    blocks. The augmented system solves:

      min_B ||Z B - Y||^2 + ridge ||B||^2

    through ``np.linalg.lstsq([Z; sqrt(ridge) I], [Y; 0])``.
    """
    orig = np.asarray(values, dtype=float)
    vector_input = orig.ndim == 1
    y = safe_center(orig, cap=float(value_cap))
    n_obs = y.shape[0]

    if controls is None:
        resid = y[:, 0] if vector_input else y
        return ResidualizationResult(
            residual=resid,
            n_controls_in=0,
            n_controls_used=0,
            effective_rank=0,
            used_projection=False,
        )

    def _fallback(
        reason: str,
        *,
        n_controls_used: int,
        effective_rank: int = 0,
        condition_number: float = float("nan"),
        coef_abs: float = float("nan"),
    ) -> ResidualizationResult:
        resid = y[:, 0] if vector_input else y
        return ResidualizationResult(
            residual=resid,
            n_controls_in=n_controls_in,
            n_controls_used=n_controls_used,
            effective_rank=effective_rank,
            used_projection=False,
            fallback_reason=reason,
            condition_number=condition_number,
            max_abs_coef=coef_abs,
        )

    z_raw = np.asarray(controls, dtype=float)
    if z_raw.ndim == 1:
        z_raw = z_raw[:, None]
    if z_raw.shape[0] != n_obs:
        resid = y[:, 0] if vector_input else y
        return ResidualizationResult(
            residual=resid,
            n_controls_in=int(z_raw.shape[1]) if z_raw.ndim == 2 else 0,
            n_controls_used=0,
            effective_rank=0,
            used_projection=False,
            fallback_reason="shape_mismatch",
        )

    n_controls_in = int(z_raw.shape[1])
    if n_controls_in == 0:
        resid = y[:, 0] if vector_input else y
        return ResidualizationResult(
            residual=resid,
            n_controls_in=0,
            n_controls_used=0,
            effective_rank=0,
            used_projection=False,
        )

    z = safe_center(z_raw, cap=float(control_cap))
    control_std = np.std(z, axis=0)
    keep = np.isfinite(control_std) & (control_std > float(min_control_std))
    z = z[:, keep]
    if z.size == 0:
        return _fallback("constant_controls", n_controls_used=0)

    norms = np.sqrt(np.sum(z * z, axis=0))
    keep = np.isfinite(norms) & (norms > float(min_control_std))
    z = z[:, keep]
    norms = norms[keep]
    if z.size == 0:
        return _fallback("degenerate_controls", n_controls_used=0)

    z = z / norms
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(z).all() or float(np.abs(z).max(initial=0.0)) > 1e6:
        return _fallback("control_scale_invalid", n_controls_used=0)

    k = int(z.shape[1])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                sqrt_ridge = float(np.sqrt(max(float(ridge), 0.0)))
                lhs = np.vstack([z, sqrt_ridge * np.eye(k, dtype=float)])
                rhs = np.vstack([y, np.zeros((k, y.shape[1]), dtype=float)])
                coef, _, rank, singular_values = np.linalg.lstsq(lhs, rhs, rcond=None)
                coef = np.asarray(coef, dtype=float)
                if not np.isfinite(coef).all():
                    raise FloatingPointError("non_finite_coef")
                coef_abs = float(np.max(np.abs(coef), initial=0.0))
                if coef_abs > float(max_abs_coef):
                    raise FloatingPointError("coef_scale_invalid")
                sv = np.asarray(singular_values, dtype=float)
                sv = sv[np.isfinite(sv) & (sv > float(min_control_std))]
                condition_number = float("nan")
                if sv.size:
                    condition_number = float(np.max(sv) / np.min(sv))
                    if condition_number > float(max_condition_number):
                        raise FloatingPointError("condition_number_invalid")
                fitted = z @ coef
                if not np.isfinite(fitted).all():
                    raise FloatingPointError("non_finite_fitted")
                resid_matrix = y - fitted
                if not np.isfinite(resid_matrix).all():
                    raise FloatingPointError("non_finite_residuals")
        resid = resid_matrix[:, 0] if vector_input else resid_matrix
        return ResidualizationResult(
            residual=resid,
            n_controls_in=n_controls_in,
            n_controls_used=k,
            effective_rank=int(rank),
            used_projection=True,
            condition_number=condition_number,
            max_abs_coef=coef_abs,
        )
    except (RuntimeWarning, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        reason = str(exc) or exc.__class__.__name__
        if reason not in {
            "non_finite_coef",
            "coef_scale_invalid",
            "condition_number_invalid",
            "non_finite_fitted",
            "non_finite_residuals",
        }:
            reason = "lstsq_failed"
        return _fallback(reason, n_controls_used=k)
