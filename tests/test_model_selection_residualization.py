import warnings

import numpy as np
import model_selection.residualization as residualization

from model_selection.residualization import residualize_against_controls


def test_residualize_against_controls_handles_pathological_control_scale_without_runtime_warning() -> None:
    x = np.array([1e308, -1e308, np.inf, -np.inf, 1.0, -1.0, 0.0, 2.0, -2.0, 3.0])
    y = np.array([0.5, -0.4, np.inf, -np.inf, 0.1, -0.1, 0.0, 0.2, -0.2, 0.3])
    z = np.column_stack(
        [
            np.array([1e308, -1e308, np.inf, -np.inf, 1.0, -1.0, 0.0, 2.0, -2.0, 3.0]),
            np.array([1e308, -1e308, np.inf, -np.inf, 1.0, -1.0, 0.0, 2.0, -2.0, 3.0]),
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = residualize_against_controls(
            np.column_stack([x, y]),
            z,
            ridge=1e-4,
            value_cap=10.0,
            control_cap=10.0,
        )

    resid = np.asarray(result.residual, dtype=float)
    assert resid.shape == (10, 2)
    assert np.isfinite(resid).all()


def test_residualize_against_controls_fails_closed_on_explosive_coefficients(monkeypatch) -> None:
    x = np.linspace(-1.0, 1.0, 12)
    y = np.linspace(1.0, -1.0, 12)
    z = np.column_stack([x, y])

    def fake_lstsq(lhs, rhs, rcond=None):
        coef = np.full((lhs.shape[1], rhs.shape[1]), 1e12, dtype=float)
        return coef, np.array([], dtype=float), lhs.shape[1], np.ones(lhs.shape[1], dtype=float)

    monkeypatch.setattr(residualization.np.linalg, "lstsq", fake_lstsq)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = residualize_against_controls(
            np.column_stack([x, y]),
            z,
            ridge=1e-4,
            value_cap=10.0,
            control_cap=10.0,
        )

    resid = np.asarray(result.residual, dtype=float)
    assert resid.shape == (12, 2)
    assert np.isfinite(resid).all()
    assert not result.used_projection
    assert result.fallback_reason == "coef_scale_invalid"
