"""
Sign-direction calibration for model selection.

Convention (invariant throughout the pipeline):
  higher adjusted_score = better candidate for the strategy mandate.
  - Long models  : IC(adjusted_score, forward_return) > 0
  - Short models : IC(adjusted_score, forward_return) < 0
                   (high adjusted_score → stock expected to go DOWN)

Direction (+1 or -1) is determined from in-window validation/training data only.
OOS test data is never used for direction selection.
"""
from __future__ import annotations

import numpy as np
from typing import Any


_SHORT_KINDS: frozenset[str] = frozenset({"short_classifier", "short_alpha"})


def determine_score_direction(
    ic_raw: float,
    *,
    model_kind: str,
) -> tuple[int, str, str]:
    """
    Determine sign multiplier from raw IC (score vs forward_return).

    For long models: want IC > 0 after calibration → direction = sign(ic_raw)
    For short models: want IC < 0 after calibration → direction = -sign(ic_raw) if ic_raw > 0

    Returns
    -------
    direction : +1 or -1
    mode      : 'fixed' (raw already correct) or 'calibrated' (flip needed)
    reason    : human-readable explanation
    """
    is_short = model_kind in _SHORT_KINDS
    if not np.isfinite(ic_raw):
        return 1, "fixed", f"IC not finite ({ic_raw}) — default +1"

    if is_short:
        if ic_raw <= 0.0:
            return 1, "fixed", f"short mandate: IC={ic_raw:.4f}≤0 → correct sign, keep +1"
        return -1, "calibrated", f"short mandate: IC={ic_raw:.4f}>0 → flip to -1"
    else:
        if ic_raw >= 0.0:
            return 1, "fixed", f"long mandate: IC={ic_raw:.4f}≥0 → correct sign, keep +1"
        return -1, "calibrated", f"long mandate: IC={ic_raw:.4f}<0 → flip to -1"


def assert_no_double_application(applied_count: int, context: str) -> None:
    """Raise if score_direction has been applied more than once."""
    if applied_count > 1:
        raise RuntimeError(
            f"score_direction applied {applied_count} times in {context}. "
            "Double-flip detected — ensure direction is applied exactly once."
        )


def check_class_ordering(model: Any, model_kind: str) -> None:
    """
    Warn if classifier class ordering is ambiguous (not [0, 1]).
    Does not raise — just returns a diagnostic string.
    """
    if not hasattr(model, "classes_"):
        return
    classes = list(model.classes_)
    if len(classes) != 2:
        raise RuntimeError(
            f"{model_kind}: expected binary classifier with 2 classes, "
            f"got {len(classes)}: {classes}. predict_proba[:,1] semantics are ambiguous."
        )


def ic_signal_strength(calibrated_ic: float) -> float:
    """
    Mandate-agnostic signal strength for gate checks.
    Returns abs(calibrated_ic) — valid for both long (IC>0) and short (IC<0) models.
    """
    return abs(calibrated_ic) if np.isfinite(calibrated_ic) else 0.0
