"""
Regime exposure utilities.

Canonical exposure mapping:
    Bull -> 1.0
    Sideways -> 0.8
    Bear -> 0.6
    Crisis -> 0.3
"""

from __future__ import annotations

from typing import Iterable
import math

CANONICAL_REGIME_EXPOSURE: dict[str, float] = {
    "Bull": 1.0,
    "Sideways": 0.8,
    "Bear": 0.6,
    "Crisis": 0.3,
}


def validate_regime_keys(regime_adjustments: dict[str, dict] | None) -> tuple[set[str], set[str]]:
    """
    Return (missing, unexpected) regime names relative to canonical set.
    """
    if regime_adjustments is None:
        return (set(CANONICAL_REGIME_EXPOSURE.keys()), set())
    keys = set(regime_adjustments.keys())
    canonical = set(CANONICAL_REGIME_EXPOSURE.keys())
    return (canonical - keys, keys - canonical)


def resolve_regime_position_scale(
    regime: str,
    regime_adjustments: dict[str, dict] | None = None,
    *,
    safe_default: float = 0.8,
    strict: bool = False,
) -> float:
    """
    Resolve position scale for a regime with safe unknown handling.

    Precedence:
    1) regime_adjustments[regime]['position_scale'] when present/finite
    2) canonical mapping
    3) safe_default (or raise ValueError when strict=True)
    """
    r = str(regime)
    if regime_adjustments and r in regime_adjustments:
        entry = regime_adjustments[r]
        v = entry.get("position_scale") if isinstance(entry, dict) else None
        if v is not None:
            try:
                fv = float(v)
                if math.isfinite(fv) and fv >= 0:
                    return fv
            except Exception:
                pass
    if r in CANONICAL_REGIME_EXPOSURE:
        return float(CANONICAL_REGIME_EXPOSURE[r])
    if strict:
        raise ValueError(f"Unknown regime: {r}")
    return float(safe_default)


def exposure_path_for_regimes(
    regimes: Iterable[str],
    regime_adjustments: dict[str, dict] | None = None,
    *,
    safe_default: float = 0.8,
    strict: bool = False,
) -> list[float]:
    return [
        resolve_regime_position_scale(
            r,
            regime_adjustments=regime_adjustments,
            safe_default=safe_default,
            strict=strict,
        )
        for r in regimes
    ]

