"""
Regime-specific signal confidence multiplier loader/validator.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

SUPPORTED_REGIMES = ["Bull", "Bear", "Sideways", "Crisis"]
_KEY_MAP = {
    "Bull": "signal_confidence_multiplier_bull",
    "Bear": "signal_confidence_multiplier_bear",
    "Sideways": "signal_confidence_multiplier_sideways",
    "Crisis": "signal_confidence_multiplier_crisis",
}


def _coerce_multiplier(value, regime: str) -> float:
    """Return sanitized multiplier for one regime."""
    try:
        v = float(value)
    except Exception:
        logger.warning("%s multiplier is not numeric (%r). Using 1.0 fallback.", regime, value)
        return 1.0
    if not (v > 0 and v <= 3.0):
        logger.warning("%s multiplier %.4f is outside (0, 3.0]. Using 1.0 fallback.", regime, v)
        return 1.0
    if v < 0.5 or v > 2.0:
        logger.warning("%s multiplier %.4f is outside recommended 0.5-2.0 range.", regime, v)
    return v


def load_regime_multipliers(config_path: str = "backtest_config.yaml") -> dict:
    """
    Load regime-specific confidence multipliers from YAML config.

    Returns
    -------
    dict
        Example:
        {'Bull': 1.0, 'Bear': 1.0, 'Sideways': 0.3, 'Crisis': 1.0}
    """
    out = {r: 1.0 for r in SUPPORTED_REGIMES}
    cfg_file = Path(config_path)

    if not os.path.exists(cfg_file):
        logger.warning("Config file not found at %s. Using all multipliers=1.0.", cfg_file)
        return out

    try:
        with open(cfg_file, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Failed to parse config %s (%s). Using all multipliers=1.0.", cfg_file, exc)
        return out

    # Nested sections (same keys may appear under signals or backtest)
    signals = cfg.get("signals", {}) if isinstance(cfg, dict) else {}
    backtest = cfg.get("backtest", {}) if isinstance(cfg, dict) else {}
    for regime in SUPPORTED_REGIMES:
        key = _KEY_MAP[regime]
        raw = None
        if isinstance(signals, dict) and key in signals:
            raw = signals.get(key)
        elif isinstance(backtest, dict) and key in backtest:
            raw = backtest.get(key)
        elif isinstance(cfg, dict) and key in cfg:
            raw = cfg.get(key)
        if raw is None:
            out[regime] = 1.0
            continue
        out[regime] = _coerce_multiplier(raw, regime)

    return out


def get_multiplier(regime: str, multipliers: dict, default: float = 1.0) -> float:
    """Case-insensitive regime lookup with safe fallback."""
    if not isinstance(multipliers, dict):
        return float(default)
    regime_norm = str(regime).strip().lower()
    canon = {r.lower(): r for r in SUPPORTED_REGIMES}
    key = canon.get(regime_norm)
    if key is None:
        return float(default)
    try:
        return float(multipliers.get(key, default))
    except Exception:
        return float(default)


def validate_multiplier_config(multipliers: dict) -> list[str]:
    """Validate loaded multipliers and return human-readable warnings."""
    warnings: list[str] = []
    for regime in SUPPORTED_REGIMES:
        try:
            v = float(multipliers.get(regime, 1.0))
        except Exception:
            warnings.append(f"{regime} multiplier is non-numeric — defaulting to 1.0")
            continue
        if v <= 0 or v > 3.0:
            warnings.append(f"{regime} multiplier {v} is invalid — expected (0, 3.0]")
        if v < 0.5 or v > 2.0:
            warnings.append(
                f"{regime} multiplier {v} is aggressive — consider 0.5-1.0"
                if v < 0.5
                else f"{regime} multiplier {v} is high — consider 0.5-2.0"
            )
    return warnings


import math


_REGIME_CFG_ATTRS = {
    "Bull":     "signal_confidence_multiplier_bull",
    "Bear":     "signal_confidence_multiplier_bear",
    "Sideways": "signal_confidence_multiplier_sideways",
    "Crisis":   "signal_confidence_multiplier_crisis",
}


def signal_confidence_multiplier_for_regime(
    regime_today: str,
    config,
    loaded_multipliers: dict | None = None,
) -> float:
    """
    Return the signal-confidence multiplier for *regime_today*.

    Resolution order:
    1. ``config`` attribute (e.g. ``signal_confidence_multiplier_bull``)
    2. ``loaded_multipliers`` dict (from ``load_regime_multipliers``)
    3. Fallback map on ``config``
    4. ``config.signal_confidence_multiplier`` or 1.0

    Extracted from ``Backtester._signal_confidence_multiplier_for_regime``.
    """
    regime_key = str(regime_today or "Sideways")
    attr = _REGIME_CFG_ATTRS.get(regime_key)
    multiplier: float | None = None

    if attr:
        raw = getattr(config, attr, None)
        if raw is not None:
            try:
                fv = float(raw)
                if math.isfinite(fv) and fv > 0:
                    multiplier = fv
            except (TypeError, ValueError):
                pass

    if multiplier is None and loaded_multipliers is not None:
        try:
            multiplier = float(get_multiplier(regime_key, loaded_multipliers, default=1.0))
        except Exception:
            multiplier = None

    if multiplier is None:
        fallback = {
            "Bull":     getattr(config, "signal_confidence_multiplier_bull", None),
            "Sideways": getattr(config, "signal_confidence_multiplier_sideways", None),
            "Bear":     getattr(config, "signal_confidence_multiplier_bear", None),
            "Crisis":   getattr(config, "signal_confidence_multiplier_crisis", None),
        }.get(regime_key)
        if fallback is not None:
            try:
                multiplier = float(fallback)
            except (TypeError, ValueError):
                multiplier = None

    if multiplier is None:
        base = getattr(config, "signal_confidence_multiplier", None)
        multiplier = float(base) if base is not None else 1.0

    return multiplier


def threshold_aggressiveness_for_regime(regime_today: str, config) -> float:
    """
    Return the rolling-std gate threshold scaler for *regime_today*.

    Values > 1.0 require stronger |score| to enter. Extracted from
    ``Backtester._threshold_aggressiveness_for_regime``.
    """
    regime_key = str(regime_today or "Sideways")
    raw_map = getattr(config, "regime_threshold_aggressiveness", None)
    if not isinstance(raw_map, dict):
        return 1.0
    v = raw_map.get(regime_key)
    if v is None:
        return 1.0
    try:
        a = float(v)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(a) or a <= 0:
        return 1.0
    return float(min(a, 20.0))


if __name__ == "__main__":
    loaded = load_regime_multipliers("backtest_config.yaml")
    print("Loaded regime multipliers:")
    for r in SUPPORTED_REGIMES:
        print(f"  {r:<9}: {loaded.get(r, 1.0)}")
    msgs = validate_multiplier_config(loaded)
    if msgs:
        print("Validation warnings:")
        for m in msgs:
            print(f"  - {m}")
    else:
        print("No validation warnings.")
