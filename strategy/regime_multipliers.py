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
        with open(cfg_file, "r", encoding="utf-8") as f:
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
