from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DrawdownOverlayState:
    peak_equity: float
    current_equity: float
    drawdown: float
    gross_multiplier: float
    halt_new_risk: bool
    flatten_all: bool


def compute_drawdown_overlay(
    *,
    current_equity: float,
    peak_equity: float,
    overlay_cfg: dict | None,
) -> DrawdownOverlayState:
    """
    Shared drawdown overlay used by research and live paths.

    The overlay scales gross exposure linearly once drawdown exceeds
    ``start_pct`` and reaches ``min_gross_multiplier`` at ``max_pct``.
    """
    cfg = overlay_cfg or {}
    enabled = bool(cfg.get("enabled", False))
    current = float(current_equity or 0.0)
    peak = float(max(peak_equity or 0.0, current))
    drawdown = 0.0 if peak <= 0 else max((peak - current) / peak, 0.0)

    if not enabled:
        return DrawdownOverlayState(
            peak_equity=peak,
            current_equity=current,
            drawdown=drawdown,
            gross_multiplier=1.0,
            halt_new_risk=False,
            flatten_all=False,
        )

    start_pct = float(cfg.get("start_pct", 0.05) or 0.05)
    max_pct = float(cfg.get("max_pct", 0.12) or 0.12)
    min_gross = float(cfg.get("min_gross_multiplier", 0.35) or 0.35)
    halt_pct = float(cfg.get("halt_new_risk_pct", max_pct) or max_pct)
    flatten_pct = float(cfg.get("flatten_all_pct", 0.0) or 0.0)

    if drawdown <= start_pct:
        gross_multiplier = 1.0
    elif max_pct <= start_pct:
        gross_multiplier = min_gross
    else:
        frac = min(max((drawdown - start_pct) / (max_pct - start_pct), 0.0), 1.0)
        gross_multiplier = 1.0 - frac * (1.0 - min_gross)

    gross_multiplier = max(min(float(gross_multiplier), 1.0), min_gross)
    halt_new_risk = drawdown >= halt_pct if halt_pct > 0 else False
    flatten_all = drawdown >= flatten_pct if flatten_pct > 0 else False

    return DrawdownOverlayState(
        peak_equity=peak,
        current_equity=current,
        drawdown=drawdown,
        gross_multiplier=gross_multiplier,
        halt_new_risk=halt_new_risk,
        flatten_all=flatten_all,
    )
