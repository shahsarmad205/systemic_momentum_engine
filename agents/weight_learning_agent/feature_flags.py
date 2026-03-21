"""
Feature ablation flags for weight learning (Phase 1 baseline vs Phase 2 groups).

Set environment variable before training/backtests:

    export TSE_ABLATION_STEP=0   # Phase 1 baseline only (~10 core features)
    export TSE_ABLATION_STEP=1   # + Group A (RSI + BB)
    export TSE_ABLATION_STEP=2   # + Group B (dist_high, dist_low)
    export TSE_ABLATION_STEP=3   # + Group C (overnight_gap, intraday_rev)
    export TSE_ABLATION_STEP=4   # + Group D (sector_relative 20d/60d)
    export TSE_ABLATION_STEP=5   # + Group E (vix_zscore, vol_spike, vix_term_zscore)
    export TSE_ABLATION_STEP=6   # + vol structure extras (rolling 5/60, vov, jump, cs_momentum)

Unset TSE_ABLATION_STEP → full feature matrix (no zeroing).
"""

from __future__ import annotations

import os

# Phase 1 core: trend, momentum, core vol, liquidity, market corr, regime flag, sentiment slots
PHASE1_BASELINE_FEATURES: frozenset[str] = frozenset(
    {
        "f_trend",
        "f_regional",
        "f_global",
        "f_social",
        "ret_5d",
        "ret_10d",
        "rolling_vol_10",
        "rolling_vol_20",
        "relative_volume",
        "volume_zscore",
        "rolling_corr_market_20",
        "is_high_vol_regime",
    }
)

GROUP_A_MR_RSI_BB: frozenset[str] = frozenset({"rsi_zscore", "bb_position"})
GROUP_B_MR_RANGE: frozenset[str] = frozenset({"dist_high", "dist_low"})
GROUP_C_MR_MICRO: frozenset[str] = frozenset({"overnight_gap", "intraday_rev"})
GROUP_D_SECTOR_REL: frozenset[str] = frozenset({"sector_relative_20d", "sector_relative_60d"})
GROUP_E_MACRO_VIX: frozenset[str] = frozenset({"vix_zscore", "vol_spike", "vix_term_zscore"})

# Richer book beyond cumulative A–E (matches full COMPOUND tail)
STRUCTURE_AND_CS_FEATURES: frozenset[str] = frozenset(
    {
        "rolling_vol_5",
        "rolling_vol_60",
        "vol_of_vol_20",
        "jump_indicator",
        "cs_momentum_percentile",
    }
)


def ablation_step_from_env() -> int | None:
    raw = os.environ.get("TSE_ABLATION_STEP", "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except ValueError:
        return None
    return max(0, min(6, v))


def active_features_for_step(step: int) -> frozenset[str]:
    s = set(PHASE1_BASELINE_FEATURES)
    if step >= 1:
        s |= set(GROUP_A_MR_RSI_BB)
    if step >= 2:
        s |= set(GROUP_B_MR_RANGE)
    if step >= 3:
        s |= set(GROUP_C_MR_MICRO)
    if step >= 4:
        s |= set(GROUP_D_SECTOR_REL)
    if step >= 5:
        s |= set(GROUP_E_MACRO_VIX)
    if step >= 6:
        s |= set(STRUCTURE_AND_CS_FEATURES)
    return frozenset(s)


def feature_columns_to_zero_for_ablation() -> set[str]:
    """
    Names in COMPOUND_AND_PRICE_FEATURES that should be forced to 0 for the current
    ablation step. Empty set when TSE_ABLATION_STEP is unset.
    """
    step = ablation_step_from_env()
    if step is None:
        return set()
    from agents.weight_learning_agent.weight_model import COMPOUND_AND_PRICE_FEATURES

    active = active_features_for_step(step)
    return {c for c in COMPOUND_AND_PRICE_FEATURES if c not in active}


def describe_ablation_step(step: int) -> str:
    labels = [
        "Phase1 baseline",
        "+ Group A (RSI+BB)",
        "+ Group B (range)",
        "+ Group C (micro)",
        "+ Group D (sector rel)",
        "+ Group E (VIX macro)",
        "+ structure + cs_momentum",
    ]
    parts = [labels[0]]
    for i in range(1, min(step + 1, len(labels))):
        parts.append(labels[i])
    return " → ".join(parts)
