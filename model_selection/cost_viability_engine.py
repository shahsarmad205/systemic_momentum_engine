"""
Cost Viability Engine — Institutional
======================================
Evaluates every feature, horizon, sleeve, and candidate signal on expected net
alpha after realistic execution costs, turnover, decay, liquidity, and capacity.

This module replaces the flat 10bps cost proxy in horizon_eligibility.py with
a full institutional cost model.

Architecture:
  CostModel (abstract)
    ├── SimpleBpsCostModel          — flat bps baseline
    └── SqrtImpactCostModel         — Almgren-Chriss square-root impact
  
  CostViabilityEngine
    ├── compute_gross_alpha()
    ├── compute_expected_cost()
    ├── compute_alpha_cost_ratio()
    ├── compute_turnover_adjusted_ic()
    ├── compute_decay_adjusted_alpha()
    ├── compute_net_expected_alpha()
    ├── compute_capacity_score()
    ├── compute_liquidity_diagnostics()
    ├── classify_cost_status()
    └── run_stress_test()
  
  AlphaToTradeDecision
    ├── compute_incremental_alpha()
    ├── compute_incremental_cost()
    └── decide_trade()
  
  NoTradeBandEngine
    ├── compute_band_width()
    └── apply_bands()
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Classification enums
# ---------------------------------------------------------------------------

class CostStatus(str, Enum):
    COST_VIABLE = "cost_viable"
    MARGINAL = "marginal"
    COST_DOMINATED = "cost_dominated"
    TURNOVER_DOMINATED = "turnover_dominated"
    IMPACT_DOMINATED = "impact_dominated"
    BORROW_DOMINATED = "borrow_dominated"
    LIQUIDITY_INSUFFICIENT = "liquidity_insufficient"
    ALPHA_TOO_WEAK = "alpha_too_weak"
    CAPACITY_INSUFFICIENT = "capacity_insufficient"
    UNSTABLE_COST_ESTIMATE = "unstable_cost_estimate"
    MISSING_PIT_DIAGNOSTICS = "missing_pit_diagnostics"


class CostModelMode(str, Enum):
    SIMPLE_BPS = "simple_bps"
    SQRT_IMPACT = "sqrt_impact"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_cost_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load cost viability config from YAML. Falls back to defaults if missing."""
    if path is None:
        path = Path(__file__).parent.parent / "cost_viability_config.yaml"
    path = Path(path)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return _default_config()


def _default_config() -> dict[str, Any]:
    """Inline defaults matching cost_viability_config.yaml."""
    return {
        "cost_model": {
            "mode": "sqrt_impact",
            "commission_bps": 1.0,
            "spread_bps": 1.0,
            "borrow_bps_annual": 50.0,
            "impact_eta": 0.142,
            "impact_alpha": 0.314,
            "impact_gamma": 0.6,
            "default_adv_usd": 50_000_000,
            "default_daily_vol": 0.02,
            "max_participation_rate": 0.10,
            "max_impact_bps": 150.0,
            "min_adv_usd": 5_000_000,
            "permanent_impact_decay_days": 5,
            "financing_rate_annual": 0.00,
        },
        "alpha_estimation": {
            "default_sigma_annual": 0.20,
            "min_observations": 100,
            "min_dates": 30,
        },
        "classification": {
            "min_alpha_cost_ratio_viable": 1.5,
            "min_alpha_cost_ratio_marginal": 1.0,
            "min_net_alpha_bps": 0.0,
            "max_expected_turnover": 0.80,
            "max_turnover_to_ic_ratio": 50.0,
            "max_impact_fraction_of_cost": 0.75,
            "max_borrow_fraction_of_cost": 0.50,
            "min_adv_usd_viable": 10_000_000,
            "max_participation_for_viable": 0.05,
            "min_gross_alpha_bps": 3.0,
            "min_ic_absolute": 0.005,
            "min_capacity_score": 1.0,
            "min_cost_data_coverage": 0.80,
        },
        "promotion_gates": {
            "min_alpha_cost_ratio": 1.5,
            "min_net_expected_alpha_bps": 0.0,
            "min_halflife_days": 5.0,
            "max_turnover": 0.80,
            "min_capacity_score": 1.0,
            "min_bh_significant": True,
            "min_breadth": 50,
            "min_dates": 252,
            "survive_base_stress": True,
            "survive_high_stress": False,
        },
        "stress_scenarios": {
            "low_cost": {
                "label": "Low Cost (Optimistic)",
                "overrides": {
                    "commission_bps": 0.5, "spread_bps": 0.5,
                    "impact_eta": 0.100, "borrow_bps_annual": 25.0,
                },
            },
            "base_cost": {"label": "Base Cost (Expected)", "overrides": {}},
            "high_cost": {
                "label": "High Cost (Conservative)",
                "overrides": {
                    "commission_bps": 2.0, "spread_bps": 3.0,
                    "impact_eta": 0.250, "borrow_bps_annual": 100.0,
                },
            },
            "stressed_spread": {
                "label": "Stressed Spread (Widened Markets)",
                "overrides": {"spread_bps": 10.0},
            },
            "stressed_impact": {
                "label": "Stressed Impact (Low Liquidity)",
                "overrides": {"impact_eta": 0.400, "impact_gamma": 0.7},
            },
            "stressed_liquidity": {
                "label": "Stressed Liquidity (ADV Halved)",
                "overrides": {"default_adv_usd": 25_000_000},
            },
            "stressed_borrow": {
                "label": "Stressed Borrow (Hard-to-Borrow)",
                "overrides": {"borrow_bps_annual": 300.0},
            },
            "crisis": {
                "label": "Crisis (All Costs Elevated)",
                "overrides": {
                    "commission_bps": 3.0, "spread_bps": 15.0,
                    "impact_eta": 0.500, "borrow_bps_annual": 500.0,
                    "max_participation_rate": 0.03,
                },
            },
        },
        "alpha_to_trade": {
            "enabled": True,
            "min_alpha_to_trade_ratio": 1.5,
            "min_incremental_alpha_bps": 1.0,
            "max_single_rebalance_turnover": 0.20,
        },
        "no_trade_bands": {
            "enabled": True,
            "base_band_width": 0.010,
            "cost_scaling_factor": 2.0,
            "vol_scaling_factor": 0.5,
            "liquidity_scaling_factor": 0.3,
            "signal_scaling_factor": -0.01,
            "min_band_width": 0.002,
            "max_band_width": 0.050,
            "max_total_drift": 0.05,
        },
    }


# ---------------------------------------------------------------------------
# Cost model interface
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostBreakdown:
    """Detailed cost breakdown in basis points (round-trip)."""
    total_bps: float
    commission_bps: float
    spread_bps: float
    temporary_impact_bps: float
    permanent_impact_bps: float
    borrow_bps: float
    financing_bps: float
    participation_rate: float
    adv_usd: float
    daily_vol: float
    degraded_quality: bool = False
    degradation_reason: str = ""


class CostModel(ABC):
    """Abstract cost model interface."""

    @abstractmethod
    def compute_round_trip_cost(
        self,
        order_usd: float,
        adv_usd: float,
        daily_vol: float,
        horizon_days: int,
        is_short: bool = False,
        position_weight: float = 0.0,
        capital: float = 10_000_000.0,
    ) -> CostBreakdown:
        ...


class SimpleBpsCostModel(CostModel):
    """Flat bps cost model — baseline only."""

    def __init__(self, config: dict[str, Any]) -> None:
        cm = config.get("cost_model", {})
        self.commission_bps = float(cm.get("commission_bps", 1.0))
        self.spread_bps = float(cm.get("spread_bps", 1.0))
        self.borrow_bps_annual = float(cm.get("borrow_bps_annual", 50.0))
        self.financing_rate_annual = float(cm.get("financing_rate_annual", 0.0))

    def compute_round_trip_cost(
        self,
        order_usd: float,
        adv_usd: float,
        daily_vol: float,
        horizon_days: int,
        is_short: bool = False,
        position_weight: float = 0.0,
        capital: float = 10_000_000.0,
    ) -> CostBreakdown:
        commission = self.commission_bps * 2  # round-trip
        spread = self.spread_bps              # round-trip (full spread)
        borrow = (self.borrow_bps_annual / 252.0 * horizon_days) if is_short else 0.0
        financing = (self.financing_rate_annual * 10000 / 252.0 * horizon_days * abs(position_weight)) if self.financing_rate_annual > 0 else 0.0
        total = commission + spread + borrow + financing
        return CostBreakdown(
            total_bps=total,
            commission_bps=commission,
            spread_bps=spread,
            temporary_impact_bps=0.0,
            permanent_impact_bps=0.0,
            borrow_bps=borrow,
            financing_bps=financing,
            participation_rate=0.0,
            adv_usd=adv_usd,
            daily_vol=daily_vol,
        )


class SqrtImpactCostModel(CostModel):
    """Almgren-Chriss square-root impact model."""

    def __init__(self, config: dict[str, Any]) -> None:
        cm = config.get("cost_model", {})
        self.commission_bps = float(cm.get("commission_bps", 1.0))
        self.spread_bps = float(cm.get("spread_bps", 1.0))
        self.borrow_bps_annual = float(cm.get("borrow_bps_annual", 50.0))
        self.eta = float(cm.get("impact_eta", 0.142))
        self.alpha_coef = float(cm.get("impact_alpha", 0.314))
        self.gamma = float(cm.get("impact_gamma", 0.6))
        self.max_participation = float(cm.get("max_participation_rate", 0.10))
        self.max_impact_bps = float(cm.get("max_impact_bps", 150.0))
        self.min_adv_usd = float(cm.get("min_adv_usd", 5_000_000))
        self.financing_rate_annual = float(cm.get("financing_rate_annual", 0.0))

    def _participation_rate(self, order_usd: float, adv_usd: float) -> tuple[float, bool]:
        if adv_usd <= 0:
            return self.max_participation, True
        rate = abs(order_usd) / adv_usd
        capped = rate > self.max_participation
        return min(rate, self.max_participation), capped

    def _impact_bps(self, order_usd: float, adv_usd: float, daily_vol: float, coef: float) -> float:
        if adv_usd <= 0 or daily_vol <= 0:
            return 0.0
        prate, _ = self._participation_rate(order_usd, adv_usd)
        if prate <= 0:
            return 0.0
        impact = coef * abs(daily_vol) * (prate ** self.gamma)
        return min(impact * 10_000, self.max_impact_bps)

    def compute_round_trip_cost(
        self,
        order_usd: float,
        adv_usd: float,
        daily_vol: float,
        horizon_days: int,
        is_short: bool = False,
        position_weight: float = 0.0,
        capital: float = 10_000_000.0,
    ) -> CostBreakdown:
        degraded = False
        degradation_reason = ""

        if adv_usd < self.min_adv_usd:
            degraded = True
            degradation_reason = f"ADV ${adv_usd:,.0f} below minimum ${self.min_adv_usd:,.0f}"
        if daily_vol <= 0:
            degraded = True
            degradation_reason = "Invalid daily volatility"

        # Spread: full round-trip (half-spread each way)
        spread = self.spread_bps

        # Commission: both legs
        commission = self.commission_bps * 2

        # Temporary impact: both entry and exit
        temp_impact = self._impact_bps(order_usd, adv_usd, daily_vol, self.eta) * 2

        # Permanent impact: entry only (information content)
        perm_impact = self._impact_bps(order_usd, adv_usd, daily_vol, self.alpha_coef)

        # Borrow cost: short positions only
        borrow = (self.borrow_bps_annual / 252.0 * horizon_days) if is_short else 0.0

        # Financing
        financing = (self.financing_rate_annual * 10000 / 252.0 * horizon_days * abs(position_weight)) if self.financing_rate_annual > 0 else 0.0

        total = spread + commission + temp_impact + perm_impact + borrow + financing

        prate, _ = self._participation_rate(order_usd, adv_usd)

        return CostBreakdown(
            total_bps=total,
            commission_bps=commission,
            spread_bps=spread,
            temporary_impact_bps=temp_impact,
            permanent_impact_bps=perm_impact,
            borrow_bps=borrow,
            financing_bps=financing,
            participation_rate=prate,
            adv_usd=adv_usd,
            daily_vol=daily_vol,
            degraded_quality=degraded,
            degradation_reason=degradation_reason,
        )


def build_cost_model(config: dict[str, Any]) -> CostModel:
    """Factory: build cost model from config."""
    mode = config.get("cost_model", {}).get("mode", "sqrt_impact")
    if mode == "simple_bps":
        return SimpleBpsCostModel(config)
    return SqrtImpactCostModel(config)


def build_cost_model_with_overrides(
    base_config: dict[str, Any],
    overrides: dict[str, Any],
) -> CostModel:
    """Build cost model with scenario-specific overrides."""
    merged = dict(base_config)
    cm = dict(merged.get("cost_model", {}))
    cm.update(overrides)
    merged["cost_model"] = cm
    return build_cost_model(merged)


# ---------------------------------------------------------------------------
# Viability result
# ---------------------------------------------------------------------------

@dataclass
class ViabilityResult:
    """Full cost viability assessment for one candidate."""
    candidate_id: str
    feature: str
    family: str
    sleeve: str = ""
    horizon: int = 0
    regime: str = ""

    # Input metrics
    ic: float = 0.0
    icir: float = 0.0
    t_stat: float = 0.0
    halflife: float = 0.0
    sigma_annual: float = 0.20
    expected_turnover: float = 0.0
    adv_usd: float = 50_000_000.0
    daily_vol: float = 0.02
    is_short: bool = False
    position_weight: float = 0.0
    capital: float = 10_000_000.0
    n_dates: int = 0
    avg_breadth: int = 0
    bh_significant: bool = False
    bhy_significant: bool = False

    # Decay diagnostics
    ic_decay_curve: dict[int, float] = field(default_factory=dict)
    persistence_survival: float = 1.0

    # Computed outputs
    expected_alpha_bps: float = 0.0
    expected_cost_bps: float = 0.0
    cost_breakdown: Optional[CostBreakdown] = None
    decay_adjusted_alpha_bps: float = 0.0
    net_expected_alpha_bps: float = 0.0
    alpha_cost_ratio: float = 0.0
    turnover_adjusted_ic: float = 0.0
    capacity_score: float = 0.0

    # Liquidity diagnostics
    adv_bucket: str = ""
    spread_bucket: str = ""
    vol_bucket: str = ""
    pct_adv_traded: float = 0.0
    liquidity_adjusted_ic: float = 0.0
    spread_weighted_ic: float = 0.0
    adv_weighted_ic: float = 0.0

    # Classification
    cost_status: CostStatus = CostStatus.COST_DOMINATED
    rejection_reason: str = ""

    # Metadata
    cost_data_coverage: float = 1.0
    degraded_quality: bool = False
    degradation_reason: str = ""
    computed_at: str = ""


# ---------------------------------------------------------------------------
# Cost Viability Engine
# ---------------------------------------------------------------------------

class CostViabilityEngine:
    """
    Institutional cost viability engine.

    Computes for each candidate:
    1. Expected gross alpha (bps)
    2. Expected cost (bps) — via pluggable cost model
    3. Alpha/cost ratio
    4. Turnover-adjusted IC
    5. Decay-adjusted alpha
    6. Net expected alpha
    7. Capacity score
    8. Liquidity diagnostics
    9. Cost domination classification
    """

    def __init__(self, config: dict[str, Any] | None = None, config_path: str | Path | None = None) -> None:
        if config is None:
            config = load_cost_config(config_path)
        self.config = config
        self.cost_model = build_cost_model(config)
        self.classification = config.get("classification", {})
        self.alpha_est = config.get("alpha_estimation", {})
        self.promotion_gates = config.get("promotion_gates", {})

    # ------------------------------------------------------------------
    # 1. Expected gross alpha
    # ------------------------------------------------------------------
    @staticmethod
    def compute_gross_alpha_bps(ic: float, sigma_annual: float, horizon: int) -> float:
        """
        Expected gross alpha in bps.

        Formula: |IC| × sigma_annual × sqrt(h / 252) × 10000

        This converts the rank IC into an expected return advantage,
        scaled by the volatility of the forward return and the horizon.
        """
        if np.isnan(ic) or ic == 0 or sigma_annual <= 0 or horizon <= 0:
            return 0.0
        return abs(ic) * sigma_annual * math.sqrt(horizon / 252.0) * 10000

    # ------------------------------------------------------------------
    # 2. Expected cost
    # ------------------------------------------------------------------
    def compute_expected_cost(
        self,
        ic: float,
        horizon: int,
        adv_usd: float,
        daily_vol: float,
        is_short: bool = False,
        position_weight: float = 0.0,
        capital: float = 10_000_000.0,
        expected_turnover: float = 0.0,
    ) -> CostBreakdown:
        """
        Expected round-trip cost in bps via the configured cost model.

        Order size is derived from turnover × capital (the dollar amount
        that would be traded on a typical rebalance).
        """
        order_usd = expected_turnover * capital if expected_turnover > 0 else 0.0
        if order_usd <= 0:
            # Fallback: assume a representative position
            order_usd = abs(position_weight) * capital if position_weight != 0 else capital * 0.02

        return self.cost_model.compute_round_trip_cost(
            order_usd=order_usd,
            adv_usd=adv_usd,
            daily_vol=daily_vol,
            horizon_days=horizon,
            is_short=is_short,
            position_weight=position_weight,
            capital=capital,
        )

    # ------------------------------------------------------------------
    # 3. Alpha/cost ratio
    # ------------------------------------------------------------------
    @staticmethod
    def compute_alpha_cost_ratio(expected_alpha_bps: float, expected_cost_bps: float) -> float:
        if expected_cost_bps <= 0:
            return float("inf") if expected_alpha_bps > 0 else 0.0
        return expected_alpha_bps / expected_cost_bps

    # ------------------------------------------------------------------
    # 4. Turnover-adjusted IC
    # ------------------------------------------------------------------
    @staticmethod
    def compute_turnover_adjusted_ic(ic: float, expected_turnover: float) -> float:
        """IC per unit of turnover. Higher is better (more alpha per trade)."""
        if expected_turnover <= 0:
            return abs(ic) if ic != 0 else 0.0
        return abs(ic) / expected_turnover

    # ------------------------------------------------------------------
    # 5. Decay-adjusted alpha
    # ------------------------------------------------------------------
    def compute_decay_adjusted_alpha(
        self,
        expected_alpha_bps: float,
        halflife: float,
        horizon: int,
        ic_decay_curve: dict[int, float] | None = None,
    ) -> float:
        """
        Alpha adjusted for signal decay over the holding period.

        Uses persistence survival: 2^(-horizon / halflife)
        If an IC decay curve is available, uses the IC ratio at the horizon.
        """
        if expected_alpha_bps <= 0:
            return 0.0

        if ic_decay_curve and len(ic_decay_curve) > 0:
            # Use empirical decay curve if available
            ic_0 = ic_decay_curve.get(1, 0)
            ic_h = ic_decay_curve.get(horizon, 0)
            if ic_0 != 0:
                survival = abs(ic_h) / abs(ic_0)
                return expected_alpha_bps * max(0.0, min(1.0, survival))

        # Fallback: exponential decay from halflife
        if halflife > 0:
            survival = 2.0 ** (-horizon / halflife)
            return expected_alpha_bps * survival

        return 0.0

    def compute_persistence_survival(
        self,
        halflife: float,
        horizon: int,
    ) -> float:
        """Fraction of signal remaining at horizon."""
        if halflife <= 0:
            return 0.0
        return 2.0 ** (-horizon / halflife)

    # ------------------------------------------------------------------
    # 6. Net expected alpha
    # ------------------------------------------------------------------
    @staticmethod
    def compute_net_expected_alpha(
        decay_adjusted_alpha_bps: float,
        expected_cost_bps: float,
    ) -> float:
        return decay_adjusted_alpha_bps - expected_cost_bps

    # ------------------------------------------------------------------
    # 7. Capacity score
    # ------------------------------------------------------------------
    @staticmethod
    def compute_capacity_score(
        expected_alpha_bps: float,
        tradable_dollar_volume: float,
        expected_turnover: float,
    ) -> float:
        """
        Capacity score: alpha × tradable_volume / turnover.

        Higher = more alpha available per unit of trading.
        A score of 1.0 means the alpha is just sufficient to cover
        the trading required to capture it.
        """
        if expected_turnover <= 0 or tradable_dollar_volume <= 0:
            return 0.0
        return expected_alpha_bps * tradable_dollar_volume / (expected_turnover * 10_000_000)

    # ------------------------------------------------------------------
    # 8. Liquidity diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def compute_liquidity_diagnostics(
        adv_usd: float,
        daily_vol: float,
        ic: float,
        spread_bps: float = 1.0,
        order_usd: float = 0,
    ) -> dict[str, Any]:
        """Compute liquidity bucket and adjusted IC metrics."""
        # ADV bucket
        if adv_usd >= 100_000_000:
            adv_bucket = "mega"
        elif adv_usd >= 50_000_000:
            adv_bucket = "large"
        elif adv_usd >= 10_000_000:
            adv_bucket = "mid"
        elif adv_usd >= 1_000_000:
            adv_bucket = "small"
        else:
            adv_bucket = "micro"

        # Spread bucket
        if spread_bps <= 1.0:
            spread_bucket = "tight"
        elif spread_bps <= 5.0:
            spread_bucket = "normal"
        elif spread_bps <= 15.0:
            spread_bucket = "wide"
        else:
            spread_bucket = "very_wide"

        # Vol bucket
        ann_vol = daily_vol * math.sqrt(252)
        if ann_vol <= 0.15:
            vol_bucket = "low"
        elif ann_vol <= 0.30:
            vol_bucket = "normal"
        elif ann_vol <= 0.50:
            vol_bucket = "high"
        else:
            vol_bucket = "extreme"

        # %ADV traded
        pct_adv = (order_usd / adv_usd * 100) if adv_usd > 0 and order_usd > 0 else 0.0

        # Liquidity-adjusted IC: weight IC by ADV (higher liquidity = more weight)
        # This represents the IC of the tradable portion
        adv_weight = min(1.0, adv_usd / 50_000_000)  # Normalize to $50M
        liquidity_adjusted_ic = ic * adv_weight

        # Spread-weighted IC: reduce IC by spread cost
        spread_cost_frac = spread_bps / 10000
        spread_weighted_ic = ic * max(0, 1 - spread_cost_frac * 100)

        # ADV-weighted IC
        adv_weighted_ic = ic * math.log10(max(adv_usd, 1)) / math.log10(50_000_000)

        return {
            "adv_bucket": adv_bucket,
            "spread_bucket": spread_bucket,
            "vol_bucket": vol_bucket,
            "pct_adv_traded": pct_adv,
            "liquidity_adjusted_ic": liquidity_adjusted_ic,
            "spread_weighted_ic": spread_weighted_ic,
            "adv_weighted_ic": adv_weighted_ic,
        }

    # ------------------------------------------------------------------
    # 9. Cost domination classification
    # ------------------------------------------------------------------
    def classify_cost_status(
        self,
        ic: float,
        expected_alpha_bps: float,
        expected_cost_bps: float,
        net_expected_alpha_bps: float,
        alpha_cost_ratio: float,
        expected_turnover: float,
        cost_breakdown: CostBreakdown,
        halflife: float,
        horizon: int,
        adv_usd: float,
        capacity_score: float,
        cost_data_coverage: float,
        pit_data_available: bool = True,
    ) -> tuple[CostStatus, str]:
        """
        Classify candidate into exactly ONE cost status.
        Evaluated in priority order (first match wins).
        """
        cls = self.classification

        # 0. Missing PIT diagnostics (highest priority — cannot evaluate without data)
        if not pit_data_available:
            return CostStatus.MISSING_PIT_DIAGNOSTICS, "missing_point_in_time_data"

        # 1. Unstable cost estimate
        if cost_data_coverage < cls.get("min_cost_data_coverage", 0.80):
            return CostStatus.UNSTABLE_COST_ESTIMATE, "insufficient_cost_data"

        # 2. Alpha too weak
        if abs(ic) < cls.get("min_ic_absolute", 0.005):
            return CostStatus.ALPHA_TOO_WEAK, "ic_below_minimum"
        if expected_alpha_bps < cls.get("min_gross_alpha_bps", 3.0):
            return CostStatus.ALPHA_TOO_WEAK, "gross_alpha_below_minimum"

        # 3. Liquidity insufficient
        if adv_usd < cls.get("min_adv_usd_viable", 10_000_000):
            return CostStatus.LIQUIDITY_INSUFFICIENT, "adv_below_minimum"
        if cost_breakdown.participation_rate > cls.get("max_participation_for_viable", 0.05):
            return CostStatus.LIQUIDITY_INSUFFICIENT, "participation_rate_too_high"

        # 4. Turnover dominated
        if expected_turnover > cls.get("max_expected_turnover", 0.80):
            return CostStatus.TURNOVER_DOMINATED, "turnover_exceeds_maximum"
        if expected_turnover > 0 and abs(ic) > 0:
            turnover_to_ic = expected_turnover / abs(ic)
            if turnover_to_ic > cls.get("max_turnover_to_ic_ratio", 50.0):
                return CostStatus.TURNOVER_DOMINATED, "turnover_to_ic_ratio_too_high"

        # 5. Impact dominated
        if expected_cost_bps > 0:
            impact_frac = (cost_breakdown.temporary_impact_bps + cost_breakdown.permanent_impact_bps) / expected_cost_bps
            if impact_frac > cls.get("max_impact_fraction_of_cost", 0.75):
                return CostStatus.IMPACT_DOMINATED, "market_impact_dominates_cost"

        # 6. Borrow dominated
        if expected_cost_bps > 0 and cost_breakdown.borrow_bps > 0:
            borrow_frac = cost_breakdown.borrow_bps / expected_cost_bps
            if borrow_frac > cls.get("max_borrow_fraction_of_cost", 0.50):
                return CostStatus.BORROW_DOMINATED, "borrow_cost_dominates"

        # 7. Capacity insufficient
        if capacity_score < cls.get("min_capacity_score", 1.0):
            return CostStatus.CAPACITY_INSUFFICIENT, "capacity_score_below_minimum"

        # 8. Cost dominated (alpha/cost ratio below marginal threshold)
        if alpha_cost_ratio < cls.get("min_alpha_cost_ratio_marginal", 1.0):
            return CostStatus.COST_DOMINATED, "alpha_below_cost"

        # 9. Marginal (alpha/cost between marginal and viable)
        if alpha_cost_ratio < cls.get("min_alpha_cost_ratio_viable", 1.5):
            return CostStatus.MARGINAL, "alpha_cost_ratio_below_viable_threshold"

        # 10. Net alpha check
        if net_expected_alpha_bps < cls.get("min_net_alpha_bps", 0.0):
            return CostStatus.COST_DOMINATED, "net_alpha_negative"

        # 11. Cost viable
        return CostStatus.COST_VIABLE, ""

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        candidate_id: str,
        feature: str,
        family: str,
        ic: float,
        horizon: int,
        sigma_annual: float,
        halflife: float = 0.0,
        expected_turnover: float = 0.0,
        adv_usd: float = 0.0,
        daily_vol: float = 0.0,
        icir: float = 0.0,
        t_stat: float = 0.0,
        ic_decay_curve: dict[int, float] | None = None,
        is_short: bool = False,
        position_weight: float = 0.0,
        capital: float = 10_000_000.0,
        sleeve: str = "",
        regime: str = "",
        n_dates: int = 0,
        avg_breadth: int = 0,
        bh_significant: bool = False,
        bhy_significant: bool = False,
        cost_data_coverage: float = 1.0,
        spread_bps: float = 1.0,
        pit_data_available: bool = True,
    ) -> ViabilityResult:
        """
        Full cost viability evaluation for one candidate.

        All inputs are point-in-time observable or empirically measured.
        No forward-looking assumptions.
        """
        # Defaults from config
        alpha_est = self.alpha_est
        if sigma_annual <= 0:
            sigma_annual = alpha_est.get("default_sigma_annual", 0.20)
        if adv_usd <= 0:
            adv_usd = self.config.get("cost_model", {}).get("default_adv_usd", 50_000_000)
        if daily_vol <= 0:
            daily_vol = self.config.get("cost_model", {}).get("default_daily_vol", 0.02)

        # 1. Gross alpha
        expected_alpha_bps = self.compute_gross_alpha_bps(ic, sigma_annual, horizon)

        # 2. Expected cost
        cost_breakdown = self.compute_expected_cost(
            ic=ic,
            horizon=horizon,
            adv_usd=adv_usd,
            daily_vol=daily_vol,
            is_short=is_short,
            position_weight=position_weight,
            capital=capital,
            expected_turnover=expected_turnover,
        )
        expected_cost_bps = cost_breakdown.total_bps

        # 3. Alpha/cost ratio
        alpha_cost_ratio = self.compute_alpha_cost_ratio(expected_alpha_bps, expected_cost_bps)

        # 4. Turnover-adjusted IC
        turnover_adjusted_ic = self.compute_turnover_adjusted_ic(ic, expected_turnover)

        # 5. Decay-adjusted alpha
        persistence_survival = self.compute_persistence_survival(halflife, horizon)
        decay_adjusted_alpha_bps = self.compute_decay_adjusted_alpha(
            expected_alpha_bps, halflife, horizon, ic_decay_curve
        )

        # 6. Net expected alpha
        net_expected_alpha_bps = self.compute_net_expected_alpha(
            decay_adjusted_alpha_bps, expected_cost_bps
        )

        # 7. Capacity score
        capacity_score = self.compute_capacity_score(
            expected_alpha_bps, adv_usd, expected_turnover
        )

        # 8. Liquidity diagnostics
        order_usd = expected_turnover * capital if expected_turnover > 0 else capital * 0.02
        liq_diag = self.compute_liquidity_diagnostics(
            adv_usd, daily_vol, ic, spread_bps, order_usd
        )

        # 9. Classification
        cost_status, rejection_reason = self.classify_cost_status(
            ic=ic,
            expected_alpha_bps=expected_alpha_bps,
            expected_cost_bps=expected_cost_bps,
            net_expected_alpha_bps=net_expected_alpha_bps,
            alpha_cost_ratio=alpha_cost_ratio,
            expected_turnover=expected_turnover,
            cost_breakdown=cost_breakdown,
            halflife=halflife,
            horizon=horizon,
            adv_usd=adv_usd,
            capacity_score=capacity_score,
            cost_data_coverage=cost_data_coverage,
            pit_data_available=pit_data_available,
        )

        return ViabilityResult(
            candidate_id=candidate_id,
            feature=feature,
            family=family,
            sleeve=sleeve,
            horizon=horizon,
            regime=regime,
            ic=ic,
            icir=icir,
            t_stat=t_stat,
            halflife=halflife,
            sigma_annual=sigma_annual,
            expected_turnover=expected_turnover,
            adv_usd=adv_usd,
            daily_vol=daily_vol,
            is_short=is_short,
            position_weight=position_weight,
            capital=capital,
            n_dates=n_dates,
            avg_breadth=avg_breadth,
            bh_significant=bh_significant,
            bhy_significant=bhy_significant,
            ic_decay_curve=ic_decay_curve or {},
            persistence_survival=persistence_survival,
            expected_alpha_bps=expected_alpha_bps,
            expected_cost_bps=expected_cost_bps,
            cost_breakdown=cost_breakdown,
            decay_adjusted_alpha_bps=decay_adjusted_alpha_bps,
            net_expected_alpha_bps=net_expected_alpha_bps,
            alpha_cost_ratio=alpha_cost_ratio,
            turnover_adjusted_ic=turnover_adjusted_ic,
            capacity_score=capacity_score,
            adv_bucket=liq_diag["adv_bucket"],
            spread_bucket=liq_diag["spread_bucket"],
            vol_bucket=liq_diag["vol_bucket"],
            pct_adv_traded=liq_diag["pct_adv_traded"],
            liquidity_adjusted_ic=liq_diag["liquidity_adjusted_ic"],
            spread_weighted_ic=liq_diag["spread_weighted_ic"],
            adv_weighted_ic=liq_diag["adv_weighted_ic"],
            cost_status=cost_status,
            rejection_reason=rejection_reason,
            cost_data_coverage=cost_data_coverage,
            degraded_quality=cost_breakdown.degraded_quality,
            degradation_reason=cost_breakdown.degradation_reason,
            computed_at=datetime.now().isoformat(),
        )

    # ------------------------------------------------------------------
    # Stress testing
    # ------------------------------------------------------------------
    def run_stress_test(
        self,
        candidate_id: str,
        feature: str,
        family: str,
        ic: float,
        horizon: int,
        sigma_annual: float,
        halflife: float = 0.0,
        expected_turnover: float = 0.0,
        adv_usd: float = 0.0,
        daily_vol: float = 0.0,
        ic_decay_curve: dict[int, float] | None = None,
        is_short: bool = False,
        position_weight: float = 0.0,
        capital: float = 10_000_000.0,
    ) -> dict[str, dict[str, Any]]:
        """
        Evaluate candidate under all stress scenarios.

        Returns dict of scenario_name → {net_alpha_bps, survives, cost_breakdown}
        """
        scenarios = self.config.get("stress_scenarios", {})
        results = {}

        for name, scenario in scenarios.items():
            overrides = scenario.get("overrides", {})
            model = build_cost_model_with_overrides(self.config, overrides)

            # Compute cost under this scenario
            order_usd = expected_turnover * capital if expected_turnover > 0 else capital * 0.02
            if adv_usd <= 0:
                adv_usd = self.config.get("cost_model", {}).get("default_adv_usd", 50_000_000)
            if daily_vol <= 0:
                daily_vol = self.config.get("cost_model", {}).get("default_daily_vol", 0.02)

            cost_bd = model.compute_round_trip_cost(
                order_usd=order_usd,
                adv_usd=adv_usd,
                daily_vol=daily_vol,
                horizon_days=horizon,
                is_short=is_short,
                position_weight=position_weight,
                capital=capital,
            )

            expected_alpha_bps = self.compute_gross_alpha_bps(ic, sigma_annual, horizon)
            decay_adj = self.compute_decay_adjusted_alpha(
                expected_alpha_bps, halflife, horizon, ic_decay_curve
            )
            net_alpha = decay_adj - cost_bd.total_bps

            results[name] = {
                "scenario_label": scenario.get("label", name),
                "expected_cost_bps": cost_bd.total_bps,
                "decay_adjusted_alpha_bps": decay_adj,
                "net_alpha_bps": net_alpha,
                "survives": net_alpha > 0,
                "commission_bps": cost_bd.commission_bps,
                "spread_bps": cost_bd.spread_bps,
                "temporary_impact_bps": cost_bd.temporary_impact_bps,
                "permanent_impact_bps": cost_bd.permanent_impact_bps,
                "borrow_bps": cost_bd.borrow_bps,
            }

        return results

    # ------------------------------------------------------------------
    # Promotion gate check
    # ------------------------------------------------------------------
    def check_promotion_gates(self, result: ViabilityResult) -> tuple[bool, list[str]]:
        """
        Check if a candidate passes all promotion gates.
        Returns (passes, list_of_failures).
        """
        gates = self.promotion_gates
        failures = []

        if result.alpha_cost_ratio < gates.get("min_alpha_cost_ratio", 1.5):
            failures.append(f"alpha_cost_ratio {result.alpha_cost_ratio:.2f} < {gates['min_alpha_cost_ratio']}")

        if result.net_expected_alpha_bps < gates.get("min_net_expected_alpha_bps", 0.0):
            failures.append(f"net_alpha {result.net_expected_alpha_bps:.1f}bps < {gates['min_net_expected_alpha_bps']}bps")

        if result.halflife < gates.get("min_halflife_days", 5.0):
            failures.append(f"halflife {result.halflife:.1f}d < {gates['min_halflife_days']}d")

        if result.expected_turnover > gates.get("max_turnover", 0.80):
            failures.append(f"turnover {result.expected_turnover:.3f} > {gates['max_turnover']}")

        if result.capacity_score < gates.get("min_capacity_score", 1.0):
            failures.append(f"capacity_score {result.capacity_score:.2f} < {gates['min_capacity_score']}")

        if gates.get("min_bh_significant", False) and not result.bh_significant:
            failures.append("not BH significant")

        if result.avg_breadth > 0 and result.avg_breadth < gates.get("min_breadth", 50):
            failures.append(f"breadth {result.avg_breadth} < {gates['min_breadth']}")

        if result.n_dates > 0 and result.n_dates < gates.get("min_dates", 252):
            failures.append(f"n_dates {result.n_dates} < {gates['min_dates']}")

        return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Alpha-to-Trade Decision Layer
# ---------------------------------------------------------------------------

@dataclass
class TradeDecision:
    """Result of alpha-to-trade decision."""
    candidate_id: str
    trade_approved: bool
    incremental_alpha_bps: float
    incremental_cost_bps: float
    alpha_to_trade_ratio: float
    reason_for_rejection: str = ""
    rejection_code: str = ""


class AlphaToTradeDecision:
    """
    Explicit trade approval layer.

    Rules:
    - Do not trade merely because target weights changed.
    - Trade only when expected incremental alpha exceeds expected incremental cost
      by a configurable margin of safety.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = load_cost_config()
        self.config = config.get("alpha_to_trade", {})
        self.enabled = self.config.get("enabled", True)
        self.min_ratio = self.config.get("min_alpha_to_trade_ratio", 1.5)
        self.min_alpha_bps = self.config.get("min_incremental_alpha_bps", 1.0)
        self.max_turnover = self.config.get("max_single_rebalance_turnover", 0.20)

    def decide(
        self,
        candidate_id: str,
        incremental_alpha_bps: float,
        incremental_cost_bps: float,
        incremental_turnover: float = 0.0,
        impact_bps: float = 0.0,
        adv_usd: float = 50_000_000.0,
        liquidity_sufficient: bool = True,
        borrow_cost_bps: float = 0.0,
        signal_halflife: float = 0.0,
        horizon: int = 5,
        breadth: int = 100,
    ) -> TradeDecision:
        """
        Decide whether to approve a trade.

        All inputs must be point-in-time observable.
        """
        if not self.enabled:
            return TradeDecision(
                candidate_id=candidate_id,
                trade_approved=True,
                incremental_alpha_bps=incremental_alpha_bps,
                incremental_cost_bps=incremental_cost_bps,
                alpha_to_trade_ratio=float("inf") if incremental_cost_bps <= 0 else incremental_alpha_bps / incremental_cost_bps,
            )

        if incremental_cost_bps <= 0:
            ratio = float("inf") if incremental_alpha_bps > 0 else 0.0
        else:
            ratio = incremental_alpha_bps / incremental_cost_bps

        # Check conditions in priority order
        if incremental_alpha_bps < self.min_alpha_bps:
            return TradeDecision(
                candidate_id=candidate_id,
                trade_approved=False,
                incremental_alpha_bps=incremental_alpha_bps,
                incremental_cost_bps=incremental_cost_bps,
                alpha_to_trade_ratio=ratio,
                reason_for_rejection=f"Incremental alpha {incremental_alpha_bps:.1f}bps below minimum {self.min_alpha_bps}bps",
                rejection_code="alpha_below_cost",
            )

        if incremental_alpha_bps < incremental_cost_bps:
            return TradeDecision(
                candidate_id=candidate_id,
                trade_approved=False,
                incremental_alpha_bps=incremental_alpha_bps,
                incremental_cost_bps=incremental_cost_bps,
                alpha_to_trade_ratio=ratio,
                reason_for_rejection=f"Alpha {incremental_alpha_bps:.1f}bps below cost {incremental_cost_bps:.1f}bps",
                rejection_code="alpha_below_cost",
            )

        if ratio < self.min_ratio:
            return TradeDecision(
                candidate_id=candidate_id,
                trade_approved=False,
                incremental_alpha_bps=incremental_alpha_bps,
                incremental_cost_bps=incremental_cost_bps,
                alpha_to_trade_ratio=ratio,
                reason_for_rejection=f"Alpha/cost ratio {ratio:.2f} below margin of safety {self.min_ratio}",
                rejection_code="alpha_below_margin",
            )

        if incremental_turnover > self.max_turnover:
            return TradeDecision(
                candidate_id=candidate_id,
                trade_approved=False,
                incremental_alpha_bps=incremental_alpha_bps,
                incremental_cost_bps=incremental_cost_bps,
                alpha_to_trade_ratio=ratio,
                reason_for_rejection=f"Turnover {incremental_turnover:.3f} exceeds max {self.max_turnover}",
                rejection_code="turnover_too_high",
            )

        if impact_bps > incremental_alpha_bps * 0.5:
            return TradeDecision(
                candidate_id=candidate_id,
                trade_approved=False,
                incremental_alpha_bps=incremental_alpha_bps,
                incremental_cost_bps=incremental_cost_bps,
                alpha_to_trade_ratio=ratio,
                reason_for_rejection=f"Market impact {impact_bps:.1f}bps too high relative to alpha",
                rejection_code="impact_too_high",
            )

        if not liquidity_sufficient:
            return TradeDecision(
                candidate_id=candidate_id,
                trade_approved=False,
                incremental_alpha_bps=incremental_alpha_bps,
                incremental_cost_bps=incremental_cost_bps,
                alpha_to_trade_ratio=ratio,
                reason_for_rejection="Insufficient liquidity for trade size",
                rejection_code="liquidity_too_low",
            )

        if borrow_cost_bps > incremental_alpha_bps * 0.5:
            return TradeDecision(
                candidate_id=candidate_id,
                trade_approved=False,
                incremental_alpha_bps=incremental_alpha_bps,
                incremental_cost_bps=incremental_cost_bps,
                alpha_to_trade_ratio=ratio,
                reason_for_rejection=f"Borrow cost {borrow_cost_bps:.1f}bps too high",
                rejection_code="borrow_cost_too_high",
            )

        if signal_halflife > 0 and signal_halflife < horizon * 0.5:
            return TradeDecision(
                candidate_id=candidate_id,
                trade_approved=False,
                incremental_alpha_bps=incremental_alpha_bps,
                incremental_cost_bps=incremental_cost_bps,
                alpha_to_trade_ratio=ratio,
                reason_for_rejection=f"Signal halflife {signal_halflife:.1f}d too short for {horizon}d horizon",
                rejection_code="signal_decay_too_fast",
            )

        if breadth < 10:
            return TradeDecision(
                candidate_id=candidate_id,
                trade_approved=False,
                incremental_alpha_bps=incremental_alpha_bps,
                incremental_cost_bps=incremental_cost_bps,
                alpha_to_trade_ratio=ratio,
                reason_for_rejection=f"Breadth {breadth} too low",
                rejection_code="insufficient_breadth",
            )

        return TradeDecision(
            candidate_id=candidate_id,
            trade_approved=True,
            incremental_alpha_bps=incremental_alpha_bps,
            incremental_cost_bps=incremental_cost_bps,
            alpha_to_trade_ratio=ratio,
        )


# ---------------------------------------------------------------------------
# No-Trade Band Engine
# ---------------------------------------------------------------------------

@dataclass
class BandResult:
    """Result of no-trade band application."""
    candidate_id: str
    current_weight: float
    target_weight: float
    band_lower: float
    band_upper: float
    trade_required: bool
    adjusted_weight: float
    band_width: float
    gross_turnover_before: float
    gross_turnover_after: float
    turnover_reduction: float
    alpha_lost_bps: float
    cost_saved_bps: float
    net_trade_benefit: float


class NoTradeBandEngine:
    """
    Adaptive no-trade band logic.

    If current weight is inside the band around target, do not trade.
    If outside, trade only to the nearest boundary.

    Band width is adaptive:
    - Widens when costs are high, vol is high, liquidity is low
    - Narrows when signal strength is high, costs are low
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = load_cost_config()
        self.config = config.get("no_trade_bands", {})
        self.enabled = self.config.get("enabled", True)
        self.base_width = self.config.get("base_band_width", 0.010)
        self.cost_scale = self.config.get("cost_scaling_factor", 2.0)
        self.vol_scale = self.config.get("vol_scaling_factor", 0.5)
        self.liq_scale = self.config.get("liquidity_scaling_factor", 0.3)
        self.signal_scale = self.config.get("signal_scaling_factor", -0.01)
        self.min_width = self.config.get("min_band_width", 0.002)
        self.max_width = self.config.get("max_band_width", 0.050)
        self.max_drift = self.config.get("max_total_drift", 0.05)

    def compute_band_width(
        self,
        expected_cost_bps: float,
        daily_vol: float,
        adv_usd: float,
        abs_ic: float,
    ) -> float:
        """
        Compute adaptive band width.

        Band widens with cost, vol, low liquidity.
        Band narrows with signal strength.
        """
        if not self.enabled:
            return self.base_width

        width = self.base_width

        # Cost scaling: higher cost → wider band (don't trade for small alpha)
        width += self.cost_scale * (expected_cost_bps / 10000)

        # Vol scaling: higher vol → wider band (more noise)
        width += self.vol_scale * daily_vol

        # Liquidity scaling: lower ADV → wider band (expensive to trade)
        liq_factor = 1.0 / max(adv_usd / 50_000_000, 0.1)
        width += self.liq_scale * liq_factor * 0.01

        # Signal scaling: stronger signal → narrower band (more conviction)
        width += self.signal_scale * abs_ic

        # Clamp
        return max(self.min_width, min(self.max_width, width))

    def apply(
        self,
        candidate_id: str,
        current_weight: float,
        target_weight: float,
        expected_cost_bps: float,
        daily_vol: float,
        adv_usd: float,
        abs_ic: float,
        expected_alpha_bps: float,
    ) -> BandResult:
        """
        Apply no-trade band to a single position.

        Returns band result with turnover and alpha/cost diagnostics.
        """
        if not self.enabled:
            return BandResult(
                candidate_id=candidate_id,
                current_weight=current_weight,
                target_weight=target_weight,
                band_lower=target_weight,
                band_upper=target_weight,
                trade_required=abs(current_weight - target_weight) > 0,
                adjusted_weight=target_weight,
                band_width=0.0,
                gross_turnover_before=abs(target_weight - current_weight),
                gross_turnover_after=abs(target_weight - current_weight),
                turnover_reduction=0.0,
                alpha_lost_bps=0.0,
                cost_saved_bps=0.0,
                net_trade_benefit=0.0,
            )

        band_width = self.compute_band_width(expected_cost_bps, daily_vol, adv_usd, abs_ic)
        band_lower = target_weight - band_width
        band_upper = target_weight + band_width

        # Check if inside band
        if band_lower <= current_weight <= band_upper:
            # No trade
            gross_before = abs(target_weight - current_weight)
            return BandResult(
                candidate_id=candidate_id,
                current_weight=current_weight,
                target_weight=target_weight,
                band_lower=band_lower,
                band_upper=band_upper,
                trade_required=False,
                adjusted_weight=current_weight,
                band_width=band_width,
                gross_turnover_before=gross_before,
                gross_turnover_after=0.0,
                turnover_reduction=gross_before,
                alpha_lost_bps=0.0,  # Signal not strong enough to justify cost
                cost_saved_bps=expected_cost_bps * gross_before,
                net_trade_benefit=expected_cost_bps * gross_before,
            )

        # Outside band: trade to nearest boundary
        if current_weight < band_lower:
            adjusted = band_lower
        else:
            adjusted = band_upper

        gross_before = abs(target_weight - current_weight)
        gross_after = abs(adjusted - current_weight)
        reduction = gross_before - gross_after

        # Alpha lost: the alpha from the weight change we didn't execute
        alpha_lost = expected_alpha_bps * reduction

        # Cost saved: the cost we avoided by not trading the full amount
        cost_saved = expected_cost_bps * reduction

        return BandResult(
            candidate_id=candidate_id,
            current_weight=current_weight,
            target_weight=target_weight,
            band_lower=band_lower,
            band_upper=band_upper,
            trade_required=True,
            adjusted_weight=adjusted,
            band_width=band_width,
            gross_turnover_before=gross_before,
            gross_turnover_after=gross_after,
            turnover_reduction=reduction,
            alpha_lost_bps=alpha_lost,
            cost_saved_bps=cost_saved,
            net_trade_benefit=cost_saved - alpha_lost,
        )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_scorecard(results: list[ViabilityResult], output_path: str | Path | None = None) -> str:
    """Generate cost_viability_scorecard.csv."""
    if not results:
        return ""

    columns = [
        "candidate_id", "feature", "family", "sleeve", "horizon", "regime",
        "ic", "icir", "t_stat", "halflife",
        "expected_turnover", "expected_alpha_bps", "expected_cost_bps",
        "decay_adjusted_alpha_bps", "net_expected_alpha_bps",
        "alpha_cost_ratio", "capacity_score", "cost_status", "rejection_reason",
    ]

    lines = [",".join(columns)]
    for r in results:
        row = [
            r.candidate_id, r.feature, r.family, r.sleeve, str(r.horizon), r.regime,
            f"{r.ic:.6f}", f"{r.icir:.4f}", f"{r.t_stat:.4f}", f"{r.halflife:.1f}",
            f"{r.expected_turnover:.6f}", f"{r.expected_alpha_bps:.2f}", f"{r.expected_cost_bps:.2f}",
            f"{r.decay_adjusted_alpha_bps:.2f}", f"{r.net_expected_alpha_bps:.2f}",
            f"{r.alpha_cost_ratio:.4f}", f"{r.capacity_score:.2f}",
            r.cost_status.value, r.rejection_reason,
        ]
        lines.append(",".join(row))

    csv_content = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(csv_content)

    return csv_content


def generate_stress_test_report(
    stress_results: dict[str, dict[str, dict[str, Any]]],
    output_path: str | Path | None = None,
) -> str:
    """Generate cost_stress_test.csv."""
    columns = [
        "candidate_id", "scenario", "scenario_label",
        "base_net_alpha", "low_cost_net_alpha", "high_cost_net_alpha",
        "stressed_spread_net_alpha", "stressed_impact_net_alpha",
        "stressed_liquidity_net_alpha",
        "survives_base", "survives_stress",
    ]

    lines = [",".join(columns)]
    for cid, scenarios in stress_results.items():
        row_data = {s: v for s, v in scenarios.items()}

        def get_net(scenario_name: str) -> float:
            return row_data.get(scenario_name, {}).get("net_alpha_bps", 0.0)

        def survives(scenario_name: str) -> str:
            return str(row_data.get(scenario_name, {}).get("survives", False))

        # Per-scenario rows
        for scenario_name, data in scenarios.items():
            row = [
                cid, scenario_name, data.get("scenario_label", scenario_name),
                f"{get_net('base_cost'):.2f}", f"{get_net('low_cost'):.2f}",
                f"{get_net('high_cost'):.2f}", f"{get_net('stressed_spread'):.2f}",
                f"{get_net('stressed_impact'):.2f}", f"{get_net('stressed_liquidity'):.2f}",
                survives("base_cost"),
                str(data.get("survives", False)),
            ]
            lines.append(",".join(row))

    csv_content = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(csv_content)

    return csv_content


def generate_turnover_attribution_report(
    band_results: list[BandResult],
    output_path: str | Path | None = None,
) -> str:
    """Generate turnover_attribution.csv."""
    columns = [
        "candidate_id", "current_weight", "target_weight",
        "band_lower", "band_upper", "trade_required",
        "adjusted_weight", "band_width",
        "gross_turnover_before", "gross_turnover_after",
        "turnover_reduction", "alpha_lost_bps", "cost_saved_bps",
        "net_trade_benefit",
    ]

    lines = [",".join(columns)]
    for br in band_results:
        row = [
            br.candidate_id,
            f"{br.current_weight:.6f}", f"{br.target_weight:.6f}",
            f"{br.band_lower:.6f}", f"{br.band_upper:.6f}",
            str(br.trade_required),
            f"{br.adjusted_weight:.6f}", f"{br.band_width:.6f}",
            f"{br.gross_turnover_before:.6f}", f"{br.gross_turnover_after:.6f}",
            f"{br.turnover_reduction:.6f}",
            f"{br.alpha_lost_bps:.2f}", f"{br.cost_saved_bps:.2f}",
            f"{br.net_trade_benefit:.2f}",
        ]
        lines.append(",".join(row))

    csv_content = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(csv_content)

    return csv_content


def generate_cost_dominated_report(
    results: list[ViabilityResult],
    output_path: str | Path | None = None,
) -> str:
    """Generate cost_dominated_candidates.csv."""
    dominated = [r for r in results if r.cost_status != CostStatus.COST_VIABLE]

    columns = [
        "candidate_id", "feature", "family", "sleeve", "horizon", "regime",
        "ic", "halflife", "expected_alpha_bps", "expected_cost_bps",
        "net_expected_alpha_bps", "alpha_cost_ratio", "cost_status",
        "rejection_reason", "adv_usd", "expected_turnover", "capacity_score",
    ]

    lines = [",".join(columns)]
    for r in dominated:
        row = [
            r.candidate_id, r.feature, r.family, r.sleeve, str(r.horizon), r.regime,
            f"{r.ic:.6f}", f"{r.halflife:.1f}",
            f"{r.expected_alpha_bps:.2f}", f"{r.expected_cost_bps:.2f}",
            f"{r.net_expected_alpha_bps:.2f}", f"{r.alpha_cost_ratio:.4f}",
            r.cost_status.value, r.rejection_reason,
            f"{r.adv_usd:.0f}", f"{r.expected_turnover:.6f}", f"{r.capacity_score:.2f}",
        ]
        lines.append(",".join(row))

    csv_content = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(csv_content)

    return csv_content
