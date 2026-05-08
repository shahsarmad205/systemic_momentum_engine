"""
Horizon Gate — Hardened
========================
Blocks ineligible feature-horizon combinations from entering model training.

Hardening changes:
- HorizonGateConfig for all thresholds (no hardcoded values).
- Two-level evaluation: statistical + production.
- Family diversity enforcement (min families, max concentration).
- Effective signal diversity via eigenvalue participation ratio.
- Detailed rejection diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from model_selection.horizon_eligibility import (
    HorizonEligibilityContract,
    compute_all_eligibility,
    compute_effective_signal_count,
    format_eligibility_report,
)


@dataclass(frozen=True)
class HorizonGateConfig:
    """
    Configuration for HorizonGate thresholds.
    All values are config-driven; no hardcoded thresholds in gate logic.
    """
    # Minimum number of production-eligible features required
    min_production_features: int = 3

    # Minimum number of distinct families required
    min_families: int = 2

    # Maximum fraction of features from a single family
    max_family_concentration: float = 0.6

    # Minimum effective signal count (participation ratio)
    min_effective_signals: float = 1.5

    # Whether to use production or statistical admissibility
    use_production_level: bool = True


@dataclass(frozen=True)
class HorizonGateResult:
    """Result of horizon gate evaluation."""
    horizon: int
    eligible_features: list[str]
    rejected_features: dict[str, str]
    block_horizon: bool
    block_reasons: list[str]
    config: HorizonGateConfig

    # Diagnostics
    n_eligible: int = 0
    n_families: int = 0
    family_concentration: float = 0.0
    effective_signals: float = 0.0
    report: str = ""


class HorizonGate:
    """
    Gate that blocks ineligible feature-horizon combinations.

    Usage:
        config = HorizonGateConfig(min_production_features=3, min_families=2)
        gate = HorizonGate(contracts, config=config)
        result = gate.evaluate(horizon=10)
        if result.block_horizon:
            print(f"Blocking h{result.horizon}d: {result.block_reasons}")
    """

    def __init__(
        self,
        contracts: dict[str, HorizonEligibilityContract],
        config: Optional[HorizonGateConfig] = None,
    ):
        self.contracts = contracts
        self.config = config or HorizonGateConfig()

    def _get_admissible_horizons(self, contract: HorizonEligibilityContract) -> list[int]:
        """Get the appropriate admissible horizons based on config level."""
        if self.config.use_production_level:
            return contract.production_admissible_horizons
        return contract.statistically_admissible_horizons

    def _get_rejections(self, contract: HorizonEligibilityContract) -> dict[int, str]:
        """Get the appropriate rejections based on config level."""
        if self.config.use_production_level:
            return contract.production_rejections
        return contract.statistical_rejections

    def evaluate(self, horizon: int) -> HorizonGateResult:
        """Evaluate gate for a specific horizon."""
        eligible = []
        rejected = {}
        family_counts: dict[str, int] = {}

        for feat, contract in self.contracts.items():
            admissible = self._get_admissible_horizons(contract)
            rejections = self._get_rejections(contract)

            if horizon in admissible:
                eligible.append(feat)
                fam = contract.family
                family_counts[fam] = family_counts.get(fam, 0) + 1
            elif horizon in rejections:
                rejected[feat] = rejections[horizon]
            else:
                rejected[feat] = "NOT_EVALUATED"

        n_eligible = len(eligible)
        n_families = len(family_counts)
        max_concentration = max(family_counts.values()) / n_eligible if n_eligible > 0 else 1.0

        # Compute effective signal diversity
        effective_signals = self._compute_effective_signals(eligible, horizon)

        # Evaluate blocking conditions
        block_reasons = []

        if n_eligible < self.config.min_production_features:
            block_reasons.append(
                f"INSUFFICIENT_FEATURES: {n_eligible} < {self.config.min_production_features}"
            )

        if n_families < self.config.min_families:
            block_reasons.append(
                f"INSUFFICIENT_FAMILIES: {n_families} < {self.config.min_families}"
            )

        if n_eligible >= 2 and max_concentration > self.config.max_family_concentration:
            block_reasons.append(
                f"FAMILY_CONCENTRATION: {max_concentration:.2f} > {self.config.max_family_concentration}"
            )

        if effective_signals < self.config.min_effective_signals:
            block_reasons.append(
                f"LOW_EFFECTIVE_SIGNALS: {effective_signals:.2f} < {self.config.min_effective_signals}"
            )

        block = len(block_reasons) > 0

        # Generate report
        report_lines = [f"Horizon Gate: h{horizon}d"]
        report_lines.append(f"  Eligible: {n_eligible}/{len(self.contracts)}")
        report_lines.append(f"  Families: {n_families}")
        report_lines.append(f"  Max family concentration: {max_concentration:.2f}")
        report_lines.append(f"  Effective signals: {effective_signals:.2f}")
        report_lines.append(f"  Min features required: {self.config.min_production_features}")
        report_lines.append(f"  Min families required: {self.config.min_families}")
        report_lines.append(f"  Blocked: {block}")
        if block_reasons:
            report_lines.append(f"  Block reasons: {'; '.join(block_reasons)}")
        if eligible:
            report_lines.append(f"  Features: {', '.join(eligible)}")
        if rejected:
            report_lines.append(f"  Rejected ({len(rejected)}):")
            for feat, reason in sorted(rejected.items()):
                report_lines.append(f"    {feat}: {reason}")
        report = "\n".join(report_lines)

        return HorizonGateResult(
            horizon=horizon,
            eligible_features=eligible,
            rejected_features=rejected,
            block_horizon=block,
            block_reasons=block_reasons,
            config=self.config,
            n_eligible=n_eligible,
            n_families=n_families,
            family_concentration=max_concentration,
            effective_signals=effective_signals,
            report=report,
        )

    def _compute_effective_signals(self, eligible: list[str], horizon: int) -> float:
        """Compute effective signal count for eligible features."""
        if len(eligible) < 2:
            return float(len(eligible))

        # Build correlation matrix from IC values across horizons
        ic_vectors = []
        for feat in eligible:
            contract = self.contracts.get(feat)
            if contract is None:
                continue
            ic_vec = [contract.ic_by_horizon.get(h, 0) for h in range(1, 21)]
            ic_vectors.append(ic_vec)

        if len(ic_vectors) < 2:
            return float(len(eligible))

        ic_matrix = np.array(ic_vectors)
        if ic_matrix.shape[0] < 2:
            return float(len(eligible))

        # Compute correlation matrix
        try:
            corr = np.corrcoef(ic_matrix)
            corr = np.nan_to_num(corr, nan=0.0)
            return compute_effective_signal_count(corr)
        except Exception:
            return float(len(eligible))

    def evaluate_all(self, horizons: list[int]) -> dict[int, HorizonGateResult]:
        """Evaluate gate for multiple horizons."""
        return {h: self.evaluate(h) for h in horizons}


def filter_eligible_features(
    feature_list: list[str],
    contracts: dict[str, HorizonEligibilityContract],
    horizon: int,
    use_production: bool = True,
) -> list[str]:
    """
    Filter feature list to only include features eligible for the horizon.
    Replacement for filter_horizon_aligned_features().
    """
    eligible = []
    for feat in feature_list:
        contract = contracts.get(feat)
        if contract is None:
            continue
        admissible = (
            contract.production_admissible_horizons if use_production
            else contract.statistically_admissible_horizons
        )
        if horizon in admissible:
            eligible.append(feat)
    return eligible


class HorizonIneligibleError(Exception):
    """Raised when no features are eligible for a horizon."""

    def __init__(self, horizon: int, n_features: int, min_required: int, reasons: list[str] = None):
        self.horizon = horizon
        self.n_features = n_features
        self.min_required = min_required
        self.reasons = reasons or []
        reason_str = "; ".join(self.reasons) if self.reasons else "no eligible features"
        super().__init__(
            f"Horizon h{horizon}d has {n_features} eligible features "
            f"(minimum {min_required} required). Blocking: {reason_str}"
        )
