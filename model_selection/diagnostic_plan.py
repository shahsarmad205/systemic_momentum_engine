from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DIAGNOSTIC_REGISTRY: dict[str, dict[str, Any]] = {
    "research_diagnostics": {
        "description": "Per-candidate research diagnostics (decay, regime, attribution, etc.)",
        "affects_gates": False,
        "module": "model_selection.research_diagnostics",
        "cost_profile": "expensive",
        "default_enabled": False,
    },
    "empirical_baselines": {
        "description": "Per-candidate empirical baseline comparisons",
        "affects_gates": False,
        "module": "model_selection.empirical_baselines",
        "cost_profile": "moderate",
        "default_enabled": False,
    },
}


def resolve_diagnostics_config(diagnostics: dict[str, Any] | None) -> dict[str, bool]:
    diags = diagnostics or {}
    resolved: dict[str, bool] = {}
    for name, spec in DIAGNOSTIC_REGISTRY.items():
        raw = diags.get(name)
        if raw is None:
            resolved[name] = bool(spec["default_enabled"])
        elif isinstance(raw, bool):
            resolved[name] = raw
        elif isinstance(raw, dict):
            resolved[name] = bool(raw.get("enabled", spec["default_enabled"]))
        else:
            resolved[name] = bool(raw)
    return resolved


@dataclass
class DiagnosticExecutionPlan:
    """Controls per-candidate optional research diagnostics.

    Pipeline-level diagnostics (P30-P34) have been removed from the
    institutional research pipeline. This plan manages only per-candidate
    exploratory diagnostics that exist for post-hoc analysis.
    """

    research_diagnostics: bool = False
    empirical_baselines: bool = False

    executed: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> DiagnosticExecutionPlan:
        ms = cfg.get("model_selection", {}) or {}
        val = (ms.get("validation", {}) or {}) if isinstance(ms, dict) else {}
        raw = (val.get("diagnostics", {}) or {}) if isinstance(val, dict) else {}
        resolved = resolve_diagnostics_config(raw)
        return cls(
            research_diagnostics=resolved.get("research_diagnostics", False),
            empirical_baselines=resolved.get("empirical_baselines", False),
        )

    def record(self, category: str, candidate_name: str) -> None:
        self.executed.setdefault(category, []).append(candidate_name)

    @property
    def any_enabled(self) -> bool:
        return self.research_diagnostics or self.empirical_baselines

    @property
    def summary(self) -> str:
        parts = [
            f"research_diagnostics={'ON' if self.research_diagnostics else 'OFF'}",
            f"empirical_baselines={'ON' if self.empirical_baselines else 'OFF'}",
        ]
        counts = []
        for cat, names in self.executed.items():
            counts.append(f"{cat}={len(names)}")
        if counts:
            parts.append("executed: " + ", ".join(counts))
        return f"DiagnosticExecutionPlan({', '.join(parts)})"
