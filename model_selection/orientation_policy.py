"""
P37: OrientationPolicy — governed signal sign-direction calibration.

Ensures score_direction (+1/-1) decisions are determined from in-sample
evidence only, are auditable per-window, and follow a configurable aggregation
policy rather than a bare majority vote.

Design principles:
  1. Train/validation evidence only — never OOS data for sign decisions.
  2. Every decision is recorded (per-window orientation manifest).
  3. Low-confidence detection — if IC_raw is near zero, follow config.
  4. Aggregate mode is versioned — changes in policy version invalidate
     cached artifacts so orientation decisions are reproducible.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any


# ── Enums ─────────────────────────────────────────────────────────────────────

class AggregateMode(str, enum.Enum):
    """How per-window orientation decisions are aggregated."""
    MAJORITY_VOTE = "majority_vote"
    IC_WEIGHTED = "ic_weighted"
    TRAIN_ONLY = "train_only"


class LowConfidenceMode(str, enum.Enum):
    """What to do when per-window IC evidence is weak."""
    FIXED_PLUS_ONE = "fixed_plus_one"
    ABSTAIN = "abstain"
    FALLBACK_TO_AGGREGATE = "fallback_to_aggregate"


class EvidenceSource(str, enum.Enum):
    """Where IC_raw for direction is computed."""
    VALIDATION_SPLIT = "validation_split"
    TRAINING_SPLIT = "training_split"
    FIXED_MANDATE = "fixed_mandate"


# ── Per-window orientation record ─────────────────────────────────────────────

@dataclass(frozen=True)
class OrientationRecord:
    """Single-window orientation decision record."""
    window_idx: int
    direction: int                      # +1 or -1
    ic_raw: float                        # IC before direction application
    ic_calibrated: float                 # IC after direction (ic_raw × direction)
    calibration_slope: float | None = None   # forecast calibration slope
    calibration_tstat: float | None = None    # calibration t-stat
    mode: str = "fixed"                  # fixed | calibrated
    reason: str = ""
    n_features: int = 0
    train_start: str = ""
    train_end: str = ""
    eval_start: str = ""
    eval_end: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_idx": self.window_idx,
            "direction": self.direction,
            "ic_raw": round(self.ic_raw, 6),
            "ic_calibrated": round(self.ic_calibrated, 6),
            "calibration_slope": round(self.calibration_slope, 4) if self.calibration_slope is not None else None,
            "calibration_tstat": round(self.calibration_tstat, 2) if self.calibration_tstat is not None else None,
            "mode": self.mode,
            "reason": self.reason,
            "n_features": self.n_features,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "eval_start": self.eval_start,
            "eval_end": self.eval_end,
        }


# ── OrientationPolicy ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OrientationPolicy:
    """
    P37: Governed score-direction calibration policy.

    Fields:
      policy_version         — bump when aggregation rules change
      allowed_sources        — which evidence sources are permitted
      aggregate_mode         — how per-window decisions are combined
      low_confidence_mode    — what to do when IC_raw is near zero
      low_confidence_ic_abs  — threshold below which IC is "low confidence"
      allow_per_window_overrides — if True, per-window direction CAN differ
                                     from aggregate when evidence is strong
      require_train_only_evidence — if True, direction MUST come from
                                     training/inner-validation, never OOS
      record_orientation_manifest — if True, write per-window records to CSV
      manifest_path          — output path for orientation manifest
    """

    policy_version: int = 1
    allowed_sources: tuple[EvidenceSource, ...] = (
        EvidenceSource.VALIDATION_SPLIT,
    )
    aggregate_mode: AggregateMode = AggregateMode.IC_WEIGHTED
    low_confidence_mode: LowConfidenceMode = LowConfidenceMode.FALLBACK_TO_AGGREGATE
    low_confidence_ic_abs: float = 0.001
    allow_per_window_overrides: bool = False
    require_train_only_evidence: bool = True
    record_orientation_manifest: bool = True
    manifest_path: str = "output/models/orientation_manifest.csv"

    def validate(self) -> list[str]:
        errors = []
        if self.low_confidence_ic_abs < 0:
            errors.append(f"low_confidence_ic_abs must be >= 0, got {self.low_confidence_ic_abs}")
        if not self.allowed_sources:
            errors.append("At least one allowed_source is required")
        return errors

    def is_low_confidence(self, ic_raw: float) -> bool:
        """Check if IC_raw magnitude is below the low-confidence threshold."""
        return abs(ic_raw) < self.low_confidence_ic_abs

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a pure-dict for JSON/YAML export."""
        return {
            "policy_version": self.policy_version,
            "allowed_sources": [s.value for s in self.allowed_sources],
            "aggregate_mode": self.aggregate_mode.value,
            "low_confidence_mode": self.low_confidence_mode.value,
            "low_confidence_ic_abs": self.low_confidence_ic_abs,
            "allow_per_window_overrides": self.allow_per_window_overrides,
            "require_train_only_evidence": self.require_train_only_evidence,
            "record_orientation_manifest": self.record_orientation_manifest,
            "manifest_path": self.manifest_path,
        }

    def aggregate_direction(
        self,
        per_window_records: list[OrientationRecord],
    ) -> tuple[int, str, dict[str, Any]]:
        """
        Aggregate per-window orientation records into a single direction.

        Returns:
            direction  — +1 or -1
            reason     — human-readable explanation
            diagnostics — per-window summary for logging
        """
        if not per_window_records:
            return 1, "no_windows_default", {"n_windows": 0}

        n_windows = len(per_window_records)
        dirs = [r.direction for r in per_window_records]
        n_pos = dirs.count(1)
        n_neg = dirs.count(-1)
        ic_raws = [(r.ic_raw, r.direction) for r in per_window_records]

        diag: dict[str, Any] = {
            "n_windows": n_windows,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "per_window_directions": dirs,
        }

        if self.aggregate_mode == AggregateMode.TRAIN_ONLY:
            # Only count windows where IC_raw is clearly non-zero
            strong = [
                r for r in per_window_records
                if not self.is_low_confidence(r.ic_raw)
            ]
            if not strong:
                if self.low_confidence_mode == LowConfidenceMode.FIXED_PLUS_ONE:
                    return 1, "train_only_no_strong_evidence_fallback_plus_one", diag
                return 1, "train_only_no_strong_evidence", diag
            s_pos = sum(1 for r in strong if r.direction > 0)
            s_neg = len(strong) - s_pos
            diag["strong_windows"] = len(strong)
            diag["strong_positive"] = s_pos
            diag["strong_negative"] = s_neg
            direction = 1 if s_pos >= s_neg else -1
            return direction, f"train_only_majority_{direction:+d}", diag

        if self.aggregate_mode == AggregateMode.IC_WEIGHTED:
            # Weight by |IC_raw| magnitude so windows with weak evidence count less
            weighted_sum = 0.0
            total_weight = 0.0
            for r in per_window_records:
                w = abs(r.ic_raw)
                weighted_sum += r.direction * w
                total_weight += w
            if total_weight < 1e-12:
                return 1, "ic_weighted_zero_total_weight", diag
            direction = 1 if weighted_sum >= 0 else -1
            diag["weighted_sum"] = round(weighted_sum, 6)
            diag["total_weight"] = round(total_weight, 6)
            diag["weighted_strength"] = round(abs(weighted_sum) / total_weight, 4) if total_weight > 0 else 0.0
            return direction, f"ic_weighted_{weighted_sum:.4f}_direction_{direction:+d}", diag

        # MAJORITY_VOTE
        direction = 1 if n_pos >= n_neg else -1
        return direction, f"majority_vote_{n_pos}_vs_{n_neg}", diag


# ── YAML Parsing ──────────────────────────────────────────────────────────────

def parse_orientation_policy(raw: dict[str, Any] | None) -> OrientationPolicy:
    """P37: Parse orientation_policy block from YAML."""
    if not raw:
        return OrientationPolicy()

    # Allowed sources
    sources_raw = raw.get("allowed_sources", ["validation_split"])
    sources = tuple(
        EvidenceSource(str(s).strip().lower())
        for s in sources_raw
    )

    # Aggregate mode
    agg_str = str(raw.get("aggregate_mode", "ic_weighted")).strip().lower()
    try:
        agg_mode = AggregateMode(agg_str)
    except ValueError:
        raise ValueError(
            f"Invalid orientation_policy.aggregate_mode: '{agg_str}'. "
            f"Valid: {[m.value for m in AggregateMode]}"
        )

    # Low confidence mode
    lc_str = str(raw.get("low_confidence_mode", "fallback_to_aggregate")).strip().lower()
    try:
        lc_mode = LowConfidenceMode(lc_str)
    except ValueError:
        raise ValueError(
            f"Invalid orientation_policy.low_confidence_mode: '{lc_str}'. "
            f"Valid: {[m.value for m in LowConfidenceMode]}"
        )

    return OrientationPolicy(
        policy_version=int(raw.get("policy_version", 1)),
        allowed_sources=sources,
        aggregate_mode=agg_mode,
        low_confidence_mode=lc_mode,
        low_confidence_ic_abs=float(raw.get("low_confidence_ic_abs", 0.001)),
        allow_per_window_overrides=bool(raw.get("allow_per_window_overrides", False)),
        require_train_only_evidence=bool(raw.get("require_train_only_evidence", True)),
        record_orientation_manifest=bool(raw.get("record_orientation_manifest", True)),
        manifest_path=str(raw.get("manifest_path", "output/models/orientation_manifest.csv")),
    )


def orientation_manifest_to_dataframe(
    records: list[OrientationRecord],
    *,
    model_name: str = "",
    policy_version: int = 1,
) -> "pd.DataFrame":
    """Convert orientation records to a DataFrame for CSV export."""
    try:
        import pandas as pd
    except ImportError:
        return None

    rows = []
    for r in records:
        d = r.to_dict()
        d["model_name"] = model_name
        d["policy_version"] = policy_version
        rows.append(d)

    if not rows:
        return pd.DataFrame()

    cols = [
        "model_name", "window_idx", "direction", "ic_raw", "ic_calibrated",
        "calibration_slope", "calibration_tstat", "mode", "reason",
        "n_features", "train_start", "train_end", "eval_start", "eval_end",
        "policy_version",
    ]
    return pd.DataFrame(rows)[[c for c in cols if c in pd.DataFrame(rows).columns]]
