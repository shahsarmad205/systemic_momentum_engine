"""Pipeline stage: report writer.

Responsibility: Write all pipeline reports to disk.
Each stage produces one or more reports.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def write_csv(
    df: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
) -> Path:
    """Write DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logger.info("Report written: %s (%d rows)", path, len(df))
    return path


def write_json(
    data: dict[str, Any],
    path: str | Path,
) -> Path:
    """Write dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Report written: %s", path)
    return path


def write_text(
    text: str,
    path: str | Path,
) -> Path:
    """Write text to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    logger.info("Report written: %s", path)
    return path


def write_schema_validation_report(
    report: Any,  # SchemaValidationReport
    path: str | Path,
) -> Path:
    """Write schema validation report."""
    data = {
        "n_rows": report.n_rows,
        "n_columns": report.n_columns,
        "n_tickers": report.n_tickers,
        "n_dates": report.n_dates,
        "n_features": report.n_features,
        "n_targets": report.n_targets,
        "n_blocked_columns": report.n_blocked_columns,
        "blocked_columns": list(report.blocked_columns),
        "missing_required": list(report.missing_required),
        "non_numeric_features": list(report.non_numeric_features),
        "null_rate_by_feature": report.null_rate_by_feature,
    }
    return write_json(data, path)


def write_target_build_report(
    report: Any,  # TargetBuildReport
    path: str | Path,
) -> Path:
    """Write target build report."""
    data = {
        "target_column": report.target_column,
        "n_positive": report.n_positive,
        "n_negative": report.n_negative,
        "positive_rate": report.positive_rate,
        "n_missing": report.n_missing,
        "missing_rate": report.missing_rate,
        "risk_adjusted_available": report.risk_adjusted_available,
    }
    return write_json(data, path)


def write_feature_registry_report(
    registry: Any,  # FeatureFamilyRegistry
    path: str | Path,
) -> Path:
    """Write feature registry report."""
    rows = []
    for family in registry.families:
        features = registry.features_by_family(family)
        rows.append({
            "family": family,
            "n_features": len(features),
            "features": ",".join(features),
        })
    df = pd.DataFrame(rows)
    return write_csv(df, path)


def write_horizon_diagnostics(
    diagnostics: list,  # list[HorizonDiagnostic]
    path: str | Path,
) -> Path:
    """Write horizon diagnostics report."""
    rows = []
    for d in diagnostics:
        rows.append({
            "horizon": d.horizon,
            "mean_ic": d.mean_ic,
            "ic_std": d.ic_std,
            "ic_tstat": d.ic_tstat,
            "icir": d.icir,
            "signal_halflife_days": d.signal_halflife_days,
            "n_valid_dates": d.n_valid_dates,
            "eligible": d.eligible,
            "cost_viable": d.cost_viable,
            "rejection_reason": d.rejection_reason,
        })
    df = pd.DataFrame(rows)
    return write_csv(df, path)


def write_feature_admission_report(
    results: list,  # list[FeatureAdmissionResult]
    path: str | Path,
) -> Path:
    """Write feature admission report."""
    rows = []
    for r in results:
        rows.append({
            "feature": r.feature,
            "family": r.family,
            "admitted": r.admitted,
            "ic": r.ic,
            "ic_tstat": r.ic_tstat,
            "icir": r.icir,
            "halflife": r.halflife,
            "rejection_reason": r.rejection_reason,
        })
    df = pd.DataFrame(rows)
    return write_csv(df, path)


def write_cost_viability_scorecard(
    results: list,  # list[CostViabilityAdapterResult]
    path: str | Path,
) -> Path:
    """Write cost viability scorecard."""
    rows = []
    for r in results:
        rows.append({
            "candidate_id": r.candidate_id,
            "feature": r.feature,
            "family": r.family,
            "horizon": r.horizon,
            "cost_status": r.cost_status.value if hasattr(r.cost_status, 'value') else str(r.cost_status),
            "expected_alpha_bps": r.expected_alpha_bps,
            "expected_cost_bps": r.expected_cost_bps,
            "net_expected_alpha_bps": r.net_expected_alpha_bps,
            "alpha_cost_ratio": r.alpha_cost_ratio,
            "rejection_reason": r.rejection_reason,
        })
    df = pd.DataFrame(rows)
    return write_csv(df, path)


def write_candidate_factory_report(
    candidates: list,  # list[CandidateSpec]
    path: str | Path,
) -> Path:
    """Write candidate factory report."""
    rows = []
    for c in candidates:
        rows.append({
            "candidate_id": c.candidate_id,
            "model_name": c.model_name,
            "model_kind": c.model_kind,
            "feature_view": c.feature_view,
            "n_features": len(c.active_features),
            "horizon": c.horizon,
            "uses_proba": c.uses_proba,
        })
    df = pd.DataFrame(rows)
    return write_csv(df, path)


def write_walk_forward_validation_report(
    results: list,  # list[ValidationResult]
    path: str | Path,
) -> Path:
    """Write walk-forward validation report."""
    rows = []
    for r in results:
        rows.append({
            "candidate_id": r.candidate_id,
            "model_name": r.model_name,
            "horizon": r.horizon,
            "windows_evaluated": r.windows_evaluated,
            "oos_sharpe": r.oos_sharpe,
            "oos_ic": r.oos_ic,
            "oos_deflated_sharpe": r.oos_deflated_sharpe,
            "cost_adjusted_sharpe": r.cost_adjusted_sharpe,
            "max_drawdown": r.max_drawdown,
            "win_rate": r.win_rate,
            "turnover": r.turnover,
            "passed": r.passed,
            "gate_failures": ";".join(r.gate_failures),
        })
    df = pd.DataFrame(rows)
    return write_csv(df, path)


def write_promotion_gate_report(
    results: list,  # list[PromotionGateResult]
    path: str | Path,
) -> Path:
    """Write promotion gate report."""
    rows = []
    for r in results:
        rows.append({
            "candidate_id": r.candidate_id,
            "passed": r.passed,
            "failures": ";".join(r.failures),
            **{f"metric_{k}": v for k, v in r.metrics.items()},
        })
    df = pd.DataFrame(rows)
    return write_csv(df, path)
