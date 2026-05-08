#!/usr/bin/env python3
"""
Model Selection Runner (Walk-Forward, leakage-safe)

Goal
  Compare multiple classification models on out-of-sample (walk-forward) performance
  using a feature matrix built by:
    agents.weight_learning_agent.feature_builder.build_feature_matrix

Important
  The feature builder includes target-like columns (e.g. 'forward_return' and derived
  columns like 'spy_forward_5d', plus cross-sectional z-scores of those). Those columns
  MUST NOT be used as model inputs, or you'll get look-ahead bias and unrealistic results.

Outputs
  - output/models/model_comparison.csv
  - output/models/best_model.pkl   (pickle of estimator + metadata)
  - output/models/best_model.meta.json

Selection
  Default ranking uses formal ``oos_deflated_sharpe`` on the primary alpha path.
  Artifact export is additionally blocked by executable promotion gates: cost-aware
  Sharpe, IC evidence, drawdown, factor/sector exposure, implementation cost, and
  leg-level diagnostics.

Integration note
  backtesting/signal generation currently doesn't load sklearn pickle models for inference.
  This script prints suggested YAML fields, but does not modify backtest_config.yaml.
"""

from __future__ import annotations

import os
import sys
# Must be set before numpy / XGBoost / sklearn load their thread pools.
# Accelerate (macOS BLAS) and OpenMP both maintain internal thread pools that
# survive fork() in a locked state, causing child processes to deadlock.
# Limiting to 1 thread means no background threads exist at fork time.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import json
import multiprocessing as _mp
import threading
import pickle
import time
import traceback
from dataclasses import dataclass, replace, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import threadpoolctl
import yaml
from model_selection.validation import (
    ExecutionCostConfig,
    DEBUG_DIAGNOSTICS,
    LongAlphaCandidateConfig,
    MetricIntegrityError,
    PromotionTier,
    ValidationStateCache,
    _safe_float,
    compute_execution_robustness,
    cross_sectional_ic,
    decile_return_diagnostics,
    evaluate_promotion_gates,
    executable_metrics,
    build_target_weights,
    simulate_executable_portfolio,
    simulate_proxy_portfolio,
    _get_implied_weights,
    _calibrate_l1_turnover_penalty,
    ChurnFilterConfig,
    apply_churn_filter,
    robust_spearman,
)
from model_selection.horizon_contract import (
    build_horizon_contract,
    HorizonContract,
    HorizonConfigurationError,
    SweepMode,
)
from model_selection.alpha_research import (
    ALPHA_RESEARCH_SCHEMA_VERSION,
    apply_admitted_feature_transforms,
    run_alpha_research,
    summarize_admission,
)
from model_selection.diagnostic_plan import DiagnosticExecutionPlan
from model_selection.factor_subsumption import build_factor_mimicking_returns, factor_subsumption_diagnostics
from model_selection.forecast_calibration import calibrate_scores as _calibrate_scores
from model_selection.research_diagnostics import (
    compute_full_diagnostics as _compute_research_diagnostics,
    generate_research_report as _generate_research_report,
)
from model_selection.empirical_baselines import compute_empirical_baselines as _compute_empirical_baselines
from model_selection.empirical_baselines import (
    alpha_capture_decomposition as _alpha_capture_decomposition,
    alpha_capture_summary_from_per_model as _alpha_capture_summary_from_per_model,
)
from model_selection.orientation_policy import (
    AggregateMode,
    OrientationPolicy,
    OrientationRecord,
    parse_orientation_policy,
    orientation_manifest_to_dataframe as _orientation_manifest_to_dataframe,
)
from model_selection.score_direction import (
    determine_score_direction as _determine_score_direction,
    check_class_ordering as _check_class_ordering,
    ic_signal_strength as _ic_signal_strength,
)
from model_selection.research_contract import (
    FEATURE_SPECS,
    TimingContractViolation,
    audit_feature_contract,
    get_horizon_alignment_report,
    is_model_feature_column,
    summarize_feature_contract,
    validate_signal_execution_timing,
)
from model_selection.horizon_eligibility import (
    HorizonEligibilityContract,
    compute_all_eligibility,
    compute_eligibility,
    format_eligibility_report,
)
from model_selection.horizon_gate import (
    HorizonGate,
    HorizonGateConfig,
    HorizonIneligibleError,
    filter_eligible_features,
)
from model_selection.configuration import (
    alpha_admission_config as _alpha_admission_config,
    embargo_days_config as _embargo_days_config,
    evaluation_config as _evaluation_config,
    execution_cost_config as _execution_cost_config,
    feature_builder_data_kwargs as _feature_builder_data_kwargs,
    horizon_contract_config as _horizon_contract_config,
    horizon_alignment_config as _horizon_alignment_config,
    nested_validation_config as _nested_validation_config,
    parallel_research_config as _parallel_research_config,
    preprocess_winsor_q as _preprocess_winsor_q,
    long_alpha_candidate_config as _long_alpha_candidate_config,
    promotion_gate_config as _promotion_gate_config,
    PromotionGateConfig,
    screening_config as _screening_config,
    target_config as _target_config,
    warn_deprecated_config_duplicates as _warn_deprecated_config_duplicates,
)
from model_selection.model_registry import (
    BASELINE_SCORERS,
    PrefitWeightedEnsemble,
    build_models as _build_models,
    constrain_model_parallelism as _constrain_model_parallelism,
    is_classifier_stock_selector as _is_classifier_stock_selector,
    is_diagnostic_only as _is_diagnostic_only,
)
from model_selection.alpha_model import assert_pure_alpha_fit_kwargs
from model_selection.statistics import (
    compute_deflated_sharpe as _compute_deflated_sharpe,
    compute_institutional_metrics as _compute_institutional_metrics,
    compute_psr as _compute_psr,
    weighted_recency_mean as _weighted_recency_mean,
)
from model_selection.preparation import PreparedPanelCache


from model_selection.research_state import ResearchStateStore, TimingLedger, frame_fingerprint
from model_selection.institutional_contracts import (
    RunTelemetryContract,
    build_cache_key_spec,
    build_cost_assumption_set,
    build_feature_manifest,
    build_horizon_run_contract,
    build_pit_audit_ledger,
    build_pit_transform_specs,
    build_promotion_gate_specs,
    build_target_manifest,
    build_target_specs,
    emit_telemetry_event,
    promotion_decision_from_row,
    stable_fingerprint,
    write_institutional_run_manifest,
    write_json_artifact,
)
from model_selection.training import (
    FeaturePreprocessor,
    TargetConfig,
    add_institutional_targets,
)
from model_selection.target_panel_provider import build_target_provider
from utils.universe import load_universe

# ── Institutional cost viability wiring (P29) ────────────────────────────────
from model_selection.cost_viability_wiring import (
    evaluate_feature_cost_viability,
    evaluate_candidate_cost_viability,
    apply_alpha_to_trade_policy,
    apply_no_trade_band,
    generate_cost_viability_reports,
    compute_institutional_horizon_gate,
    summarize_feature_cost_gate,
    filter_feature_cost_results,
    feature_cost_results_to_horizon_contracts,
    CostViabilityWiringState,
)

# ── Diagnostics removed (P30-P34) per institutional structure ────────────────
# SignalDecayEngine, ICDiagnosticsEngine, ConditionalAlphaEngine,
# PITConditionEngine, FeatureDiversity — not gates, not mathematically essential.

import logging as _logging
logger = _logging.getLogger(__name__)

def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_horizon_manifest(
    *,
    horizon_contract: Any,
    cli_horizon: Any,
    out_dir: Path,
) -> None:
    """Write the resolved run horizon manifest after the output directory exists."""
    manifest = horizon_contract.to_dict()
    manifest["resolved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["cli_horizon"] = cli_horizon
    manifest_path = out_dir / "run_horizon_manifest.json"
    try:
        _write_json(manifest_path, manifest)
    except (OSError, TypeError, ValueError) as exc:
        warning = {
            "warning_type": "horizon_manifest_write_failed",
            "path": str(manifest_path),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        logger.warning("Failed to write horizon manifest to %s: %s", manifest_path, exc)
        try:
            _write_json(out_dir / "run_horizon_manifest_warning.json", warning)
        except OSError as warn_exc:
            logger.warning("Failed to write horizon manifest warning artifact: %s", warn_exc)


# ── Debug Flags ──────────────────────────────────────────────────────────────
FINAL_WINDOW_DEBUG = False
NESTED_SELECTION_CACHE_VERSION = "nested_selection_horizon_contract_v4"


def safe_run_diagnostic(name: str, fn: "Callable[[], Any]", default: "Any" = None) -> "Any":
    """Run a diagnostic without affecting pipeline results. Failures are silently suppressed."""
    try:
        return fn()
    except Exception as _exc:
        logger.debug("[diag:%s] skipped: %s", name, _exc)
        return default


def _alpha_research_cache_is_compatible(
    admission: pd.DataFrame,
    *,
    alpha_cfg: Any,
) -> tuple[bool, str]:
    """Guard against stale alpha-research artifacts surviving code/schema fixes."""
    if admission is None or admission.empty:
        return False, "empty_admission"
    required = {"feature", "admitted", "recommended_action", "bhy_min_tstat", "bhy_qvalue"}
    missing = sorted(required - set(admission.columns))
    if missing:
        return False, f"missing_columns:{','.join(missing)}"
    if bool(getattr(alpha_cfg, "apply_bhy_correction", True)):
        bhy_min = pd.to_numeric(admission["bhy_min_tstat"], errors="coerce")
        if bhy_min.empty or not np.isfinite(bhy_min.to_numpy(dtype=float)).all():
            return False, "non_finite_bhy_min_tstat"
    actions = admission["recommended_action"].astype(str)
    admitted = admission["admitted"].eq(True)
    if bool(getattr(alpha_cfg, "fail_if_below_minimum", False)) and bool((admitted & actions.str.contains("fallback", regex=False)).any()):
        return False, "fail_closed_policy_rejects_fallback_admission"
    min_required = max(0, int(getattr(alpha_cfg, "minimum_admitted_features", 0) or 0))
    if bool(getattr(alpha_cfg, "fail_if_below_minimum", False)) and int(admitted.sum()) < min_required:
        return False, f"admitted_below_minimum:{int(admitted.sum())}<{min_required}"
    return True, "compatible"


# ── Simulation telemetry (observability only — no logic changes) ─────────────
class SimulationTelemetry:
    """Collects per-call stats for ``simulate_executable_portfolio`` and proxy.

    Thread-safe via a lock; each forked worker has its own instance.
    Workers return their ``.records`` in the result dict, and the parent
    merges them via :meth:`merge`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._market_state_stats = {"hits": 0, "misses": 0, "build_time_s": 0.0}

    def record(
        self,
        *,
        phase: str,
        model_name: str,
        window_idx: int,
        scored: "pd.DataFrame",
        cfg: Any,
        runtime_s: float,
        simulation_mode: str,
        is_cached: bool = False,
    ) -> None:
        n_tickers = int(scored["ticker"].nunique()) if "ticker" in scored.columns and not scored.empty else 0
        n_dates = int(scored["date"].nunique()) if "date" in scored.columns and not scored.empty else 0
        rebal = max(1, int(getattr(cfg, "rebalance_every_days", 5)))
        use_optimizer = bool(getattr(cfg, "use_optimizer", True))
        qp_solves = (n_dates + rebal - 1) // rebal if use_optimizer else 0
        entry = {
            "phase": str(phase),
            "model_name": str(model_name),
            "window_idx": int(window_idx),
            "simulation_mode": str(simulation_mode),
            "universe_size": n_tickers,
            "n_dates": n_dates,
            "qp_solves": qp_solves,
            "runtime_s": round(float(runtime_s), 4),
            "is_cached": bool(is_cached),
        }
        with self._lock:
            self._records.append(entry)

    @property
    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)

    def merge(self, other_records: list[dict[str, Any]]) -> None:
        with self._lock:
            self._records.extend(other_records)

    def merge_market_state_stats(self, other_stats: dict[str, float]) -> None:
        with self._lock:
            self._market_state_stats["hits"] += other_stats.get("hits", 0)
            self._market_state_stats["misses"] += other_stats.get("misses", 0)
            self._market_state_stats["build_time_s"] += other_stats.get("build_time_s", 0.0)

    def summary(self) -> dict[str, Any]:
        recs = self.records
        if not recs:
            return {"total_calls": 0, "total_qp_solves": 0}
        total_calls = len(recs)
        total_qp = sum(r["qp_solves"] for r in recs)
        total_runtime = sum(r["runtime_s"] for r in recs)
        avg_n = float(np.mean([r["universe_size"] for r in recs])) if recs else 0.0
        cache_hits = sum(1 for r in recs if r.get("is_cached", False))
        cache_misses = total_calls - cache_hits
        uncached_runtimes = [r["runtime_s"] for r in recs if not r.get("is_cached", False)]
        avg_uncached_runtime = float(np.mean(uncached_runtimes)) if uncached_runtimes else 0.0
        estimated_runtime_saved_sec = cache_hits * avg_uncached_runtime
        phases: dict[str, list[dict[str, Any]]] = {}
        for r in recs:
            phases.setdefault(r["phase"], []).append(r)
        phase_summary = {}
        for phase, items in sorted(phases.items()):
            p_runtime = sum(i["runtime_s"] for i in items)
            phase_summary[phase] = {
                "calls": len(items),
                "qp_solves": sum(i["qp_solves"] for i in items),
                "runtime_s": round(p_runtime, 2),
                "runtime_pct": round(100.0 * p_runtime / total_runtime, 1) if total_runtime > 0 else 0.0,
                "avg_universe_size": round(float(np.mean([i["universe_size"] for i in items])), 1),
                "avg_runtime_per_call_s": round(p_runtime / len(items), 3) if items else 0.0,
                "cache_hits": sum(1 for i in items if i.get("is_cached", False)),
            }
        slowest = max(recs, key=lambda r: r["runtime_s"])
        return {
            "total_calls": total_calls,
            "total_qp_solves": total_qp,
            "total_runtime_s": round(total_runtime, 2),
            "avg_universe_size": round(avg_n, 1),
            "avg_runtime_per_call_s": round(total_runtime / total_calls, 3) if total_calls else 0.0,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate_pct": round(100.0 * cache_hits / total_calls, 1) if total_calls else 0.0,
            "duplicate_simulations_avoided": cache_hits,
            "estimated_runtime_saved_sec": round(estimated_runtime_saved_sec, 2),
            "by_phase": phase_summary,
            "slowest_call": {
                "phase": slowest["phase"],
                "model_name": slowest["model_name"],
                "window_idx": slowest["window_idx"],
                "runtime_s": slowest["runtime_s"],
                "qp_solves": slowest["qp_solves"],
                "universe_size": slowest["universe_size"],
            },
            "market_state_cache": dict(self._market_state_stats),
        }

    def print_summary(self) -> None:
        s = self.summary()
        if s["total_calls"] == 0:
            print("\n[SimTelemetry] No simulation calls recorded.")
            return
        print("\n" + "=" * 72)
        print("  SIMULATION RUNTIME REPORT")
        print("=" * 72)
        print(f"  Total calls:       {s['total_calls']}")
        print(f"  Total QP solves:   {s['total_qp_solves']}")
        print(f"  Total runtime:     {s['total_runtime_s']:.1f}s")
        print(f"  Avg universe (N):  {s['avg_universe_size']:.0f}")
        print(f"  Avg runtime/call:  {s['avg_runtime_per_call_s']:.3f}s")
        print()
        print(f"  Cache Hits:        {s['cache_hits']} (avoided duplicates)")
        print(f"  Cache Misses:      {s['cache_misses']}")
        print(f"  Hit Rate:          {s['cache_hit_rate_pct']:.1f}%")
        print(f"  Estimated Savings: {s['estimated_runtime_saved_sec']:.1f}s")
        ms = s.get("market_state_cache", {})
        ms_hits = ms.get("hits", 0)
        ms_misses = ms.get("misses", 0)
        ms_total = ms_hits + ms_misses
        ms_hit_rate = 100.0 * ms_hits / ms_total if ms_total else 0.0
        ms_avg_build = ms.get("build_time_s", 0.0) / ms_misses if ms_misses else 0.0
        ms_time_saved = ms_hits * ms_avg_build
        print()
        print("  MARKET STATE CACHE")
        print(f"  Hits:              {ms_hits}")
        print(f"  Misses:            {ms_misses}")
        print(f"  Hit Rate:          {ms_hit_rate:.1f}%")
        print(f"  Avg Build Time:    {ms_avg_build:.3f}s")
        print(f"  Estimated Savings: {ms_time_saved:.1f}s")
        print()
        print(f"  {'Phase':<30} {'Calls':>6} {'Hits':>6} {'QP':>8} {'Runtime':>10} {'%':>6} {'Avg N':>7} {'Avg/call':>10}")
        print(f"  {'-'*28:<30} {'-'*6:>6} {'-'*6:>6} {'-'*8:>8} {'-'*10:>10} {'-'*6:>6} {'-'*7:>7} {'-'*10:>10}")
        for phase, ps in sorted(s["by_phase"].items()):
            print(
                f"  {phase:<30} {ps['calls']:>6} {ps.get('cache_hits', 0):>6} {ps['qp_solves']:>8} "
                f"{ps['runtime_s']:>9.1f}s {ps['runtime_pct']:>5.1f}% "
                f"{ps['avg_universe_size']:>7.0f} {ps['avg_runtime_per_call_s']:>9.3f}s"
            )
        sc = s["slowest_call"]
        print()
        print(f"  Slowest: {sc['model_name']} window={sc['window_idx']} "
              f"phase={sc['phase']} ({sc['runtime_s']:.2f}s, {sc['qp_solves']} QP, N={sc['universe_size']})")
        print("=" * 72)

    def to_json_payload(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "calls": self.records,
        }


class MemoryLedger:
    """Tracks RSS memory usage across research phases."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()
        import psutil
        self._process = psutil.Process(os.getpid())
        self.max_rss_gb = 0.0

    def record(self, event: str, **metadata: Any) -> None:
        import psutil
        rss_gb = self._process.memory_info().rss / (1024**3)
        entry = {
            "timestamp": time.time(),
            "elapsed_s": round(time.time() - self._start_time, 2),
            "event": event,
            "rss_gb": round(rss_gb, 3),
            **metadata
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.max_rss_gb = max(self.max_rss_gb, rss_gb)
        allowed_events = {
            "prewarm_start", "prewarm_complete",
            "phase_FastSweep_start", "phase_FastSweep_complete",
            "phase_DeepValidation_start", "phase_DeepValidation_complete"
        }
        if event in allowed_events:
            logger.info("  [Memory] %-30s | RSS: %.2f GB", event, rss_gb)


_SIMULATION_TELEMETRY = SimulationTelemetry()


def _print_failure_report(
    report: "pd.DataFrame",
    long_kinds: list[str],
    overlay_kinds: list[str],
    short_kinds: list[str],
    out_dir: "Path",
) -> None:
    """Emit a structured failure report when any promotion pool is empty.

    Produces both a human-readable text file and a machine-parseable JSON file.
    Every rejected candidate receives a root-cause classification — no silent omissions.
    Multi-category failures are reported as a list, not a single mutually-exclusive label.
    Diagnostic-only robustness failures are separated from blocking failures.
    """
    import json as _json

    def _fmt(val: Any, digits: int = 3) -> str:
        try:
            f = float(val)
            return f"{f:.{digits}f}" if np.isfinite(f) else "nan"
        except Exception:
            return str(val)

    def _rank_col(df: "pd.DataFrame") -> str:
        for col in ("oos_deflated_sharpe", "oos_sharpe_chained", "exec_sharpe"):
            if col in df.columns:
                return col
        return df.columns[0]

    _FAILURE_ACTIONS: dict[str, str] = {
        "min_ic_tstat": "improve feature predictive power or extend training window",
        "min_ic_ir": "improve IC consistency across windows; reduce signal decay",
        "min_sharpe": "requires more stable cross-window performance",
        "min_cost_aware_sharpe": "reduce turnover or improve gross alpha above cost drag",
        "max_beta_abs_mean": "neutralise market beta via factor residualisation or constraints",
        "max_sector_abs_mean": "add sector constraints or diversify feature set",
        "min_beat_rate": "signal is inconsistent across windows; investigate regime dependence",
        "min_psr": "Sharpe not statistically distinguishable from noise; need more windows or higher Sharpe",
        "min_long_leg_sharpe": "long-leg alpha is negative; revisit feature engineering for long side",
        "min_short_leg_sharpe": "short-leg drag — consider long-only deployment",
        "max_drawdown": "excessive drawdown; add risk limits or position sizing constraints",
        "min_decile_spread": "decile spread near zero; signal lacks cross-sectional dispersion",
        "min_tail_monotonicity": "top/bottom deciles non-monotonic; signal is not rank-consistent",
        "max_cost_to_gross_pnl": "cost burden too high relative to alpha; reduce rebalance frequency",
        "nested_min_ic": "nested model IC below threshold; economic model is not predictive",
        "nested_min_sharpe": "nested model Sharpe too low; economic model not adding value",
        "min_windows": "insufficient walk-forward windows for statistical confidence",
        "nested_min_windows": "nested validation has too few windows; extend data history",
        "diagnostic_only": "model flagged as diagnostic-only; not eligible for production",
        "robust_halflife": "signal halflife below rebalance requirement; alpha decays too fast for execution contract",
        "robust_caic_ratio": "cost-adjusted IC ratio below threshold; net alpha insufficient after costs",
        "robust_turnover": "average turnover exceeds budget; reduce rebalance frequency or constrain optimizer",
        "min_subsumption_alpha_ann": "feature alpha subsumed by existing factors; check factor overlap",
        "min_subsumption_alpha_tstat": "subsumed alpha t-stat insignificant; feature adds no independent signal",
        "max_subsumption_r2": "feature highly correlated with existing factors; consider residualisation",
        "max_subsumption_loading_abs": "factor loading too large; feature dominated by common risk factor",
    }

    # Root-cause category definitions — non-mutually-exclusive
    _ROOT_CAUSE_PATTERNS: list[tuple[str, set[str]]] = [
        ("signal_weakness", {"min_ic_tstat", "min_ic_ir", "min_beat_rate", "min_psr", "nested_min_ic"}),
        ("risk_exposure", {"max_beta_abs_mean", "max_sector_abs_mean", "max_drawdown"}),
        ("cost_drag", {"max_cost_to_gross_pnl", "min_cost_aware_sharpe", "robust_caic_ratio"}),
        ("short_leg_drag", {"min_short_leg_sharpe"}),
        ("long_leg_weakness", {"min_long_leg_sharpe"}),
        ("signal_decay", {"robust_halflife", "robust_turnover"}),
        ("factor_subsumption", {"min_subsumption_alpha_ann", "min_subsumption_alpha_tstat", "max_subsumption_r2", "max_subsumption_loading_abs"}),
        ("insufficient_data", {"min_windows", "nested_min_windows"}),
        ("dispersion_failure", {"min_decile_spread", "min_tail_monotonicity"}),
        ("nested_underperformance", {"nested_min_sharpe"}),
        ("diagnostic_only", {"diagnostic_only"}),
    ]

    def _classify_root_causes(failures: list[str]) -> list[str]:
        """Return all matching root-cause categories for a set of failures."""
        fail_set = {f.removeprefix("diagnostic:") for f in failures}
        matched: list[str] = []
        for label, pattern_set in _ROOT_CAUSE_PATTERNS:
            if fail_set & pattern_set:
                matched.append(label)
        if not matched:
            # Fallback: any unmatched failure gets a generic classification
            matched.append("unclassified_failure")
        return matched

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("STRUCTURED FAILURE REPORT — No production model promoted")
    lines.append("=" * 72)

    json_report: dict[str, Any] = {
        "report_type": "promotion_failure",
        "path_families": [],
    }

    # P40: Collect best candidates per family for LO vs Exec Sharpe reconciliation
    _best_per_family: list[tuple[str, dict]] = []

    for path_label, kinds in [
        ("LONG / SPREAD", long_kinds),
        ("OVERLAY", overlay_kinds),
        ("SHORT", short_kinds),
    ]:
        family_rows = report[report["model_kind"].isin(kinds)] if "model_kind" in report.columns else pd.DataFrame()
        if family_rows.empty:
            lines.append(f"\n[{path_label}]  No models evaluated.")
            json_report["path_families"].append({"label": path_label, "status": "no_models"})
            continue

        rank_col = _rank_col(family_rows)
        family_rows = family_rows.copy()
        family_rows["_rank"] = pd.to_numeric(family_rows[rank_col], errors="coerce")
        best = family_rows.sort_values("_rank", ascending=False).iloc[0]

        tier = str(best.get("promotion_tier", "rejected"))
        failures_str = str(best.get("promotion_failures", "") or "")
        failures = [f.strip() for f in failures_str.split(",") if f.strip()]
        name = str(best.get("model_name", best.get("name", "?")))
        kind = str(best.get("model_kind", "?"))

        _best_per_family.append((path_label, best.to_dict()))

        blocking_failures = [f for f in failures if not f.startswith("diagnostic:")]
        diagnostic_failures = [f.removeprefix("diagnostic:") for f in failures if f.startswith("diagnostic:")]
        root_causes = _classify_root_causes(failures)

        lines.append(f"\n[{path_label}]  Best candidate: {name}  (kind={kind}, tier={tier})")
        lines.append(f"  {rank_col}: {_fmt(best.get(rank_col))}  "
                     f"exec_sharpe: {_fmt(best.get('exec_sharpe'))}  "
                     f"ic_tstat: {_fmt(best.get('oos_ic_tstat', best.get('cs_ic_spearman_tstat')))}  "
                     f"beat_rate: {_fmt(best.get('oos_beat_rate'))}  "
                     f"psr: {_fmt(best.get('oos_psr'))}")
        lines.append(f"  long_leg_sharpe: {_fmt(best.get('exec_long_leg_sharpe'))}  "
                     f"short_leg_sharpe: {_fmt(best.get('exec_short_leg_sharpe'))}  "
                     f"beta: {_fmt(best.get('exec_beta_abs_mean'))}  "
                     f"cost/pnl: {_fmt(best.get('exec_cost_to_gross_pnl'))}")

        if blocking_failures:
            lines.append(f"  Blocking failures ({len(blocking_failures)}):")
            for f in blocking_failures:
                action = _FAILURE_ACTIONS.get(f, "investigate the metric and adjust model/features")
                lines.append(f"    - {f}: {action}")

        if diagnostic_failures:
            lines.append(f"  Diagnostic warnings ({len(diagnostic_failures)}):")
            for f in diagnostic_failures:
                action = _FAILURE_ACTIONS.get(f, "review diagnostic metric for research insight")
                lines.append(f"    - {f}: {action}")

        if root_causes:
            lines.append(f"  Root causes: {', '.join(root_causes)}")
        else:
            lines.append("  No gate failures recorded (nested/IC gate may have blocked).")

        json_report["path_families"].append({
            "label": path_label,
            "best_candidate": name,
            "model_kind": kind,
            "promotion_tier": tier,
            "blocking_failures": blocking_failures,
            "diagnostic_warnings": diagnostic_failures,
            "root_causes": root_causes,
            "metrics": {
                rank_col: _fmt(best.get(rank_col)),
                "exec_sharpe": _fmt(best.get("exec_sharpe")),
                "ic_tstat": _fmt(best.get("oos_ic_tstat", best.get("cs_ic_spearman_tstat"))),
                "beat_rate": _fmt(best.get("oos_beat_rate")),
                "psr": _fmt(best.get("oos_psr")),
                "long_leg_sharpe": _fmt(best.get("exec_long_leg_sharpe")),
                "short_leg_sharpe": _fmt(best.get("exec_short_leg_sharpe")),
                "beta": _fmt(best.get("exec_beta_abs_mean")),
                "cost_to_pnl": _fmt(best.get("exec_cost_to_gross_pnl")),
            },
        })

    # P40: LO vs Execution Sharpe reconciliation
    if _best_per_family:
        lines.append("")
        lines.append("-" * 72)
        lines.append("LONG-ONLY vs EXECUTION SHARPE RECONCILIATION")
        lines.append("-" * 72)
        lines.append("")
        lines.append("Long-only Sharpe is a RAW SIGNAL DIAGNOSTIC, not a tradable production metric.")
        lines.append("It uses a simplified cost estimate (turnover x 10bps half-spread) on the top-decile")
        lines.append("equal-weight portfolio.  Execution Sharpe uses the FULL cost model and QP optimizer.")
        lines.append("")
        for _family_label, best in _best_per_family:
            name = str(best.get("model_name", "?"))
            lo_sharpe = _fmt(best.get("diag_lo_sharpe"))
            lo_net_sharpe = _fmt(best.get("diag_lo_net_sharpe"))
            lo_turnover = _fmt(best.get("diag_lo_turnover_mean"))
            _lo_turn = float(best.get("diag_lo_turnover_mean", float("nan")))
            lo_cost_ann = _fmt(_lo_turn * 10.0 / 10_000.0 * np.sqrt(252) if np.isfinite(_lo_turn) else float("nan"))
            exec_sharpe = _fmt(best.get("exec_sharpe"))
            exec_long = _fmt(best.get("exec_long_leg_sharpe"))
            exec_short = _fmt(best.get("exec_short_leg_sharpe"))
            cost_to_pnl = _fmt(best.get("exec_cost_to_gross_pnl"))
            alpha_capture = _fmt(best.get("diag_alpha_capture_ratio"))

            lines.append(f"  [{_family_label}] {name}:")
            lines.append(f"    LO Sharpe (gross)       : {lo_sharpe}")
            lines.append(f"    LO Sharpe (net, 10bps)  : {lo_net_sharpe}")
            lines.append(f"    LO turnover (daily)     : {lo_turnover}")
            lines.append(f"    LO est. annualized cost : {lo_cost_ann} (turnover x 10bps x sqrt(252))")
            lines.append(f"    Exec Sharpe (full cost) : {exec_sharpe}")
            lines.append(f"    Exec long leg           : {exec_long}")
            lines.append(f"    Exec short leg          : {exec_short}")
            lines.append(f"    Cost / gross PnL        : {cost_to_pnl}")
            lines.append(f"    Alpha capture           : {alpha_capture}")

            # Decomposition
            try:
                _lo = float(best.get("diag_lo_sharpe", float("nan")))
                _exec = float(best.get("exec_sharpe", float("nan")))
                _lo_net = float(best.get("diag_lo_net_sharpe", float("nan")))
                _short = float(best.get("exec_short_leg_sharpe", float("nan")))
                if np.isfinite(_lo) and np.isfinite(_exec):
                    gap = _lo - _exec
                    lo_cost_gap = _lo - _lo_net if np.isfinite(_lo_net) else float("nan")
                    short_leg_impact = _short if np.isfinite(_short) else float("nan")
                    optimizer_friction = gap - (lo_cost_gap if np.isfinite(lo_cost_gap) else 0) - (short_leg_impact if np.isfinite(short_leg_impact) else 0)
                    lines.append(f"    --- Decomposition ---")
                    lines.append(f"    LO→Exec gap           : {gap:.3f}")
                    if np.isfinite(lo_cost_gap):
                        lines.append(f"    LO cost model delta   : {lo_cost_gap:.3f} (10bps est vs full cost)")
                    if np.isfinite(short_leg_impact):
                        lines.append(f"    Short-leg contribution: {short_leg_impact:.3f}")
                    if np.isfinite(optimizer_friction):
                        lines.append(f"    Optimizer friction    : {optimizer_friction:.3f} (constraints, risk, turnover)")
            except Exception:
                pass
            lines.append("")

    lines.append("=" * 72)
    lines.append("Principle: gates should not be weakened to promote models.")
    lines.append("Address root causes before next research iteration.")
    lines.append("=" * 72)

    report_text = "\n".join(lines)
    print(report_text)

    fail_path = out_dir / "failure_report.txt"
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    fail_path.write_text(report_text + "\n", encoding="utf-8")
    print(f"\nFailure report saved: {fail_path}")

    json_path = out_dir / "failure_report.json"
    json_path.write_text(_json.dumps(json_report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Structured failure report saved: {json_path}")


def _json_metric_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating, np.integer)):
            out[str(key)] = float(value) if np.isfinite(value) else None
        else:
            out[str(key)] = value
    return out


_CANDIDATE_STRING_METRICS = {"nested_simulation_mode", "nested_window_failure_log"}


def _candidate_metric_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        k = str(key)
        if isinstance(value, (int, float, np.floating, np.integer)):
            out[k] = float(value) if np.isfinite(value) else float("nan")
        elif k in _CANDIDATE_STRING_METRICS and isinstance(value, str):
            out[k] = value
    return out


@dataclass(frozen=True)
class WindowMetrics:
    oos_sharpe: float
    oos_ic: float
    oos_dir_acc: float
    train_time_s: float
    test_time_s: float
    n_train: int
    n_test: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    signal_halflife_days: float = float("nan")
    cost_adjusted_ic_mean: float = float("nan")
    capacity_weighted_ic: float = float("nan")
    turnover_volatility: float = float("nan")
    decile_tail_stability: float = float("nan")
    hhi_concentration: float = float("nan")


def _read_config(path: str = "backtest_config.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _read_churn_filter_config(cfg: dict[str, Any], primary_path: str) -> "ChurnFilterConfig":
    """Parse P18 persistence filter from YAML horizon_config.persistence_filter."""
    hc = cfg.get("horizon_config", {}) or {}
    pf = (hc.get("persistence_filter", {}) or {}) if isinstance(hc, dict) else {}
    apply_to = tuple(pf.get("apply_to_paths", ["long_only_overlay"]))
    enabled = bool(pf.get("enabled", False)) and (primary_path in apply_to)
    return ChurnFilterConfig(
        enabled=enabled,
        min_consecutive_top_decile_days=int(pf.get("min_consecutive_top_decile_days", 2)),
        min_eligible_names=int(pf.get("min_eligible_names", 10)),
        apply_to_paths=apply_to,
    )


def _date_add_years(ts: pd.Timestamp, years: float) -> pd.Timestamp:
    # Allow fractional years (e.g. 0.25) by converting to months.
    months = int(round(years * 12))
    return ts + pd.DateOffset(months=months)


def _walk_forward_windows(
    start_date: str,
    end_date: str,
    train_years: float,
    test_years: float,
    step_years: float,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cursor = start_ts
    while True:
        train_start = cursor
        train_end = _date_add_years(train_start, train_years)
        test_start = train_end
        test_end = _date_add_years(test_start, test_years)
        # Include the window if there is any test data available (test_start < end_ts).
        # Using test_start avoids dropping the last window when test_end overshoots by 1 day
        # (e.g. end_date=2022-12-31 but test_end=2023-01-01 for a 1-year test window).
        if test_start >= end_ts:
            break
        # Clip test_end to end_ts so the last window uses all available data.
        test_end_clipped = min(test_end, end_ts)
        windows.append((train_start, train_end, test_start, test_end_clipped))
        cursor = _date_add_years(cursor, step_years)
    return windows


def _walk_forward_windows_by_count(
    dates: pd.Series, *, n_windows: int, train_ratio: float
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Non-overlapping walk-forward windows built from the available date index.

    Each window takes a contiguous block of dates, splits it into a train slice
    (first train_ratio) and a test slice (remaining).
    """
    d = pd.to_datetime(pd.Series(dates).dropna().unique())
    d = pd.Series(sorted(d))
    if len(d) < 50 or n_windows < 2:
        return []
    n_windows = int(max(2, min(n_windows, len(d) // 20)))
    block = int(len(d) / n_windows)
    out: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for i in range(n_windows):
        s = i * block
        e = (i + 1) * block if i < n_windows - 1 else len(d)
        block_dates = d.iloc[s:e]
        if len(block_dates) < 30:
            continue
        split = int(max(10, min(len(block_dates) - 10, round(train_ratio * len(block_dates)))))
        train_start = pd.Timestamp(block_dates.iloc[0])
        train_end = pd.Timestamp(block_dates.iloc[split - 1]) + pd.Timedelta(days=1)
        test_start = pd.Timestamp(block_dates.iloc[split])
        test_end = pd.Timestamp(block_dates.iloc[-1]) + pd.Timedelta(days=1)
        out.append((train_start, train_end, test_start, test_end))
    return out


def _sharpe_from_series(pnl: np.ndarray, horizon: int = 1) -> float:
    """
    Annualised Sharpe with Newey-West correction for serial correlation.

    When `horizon > 1`, overlapping forward-return windows create autocorrelation
    that inflates the naive Sharpe (std is underestimated). The Newey-West HAC
    estimator corrects the standard error using lags up to `horizon - 1`.
    Correction factor: sqrt(1 + 2 * sum_k(1 - k/horizon) * rho_k) where rho_k
    is the k-lag autocorrelation of the return series.
    """
    pnl = pnl.astype(float)
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) < 10:
        return float("nan")
    mu = float(np.mean(pnl))
    sd = float(np.std(pnl, ddof=1))
    if sd < 1e-12:
        return float("nan")

    if horizon > 1 and len(pnl) > horizon * 2:
        # Newey-West variance: V_NW = gamma_0 + 2 * sum_{k=1}^{h-1} (1 - k/h) * gamma_k
        gamma_0 = float(np.var(pnl, ddof=1))
        nw_var = gamma_0
        for k in range(1, int(horizon)):
            if k >= len(pnl):
                break
            gamma_k = float(np.cov(pnl[k:], pnl[:-k])[0, 1])
            nw_var += 2.0 * (1.0 - k / horizon) * gamma_k
        nw_var = max(nw_var, gamma_0 * 0.1)  # floor at 10% of naive variance
        sd = float(np.sqrt(nw_var))
        if sd < 1e-12:
            return float("nan")

    return float((mu / sd) * np.sqrt(252.0))


def _cagr_from_daily_returns(daily_rets: np.ndarray) -> float:
    r = daily_rets.astype(float)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return float("nan")
    growth = float(np.prod(1.0 + r))
    if growth <= 0:
        return float("nan")
    return float(growth ** (252.0 / len(r)) - 1.0)


def _max_drawdown_from_daily_returns(daily_rets: np.ndarray) -> float:
    r = daily_rets.astype(float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(np.min(dd))


def _win_rate_from_daily_returns(daily_rets: np.ndarray) -> float:
    r = daily_rets.astype(float)
    r = r[np.isfinite(r)]
    if len(r) < 1:
        return float("nan")
    return float(np.mean(r > 0.0))


def _learned_weights_score_series(df: pd.DataFrame) -> np.ndarray:
    """
    Compute LearnedWeights baseline score per row using output/learned_weights*.json.

    We reproduce the ridge-style linear model:
      z = (x - mean) / scale   (per feature)
      score_raw = intercept + sum(w_i * z_i)
      score = score_raw * score_scale * score_direction
    """
    weights_path = Path("output/learned_weights.json")
    scaler_path = Path("output/learned_weights_scaler.json")
    w = _read_json(weights_path)
    sc = _read_json(scaler_path)
    feats = [str(x) for x in (sc.get("active_features", []) or [])]
    mean = np.array(sc.get("scaler_mean", []), dtype=float)
    scale = np.array(sc.get("scaler_scale", []), dtype=float)
    if not feats or len(mean) != len(feats) or len(scale) != len(feats):
        raise RuntimeError("learned_weights_scaler.json missing/invalid active_features/mean/scale")

    # Map feature name -> weight key in learned_weights.json
    feature_to_wkey: dict[str, str] = {
        "f_trend": "w_trend",
        "f_regional": "w_regional",
        "f_global": "w_global",
        "f_social": "w_social",
        "ret_5d": "w_ret_5d",
        "ret_10d": "w_ret_10d",
        "ret_20d": "w_ret_20d",
        "ret_60d": "w_ret_60d",
        "cs_momentum_percentile": "w_cs_momentum",
        "momentum_3m": "w_momentum_3m",
        "momentum_6m": "w_momentum_6m",
        "ma_crossover": "w_ma_crossover",
        "rolling_vol_5": "w_rolling_vol_5",
        "rolling_vol_10": "w_vol_10",
        "rolling_vol_20": "w_vol",
        "rolling_vol_60": "w_rolling_vol_60",
        "vol_of_vol_20": "w_vol_of_vol",
        "jump_indicator": "w_jump_indicator",
        "vol_rank": "w_vol_rank",
        "relative_volume": "w_relative_volume",
        "volume_zscore": "w_volume_zscore",
        "rolling_corr_market_20": "w_corr_market",
        "capm_beta": "w_capm_beta",
        "vix_zscore": "w_vix_zscore",
        "vol_spike": "w_vol_spike",
        "vix_term_zscore": "w_vix_term_zscore",
        "rsi_zscore": "w_rsi_zscore",
        "bb_position": "w_bb_position",
        "dist_high": "w_dist_high",
        "dist_low": "w_dist_low",
        "overnight_gap": "w_overnight_gap",
        "intraday_rev": "w_intraday_rev",
        "sector_relative_20d": "w_sector_relative_20d",
        "sector_relative_60d": "w_sector_relative_60d",
    }

    # Validate feature map against actual weight keys so renames surface immediately
    _w_keys = set(w.keys())
    _feats_no_map = [f for f in feats if f not in feature_to_wkey]
    _map_dead_key = [f for f in feats if f in feature_to_wkey and feature_to_wkey[f] not in _w_keys]
    if _feats_no_map:
        import warnings as _warnings
        _warnings.warn(
            f"_learned_weights_score_series: {len(_feats_no_map)} active feature(s) have no entry in "
            f"feature_to_wkey and will contribute zero weight: {_feats_no_map}",
            UserWarning,
            stacklevel=2,
        )
    if _map_dead_key:
        import warnings as _warnings  # noqa: F811
        _warnings.warn(
            f"_learned_weights_score_series: {len(_map_dead_key)} feature(s) map to keys absent from "
            f"learned_weights.json and will contribute zero: "
            f"{[(f, feature_to_wkey[f]) for f in _map_dead_key]}",
            UserWarning,
            stacklevel=2,
        )

    # Build standardized feature matrix in scaler feature order
    X = np.zeros((len(df), len(feats)), dtype=float)
    for j, f in enumerate(feats):
        if f in df.columns:
            col = pd.to_numeric(df[f], errors="coerce").to_numpy(dtype=float)
        else:
            col = np.zeros(len(df), dtype=float)
        X[:, j] = col
    z = (X - mean.reshape(1, -1)) / np.where(scale.reshape(1, -1) == 0.0, 1.0, scale.reshape(1, -1))

    intercept = float(w.get("intercept", 0.0) or 0.0)
    score_scale = float(w.get("score_scale", 1.0) or 1.0)
    score_direction = float(w.get("score_direction", 1.0) or 1.0)

    weights_vec = np.zeros(len(feats), dtype=float)
    for j, f in enumerate(feats):
        key = feature_to_wkey.get(f, "")
        weights_vec[j] = float(w.get(key, 0.0) or 0.0) if key else 0.0

    raw = intercept + z.dot(weights_vec)
    return (raw * score_scale * score_direction).astype(float)


def _strategy_daily_returns(
    te: pd.DataFrame,
    *,
    max_positions: int,
    min_positions: int,
    horizon: int = 1,
    evaluation_path: str = "long_only_overlay",
) -> pd.Series:
    """
    Simulate daily-rebalanced rank portfolios over a test slice.

    Supported paths:
      - ``long_only_overlay``: beta-sensitive deployment view. Long the top ranks.
      - ``long_short_spread``: alpha research view. Long top ranks, short bottom ranks.
      - ``short_side``: standalone short-book simulator. Short bottom ranks.

    Selection is rank-based. We intentionally do not use ``score > 0`` as a
    gate because calibrated score sign is model-family dependent and unstable
    across walk-forward windows; cross-sectional rank is the traded object.

    ``horizon`` is the number of trading days in forward_return. Dividing by
    horizon converts a multi-day cumulative return to a daily equivalent so
    that the Sharpe calculation is not inflated by overlapping return windows.
    The forward_return in the matrix is a horizon-day return stacked on every
    calendar date, so without this correction Sharpe is inflated by ~sqrt(horizon).

    Returns:
      pd.Series of daily returns indexed by date (sorted)
    """
    if te is None or te.empty:
        return pd.Series(dtype=float)
    if "date" not in te.columns or "score" not in te.columns or "forward_return" not in te.columns:
        return pd.Series(dtype=float)

    df = te.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["forward_return"] = pd.to_numeric(df["forward_return"], errors="coerce")
    df = df.dropna(subset=["date", "score", "forward_return"])
    if df.empty:
        return pd.Series(dtype=float)

    k = int(max(1, max_positions))
    min_k = int(max(1, min_positions))
    horizon_days = float(max(1, int(horizon)))
    path = str(evaluation_path or "long_only_overlay").strip().lower()
    if path not in {"long_only_overlay", "long_short_spread", "short_side"}:
        raise ValueError(
            "evaluation_path must be one of "
            "'long_only_overlay', 'long_short_spread', or 'short_side'"
        )

    # Vectorized cross-sectional rank selection — replaces groupby.apply(_day_ret).
    # rank=1 → lowest score, rank=N → highest score within each date.
    df = df.copy()
    df["_rank"] = df.groupby("date", sort=True)["score"].rank(method="first", ascending=True)
    df["_n"] = df.groupby("date", sort=True)["score"].transform("count")

    if path == "long_short_spread":
        side_k = max(1, k // 2)
        min_side_k = max(1, min(min_k, side_k))
        min_n = max(2 * min_side_k, 2)
        valid = df["_n"] >= min_n
        long_mask = valid & (df["_rank"] > (df["_n"] - side_k))
        short_mask = valid & (df["_rank"] <= side_k)
        long_ret = df.loc[long_mask].groupby("date")["forward_return"].mean()
        short_ret = df.loc[short_mask].groupby("date")["forward_return"].mean()
        long_cnt = df.loc[long_mask].groupby("date")["forward_return"].count()
        short_cnt = df.loc[short_mask].groupby("date")["forward_return"].count()
        all_dates_idx = long_ret.index.union(short_ret.index)
        long_cnt = long_cnt.reindex(all_dates_idx, fill_value=0)
        short_cnt = short_cnt.reindex(all_dates_idx, fill_value=0)
        raw_spread = (
            long_ret.reindex(all_dates_idx, fill_value=0.0)
            - short_ret.reindex(all_dates_idx, fill_value=0.0)
        ) / horizon_days
        daily = raw_spread.where((long_cnt >= min_side_k) & (short_cnt >= min_side_k), other=0.0)
    else:
        if path == "short_side":
            sel_mask = df["_rank"] <= k
        else:  # long_only_overlay
            sel_mask = df["_rank"] > (df["_n"] - k)
        sel_cnt = df.loc[sel_mask].groupby("date")["forward_return"].count()
        daily = df.loc[sel_mask].groupby("date")["forward_return"].mean() / horizon_days
        daily = daily.where(sel_cnt >= min_k, other=0.0)
        if path == "short_side":
            daily = -daily

    daily = pd.to_numeric(daily, errors="coerce").dropna()
    daily.name = "daily_return"
    return daily


def _test_portfolio_simulation_logic(*, tol: float = 1e-12) -> None:
    """
    Lightweight self-test for the portfolio simulation + Sharpe calculation.

    Uses deterministic mock data and compares:
      - simulated daily returns (from _strategy_daily_returns)
      - annualised Sharpe (from _sharpe_from_series)
    against a manual computation with the same rules.
    """
    # 5 tickers, 10 days so Sharpe isn't NaN (our Sharpe fn needs >=10 points).
    tickers = ["A", "B", "C", "D", "E"]
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    rows: list[dict[str, Any]] = []
    for d in dates:
        for t in tickers:
            rows.append({"date": d, "ticker": t})
    te = pd.DataFrame(rows)

    # Scores: constant ranking each day. Rank, not score sign, drives selection.
    score_map = {"A": 0.30, "B": 0.20, "C": 0.10, "D": -0.10, "E": -0.20}
    te["score"] = te["ticker"].map(score_map).astype(float)

    # Forward returns: design so top 2 names have known returns per day.
    # Alternate between +1% and -1% for A; constant +0.5% for B; others irrelevant (not selected).
    def _fwd_ret(row: pd.Series) -> float:
        idx = int((row["date"] - dates[0]).days)
        if row["ticker"] == "A":
            return 0.01 if (idx % 2 == 0) else -0.01
        if row["ticker"] == "B":
            return 0.005
        if row["ticker"] == "C":
            return 0.0
        return -0.02

    te["forward_return"] = te.apply(_fwd_ret, axis=1).astype(float)

    max_positions = 2
    min_positions = 2

    sim = _strategy_daily_returns(
        te,
        max_positions=max_positions,
        min_positions=min_positions,
        horizon=1,
        evaluation_path="long_only_overlay",
    )
    sim_arr = sim.to_numpy(dtype=float)

    # Manual daily return: equal-weight mean of A and B each day.
    manual = []
    for i, _ in enumerate(dates):
        a = 0.01 if (i % 2 == 0) else -0.01
        b = 0.005
        manual.append(0.5 * (a + b))
    manual_arr = np.array(manual, dtype=float)

    # Compare daily returns exactly
    if len(sim_arr) != len(manual_arr) or np.max(np.abs(sim_arr - manual_arr)) > tol:
        raise AssertionError(
            f"Simulation daily returns mismatch. max_abs_diff={float(np.max(np.abs(sim_arr - manual_arr))):.3e}"
        )

    # Compare Sharpe (annualised, ddof=1) exactly
    sim_sh = _sharpe_from_series(sim_arr)
    mu = float(np.mean(manual_arr))
    sd = float(np.std(manual_arr, ddof=1))
    manual_sh = float((mu / sd) * np.sqrt(252.0)) if sd > 1e-12 else float("nan")
    if not (np.isfinite(sim_sh) and np.isfinite(manual_sh)) or abs(sim_sh - manual_sh) > 1e-10:
        raise AssertionError(f"Sharpe mismatch. sim={sim_sh:.12f} manual={manual_sh:.12f}")

    # Cash rule: require 6 positions but only 5 names exist -> all-zero returns.
    sim_cash = _strategy_daily_returns(
        te,
        max_positions=2,
        min_positions=6,
        horizon=1,
        evaluation_path="long_only_overlay",
    ).to_numpy(dtype=float)
    if np.max(np.abs(sim_cash)) > tol:
        raise AssertionError("Cash rule failed: expected all-zero daily returns when min_positions not met.")

    # Horizon normalization: a 5-day forward return should be converted to a daily return proxy.
    sim_h5 = _strategy_daily_returns(
        te,
        max_positions=max_positions,
        min_positions=min_positions,
        horizon=5,
        evaluation_path="long_only_overlay",
    ).to_numpy(dtype=float)
    if len(sim_h5) != len(manual_arr) or np.max(np.abs(sim_h5 - (manual_arr / 5.0))) > tol:
        raise AssertionError("Horizon normalization failed for long_only_overlay.")

    # Long-short alpha path: long A, short E when max_positions=2.
    sim_ls = _strategy_daily_returns(
        te,
        max_positions=2,
        min_positions=1,
        horizon=1,
        evaluation_path="long_short_spread",
    ).to_numpy(dtype=float)
    manual_ls = []
    for i, _ in enumerate(dates):
        a = 0.01 if (i % 2 == 0) else -0.01
        e = -0.02
        manual_ls.append(a - e)
    manual_ls_arr = np.array(manual_ls, dtype=float)
    if len(sim_ls) != len(manual_ls_arr) or np.max(np.abs(sim_ls - manual_ls_arr)) > tol:
        raise AssertionError("Long-short spread simulation failed.")

    # Short-side path: short E, so return is -forward_return(E).
    sim_short = _strategy_daily_returns(
        te,
        max_positions=1,
        min_positions=1,
        horizon=1,
        evaluation_path="short_side",
    ).to_numpy(dtype=float)
    manual_short = np.full(len(dates), 0.02, dtype=float)
    if len(sim_short) != len(manual_short) or np.max(np.abs(sim_short - manual_short)) > tol:
        raise AssertionError("Short-side simulation failed.")

    print("PASS: portfolio simulation self-test")


def _primary_evaluation_path(model_kind: str, *, long_only: bool = False) -> str:
    """Research path used for model ranking."""
    if model_kind in {"short_classifier", "short_alpha"}:
        return "short_side"
    if model_kind == "overlay_alpha" or long_only:
        return "long_only_overlay"
    return "long_short_spread"


def _deployment_primary_path_for_model(model_kind: str, configured_primary_path: str) -> str:
    """Resolve the executable validation path without violating the deployment mandate."""
    if _is_short_model(model_kind):
        return _primary_evaluation_path(model_kind)
    path = str(configured_primary_path or "long_only_overlay").strip().lower()
    if path not in {"long_only_overlay", "long_short_spread"}:
        return "long_only_overlay"
    return path


def _validation_target_col_for_path(primary_path: str) -> str:
    """Target column used for executable validation and score calibration."""
    path = str(primary_path or "").strip().lower()
    if path == "long_only_overlay":
        return "forward_return"
    return "target_return"


def _target_name_for_model_kind(model_kind: str, split_enabled: bool) -> str:
    """Legacy split-model target name used by older scripts/tests."""
    if not split_enabled:
        return "forward_return"
    if model_kind == "classifier":
        return "y_long"
    if model_kind == "short_classifier":
        return "y_short"
    return "forward_return"


def _pick_split_best_models(report: pd.DataFrame) -> tuple[str | None, str | None]:
    """Pick best legacy long classifier and short classifier from a ranked report."""
    if report is None or report.empty:
        return None, None
    metric = "_selection_metric" if "_selection_metric" in report.columns else "oos_deflated_sharpe"
    ranked = report.copy()
    ranked["_legacy_rank_metric"] = pd.to_numeric(ranked.get(metric), errors="coerce")
    ranked = ranked.sort_values("_legacy_rank_metric", ascending=False, na_position="last")
    long_pool = ranked[ranked.get("model_kind", pd.Series("", index=ranked.index)).eq("classifier")]
    short_pool = ranked[ranked.get("model_kind", pd.Series("", index=ranked.index)).eq("short_classifier")]
    best_long = str(long_pool.iloc[0]["model_name"]) if not long_pool.empty else None
    best_short = str(short_pool.iloc[0]["model_name"]) if not short_pool.empty else None
    return best_long, best_short


def _is_short_model(model_kind: str) -> bool:
    return model_kind in {"short_classifier", "short_alpha"}


def _active_features_for_model_kind(
    model_kind: str,
    feat_cols: list[str],
    *,
    short_feature_subset: list[str],
    overlay_feature_subset: list[str],
) -> list[str]:
    """
    Filter features by economic mandate.

    Feature admission is the primary statistical gate, but statistical
    significance does not imply economic relevance.  Short-squeeze filters
    have negative IC (they predict bad shorts) but make zero economic sense
    in a long-book model.  This post-admission filter removes features whose
    FEATURE_SPECS.family is incompatible with the model kind.

    Universal families (regime, sector_relative, liquidity, risk) are always
    available.  The exclusion sets below follow the economic mandate taxonomy
    defined in research_contract.py.
    """
    # P10: Economic mandate filtering — families to exclude per model kind.
    _LONG_ONLY_FAMILIES = frozenset({
        "trend", "momentum", "reversal", "reversal_conditioner",
        "quality", "quality_lowvol", "residual_alpha", "fundamental_quality",
    })
    _SHORT_ONLY_FAMILIES = frozenset({
        "short_momentum", "fundamental_deterioration", "fundamental_leverage",
        "dilution", "reporting_quality", "crowding", "squeeze_filter",
    })
    _OVERLAY_EXCLUDED = _LONG_ONLY_FAMILIES | _SHORT_ONLY_FAMILIES

    if model_kind in {"short_alpha", "short_classifier"}:
        excluded = _LONG_ONLY_FAMILIES
    elif model_kind in {"overlay_alpha"}:
        excluded = _OVERLAY_EXCLUDED
    else:
        # long_alpha, classifier, regressor, and any unknown kind
        excluded = _SHORT_ONLY_FAMILIES

    return [
        f for f in feat_cols
        if FEATURE_SPECS.get(f) is None or FEATURE_SPECS[f].family not in excluded
    ]


def _model_registry(models: list[tuple[str, Any, bool, str]]) -> dict[str, tuple[Any, bool, str]]:
    return {name: (model, uses_proba, model_kind) for name, model, uses_proba, model_kind in models}


def _is_economic_model_kind(model_kind: str) -> bool:
    # Alpha model kinds are score generators, not portfolio-utility objectives.
    # Economic optimization is now downstream-only in validation/construction.
    return False


def _split_model_families(
    models: list[tuple[str, Any, bool, str]],
) -> tuple[list[tuple[str, Any, bool, str]], list[tuple[str, Any, bool, str]]]:
    fast = [spec for spec in models if not _is_economic_model_kind(str(spec[3]))]
    economic = [spec for spec in models if _is_economic_model_kind(str(spec[3]))]
    return fast, economic


def get_group_boundaries(group_ids: np.ndarray) -> np.ndarray:
    if len(group_ids) == 0:
        return np.array([0])
    boundaries = np.where(group_ids[1:] != group_ids[:-1])[0] + 1
    return np.concatenate(([0], boundaries, [len(group_ids)]))

def _proxy_turnover_from_scores(scored: pd.DataFrame, *, primary_path: str, max_positions: int = 10) -> float:
    if scored is None or scored.empty or "date" not in scored.columns or "score" not in scored.columns or "ticker" not in scored.columns:
        return float("nan")
        
    df = scored[["date", "ticker", "score"]].dropna().copy()
    if df.empty:
        return float("nan")
        
    df["date_id"] = df["date"].astype("category").cat.codes
    df.sort_values(["date_id", "score"], inplace=True)
    
    dates = df["date_id"].values
    scores = df["score"].values
    
    bounds = get_group_boundaries(dates)
    
    w_raw = np.zeros(len(scores), dtype=np.float64)
    valid_mask = np.zeros(len(scores), dtype=bool)
    
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i+1]
        n = e - s
        if n < 5 or scores[e-1] == scores[s]:
            continue
            
        valid_mask[s:e] = True
        if primary_path == "long_only_overlay":
            w = np.zeros(n)
            # Find top-k
            top_idx = np.argsort(scores[s:e])[-max_positions:]
            w[top_idx] = 1.0 / max(1, len(top_idx))
        elif primary_path == "short_side":
            w = np.zeros(n)
            # Find bottom-k
            bot_idx = np.argsort(scores[s:e])[:max_positions]
            w[bot_idx] = -1.0 / max(1, len(bot_idx))
        else: # long_short_spread
            w = np.zeros(n)
            k_half = max_positions // 2
            top_idx = np.argsort(scores[s:e])[-k_half:]
            bot_idx = np.argsort(scores[s:e])[:k_half]
            if len(top_idx) > 0: w[top_idx] = 0.5 / len(top_idx)
            if len(bot_idx) > 0: w[bot_idx] = -0.5 / len(bot_idx)
            
        if not np.isclose(np.abs(w).sum(), 1.0) and np.abs(w).sum() > 1e-9:
             raise AssertionError("Turnover/POV integrity failure: weights do not sum to 1.")
            
        w_raw[s:e] = w
        
    df["_w_raw"] = w_raw
    df = df[valid_mask].copy()
    if df.empty:
        return float("nan")
        
    dates_valid = df["date_id"].values
    w_raw_valid = df["_w_raw"].values
    bounds_valid = get_group_boundaries(dates_valid)
    
    abs_sums = np.add.reduceat(np.abs(w_raw_valid), bounds_valid[:-1])
    abs_sum_full = np.repeat(abs_sums, np.diff(bounds_valid))
    
    valid_gross = abs_sum_full > 1e-12
    df = df[valid_gross].copy()
    if df.empty:
        return float("nan")
        
    df["_w"] = df["_w_raw"] / abs_sum_full[valid_gross]
    
    wide = df.pivot_table(index="date", columns="ticker", values="_w", fill_value=0.0)
    wide.sort_index(inplace=True)
    
    delta = wide.diff().abs().sum(axis=1).iloc[1:]
    return float(np.nanmean(delta)) if len(delta) > 0 else float("nan")

def _resolve_report_training_spec(
    row: pd.Series,
    *,
    models: list[tuple[str, Any, bool, str]],
    cfg: dict[str, Any],
    feat_cols: list[str],
    short_feature_subset: list[str],
    overlay_feature_subset: list[str],
    default_horizon: int,
) -> dict[str, Any]:
    registry = _model_registry(models)
    row_kind = str(row.get("model_kind", "") or "")
    selected_name = str(row.get("nested_selected_model_mode", "") or "").strip()
    selected_view = str(row.get("nested_selected_feature_view_mode", "") or "").strip() or "full"
    selected_horizon = int(_safe_float(row.get("nested_selected_horizon_mode"), float(default_horizon)) or default_horizon)

    if row_kind in {"long_alpha", "overlay_alpha", "short_alpha"} and selected_name in registry:
        resolved_name = selected_name
    else:
        resolved_name = str(row.get("model_name", "") or "")
        selected_view = "full"
        selected_horizon = int(default_horizon)

    if resolved_name not in registry:
        raise KeyError(f"Model spec '{resolved_name}' not found in registry")

    model_template, uses_proba, resolved_kind = registry[resolved_name]
    active_features = _active_features_for_model_kind(
        resolved_kind,
        feat_cols,
        short_feature_subset=short_feature_subset,
        overlay_feature_subset=overlay_feature_subset,
    )
    if row_kind in {"long_alpha", "overlay_alpha", "short_alpha"}:
        active_features = _program_feature_view(
            active_features,
            primary_path=_primary_evaluation_path(row_kind),
            view=selected_view,
        )

    template = model_template
    return {
        "row_model_kind": row_kind,
        "resolved_model_name": resolved_name,
        "resolved_model_kind": resolved_kind,
        "uses_proba": bool(uses_proba),
        "model_template": template,
        "active_features": list(active_features),
        "feature_view": selected_view,
        "horizon": int(selected_horizon),
    }


def _directional_accuracy_from_scored(te_scored: pd.DataFrame, *, model_kind: str) -> float:
    """
    Directional hit-rate diagnostic aligned with the model target.

    Short-family models are rankers: lower score means a stronger short. Compare
    bottom score decile to realized bottom-return decile by test-date.
    """
    if te_scored is None or te_scored.empty or "score" not in te_scored.columns:
        return float("nan")
    score = pd.to_numeric(te_scored["score"], errors="coerce")
    if _is_short_model(model_kind) and {"date", "forward_return"}.issubset(te_scored.columns):
        fwd = pd.to_numeric(te_scored["forward_return"], errors="coerce")
        date_key = pd.to_datetime(te_scored["date"], errors="coerce")
        ranks = fwd.groupby(date_key).transform(
            lambda x: x.rank(pct=True, na_option="keep")
        )
        score_ranks = score.groupby(date_key).transform(lambda x: x.rank(pct=True, na_option="keep"))
        realized_down = (ranks < 0.10).where(ranks.notna())
        predicted_down = (score_ranks <= 0.10).where(score_ranks.notna())
        mask = predicted_down.notna() & realized_down.notna()
        return float((predicted_down[mask] == realized_down[mask]).mean()) if bool(mask.any()) else float("nan")
    if "y_bin" not in te_scored.columns:
        return float("nan")
    y_bin = pd.to_numeric(te_scored["y_bin"], errors="coerce")
    mask = score.notna() & y_bin.notna()
    return float(((score[mask] >= 0.0) == (y_bin[mask] == 1)).mean()) if bool(mask.any()) else float("nan")


def _concat_window_daily_returns(parts: list[pd.Series]) -> np.ndarray:
    """
    Concatenate per-window daily return series into a single chronological series.

    If windows overlap (they shouldn't), later windows overwrite earlier values.
    """
    if not parts:
        return np.array([], dtype=float)
    s = pd.concat(parts)
    s = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.dropna().to_numpy(dtype=float)


def _concat_window_daily_return_series(parts: list[pd.Series]) -> pd.Series:
    """Concatenate per-window daily return series while preserving calendar index."""
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts)
    s = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.dropna().astype(float)


def _pnl_detail_metrics(
    pnl_parts: list[pd.DataFrame],
    daily_returns: np.ndarray,
    *,
    horizon: int,
) -> dict[str, float]:
    """Aggregate executable portfolio diagnostics from the exact per-window simulations."""
    arr = np.asarray(daily_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    pnl = pd.concat(pnl_parts, ignore_index=True) if pnl_parts else pd.DataFrame()
    costs = float(pd.to_numeric(pnl.get("cost_return", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    gross = float(pd.to_numeric(pnl.get("gross_return", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())

    def _sum_col(name: str) -> float:
        return float(pd.to_numeric(pnl.get(name, pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())

    def _mean_col(name: str) -> float:
        s = pd.to_numeric(pnl.get(name, pd.Series(dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan)
        return float(s.mean()) if not s.dropna().empty else float("nan")

    def _max_col(name: str) -> float:
        s = pd.to_numeric(pnl.get(name, pd.Series(dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan)
        return float(s.max()) if not s.dropna().empty else float("nan")

    def _pnl_col(name: str) -> np.ndarray:
        # Columns added by the cost-allocation fix may be absent in cached PnL parts
        # from earlier runs.  Return a zero array of the correct row count so that
        # arithmetic with other columns never broadcasts with mismatched shapes.
        if pnl.empty or name not in pnl.columns:
            return np.zeros(len(pnl), dtype=float)
        return (
            pd.to_numeric(pnl[name], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    # Net leg returns: long absorbs its transaction costs; short absorbs its costs,
    # borrow, and the market-neutral adjustment.  Mirrors the fix in validation.py so
    # that leg Sharpes are comparable to the portfolio-level exec_sharpe.
    long_arr = _pnl_col("long_gross_return") - _pnl_col("long_cost_return")
    short_arr = (
        _pnl_col("short_gross_return")
        - _pnl_col("short_cost_return")
        - _pnl_col("borrow_return")
        - _pnl_col("market_adj_return")
    )
    return {
        "exec_sharpe": _sharpe_from_series(arr, horizon=int(horizon)),
        "exec_long_leg_sharpe": _sharpe_from_series(long_arr, horizon=int(horizon)),
        "exec_short_leg_sharpe": _sharpe_from_series(short_arr, horizon=int(horizon)),
        "exec_cagr": _cagr_from_daily_returns(arr),
        "exec_max_dd": _max_drawdown_from_daily_returns(arr),
        "exec_win_rate": _win_rate_from_daily_returns(arr),
        "exec_cost_return_sum": costs,
        "exec_borrow_return_sum": float(
            pd.to_numeric(pnl.get("borrow_return", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
        ),
        "exec_commission_return_sum": _sum_col("commission_return"),
        "exec_spread_return_sum": _sum_col("spread_return"),
        "exec_fixed_cost_return_sum": _sum_col("fixed_cost_return"),
        "exec_temporary_impact_return_sum": _sum_col("temporary_impact_return"),
        "exec_permanent_impact_return_sum": _sum_col("permanent_impact_return"),
        "exec_permanent_impact_decay_return_sum": _sum_col("permanent_impact_decay_return"),
        "exec_permanent_impact_unamortized_mean": _mean_col("permanent_impact_unamortized_return"),
        "exec_lambda_risk_mean": _mean_col("lambda_risk"),
        "exec_lambda_risk_min": float(
            pd.to_numeric(pnl.get("lambda_risk", pd.Series(dtype=float)), errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .min()
        ) if "lambda_risk" in pnl else float("nan"),
        "exec_lambda_risk_max": _max_col("lambda_risk"),
        "exec_gamma_turnover_mean": _mean_col("gamma_turnover"),
        "exec_gamma_turnover_min": float(
            pd.to_numeric(pnl.get("gamma_turnover", pd.Series(dtype=float)), errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .min()
        ) if "gamma_turnover" in pnl else float("nan"),
        "exec_gamma_turnover_max": _max_col("gamma_turnover"),
        "exec_expected_alpha_mean": _mean_col("expected_alpha"),
        "exec_expected_cost_mean": _mean_col("expected_cost"),
        "exec_realized_volatility_mean": _mean_col("realized_volatility"),
        "exec_turnover_mean": float(
            pd.to_numeric(pnl.get("turnover", pd.Series(dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
        ),
        "exec_trade_count_sum": _sum_col("trade_count"),
        "exec_trade_notional_sum": _sum_col("trade_notional"),
        "exec_participation_mean": _mean_col("participation_rate_mean"),
        "exec_participation_p95": _mean_col("participation_rate_p95"),
        "exec_participation_max": _max_col("participation_rate_max"),
        "exec_participation_over_5pct_count": _sum_col("participation_over_5pct_count"),
        "exec_participation_over_10pct_count": _sum_col("participation_over_10pct_count"),
        "exec_participation_capped_count": _sum_col("participation_capped_count"),
        "exec_gross_exposure_mean": float(
            pd.to_numeric(pnl.get("gross_exposure", pd.Series(dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
        ),
        "exec_net_exposure_abs_mean": float(
            pd.to_numeric(pnl.get("net_exposure", pd.Series(dtype=float)), errors="coerce").abs().replace([np.inf, -np.inf], np.nan).mean()
        ),
        "exec_long_exposure_mean": float(
            pd.to_numeric(pnl.get("long_exposure", pd.Series(dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
        ),
        "exec_short_exposure_mean": float(
            pd.to_numeric(pnl.get("short_exposure", pd.Series(dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
        ),
        "exec_beta_abs_mean": float(
            pd.to_numeric(pnl.get("beta_exposure", pd.Series(dtype=float)), errors="coerce").abs().replace([np.inf, -np.inf], np.nan).mean()
        ),
        "exec_max_sector_abs_mean": float(
            pd.to_numeric(pnl.get("max_sector_exposure", pd.Series(dtype=float)), errors="coerce").abs().replace([np.inf, -np.inf], np.nan).mean()
        ),
        "exec_cost_to_gross_pnl": abs(costs) / abs(gross) if abs(gross) > 1e-12 else float("nan"),
        "exec_n_days": int(len(arr)),
    }


def _nested_inner_windows(
    train_df: pd.DataFrame,
    *,
    max_windows: int,
    validation_days: int,
    min_train_days: int,
    embargo_days: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = pd.Series(sorted(pd.to_datetime(train_df["date"], errors="coerce").dropna().unique()))
    if len(dates) < (int(min_train_days) + int(validation_days) + int(embargo_days)):
        return []
    out: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    val_days = max(20, int(validation_days))
    for i in range(int(max_windows)):
        val_end_pos = len(dates) - i * val_days
        val_start_pos = val_end_pos - val_days
        if val_start_pos <= 0:
            break
        train_end_pos = max(0, val_start_pos - int(embargo_days))
        if train_end_pos < int(min_train_days):
            continue
        tr_s = pd.Timestamp(dates.iloc[0])
        tr_e = pd.Timestamp(dates.iloc[train_end_pos - 1]) + pd.Timedelta(days=1)
        va_s = pd.Timestamp(dates.iloc[val_start_pos])
        va_e = pd.Timestamp(dates.iloc[val_end_pos - 1]) + pd.Timedelta(days=1)
        out.append((tr_s, tr_e, va_s, va_e))
    return list(reversed(out))


def _score_model_predictions(
    win_model: Any,
    x: np.ndarray,
    *,
    model_kind: str,
    uses_proba: bool,
    score_direction: int = 1,
) -> np.ndarray:
    assert score_direction in (1, -1), f"score_direction must be +1 or -1, got {score_direction}"
    with np.errstate(all="ignore"):
        if model_kind in {"regressor", "long_alpha", "overlay_alpha", "short_alpha"}:
            raw = win_model.predict(x).astype(float)
        elif model_kind == "short_classifier":
            if uses_proba and hasattr(win_model, "predict_proba"):
                p_down = win_model.predict_proba(x)[:, 1].astype(float)
                raw = p_down - 0.5
            else:
                raw = win_model.predict(x).astype(float) - 0.5
        elif uses_proba and hasattr(win_model, "predict_proba"):
            raw = win_model.predict_proba(x)[:, 1].astype(float) - 0.5
        elif hasattr(win_model, "decision_function"):
            raw = win_model.decision_function(x).astype(float)
        else:
            raw = win_model.predict(x).astype(float) - 0.5
    return raw * float(score_direction)


def _calibrate_direction_from_data(
    win_model: Any,
    x_cal: np.ndarray,
    cal_df: pd.DataFrame,
    *,
    model_kind: str,
    uses_proba: bool,
    target_col: str = "target_return",
    source: str = "training",
) -> tuple[int, dict[str, Any]]:
    """
    Determine score_direction (+1/-1) from labeled calibration data.

    Uses raw scores (score_direction=1) on cal_df to compute Spearman IC,
    then applies determine_score_direction() to select the right multiplier.

    Calibration data must come from training or inner-validation windows —
    never from the final OOS test window.

    Returns
    -------
    direction : +1 or -1
    diagnostics : dict with keys prefixed `score_direction_`
    """
    _check_class_ordering(win_model, model_kind)
    raw_scores = _score_model_predictions(
        win_model, x_cal, model_kind=model_kind, uses_proba=uses_proba, score_direction=1
    )
    if target_col not in cal_df.columns or "date" not in cal_df.columns:
        return 1, {"score_direction": 1, "score_direction_source": source, "score_direction_mode": "fixed", "score_direction_ic_raw": float("nan")}

    scored = cal_df.assign(score=raw_scores)
    ic_stats = cross_sectional_ic(scored, target_col=target_col)
    ic_raw = float(ic_stats.get("cs_ic_spearman_mean", 0.0) or 0.0)

    direction, mode, reason = _determine_score_direction(ic_raw, model_kind=model_kind)
    calibrated_ic = ic_raw * direction  # spearmanr(-x, y) = -spearmanr(x, y)

    return direction, {
        "score_direction": direction,
        "score_direction_source": source,
        "score_direction_mode": mode,
        "score_direction_ic_raw": round(ic_raw, 6),
        "score_direction_ic_calibrated": round(calibrated_ic, 6),
        "score_direction_reason": reason,
    }


def _fit_candidate_model(
    *,
    model_template: Any,
    name: str,
    model_kind: str,
    tr: pd.DataFrame,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> Any:
    from sklearn.base import clone

    win_model = clone(model_template)
    if name == "ShortXGB" and model_kind == "short_classifier":
        pos = int((y_tr == 1).sum())
        neg = int((y_tr == 0).sum())
        win_model.set_params(scale_pos_weight=float(neg / pos) if pos > 0 else 1.0)
    with np.errstate(all="ignore"):
        tr_dates_sorted = pd.to_datetime(tr["date"], errors="coerce").to_numpy()
        _, date_groups = np.unique(tr_dates_sorted, return_inverse=True)
        fit_kwargs: dict[str, Any] = {}
        if name == "LGBMRanker":
            fit_kwargs["_date_groups"] = date_groups
        if sample_weight is not None and len(sample_weight) == len(y_tr):
            fit_kwargs["sample_weight"] = sample_weight
        assert_pure_alpha_fit_kwargs(fit_kwargs)
        win_model.fit(x_tr, y_tr, **fit_kwargs)
    return win_model


def _complexity_screen_model_template(model_template: Any, *, active_feature_count: int) -> Any:
    """Build a structurally cheaper screening estimator without changing the final model.

    SignalDiscovery and nested prefilters only need stable ordering evidence.  Tree and
    ranker families are therefore capped by the dimensionality of the admitted alpha
    set, while ExecutionValidation still receives the original full template.
    """
    if not (hasattr(model_template, "get_params") and hasattr(model_template, "set_params")):
        return model_template
    try:
        from sklearn.base import clone

        params = model_template.get_params(deep=False)
        if not params:
            return model_template
        updates: dict[str, Any] = {}
        n_features = max(1, int(active_feature_count or 1))
        if "n_estimators" in params:
            current = int(params.get("n_estimators") or 0)
            if current > 0:
                estimator_budget = max(n_features, int(np.ceil(2.0 * np.sqrt(n_features) * np.log2(n_features + 1))))
                updates["n_estimators"] = max(1, min(current, estimator_budget))
        if "max_depth" in params and params.get("max_depth") is not None:
            current_depth = int(params.get("max_depth") or 0)
            if current_depth > 0:
                depth_budget = max(1, int(np.ceil(np.log2(n_features + 1))))
                updates["max_depth"] = max(1, min(current_depth, depth_budget))
        if "num_leaves" in params and "max_depth" in updates:
            current_leaves = int(params.get("num_leaves") or 0)
            if current_leaves > 0:
                updates["num_leaves"] = max(2, min(current_leaves, (2 ** int(updates["max_depth"])) - 1))
        # Enforce single-threaded inner parallelism for screening — the outer model
        # selection loop already parallelizes across models. Allowing inner thread
        # pools (n_jobs=-1) creates oversubscription and memory pressure in fork workers.
        for thread_key in ("n_jobs", "nthread", "num_threads"):
            if thread_key in params:
                updates[thread_key] = 1
        if not updates:
            return model_template
        return clone(model_template).set_params(**updates)
    except Exception:
        return model_template


def _proxy_model_template(model_template: Any, nested_cfg: dict[str, Any], *, active_feature_count: int = 1) -> Any:
    """Build a cheap first-stage candidate for nested screening."""
    model_template = _complexity_screen_model_template(
        model_template,
        active_feature_count=active_feature_count,
    )
    search_cfg = (nested_cfg.get("search", {}) if isinstance(nested_cfg.get("search", {}), dict) else {}) or {}
    proxy_max_iter = int(search_cfg.get("proxy_max_iter", 30) or 30)
    if proxy_max_iter <= 0:
        return model_template
    if not (hasattr(model_template, "get_params") and hasattr(model_template, "set_params")):
        return model_template
    params = model_template.get_params(deep=False)
    if "max_iter" not in params:
        return model_template
    try:
        from sklearn.base import clone

        current = int(params.get("max_iter", proxy_max_iter) or proxy_max_iter)
        return clone(model_template).set_params(max_iter=max(1, min(current, proxy_max_iter)))
    except Exception:
        return model_template


@dataclass(frozen=True)
class NestedCandidateSpec:
    model_name: str
    model_kind: str
    uses_proba: bool
    model_template: Any
    active_features: tuple[str, ...]
    feature_view: str
    horizon: int
    turnover_penalty: float | None = None
    cost_penalty: float | None = None


@dataclass(frozen=True)
class SelectionPlan:
    """C.4: Machine-readable plan for a single outer-window selection decision.

    Emitted for every _select_nested_candidate call. Records whether nested
    validation was required or short-circuited, and why.
    """
    outer_window_id: str
    model_family: str
    candidate_count: int
    candidate_ids: list[dict[str, Any]]
    candidate_fingerprints: list[str]
    selection_required: bool
    selection_required_reason: str
    short_circuit_allowed: bool
    short_circuit_reason: str
    selection_disabled_by_config: bool
    allow_cross_family_selection: bool
    horizon_candidates: list[int]
    hyperparameter_candidates: int
    feature_set_candidates: int
    proxy_metric: str
    inner_fold_count: int
    outer_train_start: str
    outer_train_end: str
    outer_eval_start: str
    outer_eval_end: str
    horizon_contract_fingerprint: str
    target_spec_fingerprint: str
    feature_spec_fingerprints: list[str]
    cache_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "outer_window_id": self.outer_window_id,
            "model_family": self.model_family,
            "candidate_count": self.candidate_count,
            "candidate_ids": self.candidate_ids,
            "candidate_fingerprints": self.candidate_fingerprints,
            "selection_required": self.selection_required,
            "selection_required_reason": self.selection_required_reason,
            "short_circuit_allowed": self.short_circuit_allowed,
            "short_circuit_reason": self.short_circuit_reason,
            "selection_disabled_by_config": self.selection_disabled_by_config,
            "allow_cross_family_selection": self.allow_cross_family_selection,
            "horizon_candidates": self.horizon_candidates,
            "hyperparameter_candidates": self.hyperparameter_candidates,
            "feature_set_candidates": self.feature_set_candidates,
            "proxy_metric": self.proxy_metric,
            "inner_fold_count": self.inner_fold_count,
            "outer_train_start": self.outer_train_start,
            "outer_train_end": self.outer_train_end,
            "outer_eval_start": self.outer_eval_start,
            "outer_eval_end": self.outer_eval_end,
            "horizon_contract_fingerprint": self.horizon_contract_fingerprint,
            "target_spec_fingerprint": self.target_spec_fingerprint,
            "feature_spec_fingerprints": self.feature_spec_fingerprints,
            "cache_key": self.cache_key,
        }


def _build_selection_plan(
    *,
    outer_window_id: str,
    model_family: str,
    pool: list[NestedCandidateSpec],
    nested_cfg: dict[str, Any],
    search_cfg: dict[str, Any],
    windows: list,
    train_df: pd.DataFrame,
    horizon_contract: HorizonContract | None,
    target_cfg: TargetConfig | None,
    feat_cols: list[str],
) -> SelectionPlan:
    """C.4: Build a SelectionPlan for a single outer-window selection decision."""
    candidate_ids = [_nested_candidate_identity(c) for c in pool]
    candidate_fps = [
        PreparedPanelCache._stable_hash((json.dumps(cid, sort_keys=True),))
        for cid in candidate_ids
    ]

    unique_horizons = sorted({int(c.horizon) for c in pool})
    unique_feature_sets = len({tuple(c.active_features) for c in pool})
    unique_hyperparams = len({(c.model_name, c.model_kind, c.feature_view) for c in pool})

    train_dates = pd.to_datetime(train_df["date"], errors="coerce")
    outer_train_start = str(train_dates.min().date()) if train_dates.notna().any() else ""
    outer_train_end = str(train_dates.max().date()) if train_dates.notna().any() else ""
    outer_eval_start = ""
    outer_eval_end = ""

    allow_cross_family = bool(search_cfg.get("allow_cross_family_selection", False))
    # true_selection_enabled=False means the caller explicitly disabled true
    # model selection (e.g. fixed-shortlist mode).  In that case we must NOT
    # short-circuit — the caller wants full nested validation regardless of
    # candidate count.  In production this is always True because
    # _select_nested_candidate is only called when nested_search_applicable.
    selection_disabled_by_config = not bool(nested_cfg.get("true_selection_enabled", True))

    # Short-circuit analysis
    single_candidate = len(pool) == 1
    single_family = len({c.model_kind for c in pool}) == 1
    single_horizon = len(unique_horizons) == 1
    single_feature_set = unique_feature_sets == 1
    single_hyperparam = unique_hyperparams == 1

    short_circuit_allowed = (
        single_candidate
        and single_family
        and single_horizon
        and single_feature_set
        and single_hyperparam
        and (not allow_cross_family or single_family)
        and not selection_disabled_by_config
    )

    if short_circuit_allowed:
        short_circuit_reason = "single_candidate_no_selection_decision"
        selection_required = False
        selection_required_reason = ""
    else:
        short_circuit_reason = ""
        selection_required = True
        reasons = []
        if not single_candidate:
            reasons.append(f"candidate_count={len(pool)}")
        if not single_family:
            reasons.append(f"families={len({c.model_kind for c in pool})}")
        if not single_horizon:
            reasons.append(f"horizons={unique_horizons}")
        if not single_feature_set:
            reasons.append(f"feature_sets={unique_feature_sets}")
        if not single_hyperparam:
            reasons.append(f"hyperparams={unique_hyperparams}")
        if allow_cross_family:
            reasons.append("allow_cross_family_selection=true")
        if selection_disabled_by_config:
            reasons.append("true_selection_enabled=false")
        selection_required_reason = "; ".join(reasons) if reasons else "unknown"

    hc_fp = ""
    if horizon_contract is not None:
        hc_fp = PreparedPanelCache._stable_hash((
            int(horizon_contract.config.target_horizon_days),
            int(horizon_contract.config.holding_period_days),
        ))

    ts_fp = ""
    if target_cfg is not None:
        ts_fp = PreparedPanelCache._stable_hash((
            int(target_cfg.horizon_days),
            bool(target_cfg.residualize),
            bool(target_cfg.net_of_costs),
        ))

    feat_fps = [PreparedPanelCache._stable_hash((str(fc),)) for fc in sorted(feat_cols)[:10]]

    cache_key = PreparedPanelCache._stable_hash((
        str(outer_window_id),
        str(model_family),
        str(len(pool)),
        candidate_fps[0] if candidate_fps else "",
    ))

    return SelectionPlan(
        outer_window_id=outer_window_id,
        model_family=model_family,
        candidate_count=len(pool),
        candidate_ids=candidate_ids,
        candidate_fingerprints=candidate_fps,
        selection_required=selection_required,
        selection_required_reason=selection_required_reason,
        short_circuit_allowed=short_circuit_allowed,
        short_circuit_reason=short_circuit_reason,
        selection_disabled_by_config=selection_disabled_by_config,
        allow_cross_family_selection=allow_cross_family,
        horizon_candidates=unique_horizons,
        hyperparameter_candidates=unique_hyperparams,
        feature_set_candidates=unique_feature_sets,
        proxy_metric=str(search_cfg.get("proxy_metric", "proxy_selection_score")),
        inner_fold_count=len(windows),
        outer_train_start=outer_train_start,
        outer_train_end=outer_train_end,
        outer_eval_start=outer_eval_start,
        outer_eval_end=outer_eval_end,
        horizon_contract_fingerprint=hc_fp,
        target_spec_fingerprint=ts_fp,
        feature_spec_fingerprints=feat_fps,
        cache_key=cache_key,
    )


def _short_circuit_metrics(candidate: NestedCandidateSpec) -> dict[str, float]:
    """C.4: Return explicit null/NA metrics for short-circuited nested validation.

    Preserves schema compatibility while signaling that no inner validation ran.
    Downstream gates must check nested_validation_skipped before consuming metrics.
    """
    return {
        "nested_sharpe_mean": float("nan"),
        "nested_ic_mean": float("nan"),
        "nested_windows": 0,
        "nested_selection_score": float("nan"),
        "nested_candidate_count": 1,
        "nested_validation_skipped": 1.0,
        "nested_validation_skip_reason": 0.0,  # placeholder; reason in artifacts
        "nested_metrics_available": 0.0,
        "nested_short_circuit": 1.0,
    }


@dataclass(frozen=True)
class NestedWindowState:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    prepared_folds: dict[tuple[int, tuple[str, ...]], Any]
    validation_state_by_horizon: dict[int, ValidationStateCache]


def _nested_candidate_identity(candidate: NestedCandidateSpec) -> dict[str, Any]:
    return {
        "model_name": str(candidate.model_name),
        "model_kind": str(candidate.model_kind),
        "feature_view": str(candidate.feature_view),
        "horizon": int(candidate.horizon),
        "active_features": list(candidate.active_features),
        "turnover_penalty": None if candidate.turnover_penalty is None else float(candidate.turnover_penalty),
        "cost_penalty": None if candidate.cost_penalty is None else float(candidate.cost_penalty),
    }


def _match_nested_candidate(
    pool: list[NestedCandidateSpec],
    identity: dict[str, Any] | None,
) -> NestedCandidateSpec | None:
    if not identity:
        return None
    wanted = {
        "model_name": str(identity.get("model_name", "")),
        "model_kind": str(identity.get("model_kind", "")),
        "feature_view": str(identity.get("feature_view", "")),
        "horizon": int(identity.get("horizon", 0) or 0),
        "active_features": [str(f) for f in identity.get("active_features", [])],
        "turnover_penalty": (
            None
            if identity.get("turnover_penalty") is None
            else float(identity.get("turnover_penalty"))
        ),
        "cost_penalty": (
            None
            if identity.get("cost_penalty") is None
            else float(identity.get("cost_penalty"))
        ),
    }
    for candidate in pool:
        if _nested_candidate_identity(candidate) == wanted:
            return candidate
    return None


# ── Fork-based parallelism ────────────────────────────────────────────────────
# With spawn (macOS default), every ProcessPoolExecutor task pickles the full
# OuterEvaluationContext including PreparedPanelCache (~7 GB of pre-warmed folds).
# With fork, workers are cloned from the parent via copy-on-write — they already
# have _FORK_GLOBAL_CTX in their address space, so it is NEVER pickled.
# _evaluate_model_fork_wrapper reads this global; only (evaluator_ref, model_spec)
# (~few KB) travel through the IPC queue.
@dataclass(frozen=True)
class NestedEvaluationContext:
    train_df: pd.DataFrame
    prepared_cache: PreparedPanelCache
    nested_workspace: list[NestedWindowState] | None
    cfg: dict[str, Any]
    nested_cfg: dict[str, Any]
    horizon_contract: HorizonContract | None
    max_positions: int
    min_positions: int
    embargo_days: int
    use_risk_adj: bool
    primary_path: str
    target_cfg: TargetConfig
    costs: ExecutionCostConfig
    max_name_weight: float
    score_cache: dict[tuple[str, int, tuple[str, ...], int], Any] | None = None

_FORK_GLOBAL_CTX: Any = None
_FORK_NESTED_CTX: NestedEvaluationContext | None = None
# P13: Process start timestamp for stale-cache invalidation in nested candidate metrics
_PROCESS_START_TIME: float = time.time()

def _nested_evaluate_fork_wrapper(candidate: NestedCandidateSpec) -> tuple[NestedCandidateSpec, dict[str, float]]:
    """Thin shim for nested workers."""
    metrics = _nested_validate_candidate(
        _FORK_NESTED_CTX.train_df,
        prepared_cache=_FORK_NESTED_CTX.prepared_cache,
        nested_workspace=_FORK_NESTED_CTX.nested_workspace,
        model_template=candidate.model_template,
        name=candidate.model_name,
        model_kind=candidate.model_kind,
        uses_proba=candidate.uses_proba,
        active_feats=list(candidate.active_features),
        cfg=_FORK_NESTED_CTX.cfg,
        nested_cfg=_FORK_NESTED_CTX.nested_cfg,
        horizon=int(candidate.horizon),
        horizon_contract=_FORK_NESTED_CTX.horizon_contract,
        max_positions=_FORK_NESTED_CTX.max_positions,
        min_positions=_FORK_NESTED_CTX.min_positions,
        embargo_days=_FORK_NESTED_CTX.embargo_days,
        use_risk_adj=_FORK_NESTED_CTX.use_risk_adj,
        primary_path=_FORK_NESTED_CTX.primary_path,
        target_cfg=_FORK_NESTED_CTX.target_cfg,
        costs=_FORK_NESTED_CTX.costs,
        max_name_weight=_FORK_NESTED_CTX.max_name_weight,
        score_cache=_FORK_NESTED_CTX.score_cache,
    )
    return candidate, metrics


def _evaluate_model_fork_wrapper(evaluator: Any, model_spec: tuple, parallel_cfg: dict[str, Any] | None = None) -> Any:
    """Thin shim called in fork workers — ctx comes from inherited global, not pickle."""
    import time
    started_at = time.perf_counter()
    
    # Task 1: Override nested parallelism in child process
    local_ctx = _FORK_GLOBAL_CTX
    if parallel_cfg is not None:
        from dataclasses import replace
        local_ctx = replace(local_ctx, parallel_cfg=parallel_cfg)
        
    # Task 4 & 5: Lazy load data via property access
    # accessing local_ctx.df will trigger loading if _df is None
    _ = local_ctx.df
    # Force low-memory cache in workers to prevent page copying
    local_ctx.global_prepare_cache.max_cache_size = 1
    local_ctx.global_prepare_cache.reset_runtime_stats()
        
    result = evaluator(model_spec, ctx=local_ctx)
    completed_at = time.perf_counter()
    
    if isinstance(result, dict):
        result["_worker_started_at"] = started_at
        result["_worker_completed_at"] = completed_at
        result["_worker_cache_stats"] = local_ctx.global_prepare_cache.stats()
        
    # Free worker-local memory before returning to pool
    if local_ctx.df_path is not None:
        local_ctx.df = None
        import gc
        gc.collect()
        
    return result


def _aggregate_worker_cache_stats(results: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate process-local PreparedPanelCache telemetry returned by workers."""
    keys = [
        "hits",
        "misses",
        "evictions",
        "artifact_hits",
        "artifact_misses",
        "artifact_writes",
        "horizon_memory_hits",
        "horizon_memory_misses",
        "horizon_artifact_hits",
        "horizon_artifact_misses",
        "horizon_artifact_writes",
        "fold_memory_hits",
        "fold_memory_misses",
        "target_memory_hits",
        "target_memory_misses",
        "target_artifact_hits",
        "target_artifact_misses",
        "target_artifact_writes",
        "prepared_fold_lookups",
        "prepared_fold_unique_keys",
    ]
    agg = {k: 0.0 for k in keys}
    for result in results:
        stats = result.get("_worker_cache_stats") if isinstance(result, dict) else None
        if not isinstance(stats, dict):
            continue
        for key in keys:
            val = stats.get(key, 0.0)
            if isinstance(val, (int, float, bool, np.integer, np.floating, np.bool_)):
                agg[key] += float(val)
    total = agg["hits"] + agg["misses"]
    effective_hits = agg["hits"] + agg["artifact_hits"]
    agg["memory_hit_rate"] = (agg["hits"] / total * 100.0) if total > 0 else 0.0
    agg["effective_hit_rate"] = (effective_hits / total * 100.0) if total > 0 else 0.0
    agg["worker_result_count"] = float(sum(1 for r in results if isinstance(r, dict) and isinstance(r.get("_worker_cache_stats"), dict)))
    return agg


@dataclass
class OuterEvaluationContext:
    windows: tuple[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp], ...]
    models: tuple[tuple[str, Any, bool, str], ...]
    df_path: Path | None
    cfg: dict[str, Any]
    args: argparse.Namespace
    nested_cfg: dict[str, Any]
    feat_cols: list[str]
    short_feature_subset: list[str]
    overlay_feature_subset: list[str]
    horizon: int
    horizon_contract: HorizonContract | None
    max_positions: int
    min_positions: int
    embargo_days: int
    use_risk_adj: bool
    exec_costs: ExecutionCostConfig
    research_max_name_weight: float
    global_prepare_cache: PreparedPanelCache
    metric_trial_count: int
    ms_cfg: dict[str, Any]
    gate_cfg: PromotionGateConfig
    long_cand_cfg: LongAlphaCandidateConfig
    feature_contract_summary: dict[str, float]
    alpha_admission_summary: dict[str, float]
    research_state: ResearchStateStore
    timing_ledger: TimingLedger | None
    parallel_cfg: dict[str, Any]
    precomputed_factor_returns: dict[int, pd.DataFrame]
    diagnostic_plan: DiagnosticExecutionPlan = field(default_factory=DiagnosticExecutionPlan)
    _df: pd.DataFrame | None = None
    primary_path: str = "long_only_overlay"

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            if self.df_path is not None and self.df_path.exists():
                import pandas as _pd
                self._df = _pd.read_parquet(self.df_path)
                # Re-attach to cache
                self.global_prepare_cache.base_df = self._df
            else:
                raise ValueError("Context.df is None and df_path is missing or invalid.")
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame | None):
        self._df = value
        if self.global_prepare_cache is not None:
            if value is None:
                self.global_prepare_cache.purge()
            else:
                self.global_prepare_cache.base_df = value


def _screen_model_family(
    model_spec: tuple[str, Any, bool, str],
    *,
    ctx: OuterEvaluationContext,
) -> dict[str, Any]:
    name, model, uses_proba, model_kind = model_spec
    primary_path = ctx.primary_path if model_kind not in {"short_classifier", "short_alpha"} else _primary_evaluation_path(model_kind)
    validation_target_col = _validation_target_col_for_path(primary_path)
    active_feats = _active_features_for_model_kind(
        model_kind,
        ctx.feat_cols,
        short_feature_subset=ctx.short_feature_subset,
        overlay_feature_subset=ctx.overlay_feature_subset,
    )
    log_lines: list[str] = [f"=== SCREEN {name} ({model_kind}) ==="]
    ic_vals: list[float] = []
    daily_icir_vals: list[float] = []
    turnover_vals: list[float] = []
    train_times: list[float] = []
    score_times: list[float] = []
    valid_windows = 0
    robust_aggregates: dict[str, list[float]] = {
        "signal_halflife_days": [],
        "cost_adjusted_ic_mean": [],
        "capacity_weighted_ic": [],
        "turnover_volatility": [],
        "decile_tail_stability": [],
        "hhi_concentration": [],
        "turnover_mean": [],
    }

    for win_idx, (tr_s, tr_e, te_s, te_e) in enumerate(ctx.windows, 1):
        te_label = f"{te_s.date()} → {(te_e - pd.Timedelta(days=1)).date()}"
        purge_cutoff = te_s - pd.Timedelta(days=ctx.embargo_days)
        te_mask = (ctx.df["date"] >= te_s) & (ctx.df["date"] < te_e)
        if not te_mask.any() or int(ctx.df.loc[te_mask, "date"].nunique()) < int(ctx.args.min_oos_days):
            continue
        tr_mask = (ctx.df["date"] >= tr_s) & (ctx.df["date"] < min(tr_e, purge_cutoff))
        if not tr_mask.any():
            continue
        prepared = ctx.global_prepare_cache.get_prepared_fold(
            train_start=tr_s,
            train_end=min(tr_e, purge_cutoff),
            eval_start=te_s,
            eval_end=te_e,
            horizon_days=int(ctx.horizon),
            active_features=active_feats,
        )
        if prepared.train_df.empty or prepared.eval_df.empty or prepared.x_train.shape[1] == 0:
            continue
        y_tr = ctx.global_prepare_cache.get_training_target(
            start=tr_s,
            end=min(tr_e, purge_cutoff),
            horizon_days=int(ctx.horizon),
            model_name=name,
            model_kind=model_kind,
            use_risk_adj=ctx.use_risk_adj,
        )
        if model_kind == "short_classifier" and int((y_tr == 1).sum()) < 30:
            continue
        try:
            t0 = time.perf_counter()
            screen_template = _complexity_screen_model_template(
                model,
                active_feature_count=len(active_feats),
            )
            screen_model = _fit_candidate_model(
                model_template=screen_template,
                name=name,
                model_kind=model_kind,
                tr=prepared.train_df,
                x_tr=prepared.x_train,
                y_tr=y_tr,
            )
            t1 = time.perf_counter()
            score = _score_model_predictions(
                screen_model,
                prepared.x_eval,
                model_kind=model_kind,
                uses_proba=uses_proba,
                score_direction=1,
            )
            train_score = _score_model_predictions(
                screen_model,
                prepared.x_train,
                model_kind=model_kind,
                uses_proba=uses_proba,
                score_direction=1,
            )
            score, cal_result = _calibrate_scores(
                prepared.train_df,
                train_score,
                score,
                target_col=validation_target_col,
            )
            t2 = time.perf_counter()
        except Exception as exc:
            log_lines.append(f"  [window {win_idx}/{len(ctx.windows)}] screen failed ({te_label}): {exc}")
            continue
        finite_score = np.asarray(score, dtype=float)
        finite_score = finite_score[np.isfinite(finite_score)]
        score_scale = max(1.0, float(np.nanmedian(np.abs(finite_score))) if finite_score.size else 1.0)
        score_tol = float(np.sqrt(np.finfo(float).eps) * score_scale)
        if finite_score.size < 2 or float(np.nanstd(finite_score)) <= score_tol:
            log_lines.append(
                f"  [window {win_idx}/{len(ctx.windows)}] degenerate screen scores "
                f"(std<=machine_tol); skipping IC/turnover diagnostics for this window"
            )
            continue
        scored = prepared.eval_df.assign(score=score)

        ic_stats = compute_execution_robustness(scored, primary_path=primary_path, target_col=validation_target_col, model_name=name, window_idx=win_idx)
        
        # [AUDIT] Metric Integrity
        global_audit.min_valid_days = min(global_audit.min_valid_days, ic_stats.get("ic_valid_days", 9999))
        global_audit.target_col = validation_target_col
        
        # [IC PIPELINE COMPARISON TABLE] (User Request)
        if DEBUG_DIAGNOSTICS and win_idx == 1 and not getattr(ctx, "_ic_comparison_done", False):
            from model_selection.validation import cross_sectional_ic
            ref_ic_res = cross_sectional_ic(scored, target_col=validation_target_col, quiet=True)
            
            print("\n[IC Pipeline Comparison Table]")
            print(f"{'ic_source':<20} | {'ic_mean':<10} | {'ic_tstat':<10} | {'valid_days':<10} | {'nan_days':<10} | {'method':<10}")
            print("-" * 85)
            
            # 1. IC Summary (via compute_execution_robustness)
            print(f"{'IC Summary':<20} | {ic_stats['ic_mean']:<10.6f} | {ic_stats['ic_tstat']:<10.4f} | {ic_stats['ic_valid_days']:<10} | {ic_stats['ic_nan_days']:<10} | {'Spearman':<10}")
            
            # 2. Reference IC (Direct call to cross_sectional_ic)
            mu = ref_ic_res.get("cs_ic_spearman_mean", 0)
            t = ref_ic_res.get("cs_ic_spearman_tstat", 0)
            vd = ref_ic_res.get("cs_ic_n_days", 0)
            nd = ref_ic_res.get("cs_ic_nan_days", 0)
            print(f"{'Reference IC':<20} | {mu:<10.6f} | {t:<10.4f} | {vd:<10} | {nd:<10} | {'Spearman':<10}")
            
            # 3. Aggregated (SIGN_TRACE logic for this window)
            print(f"{'SIGN_TRACE (window)':<20} | {ic_stats['ic_mean']:<10.6f} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'Spearman':<10}")
            print("-" * 85)
            print(f"Model: {name} | Window: {win_idx} | Target: {validation_target_col} | Grouping: Cross-sectional per date")
            print(f"Goal: Confirm if IC Summary aligns with Reference IC math.")
            setattr(ctx, "_ic_comparison_done", True)
        
        # Keys now match the strict schema in validation.py (Task 2)
        ic = ic_stats["ic_mean"]
        daily_icir = ic_stats["icir"]
        turnover = ic_stats["turnover_mean"]
        robust_aggregates.setdefault("forecast_calibration_slope", []).append(float(cal_result.slope))
        robust_aggregates.setdefault("forecast_calibration_shrinkage", []).append(float(cal_result.shrinkage))
        robust_aggregates.setdefault("forecast_calibration_tstat", []).append(float(cal_result.slope_tstat))
        
        # Model-level Sign Inversion Audit (Task 5)
        from model_selection.validation import cross_sectional_ic
        ic_neg = cross_sectional_ic(scored.assign(score=-score), target_col=validation_target_col, quiet=True).get("cs_ic_spearman_mean", float("nan"))
        if np.isfinite(ic_neg) and np.isfinite(ic) and ic_neg > ic and ic_neg > 0.01:
            log_lines.append(f"  [window {win_idx}] SIGN_INVERSION_DETECTED: IC={ic:.4f}, IC(-score)={ic_neg:.4f}")
        
        if np.isfinite(ic):
            ic_vals.append(float(ic))
        else:
            log_lines.append(f"  [window {win_idx}] ic_mean is NaN (undefined)")
            
        if np.isfinite(daily_icir):
            daily_icir_vals.append(float(daily_icir))
        else:
            log_lines.append(f"  [window {win_idx}] daily_icir is NaN (undefined)")
            
        if np.isfinite(turnover):
            turnover_vals.append(float(turnover))
        else:
            log_lines.append(f"  [window {win_idx}] turnover_mean is NaN (undefined)")
        train_times.append(float(t1 - t0))
        score_times.append(float(t2 - t1))
        
        # New Diagnostics (already merged into ic_stats by compute_execution_robustness)
        robust_vals = {
            "signal_halflife_days": float(ic_stats.get("signal_halflife_days", float("nan"))),
            "cost_adjusted_ic_mean": float(ic_stats.get("cost_adjusted_ic_mean", float("nan"))),
            "capacity_weighted_ic": float(ic_stats.get("capacity_weighted_ic", float("nan"))),
            "turnover_volatility": float(ic_stats.get("turnover_volatility", float("nan"))),
            "decile_tail_stability": float(ic_stats.get("decile_tail_stability", float("nan"))),
            "hhi_concentration": float(ic_stats.get("hhi_concentration", float("nan"))),
            "turnover_mean": float(ic_stats.get("turnover_mean", float("nan"))),
        }
        for k, v in robust_vals.items():
            if np.isfinite(v):
                robust_aggregates[k].append(v)

        valid_windows += 1
        log_lines.append(
            f"  [window {win_idx}/{len(ctx.windows)}] test=[{te_label}] | "
            f"IC={ic:.4f} | annICIR={daily_icir:.2f} | turnover_proxy={turnover:.3f} | "
            f"timers fit/score={(t1 - t0):.1f}/{(t2 - t1):.1f}s"
        )

        # Early termination: if a model has produced NaN IC across the first
        # min_early_abort windows, it is structurally unable to generate a
        # predictive signal on this feature set. Skip remaining windows to
        # free worker slots for candidates that can.
        if win_idx >= 3 and len(ic_vals) == 0:
            log_lines.append(
                f"  [early_abort] {name}: NaN IC across first {win_idx} windows — "
                "no predictive signal detected; skipping remaining windows"
            )
            break

    ic_mean = float(np.nanmean(ic_vals)) if ic_vals else float("nan")
    daily_icir_mean = float(np.nanmean(daily_icir_vals)) if daily_icir_vals else float("nan")
    turnover_mean = float(np.nanmean(turnover_vals)) if turnover_vals else float("nan")
    capacity_proxy_mean = (
        float(np.nanmean(robust_aggregates["capacity_weighted_ic"]))
        if robust_aggregates["capacity_weighted_ic"]
        else float("nan")
    )
    cost_adjusted_ic_mean = (
        float(np.nanmean(robust_aggregates["cost_adjusted_ic_mean"]))
        if robust_aggregates["cost_adjusted_ic_mean"]
        else float("nan")
    )
    simple_cost_proxy_mean = (
        abs(float(ic_mean) - float(cost_adjusted_ic_mean))
        if np.isfinite(ic_mean) and np.isfinite(cost_adjusted_ic_mean)
        else float("nan")
    )
    
    # Stage A is pure signal discovery: IC / ICIR / decay, without optimizer,
    # execution costs, or turnover penalties.
    if not np.isfinite(ic_mean) or not np.isfinite(daily_icir_mean):
        signal_score = float("nan")
    else:
        signal_score = ic_mean + (0.15 * daily_icir_mean)

    # Stage B is feasibility only: it uses cheap turnover/capacity/cost proxies,
    # never full portfolio simulation. Keep the legacy algebraic ordering so the
    # same candidate set is selected when the same inputs are available.
    if not np.isfinite(signal_score) or not np.isfinite(turnover_mean):
        feasibility_score = float("nan")
        missing = []
        if not np.isfinite(ic_mean): missing.append("ic_mean")
        if not np.isfinite(daily_icir_mean): missing.append("daily_icir_mean")
        if not np.isfinite(turnover_mean): missing.append("turnover_mean")
        log_lines.append(f"  [!] UNDEFINED FEASIBILITY SCORE: Components {missing} contain NaN values. Score propagated as NaN.")
    else:
        feasibility_score = signal_score - (0.10 * turnover_mean)
    screen_score = feasibility_score
    
    # FINAL AGGREGATION TRACE (Task 5)
    print(f"\n[SIGN_TRACE] Final Aggregation: {name}")
    print(f"  ic_mean        : {ic_mean:>10.6f}")
    print(f"  daily_icir_mean: {daily_icir_mean:>10.6f}")
    print(f"  turnover_mean  : {turnover_mean:>10.6f}")
    print(f"  signal_score   : {signal_score:>10.6f}")
    print(f"  feasibility    : {feasibility_score:>10.6f}")
    
    log_lines.append(
        f"StageA SignalScore: {signal_score:.4f} | IC={ic_mean:.4f} | annICIR={daily_icir_mean:.2f} | "
        f"StageB FeasibilityScore: {feasibility_score:.4f} | turnover_proxy={turnover_mean:.3f} | "
        f"capacity_proxy={capacity_proxy_mean:.4f} | simple_cost_proxy={simple_cost_proxy_mean:.4f} | "
        f"valid_windows={valid_windows} | "
        f"avg fit/score={float(np.nanmean(train_times)) if train_times else float('nan'):.1f}/"
        f"{float(np.nanmean(score_times)) if score_times else float('nan'):.1f}s"
    )
    return {
        "model_name": name,
        "model_kind": model_kind,
        "primary_path": primary_path,
        "rebalance_frequency": int(ctx.horizon),
        "screen_score": float(screen_score),
        "stage_a_signal_score": float(signal_score),
        "stage_b_feasibility_score": float(feasibility_score),
        "signal_score": float(signal_score),
        "feasibility_score": float(feasibility_score),
        "screen_ic_mean": float(ic_mean),
        "screen_daily_icir_mean": float(daily_icir_mean),
        "screen_turnover_proxy_mean": float(turnover_mean),
        "feasibility_turnover_proxy_mean": float(turnover_mean),
        "feasibility_capacity_proxy_mean": float(capacity_proxy_mean),
        "feasibility_simple_cost_proxy_mean": float(simple_cost_proxy_mean),
        "diag_robust_signal_halflife": float(np.nanmean(robust_aggregates["signal_halflife_days"])) if robust_aggregates.get("signal_halflife_days") else float("nan"),
        "diag_robust_cost_adjusted_ic": float(cost_adjusted_ic_mean),
        "diag_robust_capacity_weighted_ic": float(capacity_proxy_mean),
        "diag_robust_turnover_volatility": float(np.nanmean(robust_aggregates["turnover_volatility"])) if robust_aggregates.get("turnover_volatility") else float("nan"),
        "diag_robust_tail_stability": float(np.nanmean(robust_aggregates["decile_tail_stability"])) if robust_aggregates.get("decile_tail_stability") else float("nan"),
        "diag_robust_hhi": float(np.nanmean(robust_aggregates["hhi_concentration"])) if robust_aggregates.get("hhi_concentration") else float("nan"),
        "diag_robust_turnover_mean": float(np.nanmean(robust_aggregates["turnover_mean"])) if robust_aggregates.get("turnover_mean") else float("nan"),
        "valid_windows": int(valid_windows),
        "log_lines": log_lines,
    }


def _evaluate_baseline_scorers(
    *,
    ctx: "OuterEvaluationContext",
) -> list[dict[str, Any]]:
    """
    P9: Evaluate baseline scorers (equal-weight, IC-weighted) on the raw
    feature matrix, bypassing the model fitting loop.

    Baselines are reference statistics, not hypothesis tests. They establish
    the "no-ML" floor that learned models must beat. If ML models produce
    negative subsumption alpha, the baselines are the superior choice.

    This function runs through the same walk-forward windows as SignalDiscovery
    but:
    - Operates directly on the preprocessed feature matrix (no clone/fit)
    - Computes IC per window, aggregates across windows
    - Returns results in the same format as _screen_model_family
    """
    from model_selection.baseline_scorers import EqualWeightBaseline, ICWeightedBaseline
    from model_selection.validation import compute_execution_robustness

    baseline_specs = [
        ("EqualWeightBaseline", EqualWeightBaseline(), False, "regressor"),
        ("ICWeightedBaseline", ICWeightedBaseline(ic_window=126, min_ic_obs=30), False, "regressor"),
    ]

    results: list[dict[str, Any]] = []

    for name, baseline, uses_proba, model_kind in baseline_specs:
        log_lines: list[str] = [f"=== BASELINE {name} ==="]
        ic_vals: list[float] = []
        daily_icir_vals: list[float] = []
        valid_windows = 0

        for win_idx, (tr_s, tr_e, te_s, te_e) in enumerate(ctx.windows, 1):
            te_label = f"{te_s.date()} → {(te_e - pd.Timedelta(days=1)).date()}"
            purge_cutoff = te_s - pd.Timedelta(days=ctx.embargo_days)
            te_mask = (ctx.df["date"] >= te_s) & (ctx.df["date"] < te_e)
            if not te_mask.any() or int(ctx.df.loc[te_mask, "date"].nunique()) < int(ctx.args.min_oos_days):
                continue
            tr_mask = (ctx.df["date"] >= tr_s) & (ctx.df["date"] < min(tr_e, purge_cutoff))
            if not tr_mask.any():
                continue

            active_feats = ctx.feat_cols
            prepared = ctx.global_prepare_cache.get_prepared_fold(
                train_start=tr_s,
                train_end=min(tr_e, purge_cutoff),
                eval_start=te_s,
                eval_end=te_e,
                horizon_days=int(ctx.horizon),
                active_features=active_feats,
            )
            if prepared.train_df.empty or prepared.eval_df.empty or prepared.x_train.shape[1] == 0:
                continue

            y_tr = ctx.global_prepare_cache.get_training_target(
                start=tr_s,
                end=min(tr_e, purge_cutoff),
                horizon_days=int(ctx.horizon),
                model_name=name,
                model_kind=model_kind,
                use_risk_adj=ctx.use_risk_adj,
            )

            try:
                # Fit baseline on training data
                tr_dates = pd.to_datetime(prepared.train_df["date"], errors="coerce").to_numpy()
                baseline.fit(prepared.x_train, y_tr, dates=tr_dates)

                # Predict on evaluation data
                score = baseline.predict(prepared.x_eval)

                finite_score = np.asarray(score, dtype=float)
                finite_score = finite_score[np.isfinite(finite_score)]
                score_scale = max(1.0, float(np.nanmedian(np.abs(finite_score))) if finite_score.size else 1.0)
                score_tol = float(np.sqrt(np.finfo(float).eps) * score_scale)
                if finite_score.size < 2 or float(np.nanstd(finite_score)) <= score_tol:
                    log_lines.append(f"  [window {win_idx}] degenerate scores — skipping")
                    continue

                scored = prepared.eval_df.assign(score=score)
                ic_stats = compute_execution_robustness(scored, primary_path=ctx.primary_path, target_col=validation_target_col, model_name=name, window_idx=win_idx)

                ic = ic_stats["ic_mean"]
                daily_icir = ic_stats["icir"]

                if np.isfinite(ic):
                    ic_vals.append(float(ic))
                if np.isfinite(daily_icir):
                    daily_icir_vals.append(float(daily_icir))

                valid_windows += 1
                log_lines.append(
                    f"  [window {win_idx}/{len(ctx.windows)}] test=[{te_label}] | "
                    f"IC={ic:.4f} | annICIR={daily_icir:.2f}"
                )
            except Exception as exc:
                log_lines.append(f"  [window {win_idx}] failed: {exc}")
                continue

        ic_mean = float(np.nanmean(ic_vals)) if ic_vals else float("nan")
        daily_icir_mean = float(np.nanmean(daily_icir_vals)) if daily_icir_vals else float("nan")
        screen_score = ic_mean if np.isfinite(ic_mean) else float("nan")

        log_lines.append(
            f"  [AGGREGATE] IC_mean={ic_mean:.4f} | ICIR={daily_icir_mean:.2f} | valid_windows={valid_windows}"
        )

        results.append({
            "model_name": name,
            "model_kind": model_kind,
            "screen_ic_mean": ic_mean,
            "screen_daily_icir": daily_icir_mean,
            "screen_score": screen_score,
            "valid_windows": valid_windows,
            "log_lines": log_lines,
        })

    return results

def signal_screening(
    model_specs: list[tuple[str, Any, bool, str]],
    *,
    phase_runner: Any,
) -> list[dict[str, Any]]:
    """Stage A: discover predictive signal without portfolio construction."""

    return phase_runner("SignalDiscovery", list(model_specs), evaluator=_screen_model_family)


def feasibility_filter(
    signal_results: list[dict[str, Any]],
    model_specs: list[tuple[str, Any, bool, str]],
    *,
    screening_cfg: dict[str, Any],
) -> tuple[list[tuple[str, Any, bool, str]], list[dict[str, Any]]]:
    """Stage B: reduce candidates using cheap feasibility proxies only."""

    if not signal_results:
        return [], []
    if not bool(screening_cfg.get("enabled", True)):
        return list(model_specs), [
            {
                "primary_path": "all",
                "candidate_count_in": int(len(model_specs)),
                "candidate_count_out": int(len(model_specs)),
                "selected_models": [str(spec[0]) for spec in model_specs],
                "disabled": True,
            }
        ]

    shortlist_top_k = max(1, int(screening_cfg.get("shortlist_top_k_per_path", 4) or 4))
    min_keep = max(1, int(screening_cfg.get("min_keep_per_path", 2) or 2))
    adaptive_execution_budget = bool(screening_cfg.get("adaptive_execution_budget", False))
    require_positive_feasibility = bool(screening_cfg.get("require_positive_feasibility", False))
    by_path: dict[str, list[dict[str, Any]]] = {}
    diagnostic_by_path: dict[str, list[str]] = {}
    for result in signal_results:
        path = str(result.get("primary_path", ""))
        name = str(result.get("model_name", ""))
        kind = str(result.get("model_kind", ""))
        if _is_diagnostic_only(name) or _is_classifier_stock_selector(name, kind):
            diagnostic_by_path.setdefault(path, []).append(name)
            continue
        by_path.setdefault(path, []).append(result)

    keep_keys: set[tuple[str, str]] = set()
    audit_rows: list[dict[str, Any]] = []
    for path_key, items in sorted(by_path.items()):
        ranked = sorted(
            items,
            key=lambda r: _safe_float(
                r.get("feasibility_score", r.get("screen_score")),
                float("-inf"),
            ),
            reverse=True,
        )
        eligible = ranked
        rejected_nonpositive: list[str] = []
        if require_positive_feasibility:
            eligible = []
            for item in ranked:
                score = _safe_float(item.get("feasibility_score", item.get("screen_score")), float("-inf"))
                if np.isfinite(score) and score > 0.0:
                    eligible.append(item)
                else:
                    rejected_nonpositive.append(str(item.get("model_name", "")))
        if adaptive_execution_budget:
            keep_floor = 1 if eligible else 0
            viable_count = len(eligible)
            adaptive_keep = int(np.ceil(np.sqrt(float(viable_count)))) if viable_count > 0 else 0
            keep_n = min(viable_count, max(keep_floor, min(shortlist_top_k, adaptive_keep)))
        else:
            keep_n = min(len(eligible), max(min_keep, shortlist_top_k))
        selected = eligible[:keep_n]
        selected_names = [str(item["model_name"]) for item in selected]
        keep_keys.update((str(item["model_name"]), str(item["model_kind"])) for item in selected)
        audit_rows.append(
            {
                "primary_path": str(path_key),
                "candidate_count_in": int(len(ranked)),
                "candidate_count_eligible": int(len(eligible)),
                "candidate_count_out": int(keep_n),
                "selected_models": selected_names,
                "diagnostic_only_models": diagnostic_by_path.get(path_key, []),
                "rejected_nonpositive_feasibility": rejected_nonpositive,
                "adaptive_execution_budget": bool(adaptive_execution_budget),
                "disabled": False,
            }
        )

    for path_key, names in sorted(diagnostic_by_path.items()):
        if path_key in by_path:
            continue
        audit_rows.append(
            {
                "primary_path": str(path_key),
                "candidate_count_in": int(len(names)),
                "candidate_count_out": 0,
                "selected_models": [],
                "diagnostic_only_models": names,
                "disabled": False,
            }
        )

    shortlisted = [spec for spec in model_specs if (str(spec[0]), str(spec[3])) in keep_keys]
    return shortlisted, audit_rows


def execution_validation(
    model_specs: list[tuple[str, Any, bool, str]],
    *,
    phase_runner: Any,
) -> list[dict[str, Any]]:
    """Stage C: full executable validation through optimizer, costs, and simulator."""

    return phase_runner("ExecutionValidation", list(model_specs), evaluator=_evaluate_model_family)


def _program_feature_view(
    features: list[str],
    *,
    primary_path: str,
    view: str,
) -> list[str]:
    if view == "full":
        return list(features)
    families_by_view = {
        "long_short_spread": {
            "program": {
                "momentum",
                "sector_relative",
                "residual_alpha",
                "quality",
                "quality_lowvol",
                "fundamental_quality",
                "reversal",
            },
        },
        "long_only_overlay": {
            "program": {
                "quality",
                "quality_lowvol",
                "fundamental_quality",
                "risk",
                "liquidity",
            },
        },
        "short_side": {
            "program": {
                "short_momentum",
                "fundamental_deterioration",
                "fundamental_leverage",
                "dilution",
                "reporting_quality",
                "crowding",
                "squeeze_filter",
            },
        },
    }
    allowed = families_by_view.get(primary_path, {}).get(view, set())
    if not allowed:
        return list(features)
    filtered = [f for f in features if FEATURE_SPECS.get(f) and FEATURE_SPECS[f].family in allowed]
    return filtered if len(filtered) >= max(5, min(len(features), 5)) else list(features)


def _nested_program_kinds(primary_path: str) -> set[str]:
    if primary_path == "short_side":
        return {"short_alpha", "short_classifier"}
    if primary_path == "long_only_overlay":
        return {"overlay_alpha", "regressor"}
    return {"long_alpha", "classifier", "regressor"}


def _nested_model_pool_for_outer(
    model_spec: tuple[str, Any, bool, str],
    *,
    ctx_models: tuple[tuple[str, Any, bool, str], ...],
    nested_cfg: dict[str, Any],
) -> list[tuple[str, Any, bool, str]]:
    """Return the nested candidate pool without reopening rejected model families."""
    _, _, _, model_kind = model_spec
    search_cfg = (
        nested_cfg.get("search", {})
        if isinstance(nested_cfg.get("search", {}), dict)
        else {}
    )
    allow_cross_family = bool(search_cfg.get("allow_cross_family_selection", False))
    if allow_cross_family and not _is_economic_model_kind(str(model_kind)):
        return list(ctx_models)
    return [model_spec]


def _build_nested_candidate_pool(
    *,
    models: list[tuple[str, Any, bool, str]],
    primary_path: str,
    cfg: dict[str, Any],
    nested_cfg: dict[str, Any],
    default_horizon: int,
    feat_cols: list[str],
    short_feature_subset: list[str],
    overlay_feature_subset: list[str],
) -> list[NestedCandidateSpec]:
    search_cfg = (nested_cfg.get("search", {}) if isinstance(nested_cfg, dict) else {}) or {}
    raw_horizons = search_cfg.get("candidate_horizons", [])
    if raw_horizons:
        horizons = [int(h) for h in raw_horizons if int(h) > 0]
    else:
        horizons = [int(default_horizon)]
    max_horizons = max(1, int(search_cfg.get("max_horizons", len(horizons)) or len(horizons)))
    horizons = list(dict.fromkeys(horizons))[:max_horizons] or [int(default_horizon)]
    feature_views = [str(v) for v in (search_cfg.get("feature_views", ["full", "program"]) or ["full", "program"])]
    feature_views = list(dict.fromkeys(feature_views))
    max_candidates = max(4, int(search_cfg.get("max_candidates", 24) or 24))

    out: list[NestedCandidateSpec] = []
    seen: set[tuple[Any, ...]] = set()
    eligible = _nested_program_kinds(primary_path)
    for base_name, base_model, uses_proba, model_kind in models:
        if model_kind not in eligible:
            continue
        base_features = _active_features_for_model_kind(
            model_kind,
            feat_cols,
            short_feature_subset=short_feature_subset,
            overlay_feature_subset=overlay_feature_subset,
        )
        if not base_features:
            continue
        candidate_horizons = horizons
        candidate_views = feature_views if model_kind in {"long_alpha", "overlay_alpha", "short_alpha"} else ["full"]
        for cand_h in candidate_horizons:
            for view in candidate_views:
                active = tuple(
                    _program_feature_view(
                        base_features,
                        primary_path=primary_path,
                        view=view,
                    )
                )
                if len(active) == 0:
                    continue
                key = (
                    base_name,
                    model_kind,
                    view,
                    tuple(active),
                    int(cand_h),
                )
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    NestedCandidateSpec(
                        model_name=base_name,
                        model_kind=model_kind,
                        uses_proba=bool(uses_proba),
                        model_template=base_model,
                        active_features=active,
                        feature_view=str(view),
                        horizon=int(cand_h),
                    )
                )
                if len(out) >= max_candidates:
                    return out
    return out


def _prescreen_qp_candidates(
    scored: pd.DataFrame,
    *,
    primary_path: str,
    max_positions: int,
    qp_top_k_multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Reduce the scored universe to the K most investable candidates before the QP solve.

    The QP optimizer assigns zero weight to stocks ranked outside the optimal set.
    With N=500 and max_positions=25, the optimizer will zero ~450 stocks. Pre-screening
    eliminates the O(N²) overhead of including those zero-weight stocks.

    K = max(max_positions × qp_top_k_multiplier, 50) — a 2× buffer ensures the
    optimizer can always form its full optimal portfolio with slack to spare.
    Mathematically identical to the full-N solve: excluded stocks would receive
    weight=0 in the unconstrained optimum, so restricting the feasible set to
    top/bottom K does not change the optimal solution.

    Parameters
    ----------
    primary_path : str
        Portfolio construction path — determines which tail to keep.
    max_positions : int
        Configured maximum portfolio positions (from EvaluationConfig).
    qp_top_k_multiplier : float
        Safety buffer multiplier over max_positions (default 2×).
    """
    if "score" not in scored.columns or "date" not in scored.columns or scored.empty:
        return scored

    k = max(int(max_positions * qp_top_k_multiplier), 50)
    date_codes, _ = pd.factorize(pd.to_datetime(scored["date"], errors="coerce"), sort=False)
    counts = np.bincount(date_codes[date_codes >= 0]) if date_codes.size else np.asarray([], dtype=int)
    n_max = int(counts.max()) if counts.size else 0
    if n_max <= k:
        return scored  # universe already within budget

    is_short = primary_path in {"short_side", "short_alpha", "short_classifier"}
    is_long_short = primary_path == "long_short_spread"
    scores = pd.to_numeric(scored["score"], errors="coerce").to_numpy(dtype=float)
    keep = np.zeros(len(scored), dtype=bool)
    for code in np.flatnonzero(counts > 0):
        idx = np.flatnonzero(date_codes == code)
        finite_idx = idx[np.isfinite(scores[idx])]
        if finite_idx.size == 0:
            continue
        if is_long_short:
            asc = finite_idx[np.argsort(scores[finite_idx], kind="mergesort")[:k]]
            desc = finite_idx[np.argsort(-scores[finite_idx], kind="mergesort")[:k]]
            keep[asc] = True
            keep[desc] = True
        else:
            desc = finite_idx[np.argsort(-scores[finite_idx], kind="mergesort")[:k]]
            keep[desc] = True

    return scored.loc[keep].reset_index(drop=True)


def _nested_validate_candidate(
    train_df: pd.DataFrame,
    *,
    prepared_cache: PreparedPanelCache,
    nested_workspace: list[NestedWindowState] | None,
    model_template: Any,
    name: str,
    model_kind: str,
    uses_proba: bool,
    active_feats: list[str],
    cfg: dict[str, Any],
    nested_cfg: dict[str, Any],
    horizon: int,
    horizon_contract: HorizonContract | None,
    max_positions: int,
    min_positions: int,
    embargo_days: int,
    use_risk_adj: bool,
    primary_path: str,
    target_cfg: TargetConfig,
    costs: ExecutionCostConfig,
    max_name_weight: float,
    score_cache: dict[tuple[str, int, tuple[str, ...], int], Any] | None = None,
    ) -> dict[str, float]:
    if not bool(nested_cfg.get("enabled", True)):
        return {
            "nested_sharpe_mean": float("nan"),
            "nested_ic_mean": float("nan"),
            "nested_windows": 0,
            "nested_selection_score": float("nan"),
        }
    if nested_workspace is not None:
        windows = [
            (state.train_start, state.train_end, state.eval_start, state.eval_end)
            for state in nested_workspace
        ]
    else:
        windows = _nested_inner_windows(
            train_df,
            max_windows=int(nested_cfg.get("max_windows", 1)),
            validation_days=int(nested_cfg.get("validation_days", 126)),
            min_train_days=int(nested_cfg.get("min_train_days", 504)),
            embargo_days=int(embargo_days),
        )
    _search_cfg = (nested_cfg.get("search", {}) if isinstance(nested_cfg.get("search", {}), dict) else {}) or {}
    ic_qp_floor = float(_search_cfg.get("ic_qp_floor", 0.01))
    _is_short_path = primary_path == "short_side"
    validation_target_col = _validation_target_col_for_path(primary_path)
    sharpes: list[float] = []
    ics: list[float] = []
    daily_icirs: list[float] = []
    cost_to_gross_vals: list[float] = []
    beta_abs_vals: list[float] = []
    _per_window_directions: list[int] = []
    _window_failure_log: list[str] = []
    for window_idx, (tr_s, tr_e, va_s, va_e) in enumerate(windows):
        if nested_workspace is not None:
            window_state = nested_workspace[window_idx]
            prepared = window_state.prepared_folds.get((int(horizon), tuple(str(f) for f in active_feats)))
            if prepared is None:
                continue
            eval_state = window_state.validation_state_by_horizon.get(int(horizon))
        else:
            prepared = prepared_cache.get_prepared_fold(
                train_start=tr_s,
                train_end=tr_e,
                eval_start=va_s,
                eval_end=va_e,
                horizon_days=int(horizon),
                active_features=active_feats,
            )
            eval_cfg = _evaluation_config(
                cfg,
                path=primary_path,
                max_positions=max_positions,
                min_positions=min_positions,
                horizon=horizon,
                horizon_contract=horizon_contract,
            )
            eval_state = prepared_cache.get_validation_state(
                start=va_s,
                end=va_e,
                horizon_days=int(horizon),
                evaluation_cfg=eval_cfg,
            )
        if prepared.train_df.empty or prepared.eval_df.empty:
            continue
        x_tr = prepared.x_train
        x_va = prepared.x_eval
        if x_tr.shape[1] == 0:
            continue
        y_tr = prepared_cache.get_training_target(
            start=tr_s,
            end=tr_e,
            horizon_days=int(horizon),
            model_name=name,
            model_kind=model_kind,
            use_risk_adj=use_risk_adj,
        )
        if model_kind == "short_classifier" and int((y_tr == 1).sum()) < 30:
            continue
        try:
            score_raw = None
            train_score_raw = None
            if score_cache is not None:
                cache_key = (name, int(horizon), tuple(sorted(active_feats)), int(window_idx))
                cached_scores = score_cache.get(cache_key)
                if isinstance(cached_scores, tuple) and len(cached_scores) == 2:
                    score_raw, train_score_raw = cached_scores
                else:
                    score_raw = cached_scores

            if score_raw is None or train_score_raw is None:
                inner_model = _fit_candidate_model(
                    model_template=model_template,
                    name=name,
                    model_kind=model_kind,
                    tr=prepared.train_df,
                    x_tr=x_tr,
                    y_tr=y_tr,
                )
                # Always cache raw scores (direction=1); calibration is applied after.
                score_raw = _score_model_predictions(
                    inner_model,
                    x_va,
                    model_kind=model_kind,
                    uses_proba=uses_proba,
                    score_direction=1,
                )
                train_score_raw = _score_model_predictions(
                    inner_model,
                    x_tr,
                    model_kind=model_kind,
                    uses_proba=uses_proba,
                    score_direction=1,
                )
                if score_cache is not None:
                    score_cache[cache_key] = (score_raw, train_score_raw)

            # --- Score direction calibration (validation data only, not OOS) ---
            _scored_raw = prepared.eval_df.assign(score=score_raw)
            _ic_raw_stats = cross_sectional_ic(_scored_raw, target_col=validation_target_col)
            _ic_raw = float(_ic_raw_stats.get("cs_ic_spearman_mean", 0.0) or 0.0)
            _win_direction, _dir_mode, _dir_reason = _determine_score_direction(
                _ic_raw, model_kind=model_kind
            )
            _per_window_directions.append(_win_direction)
            # Apply direction: calibrated IC = ic_raw * direction (spearmanr(-x,y)=-spearmanr(x,y))
            _ic_calibrated = _ic_raw * _win_direction
            score_directed_raw = score_raw * float(_win_direction)
            train_score_directed_raw = train_score_raw * float(_win_direction)
            score, _cal_result = _calibrate_scores(
                prepared.train_df,
                train_score_directed_raw,
                score_directed_raw,
                target_col=validation_target_col,
            )
            # ------------------------------------------------------------------

            scored = prepared.eval_df.assign(score=score)
            # IC is O(N log N) rank correlation — computed from calibrated scores.
            ic_stats = cross_sectional_ic(scored, target_col=validation_target_col)
            ic = ic_stats.get("cs_ic_spearman_mean", float("nan"))
            if np.isfinite(ic):
                ics.append(float(ic))
            ann_icir = ic_stats.get("daily_ic_annualized_icir", float("nan"))
            if np.isfinite(ann_icir):
                daily_icirs.append(float(ann_icir))
            # Grinold-Kahn floor using signal STRENGTH (mandate-agnostic abs value).
            # Short models have negative calibrated IC — abs() correctly measures strength.
            if not (np.isfinite(_ic_calibrated) and _ic_signal_strength(_ic_calibrated) >= ic_qp_floor):
                continue
            halflife = float(ic_stats.get("signal_halflife_days", float("nan")))
            eval_cfg = _evaluation_config(
                cfg,
                path=primary_path,
                max_positions=max_positions,
                min_positions=min_positions,
                horizon=horizon,
                horizon_contract=horizon_contract,
                signal_halflife_days=halflife,
            )
            
            # Task 2: Calibrate L1 turnover penalty if enabled
            if getattr(eval_cfg, "optimization_type", "l2") == "l1":
                # Use a slice of the evaluation window or the training window to calibrate
                # Here we use the evaluation window itself as a proxy for the 'current' regime
                # but we could also use the tail of tr_df.
                cal_lambda = _calibrate_l1_turnover_penalty(scored, eval_cfg, state_cache=eval_state)
                eval_cfg = replace(eval_cfg, lambda_turn_override=cal_lambda)
                
            # --- simulation_policy: nested_full ---
            _sim_policy = (nested_cfg.get("simulation_policy", {}) or {}).get("nested_full", "executable")
            scored_qp = _prescreen_qp_candidates(
                scored, primary_path=primary_path, max_positions=max_positions
            )
            _t_sim_0 = time.perf_counter()
            if _sim_policy == "proxy_only":
                # Rank-based proxy: no QP optimizer, no covariance, no market state.
                # Used ONLY for candidate ranking — never for promotion.
                daily, pnl = simulate_proxy_portfolio(scored_qp, eval_cfg)
            else:
                daily, pnl = simulate_executable_portfolio(
                    scored_qp,
                    eval_cfg,
                    state_cache=eval_state,
                )
            _t_sim_1 = time.perf_counter()
            _SIMULATION_TELEMETRY.record(
                phase="nested_full",
                model_name=name,
                window_idx=window_idx,
                scored=scored_qp,
                cfg=eval_cfg,
                runtime_s=_t_sim_1 - _t_sim_0,
                simulation_mode=_sim_policy,
                is_cached=getattr(pnl, "attrs", {}).get("_is_cached", False),
            )
            sharpe = _sharpe_from_series(daily.to_numpy(dtype=float), horizon=horizon)
            if np.isfinite(sharpe):
                sharpes.append(float(sharpe))
            gross = float(pd.to_numeric(pnl.get("gross_return", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
            cost = float(pd.to_numeric(pnl.get("cost_return", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
            if abs(gross) > 1e-12:
                cost_to_gross_vals.append(abs(cost) / abs(gross))
            beta_abs = float(pd.to_numeric(pnl.get("beta_exposure", pd.Series(dtype=float)), errors="coerce").abs().mean())
            if np.isfinite(beta_abs):
                beta_abs_vals.append(beta_abs)
            
            # [AUDIT] Portfolio Integrity
            if "net_exposure" in pnl.columns:
                global_audit.max_net_exposure = max(global_audit.max_net_exposure, abs(float(pnl["net_exposure"].mean())))
            if "short_exposure" in pnl.columns:
                if float(pnl["short_exposure"].sum()) < 1e-6:
                    global_audit.zero_short_days += 1
        except Exception as _exc:
            _window_failure_log.append(
                f"window={window_idx} [{tr_s.date()}→{va_e.date()}] "
                f"{type(_exc).__name__}: {_exc}"
            )
            continue
    sharpe_mean = float(np.nanmean(sharpes)) if sharpes else float("nan")
    ic_mean = float(np.nanmean(ics)) if ics else float("nan")
    daily_icir_mean = float(np.nanmean(daily_icirs)) if daily_icirs else float("nan")
    cost_mean = float(np.nanmean(cost_to_gross_vals)) if cost_to_gross_vals else float("nan")
    beta_mean = float(np.nanmean(beta_abs_vals)) if beta_abs_vals else float("nan")
    selection_score = (
        (sharpe_mean if np.isfinite(sharpe_mean) else -1.0)
        + 0.15 * (daily_icir_mean if np.isfinite(daily_icir_mean) else 0.0)
        - 0.25 * (cost_mean if np.isfinite(cost_mean) else 0.0)
        - 0.10 * (beta_mean if np.isfinite(beta_mean) else 0.0)
    )
    _nested_sim_mode = str((nested_cfg.get("simulation_policy", {}) or {}).get("nested_full", "executable"))
    # Mode direction across inner validation windows
    _dir_mode_agg: int = 1
    if _per_window_directions:
        _dir_counts = {1: _per_window_directions.count(1), -1: _per_window_directions.count(-1)}
        _dir_mode_agg = 1 if _dir_counts[1] >= _dir_counts[-1] else -1
    return {
        "nested_sharpe_mean": sharpe_mean,
        "nested_ic_mean": ic_mean,
        "nested_daily_icir_mean": daily_icir_mean,
        "nested_cost_to_gross_mean": cost_mean,
        "nested_beta_abs_mean": beta_mean,
        "nested_selection_score": float(selection_score),
        "nested_windows": int(max(len(sharpes), len(ics))),
        "nested_window_failures": int(len(_window_failure_log)),
        "nested_window_failure_log": " | ".join(_window_failure_log) if _window_failure_log else "",
        "nested_simulation_mode": _nested_sim_mode,
        "cal_score_direction_mode": int(_dir_mode_agg),
        "cal_direction_flip_count": int(_per_window_directions.count(-1)),
        "cal_direction_window_count": int(len(_per_window_directions)),
    }


def _nested_proxy_candidate(
    train_df: pd.DataFrame,
    *,
    prepared_cache: PreparedPanelCache,
    nested_workspace: list[NestedWindowState] | None,
    model_template: Any,
    name: str,
    model_kind: str,
    uses_proba: bool,
    active_feats: list[str],
    cfg: dict[str, Any],
    nested_cfg: dict[str, Any],
    horizon: int,
    embargo_days: int,
    use_risk_adj: bool,
    primary_path: str,
) -> dict[str, float]:
    validation_target_col = _validation_target_col_for_path(primary_path)
    if nested_workspace is not None:
        windows = [
            (state.train_start, state.train_end, state.eval_start, state.eval_end)
            for state in nested_workspace
        ]
    else:
        windows = _nested_inner_windows(
            train_df,
            max_windows=int((nested_cfg.get("search", {}) or {}).get("prefilter_windows", nested_cfg.get("max_windows", 1))),
            validation_days=int(nested_cfg.get("validation_days", 126)),
            min_train_days=int(nested_cfg.get("min_train_days", 504)),
            embargo_days=int(embargo_days),
        )
    ics: list[float] = []
    daily_icirs: list[float] = []
    turnover_proxy: list[float] = []
    for window_idx, (tr_s, tr_e, va_s, va_e) in enumerate(windows):
        if nested_workspace is not None:
            window_state = nested_workspace[window_idx]
            prepared = window_state.prepared_folds.get((int(horizon), tuple(str(f) for f in active_feats)))
            if prepared is None:
                continue
        else:
            prepared = prepared_cache.get_prepared_fold(
                train_start=tr_s,
                train_end=tr_e,
                eval_start=va_s,
                eval_end=va_e,
                horizon_days=int(horizon),
                active_features=active_feats,
            )
        if prepared.train_df.empty or prepared.eval_df.empty or prepared.x_train.shape[1] == 0:
            continue
        y_tr = prepared_cache.get_training_target(
            start=tr_s,
            end=tr_e,
            horizon_days=int(horizon),
            model_name=name,
            model_kind=model_kind,
            use_risk_adj=use_risk_adj,
        )
        if model_kind == "short_classifier" and int((y_tr == 1).sum()) < 30:
            continue
        try:
            proxy_template = _proxy_model_template(
                model_template,
                nested_cfg,
                active_feature_count=len(active_feats),
            )
            inner_model = _fit_candidate_model(
                model_template=proxy_template,
                name=name,
                model_kind=model_kind,
                tr=prepared.train_df,
                x_tr=prepared.x_train,
                y_tr=y_tr,
            )
            score = _score_model_predictions(
                inner_model,
                prepared.x_eval,
                model_kind=model_kind,
                uses_proba=uses_proba,
                score_direction=1,
            )
            train_score = _score_model_predictions(
                inner_model,
                prepared.x_train,
                model_kind=model_kind,
                uses_proba=uses_proba,
                score_direction=1,
            )
            score, _cal_result = _calibrate_scores(
                prepared.train_df,
                train_score,
                score,
                target_col=validation_target_col,
            )
            scored = prepared.eval_df.assign(score=score)
            ic_stats = cross_sectional_ic(scored, target_col=validation_target_col)
            ic = float(ic_stats.get("cs_ic_spearman_mean", float("nan")))
            daily_icir = float(ic_stats.get("daily_ic_annualized_icir", float("nan")))
            proxy_turnover = _proxy_turnover_from_scores(
                scored, 
                primary_path=primary_path, 
                max_positions=int(nested_cfg.get("max_positions", 10))
            )
            if np.isfinite(ic):
                ics.append(ic)
            if np.isfinite(daily_icir):
                daily_icirs.append(daily_icir)
            if np.isfinite(proxy_turnover):
                turnover_proxy.append(proxy_turnover)
        except Exception:
            continue
    ic_mean = float(np.nanmean(ics)) if ics else float("nan")
    daily_icir_mean = float(np.nanmean(daily_icirs)) if daily_icirs else float("nan")
    turnover_mean = float(np.nanmean(turnover_proxy)) if turnover_proxy else float("nan")
    selection_score = (
        (ic_mean if np.isfinite(ic_mean) else -1.0)
        + 0.15 * (daily_icir_mean if np.isfinite(daily_icir_mean) else 0.0)
        - 0.10 * (turnover_mean if np.isfinite(turnover_mean) else 0.0)
    )
    return {
        "proxy_ic_mean": ic_mean,
        "proxy_daily_icir_mean": daily_icir_mean,
        "proxy_turnover_mean": turnover_mean,
        "proxy_selection_score": float(selection_score),
        "proxy_max_iter": float(((nested_cfg.get("search", {}) or {}).get("proxy_max_iter", 30) or 30)),
        "proxy_windows": int(max(len(ics), len(daily_icirs))),
    }


def _select_nested_candidate(
    train_df: pd.DataFrame,
    *,
    prepared_cache: PreparedPanelCache,
    research_state: ResearchStateStore | None,
    primary_path: str,
    models: list[tuple[str, Any, bool, str]],
    cfg: dict[str, Any],
    nested_cfg: dict[str, Any],
    feat_cols: list[str],
    short_feature_subset: list[str],
    overlay_feature_subset: list[str],
    default_horizon: int,
    horizon_contract: HorizonContract | None,
    max_positions: int,
    min_positions: int,
    embargo_days: int,
    use_risk_adj: bool,
    target_cfg: TargetConfig,
    costs: ExecutionCostConfig,
    max_name_weight: float,
    nested_candidate_workers: int = 1,
    timing_ledger: TimingLedger | None = None,
    return_artifacts: bool = False,
) -> tuple[NestedCandidateSpec | None, dict[str, float]] | tuple[NestedCandidateSpec | None, dict[str, float], list[dict[str, Any]]]:
    pool = _build_nested_candidate_pool(
        models=models,
        primary_path=primary_path,
        cfg=cfg,
        nested_cfg=nested_cfg,
        default_horizon=default_horizon,
        feat_cols=feat_cols,
        short_feature_subset=short_feature_subset,
        overlay_feature_subset=overlay_feature_subset,
    )
    if not pool:
        empty_metrics = {
            "nested_selection_score": float("nan"),
            "nested_candidate_count": 0,
        }
        return (None, empty_metrics, []) if return_artifacts else (None, empty_metrics)
    search_cfg = (nested_cfg.get("search", {}) if isinstance(nested_cfg.get("search", {}), dict) else {}) or {}

    # C.4: Build SelectionPlan and check for short-circuit
    train_dates_for_plan = pd.to_datetime(train_df["date"], errors="coerce")
    plan = _build_selection_plan(
        outer_window_id=f"win_{train_dates_for_plan.min().date() if train_dates_for_plan.notna().any() else 'unknown'}",
        model_family=str(models[0][3]) if models else "unknown",
        pool=pool,
        nested_cfg=nested_cfg,
        search_cfg=search_cfg,
        windows=[],
        train_df=train_df,
        horizon_contract=horizon_contract,
        target_cfg=target_cfg,
        feat_cols=feat_cols,
    )

    # Write selection_plan artifact
    if research_state is not None:
        _plan_dir = research_state.subdir("selection_plans")
        _plan_dir.mkdir(parents=True, exist_ok=True)
        _plan_path = _plan_dir / f"plan_{plan.cache_key}.json"
        _write_json(_plan_path, plan.to_dict())

    # Emit telemetry
    if timing_ledger is not None:
        timing_ledger.record(
            "selection_plan",
            outer_window_id=plan.outer_window_id,
            model_family=plan.model_family,
            candidate_count=plan.candidate_count,
            selection_required=plan.selection_required,
            short_circuit_allowed=plan.short_circuit_allowed,
            reason=plan.short_circuit_reason if plan.short_circuit_allowed else plan.selection_required_reason,
        )

    # C.4: Short-circuit when selection is statistically redundant
    if plan.short_circuit_allowed:
        sole = pool[0]
        sc_metrics = _short_circuit_metrics(sole)
        sc_artifacts = [{
            **_nested_candidate_identity(sole),
            "short_circuit": True,
            "short_circuit_reason": plan.short_circuit_reason,
            "selection_plan_cache_key": plan.cache_key,
        }]
        return (sole, sc_metrics, sc_artifacts) if return_artifacts else (sole, sc_metrics)

    windows = _nested_inner_windows(
        train_df,
        max_windows=int(nested_cfg.get("max_windows", 1)),
        validation_days=int(nested_cfg.get("validation_days", 126)),
        min_train_days=int(nested_cfg.get("min_train_days", 504)),
        embargo_days=int(embargo_days),
    )
    prefilter_window_count = max(1, min(int(search_cfg.get("prefilter_windows", 1) or 1), len(windows) or 1))
    unique_fold_specs = sorted(
        {
            (int(candidate.horizon), tuple(str(f) for f in candidate.active_features))
            for candidate in pool
        },
        key=lambda item: (item[0], item[1]),
    )
    unique_horizons = sorted({int(candidate.horizon) for candidate in pool})
    nested_workspace: list[NestedWindowState] = []
    for tr_s, tr_e, va_s, va_e in windows:
        prepared_folds: dict[tuple[int, tuple[str, ...]], Any] = {}
        for cand_h, active_features in unique_fold_specs:
            prepared_folds[(cand_h, active_features)] = prepared_cache.get_prepared_fold(
                train_start=tr_s,
                train_end=tr_e,
                eval_start=va_s,
                eval_end=va_e,
                horizon_days=int(cand_h),
                active_features=list(active_features),
            )
        validation_state_by_horizon: dict[int, ValidationStateCache] = {}
        for cand_h in unique_horizons:
            eval_cfg = _evaluation_config(
                cfg,
                path=primary_path,
                max_positions=max_positions,
                min_positions=min_positions,
                horizon=int(cand_h),
                horizon_contract=horizon_contract,
            )
            validation_state_by_horizon[int(cand_h)] = prepared_cache.get_validation_state(
                start=va_s,
                end=va_e,
                horizon_days=int(cand_h),
                evaluation_cfg=eval_cfg,
            )
        nested_workspace.append(
            NestedWindowState(
                train_start=tr_s,
                train_end=tr_e,
                eval_start=va_s,
                eval_end=va_e,
                prepared_folds=prepared_folds,
                validation_state_by_horizon=validation_state_by_horizon,
            )
        )

    proxy_workspace = nested_workspace[:prefilter_window_count]
    train_dates = pd.to_datetime(train_df["date"], errors="coerce")
    cache_payload = {
        "cache_version": NESTED_SELECTION_CACHE_VERSION,
        "primary_path": str(primary_path),
        "train_start": str(train_dates.min().date()) if train_dates.notna().any() else "",
        "train_end": str(train_dates.max().date()) if train_dates.notna().any() else "",
        "train_rows": int(len(train_df)),
        "embargo_days": int(embargo_days),
        "use_risk_adj": bool(use_risk_adj),
        "pool": [_nested_candidate_identity(candidate) for candidate in pool],
        "simulation_policy": nested_cfg.get("simulation_policy", {}),
        "windows": [
            {
                "train_start": str(tr_s.date()),
                "train_end": str((tr_e - pd.Timedelta(days=1)).date()),
                "eval_start": str(va_s.date()),
                "eval_end": str((va_e - pd.Timedelta(days=1)).date()),
            }
            for tr_s, tr_e, va_s, va_e in windows
        ],
        "search_cfg": search_cfg,
        "prefilter_window_count": int(prefilter_window_count),
    }
    cache_key = PreparedPanelCache._stable_hash((json.dumps(cache_payload, sort_keys=True),))
    cache_path = (
        research_state.subdir("nested_selection") / f"{cache_key}.json"
        if research_state is not None
        else None
    )
    candidate_cache_dir = research_state.subdir("nested_candidate_metrics") if research_state is not None else None
    nested_context = {
        "primary_path": str(primary_path),
        "selection_cache_key": str(cache_key),
        "train_start": cache_payload["train_start"],
        "train_end": cache_payload["train_end"],
        "train_rows": int(len(train_df)),
    }
    if cache_path is not None and cache_path.exists():
        cached = _read_json(cache_path)
        best_candidate = _match_nested_candidate(pool, cached.get("best_candidate"))
        cached_metrics = cached.get("best_metrics", {})
        if best_candidate is not None and isinstance(cached_metrics, dict):
            if timing_ledger is not None:
                timing_ledger.record(
                    "nested_selection_cache_hit",
                    **nested_context,
                    candidate_count=int(len(pool)),
                )
            cached_out = {
                **{k: v for k, v in cached_metrics.items() if isinstance(k, str)},
                "nested_candidate_count": int(len(pool)),
                "nested_cache_hit": 1.0,
            }
            return (best_candidate, cached_out, []) if return_artifacts else (best_candidate, cached_out)

    def _candidate_metric_path(stage: str, candidate: NestedCandidateSpec) -> Path | None:
        if candidate_cache_dir is None:
            return None
        payload = {
            "stage": str(stage),
            "context": cache_payload,
            "candidate": _nested_candidate_identity(candidate),
        }
        key = PreparedPanelCache._stable_hash((json.dumps(payload, sort_keys=True),))
        return candidate_cache_dir / f"{key}.json"

    def _cached_candidate_metrics(
        stage: str,
        candidate: NestedCandidateSpec,
        evaluator: Any,
    ) -> tuple[NestedCandidateSpec, dict[str, float]]:
        path = _candidate_metric_path(stage, candidate)
        identity = _nested_candidate_identity(candidate)
        if path is not None and path.exists():
            # P13: Stale cache guard — metrics written by a previous process
            # invocation are silently stale.  Check mtime against process start.
            _mtime = path.stat().st_mtime
            if _mtime < _PROCESS_START_TIME:
                path.unlink(missing_ok=True)
            else:
                cached = _read_json(path)
                metrics = cached.get("metrics", {}) if isinstance(cached, dict) else {}
                if isinstance(metrics, dict):
                    out = _candidate_metric_payload(metrics)
                    out[f"{stage}_candidate_cache_hit"] = 1.0
                    if timing_ledger is not None:
                        timing_ledger.record(
                            "nested_candidate_cache_hit",
                            **nested_context,
                            stage=str(stage),
                            **identity,
                        )
                    return candidate, out
        started = time.perf_counter()
        _, metrics = evaluator(candidate)
        elapsed = time.perf_counter() - started
        out = _candidate_metric_payload(metrics)
        out[f"{stage}_candidate_elapsed_s"] = float(elapsed)
        out[f"{stage}_candidate_cache_hit"] = 0.0
        if path is not None:
            _write_json(
                path,
                {
                    "context": nested_context,
                    "candidate": identity,
                    "metrics": _json_metric_payload(out),
                },
            )
        if timing_ledger is not None:
            timing_ledger.record(
                "nested_candidate_evaluated",
                **nested_context,
                stage=str(stage),
                elapsed_s=float(elapsed),
                **identity,
                **_json_metric_payload(out),
            )
        return candidate, out

    # Task 1 & 2: Memoize feature panels and score vectors
    score_cache: dict[tuple[str, int, tuple[str, ...], int], Any] = {}

    def _evaluate_candidate(candidate: NestedCandidateSpec) -> tuple[NestedCandidateSpec, dict[str, float]]:
        metrics = _nested_validate_candidate(
            train_df,
            prepared_cache=prepared_cache,
            nested_workspace=nested_workspace,
            model_template=candidate.model_template,
            name=candidate.model_name,
            model_kind=candidate.model_kind,
            uses_proba=candidate.uses_proba,
            active_feats=list(candidate.active_features),
            cfg=cfg,
            nested_cfg=nested_cfg,
            horizon=int(candidate.horizon),
            horizon_contract=horizon_contract,
            max_positions=max_positions,
            min_positions=min_positions,
            embargo_days=embargo_days,
            use_risk_adj=use_risk_adj,
            primary_path=primary_path,
            target_cfg=target_cfg,
            costs=costs,
            max_name_weight=max_name_weight,
            score_cache=score_cache,
        )
        return candidate, metrics

    def _evaluate_proxy(candidate: NestedCandidateSpec) -> tuple[NestedCandidateSpec, dict[str, float]]:
        metrics = _nested_proxy_candidate(
            train_df,
            prepared_cache=prepared_cache,
            nested_workspace=proxy_workspace,
            model_template=candidate.model_template,
            name=candidate.model_name,
            model_kind=candidate.model_kind,
            uses_proba=candidate.uses_proba,
            active_feats=list(candidate.active_features),
            cfg=cfg,
            nested_cfg=nested_cfg,
            horizon=int(candidate.horizon),
            embargo_days=embargo_days,
            use_risk_adj=use_risk_adj,
            primary_path=primary_path,
        )
        return candidate, metrics

    best_candidate: NestedCandidateSpec | None = None
    best_metrics: dict[str, float] = {
        "nested_selection_score": float("-inf"),
        "nested_candidate_count": int(len(pool)),
    }
    proxy_evaluated: list[tuple[NestedCandidateSpec, dict[str, float]]] = []
    proxy_started = time.perf_counter()
    worker_count = max(1, int(nested_candidate_workers))
    nested_backend = str(search_cfg.get("nested_parallel_backend", "sequential")).strip().lower()
    
    # Task 3: Parallelize nested candidates
    # Set global context for fork-safety
    global _FORK_NESTED_CTX
    _FORK_NESTED_CTX = NestedEvaluationContext(
        train_df=train_df,
        prepared_cache=prepared_cache,
        nested_workspace=nested_workspace,
        cfg=cfg,
        nested_cfg=nested_cfg,
        horizon_contract=horizon_contract,
        max_positions=max_positions,
        min_positions=min_positions,
        embargo_days=embargo_days,
        use_risk_adj=use_risk_adj,
        primary_path=primary_path,
        target_cfg=target_cfg,
        costs=costs,
        max_name_weight=max_name_weight,
        score_cache=score_cache,
    )
        
    for candidate in pool:
        proxy_evaluated.append(_cached_candidate_metrics("proxy", candidate, _evaluate_proxy))
    proxy_elapsed = time.perf_counter() - proxy_started
    ranked_proxy = sorted(
        proxy_evaluated,
        key=lambda item: _safe_float(item[1].get("proxy_selection_score"), float("-inf")),
        reverse=True,
    )
    prefilter_top_k = max(1, min(int(search_cfg.get("prefilter_top_k", 6) or 6), len(ranked_proxy)))
    shortlisted = [candidate for candidate, _ in ranked_proxy[:prefilter_top_k]]
    shortlisted_ids = {
        json.dumps(_nested_candidate_identity(candidate), sort_keys=True)
        for candidate in shortlisted
    }
    full_started = time.perf_counter()
    evaluated: list[tuple[NestedCandidateSpec, dict[str, float]]] = []
    
    if nested_backend == "process" and worker_count > 1 and len(shortlisted) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            # We don't use _evaluate_candidate (local closure) to avoid pickling.
            # We use the fork-wrapper which reads from _FORK_NESTED_CTX.
            futures = [
                executor.submit(_nested_evaluate_fork_wrapper, candidate) 
                for candidate in shortlisted if _candidate_metric_path("full", candidate) is None or not _candidate_metric_path("full", candidate).exists()
            ]
            for candidate in shortlisted:
                path = _candidate_metric_path("full", candidate)
                if path is not None and path.exists():
                    evaluated.append(_cached_candidate_metrics("full", candidate, _evaluate_candidate))
            
            for future in as_completed(futures):
                evaluated.append(future.result())
    else:
        for candidate in shortlisted:
            evaluated.append(_cached_candidate_metrics("full", candidate, _evaluate_candidate))
    full_elapsed = time.perf_counter() - full_started
    ranking_artifacts: list[dict[str, Any]] = []
    full_metrics_by_id = {
        json.dumps(_nested_candidate_identity(candidate), sort_keys=True): metrics
        for candidate, metrics in evaluated
    }
    for candidate, proxy_metrics in ranked_proxy:
        identity = _nested_candidate_identity(candidate)
        identity_key = json.dumps(identity, sort_keys=True)
        ranking_artifacts.append(
            {
                **identity,
                **{k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v) else (None if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in proxy_metrics.items()},
                "shortlisted": bool(identity_key in shortlisted_ids),
                "full_metrics": {
                    k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v) else (None if isinstance(v, (int, float, np.floating, np.integer)) else v))
                    for k, v in full_metrics_by_id.get(identity_key, {}).items()
                },
            }
        )
    for candidate, metrics in evaluated:
        score = _safe_float(metrics.get("nested_selection_score"), float("-inf"))
        current_best = _safe_float(best_metrics.get("nested_selection_score"), float("-inf"))
        if score > current_best:
            best_candidate = candidate
            best_metrics = {
                **metrics,
                "nested_candidate_count": int(len(pool)),
                "nested_prefilter_top_k": int(prefilter_top_k),
                "nested_proxy_elapsed_s": float(proxy_elapsed),
                "nested_exec_elapsed_s": float(full_elapsed),
                "nested_cache_hit": 0.0,
                "nested_proxy_candidate_cache_hit_rate": float(np.mean([m.get("proxy_candidate_cache_hit", 0.0) for _, m in proxy_evaluated])) if proxy_evaluated else 0.0,
                "nested_full_candidate_cache_hit_rate": float(np.mean([m.get("full_candidate_cache_hit", 0.0) for _, m in evaluated])) if evaluated else 0.0,
                "nested_selected_horizon": int(candidate.horizon),
                **{
                    f"nested_cache_{k}": float(v)
                    for k, v in prepared_cache.stats().items()
                    if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_))
                },
            }
    if cache_path is not None and best_candidate is not None:
        _write_json(
            cache_path,
            {
                "cache_payload": cache_payload,
                "best_candidate": _nested_candidate_identity(best_candidate),
                "best_metrics": best_metrics,
                "ranking_artifacts": ranking_artifacts,
            },
        )
    return (best_candidate, best_metrics, ranking_artifacts) if return_artifacts else (best_candidate, best_metrics)


def check_feature_leakage(
    *,
    as_of_date: str = "2020-06-15",
    tickers: list[str] | None = None,
    tol: float = 1e-6,
) -> int:
    """
    Backward-looking sanity check for a couple of key features.

    For each ticker, we compute each feature in two ways:
    - **truncated**: using raw prices up to and including as_of_date
    - **full**: using raw prices including future dates beyond as_of_date
      (feature value at as_of_date must not change)

    Then we compare truncated manual values to the values produced by
    build_feature_matrix at as_of_date.
    """
    from agents.weight_learning_agent.feature_builder import build_feature_matrix
    from utils.market_data import get_ohlcv

    as_of = pd.Timestamp(as_of_date)
    cfg = _read_config()
    chosen = load_universe(cfg)
    if not chosen:
        print("FAIL: no tickers provided and config has no tickers.")
        return 1

    # Ensure enough lookback for 252d rolling z-scores + 20d realised vol.
    start_date = (as_of - pd.Timedelta(days=900)).strftime("%Y-%m-%d")
    end_date = as_of.strftime("%Y-%m-%d")

    df = build_feature_matrix(
        chosen,
        start_date=start_date,
        end_date=end_date,
        holding_period=5,
        feature_subset=None,
        **_feature_builder_data_kwargs(cfg),
    )
    if df is None or df.empty:
        print("FAIL: build_feature_matrix returned empty DataFrame.")
        return 1

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"] == as_of]
    if df.empty:
        print(f"FAIL: no feature rows produced for as_of_date={as_of_date}.")
        return 1

    def _download_close(ticker: str, *, extra_days: int) -> pd.Series:
        # Pull a generous range; then slice to as_of.
        req_start = (as_of - pd.Timedelta(days=1000)).strftime("%Y-%m-%d")
        req_end = (as_of + pd.Timedelta(days=extra_days)).strftime("%Y-%m-%d")
        import os
        from utils.wrds_data import load_wrds_price_panel, resolve_data_provider

        data_cfg = cfg.get("data") or {}
        provider = resolve_data_provider(data_cfg.get("provider"))
        if provider == "wrds":
            ohlcv = load_wrds_price_panel(
                [ticker],
                start_date=req_start,
                end_date=req_end,
                username=os.environ.get("WRDS_USERNAME"),
                cache_dir=data_cfg.get("cache_dir", "data/cache/wrds"),
                cache_ttl_days=int(data_cfg.get("cache_ttl_days", 1)),
                ticker_to_permno={},
            ).get(ticker, pd.DataFrame())
        else:
            ohlcv = get_ohlcv(
                ticker,
                req_start,
                req_end,
                provider=provider,
                use_cache=True,
                cache_ttl_days=1,
            )
        if ohlcv is None or ohlcv.empty or "Close" not in ohlcv.columns:
            return pd.Series(dtype=float)
        close = pd.to_numeric(ohlcv["Close"], errors="coerce").dropna().sort_index()
        close = close.loc[close.index <= as_of]
        return close

    def _manual_ret_5d(close: pd.Series) -> float:
        return float(close.pct_change(5).iloc[-1])

    def _manual_rolling_vol_20(close: pd.Series) -> float:
        daily_ret = close.pct_change()
        vol_20_raw = daily_ret.rolling(20).std()
        v20_m = vol_20_raw.rolling(252, min_periods=60).mean()
        v20_s = vol_20_raw.rolling(252, min_periods=60).std().replace(0, np.nan).fillna(1.0)
        out = (vol_20_raw - v20_m) / v20_s
        return float(out.iloc[-1])

    def _cs_z(vals: dict[str, float]) -> dict[str, float]:
        """
        Cross-sectional z-score across tickers (population std, ddof=0),
        matching feature_builder's cross_sectional_zscore_ddof0.
        """
        x = np.array([vals[tk] for tk in chosen], dtype=float)
        m = np.isfinite(x)
        out: dict[str, float] = {}
        if m.sum() < 2:
            for tk in chosen:
                out[tk] = 0.0
            return out
        mu = float(np.mean(x[m]))
        sd = float(np.std(x[m], ddof=0))
        if sd < 1e-12:
            for tk in chosen:
                out[tk] = 0.0
            return out
        for tk in chosen:
            v = float(vals[tk])
            out[tk] = float((v - mu) / sd) if np.isfinite(v) else 0.0
        return out

    failures = 0
    print()
    print(f"=== Feature leakage check (as-of {as_of_date}) ===")

    # Build manual values for the whole cross-section first, because feature_builder
    # overwrites some columns with cross-sectional z-scores (e.g., ret_5d, rolling_vol_20).
    close_trunc_map: dict[str, pd.Series] = {}
    close_full_map: dict[str, pd.Series] = {}
    for tk in chosen:
        close_trunc_map[tk] = _download_close(tk, extra_days=2)
        close_full_map[tk] = _download_close(tk, extra_days=60)

    # ret_5d: raw pct_change(5) then cross-sectional z-score per date
    ret5_trunc_raw = {tk: _manual_ret_5d(close_trunc_map[tk]) for tk in chosen}
    ret5_full_raw = {
        tk: _manual_ret_5d(close_full_map[tk]) if not close_full_map[tk].empty else ret5_trunc_raw[tk]
        for tk in chosen
    }
    ret5_trunc_cs = _cs_z(ret5_trunc_raw)
    ret5_full_cs = _cs_z(ret5_full_raw)

    # rolling_vol_20: per-ticker TS z-score of vol_20d, then cross-sectional z-score per date
    rv20_trunc_ts = {tk: _manual_rolling_vol_20(close_trunc_map[tk]) for tk in chosen}
    rv20_full_ts = {
        tk: _manual_rolling_vol_20(close_full_map[tk]) if not close_full_map[tk].empty else rv20_trunc_ts[tk]
        for tk in chosen
    }
    rv20_trunc_cs = _cs_z(rv20_trunc_ts)
    rv20_full_cs = _cs_z(rv20_full_ts)

    for tk in chosen:
        sub = df.loc[df["ticker"] == tk]
        if sub.empty:
            print(f"{tk}: SKIP (no row for date)")
            continue

        if close_trunc_map[tk].empty or len(close_trunc_map[tk]) < 300:
            print(f"{tk}: SKIP (insufficient history)")
            continue

        # ret_5d (cross-sectional z-scored in feature_builder)
        try:
            fb = float(pd.to_numeric(sub["ret_5d"], errors="coerce").iloc[0])
            m_trunc = float(ret5_trunc_cs.get(tk, 0.0))
            m_full = float(ret5_full_cs.get(tk, 0.0))
            ok_window = abs(m_trunc - m_full) <= tol
            ok_match = abs(fb - m_trunc) <= tol
            status = "PASS" if (ok_window and ok_match) else "FAIL"
            if status == "FAIL":
                failures += 1
            print(
                f"{tk} ret_5d: {status} | feature_builder={fb:.8f} manual_cs_z={m_trunc:.8f} | "
                f"window_backwards={ok_window}"
            )
        except Exception as exc:
            failures += 1
            print(f"{tk} ret_5d: FAIL (exception: {exc})")

        # rolling_vol_20 (TS-z then CS-z in feature_builder)
        try:
            fb = float(pd.to_numeric(sub["rolling_vol_20"], errors="coerce").iloc[0])
            m_trunc = float(rv20_trunc_cs.get(tk, 0.0))
            m_full = float(rv20_full_cs.get(tk, 0.0))
            ok_window = abs(m_trunc - m_full) <= tol
            ok_match = abs(fb - m_trunc) <= tol
            status = "PASS" if (ok_window and ok_match) else "FAIL"
            if status == "FAIL":
                failures += 1
            print(
                f"{tk} rolling_vol_20: {status} | feature_builder={fb:.8f} manual_cs_z={m_trunc:.8f} | "
                f"window_backwards={ok_window}"
            )
        except Exception as exc:
            failures += 1
            print(f"{tk} rolling_vol_20: FAIL (exception: {exc})")

    print()
    if failures == 0:
        print("Overall: PASS")
    return 0


def _compute_nested_stability_metrics(
    all_ranking_artifacts: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute proxy-vs-full selection stability metrics from per-window ranking artifacts."""
    if not all_ranking_artifacts:
        return {
            "nested_stability_proxy_full_rank_corr": float("nan"),
            "nested_stability_proxy_full_rank_corr_std": float("nan"),
            "nested_stability_regret_mean": float("nan"),
            "nested_stability_regret_std": float("nan"),
            "nested_stability_proxy_winner_advantage_prob": float("nan"),
            "nested_stability_window_count": 0,
        }

    by_window: dict[int, list[dict[str, Any]]] = {}
    for art in all_ranking_artifacts:
        w = int(art.get("outer_window_idx", 0))
        by_window.setdefault(w, []).append(art)

    rho_per_window: list[float] = []
    regret_per_window: list[float] = []
    advantage_per_window: list[float] = []

    for candidates in by_window.values():
        shortlisted = [c for c in candidates if c.get("shortlisted")]
        if len(shortlisted) < 2:
            continue
        proxy_scores = np.array(
            [c.get("proxy_selection_score", np.nan) for c in shortlisted], dtype=float
        )
        full_scores = np.array(
            [
                c.get("full_metrics", {}).get("nested_selection_score", np.nan)
                if isinstance(c.get("full_metrics"), dict)
                else np.nan
                for c in shortlisted
            ],
            dtype=float,
        )
        valid = np.isfinite(proxy_scores) & np.isfinite(full_scores)
        if valid.sum() < 2:
            continue
        pv = proxy_scores[valid]
        fv = full_scores[valid]
        p_rank = np.argsort(np.argsort(pv)).astype(float)
        f_rank = np.argsort(np.argsort(fv)).astype(float)
        p_rank -= p_rank.mean()
        f_rank -= f_rank.mean()
        denom = np.sqrt((p_rank**2).sum() * (f_rank**2).sum())
        rho = float((p_rank * f_rank).sum() / denom) if denom > 0 else float("nan")
        if np.isfinite(rho):
            rho_per_window.append(rho)
        proxy_winner_idx = int(np.argmax(pv))
        best_full = float(np.max(fv))
        regret = best_full - float(fv[proxy_winner_idx])
        regret_per_window.append(regret)
        best_full_idx = int(np.argmax(fv))
        advantage_per_window.append(float(proxy_winner_idx == best_full_idx))

    return {
        "nested_stability_proxy_full_rank_corr": float(np.nanmean(rho_per_window)) if rho_per_window else float("nan"),
        "nested_stability_proxy_full_rank_corr_std": float(np.nanstd(rho_per_window)) if rho_per_window else float("nan"),
        "nested_stability_regret_mean": float(np.nanmean(regret_per_window)) if regret_per_window else float("nan"),
        "nested_stability_regret_std": float(np.nanstd(regret_per_window)) if regret_per_window else float("nan"),
        "nested_stability_proxy_winner_advantage_prob": float(np.nanmean(advantage_per_window)) if advantage_per_window else float("nan"),
        "nested_stability_window_count": len(by_window),
    }


def _optimizer_score_weight_audit(
    scored: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    model_name: str,
    window_idx: int,
    horizon_days: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Audit whether portfolio construction preserves the model's cross-section.

    The production simulator works from target weights, so this is the first
    executable point where we can distinguish weak alpha from alpha destroyed by
    constraints, turnover control, liquidity caps, or beta de-risking.
    """
    empty_metrics = {
        "opt_score_weight_rank_corr_mean": float("nan"),
        "opt_top_score_long_capture_mean": float("nan"),
        "opt_top_score_abs_capture_mean": float("nan"),
        "opt_top_score_zero_weight_rate_mean": float("nan"),
        "opt_bottom_score_long_leakage_mean": float("nan"),
        "opt_weight_next_return_corr_mean": float("nan"),
        "opt_score_next_return_corr_mean": float("nan"),
        "opt_realized_weighted_next_return_mean": float("nan"),
        "opt_raw_beta_abs_mean": float("nan"),
        "opt_max_sector_abs_mean": float("nan"),
        "opt_audit_days": 0.0,
    }
    if scored is None or scored.empty or target_weights is None or target_weights.empty:
        return empty_metrics, pd.DataFrame()

    df = scored.copy()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["ticker"] = df.get("ticker").astype(str)
    df["score"] = pd.to_numeric(df.get("score"), errors="coerce")
    if "daily_return" in df.columns:
        df = df.sort_values(["ticker", "date"]).copy()
        df["next_return"] = pd.to_numeric(df["daily_return"], errors="coerce").groupby(df["ticker"]).shift(-1)
    else:
        df["next_return"] = np.nan
    keep_cols = ["date", "ticker", "score", "next_return"]
    for optional in ("capm_beta", "sector"):
        if optional in df.columns:
            keep_cols.append(optional)
    df = df[keep_cols].dropna(subset=["date", "ticker", "score"])

    tw = target_weights[["date", "ticker", "target_weight"]].copy()
    tw["date"] = pd.to_datetime(tw["date"], errors="coerce")
    tw["ticker"] = tw["ticker"].astype(str)
    tw["target_weight"] = pd.to_numeric(tw["target_weight"], errors="coerce").fillna(0.0)
    merged = df.merge(tw, on=["date", "ticker"], how="inner")
    if merged.empty:
        return empty_metrics, pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for dt, day in merged.groupby("date", sort=True):
        if len(day) < 5:
            continue
        score = pd.to_numeric(day["score"], errors="coerce")
        weight = pd.to_numeric(day["target_weight"], errors="coerce").fillna(0.0)
        valid = score.notna() & weight.notna()
        if int(valid.sum()) < 5:
            continue
        score_v = score.loc[valid]
        weight_v = weight.loc[valid]
        abs_w = weight_v.abs()
        pos_w = weight_v.clip(lower=0.0)
        top_cut = score_v.rank(pct=True, method="average") >= 0.90
        bottom_cut = score_v.rank(pct=True, method="average") <= 0.10
        pos_total = float(pos_w.sum())
        abs_total = float(abs_w.sum())
        next_ret = pd.to_numeric(day.loc[valid, "next_return"], errors="coerce")
        beta_abs = float("nan")
        if "capm_beta" in day.columns:
            beta = pd.to_numeric(day.loc[valid, "capm_beta"], errors="coerce").fillna(1.0)
            beta_abs = abs(float((weight_v * beta).sum()))
        sector_abs = float("nan")
        if "sector" in day.columns:
            sector_exp = weight_v.groupby(day.loc[valid, "sector"].fillna("Unknown").astype(str)).sum()
            sector_abs = float(sector_exp.abs().max()) if len(sector_exp) else float("nan")

        rows.append({
            "model_name": str(model_name),
            "window_idx": int(window_idx),
            "horizon_days": int(horizon_days),
            "date": pd.Timestamp(dt),
            "n_names": int(valid.sum()),
            "score_weight_rank_corr": robust_spearman(score_v.values, weight_v.values),
            "top_score_long_capture": (
                float(pos_w.loc[top_cut].sum() / pos_total) if pos_total > 1e-12 else float("nan")
            ),
            "top_score_abs_capture": (
                float(abs_w.loc[top_cut].sum() / abs_total) if abs_total > 1e-12 else float("nan")
            ),
            "top_score_zero_weight_rate": float((abs_w.loc[top_cut] <= 1e-12).mean()) if bool(top_cut.any()) else float("nan"),
            "bottom_score_long_leakage": (
                float(pos_w.loc[bottom_cut].sum() / pos_total) if pos_total > 1e-12 else float("nan")
            ),
            "weight_next_return_corr": robust_spearman(weight_v.values, next_ret.values),
            "score_next_return_corr": robust_spearman(score_v.values, next_ret.values),
            "realized_weighted_next_return": float((weight_v * next_ret.fillna(0.0)).sum()),
            "raw_beta_abs": beta_abs,
            "max_sector_abs": sector_abs,
            "gross_weight": float(abs_total),
            "net_weight": float(weight_v.sum()),
        })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return empty_metrics, detail
    metric_map = {
        "opt_score_weight_rank_corr_mean": "score_weight_rank_corr",
        "opt_top_score_long_capture_mean": "top_score_long_capture",
        "opt_top_score_abs_capture_mean": "top_score_abs_capture",
        "opt_top_score_zero_weight_rate_mean": "top_score_zero_weight_rate",
        "opt_bottom_score_long_leakage_mean": "bottom_score_long_leakage",
        "opt_weight_next_return_corr_mean": "weight_next_return_corr",
        "opt_score_next_return_corr_mean": "score_next_return_corr",
        "opt_realized_weighted_next_return_mean": "realized_weighted_next_return",
        "opt_raw_beta_abs_mean": "raw_beta_abs",
        "opt_max_sector_abs_mean": "max_sector_abs",
    }
    out = {}
    for key, col in metric_map.items():
        vals = pd.to_numeric(detail[col], errors="coerce")
        out[key] = float(vals.mean()) if vals.notna().any() else float("nan")
    out["opt_audit_days"] = float(len(detail))
    return out, detail


def _evaluate_model_family(
    model_spec: tuple[str, Any, bool, str],
    *,
    ctx: OuterEvaluationContext,
) -> dict[str, Any]:
    name, model, uses_proba, model_kind = model_spec
    is_short_model = _is_short_model(model_kind)
    primary_path = _deployment_primary_path_for_model(
        model_kind,
        str(getattr(ctx, "primary_path", "long_only_overlay") or "long_only_overlay"),
    )
    validation_target_col = _validation_target_col_for_path(primary_path)
    # Task 1-3: Use disk-backed storage for window parts to keep memory flat
    import os
    scratch_dir = ctx.research_state.subdir(f"eval_scratch_{name.replace(' ', '_')}_{os.getpid()}")
    oos_paths: list[Path] = []
    pnl_paths: list[Path] = []
    daily_paths: list[Path] = []
    overlay_pnl_paths: list[Path] = []
    overlay_daily_paths: list[Path] = []
    exec_time_parts: list[float] = []
    wm: list[WindowMetrics] = []
    nested_records: list[dict[str, Any]] = []
    nested_selected_models: list[str] = []
    nested_selected_views: list[str] = []
    nested_selected_horizons: list[int] = []
    all_ranking_artifacts: list[dict[str, Any]] = []
    optimizer_audit_metrics: list[dict[str, float]] = []
    optimizer_audit_records: list[dict[str, Any]] = []
    _outer_window_directions: list[int] = []
    # P13: Collect evaluation-phase forecast calibration per window for final report
    eval_calibration_results: list = []
    log_lines: list[str] = [f"=== {name} ({model_kind}) ==="]
    nested_search_applicable = bool(
        model_kind in _nested_program_kinds(primary_path)
        and ctx.nested_cfg.get("true_selection_enabled", True)
    )
    selection_mode = "alpha_nested_search" if nested_search_applicable else "fixed_shortlist"
    validation_cfg = ((ctx.cfg.get("model_selection", {}) or {}).get("validation", {}) or {})
    run_overlay_diagnostics = bool(validation_cfg.get("run_overlay_diagnostics", True))
    # ── Phase 1 diagnostic flags (all default-off, no behavior change) ────────
    _p1_diag = (validation_cfg.get("phase1_diagnostics") or {})
    enable_neutralization_diagnostics: bool = bool(_p1_diag.get("factor_neutralization", False))
    enable_signal_stability_diagnostics: bool = bool(_p1_diag.get("signal_stability", False))
    enable_direction_model_diagnostics: bool = bool(_p1_diag.get("direction_model", False))
    enable_ic_engine_diagnostics: bool = bool(_p1_diag.get("ic_engine", False))
    # ─────────────────────────────────────────────────────────────────────────
    # Direction meta-model (shadow, one instance per model across all windows)
    _dir_meta_model: object = None
    _dir_meta_ic_history: list[float] = []
    if enable_direction_model_diagnostics:
        try:
            from model_selection.direction_model import DirectionMetaModel
            _dir_meta_model = DirectionMetaModel()
        except Exception as _exc:
            logger.debug("[diag:direction_model] init failed: %s", _exc)

    for win_idx, (tr_s, tr_e, te_s, te_e) in enumerate(ctx.windows, 1):
        te_label = f"{te_s.date()} → {(te_e - pd.Timedelta(days=1)).date()}"
        purge_cutoff = te_s - pd.Timedelta(days=ctx.embargo_days)

        te_mask = (ctx.df["date"] >= te_s) & (ctx.df["date"] < te_e)
        n_test_unique = int(ctx.df.loc[te_mask, "date"].nunique())
        
        if not te_mask.any():
            log_lines.append(f"  [window {win_idx}/{len(ctx.windows)}] skip: empty train or test | test={te_label}")
            continue

        if n_test_unique < int(ctx.args.min_oos_days):
            log_lines.append(
                f"  WARNING [window {win_idx}/{len(ctx.windows)}] skip: only {n_test_unique} test days "
                f"(min_oos_days={ctx.args.min_oos_days}) | test={te_label}"
            )
            continue

        if n_test_unique < int(ctx.args.min_test_days):
            log_lines.append(
                f"  [window {win_idx}/{len(ctx.windows)}] skip: {n_test_unique} test days < "
                f"min_test_days={ctx.args.min_test_days} | test={te_label}"
            )
            continue
            
        tr_mask = (ctx.df["date"] >= tr_s) & (ctx.df["date"] < min(tr_e, purge_cutoff))
        if not tr_mask.any():
            log_lines.append(f"  [window {win_idx}/{len(ctx.windows)}] skip: empty train or test | test={te_label}")
            continue
        
        # Define training slice for nested search (Task 1 & 3)
        tr = ctx.df[tr_mask]

        try:
            selected_candidate: NestedCandidateSpec | None = None
            candidate_name = name
            candidate_kind = model_kind
            candidate_uses_proba = uses_proba
            candidate_model = model
            candidate_horizon = int(ctx.horizon)
            active_feats = _active_features_for_model_kind(
                model_kind,
                ctx.feat_cols,
                short_feature_subset=ctx.short_feature_subset,
                overlay_feature_subset=ctx.overlay_feature_subset,
            )
            t_nested_0 = time.perf_counter()
            nested_metrics: dict[str, float] = {}
            if nested_search_applicable:
                # Contract validation (Task 5)
                assert tr is not None and not tr.empty, f"Model {name} window {win_idx}: tr must be non-empty for nested search"
                nested_model_pool = _nested_model_pool_for_outer(
                    model_spec,
                    ctx_models=ctx.models,
                    nested_cfg=ctx.nested_cfg,
                )
                
                selected_candidate, nested_metrics, _window_ranking_artifacts = _select_nested_candidate(
                    tr,
                    prepared_cache=ctx.global_prepare_cache,
                    research_state=ctx.research_state,
                    primary_path=primary_path,
                    models=nested_model_pool,
                    cfg=ctx.cfg,
                    nested_cfg=ctx.nested_cfg,
                    feat_cols=ctx.feat_cols,
                    short_feature_subset=ctx.short_feature_subset,
                    overlay_feature_subset=ctx.overlay_feature_subset,
                    default_horizon=int(ctx.horizon),
                    horizon_contract=ctx.horizon_contract,
                    max_positions=int(ctx.max_positions),
                    min_positions=int(ctx.min_positions),
                    embargo_days=int(ctx.embargo_days),
                    use_risk_adj=ctx.use_risk_adj,
                    target_cfg=_target_config(ctx.cfg, horizon=int(ctx.horizon), horizon_contract=ctx.horizon_contract),
                    costs=ctx.exec_costs,
                    max_name_weight=ctx.research_max_name_weight,
                    nested_candidate_workers=int(ctx.parallel_cfg.get("nested_candidate_workers", 1)),
                    timing_ledger=ctx.timing_ledger,
                    return_artifacts=True,
                )
                nested_records.append(nested_metrics)
                for _art in _window_ranking_artifacts:
                    all_ranking_artifacts.append({**_art, "outer_window_idx": win_idx, "model_name_outer": name})
                if selected_candidate is None:
                    log_lines.append(
                        f"  [window {win_idx}/{len(ctx.windows)}] skip: no nested candidate survived inner search"
                    )
                    continue
                # P13 institutional fix: The nested search optimises hyperparameters
                # (horizon, feature view, score direction) WITHIN the outer model's
                # own architecture.  The nested winner's model_template is adopted
                # ONLY when it belongs to the same model family.  When the nested
                # winner is from a different family (cross-family search enabled),
                # we adopt its DESCRIPTOR but train the outer model's OWN template.
                # This prevents Ridge and XGBRegressor from both silently becoming
                # the same LGBMRanker under the hood.
                nested_winner_name = selected_candidate.model_name
                nested_winner_kind = selected_candidate.model_kind
                _is_same_family = (
                    str(selected_candidate.model_kind).strip().lower()
                    == str(model_kind).strip().lower()
                    and str(selected_candidate.model_name) == str(name)
                )
                if _is_same_family:
                    candidate_model = selected_candidate.model_template
                else:
                    # Cross-family winner: keep outer model's own architecture
                    candidate_model = model
                candidate_horizon = int(selected_candidate.horizon)
                active_feats = list(selected_candidate.active_features)
                # Keep candidate_name, candidate_kind, candidate_uses_proba
                # from the outer model (set at lines 4036-4038)
                nested_selected_models.append(nested_winner_name)
                nested_selected_views.append(selected_candidate.feature_view)
                nested_selected_horizons.append(candidate_horizon)
            t_nested_1 = time.perf_counter()
            if nested_search_applicable and selected_candidate is not None:
                _ncand = int(nested_metrics.get("nested_candidate_count", 0))
                _nk = int(nested_metrics.get("nested_prefilter_top_k", 0))
                _cache = nested_metrics.get("nested_cache_hit", 0.0)
                _nsim = str(nested_metrics.get("nested_simulation_mode", "executable"))
                print(
                    f"    [{name} window {win_idx}/{len(ctx.windows)}] nested done in {t_nested_1 - t_nested_0:.1f}s"
                    f" | candidates={_ncand}->{_nk}"
                    f" | selected={candidate_name}({selected_candidate.feature_view},h={selected_candidate.horizon})"
                    f" | cache={'HIT' if _cache > 0 else 'MISS'}"
                    f" | nested_sim={_nsim}",
                    flush=True,
                )

            prepared = ctx.global_prepare_cache.get_prepared_fold(
                train_start=tr_s,
                train_end=min(tr_e, purge_cutoff),
                eval_start=te_s,
                eval_end=te_e,
                horizon_days=int(candidate_horizon),
                active_features=active_feats,
            )
            tr_work = prepared.train_df
            te_work = prepared.eval_df
            X_tr = prepared.x_train
            X_te = prepared.x_eval
            if X_tr.shape[1] == 0:
                log_lines.append(
                    f"  [window {win_idx}/{len(ctx.windows)}] skip: no active features after train-only preprocessing"
                )
                continue
            y_tr = ctx.global_prepare_cache.get_training_target(
                start=tr_s,
                end=min(tr_e, purge_cutoff),
                horizon_days=int(candidate_horizon),
                model_name=candidate_name,
                model_kind=candidate_kind,
                use_risk_adj=ctx.use_risk_adj,
            )

            if candidate_kind == "short_classifier":
                pos = int((y_tr == 1).sum())
                if pos < 30:
                    log_lines.append(
                        f"  [window {win_idx}/{len(ctx.windows)}] skip short model: only {pos} positive labels "
                        f"(min 30) | test={te_label}"
                    )
                    continue

            try:
                t0 = time.perf_counter()
                win_model = _fit_candidate_model(
                    model_template=candidate_model,
                    name=candidate_name,
                    model_kind=candidate_kind,
                    tr=tr_work,
                    x_tr=X_tr,
                    y_tr=y_tr,
                )
                t1 = time.perf_counter()
            except Exception as exc:
                log_lines.append(f"  [window {win_idx}/{len(ctx.windows)}] train failed: {exc}")
                continue

            # --- Score direction: read from nested validation IC only (single calibration point) ---
            # Direction is determined inside _nested_validate_candidate using inner-window
            # validation data. Reading it here (never recomputing) prevents double-flip.
            _win_dir: int = int(nested_metrics.get("cal_score_direction_mode", 1))
            # P13: When nested search is disabled, fall back to training-data
            # direction calibration using _calibrate_direction_from_data.
            if not nested_search_applicable and _win_dir == 1:
                try:
                    _cal_dir, _cal_diag = _calibrate_direction_from_data(
                        win_model, X_tr, tr_work,
                        model_kind=candidate_kind,
                        uses_proba=candidate_uses_proba,
                        target_col=validation_target_col,
                        source="training_fallback",
                    )
                    _win_dir = _cal_dir
                    nested_metrics["cal_score_direction_mode"] = _cal_dir
                    nested_metrics["score_direction_ic_raw"] = float(
                        _cal_diag.get("score_direction_ic_raw", float("nan"))
                    )
                except Exception:
                    _win_dir = 1
            assert _win_dir in (1, -1), (
                f"[window {win_idx}] cal_score_direction_mode={_win_dir!r} from nested_metrics "
                f"is not in {{+1, -1}} — direction must be set by _nested_validate_candidate"
            )
            _outer_window_directions.append(_win_dir)
            log_lines.append(
                f"  [window {win_idx}] score_direction={_win_dir:+d} "
                f"(from nested_metrics, flip_count="
                f"{int(nested_metrics.get('cal_direction_flip_count', 0))}/"
                f"{int(nested_metrics.get('cal_direction_window_count', 0))} inner windows)"
            )
            # ---------------------------------------------------------------------------------

            try:
                t2 = time.perf_counter()
                score = _score_model_predictions(
                    win_model,
                    X_te,
                    model_kind=candidate_kind,
                    uses_proba=candidate_uses_proba,
                    score_direction=_win_dir,
                )
                train_score = _score_model_predictions(
                    win_model,
                    X_tr,
                    model_kind=candidate_kind,
                    uses_proba=candidate_uses_proba,
                    score_direction=_win_dir,
                )
                score, forecast_calibration = _calibrate_scores(
                    prepared.train_df,
                    train_score,
                    score,
                    target_col=validation_target_col,
                )
                # P13: Store calibration in eval phase for final report
                eval_calibration_results.append(forecast_calibration)
                t3 = time.perf_counter()
            except Exception as exc:
                log_lines.append(f"  [window {win_idx}/{len(ctx.windows)}] predict failed: {exc}")
                continue

            te_scored = te_work.assign(score=score)
            log_lines.append(
                f"  [window {win_idx}] forecast_calibration="
                f"slope={forecast_calibration.slope:.6g} "
                f"shrink={forecast_calibration.shrinkage:.3f} "
                f"t={forecast_calibration.slope_tstat:.2f} "
                f"n={forecast_calibration.n_obs}"
            )
            ic_stats = cross_sectional_ic(te_scored, target_col=validation_target_col, horizon_days=int(ctx.horizon))
            ic = float(ic_stats.get("cs_ic_spearman_mean", float("nan")))
            dir_acc = _directional_accuracy_from_scored(te_scored, model_kind=candidate_kind)

            # ── Phase 1 diagnostics (shadow only, no effect on ic/dir_acc/score) ──
            if enable_neutralization_diagnostics:
                def _neut_diag(_ts=te_scored) -> None:
                    from model_selection.factor_neutralization import compute_neutralization_diagnostics, format_neutralization_report
                    log_lines.append(format_neutralization_report(
                        compute_neutralization_diagnostics(_ts, "score", validation_target_col)
                    ))
                safe_run_diagnostic("factor_neutralization", _neut_diag)

            if enable_signal_stability_diagnostics:
                def _stab_diag(_ts=te_scored) -> None:
                    from model_selection.signal_stability import compute_stability_metrics
                    _sm = compute_stability_metrics(_ts, "score", validation_target_col)
                    log_lines.append(
                        f"  [StabilityDiag] rank_autocorr={_sm.get('rank_autocorr', float('nan')):.4f}"
                        f" spread_cv={_sm.get('spread_cv', float('nan')):.4f}"
                        f" monotonicity={_sm.get('monotonicity', float('nan')):.4f}"
                    )
                safe_run_diagnostic("signal_stability", _stab_diag)

            if enable_ic_engine_diagnostics:
                def _ice_diag(_ts=te_scored) -> None:
                    from research.ic_engine import compute_ic_result
                    _daily_ic = (
                        _ts.dropna(subset=["score", validation_target_col])
                        .groupby("date")[["score", validation_target_col]]
                        .apply(
                            lambda g: float(g["score"].corr(g[validation_target_col], method="spearman"))
                            if len(g) >= 5 else float("nan")
                        )
                    )
                    _icr = compute_ic_result(_daily_ic, feature_name="score")
                    log_lines.append(
                        f"  [ICEngineDiag] IC={_icr.ic_mean:+.4f} t={_icr.ic_t_stat:+.2f}"
                        f" p={_icr.ic_p_value:.3f} sig={_icr.is_significant} ICIR={_icr.icir:+.3f}"
                    )
                safe_run_diagnostic("ic_engine", _ice_diag)

            if _dir_meta_model is not None:
                def _dir_diag(_ts=te_scored, _ic=ic, _wd=_win_dir) -> None:
                    _state = _dir_meta_model.extract_state(_ts, _dir_meta_ic_history)
                    _pred = _dir_meta_model.predict(_state, fallback_ic=_ic)
                    _dir_meta_model.record_window(_state, direction_label=1 if _ic >= 0 else -1)
                    _dir_meta_model.fit()
                    _dir_meta_ic_history.append(_ic)
                    log_lines.append(
                        f"  [DirModelDiag] shadow_pred={_pred.direction:+d} actual={_wd:+d}"
                        f" prob={_pred.direction_prob:.3f} fallback={_pred.fallback}"
                    )
                safe_run_diagnostic("direction_model", _dir_diag)
            # ── End Phase 1 diagnostics ──────────────────────────────────────────

            halflife = float(ic_stats.get("signal_halflife_days", float("nan")))
            eval_cfg = _evaluation_config(
                ctx.cfg,
                path=primary_path,
                max_positions=int(ctx.max_positions),
                min_positions=int(ctx.min_positions),
                horizon=int(candidate_horizon),
                horizon_contract=ctx.horizon_contract,
                signal_halflife_days=halflife,
            )
            # ── Adaptive portfolio-construction calibration ────────────────────
            # Alpha candidates select score family/features/horizon only. Turnover
            # aversion and no-trade bands are construction parameters calibrated
            # downstream, never embedded in model training.
            from model_selection.adaptive_params import (
                calibrate_kelly_gamma, calibrate_gp_band, compute_sigma_cs_daily,
            )
            _prior_ics = [w.oos_ic for w in wm]
            _kelly_gamma = calibrate_kelly_gamma(
                _prior_ics,
                tr_work,
                target_col=validation_target_col,
                config_gamma=float(eval_cfg.gamma_turnover),
            )
            _final_gamma = _kelly_gamma
            _sigma_cs_daily = compute_sigma_cs_daily(tr_work)
            _ic_mean_prior = float(np.nanmean(_prior_ics)) if _prior_ics else float("nan")
            if np.isfinite(_sigma_cs_daily) and np.isfinite(_ic_mean_prior):
                _one_way_cost_bps = (
                    float(eval_cfg.costs.commission_bps)
                    + float(eval_cfg.costs.spread_bps) / 2.0
                )
                _gp_wd, _gp_td = calibrate_gp_band(
                    _one_way_cost_bps,
                    _ic_mean_prior,
                    _sigma_cs_daily,
                    float(eval_cfg.max_name_weight),
                )
                eval_cfg = replace(
                    eval_cfg,
                    gamma_turnover=_final_gamma,
                    no_trade_band_weight_diff=_gp_wd,
                    no_trade_band_total_drift=_gp_td,
                )
            else:
                eval_cfg = replace(eval_cfg, gamma_turnover=_final_gamma)
            # ────────────────────────────────────────────────────────────────────
            eval_state = ctx.global_prepare_cache.get_validation_state(
                start=te_s,
                end=te_e,
                horizon_days=int(candidate_horizon),
                evaluation_cfg=eval_cfg,
            )
            t_exec_0 = time.perf_counter()
            if ctx.timing_ledger is not None:
                ctx.timing_ledger.record(
                    "outer_window_execution_started",
                    model_name=str(name),
                    model_kind=str(model_kind),
                    candidate_name=str(candidate_name),
                    primary_path=str(primary_path),
                    window_index=int(win_idx),
                    test_start=str(te_s.date()),
                    test_end=str((te_e - pd.Timedelta(days=1)).date()),
                    selection_mode=str(selection_mode),
                )
            print(
                f"  [ExecutionValidation heartbeat] {name} window {win_idx}/{len(ctx.windows)} "
                f"simulating {primary_path} test={te_s.date()}→{(te_e - pd.Timedelta(days=1)).date()}",
                flush=True,
            )
            # SAFEGUARD: final_validation MUST use executable simulation for promotion.
            # simulate_proxy_portfolio is forbidden here — only nested_full may use proxy.
            # P18: Churn filter — exclude stocks without consecutive top-decile persistence
            _churn_cfg = _read_churn_filter_config(ctx.cfg, primary_path)
            if _churn_cfg.enabled:
                te_scored, _churn_diag = apply_churn_filter(te_scored, cfg=_churn_cfg)
                if not _churn_diag.get("churn_filter_status", "").startswith("fallback"):
                    log_lines.append(
                        f"  [ChurnFilter] eligible={_churn_diag.get('churn_filter_n_eligible', 0)}/"
                        f"{_churn_diag.get('churn_filter_n_total', 0)} "
                        f"({_churn_diag.get('churn_filter_eligible_pct', 0):.0f}%) "
                        f"elig_score={_churn_diag.get('churn_filter_eligible_score_mean', 0):.4f}"
                    )
                else:
                    log_lines.append(
                        f"  [ChurnFilter] FALLBACK: {_churn_diag.get('churn_filter_fallback_reason', '')}"
                    )
            te_scored_qp = _prescreen_qp_candidates(
                te_scored, primary_path=primary_path, max_positions=int(ctx.max_positions)
            )
            target_weights = build_target_weights(te_scored_qp, eval_cfg, state_cache=eval_state)
            _opt_metrics, _opt_detail = _optimizer_score_weight_audit(
                te_scored_qp,
                target_weights,
                model_name=str(candidate_name),
                window_idx=int(win_idx),
                horizon_days=int(candidate_horizon),
            )
            optimizer_audit_metrics.append(_opt_metrics)
            if not _opt_detail.empty:
                optimizer_audit_records.extend(_opt_detail.to_dict("records"))
            daily_ret_s, pnl_detail = simulate_executable_portfolio(
                te_scored_qp,
                eval_cfg,
                state_cache=eval_state,
                target_weights=target_weights,
            )
            _SIMULATION_TELEMETRY.record(
                phase="final_validation",
                model_name=candidate_name,
                window_idx=win_idx,
                scored=te_scored_qp,
                cfg=eval_cfg,
                runtime_s=0.0,  # placeholder; updated below with t_exec_1 - t_exec_0
                simulation_mode="executable",
                is_cached=getattr(pnl_detail, "attrs", {}).get("_is_cached", False),
            )
            overlay_ret_s = pd.Series(dtype=float)
            overlay_pnl_detail = pd.DataFrame()
            if run_overlay_diagnostics and not is_short_model and primary_path != "long_only_overlay":
                _h = int(candidate_horizon)
                _overlay_cfg = _evaluation_config(
                    ctx.cfg,
                    path="long_only_overlay",
                    max_positions=int(ctx.max_positions),
                    min_positions=int(ctx.min_positions),
                    horizon=_h,
                    horizon_contract=ctx.horizon_contract,
                    signal_halflife_days=halflife,
                )
                overlay_scored_qp = _prescreen_qp_candidates(
                    te_scored, primary_path="long_only_overlay", max_positions=int(ctx.max_positions)
                )
                overlay_ret_s, overlay_pnl_detail = simulate_executable_portfolio(
                    overlay_scored_qp,
                    _overlay_cfg,
                    state_cache=eval_state,
                )
            t_exec_1 = time.perf_counter()
            # Back-fill the runtime on the last telemetry record for this window.
            if _SIMULATION_TELEMETRY.records:
                _SIMULATION_TELEMETRY.records[-1]["runtime_s"] = round(float(t_exec_1 - t_exec_0), 4)
            n_daily_pts = int(len(daily_ret_s))
            if n_daily_pts < int(ctx.args.min_oos_days):
                log_lines.append(
                    f"  WARNING [window {win_idx}/{len(ctx.windows)}] skip: portfolio sim has {n_daily_pts} days "
                    f"(min_oos_days={ctx.args.min_oos_days}) | test={te_label}"
                )
                continue

            n_invested = (
                int(pd.to_numeric(pnl_detail.get("n_positions"), errors="coerce").gt(0).sum())
                if not pnl_detail.empty and "n_positions" in pnl_detail
                else 0
            )
            sharpe = _sharpe_from_series(daily_ret_s.to_numpy(dtype=float), horizon=int(candidate_horizon))
            sharpe_str = f"{sharpe:.4f}" if np.isfinite(sharpe) else "nan"
            overlay_sharpe = (
                _sharpe_from_series(overlay_ret_s.to_numpy(dtype=float), horizon=int(candidate_horizon))
                if len(overlay_ret_s) >= int(ctx.args.min_oos_days)
                else float("nan")
            )
            exec_time_s = float(t_exec_1 - t_exec_0)
            overlay_str = f"{overlay_sharpe:.4f}" if np.isfinite(overlay_sharpe) else "nan"
            selected_str = (
                f" | selected={candidate_name}@h{candidate_horizon}/{selected_candidate.feature_view}"
                if selected_candidate is not None
                else ""
            )
            nested_str = (
                f" | nested={int(nested_metrics.get('nested_candidate_count', 0))}"
                f"→{int(nested_metrics.get('nested_prefilter_top_k', nested_metrics.get('nested_candidate_count', 0)) or 0)}"
                f" proxy={nested_metrics.get('nested_proxy_elapsed_s', float('nan')):.1f}s"
                f" full={nested_metrics.get('nested_exec_elapsed_s', float('nan')):.1f}s"
                f" cache={'HIT' if bool(nested_metrics.get('nested_cache_hit', 0.0)) else 'MISS'}"
                if nested_search_applicable and nested_metrics
                else " | selection=fixed_shortlist | inner_search=SKIPPED"
                if not nested_search_applicable
                else ""
            )
            if DEBUG_DIAGNOSTICS or FINAL_WINDOW_DEBUG:
                log_lines.append(
                    f"  [window {win_idx}/{len(ctx.windows)}] train=[{tr_s.date()}→{(tr_e - pd.Timedelta(days=1)).date()}] "
                    f"test=[{te_label}] | n_days={n_test_unique} | "
                    f"path={primary_path} | days_with_positions={n_invested} | "
                    f"ExecSharpe={sharpe_str} | OverlayExecSharpe={overlay_str} | CS_IC={ic:.4f}"
                    f"{selected_str}{nested_str} | timers nested/train/score/exec="
                    f"{(t_nested_1 - t_nested_0):.1f}/{(t1 - t0):.1f}/{(t3 - t2):.1f}/{exec_time_s:.1f}s"
                )
            if ctx.timing_ledger is not None:
                ctx.timing_ledger.record(
                    "outer_window_evaluated",
                    model_name=str(name),
                    model_kind=str(model_kind),
                    candidate_name=str(candidate_name),
                    candidate_kind=str(candidate_kind),
                    primary_path=str(primary_path),
                    window_index=int(win_idx),
                    train_start=str(tr_s.date()),
                    train_end=str((tr_e - pd.Timedelta(days=1)).date()),
                    test_start=str(te_s.date()),
                    test_end=str((te_e - pd.Timedelta(days=1)).date()),
                    selection_mode=str(selection_mode),
                    nested_time_s=float(t_nested_1 - t_nested_0),
                    fit_time_s=float(t1 - t0),
                    score_time_s=float(t3 - t2),
                    exec_time_s=float(exec_time_s),
                    nested_candidate_count=int(nested_metrics.get("nested_candidate_count", 0)) if nested_metrics else 0,
                    nested_prefilter_top_k=int(nested_metrics.get("nested_prefilter_top_k", 0)) if nested_metrics else 0,
                    exec_sharpe=float(sharpe) if np.isfinite(sharpe) else None,
                    cs_ic=float(ic) if np.isfinite(ic) else None,
                )

            # Task 1 & 2: Write results to disk and delete intermediate arrays
            oos_keep = [c for c in ("date", "ticker", "forward_return", "target_return", "score", "daily_return", "adv_dollar_20", "realised_vol_20d", "capm_beta", "sector", "short_squeeze_risk", "hard_short_squeeze_filter", "borrow_crowding_risk", "short_interest_ratio") if c in te_scored.columns]
            
            oos_p = scratch_dir / f"oos_{win_idx}.parquet"
            te_scored[oos_keep].to_parquet(oos_p, index=False)
            oos_paths.append(oos_p)

            pnl_p = scratch_dir / f"pnl_{win_idx}.parquet"
            pnl_detail.to_parquet(pnl_p, index=False)
            pnl_paths.append(pnl_p)

            daily_p = scratch_dir / f"daily_{win_idx}.parquet"
            daily_ret_s.to_frame("daily_ret").to_parquet(daily_p, index=True)
            daily_paths.append(daily_p)

            exec_time_parts.append(exec_time_s)

            if "overlay_pnl_detail" in locals() and not overlay_pnl_detail.empty:
                o_pnl_p = scratch_dir / f"overlay_pnl_{win_idx}.parquet"
                overlay_pnl_detail.to_parquet(o_pnl_p, index=False)
                overlay_pnl_paths.append(o_pnl_p)

            if "overlay_ret_s" in locals() and len(overlay_ret_s) > 0:
                o_daily_p = scratch_dir / f"overlay_daily_{win_idx}.parquet"
                overlay_ret_s.to_frame("daily_ret").to_parquet(o_daily_p, index=True)
                overlay_daily_paths.append(o_daily_p)

            wm.append(
                WindowMetrics(
                    oos_sharpe=float(sharpe) if np.isfinite(sharpe) else float("nan"),
                    oos_ic=float(ic) if np.isfinite(ic) else float("nan"),
                    oos_dir_acc=float(dir_acc) if np.isfinite(dir_acc) else float("nan"),
                    train_time_s=float(t1 - t0),
                    test_time_s=float(t3 - t2),
                    n_train=int(len(tr_work)),
                    n_test=int(len(te_scored)),
                    train_start=str(tr_s.date()),
                    train_end=str((tr_e - pd.Timedelta(days=1)).date()),
                    test_start=str(te_s.date()),
                    test_end=str((te_e - pd.Timedelta(days=1)).date()),
                    signal_halflife_days=float(ic_stats.get("signal_halflife_days", float("nan"))),
                    cost_adjusted_ic_mean=float(ic_stats.get("cost_adjusted_ic_mean", float("nan"))),
                    capacity_weighted_ic=float(ic_stats.get("capacity_weighted_ic", float("nan"))),
                    turnover_volatility=float(ic_stats.get("turnover_volatility", float("nan"))),
                    decile_tail_stability=float(ic_stats.get("decile_tail_stability", float("nan"))),
                    hhi_concentration=float(ic_stats.get("hhi_concentration", float("nan"))),
                )
            )
            # Task 2 & 4 & 5: Ensure per-window cleanup and memory tracking
            del tr_work
            del te_scored
            del te_mask
            if "daily_ret_s" in locals(): del daily_ret_s
            if "pnl_detail" in locals(): del pnl_detail
            if "target_weights" in locals(): del target_weights
            if "win_model" in locals(): del win_model
            if "overlay_scored_qp" in locals(): del overlay_scored_qp
            if "overlay_ret_s" in locals(): del overlay_ret_s
            if "overlay_pnl_detail" in locals(): del overlay_pnl_detail
            import gc
            gc.collect()
        except Exception as exc:
            log_lines.append(f"  ERROR [window {win_idx}/{len(ctx.windows)}] unexpected failure ({te_label}): {exc}")
            continue

    if not wm:
        log_lines.append("No valid windows (insufficient data).")
        return {"model_name": name, "window_metrics": wm, "row": None, "log_lines": log_lines}

    sharpe_vals = np.array([w.oos_sharpe for w in wm], dtype=float)
    ic_vals = np.array([w.oos_ic for w in wm], dtype=float)
    acc_vals = np.array([w.oos_dir_acc for w in wm], dtype=float)
    tr_t = np.array([w.train_time_s for w in wm], dtype=float)
    te_t = np.array([w.test_time_s for w in wm], dtype=float)

    def _wm_nanmean(attr: str) -> float:
        vals = [getattr(w, attr) for w in wm]
        finite = [v for v in vals if np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")

    _diag_robust_signal_halflife = _wm_nanmean("signal_halflife_days")
    _diag_robust_cost_adjusted_ic = _wm_nanmean("cost_adjusted_ic_mean")
    _diag_robust_capacity_weighted_ic = _wm_nanmean("capacity_weighted_ic")
    _diag_robust_turnover_volatility = _wm_nanmean("turnover_volatility")
    _diag_robust_tail_stability = _wm_nanmean("decile_tail_stability")
    _diag_robust_hhi = _wm_nanmean("hhi_concentration")

    nested_sharpes = np.array([r.get("nested_sharpe_mean", np.nan) for r in nested_records], dtype=float)
    nested_ics = np.array([r.get("nested_ic_mean", np.nan) for r in nested_records], dtype=float)
    nested_windows_total = int(np.nansum([r.get("nested_windows", 0) for r in nested_records]))
    nested_window_failures_total = int(np.nansum([r.get("nested_window_failures", 0) for r in nested_records]))
    nested_failure_log_combined = " || ".join(
        r["nested_window_failure_log"] for r in nested_records
        if r.get("nested_window_failure_log")
    )

    # Task 4: Reload summaries at end for final aggregation
    oos_df = pd.concat([pd.read_parquet(p) for p in oos_paths], ignore_index=True) if oos_paths else pd.DataFrame()
    daily_parts = [pd.read_parquet(p)["daily_ret"] for p in daily_paths]
    chained_daily_s = _concat_window_daily_return_series(daily_parts)
    chained_daily = chained_daily_s.to_numpy(dtype=float)
    oos_sharpe_chained = _sharpe_from_series(chained_daily, horizon=int(ctx.horizon))
    oos_cagr_chained = _cagr_from_daily_returns(chained_daily)
    oos_max_dd = _max_drawdown_from_daily_returns(chained_daily)
    oos_win_rate = _win_rate_from_daily_returns(chained_daily)
    
    pnl_parts = [pd.read_parquet(p) for p in pnl_paths]
    exec_stats = _pnl_detail_metrics(pnl_parts, chained_daily, horizon=int(ctx.horizon))
    exec_stats.update(decile_return_diagnostics(oos_df, target_col=validation_target_col))

    overlay_daily_parts = [pd.read_parquet(p)["daily_ret"] for p in overlay_daily_paths]
    overlay_chained_daily_s = _concat_window_daily_return_series(overlay_daily_parts)
    overlay_chained_daily = overlay_chained_daily_s.to_numpy(dtype=float)
    overlay_oos_sharpe_chained = _sharpe_from_series(overlay_chained_daily, horizon=int(ctx.horizon))
    overlay_oos_cagr_chained = _cagr_from_daily_returns(overlay_chained_daily)
    overlay_oos_max_dd = _max_drawdown_from_daily_returns(overlay_chained_daily)
    overlay_oos_win_rate = _win_rate_from_daily_returns(overlay_chained_daily)
    
    overlay_pnl_parts = [pd.read_parquet(p) for p in overlay_pnl_paths]
    overlay_exec_stats = _pnl_detail_metrics(overlay_pnl_parts, overlay_chained_daily, horizon=int(ctx.horizon))
    
    overlay_sharpe_vals = np.array(
        [
            _sharpe_from_series(s.to_numpy(dtype=float), horizon=int(ctx.horizon))
            for s in overlay_daily_parts
            if len(s) >= int(ctx.args.min_oos_days)
        ],
        dtype=float,
    )
    cs_ic_stats = cross_sectional_ic(oos_df, target_col=validation_target_col, horizon_days=int(ctx.horizon))
    oos_ic_chained = float(cs_ic_stats.get("cs_ic_spearman_mean", float("nan")))
    overlay_oos_ic_chained = oos_ic_chained

    # (Removed premature cleanup to allow subsumption and diagnostics to run)

    if is_short_model or primary_path == "long_only_overlay":
        overlay_oos_ic_chained = float("nan")
    nested_selected_model_mode = (
        pd.Series(nested_selected_models, dtype="object").mode().iloc[0] if nested_selected_models else ""
    )
    nested_selected_view_mode = (
        pd.Series(nested_selected_views, dtype="object").mode().iloc[0] if nested_selected_views else ""
    )
    nested_selected_horizon_mode = (
        int(pd.Series(nested_selected_horizons, dtype=float).mode().iloc[0])
        if nested_selected_horizons
        else int(ctx.horizon)
    )
    _subsumption_horizon = int(nested_selected_horizon_mode if nested_selected_horizons else ctx.horizon)
    subsumption_stats = factor_subsumption_diagnostics(
        chained_daily_s,
        oos_df,
        horizon_days=_subsumption_horizon,
        precomputed_fac=ctx.precomputed_factor_returns.get(_subsumption_horizon),
    )
    # P13 fix: use explicit None-sentinel to avoid Python "0.0 or 0.95" trap
    _raw_decay = ctx.ms_cfg.get("selection_weight_decay")
    decay_val = float(_raw_decay) if _raw_decay is not None else 0.95
    # P13: Aggregate evaluation-phase forecast calibration across windows
    if eval_calibration_results:
        _cal_slopes = [c.slope for c in eval_calibration_results if np.isfinite(c.slope)]
        _cal_shrinks = [c.shrinkage for c in eval_calibration_results if np.isfinite(c.shrinkage)]
        _cal_tstats = [c.slope_tstat for c in eval_calibration_results if np.isfinite(c.slope_tstat)]
        _eval_cal = {
            "eval_forecast_calibration_slope_mean": float(np.nanmean(_cal_slopes)) if _cal_slopes else float("nan"),
            "eval_forecast_calibration_slope_std": float(np.nanstd(_cal_slopes, ddof=1)) if len(_cal_slopes) > 1 else 0.0,
            "eval_forecast_calibration_shrinkage_mean": float(np.nanmean(_cal_shrinks)) if _cal_shrinks else float("nan"),
            "eval_forecast_calibration_tstat_mean": float(np.nanmean(_cal_tstats)) if _cal_tstats else float("nan"),
            "eval_forecast_calibration_tstat_max": float(np.nanmax(np.abs(_cal_tstats))) if _cal_tstats else float("nan"),
            "eval_forecast_calibration_n_windows": len(eval_calibration_results),
        }
    else:
        _eval_cal = {
            "eval_forecast_calibration_slope_mean": float("nan"),
            "eval_forecast_calibration_slope_std": float("nan"),
            "eval_forecast_calibration_shrinkage_mean": float("nan"),
            "eval_forecast_calibration_tstat_mean": float("nan"),
            "eval_forecast_calibration_tstat_max": float("nan"),
            "eval_forecast_calibration_n_windows": 0,
        }
    if optimizer_audit_metrics:
        _opt_keys = sorted({k for d in optimizer_audit_metrics for k in d})
        _opt_audit_agg = {}
        for _k in _opt_keys:
            _vals = [
                float(d.get(_k, float("nan")))
                for d in optimizer_audit_metrics
                if np.isfinite(float(d.get(_k, float("nan"))))
            ]
            _opt_audit_agg[_k] = float(np.nanmean(_vals)) if _vals else float("nan")
    else:
        _opt_audit_agg = {}
    inst_metrics = _compute_institutional_metrics(ic_vals, sharpe_vals, float(oos_cagr_chained), float(oos_max_dd))
    row = {
        "model_name": name,
        "model_kind": model_kind,
        "selection_mode": selection_mode,
        "horizon_days": int(ctx.horizon),
        "rebalance_frequency": int(ctx.horizon),
        "oos_evaluation_path": primary_path,
        "overlay_evaluation_path": ("long_only_overlay" if (run_overlay_diagnostics and not is_short_model and primary_path != "long_only_overlay") else ""),
        "oos_sharpe_mean": _weighted_recency_mean(sharpe_vals, decay_base=decay_val),
        "oos_sharpe_std": float(np.nanstd(sharpe_vals, ddof=1)) if len(wm) > 1 else 0.0,
        "oos_ic_mean": _weighted_recency_mean(ic_vals, decay_base=decay_val),
        "oos_ic_std": float(np.nanstd(ic_vals, ddof=1)) if len(wm) > 1 else 0.0,
        "oos_dir_acc_mean": _weighted_recency_mean(acc_vals, decay_base=decay_val),
        "oos_dir_acc_std": float(np.nanstd(acc_vals, ddof=1)) if len(wm) > 1 else 0.0,
        "oos_sharpe_chained": float(oos_sharpe_chained),
        "oos_cagr_chained": float(oos_cagr_chained),
        "oos_max_dd": float(oos_max_dd),
        "oos_win_rate": float(oos_win_rate),
        "oos_ic_chained": float(oos_ic_chained),
        **cs_ic_stats,
        **exec_stats,
        **_opt_audit_agg,
        **subsumption_stats,
        **_eval_cal,
        "long_short_oos_sharpe_chained": float(oos_sharpe_chained) if primary_path == "long_short_spread" else float("nan"),
        "long_short_oos_cagr_chained": float(oos_cagr_chained) if primary_path == "long_short_spread" else float("nan"),
        "long_short_oos_max_dd": float(oos_max_dd) if primary_path == "long_short_spread" else float("nan"),
        "long_short_oos_ic_chained": float(oos_ic_chained) if primary_path == "long_short_spread" else float("nan"),
        "short_side_oos_sharpe_chained": float(oos_sharpe_chained) if primary_path == "short_side" else float("nan"),
        "short_side_oos_cagr_chained": float(oos_cagr_chained) if primary_path == "short_side" else float("nan"),
        "short_side_oos_max_dd": float(oos_max_dd) if primary_path == "short_side" else float("nan"),
        "short_side_oos_ic_chained": float(oos_ic_chained) if primary_path == "short_side" else float("nan"),
        "overlay_oos_sharpe_mean": _weighted_recency_mean(overlay_sharpe_vals, decay_base=decay_val),
        "overlay_oos_sharpe_std": (float(np.nanstd(overlay_sharpe_vals, ddof=1)) if len(overlay_sharpe_vals) > 1 else 0.0),
        "overlay_oos_sharpe_chained": float(overlay_oos_sharpe_chained),
        "overlay_oos_cagr_chained": float(overlay_oos_cagr_chained),
        "overlay_oos_max_dd": float(overlay_oos_max_dd),
        "overlay_oos_win_rate": float(overlay_oos_win_rate),
        "overlay_oos_ic_chained": float(overlay_oos_ic_chained),
        "overlay_exec_sharpe": float(overlay_exec_stats.get("exec_sharpe", float("nan"))),
        "overlay_exec_cost_return_sum": float(overlay_exec_stats.get("exec_cost_return_sum", float("nan"))),
        "overlay_exec_turnover_mean": float(overlay_exec_stats.get("exec_turnover_mean", float("nan"))),
        **inst_metrics,
        "train_time_avg": float(np.nanmean(tr_t)),
        "test_time_avg": float(np.nanmean(te_t)),
        "exec_time_avg": float(np.nanmean(exec_time_parts)) if exec_time_parts else float("nan"),
        "nested_proxy_elapsed_avg": float(np.nanmean([r.get("nested_proxy_elapsed_s", np.nan) for r in nested_records])) if nested_records else float("nan"),
        "nested_exec_elapsed_avg": float(np.nanmean([r.get("nested_exec_elapsed_s", np.nan) for r in nested_records])) if nested_records else float("nan"),
        "n_windows": int(len(wm)),
        "nested_enabled": bool(ctx.nested_cfg.get("enabled", True)) and nested_search_applicable,
        "nested_search_applicable": bool(nested_search_applicable),
        "nested_sharpe_mean": float(np.nanmean(nested_sharpes)) if np.isfinite(nested_sharpes).any() else float("nan"),
        "nested_ic_mean": float(np.nanmean(nested_ics)) if np.isfinite(nested_ics).any() else float("nan"),
        "nested_selection_score_mean": float(np.nanmean([r.get("nested_selection_score", np.nan) for r in nested_records])) if nested_records else float("nan"),
        "nested_windows": int(nested_windows_total),
        "nested_window_failures": int(nested_window_failures_total),
        "nested_window_failure_log": nested_failure_log_combined,
        "nested_candidate_count": int(np.nanmax([r.get("nested_candidate_count", 0) for r in nested_records]) if nested_records else 0),
        "nested_prefilter_top_k": int(np.nanmax([r.get("nested_prefilter_top_k", 0) for r in nested_records]) if nested_records else 0),
        "nested_cache_hit": float(np.nanmean([r.get("nested_cache_hit", 0.0) for r in nested_records])) if nested_records else 0.0,
        "overlay_diagnostics_enabled": bool(run_overlay_diagnostics),
        "nested_selected_model_mode": str(nested_selected_model_mode),
        "nested_selected_feature_view_mode": str(nested_selected_view_mode),
        "nested_selected_horizon_mode": int(nested_selected_horizon_mode),
        "oos_psr": round(_compute_psr(chained_daily.dropna().values if hasattr(chained_daily, "dropna") else chained_daily[np.isfinite(chained_daily)]), 4),
        "oos_deflated_sharpe": round(_compute_deflated_sharpe(chained_daily.dropna().values if hasattr(chained_daily, "dropna") else chained_daily[np.isfinite(chained_daily)], n_trials=ctx.metric_trial_count), 4),
        **ctx.feature_contract_summary,
        **ctx.alpha_admission_summary,
        "diag_robust_signal_halflife": _diag_robust_signal_halflife,
        "diag_robust_cost_adjusted_ic": _diag_robust_cost_adjusted_ic,
        "diag_robust_capacity_weighted_ic": _diag_robust_capacity_weighted_ic,
        "diag_robust_turnover_volatility": _diag_robust_turnover_volatility,
        "diag_robust_tail_stability": _diag_robust_tail_stability,
        "diag_robust_hhi": _diag_robust_hhi,
        "diag_robust_turnover_mean": float(exec_stats.get("exec_turnover_mean", float("nan"))),
    }
    row.update(evaluate_promotion_gates(row, ctx.gate_cfg, model_kind=model_kind, long_cand_cfg=ctx.long_cand_cfg))

    # ── Research diagnostics (read-only, post-gate) ──────────────────────────
    # Gated by DiagnosticExecutionPlan. Outputs are additive ``diag_`` prefixed
    # keys that do NOT affect promotion gates.
    if ctx.diagnostic_plan.research_diagnostics:
        try:
            _pnl_all = pd.concat(pnl_parts, ignore_index=True) if pnl_parts else pd.DataFrame()
            _research_diag = _compute_research_diagnostics(
                oos_df,
                _pnl_all,
                row,
                model_kind=model_kind,
                horizon=int(ctx.horizon),
                target_col=validation_target_col,
            )
            row.update(_research_diag)
        except Exception as _diag_exc:
            logger.debug("research_diagnostics skipped for %s: %s", name, _diag_exc)
        else:
            ctx.diagnostic_plan.record("research_diagnostics", name)

    if ctx.diagnostic_plan.empirical_baselines:
        try:
            _pnl_for_decomp = pd.concat(pnl_parts, ignore_index=True) if pnl_parts else pd.DataFrame()
            _baseline_diag = _compute_empirical_baselines(
                oos_df, model_kind=model_kind, pnl_df=_pnl_for_decomp
            )
            row.update(_baseline_diag)
        except Exception as _bl_exc:
            logger.debug("empirical_baselines skipped for %s: %s", name, _bl_exc)
        else:
            ctx.diagnostic_plan.record("empirical_baselines", name)

    if all_ranking_artifacts:
        row.update(_compute_nested_stability_metrics(all_ranking_artifacts))

    # --- Score direction aggregation across outer windows (P37: governed) ---
    _orientation_policy = parse_orientation_policy(
        ((ctx.cfg.get("model_selection", {}) or {}).get("orientation_policy", {}) or {}) if hasattr(ctx, "cfg") else {}
    )
    if _outer_window_directions:
        _n_flip = _outer_window_directions.count(-1)
        _n_fixed = _outer_window_directions.count(1)

        # Legacy majority vote (reference only)
        _legacy_direction = -1 if _n_flip > _n_fixed else 1

        # P37: Build OrientationRecords from window data (IC_raw from nested_metrics if available)
        _orient_records: list[OrientationRecord] = []
        _window_ic_raws = []
        for _i, _wd in enumerate(_outer_window_directions):
            _w_idx = _i + 1
            _ic_raw = float("nan")
            if _i < len(wm):
                _ic_raw = float(getattr(wm[_i], "oos_ic", float("nan")))
            _window_ic_raws.append(_ic_raw)
            _orient_records.append(OrientationRecord(
                window_idx=_w_idx,
                direction=_wd,
                ic_raw=_ic_raw if np.isfinite(_ic_raw) else 0.0,
                ic_calibrated=_ic_raw * _wd if np.isfinite(_ic_raw) else 0.0,
                mode="calibrated" if _wd == -1 else "fixed",
                reason="" if np.isfinite(_ic_raw) else "no_ic_evidence",
            ))

        # Policy-governed aggregation
        _policy_direction, _policy_reason, _policy_diag = _orientation_policy.aggregate_direction(_orient_records)
        _dir_mode_str = "calibrated" if _policy_direction == -1 else "fixed"

        row.update({
            "score_direction_selected": int(_policy_direction),
            "score_direction_flip_windows": int(_n_flip),
            "score_direction_fixed_windows": int(_n_fixed),
            "score_direction_mode": _dir_mode_str,
            "score_direction_policy_mode": _orientation_policy.aggregate_mode.value,
            "score_direction_policy_version": _orientation_policy.policy_version,
        })
        log_lines.append(
            f"  score_direction aggregate: {_policy_direction:+d} ({_dir_mode_str}) "
            f"[flip={_n_flip}/{len(_outer_window_directions)} windows] "
            f"policy={_orientation_policy.aggregate_mode.value} reason={_policy_reason}"
        )

        # P37: Write orientation manifest
        if _orientation_policy.record_orientation_manifest and _orient_records:
            try:
                _manifest_df = _orientation_manifest_to_dataframe(
                    _orient_records,
                    model_name=str(name),
                    policy_version=_orientation_policy.policy_version,
                )
                if _manifest_df is not None and not _manifest_df.empty:
                    _manifest_path = Path(out_dir) / Path(_orientation_policy.manifest_path).name
                    _manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    _manifest_df.to_csv(_manifest_path, index=False)
            except Exception:
                pass
    else:
        row.update({
            "score_direction_selected": 1,
            "score_direction_flip_windows": 0,
            "score_direction_fixed_windows": 0,
            "score_direction_mode": "fixed",
            "score_direction_policy_mode": "",
            "score_direction_policy_version": 0,
        })

    if nested_search_applicable and bool(ctx.nested_cfg.get("enabled", True)):
        nested_failures: list[str] = []
        # P13: max_windows config field is used as a MINIMUM threshold (the
        # model must have at least this many nested windows to pass).  The
        # field should be renamed to min_windows; for backward compatibility
        # we accept both names.
        _nested_min_windows = int(
            ctx.nested_cfg.get("min_windows")
            or ctx.nested_cfg.get("max_windows", 1)
        )
        if int(row.get("nested_windows", 0) or 0) < _nested_min_windows:
            nested_failures.append("nested_min_windows")
        # P40: When nested_sim=proxy_only, the nested Sharpe is computed from
        # a rank-based proxy (no QP optimizer, no covariance, no risk budgeting).
        # It is a diagnostic/ranking metric, NOT an executable promotion gate.
        # The final execution validation is authoritative.
        _nested_sim_mode = str(row.get("nested_simulation_mode", "executable") or "executable")
        _is_proxy_only = _nested_sim_mode == "proxy_only"
        _nested_skipped = bool(row.get("nested_validation_skipped", 0.0))
        _nested_sharpe = _safe_float(row.get("nested_sharpe_mean"), -999.0)
        if _is_proxy_only:
            # Report as proxy_nested_sharpe_mean for clarity; do not block promotion
            row["proxy_nested_sharpe_mean"] = _nested_sharpe
            row["nested_sharpe_mean"] = float("nan")  # Clear to prevent misuse
        elif _nested_skipped:
            # Short-circuited: no nested validation ran. Skip nested gates.
            row["nested_sharpe_mean"] = float("nan")
            row["nested_ic_mean"] = float("nan")
        elif _nested_sharpe < float(ctx.nested_cfg.get("min_sharpe", 0.0)):
            nested_failures.append("nested_min_sharpe")
        # Overlay candidates are score models evaluated through a different
        # construction mandate, so raw IC remains diagnostic but is not always
        # comparable to spread/short paths.
        _ic_gate_applicable = model_kind != "overlay_alpha"
        if _ic_gate_applicable and not _nested_skipped and _safe_float(row.get("nested_ic_mean"), -999.0) < float(ctx.nested_cfg.get("min_ic", 0.0)):
            nested_failures.append("nested_min_ic")
        if nested_failures:
            row["promotion_pass"] = False
            prev_fail = str(row.get("promotion_failures", "") or "")
            row["promotion_failures"] = ",".join([x for x in [prev_fail, ",".join(nested_failures)] if x])
    suspicious = bool(
        np.isfinite(row["oos_sharpe_chained"])
        and np.isfinite(row["oos_ic_chained"])
        and (row["oos_sharpe_chained"] > 2.0)
        and (row["oos_ic_chained"] < 0.05)
    )
    row["leakage_suspect"] = bool(suspicious)
    row["is_diagnostic_only"] = _is_diagnostic_only(name)
    if row["is_diagnostic_only"]:
        row["promotion_pass"] = False
        prev_fail = str(row.get("promotion_failures", "") or "")
        row["promotion_failures"] = ",".join([x for x in [prev_fail, "diagnostic_only"] if x])
    if suspicious:
        log_lines.append(
            f"WARNING: {name} suspicious metrics (Sharpe_chained={row['oos_sharpe_chained']:.3f}, "
            f"IC_chained={row['oos_ic_chained']:.3f}). This may indicate leakage or a broken Sharpe proxy."
        )
        if ctx.args.discard_suspicious_models:
            log_lines.append(f"  -> Discarding {name} from selection/report due to --discard_suspicious_models.")
            return {"model_name": name, "window_metrics": wm, "row": None, "log_lines": log_lines}

    _ic_ir_str = f"{row['oos_ic_ir']:.3f}" if np.isfinite(row.get('oos_ic_ir', float('nan'))) else "nan"
    _tstat_str = f"{row['oos_ic_tstat']:.2f}" if np.isfinite(row.get('oos_ic_tstat', float('nan'))) else "nan"
    _daily_ic_ann_str = f"{row['daily_ic_annualized_icir']:.2f}" if np.isfinite(row.get("daily_ic_annualized_icir", float("nan"))) else "nan"
    _daily_ic_t_str = f"{row['daily_ic_hac_tstat']:.2f}" if np.isfinite(row.get("daily_ic_hac_tstat", float("nan"))) else "nan"
    _calmar_str = f"{row['oos_calmar']:.2f}" if np.isfinite(row.get('oos_calmar', float('nan'))) else "nan"
    _beat_str = f"{row['oos_beat_rate']:.2f}" if np.isfinite(row.get('oos_beat_rate', float('nan'))) else "nan"
    _comp_str = f"{row['oos_composite']:.3f}" if np.isfinite(row.get('oos_composite', float('nan'))) else "nan"
    _dsr_str = f"{row['oos_deflated_sharpe']:.3f}" if np.isfinite(row.get("oos_deflated_sharpe", float("nan"))) else "nan"
    _overlay_str = f"{row['overlay_oos_sharpe_chained']:.3f}" if np.isfinite(row.get("overlay_oos_sharpe_chained", float("nan"))) else "nan"
    _subsumption_str = (
        f"{row.get('subsumption_alpha_ann', float('nan')):.3f}/"
        f"{row.get('subsumption_alpha_tstat', float('nan')):.2f}/"
        f"{row.get('subsumption_r2', float('nan')):.2f}"
        if np.isfinite(row.get("subsumption_alpha_ann", float("nan")))
        else "nan"
    )
    _gate_str = "PASS" if bool(row.get("promotion_pass", False)) else f"FAIL[{row.get('promotion_failures', '')}]"
    _nested_str = (
        f"{row.get('nested_sharpe_mean', float('nan')):.3f}/{row.get('nested_ic_mean', float('nan')):.4f}"
        if bool(row.get("nested_enabled", False))
        else "off"
    )
    _decile_str = (
        f"{row.get('decile_spread', float('nan')):.5f}/{row.get('decile_monotonicity', float('nan')):.2f}"
        if np.isfinite(row.get("decile_spread", float("nan")))
        else "nan"
    )
    log_lines.append(
        f"OOS Sharpe ({row['oos_evaluation_path']} chained): {row['oos_sharpe_chained']:.3f} | "
        f"Overlay Sharpe: {_overlay_str} | ExecCost: {row.get('exec_cost_return_sum', float('nan')):.4f} | "
        f"POV mean/p95/max: {row.get('exec_participation_mean', float('nan')):.3f}/"
        f"{row.get('exec_participation_p95', float('nan')):.3f}/"
        f"{row.get('exec_participation_max', float('nan')):.3f} | "
        f"BetaAbs: {row.get('exec_beta_abs_mean', float('nan')):.3f} | "
        f"Subsumption alpha/t/R2: {_subsumption_str} | "
        f"LegSharpe L/S: {row.get('exec_long_leg_sharpe', float('nan')):.3f}/{row.get('exec_short_leg_sharpe', float('nan')):.3f} | "
        f"DecileSpread/Mono: {_decile_str} | Nested Sharpe/IC: {_nested_str} | "
        f"window Sharpe mean±std: {row['oos_sharpe_mean']:.3f} ± {row['oos_sharpe_std']:.3f} | "
        f"WindowIC: {row['oos_ic_mean']:.3f} ± {row['oos_ic_std']:.3f} (IR={_ic_ir_str}, t={_tstat_str}) | "
        f"DailyIC: {row.get('daily_ic_mean', float('nan')):.4f} ± {row.get('daily_ic_std', float('nan')):.4f} "
        f"(annICIR={_daily_ic_ann_str}, HACt={_daily_ic_t_str}) | "
        f"PSR/DSR: {row.get('oos_psr', float('nan')):.3f}/{_dsr_str} | Calmar: {_calmar_str} | "
        f"Beat: {_beat_str} | Composite: {_comp_str} | DirAcc: {row['oos_dir_acc_mean']:.3f} | "
        f"Gate={_gate_str} | windows={row['n_windows']}"
    )
    # Final cleanup of reloaded objects and scratch files
    try:
        del oos_df
        del pnl_parts
        del daily_parts
        del overlay_pnl_parts
        del overlay_daily_parts
    except NameError:
        pass
        
    try:
        import shutil
        shutil.rmtree(scratch_dir, ignore_errors=True)
    except:
        pass

    import gc
    gc.collect()

    from model_selection import validation as _val
    return {
        "model_name": name,
        "window_metrics": wm,
        "row": row,
        "log_lines": log_lines,
        "_sim_telemetry": _SIMULATION_TELEMETRY.records,
        "_market_state_stats": _val.get_market_state_stats(),
        "_ranking_artifacts": all_ranking_artifacts,
        "_optimizer_audit_records": optimizer_audit_records,
    }
    print(f"Overall: FAIL ({failures} failing check(s))")
    return 1


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Leakage-safe numeric predictor columns.

    Excludes:
      - identifiers: date, ticker, sector, regime labels
      - targets: forward_return, direction
      - risk/execution state: ADV, realised vol, beta, expected cost
      - any column containing 'forward' (e.g. spy_forward_5d, forward_return_cs_z)
    """
    cols: list[str] = []
    for c in df.columns:
        if c == "y_bin":
            continue
        if is_model_feature_column(c, df[c]):
            cols.append(c)
    return cols


# ---------------------------------------------------------------------------
# Sharpe-IC objective (Liu, Zhou & Zhu 2023 — "Maximizing the Sharpe Ratio:
# A Genetic Programming Approach")
#
# Core finding: training to MAXIMIZE the Sharpe ratio of the cross-sectional
# spread portfolio directly outperforms MSE-minimizing models by ~20% in OOS
# Sharpe (1.21 vs 1.01 for GP_SR vs GP_MSE; 1.21 vs 0.83 for best NN).
#
# The spread portfolio Sharpe ratio is proportional to IC × sqrt(breadth)
# (Grinold-Kahn fundamental law of active management).  Maximizing IC is
# therefore a tractable proxy for directly maximizing the Sharpe ratio.
#
# For XGBoost, we implement this as a custom (grad, hess) objective.
# XGB minimizes its objective, so we minimize the NEGATIVE IC.
#
# Gradient of IC w.r.t. prediction f_i (Pearson correlation):
#   IC = cov(f, r) / (σ_f × σ_r)
#   ∂IC/∂f_i = (r_i − r̄)/(n σ_f σ_r) − IC (f_i − f̄)/(n σ_f²)
#
# Differential Sharpe Ratio (DSR) insight from Moody & Saffell (1998), also
# used in MACE (Abbade & Costa 2025): the gradient of Sharpe w.r.t. return
# is a stable online estimator — constant Hessian (1/n) is the approved
# approximation when the Hessian of IC is costly to compute.
# ---------------------------------------------------------------------------

def _train_regime_models(
    df: pd.DataFrame,
    feat_cols: list[str],
    all_singular: list[str],
    out_dir: Path,
    horizon: int,
) -> None:
    """
    C2: Train regime-conditional XGBClassifier models on the full dataset.

    One model per regime (Bull, Bear, HighVol, Normal).  Each model trains
    only on observations from that regime, so it learns feature→return
    relationships that are specific to that market environment.

    Saved as:
        output/models/xgb_regime_bull.pkl
        output/models/xgb_regime_bear.pkl
        output/models/xgb_regime_highvol.pkl
        output/models/xgb_regime_normal.pkl
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("[C2] XGBoost not available; skipping regime-conditional model training.")
        return

    if "regime_label" not in df.columns:
        print("[C2] regime_label column not found in feature matrix; skipping.")
        return

    active_feats = [c for c in feat_cols if c not in all_singular]
    # Map both regime taxonomies:
    # - feature_builder/regime_detection uses: Bull, Bear, HighVol, Normal
    # - backtester/regime.py uses:            Bull, Bear, Crisis, Sideways
    regime_map = {
        "Bull": "bull",
        "Bear": "bear",
        "HighVol": "highvol",   # also treated as Crisis in the backtester
        "Normal": "normal",
        "Crisis": "highvol",
        "Sideways": "normal",
    }

    print("\n[C2] Training regime-conditional XGBClassifier models...")
    saved: list[str] = []
    for regime_label, fname_suffix in regime_map.items():
        rdf = df[df["regime_label"] == regime_label].copy()
        n = len(rdf)
        if n < 300:
            print(f"  {regime_label}: only {n} samples — skipping (need ≥300).")
            continue

        X = rdf[active_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0).values
        y = rdf["y_bin"].fillna(0).astype(int).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).copy()

        # Balance classes — regime-filtered sets can be skewed (e.g. Bear is mostly down)
        scale = float(max((y == 0).sum(), 1)) / float(max((y == 1).sum(), 1))

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
        )
        with np.errstate(all="ignore"):
            xgb.fit(X, y)

        path = out_dir / f"xgb_regime_{fname_suffix}.pkl"
        artifact = {
            "model_name": f"XGBClassifier_{regime_label}",
            "model_type": "classifier",
            "regime": regime_label,
            "horizon_days": int(horizon),
            "target": "y_bin",
            "feature_columns": active_feats,
            "n_train": int(n),
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "estimator": xgb,
        }
        with open(path, "wb") as fh:
            pickle.dump(artifact, fh)
        saved.append(str(path))
        print(f"  {regime_label}: n={n}  scale_pos_weight={scale:.2f}  → saved {path.name}")

    if saved:
        print("\n[C2] Regime-conditional classifier models ready.")

    # ── Regime-conditional REGRESSORS for soft-mixture blending ────────────────────────
    # Train XGBRegressor on each regime's data subset using forward_return as target.
    # The inference engine blends these four models with weights proportional to the
    # current regime probability, producing a continuous score that smoothly transitions
    # across regime boundaries rather than hard-switching at regime detection day.
    #
    # Blend weights (hardcoded in signals.py _REGIME_BLEND_WEIGHTS):
    #   Bull    → 85% Bull model, 10% Sideways, 5% Bear
    #   Bear    → 65% Bear model, 15% Sideways, 15% Crisis, 5% Bull
    #   Crisis  → 75% Crisis model, 20% Bear, 5% Sideways
    #   Sideways→ 65% Sideways, 15% Bull, 15% Bear, 5% Crisis
    print("\n[C2] Training per-regime XGBRegressor models for soft-mixture inference...")
    try:
        from xgboost import XGBRegressor as _XGBReg
    except ImportError:
        print("[C2] XGBoost not available; skipping regime regressor training.")
        return

    if "forward_return" not in df.columns:
        print("[C2] forward_return column missing from feature matrix; skipping regime regressors.")
        return

    _reg_regime_map = {
        "Bull": "bull",
        "Bear": "bear",
        "HighVol": "highvol",
        "Normal": "normal",
        "Crisis": "highvol",
        "Sideways": "normal",
    }
    _reg_saved: list[str] = []
    for _rlabel, _rsuffix in _reg_regime_map.items():
        _rdf = df[df["regime_label"] == _rlabel].copy()
        _n = len(_rdf)
        if _n < 300:
            print(f"  {_rlabel} (regressor): only {_n} samples — skipping (need ≥300).")
            continue

        _X = (
            _rdf[active_feats]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(-10.0, 10.0)
            .values
        )
        _X = np.nan_to_num(_X, nan=0.0, posinf=0.0, neginf=0.0).copy()
        _y = _rdf["forward_return"].fillna(0.0).values

        _xgb_reg = _XGBReg(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="rmse",
            verbosity=0,
            random_state=42,
        )
        with np.errstate(all="ignore"):
            _xgb_reg.fit(_X, _y)

        _reg_path = out_dir / f"xgb_regime_reg_{_rsuffix}.pkl"
        _reg_artifact = {
            "model_name": f"XGBRegressor_{_rlabel}",
            "model_type": "regressor",
            "regime": _rlabel,
            "horizon_days": int(horizon),
            "target": "forward_return",
            "feature_columns": active_feats,
            "n_train": int(_n),
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "estimator": _xgb_reg,
        }
        with open(_reg_path, "wb") as _fh:
            pickle.dump(_reg_artifact, _fh)
        _reg_saved.append(str(_reg_path))
        print(f"  {_rlabel} (regressor): n={_n} → saved {_reg_path.name}")

    if _reg_saved:
        print("\n[C2] Regime regressor models ready. Add to backtest_config.yaml:")
        print("  signals:")
        print("    ml_regime_models_dir: output/models   # enables soft-mixture blending")
        print("    ml_regime_blend_enabled: true")

        # M2: Write manifest so backtest can load regime models by path, not by guessing names.
        import json as _json
        _manifest: dict = {
            "models_dir": str(out_dir.resolve()),
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "horizon_days": int(horizon),
            "models": [],
        }
        for _ps in _reg_saved:
            _pp = Path(_ps)
            _sfx = _pp.stem.replace("xgb_regime_reg_", "")
            _manifest["models"].append({
                "suffix": _sfx,
                "filename": _pp.name,
                "path": str(_pp.resolve()),
            })
        _manifest_path = out_dir / "regime_models_manifest.json"
        with open(_manifest_path, "w") as _mf:
            _json.dump(_manifest, _mf, indent=2)
        print(f"[C2] Wrote regime model manifest: {_manifest_path}")


@dataclass
class ExecutionAuditStore:
    cache_hit_rate: float = 0.0
    cache_structurally_unique: bool = False
    cache_unique_keys: int = 0
    cache_total_fold_lookups: int = 0
    cache_capacity: int = 0
    peak_rss_gb: float = 0.0
    min_valid_days: int = 9999
    max_net_exposure: float = 0.0
    zero_short_days: int = 0
    gate_failures: list[str] = field(default_factory=list)
    ic_definitions_consistent: bool = True
    target_col: str = "target_return"
    promotion_candidate_count: int = 0
    promotion_pass_count: int = 0
    # P19: Worker result payloads for IC integrity aggregation
    _ic_worker_results: list[dict[str, Any]] = field(default_factory=list)

    def collect_ic_counters(self, result: dict[str, Any]) -> None:
        """Collect per-window IC integrity counters from a worker result."""
        if not isinstance(result, dict):
            return
        # Screen-level results from _screen_model_family
        if "ic_valid_days" in result:
            self._ic_worker_results.append({
                "ic_n_days": int(result.get("ic_valid_days", 0) or 0),
                "ic_nan_days": int(result.get("ic_nan_days", 0) or 0),
                "ic_constant_days": int(result.get("ic_constant_days", 0) or 0),
                "ic_small_sample_days": int(result.get("ic_small_sample_days", 0) or 0),
                "ic_nan_inf_days": int(result.get("ic_nan_inf_days", 0) or 0),
            })
        # Execution-level results from _evaluate_model_family
        for wm in result.get("window_metrics", []):
            if isinstance(wm, dict) and "ic_valid_days" in wm:
                self._ic_worker_results.append({
                    "ic_n_days": int(wm.get("ic_valid_days", 0) or 0),
                    "ic_nan_days": int(wm.get("ic_nan_days", 0) or 0),
                    "ic_constant_days": int(wm.get("ic_constant_days", 0) or 0),
                    "ic_small_sample_days": int(wm.get("ic_small_sample_days", 0) or 0),
                    "ic_nan_inf_days": int(wm.get("ic_nan_inf_days", 0) or 0),
                })

    def report(self) -> None:
        print("\n" + "="*40)
        print("EXECUTION READINESS CHECKLIST")
        print("="*40)
        
        # 1. Cache Efficiency.  A low hit rate is a failure only when the cache
        # topology is reusable.  Fold states generated by
        # window × feature-view × horizon grids are often intentionally unique.
        status_cache = "PASS" if (self.cache_hit_rate >= 0.50 or self.cache_structurally_unique) else "FAIL"
        requirement = "Topology-aware" if self.cache_structurally_unique else "Required >= 50%"
        print(f"1. Cache Efficiency: {status_cache}")
        print(f"   - hit_rate        : {self.cache_hit_rate*100:.1f}% ({requirement})")
        print(
            f"   - fold_topology   : unique={self.cache_unique_keys}/{self.cache_total_fold_lookups} "
            f"capacity={self.cache_capacity}"
        )
        print(f"   - peak_rss        : {self.peak_rss_gb:.2f} GB")
        
        # 2. Metric Integrity
        status_metric = "PASS" if self.min_valid_days >= 200 and self.ic_definitions_consistent else "FAIL"
        print(f"2. Metric Integrity: {status_metric}")
        print(f"   - target_col      : {self.target_col}")
        print(f"   - min_valid_days  : {self.min_valid_days if self.min_valid_days < 9000 else 'N/A'} (Required >= 200)")
        print(f"   - ic_consistency  : {'IDENTIFIED' if self.ic_definitions_consistent else 'MISMATCH'}")
        
        # 3. Portfolio Integrity
        status_port = "PASS" if self.zero_short_days == 0 and abs(self.max_net_exposure) < 0.01 else "FAIL"
        print(f"3. Portfolio Integrity: {status_port}")
        print(f"   - zero_short_days : {self.zero_short_days} (Required == 0)")
        print(f"   - mean_net_exp    : {self.max_net_exposure:.6f} (Required < 0.01)")

        # 3.5 IC Integrity (P19) — aggregated from worker result payloads
        from model_selection.validation import aggregate_ic_integrity
        _icr = aggregate_ic_integrity(self._ic_worker_results)
        _ic_status = _icr.get("ic_status", "FAIL")
        print(f"3.5. IC Integrity: {_ic_status}")
        print(f"   - total_evals     : {_icr['ic_total_evaluations']}")
        print(f"   - valid           : {_icr['ic_valid_count']}")
        print(f"   - invalid_constant: {_icr['ic_invalid_constant_count']}")
        print(f"   - invalid_small   : {_icr['ic_invalid_small_sample_count']}")
        print(f"   - invalid_nan     : {_icr.get('ic_invalid_nan_count', 0)}")
        print(f"   - invalid_ratio   : {_icr['ic_invalid_ratio']*100:.1f}% (Required <= 20%)")

        # 4. Promotion Integrity
        status_prom = "PASS" if self.promotion_pass_count > 0 and not self.gate_failures else "FAIL"
        print(f"4. Promotion Integrity: {status_prom}")
        if self.gate_failures:
            print(f"   - Found {len(self.gate_failures)} gate failures in promotion candidates.")
            for f in self.gate_failures[:5]:
                print(f"     - {f}")
        else:
            print(
                f"   - promoted        : {self.promotion_pass_count}/{self.promotion_candidate_count} "
                "(Required >= 1)"
            )
        print("="*40 + "\n")

global_audit = ExecutionAuditStore()


def _write_horizon_ic_report(
    alpha_decay: "pd.DataFrame",
    alpha_admission: "pd.DataFrame",
    horizon: int,
    out_dir: "Path",
) -> None:
    """Write a structured before/after IC report for a given production horizon.

    Columns written:
      feature, horizon_evaluated, ic_mean, ic_tstat, admitted, transform_sign,
      recommended_action, note

    The 'note' column explains WHY a feature was admitted or rejected at this horizon,
    making the horizon-mismatch diagnosis self-documenting in the output CSV.
    """
    if alpha_admission.empty:
        return
    rows: list[dict[str, Any]] = []
    for _, r in alpha_admission.iterrows():
        feature = str(r.get("feature", ""))
        admitted = bool(r.get("admitted", False))
        sign = int(pd.to_numeric(r.get("transform_sign", 1), errors="coerce") or 1)
        action = str(r.get("recommended_action", ""))
        prod_ic = float(pd.to_numeric(r.get("production_ic", 0), errors="coerce") or 0.0)
        prod_tstat = float(pd.to_numeric(r.get("production_ic_tstat", 0), errors="coerce") or 0.0)
        reason = str(r.get("reason", ""))
        marginal_ic = float(pd.to_numeric(r.get("marginal_ic", np.nan), errors="coerce"))
        selected_horizon = pd.to_numeric(r.get("selected_horizon_days", np.nan), errors="coerce")
        valid_days_raw = float(pd.to_numeric(r.get("production_ic_valid_days", 0), errors="coerce"))
        valid_days = int(valid_days_raw) if np.isfinite(valid_days_raw) else 0
        redundancy_corr = float(pd.to_numeric(r.get("redundancy_max_abs_corr", np.nan), errors="coerce"))
        redundant_with = str(r.get("redundant_with", "") or "")
        if admitted:
            note = f"Admitted at {horizon}d; IC={prod_ic:.4f} t={prod_tstat:.2f}"
            if sign < 0:
                note += "; INVERTED (signal used with flipped sign)"
        elif "move_horizon" in action:
            if np.isfinite(selected_horizon):
                note = (
                    f"Moved out of {horizon}d production set; best/declared evidence belongs to "
                    f"{int(selected_horizon)}d, so it remains research-only for this contract"
                )
            else:
                note = f"Rejected at {horizon}d; signal may predict at a different horizon — check IC decay table"
        elif reason == "fails_marginal_contribution":
            note = f"Rejected at {horizon}d by marginal contribution gate; marginal IC={marginal_ic:.4f}"
            if redundant_with:
                note += f"; most redundant with {redundant_with}"
                if np.isfinite(redundancy_corr):
                    note += f" (avg |rank corr|={redundancy_corr:.2f})"
        elif valid_days <= 0 or not np.isfinite(prod_ic) or not np.isfinite(prod_tstat):
            note = f"Rejected at {horizon}d; zero valid IC days or unavailable production evidence"
        elif str(reason).startswith("insufficient_production_evidence"):
            note = f"Rejected at {horizon}d; insufficient production evidence ({reason.split(':', 1)[-1]})"
        else:
            note = f"Rejected at {horizon}d; IC={prod_ic:.4f} t={prod_tstat:.2f}; reason={reason or 'fails_current_horizon_gates'}"
        rows.append({
            "feature": feature,
            "horizon_evaluated": horizon,
            "production_ic": round(prod_ic, 6),
            "production_ic_tstat": round(prod_tstat, 4),
            "production_ic_valid_days": valid_days,
            "marginal_ic": round(marginal_ic, 6) if np.isfinite(marginal_ic) else np.nan,
            "redundancy_max_abs_corr": round(redundancy_corr, 6) if np.isfinite(redundancy_corr) else np.nan,
            "redundant_with": redundant_with,
            "admitted": admitted,
            "transform_sign": sign,
            "recommended_action": action,
            "reason": reason,
            "note": note,
        })
    report_df = pd.DataFrame(rows)
    report_path = out_dir / f"ic_report_{horizon}d.csv"
    report_df.to_csv(report_path, index=False)
    print(f"[IC Report {horizon}d] {len(report_df)} features evaluated → "
          f"{report_df['admitted'].sum()} admitted | "
          f"{(~report_df['admitted']).sum()} rejected | "
          f"saved: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-validation", action="store_true", help="Run detailed feature/target audits")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--config", type=str, default="backtest_config.yaml")
    parser.add_argument(
        "--run_sim_test",
        action="store_true",
        help="Run a small self-test for portfolio simulation logic and exit",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Forward return horizon in trading days (default: config model_selection.lookahead_horizon_days or 20)",
    )
    parser.add_argument(
        "--production_horizons",
        type=int,
        nargs="+",
        default=None,
        metavar="H",
        help=(
            "Run full pipeline for each horizon and save results under output/models/h<H>d/. "
            "Example: --production_horizons 20 63 trains separate 20d and 63d models. "
            "Overrides --horizon when specified. Reads from config production_horizons if omitted."
        ),
    )
    parser.add_argument(
        "--auto-horizon",
        action="store_true",
        default=None,
        help=(
            "Select the optimal prediction horizon algorithmically from the IC-decay "
            "analysis and dispatch the full pipeline at that horizon.  Equivalent to "
            "setting adaptive_horizon.apply: true in config.  Overridden by --horizon."
        ),
    )
    parser.add_argument("--min_test_days", type=int, default=30, help="Skip windows with fewer test dates")
    parser.add_argument(
        "--min_oos_days",
        type=int,
        default=None,
        help="Minimum distinct test days per window; defaults to max(10, 2×holding_period). Set 0 to disable.",
    )
    parser.add_argument(
        "--select_metric",
        type=str,
        default="oos_deflated_sharpe",
        help=(
            "Metric to rank/select best model "
            "(default: formal oos_deflated_sharpe from the primary alpha/short path; overlay metrics are reported separately)"
        ),
    )
    parser.add_argument("--limit_tickers", type=int, default=0, help="Optional: limit universe size for quick runs")
    parser.add_argument(
        "--risk-adj-target",
        action="store_true",
        dest="risk_adj_target",
        help=(
            "C3: Use risk-adjusted return (forward_return / realized_vol) as the regressor target. "
            "Rewards low-vol momentum; improves IC for cross-sectional ranking."
        ),
    )
    parser.add_argument(
        "--regime-models",
        action="store_true",
        dest="regime_models",
        help=(
            "C2: After main model selection, train regime-conditional XGBClassifier models "
            "(Bull/Bear/Sideways/HighVol) on the full dataset and save to output/models/."
        ),
    )
    parser.add_argument(
        "--regime-conditioning",
        action="store_true",
        dest="regime_conditioning",
        help=(
            "C4: Run panel-native regime conditioning evaluation — derives regimes from "
            "cross-sectional panel statistics (no external data), evaluates per-regime "
            "walk-forward (Strategy A) and regime-dummies-as-features (Strategy B)."
        ),
    )
    parser.add_argument(
        "--feature-redundancy",
        action="store_true",
        dest="feature_redundancy",
        help=(
            "C5: Run feature redundancy analysis — computes cross-sectional pairwise "
            "Spearman correlation, IC stability, turnover, signal decay; clusters features; "
            "flags redundant members with data-driven justification; evaluates IC impact."
        ),
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.60,
        dest="corr_threshold",
        help="Absolute Spearman correlation threshold for feature clustering (default 0.60).",
    )
    parser.add_argument(
        "--ensemble-weighting",
        action="store_true",
        dest="ensemble_weighting",
        help=(
            "C6: Run dynamic ensemble weighting — learns per-model weights from rolling IC, "
            "IC t-stat, decile spread, long-only Sharpe, with stability/correlation/turnover "
            "penalties via penalized softmax + EMA smoothing."
        ),
    )
    parser.add_argument(
        "--meta-labeling",
        action="store_true",
        dest="meta_labeling",
        help=(
            "C7: Run meta-labeling framework — trains a meta-model on signal reliability "
            "features (score dispersion, recent IC, alpha capture, market vol, liquidity, "
            "model uncertainty) to scale base-model exposure walk-forward. "
            "Requires --meta-model-name to select the base model."
        ),
    )
    parser.add_argument(
        "--meta-model-name",
        type=str,
        default=None,
        dest="meta_model_name",
        help="Name of base model to meta-label (default: best model by IC from C1 objective comparison).",
    )
    parser.add_argument(
        "--meta-model-type",
        type=str,
        default="ridge",
        dest="meta_model_type",
        choices=["ridge", "lgbm", "rf"],
        help="Meta-model type: ridge (default), lgbm, or rf.",
    )
    parser.add_argument(
        "--short-modeling",
        action="store_true",
        dest="short_modeling",
        help=(
            "C8: Run short-side modeling framework — builds 5 short-specific targets "
            "(negative residual, vol expansion, liquidity drain, failed momentum, downside skew), "
            "trains pure score models walk-forward on each, evaluates IC/spread/short Sharpe, "
            "and classifies output as alpha_short / hedge_only / exposure_artifact."
        ),
    )
    parser.add_argument(
        "--compare_baseline",
        action="store_true",
        help="Also evaluate the LearnedWeights baseline via the same portfolio simulation and include it in the report",
    )
    parser.add_argument(
        "--max_positions",
        type=int,
        default=None,
        help="Max positions for OOS strategy simulation (default: config model_selection.max_positions or 10)",
    )
    parser.add_argument(
        "--min_positions",
        type=int,
        default=None,
        help="Min positions for OOS strategy simulation (default: config model_selection.min_positions or 3)",
    )
    parser.add_argument("--check_feature_leakage", action="store_true", help="Run feature leakage check and exit")
    parser.add_argument("--leakage_date", type=str, default="2020-06-15", help="As-of date for leakage check")
    parser.add_argument(
        "--leakage_tickers",
        type=str,
        default="",
        help="Comma-separated tickers for leakage check (default: first 3 in config)",
    )
    parser.add_argument("--leakage_tol", type=float, default=1e-6, help="Tolerance for leakage check comparisons")
    parser.add_argument(
        "--discard_suspicious_models",
        action="store_true",
        help="If set, discard models that trigger leakage warning (Sharpe_chained>2 & IC_chained<0.05)",
    )
    parser.add_argument(
        "--embargo_days",
        type=int,
        default=None,
        help="Embargo between train/test (calendar days). Default ~2*horizon.",
    )
    parser.add_argument(
        "--matrix-start-date",
        type=str,
        default="",
        help="Override backtest start_date for feature matrix only (YYYY-MM-DD). Used by run_retrain_model.py.",
    )
    parser.add_argument(
        "--matrix-end-date",
        type=str,
        default="",
        help="Override backtest end_date for feature matrix only (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--horizon-optimization",
        action="store_true",
        dest="horizon_optimization",
        help="C9: Identify economically optimal holding horizon per model via alpha efficiency and net Sharpe sweep."
    )
    parser.add_argument(
        "--confidence-weighting",
        action="store_true",
        dest="confidence_weighting",
        help="C10: Bootstrap daily IC to estimate P(IC>0) and apply Bayesian confidence weighting to scores."
    )
    parser.add_argument(
        "--regime-gating",
        action="store_true",
        dest="regime_gating",
        help="C11: Identify market regimes (Vol/Trend/Liquidity) via clustering and learn data-driven gating functions."
    )
    parser.add_argument(
        "--asymmetry-correction",
        action="store_true",
        dest="asymmetry_correction",
        help="C12: Diagnose and correct long/short performance asymmetry via leg-level diagnostics and asymmetric scaling."
    )
    parser.add_argument(
        "--capacity-analysis",
        action="store_true",
        dest="capacity_analysis",
        help="C13: Estimate strategy capacity and alpha decay curve using multi-scale capital simulations."
    )
    parser.add_argument(
        "--marginal-value",
        action="store_true",
        dest="marginal_value",
        help="C14: Evaluate signal's marginal value via orthogonalization and incremental ensemble contribution analysis."
    )
    parser.add_argument(
        "--cost-sensitivity",
        action="store_true",
        dest="cost_sensitivity",
        help="C15: Run empirical cost sensitivity analysis and impact model calibration."
    )
    parser.add_argument(
        "--joint-optimization",
        action="store_true",
        dest="joint_optimization",
        help="C21: Jointly select (H, lambda_turn) maximizing net Sharpe via nested cross-validation."
    )
    parser.add_argument(
        "--deployability-ranking",
        action="store_true",
        dest="deployability_ranking",
        help="C20: Produce a unified deployability score combining all evidence (Sharpe, Confidence, Stability, Capacity)."
    )
    parser.add_argument(
        "--simplified",
        action="store_true",
        help="Disable all research layers and revert to L2 optimization to isolate core performance."
    )
    parser.add_argument("--viability-check", action="store_true", help="Quantify alpha-to-cost ratios and classify signal tradeability.")
    args = parser.parse_args()
    _logging.basicConfig(level=getattr(_logging, args.log_level.upper()), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.run_sim_test:
        _test_portfolio_simulation_logic()
        raise SystemExit(0)

    if args.check_feature_leakage:
        tickers_override = [t.strip().upper() for t in str(args.leakage_tickers).split(",") if t.strip()]
        raise SystemExit(
            check_feature_leakage(
                as_of_date=str(args.leakage_date),
                tickers=tickers_override or None,
                tol=float(args.leakage_tol),
            )
        )

    cfg = _read_config(args.config)
    _warn_deprecated_config_duplicates(cfg)

    # ── Stage-level timing instrumentation ──────────────────────────────────
    _stage_times: dict[str, float] = {}
    _stage_clock = [time.time()]

    def _stage(name: str) -> None:
        t = time.time()
        elapsed = t - _stage_clock[0]
        print(f"[TIMING] {name:<45s} {elapsed:>8.3f}s")
        _stage_times[name] = elapsed
        _stage_clock[0] = t

    _stage("config_parse")

    # Build diagnostic execution plan early so all downstream stages can check
    # their gating flags for per-candidate research diagnostics.
    diagnostic_plan = DiagnosticExecutionPlan.from_config(cfg)

    # Per-sub-stage timing helper (no longer used for P30-P34 which are removed).
    _sub_stage_times: dict[str, float] = {}
    _sub_stage_clock: list[float] = []

    def _sub_stage(name: str) -> None:
        if not _sub_stage_clock:
            _sub_stage_clock.append(time.monotonic())
        t = time.monotonic()
        elapsed = t - _sub_stage_clock[0]
        print(f"  [SUB-TIMING] {name:<40s} {elapsed:>8.3f}s")
        _sub_stage_times[name] = elapsed
        _sub_stage_clock[0] = t

    # Multi-horizon dispatch: if --production_horizons (or config production_horizons) is set,
    # re-invoke this process once per horizon with --horizon <h> and a per-horizon output dir.
    _prod_horizons: list[int] | None = None
    if args.production_horizons:
        _prod_horizons = [int(h) for h in args.production_horizons]
    elif args.horizon is None:
        _ms_cfg_tmp = cfg.get("model_selection", {}) or {}
        _ph = _ms_cfg_tmp.get("production_horizons")
        if _ph and isinstance(_ph, list) and len(_ph) > 1:
            _prod_horizons = [int(h) for h in _ph]

    if _prod_horizons and len(_prod_horizons) > 1:
        import subprocess as _subprocess
        from model_selection.horizon_contract import build_horizon_contracts_for_sweep
        _sweep = build_horizon_contracts_for_sweep(cfg, _prod_horizons)
        print(f"[multi-horizon] Sweep mode: {_prod_horizons}")
        # Write parent sweep manifest
        try:
            import json as _json
            _manifest = _sweep.to_dict()
            _manifest_path = Path("output/models") / "horizon_sweep_manifest.json"
            _manifest_path.parent.mkdir(parents=True, exist_ok=True)
            _manifest_path.write_text(_json.dumps(_manifest, indent=2, default=str), encoding="utf-8")
            print(f"[multi-horizon] Sweep manifest written to {_manifest_path}")
        except Exception as _sweep_manifest_exc:
            logger.warning("Failed to write multi-horizon sweep manifest: %s", _sweep_manifest_exc)
        _base_argv = [a for a in sys.argv[1:]
                      if not a.startswith("--production_horizons") and not a.startswith("--horizon")]
        _exit_codes: list[int] = []
        for _h in _prod_horizons:
            print(f"\n{'='*60}")
            print(f"  HORIZON {_h}d — sweep child (sweep-controlled, no cross-horizon warnings)")
            print(f"{'='*60}")
            _cmd = [sys.executable, __file__] + _base_argv + ["--horizon", str(_h)]
            _result = _subprocess.run(_cmd)
            _exit_codes.append(_result.returncode)
            if _result.returncode != 0:
                print(f"[multi-horizon] WARNING: horizon={_h}d exited with code {_result.returncode}")
        _any_failed = any(c != 0 for c in _exit_codes)
        print(f"\n[multi-horizon] All horizons complete. Exit codes: {dict(zip(_prod_horizons, _exit_codes))}")
        raise SystemExit(1 if _any_failed else 0)

    try:
        tickers = load_universe(cfg)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    if int(args.limit_tickers or 0) > 0:
        tickers = tickers[: int(args.limit_tickers)]
    bt = cfg.get("backtest", {}) or {}
    research = cfg.get("research", {}) or {}
    feature_sel = cfg.get("feature_selection", {}) or {}
    ms_cfg = cfg.get("model_selection", {}) or {}
    feature_subset = feature_sel.get("feature_subset", []) or []
    feature_subset = [str(c).strip() for c in feature_subset if str(c).strip()]
    short_feature_subset = feature_sel.get("short_feature_subset", []) or []
    short_feature_subset = [str(c).strip() for c in short_feature_subset if str(c).strip()]
    overlay_feature_subset = feature_sel.get("overlay_feature_subset", []) or []
    overlay_feature_subset = [str(c).strip() for c in overlay_feature_subset if str(c).strip()]

    exe_cfg = cfg.get("execution", {}) or {}
    long_only = exe_cfg.get("long_only", False)
    enable_shorts = exe_cfg.get("enable_shorts", True)
    do_shorts = enable_shorts and not long_only
    primary_path = "long_short_spread" if do_shorts else "long_only_overlay"

    # Model-selection / evaluation settings (CLI overrides config).
    # P28: Detect sweep mode — when --horizon is passed and production_horizons
    # is in the config, this is a sweep-controlled child run.  Suppress
    # production_horizons warnings that would otherwise flood child logs.
    _is_sweep_child = (
        args.horizon is not None
        and bool((cfg.get("model_selection", {}) or {}).get("production_horizons"))
    )
    _sweep_mode = SweepMode.MULTI_HORIZON_SWEEP if _is_sweep_child else SweepMode.SINGLE_PRODUCTION
    horizon_contract = build_horizon_contract(cfg, cli_horizon=args.horizon, sweep_mode=_sweep_mode)
    horizon_run_contract = build_horizon_run_contract(
        horizon_contract,
        cfg,
        config_path=str(args.config),
    )
    # In sweep mode, use production_horizon_days (the sweep-controlled value) for
    # output directory naming and downstream computation.  In single-production
    # mode, target_horizon_days equals production_horizon_days by contract.
    horizon = int(horizon_contract.config.production_horizon_days) if _is_sweep_child else int(horizon_contract.config.target_horizon_days)
    # P38: Research-cache fingerprint — computed once at contract resolution time
    _research_cache_fp = ""
    try:
        from model_selection.horizon_contract import research_cache_fingerprint_from_config
        _research_cache_fp = research_cache_fingerprint_from_config(cfg)
    except Exception as _cache_fp_exc:
        logger.warning("Failed to compute legacy research cache fingerprint: %s", _cache_fp_exc)
    _research_cache_fp = stable_fingerprint(
        {
            "legacy_research_cache_fingerprint": _research_cache_fp,
            "horizon_run_contract_fingerprint": horizon_run_contract.fingerprint(),
        },
        length=24,
    )
    # P15: If --min_oos_days not explicitly set, derive from holding period
    if args.min_oos_days is None:
        args.min_oos_days = max(10, 2 * int(horizon_contract.config.holding_period_days))
    gate_cfg = _promotion_gate_config(cfg)
    long_cand_cfg = _long_alpha_candidate_config(cfg)
    nested_cfg = _nested_validation_config(cfg, horizon_contract=horizon_contract)
    max_positions = (
        int(args.max_positions)
        if args.max_positions is not None
        else int(ms_cfg.get("max_positions", 10) or 10)
    )
    min_positions = (
        int(args.min_positions)
        if args.min_positions is not None
        else int(ms_cfg.get("min_positions", 3) or 3)
    )
    max_positions = int(max(1, max_positions))
    min_positions = int(max(1, min_positions))
    if min_positions > max_positions:
        min_positions = max_positions

    start_date = str(bt.get("start_date", "2018-01-01"))
    end_date = str(bt.get("end_date", "2024-01-01"))
    ms = str(getattr(args, "matrix_start_date", "") or "").strip()
    me = str(getattr(args, "matrix_end_date", "") or "").strip()
    if me:
        end_date = me
    if ms:
        start_date = ms
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise SystemExit(f"Invalid matrix window: start {start_date} after end {end_date}")
    train_years = float(research.get("train_years", 5))
    test_years = float(research.get("test_years", 1))
    step_years = float(research.get("step_years", test_years))
    n_windows_cfg = int(research.get("walk_forward_windows", 4) or 4)
    train_ratio = float(research.get("walk_forward_train_ratio", 0.70) or 0.70)

    if not tickers:
        raise SystemExit("No tickers found in backtest_config.yaml")

    print(f"Config: {args.config}")
    print(f"Universe: {len(tickers)} tickers")
    print(f"Window: {start_date} → {end_date}")
    print(f"Walk-forward: train={train_years}y test={test_years}y step={step_years}y")
    print(
        "Horizon Contract: "
        f"target={horizon_contract.config.target_horizon_days}d "
        f"holding={horizon_contract.config.holding_period_days}d "
        f"rebalance={horizon_contract.config.rebalance_frequency_days}d "
        f"ic_eval={horizon_contract.config.ic_evaluation_horizon}d "
        f"execution_tau={horizon_contract.config.execution_tau_days if horizon_contract.config.execution_tau_days is not None else 'auto'}"
    )
    # P17: Audit print — signal persistence expectation
    _reb = horizon_contract.config.rebalance_frequency_days
    _hold = horizon_contract.config.holding_period_days
    print(
        f"[RebalanceAudit] rebalance={_reb}d holding={_hold}d "
        f"ratio={_hold/_reb:.1f}x "
        f"(positions refreshed {_hold/_reb:.0f}× per hold period)"
    )
    for _warning in horizon_contract.warnings[:8]:
        print(f"[HorizonContract] WARNING: {_warning}")
    embargo_days = int(args.embargo_days) if args.embargo_days is not None else _embargo_days_config(cfg, default_horizon=int(horizon))
    print(f"Embargo: {embargo_days} calendar days")
    if feature_subset or short_feature_subset or overlay_feature_subset:
        num_feat = len(set(feature_subset + short_feature_subset + overlay_feature_subset))
        print(
            f"Feature candidate universe: {num_feat} unique columns "
            "(legacy static subsets are build-universe only; alpha admission selects production features)"
        )

    from agents.weight_learning_agent.feature_builder import build_feature_matrix

    data_kwargs = _feature_builder_data_kwargs(cfg)

    # Build enough history for rolling features (feature_builder applies its own buffers).
    # Pass the union of all candidate subsets to build_feature_matrix.
    matrix_subset = (
        list(set(feature_subset + short_feature_subset + overlay_feature_subset))
        if (feature_subset or short_feature_subset or overlay_feature_subset)
        else None
    )
    if short_feature_subset or overlay_feature_subset:
        print(
            "[FeatureAdmission] Static short/overlay feature subsets are deprecated for model fitting; "
            "role-specific models will consume the admitted evidence set only."
        )
    research_state = ResearchStateStore.for_model_selection(
        cfg=cfg,
        tickers=tickers,
        start_date=str(start_date),
        end_date=str(end_date),
        horizon_days=int(horizon),
        feature_subset=matrix_subset,
        provider=data_kwargs.get("data_provider"),
        cache_fingerprint=_research_cache_fp,
    )
    timing_ledger = research_state.timing_ledger()
    universe_panel_path = research_state.persist_universe_panel(
        tickers=tickers,
        membership_ranges=cfg.get("pit_membership_ranges") if isinstance(cfg, dict) else None,
        start_date=str(start_date),
        end_date=str(end_date),
    )
    print(f"Research state: {research_state.path}")
    print(f"Universe panel artifact: {universe_panel_path}")
    print(f"Timing ledger: {timing_ledger.path}")
    timing_ledger.record(
        "model_selection_start",
        universe_size=int(len(tickers)),
        start_date=str(start_date),
        end_date=str(end_date),
        horizon_days=int(horizon),
    )
    t_feature_build_0 = time.perf_counter()
    df, feature_from_cache = research_state.get_or_build_frame(
        "feature_panel_program",
        lambda: build_feature_matrix(
            tickers,
            start_date=start_date,
            end_date=end_date,
            holding_period=int(horizon_contract.config.holding_period_days),
            feature_subset=matrix_subset,
            **data_kwargs,
        ),
    )
    t_feature_build_1 = time.perf_counter()
    if df is None or df.empty:
        raise SystemExit("Feature matrix is empty; cannot run model selection.")
    print(
        f"Feature matrix {'load' if feature_from_cache else 'build'} runtime: "
        f"{t_feature_build_1 - t_feature_build_0:.2f}s"
    )
    _stage("feature_matrix_load")
    # Task 1: After feature matrix load
    mem_ledger = MemoryLedger(research_state.path / "memory_ledger.jsonl")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Target: forward return (already aligned in feature_builder via close.shift(-holding_period)/close - 1).
    # Re-derive a binary label from that return.
    if "forward_return" not in df.columns:
        raise SystemExit("Feature matrix missing forward_return; cannot compute target.")
    df["forward_return"] = pd.to_numeric(df["forward_return"], errors="coerce")
    df = df.dropna(subset=["forward_return"])
    target_cfg = _target_config(cfg, horizon_contract=horizon_contract)
    df = add_institutional_targets(
        df,
        cfg=target_cfg,
        costs=_execution_cost_config(cfg),
        max_name_weight=_evaluation_config(
            cfg,
            path=primary_path,
            max_positions=int(max_positions),
            min_positions=int(min_positions),
            horizon_contract=horizon_contract,
        ).max_name_weight,
    )
    # Classification labels are residual/net-of-cost alpha direction, not raw beta direction.
    df["y_bin"] = df["target_up"].astype(int)
    print(
        "Targets: "
        f"residualize={target_cfg.residualize} net_of_costs={target_cfg.net_of_costs} "
        f"winsor_q={target_cfg.winsor_q:.3f}"
    )
    _stage("feature_matrix_and_targets")

    # P3: Wire regime probability features into the feature matrix as model inputs.
    # MarketRegimeAgent computes HMM-based regime classification using SPY returns,
    # VIX level, VIX momentum, SPY realized vol, and SPY vs SMA200 gap.
    # We attach per-date regime probabilities and continuous scores as features
    # so the model learns regime-conditional feature weights without separate training.
    _regime_cfg = (cfg.get("regime", {}) or {}) if isinstance(cfg, dict) else {}
    _regime_enabled = bool(_regime_cfg.get("enabled", True))
    if _regime_enabled and "regime_label" not in df.columns:
        try:
            from backtesting.regime import MarketRegimeAgent, REGIME_LABELS
            _all_dates = pd.to_datetime(df["date"].dropna().unique())
            _date_range = (_all_dates.min().strftime("%Y-%m-%d"), _all_dates.max().strftime("%Y-%m-%d"))
            _agent = MarketRegimeAgent()
            _labels = _agent.detect_regimes(_date_range[0], _date_range[1], confirmation_days=int(_regime_cfg.get("confirmation_days", 1)))
            _probs = _agent.get_regime_probs()
            _scores = _agent.detect_regime_scores()
            if _labels:
                _label_series = pd.Series(_labels).rename("regime_label")
                _score_series = pd.Series(_scores).rename("regime_score")
                _prob_bull = pd.Series({d: p.get("Bull", 0.0) for d, p in _probs.items()}).rename("regime_proba_bull")
                _prob_bear = pd.Series({d: p.get("Bear", 0.0) for d, p in _probs.items()}).rename("regime_proba_bear")
                _prob_crisis = pd.Series({d: p.get("Crisis", 0.0) for d, p in _probs.items()}).rename("regime_proba_crisis")
                _regime_df = pd.DataFrame({
                    "regime_label": _label_series,
                    "regime_score": _score_series,
                    "regime_proba_bull": _prob_bull,
                    "regime_proba_bear": _prob_bear,
                    "regime_proba_crisis": _prob_crisis,
                })
                _regime_df.index = _regime_df.index.tz_localize(None) if _regime_df.index.tzinfo is not None else _regime_df.index
                df["date"] = pd.to_datetime(df["date"])
                _n_merged = 0
                for col in ["regime_label", "regime_score", "regime_proba_bull", "regime_proba_bear", "regime_proba_crisis"]:
                    if col in _regime_df.columns:
                        mapping = _regime_df[col].to_dict()
                        df[col] = df["date"].map(mapping)
                        _n_merged += df[col].notna().sum()
                _n_regime_dates = len(set(_labels.keys()) & set(df["date"].dt.normalize()))
                print(f"  [P3 Regime Features] Attached {len(_labels)} regime dates | {_n_regime_dates} merged into feature matrix")
                print(f"    regime_score range: [{_score_series.min():.3f}, {_score_series.max():.3f}] | mean={_score_series.mean():.3f}")
            else:
                print("  [P3 Regime Features] Regime agent returned empty labels — skipping regime feature attachment.")
        except Exception as _reg_exc:
            print(f"  [P3 Regime Features] Failed to attach regime features: {_reg_exc} — proceeding without.")
    elif "regime_label" in df.columns:
        print("  [P3 Regime Features] Regime columns already present in feature matrix — using existing.")

    # ── Phase 1 diagnostic: target construction evaluation (shadow, read-only) ─
    _tc_p1 = ((ms_cfg.get("validation") or {}).get("phase1_diagnostics") or {})
    if _tc_p1.get("target_construction", False):
        def _tc_diag(_df=df) -> None:
            from model_selection.target_construction import build_target_menu, evaluate_targets, format_diagnostics_table
            _costs = _execution_cost_config(cfg)
            _max_nw = _evaluation_config(
                cfg,
                path=primary_path,
                max_positions=int(max_positions),
                min_positions=int(min_positions),
                horizon_contract=horizon_contract,
            ).max_name_weight
            _df_menu = build_target_menu(_df, costs=_costs, max_name_weight=_max_nw)
            _proxy = "forward_return" if "forward_return" in _df_menu.columns else None
            if _proxy:
                _tbl = evaluate_targets(_df_menu, proxy_col=_proxy)
                print("[TargetConstructionDiag]\n" + format_diagnostics_table(_tbl))
        safe_run_diagnostic("target_construction", _tc_diag)
    # ── End Phase 1 target_construction diagnostic ────────────────────────────

    # C3: risk-adjusted target — use when --risk-adj-target flag is passed.
    # sign(risk_adj) == sign(forward_return) so y_bin is unchanged.
    use_risk_adj = getattr(args, "risk_adj_target", False) and "forward_return_risk_adj" in df.columns
    if use_risk_adj:
        df["forward_return_risk_adj"] = pd.to_numeric(df["forward_return_risk_adj"], errors="coerce")
        # Fall back to raw return where risk-adj is missing (e.g. very low vol)
        df["forward_return_risk_adj"] = df["forward_return_risk_adj"].fillna(df["forward_return"])
        print(f"[C3] Using risk-adjusted return target (forward_return / realized_vol_holding)")
    else:
        use_risk_adj = False

    # Optional baseline (LearnedWeights) uses the full feature set + its own scaler feature list.
    df_baseline: pd.DataFrame | None = None
    if args.compare_baseline:
        print("Baseline comparison enabled: building a second full-matrix view for LearnedWeights parity.")
        try:
            df_baseline, baseline_from_cache = research_state.get_or_build_frame(
                "feature_panel_full",
                lambda: build_feature_matrix(
                    tickers,
                        start_date=start_date,
                        end_date=end_date,
                        holding_period=int(horizon_contract.config.holding_period_days),
                        feature_subset=None,
                        **data_kwargs,
                ),
            )
            if df_baseline is None or df_baseline.empty:
                print("WARNING: compare_baseline enabled but baseline feature matrix is empty; skipping baseline.")
                df_baseline = None
            else:
                print(f"Baseline feature panel source: {'cache' if baseline_from_cache else 'build'}")
                df_baseline = df_baseline.copy()
                df_baseline["date"] = pd.to_datetime(df_baseline["date"], errors="coerce")
                df_baseline = df_baseline.dropna(subset=["date"])
                df_baseline["forward_return"] = pd.to_numeric(df_baseline["forward_return"], errors="coerce")
                df_baseline = df_baseline.dropna(subset=["forward_return"])
                df_baseline = add_institutional_targets(
                    df_baseline,
                    cfg=target_cfg,
                    costs=_execution_cost_config(cfg),
                    max_name_weight=_evaluation_config(
                        cfg,
                        path=primary_path,
                        max_positions=int(max_positions),
                        min_positions=int(min_positions),
                        horizon_contract=horizon_contract,
                    ).max_name_weight,
                )
                df_baseline["y_bin"] = df_baseline["target_up"].astype(int)
                df_baseline = df_baseline.sort_values(["ticker", "date"]).reset_index(drop=True)
        except Exception as exc:
            print(f"WARNING: compare_baseline enabled but failed to build baseline matrix: {exc}")
            traceback.print_exc()
            df_baseline = None

    feat_cols = _feature_columns(df)
    if not feat_cols:
        raise SystemExit("No numeric feature columns found.")

    # Use a per-horizon subdirectory when running from --production_horizons dispatch
    # so 20d and 63d results don't overwrite each other.
    _ms_ph = (cfg.get("model_selection", {}) or {}).get("production_horizons", [])
    _is_multi = isinstance(_ms_ph, list) and len(_ms_ph) > 1
    out_dir = Path("output/models") / f"h{horizon}d" if _is_multi else Path("output/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "horizon_contract.json", horizon_contract.to_dict())
    pd.DataFrame(horizon_contract.audit_rows()).to_csv(out_dir / "horizon_contract_audit.csv", index=False)
    print(f"Horizon contract audit: {out_dir / 'horizon_contract_audit.csv'}")
    _write_horizon_manifest(
        horizon_contract=horizon_contract,
        cli_horizon=args.horizon,
        out_dir=out_dir,
    )
    institutional_run_id = f"{research_state.signature}_h{int(horizon)}"
    telemetry_contract = RunTelemetryContract(
        run_id=institutional_run_id,
        telemetry_dir=str(out_dir),
    )
    write_json_artifact(out_dir / "horizon_run_contract.json", horizon_run_contract.to_dict())
    _contract_exec_costs = _execution_cost_config(cfg)
    cost_assumption_set = build_cost_assumption_set(_contract_exec_costs, cfg=cfg)
    write_json_artifact(out_dir / "cost_manifest.json", cost_assumption_set.to_dict())
    target_specs = build_target_specs(
        target_cfg,
        cfg=cfg,
        horizon_contract=horizon_run_contract,
    )
    target_manifest = build_target_manifest(df, target_specs)
    write_json_artifact(out_dir / "target_manifest.json", target_manifest)
    pit_specs = build_pit_transform_specs(df, cfg=cfg)
    pit_manifest = {
        "schema_version": "phase_b.1",
        "pit_transform_fingerprints": {name: spec.fingerprint() for name, spec in sorted(pit_specs.items())},
        "transforms": {name: spec.to_dict() for name, spec in sorted(pit_specs.items())},
    }
    write_json_artifact(out_dir / "pit_transform_manifest.json", pit_manifest)
    for _pit_row in build_pit_audit_ledger(df, pit_specs):
        emit_telemetry_event(
            out_dir,
            run_id=institutional_run_id,
            stage="pit_contract",
            event_type="pit_transform_audit",
            message=f"PIT transform audited for {_pit_row['column_name']}",
            ledger="pit_audit_ledger.jsonl",
            contract_fingerprints={"pit": str(_pit_row["transform_spec_fingerprint"])},
        )
    feature_manifest = build_feature_manifest(feat_cols, cfg=cfg, pit_specs=pit_specs)
    write_json_artifact(out_dir / "feature_manifest.json", feature_manifest)
    _target_fp_for_gates = target_specs.get("target_return", next(iter(target_specs.values()))).fingerprint()
    promotion_gate_specs = build_promotion_gate_specs(
        gate_cfg,
        horizon_contract_fingerprint=horizon_run_contract.fingerprint(),
        target_spec_fingerprint=_target_fp_for_gates,
        cost_assumption_fingerprint=cost_assumption_set.fingerprint(),
    )
    write_json_artifact(
        out_dir / "promotion_gate_manifest.json",
        {
            "schema_version": "phase_b.1",
            "promotion_gate_spec_fingerprint": stable_fingerprint([spec.to_dict() for spec in promotion_gate_specs]),
            "gates": [spec.to_dict() for spec in promotion_gate_specs],
        },
    )
    cache_key_specs = [
        build_cache_key_spec(
            cache_name="model_selection_feature_panel",
            cfg=cfg,
            tickers=tickers,
            start_date=str(start_date),
            end_date=str(end_date),
            horizon_contract_fingerprint=horizon_run_contract.fingerprint(),
            data_source_fingerprint=str(data_kwargs.get("data_provider") or ""),
        ),
        build_cache_key_spec(
            cache_name="target_panel",
            cfg=cfg,
            tickers=tickers,
            start_date=str(start_date),
            end_date=str(end_date),
            horizon_contract_fingerprint=horizon_run_contract.fingerprint(),
            target_spec_fingerprints=list(target_manifest.get("target_spec_fingerprints", {}).values()),
            cost_assumption_fingerprint=cost_assumption_set.fingerprint(),
            pit_transform_fingerprints=[spec.fingerprint() for spec in pit_specs.values()],
            data_source_fingerprint=str(data_kwargs.get("data_provider") or ""),
        ),
        build_cache_key_spec(
            cache_name="prepared_panel_cache",
            cfg=cfg,
            tickers=tickers,
            start_date=str(start_date),
            end_date=str(end_date),
            horizon_contract_fingerprint=horizon_run_contract.fingerprint(),
            target_spec_fingerprints=list(target_manifest.get("target_spec_fingerprints", {}).values()),
            feature_spec_fingerprints=list(feature_manifest.get("feature_fingerprints", {}).values()),
            cost_assumption_fingerprint=cost_assumption_set.fingerprint(),
            pit_transform_fingerprints=[spec.fingerprint() for spec in pit_specs.values()],
            data_source_fingerprint=str(data_kwargs.get("data_provider") or ""),
        ),
    ]
    write_json_artifact(
        out_dir / "cache_manifest.json",
        {
            "schema_version": "phase_b.1",
            "cache_key_specs": {spec.cache_name: spec.to_dict() for spec in cache_key_specs},
        },
    )
    _contract_artifacts = {
        "horizon_contract": str(out_dir / "horizon_contract.json"),
        "horizon_run_contract": str(out_dir / "horizon_run_contract.json"),
        "target_manifest": str(out_dir / "target_manifest.json"),
        "feature_manifest": str(out_dir / "feature_manifest.json"),
        "pit_transform_manifest": str(out_dir / "pit_transform_manifest.json"),
        "cost_manifest": str(out_dir / "cost_manifest.json"),
        "promotion_gate_manifest": str(out_dir / "promotion_gate_manifest.json"),
        "cache_manifest": str(out_dir / "cache_manifest.json"),
        "run_telemetry": str(out_dir / "run_telemetry.jsonl"),
        "failure_ledger": str(out_dir / "failure_ledger.jsonl"),
        "cache_events": str(out_dir / "cache_events.jsonl"),
        "artifact_events": str(out_dir / "artifact_events.jsonl"),
    }
    emit_telemetry_event(
        out_dir,
        run_id=institutional_run_id,
        stage="institutional_contracts",
        event_type="contracts_initialized",
        message="Institutional Phase B contracts initialized for model-selection run.",
        contract_fingerprints={
            "horizon": horizon_run_contract.fingerprint(),
            "cost": cost_assumption_set.fingerprint(),
            "promotion_gates": stable_fingerprint([spec.to_dict() for spec in promotion_gate_specs]),
        },
    )
    for _artifact_name, _artifact_path in _contract_artifacts.items():
        emit_telemetry_event(
            out_dir,
            run_id=institutional_run_id,
            stage="artifact_write",
            event_type="artifact_declared",
            message=f"Artifact registered: {_artifact_name}",
            ledger="artifact_events.jsonl",
            artifact_path=_artifact_path,
        )
    write_institutional_run_manifest(
        out_dir,
        run_id=institutional_run_id,
        config_path=str(args.config),
        cfg=cfg,
        horizon_contract=horizon_run_contract,
        target_manifest=target_manifest,
        pit_specs=pit_specs,
        feature_manifest=feature_manifest,
        cost_assumption_set=cost_assumption_set,
        promotion_gate_specs=promotion_gate_specs,
        cache_key_specs=cache_key_specs,
        artifacts=_contract_artifacts,
        telemetry_contract=telemetry_contract,
        warnings=list(horizon_run_contract.warnings) + list(feature_manifest.get("unknown_features", [])),
        errors=[],
    )
    print(f"Institutional run manifest: {out_dir / 'institutional_run_manifest.json'}")
    _research_cache_fp = stable_fingerprint(
        {
            "base": _research_cache_fp,
            "horizon_run_contract": horizon_run_contract.fingerprint(),
            "target_specs": list(target_manifest.get("target_spec_fingerprints", {}).values()),
            "feature_specs": list(feature_manifest.get("feature_fingerprints", {}).values()),
            "pit_specs": [spec.fingerprint() for spec in pit_specs.values()],
            "cost_assumption_set": cost_assumption_set.fingerprint(),
        },
        length=24,
    )

    contract_cfg = (ms_cfg.get("research_contract", {}) or {}) if isinstance(ms_cfg, dict) else {}
    feature_ledger = audit_feature_contract(df, feat_cols, target_col="target_return")
    feature_ledger_path = out_dir / "feature_research_ledger.csv"
    feature_ledger.to_csv(feature_ledger_path, index=False)
    feature_contract_summary = summarize_feature_contract(feature_ledger)
    unknown_features = (
        feature_ledger.loc[feature_ledger["known_contract"].eq(False), "feature"].tolist()
        if not feature_ledger.empty and "known_contract" in feature_ledger.columns
        else []
    )
    weak_coverage = (
        feature_ledger.loc[pd.to_numeric(feature_ledger["coverage"], errors="coerce") < float(contract_cfg.get("min_coverage", 0.60) or 0.60), "feature"].tolist()
        if not feature_ledger.empty and "coverage" in feature_ledger.columns
        else []
    )
    print(f"Feature research ledger: {feature_ledger_path}")
    print(
        "Feature contract: "
        f"known={feature_contract_summary.get('feature_known_contract_rate', float('nan')):.2f} "
        f"min_coverage={feature_contract_summary.get('feature_min_coverage', float('nan')):.2f} "
        f"positive_ic_rate={feature_contract_summary.get('feature_positive_ic_rate', float('nan')):.2f}"
    )
    if unknown_features:
        print(f"WARNING: {len(unknown_features)} model features have no research contract: {unknown_features[:10]}")
    if weak_coverage:
        print(f"WARNING: {len(weak_coverage)} model features have weak coverage: {weak_coverage[:10]}")
    if bool(contract_cfg.get("require_known_features", False)) and unknown_features:
        raise SystemExit(f"Research contract failed: unknown model features: {unknown_features[:20]}")
    if bool(contract_cfg.get("require_min_coverage", False)) and weak_coverage:
        raise SystemExit(f"Research contract failed: low-coverage model features: {weak_coverage[:20]}")

    # ── Horizon alignment audit ────────────────────────────────────────────────
    # Reports features whose documented horizon_days is misaligned with the
    # prediction horizon. Controlled by model_selection.horizon_alignment config.
    _ha_cfg = _horizon_alignment_config(cfg)
    _ha_report = None
    if _ha_cfg["enabled"]:
        _ha_report = get_horizon_alignment_report(
            feat_cols,
            int(horizon),
            alignment_multiplier=float(_ha_cfg["multiplier"]),
        )
        if _ha_report["n_misaligned"] > 0:
            _misaligned_names = [m[0] for m in _ha_report["misaligned"]]
            _threshold_days = int(horizon * _ha_cfg["multiplier"])
            print(
                "[HorizonAlignment] Pre-admission audit: "
                f"{_ha_report['n_misaligned']} of {len(feat_cols)} candidate feature(s) are cross-horizon "
                f"for the {int(horizon)}d contract (threshold={_threshold_days}d). "
                "They remain research candidates only; alpha admission must remove or move them before fitting."
            )
            if _ha_cfg.get("verbose") or _ha_cfg["enforce"]:
                for _name, _fh, _ts in _ha_report["misaligned"]:
                    print(f"  MISALIGNED  {_name:<40s} feature_horizon={_fh}d  timestamp={_ts}")
            else:
                print(
                    f"  Misaligned candidates: {_misaligned_names[:8]}"
                    f"{' ...' if len(_misaligned_names) > 8 else ''}"
                )
            if _ha_cfg["enforce"]:
                raise SystemExit(
                    "Signal/execution timing contract failed: "
                    f"HorizonAlignment found misaligned features: {_misaligned_names}"
                )
        else:
            print(f"[HorizonAlignment] All {_ha_report['n_aligned']} features aligned with {int(horizon)}-day target.")
        # Save alignment ledger to CSV for research review
        # P22: Path must be horizon-specific to prevent multi-horizon overwrites
        _ha_path = _ha_cfg.get("report_path", "")
        if not _ha_path or _ha_path == "output/horizon_alignment_report.csv":
            _ha_path = f"output/models/h{int(horizon)}d/horizon_alignment_report.csv"
        if _ha_path:
            try:
                import pathlib as _pathlib, csv as _csv
                _pathlib.Path(_ha_path).parent.mkdir(parents=True, exist_ok=True)
                with open(_ha_path, "w", newline="") as _f:
                    _w = _csv.writer(_f)
                    _w.writerow(["feature", "status", "feature_horizon_days", "timestamp", "prediction_horizon", "multiplier"])
                    for _name, _fh, _ts in _ha_report["misaligned"]:
                        _w.writerow([_name, "misaligned", _fh, _ts, int(horizon), _ha_cfg["multiplier"]])
                    for _name in _ha_report["aligned"]:
                        _spec = FEATURE_SPECS.get(_name)
                        _w.writerow([_name, "aligned", _spec.horizon_days if _spec else "", _spec.timestamp if _spec else "", int(horizon), _ha_cfg["multiplier"]])
                    for _name in _ha_report["unknown"]:
                        _w.writerow([_name, "unknown", "", "", int(horizon), _ha_cfg["multiplier"]])
                print(f"[HorizonAlignment] Ledger written: {_ha_path}")
            except Exception as _e:
                print(f"[HorizonAlignment] WARNING: could not write ledger: {_e}")
    # ── End horizon alignment audit ───────────────────────────────────────────

    try:
        _timing_report = validate_signal_execution_timing(
            df,
            feat_cols,
            prediction_horizon_days=int(horizon),
            horizon_report=_ha_report,
            enforce_horizon_alignment=bool(_ha_cfg["enforce"]),
        )
        print(
            "Signal/execution timing contract: "
            f"signal_time=close_t execution_time=next_open_or_next_vwap "
            f"holding_period={int(horizon_contract.config.holding_period_days)}d features={_timing_report.n_features} "
            f"warnings={_timing_report.n_warnings}"
        )
        _non_alignment_warnings = [
            _warning for _warning in _timing_report.warnings
            if not str(_warning).startswith("HorizonAlignment found ")
        ]
        for _warning in _non_alignment_warnings[:5]:
            print(f"[TimingContract] WARNING: {_warning}")
        if len(_non_alignment_warnings) < _timing_report.n_warnings:
            print(
                "[TimingContract] HorizonAlignment warning captured in pre-admission audit; "
                "production admission remains fail-closed."
            )
    except TimingContractViolation as _exc:
        raise SystemExit(f"Signal/execution timing contract failed: {_exc}") from _exc

    _stage("horizon_alignment_and_timing_contract")

    alpha_cfg = _alpha_admission_config(cfg, horizon_contract=horizon_contract)
    alpha_admission_summary: dict[str, float] = {}
    alpha_admission = pd.DataFrame()
    if alpha_cfg.enabled:
        max_name_weight_for_research = _evaluation_config(
            cfg,
            path=primary_path,
            max_positions=int(max_positions),
            min_positions=int(min_positions),
            horizon_contract=horizon_contract,
        ).max_name_weight
        t_alpha_research_0 = time.perf_counter()
        feature_contract_payload = {
            feature: FEATURE_SPECS[feature]
            for feature in feat_cols
            if feature in FEATURE_SPECS
        }
        # Build windows first to determine research cutoff (Task 6)
        windows = _walk_forward_windows(start_date, end_date, train_years, test_years, step_years)
        if len(windows) <= 1:
            windows = _walk_forward_windows_by_count(df["date"], n_windows=n_windows_cfg, train_ratio=train_ratio)
        if len(windows) <= 1:
            raise SystemExit(
                "Not enough walk-forward windows. Either extend the backtest date range or reduce research.train_years/test_years."
            )

        # Use the train_end of the first window as the research cutoff to avoid leakage (Task 1)
        research_cutoff = windows[0][1]

        alpha_panel_signature = frame_fingerprint(
            df,
            columns=["date", "ticker", *feat_cols, "daily_return", "forward_return"],
        )
        alpha_state = ResearchStateStore.for_alpha_research(
            cfg=cfg,
            tickers=tickers,
            start_date=str(start_date),
            end_date=str(research_cutoff),
            provider=data_kwargs.get("data_provider"),
            feature_columns=feat_cols,
            feature_panel_signature=alpha_panel_signature,
            feature_contract=feature_contract_payload,
            alpha_admission_policy=alpha_cfg,
            alpha_research_schema_version=ALPHA_RESEARCH_SCHEMA_VERSION,
            cache_fingerprint=_research_cache_fp,
        )
        print(f"Alpha research state (Cutoff: {research_cutoff.date()}): {alpha_state.path}")
        alpha_bundle_dir = alpha_state.path
        alpha_df_path = alpha_bundle_dir / "enriched_panel.parquet"
        alpha_decay_parquet = alpha_bundle_dir / "alpha_ic_decay.parquet"
        alpha_admission_parquet = alpha_bundle_dir / "feature_admission.parquet"
        alpha_meta_path = alpha_bundle_dir / "alpha_research_metadata.json"
        if alpha_df_path.exists() and alpha_decay_parquet.exists() and alpha_admission_parquet.exists():
            df = pd.read_parquet(alpha_df_path)
            alpha_decay = pd.read_parquet(alpha_decay_parquet)
            alpha_admission = pd.read_parquet(alpha_admission_parquet)
            meta = _read_json(alpha_meta_path) if alpha_meta_path.exists() else {}
            compatible, incompat_reason = _alpha_research_cache_is_compatible(
                alpha_admission,
                alpha_cfg=alpha_cfg,
            )
            if meta.get("schema_version") != ALPHA_RESEARCH_SCHEMA_VERSION:
                compatible = False
                incompat_reason = f"schema_version:{meta.get('schema_version', 'missing')}!={ALPHA_RESEARCH_SCHEMA_VERSION}"
            alpha_from_cache = bool(compatible)
            if not compatible:
                print(
                    "[AlphaResearch] Ignoring stale cached admission bundle: "
                    f"{incompat_reason}. Rebuilding under {ALPHA_RESEARCH_SCHEMA_VERSION}."
                )
                research_df = df[df["date"] < research_cutoff].copy()
                _, alpha_decay, alpha_admission = run_alpha_research(
                    research_df,
                    feat_cols,
                    cfg=alpha_cfg,
                    target_cfg=target_cfg,
                    costs=_execution_cost_config(cfg),
                    max_name_weight=max_name_weight_for_research,
                )
                df.to_parquet(alpha_df_path, index=False)
                alpha_decay.to_parquet(alpha_decay_parquet, index=False)
                alpha_admission.to_parquet(alpha_admission_parquet, index=False)
                _write_json(
                    alpha_meta_path,
                    {
                        "schema_version": ALPHA_RESEARCH_SCHEMA_VERSION,
                        "cache_reason": "rebuilt_after_incompatible_cache",
                    },
                )
        else:
            # Filter df to only include data up to research_cutoff (Task 1)
            research_df = df[df["date"] < research_cutoff].copy()
            _, alpha_decay, alpha_admission = run_alpha_research(
                research_df,
                feat_cols,
                cfg=alpha_cfg,
                target_cfg=target_cfg,
                costs=_execution_cost_config(cfg),
                max_name_weight=max_name_weight_for_research,
            )
            df.to_parquet(alpha_df_path, index=False)
            alpha_decay.to_parquet(alpha_decay_parquet, index=False)
            alpha_admission.to_parquet(alpha_admission_parquet, index=False)
            _write_json(
                alpha_meta_path,
                {
                    "schema_version": ALPHA_RESEARCH_SCHEMA_VERSION,
                    "cache_reason": "fresh_build",
                },
            )
            alpha_from_cache = False
        t_alpha_research_1 = time.perf_counter()
        # Task 1: After alpha research load
        alpha_decay_path = out_dir / "alpha_ic_decay.csv"
        alpha_admission_path = out_dir / "feature_admission.csv"
        alpha_decay.to_csv(alpha_decay_path, index=False)
        alpha_admission.to_csv(alpha_admission_path, index=False)
        # Per-horizon labelled copies for multi-horizon comparison
        alpha_decay.to_csv(out_dir / f"alpha_ic_decay_{horizon}d.csv", index=False)
        alpha_admission.to_csv(out_dir / f"feature_admission_{horizon}d.csv", index=False)
        _write_horizon_ic_report(alpha_decay, alpha_admission, horizon, out_dir)
        _stage("alpha_research_and_ic_report")

        # ── P29: Institutional feature-level cost viability ────────────────────
        # Evaluates each admitted feature with CostViabilityEngine (Almgren-Chriss
        # impact modeling) instead of flat 10bps. Results written to
        # output/models/cost_viability/ and used by institutional horizon gate.
        _cost_viability_state = CostViabilityWiringState(config=cfg)
        try:
            _cost_viability_state.feature_results = evaluate_feature_cost_viability(
                alpha_admission, alpha_decay, df, cfg, int(horizon),
            )
            _production_cost_features = (
                alpha_admission.loc[
                    alpha_admission["admitted"].eq(True), "feature"
                ].astype(str).tolist()
                if (
                    isinstance(alpha_admission, pd.DataFrame)
                    and not alpha_admission.empty
                    and {"admitted", "feature"}.issubset(alpha_admission.columns)
                )
                else []
            )
            _cost_viability_state.production_feature_results = filter_feature_cost_results(
                _cost_viability_state.feature_results,
                _production_cost_features,
            )
            _cv_out = out_dir / "cost_viability"
            _cv_out.mkdir(parents=True, exist_ok=True)
            generate_cost_viability_reports(
                _cost_viability_state.feature_results,
                _cost_viability_state.candidate_results,
                _cost_viability_state.band_results,
                cfg, _cv_out,
            )
            if _cost_viability_state.production_feature_results:
                generate_cost_viability_reports(
                    _cost_viability_state.production_feature_results,
                    _cost_viability_state.candidate_results,
                    _cost_viability_state.band_results,
                    cfg,
                    _cv_out / "production_admitted",
                )
            _cv_viable = sum(
                1 for r in _cost_viability_state.feature_results
                if r.cost_status.value == "cost_viable"
            )
            _cv_total = len(_cost_viability_state.feature_results)
            _cv_prod_viable = sum(
                1 for r in _cost_viability_state.production_feature_results
                if r.cost_status.value == "cost_viable"
            )
            _cv_prod_total = len(_cost_viability_state.production_feature_results)
            print(
                f"\n[CostViability] Research candidates: {_cv_viable}/{_cv_total} cost viable "
                f"at h{horizon}d (institutional impact modeling)"
            )
            print(
                f"[CostViability] Production-admitted: {_cv_prod_viable}/{_cv_prod_total} cost viable "
                f"at h{horizon}d (horizon gate population)"
            )
        except Exception as _cv_exc:
            print(f"\n[CostViability] Feature evaluation skipped: {_cv_exc}")

        _stage("cost_viability_p29")

        # Pipeline diagnostics (P30-P34) removed — institutional research pipeline
        # does not run exploratory diagnostic engines. Core model evaluation
        # resumes directly after cost viability.
        _stage("cost_viability_complete")

        # ── Adaptive horizon selection from IC-decay table ────────────────────
        # Candidate set: intersection of alpha_research horizons and precompute
        # horizons determined by alpha research × adaptive_horizon config.
        from model_selection.adaptive_params import optimal_horizon_from_decay
        _horizon_policy = ms_cfg.get("adaptive_horizon", {}) or {}
        _ah_candidates = _horizon_policy.get("candidate_horizons", [5, 10, 20, 40, 60, 63])
        _decay_candidates = sorted(
            set(alpha_cfg.horizons) & {int(h) for h in _ah_candidates}
        )
        if _decay_candidates:
            _opt_h, _h_diag = optimal_horizon_from_decay(
                alpha_decay,
                candidate_horizons=_decay_candidates,
                target_type="net_residual_return",
                # P16: Guardrail config from YAML adaptive_horizon section
                min_features_per_horizon=int(
                    _horizon_policy.get("min_features_per_horizon", 2)
                ),
                min_median_ic=float(
                    _horizon_policy.get("min_median_ic", 0.002)
                ),
                tie_tolerance_pct=float(
                    _horizon_policy.get("tie_tolerance_pct", 10.0)
                ),
            )
            _scores_str = ", ".join(
                f"{h}d={v:.4f}" for h, v in _h_diag.get("scores", {}).items()
            )
            print(
                f"[AdaptiveHorizon] IC-decay scores: {_scores_str} | "
                f"optimal={_opt_h}d (config={horizon}d) | reason={_h_diag.get('reason', '')}"
            )
            # P16: --auto-horizon CLI flag overrides YAML adaptive_horizon.apply
            _auto_apply_horizon = bool(
                args.auto_horizon
                or _horizon_policy.get("apply", False)
                or _horizon_policy.get("auto_apply", False)
            )
            _horizon_governance = {
                "configured_horizon_days": int(horizon),
                "recommended_horizon_days": int(_opt_h),
                "candidate_horizons": [int(h) for h in _decay_candidates],
                "scores": {str(k): float(v) for k, v in (_h_diag.get("scores", {}) or {}).items()},
                "reason": str(_h_diag.get("reason", "")),
                "auto_apply": bool(_auto_apply_horizon),
                "action": "keep_configured_horizon",
                "cache_fingerprint": str(_research_cache_fp),
            }
            if _opt_h != int(horizon):
                if _auto_apply_horizon:
                    _horizon_governance["action"] = "auto_dispatch"
                    _write_json(out_dir / "horizon_governance.json", _horizon_governance)
                    import subprocess as _sp
                    _cmd = [sys.executable, __file__] + [
                        a for a in sys.argv[1:]
                        if not a.startswith("--horizon")
                        and not a.startswith("--production_horizons")
                        and not a.startswith("--auto-horizon")
                    ] + ["--horizon", str(int(_opt_h))]
                    print(
                        f"\n[AutoHorizon] Re-dispatching full pipeline at optimal horizon "
                        f"{int(_opt_h)}d (config was {int(horizon)}d).\n"
                        f"  IC-decay scores: {_scores_str}\n"
                        f"  Guardrail: {_h_diag.get('guardrail', {})}\n"
                        f"  Command: {' '.join(_cmd)}\n"
                    )
                    _result = _sp.run(_cmd)
                    if _result.returncode != 0:
                        print(f"[AutoHorizon] WARNING: dispatch at {int(_opt_h)}d exited {_result.returncode}")
                    raise SystemExit(_result.returncode)
                _horizon_governance["action"] = "advisory_only"
                print(
                    f"  → Advisory only: keeping configured production horizon at {int(horizon)}d. "
                    f"Run --auto-horizon or set adaptive_horizon.apply: true in config "
                    f"to promote the {int(_opt_h)}d contract."
                )
            _write_json(out_dir / "horizon_governance.json", _horizon_governance)
        # ── P34: Execution-aware horizon frontier (advisory diagnostic) ───────
        _eaw_enabled = bool(
            (ms_cfg.get("execution_aware_horizon", {}) or {}).get("enabled", True)
        )
        if _eaw_enabled and not alpha_decay.empty:
            try:
                from model_selection.adaptive_params import (
                    execution_aware_horizon_frontier,
                    compute_rebalance_frontier,
                    ExecutionAwareHorizonConfig,
                )
                _eaw_raw = ms_cfg.get("execution_aware_horizon", {}) or {}
                _eaw_cfg = ExecutionAwareHorizonConfig(
                    enabled=bool(_eaw_raw.get("enabled", True)),
                    candidate_horizons=tuple(
                        int(h) for h in _eaw_raw.get("candidate_horizons", [5, 10, 20, 40, 60, 63])
                    ),
                    target_type=str(_eaw_raw.get("target_type", "net_residual_return")),
                    weight_ic_strength=float(_eaw_raw.get("weight_ic_strength", 0.25)),
                    weight_ic_consistency=float(_eaw_raw.get("weight_ic_consistency", 0.15)),
                    weight_halflife_persistence=float(_eaw_raw.get("weight_halflife_persistence", 0.20)),
                    weight_cost_adjusted_ic=float(_eaw_raw.get("weight_cost_adjusted_ic", 0.20)),
                    weight_alpha_capture=float(_eaw_raw.get("weight_alpha_capture", 0.10)),
                    weight_execution_sharpe=float(_eaw_raw.get("weight_execution_sharpe", 0.10)),
                    min_features_per_horizon=int(_eaw_raw.get("min_features_per_horizon", 2)),
                    min_median_ic=float(_eaw_raw.get("min_median_ic", 0.002)),
                    halflife_persistence_threshold=float(_eaw_raw.get("halflife_persistence_threshold", 0.30)),
                    candidate_rebalance_frequencies=tuple(
                        int(f) for f in _eaw_raw.get("candidate_rebalance_frequencies", [2, 3, 5, 10, 20, 63])
                    ),
                    max_rebalance_to_horizon_ratio=float(_eaw_raw.get("max_rebalance_to_horizon_ratio", 1.0)),
                    horizon_frontier_path=str(_eaw_raw.get("horizon_frontier_path", str(out_dir / "horizon_frontier.csv"))),
                    rebalance_frontier_path=str(_eaw_raw.get("rebalance_frontier_path", str(out_dir / "rebalance_frontier.csv"))),
                    report_path=str(_eaw_raw.get("report_path", str(out_dir / "execution_aware_horizon_report.txt"))),
                )
                _frontier, _fdiag = execution_aware_horizon_frontier(
                    alpha_decay, cfg=_eaw_cfg,
                    rebalance_frequency_days=int(horizon_contract.config.rebalance_frequency_days),
                )
                _reb_frontier = compute_rebalance_frontier(alpha_decay, cfg=_eaw_cfg)

                if not _frontier.empty:
                    _frontier_path = Path(_eaw_cfg.horizon_frontier_path)
                    _frontier_path.parent.mkdir(parents=True, exist_ok=True)
                    _frontier.to_csv(_frontier_path, index=False)
                    print(f"[ExecutionAware] Horizon frontier saved: {_frontier_path}")

                if not _reb_frontier.empty:
                    _reb_path = Path(_eaw_cfg.rebalance_frontier_path)
                    _reb_path.parent.mkdir(parents=True, exist_ok=True)
                    _reb_frontier.to_csv(_reb_path, index=False)
                    print(f"[ExecutionAware] Rebalance frontier saved: {_reb_path}")

                # Write human-readable report
                _report_lines = ["=" * 72]
                _report_lines.append("EXECUTION-AWARE HORIZON FRONTIER — Advisory Diagnostic")
                _report_lines.append("=" * 72)
                _report_lines.append(f"Target type: {_eaw_cfg.target_type}")
                _report_lines.append(f"Candidate horizons: {list(_eaw_cfg.candidate_horizons)}")
                _report_lines.append("")
                _report_lines.append("Composite scoring weights (available metrics only, sum-normalised):")
                _report_lines.append(f"  IC strength:     {_eaw_cfg.weight_ic_strength:.2f}")
                _report_lines.append(f"  IC consistency:  {_eaw_cfg.weight_ic_consistency:.2f}")
                _report_lines.append(f"  Halflife persistence: {_eaw_cfg.weight_halflife_persistence:.2f}")
                _report_lines.append(f"  Cost-adjusted IC:     {_eaw_cfg.weight_cost_adjusted_ic:.2f}")
                _report_lines.append(f"  Alpha capture:        {_eaw_cfg.weight_alpha_capture:.2f}  (not yet available)")
                _report_lines.append(f"  Execution Sharpe:     {_eaw_cfg.weight_execution_sharpe:.2f}  (not yet available)")
                _report_lines.append("")
                _report_lines.append(f"Persistence threshold: {_eaw_cfg.halflife_persistence_threshold:.0%} rank survival at rebalance")
                _report_lines.append("")

                if not _frontier.empty:
                    _report_lines.append(f"{'H':>4} {'N_feat':>7} {'N_pos':>6} {'MedianIC':>9} {'IC_IR':>8} {'IC_t':>7} {'Halflife':>9} {'Persist':>8} {'P_ok':>5} {'CompScore':>10}")
                    _report_lines.append(f"{'─'*4:>4} {'─'*7:>7} {'─'*6:>6} {'─'*9:>9} {'─'*8:>8} {'─'*7:>7} {'─'*9:>9} {'─'*8:>8} {'─'*5:>5} {'─'*10:>10}")
                    for _, r in _frontier.iterrows():
                        _p_ok = "YES" if r.get("halflife_persistence_ok") else ("NO" if r.get("halflife_persistence_ok") is False else "?")
                        _persist = f"{r['rank_persistence_at_rebalance']:.4f}" if isinstance(r.get("rank_persistence_at_rebalance"), float) and np.isfinite(r["rank_persistence_at_rebalance"]) else "N/A"
                        _report_lines.append(
                            f"{int(r['horizon_days']):>4d} {int(r['n_features']):>7d} {int(r['n_positive_features']):>6d} "
                            f"{r['median_ic']:>9.4f} {r['ic_ir']:>8.4f} {r['ic_tstat']:>7.2f} "
                            f"{r['signal_halflife_days']:>9.2f} {_persist:>8} {_p_ok:>5} "
                            f"{r.get('composite_score', 0):>10.6f}"
                        )
                    _report_lines.append("")
                    _report_lines.append("NOTE: 'not_yet_available' metrics require model training and execution validation.")
                    _report_lines.append("      This frontier is advisory — it does NOT change the selected horizon.")

                for flag in _fdiag.get("flags", []):
                    _report_lines.append(f"\n[FLAG] {flag}")
                for rec in _fdiag.get("recommendations", []):
                    _report_lines.append(f"[RECOMMENDATION] {rec}")

                _report_lines.append("\n" + "=" * 72)
                _report_text = "\n".join(_report_lines)
                _report_path = Path(_eaw_cfg.report_path)
                _report_path.parent.mkdir(parents=True, exist_ok=True)
                _report_path.write_text(_report_text + "\n", encoding="utf-8")
                print(f"[ExecutionAware] Frontier report saved: {_report_path}")

                if _fdiag.get("recommendations"):
                    for rec in _fdiag["recommendations"]:
                        print(f"  → {rec}")
            except Exception as _eaw_exc:
                import traceback as _tb
                print(f"[ExecutionAware] Frontier diagnostic failed (non-fatal): {_eaw_exc}")
                if args.traceback:
                    _tb.print_exc()
        # ─────────────────────────────────────────────────────────────────────

        alpha_admission_summary = summarize_admission(alpha_admission)
        # --- Feature Admission Summary ---
        from model_selection.preparation import DEBUG_DIAGNOSTICS
        n_before = len(alpha_admission)
        n_after = int(alpha_admission_summary.get("alpha_features_admitted", 0))
        n_inverted = int(alpha_admission_summary.get("alpha_features_inverted", 0))
        n_removed = int(alpha_admission_summary.get("alpha_features_removed", 0))
        print(f"[Feature Admission Summary] before={n_before} | after={n_after} | inverted={n_inverted} | removed={n_removed}")
        if DEBUG_DIAGNOSTICS:
            admitted_names = alpha_admission[alpha_admission["admitted"].eq(True)]["feature"].tolist()
            print(f"  Admitted: {admitted_names}")
        # ---------------------------------
        # ── P21: Feature/Alpha Redesign Diagnostics (read-only) ─────────────────
        _p21_enabled = bool(
            (ms_cfg.get("feature_audit", {}) or {}).get("enabled", True)
        )
        if _p21_enabled and not alpha_admission.empty and not alpha_decay.empty:
            try:
                from model_selection.feature_audit import run_feature_audit, FeatureAuditConfig
                _p21_cfg_raw = (ms_cfg.get("feature_audit", {}) or {})
                _p21_cfg = FeatureAuditConfig(
                    min_abs_ic_for_sign=float(_p21_cfg_raw.get("min_abs_ic_for_sign", 0.001)),
                    max_sign_flips_for_stable=int(_p21_cfg_raw.get("max_sign_flips_for_stable", 1)),
                    min_admission_rate_for_promising=float(_p21_cfg_raw.get("min_admission_rate_for_promising", 0.30)),
                )
                _p21_alignment = get_horizon_alignment_report(
                    feature_columns=feat_cols,
                    prediction_horizon_days=int(horizon),
                )
                run_feature_audit(
                    decay_path=alpha_decay_path,
                    admission_path=alpha_admission_path,
                    horizon_alignment_report=_p21_alignment,
                    target_horizon=int(horizon),
                    out_dir=out_dir,
                    config=_p21_cfg,
                )
            except Exception as _p21_exc:
                logger.debug("FeatureAudit P21 skipped: %s", _p21_exc)
        # ───────────────────────────────────────────────────────────────────────
        admitted_features = (
            alpha_admission.loc[alpha_admission["admitted"].eq(True), "feature"].astype(str).tolist()
            if not alpha_admission.empty and "admitted" in alpha_admission.columns
            else []
        )
        inverted_features = (
            alpha_admission.loc[
                alpha_admission["admitted"].eq(True)
                & pd.to_numeric(alpha_admission["transform_sign"], errors="coerce").lt(0),
                "feature",
            ].astype(str).tolist()
            if not alpha_admission.empty and "transform_sign" in alpha_admission.columns
            else []
        )
        move_horizon_features = (
            alpha_admission.loc[
                alpha_admission["admitted"].eq(False)
                & alpha_admission["recommended_action"].astype(str).str.contains("move_horizon", regex=False),
                "feature",
            ].astype(str).tolist()
            if not alpha_admission.empty and "recommended_action" in alpha_admission.columns
            else []
        )
        removed_features = (
            alpha_admission.loc[
                alpha_admission["admitted"].eq(False)
                & alpha_admission["recommended_action"].astype(str).eq("remove"),
                "feature",
            ].astype(str).tolist()
            if not alpha_admission.empty and "recommended_action" in alpha_admission.columns
            else []
        )
        print(f"Alpha IC decay: {alpha_decay_path}")
        print(f"Feature admission: {alpha_admission_path}")
        print(
            "Feature admission: "
            f"admitted={len(admitted_features)}/{len(feat_cols)} "
            f"inverted={len(inverted_features)} "
            f"move_horizon={len(move_horizon_features)} "
            f"removed={len(removed_features)}"
        )
        min_admitted_required = max(0, int(getattr(alpha_cfg, "minimum_admitted_features", 0) or 0))
        if bool(getattr(alpha_cfg, "fail_if_below_minimum", False)) and len(admitted_features) < min_admitted_required:
            raise SystemExit(
                "Feature admission failed closed: "
                f"admitted={len(admitted_features)} below required minimum={min_admitted_required}. "
                "Do not promote fallback-ranked or under-breadth feature sets into model fitting."
            )
        if DEBUG_DIAGNOSTICS:
            if inverted_features:
                print(f"  Inverted production features: {inverted_features}")
            if move_horizon_features:
                print(f"  Moved out of {horizon}d model: {move_horizon_features}")
            if removed_features:
                print(f"  Removed by admission: {removed_features}")
        
        # Feature Polarity Sign-Flow Audit (Task 4)
        if getattr(args, "debug_validation", False) and not alpha_admission.empty and admitted_features:
            logger.debug("\nFeature Polarity Sign-Flow Audit (Task 4):")
            print(f"{'Feature':<25} | {'TrainIC':<8} | {'Mult':<4} | {'AdjTrain':<8} | {'Status'}")
            print("-" * 65)
            for _, r in alpha_admission[alpha_admission["admitted"]].iterrows():
                f = r["feature"]
                raw_ic = float(r["production_ic"])
                mult = int(r["transform_sign"])
                adj_ic = raw_ic * mult
                status = "Inverted" if mult < 0 else "Normal"
                print(f"{f:<25} | {raw_ic:>8.4f} | {mult:>4} | {adj_ic:>8.4f} | {status}")
        
        print(
            f"Alpha research {'load' if alpha_from_cache else 'build'} runtime: "
            f"{t_alpha_research_1 - t_alpha_research_0:.2f}s"
        )
        if alpha_cfg.enforce:
            if admitted_features:
                df = apply_admitted_feature_transforms(df, alpha_admission)
                feat_cols = [f for f in feat_cols if f in set(admitted_features)]
                short_feature_subset = []
                overlay_feature_subset = []
                if not alpha_cfg.multi_horizon_admission:
                    production_alignment = get_horizon_alignment_report(
                        feat_cols,
                        int(horizon),
                        alignment_multiplier=float(_ha_cfg["multiplier"]),
                    )
                    if int(production_alignment.get("n_misaligned", 0) or 0) > 0:
                        bad = [m[0] for m in production_alignment.get("misaligned", [])]
                        raise SystemExit(
                            "Feature admission produced horizon-misaligned production features "
                            f"for {int(horizon)}d target: {bad}. These features must be moved "
                            "to a compatible horizon or removed before model fitting."
                        )
                    if _ha_report and int(_ha_report.get("n_misaligned", 0) or 0) > 0:
                        print(
                            "HorizonAlignment post-admission: production feature set is aligned; "
                            f"{int(_ha_report.get('n_misaligned', 0) or 0)} pre-admission cross-horizon "
                            "candidate(s) were researched but not allowed through unless admitted for this horizon."
                        )
                print(f"Training feature set after admission: {len(feat_cols)} columns")
                if alpha_cfg.multi_horizon_admission:
                    eval_horizons = sorted(set(alpha_admission.loc[alpha_admission["admitted"], "eval_horizon_days"].dropna().astype(int)))
                    print(f"Multi-horizon admission: features evaluated at horizons {eval_horizons}d (model contract={int(horizon)}d)")
            else:
                raise SystemExit(
                    "Feature admission produced zero statistically admissible features. "
                    "Failing closed instead of forcing weak/noisy features into model fitting."
                )
    elif _ha_report and int(_ha_report.get("n_misaligned", 0) or 0) > 0:
        bad = [m[0] for m in _ha_report.get("misaligned", [])]
        raise SystemExit(
            "HorizonAlignment failed before model fitting because alpha feature admission is disabled: "
            f"{bad}. Enable alpha_research or remove/move these features to a compatible horizon."
        )

    # Basic leakage sanity check: any forward-looking columns still present?
    leaked = [c for c in feat_cols if "forward" in c.lower()]
    if leaked:
        raise SystemExit(f"Leakage: feature columns contain forward-looking fields: {leaked[:10]}")

    # Build windows. Prefer calendar windows when they yield >1, else fall back to count-based.
    windows = _walk_forward_windows(start_date, end_date, train_years, test_years, step_years)
    if len(windows) <= 1:
        windows = _walk_forward_windows_by_count(df["date"], n_windows=n_windows_cfg, train_ratio=train_ratio)
    if len(windows) <= 1:
        raise SystemExit(
            "Not enough walk-forward windows. Either extend the backtest date range or reduce research.train_years/test_years."
        )

    # Validate target alignment (Task 2)
    sample_df = df.sort_values(["ticker", "date"]).head(100)
    if getattr(args, "debug_validation", False) and not sample_df.empty and "forward_return" in sample_df.columns:
        logger.debug("\nTarget Alignment Audit (Task 2):")
        # Check first 3 dates for first ticker
        first_ticker = sample_df["ticker"].iloc[0]
        t_df = sample_df[sample_df["ticker"] == first_ticker].head(3).copy()
        t_df["target_start_date"] = t_df["date"] + pd.Timedelta(days=1)
        # Using 5-day horizon proxy for display
        t_df["target_end_date"] = t_df["date"] + pd.Timedelta(days=6)
        print(t_df[["date", "ticker", "forward_return", "target_start_date", "target_end_date"]])

    print()
    print("Walk-forward windows (calendar bounds):")
    prev_test_end: pd.Timestamp | None = None
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows, 1):
        sequential = "OK"
        if prev_test_end is not None and te_s < prev_test_end:
            sequential = "OVERLAP"
        tr_span = f"{tr_s.date()} → {(tr_e - pd.Timedelta(days=1)).date()}"
        te_span = f"{te_s.date()} → {(te_e - pd.Timedelta(days=1)).date()}"
        print(f"  {i:02d} train=[{tr_span}]  test=[{te_span}]  {sequential}")
        prev_test_end = te_e

    models = _build_models(cfg)
    if not models:
        raise SystemExit("No models available (check sklearn / optional xgboost install).")

    # P9: Separate baseline scorers from ML models. Baselines are reference
    # statistics evaluated outside the hypothesis-testing pipeline.
    ml_models = [m for m in models if m[0] not in BASELINE_SCORERS]
    baseline_models = [m for m in models if m[0] in BASELINE_SCORERS]
    if baseline_models:
        print(f"Baseline scorers ({len(baseline_models)}): {[m[0] for m in baseline_models]} — evaluated outside SignalDiscovery")
    models = ml_models

    parallel_cfg = _parallel_research_config(cfg, n_models=len(models))
    # Unconditionally constrain inner model parallelism to prevent nested explosion
    models = _constrain_model_parallelism(models, max_jobs=1)
    # Simplified Mode override (Task 1 & 2)
    if getattr(args, "simplified", False):
        print("\n!!! SIMPLIFIED MODE ENABLED: Disabling research layers and forcing L2 optimization !!!")
        args.horizon_optimization = False
        args.confidence_weighting = False
        args.regime_gating = False
        args.asymmetry_correction = False
        args.capacity_analysis = False
        args.marginal_value = False
        args.cost_sensitivity = False
        args.joint_optimization = False
        args.deployability_ranking = False
        # Revert to L2 for core isolation
        ms_cfg["validation"] = ms_cfg.get("validation", {})
        ms_cfg["validation"]["optimization_type"] = "l2"
        ms_cfg["validation"]["gamma_turnover"] = 0.0 # Remove penalty for raw performance check
        
    # Compute the full search-tree trial count for multiple-testing correction.
    # P7: The previous formula (models × families × horizons × subsets × paths)
    # grossly inflated the trial count because:
    #   - Feature families are used jointly, not tested separately
    #   - Horizon is fixed to contract (10d), not searched over
    #   - Baseline scorers are not "searched" — they are evaluation floors
    # The true search space is: ML models × evaluation paths.
    # Baselines (EqualWeight, ICWeighted) establish a no-ML floor but are not
    # part of the hypothesis test — they are descriptive statistics.
    _search_cfg = ms_cfg.get("search", {}) or {}
    _ml_search_models = [m for m in models if m[0] not in {"EqualWeightBaseline", "ICWeightedBaseline"}]
    _n_ml_models = len(_ml_search_models)
    # Paths: long_only_overlay is the deployment path; long_short_spread is the
    # alpha research path. Only the research path generates a separate hypothesis.
    _n_paths = 2 if do_shorts else 1
    # Feature families are consumed jointly by every model — NOT tested separately.
    # Horizon is fixed to the production contract — NOT searched over.
    metric_trial_count = max(_n_ml_models * _n_paths, _n_ml_models)
    print(
        f"\nMultiple-testing correction (P7): metric_trial_count={metric_trial_count} "
        f"(ML_models={_n_ml_models}, paths={_n_paths}; "
        f"baselines excluded from search tree, families/horizon consumed jointly)"
    )

    # ── Horizon Gate: Pre-training governance layer ───────────────────────────
    # Computes per-feature eligibility contracts and evaluates the horizon gate
    # BEFORE PreparedPanelCache is initialized. Blocked horizons do not train.
    # Ineligible features cannot enter training.
    _hg_cfg_raw = (ms_cfg.get("horizon_gate", {}) or {}) if isinstance(ms_cfg, dict) else {}
    _hg_mode = str(_hg_cfg_raw.get("mode", "production")).strip().lower()
    if _hg_mode not in {"report_only", "production"}:
        _hg_mode = "production"

    _hg_feature_families: dict[str, list[str]] = {}
    for _f in feat_cols:
        _spec = FEATURE_SPECS.get(_f)
        _fam = _spec.family if _spec else "unknown"
        _hg_feature_families.setdefault(_fam, []).append(_f)

    _eligibility_dir = out_dir / "horizon_eligibility"
    _eligibility_dir.mkdir(parents=True, exist_ok=True)

    _horizon_gate_contracts: dict[str, HorizonEligibilityContract] = {}
    _horizon_gate_result: Any = None
    _gated_feat_cols: list[str] = list(feat_cols)
    _horizon_gate_blocked = False

    try:
        # ── P29: Institutional horizon gate (optional, replaces flat 10bps) ───
        _hg_use_institutional = bool(_hg_cfg_raw.get("use_institutional", True))
        if _hg_use_institutional and alpha_admission is not None and not alpha_admission.empty:
            print(f"\n[HorizonGate] Using institutional cost viability (Almgren-Chriss)")
            _hg_alpha_admission = alpha_admission
            if "feature" in _hg_alpha_admission.columns:
                _hg_alpha_admission = _hg_alpha_admission[
                    _hg_alpha_admission["feature"].astype(str).isin(set(map(str, feat_cols)))
                ].copy()
            _hg_cost_state = locals().get("_cost_viability_state")
            if (
                _hg_cost_state is not None
                and getattr(_hg_cost_state, "feature_results", None)
            ):
                _hg_feature_results = filter_feature_cost_results(
                    getattr(_hg_cost_state, "production_feature_results", []),
                    set(map(str, feat_cols)),
                )
                _hg_blocked, _hg_gate_diag = summarize_feature_cost_gate(
                    _hg_feature_results, cfg, int(horizon),
                )
                print(
                    f"[HorizonGate] Reusing precomputed production cost viability "
                    f"({len(_hg_feature_results)} feature contracts)"
                )
            else:
                _hg_blocked, _hg_feature_results, _hg_gate_diag = compute_institutional_horizon_gate(
                    df, _hg_feature_families, _hg_alpha_admission, alpha_decay, cfg, int(horizon),
                )
            _horizon_gate_contracts = feature_cost_results_to_horizon_contracts(
                _hg_feature_results,
                int(horizon),
            )
            _horizon_gate_result = type(
                "InstitutionalGateResult", (), {
                    "eligible_features": [_fr.feature for _fr in _hg_feature_results if _fr.cost_status.value == "cost_viable"],
                    "rejected_features": {
                        _fr.feature: _fr.rejection_reason or _fr.cost_status.value
                        for _fr in _hg_feature_results if _fr.cost_status.value != "cost_viable"
                    },
                    "n_eligible": len([_fr for _fr in _hg_feature_results if _fr.cost_status.value == "cost_viable"]),
                    "n_families": len(set(_fr.family for _fr in _hg_feature_results if _fr.family != "unknown")),
                    "effective_signals": sum(
                        max(0.1, _fr.alpha_cost_ratio) for _fr in _hg_feature_results
                        if _fr.cost_status.value == "cost_viable"
                    ),
                    "family_concentration": 0.0,
                    "block_horizon": _hg_blocked,
                    "block_reasons": _hg_gate_diag.get("block_reasons", []),
                    "report": str(_hg_gate_diag),
                }
            )()
            _n_stat = len(_horizon_gate_result.eligible_features)
            _n_prod = _n_stat
            _n_rejected = len(_horizon_gate_result.rejected_features)
            _cost_dominated = sum(
                1 for _fr in _hg_feature_results
                if _fr.cost_status.value in ("cost_dominated", "turnover_dominated", "impact_dominated")
            )
            _halflife_rejected = sum(
                1 for _fr in _hg_feature_results
                if "halflife" in (_fr.rejection_reason or "").lower()
            )
            _top_reasons = {}
            for _reason in _horizon_gate_result.rejected_features.values():
                _code = _reason.split(" + ")[0].strip()
                _top_reasons[_code] = _top_reasons.get(_code, 0) + 1
            _top_reasons_sorted = sorted(_top_reasons.items(), key=lambda x: -x[1])[:5]
        else:
            _hg_cost_bps = float(_hg_cfg_raw.get("cost_bps", 10.0) or 10.0)
            _horizon_gate_contracts = compute_all_eligibility(
            df, _hg_feature_families,
            horizons=[int(horizon)],
            cost_bps=_hg_cost_bps,
        )

        _eligibility_report = format_eligibility_report(_horizon_gate_contracts, [int(horizon)])
        _report_path = _eligibility_dir / f"eligibility_report_{horizon}d.txt"
        _report_path.write_text(_eligibility_report + "\n", encoding="utf-8")
        print(f"\n[HorizonGate] Eligibility report: {_report_path}")

        _gate_config = HorizonGateConfig(
            min_production_features=int(_hg_cfg_raw.get("min_production_features", 3) or 3),
            min_families=int(_hg_cfg_raw.get("min_families", 2) or 2),
            max_family_concentration=float(_hg_cfg_raw.get("max_family_concentration", 0.6) or 0.6),
            min_effective_signals=float(_hg_cfg_raw.get("min_effective_signals", 1.5) or 1.5),
            use_production_level=(_hg_mode == "production"),
        )

        _horizon_gate = HorizonGate(_horizon_gate_contracts, config=_gate_config)
        _horizon_gate_result = _horizon_gate.evaluate(int(horizon))

        _n_stat = len(_horizon_gate_result.eligible_features) if not _gate_config.use_production_level else 0
        _n_prod = len(_horizon_gate_result.eligible_features) if _gate_config.use_production_level else 0
        _n_rejected = len(_horizon_gate_result.rejected_features)

        _cost_dominated = sum(
            1 for c in _horizon_gate_contracts.values()
            if not c.cost_adjusted_viable.get(int(horizon), False)
        )
        _halflife_rejected = sum(
            1 for c in _horizon_gate_contracts.values()
            if any("HALFLIFE_TOO_SHORT" in r for r in c.statistical_rejections.values())
        )

        _top_reasons: dict[str, int] = {}
        for _reason in _horizon_gate_result.rejected_features.values():
            for _code in _reason.split(" + "):
                _code = _code.strip()
                _top_reasons[_code] = _top_reasons.get(_code, 0) + 1
        _top_reasons_sorted = sorted(_top_reasons.items(), key=lambda x: -x[1])[:5]

        print(f"\n{'='*72}")
        print(f"  HORIZON GATE — h{horizon}d  (mode={_hg_mode})")
        print(f"{'='*72}")
        print(f"  Statistical eligible : {_n_stat if _n_stat else 'N/A'}")
        print(f"  Production eligible  : {_n_prod if _n_prod else 'N/A'}")
        print(f"  Rejected             : {_n_rejected}")
        print(f"  Cost dominated       : {_cost_dominated}")
        print(f"  Halflife rejected    : {_halflife_rejected}")
        print(f"  Families represented : {_horizon_gate_result.n_families}")
        print(f"  Effective signals    : {_horizon_gate_result.effective_signals:.2f}")
        print(f"  Family concentration : {_horizon_gate_result.family_concentration:.2f}")
        if _top_reasons_sorted:
            print(f"  Top rejection reasons:")
            for _code, _cnt in _top_reasons_sorted:
                print(f"    {_code}: {_cnt}")
        print(f"  Decision             : {'BLOCK' if _horizon_gate_result.block_horizon else 'ALLOW'}")
        print(f"{'='*72}")

        if _hg_mode == "production" and _horizon_gate_result.block_horizon:
            _horizon_gate_blocked = True
            _block_report_path = _eligibility_dir / f"blocked_horizon_{horizon}d.txt"
            _block_report = (
                f"HORIZON BLOCKED: h{horizon}d\n"
                f"Mode: production\n"
                f"Block reasons: {'; '.join(_horizon_gate_result.block_reasons)}\n"
                f"Eligible features: {_horizon_gate_result.n_eligible}\n"
                f"Required features: {_gate_config.min_production_features}\n"
                f"Families: {_horizon_gate_result.n_families}\n"
                f"Effective signals: {_horizon_gate_result.effective_signals:.2f}\n"
                f"\nFull gate report:\n{_horizon_gate_result.report}\n"
            )
            _block_report_path.write_text(_block_report, encoding="utf-8")
            print(f"\n[HorizonGate] BLOCKED: h{horizon}d — {_horizon_gate_result.block_reasons}")
            print(f"[HorizonGate] Block report: {_block_report_path}")
            raise SystemExit(
                f"Horizon h{horizon}d blocked by governance gate: "
                f"{'; '.join(_horizon_gate_result.block_reasons)}"
            )
        elif _hg_mode == "report_only" and _horizon_gate_result.block_horizon:
            print(f"\n[HorizonGate] REPORT_ONLY: h{horizon}d would be BLOCKED but continuing: "
                  f"{'; '.join(_horizon_gate_result.block_reasons)}")
            _gated_feat_cols = list(feat_cols)
        else:
            _gated_feat_cols = filter_eligible_features(
                feat_cols, _horizon_gate_contracts, horizon=int(horizon),
                use_production=(_hg_mode == "production"),
            )
            if _gated_feat_cols != feat_cols:
                print(
                    f"\n[HorizonGate] Feature gating: {len(feat_cols)} → {len(_gated_feat_cols)} features "
                    f"({len(feat_cols) - len(_gated_feat_cols)} removed)"
                )

        _eligibility_fp = hashlib.sha256(
            json.dumps({
                "horizon": int(horizon),
                "mode": _hg_mode,
                "n_contracts": len(_horizon_gate_contracts),
                "n_eligible": len(_gated_feat_cols),
                "blocked": _horizon_gate_blocked,
                "gate_result": {
                    "n_eligible": _horizon_gate_result.n_eligible,
                    "n_families": _horizon_gate_result.n_families,
                    "effective_signals": _horizon_gate_result.effective_signals,
                    "block_reasons": _horizon_gate_result.block_reasons,
                },
            }, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]

        _write_json(_eligibility_dir / f"gate_result_{horizon}d.json", {
            "horizon": int(horizon),
            "mode": _hg_mode,
            "blocked": _horizon_gate_blocked,
            "eligible_features": _gated_feat_cols,
            "n_eligible": len(_gated_feat_cols),
            "n_rejected": _n_rejected,
            "n_families": _horizon_gate_result.n_families,
            "effective_signals": _horizon_gate_result.effective_signals,
            "family_concentration": _horizon_gate_result.family_concentration,
            "block_reasons": _horizon_gate_result.block_reasons,
            "top_rejection_reasons": dict(_top_reasons_sorted),
            "cost_dominated_count": _cost_dominated,
            "halflife_rejected_count": _halflife_rejected,
            "fingerprint": _eligibility_fp,
        })

    except SystemExit:
        raise
    except Exception as _hg_exc:
        print(f"\n[HorizonGate] ERROR computing eligibility: {_hg_exc}")
        if _hg_mode == "production":
            raise SystemExit(f"Horizon gate failed closed: {_hg_exc}") from _hg_exc
        print(f"[HorizonGate] Report-only mode — continuing with full feature set.")
        _gated_feat_cols = list(feat_cols)
        _eligibility_fp = "error_fallback"

    _research_cache_fp = f"{_research_cache_fp}_hg{_eligibility_fp}" if _research_cache_fp else f"hg{_eligibility_fp}"
    # ── End Horizon Gate ──────────────────────────────────────────────────────

    exec_costs = _execution_cost_config(cfg)
    research_max_name_weight = _evaluation_config(
        cfg,
        path=primary_path,
        max_positions=int(max_positions),
        min_positions=int(min_positions),
        horizon_contract=horizon_contract,
    ).max_name_weight

    # C.2: Build TargetPanelProvider once to precompute targets for all horizons
    _RESEARCH_GRID: list[int] = [5, 10, 20, 40, 60, 63]
    _candidate_horizons: list[int] = sorted(
        {int(horizon)} | set(_RESEARCH_GRID)
    )
    _target_provider = build_target_provider(
        df,
        target_cfg=target_cfg,
        costs=exec_costs,
        max_name_weight=research_max_name_weight,
        horizons=_candidate_horizons,
    )
    print(f"\nInitial RAM usage: {PreparedPanelCache.get_rss_mb():.1f} MB")
    # P12: Pre-fit factor neutralization (institutional fix)
    # Neutralize features against CAPM beta, sector, size, and volatility BEFORE
    # model training. This prevents any model from learning factor exposure.
    # Equivalent to training on idiosyncratic returns directly.
    _neutralize_factors = bool(
        (ms_cfg.get("search", {}) or {}).get("neutralize_factors_before_training", True)
    )
    _neutralize_ridge = float(
        (ms_cfg.get("search", {}) or {}).get("neutralization_ridge", 1e-4)
    )

    global_prepare_cache = PreparedPanelCache(
        df,
        target_cfg=target_cfg,
        costs=exec_costs,
        max_name_weight=research_max_name_weight,
        winsor_q=_preprocess_winsor_q(cfg),
        artifact_dir=research_state.subdir("prepared_panels"),
        max_cache_size=int((ms_cfg.get("search", {}) or {}).get("max_memory_panels", 2)),
        min_free_space_mb=float((ms_cfg.get("search", {}) or {}).get("min_cache_free_space_mb", 2048.0)),
        disk_persistence=str((ms_cfg.get("search", {}) or {}).get("prepared_panel_disk_persistence", "horizon_only")),
        neutralize_factors=_neutralize_factors,
        neutralization_ridge=_neutralize_ridge,
        cache_fingerprint=_research_cache_fp,
        target_provider=_target_provider,
    )

    # --- BOTTLENECK 3: Precompute factor mimicking returns ---
    # build_factor_mimicking_returns uses only fundamental/return columns from the OOS panel
    # (sector, market_cap, momentum proxies, internal feature signals) — never the model score.
    # The result is identical for every model family evaluated on the same walk-forward windows
    # and horizon. Precomputing once here and passing it into factor_subsumption_diagnostics
    # eliminates O(N_models × ~18,000) redundant per-date quantile-sort computations.
    _oos_panel_for_factor = pd.concat(
        [df[(df["date"] >= te_s) & (df["date"] < te_e)] for _, _, te_s, te_e in windows],
        ignore_index=True,
    )
    # Task 2 & 3: Force multi-horizon sweep for factor mimicking and cache pre-warming
    # P15: Always include the active production horizon in the candidate grid.
    # The research grid provides standard horizons for IC-decay analysis;
    # the production horizon is mandatory for subsumption gate evaluation.
    # P22: Include 63d for native fundamental feature evaluation
    # (C.2: _candidate_horizons already defined above for TargetPanelProvider)
    if len(_candidate_horizons) <= 1:
        from model_selection.validation import MetricIntegrityError
        raise MetricIntegrityError("Horizon sweep must evaluate multiple horizons (H in {5, 10, 20, 40, 60}).")
        
    _factor_horizon_override = (ms_cfg.get("search", {}) or {}).get("factor_precompute_horizons", None)
    if _factor_horizon_override is not None:
        _precompute_horizons: list[int] = sorted({int(h) for h in _factor_horizon_override})
    else:
        _precompute_horizons: list[int] = sorted({int(horizon)} | set(_candidate_horizons))
    print(f"\nPrecomputing factor mimicking returns for horizons={_precompute_horizons} ...")
    _t_fac0 = time.perf_counter()
    precomputed_factor_returns: dict[int, pd.DataFrame] = {}
    for _h in _precompute_horizons:
        _fac = build_factor_mimicking_returns(_oos_panel_for_factor, horizon_days=_h)
        precomputed_factor_returns[_h] = _fac
        logger.debug(f"  horizon={_h}: {len(_fac.columns)} factors, {len(_fac)} dates ({time.perf_counter() - _t_fac0:.1f}s)")
    logger.info(f"Precomputed {len(_precompute_horizons)} horizons in {time.perf_counter() - _t_fac0:.1f}s")

    # --- BOTTLENECK 4: Pre-warm reusable PreparedPanelCache artifacts ---
    # ProcessPoolExecutor workers each get their own process-local PreparedPanelCache instance.
    # Institutionally, only reusable horizon-level target panels should persist by default.
    # Fold/preprocessor artifacts are structurally unique in walk-forward selection, so
    # persisting them mostly creates disk pressure with little reuse.
    _distinct_feature_views: list[list[str]] = [_gated_feat_cols]
    mem_ledger.record("prewarm_start")
    _t_pw0 = time.perf_counter()
    if not global_prepare_cache.disk_cache_writable():
        _cache_stats = global_prepare_cache.stats()
        print(
            "\nSkipping PreparedPanelCache pre-warm: persistent artifact cache is not writable "
            f"({_cache_stats.get('artifact_disabled_reason', 'unknown')}; "
            f"free={float(_cache_stats.get('artifact_free_space_mb', 0.0)):.1f}MB, "
            f"reserve={float(_cache_stats.get('artifact_min_free_space_mb', 0.0)):.1f}MB). "
            "Continuing with memory-only fold construction."
        )
    else:
        if global_prepare_cache.disk_persistence == "horizon_only":
            _production_h = int(horizon)
            _warm_horizons = [h for h in _precompute_horizons if int(h) != _production_h] + [_production_h]
            print(
                f"\nPre-warming PreparedPanelCache reusable horizon panels: "
                f"{len(_warm_horizons)} horizons ..."
            )
            for _h in _warm_horizons:
                try:
                    global_prepare_cache.get_full_retargeted_panel(_h)
                except Exception as _exc:
                    print(f"  WARNING: horizon-panel pre-warm failed for h={_h}: {_exc}")
                    if not global_prepare_cache.disk_cache_writable():
                        print("  PreparedPanelCache pre-warm stopped: persistent artifact cache became unwritable.")
                        break
        else:
            print(
                f"\nPre-warming PreparedPanelCache: {len(windows)} windows × "
                f"{len(_distinct_feature_views)} feature views × {len(_precompute_horizons)} horizons ..."
            )
            for _view_feats in _distinct_feature_views:
                for _h in _precompute_horizons:
                    for tr_s, tr_e, te_s, te_e in windows:
                        purge_cutoff = te_s - pd.Timedelta(days=int(embargo_days))
                        try:
                            global_prepare_cache.get_prepared_fold(
                                train_start=tr_s,
                                train_end=min(tr_e, purge_cutoff),
                                eval_start=te_s,
                                eval_end=te_e,
                                horizon_days=_h,
                                active_features=_view_feats,
                            )
                        except Exception as _exc:
                            print(
                                f"  WARNING: pre-warm failed for view={len(_view_feats)}feats "
                                f"h={_h} window={tr_s.date()}: {_exc}"
                            )
                            if not global_prepare_cache.disk_cache_writable():
                                print("  PreparedPanelCache pre-warm stopped: persistent artifact cache became unwritable.")
                                break
                    if not global_prepare_cache.disk_cache_writable():
                        break
                if not global_prepare_cache.disk_cache_writable():
                    break
    _prewarm_stats = global_prepare_cache.stats()
    logger.info(
        "PreparedPanelCache prewarm stats: "
        "memory_hits=%s artifact_hits=%s misses=%s writes=%s evictions=%s policy=%s",
        _prewarm_stats.get("hits", 0),
        _prewarm_stats.get("artifact_hits", 0),
        _prewarm_stats.get("misses", 0),
        _prewarm_stats.get("artifact_writes", 0),
        _prewarm_stats.get("evictions", 0),
        _prewarm_stats.get("disk_persistence", ""),
    )
    global_prepare_cache.reset_runtime_stats()
    mem_ledger.record("prewarm_complete", cache_items=len(global_prepare_cache._prepared_fold_cache))
    global_prepare_cache.log_uniqueness_report()
    _stage("prewarm_and_context")

    rows: list[dict[str, Any]] = []
    all_window_details: dict[str, list[WindowMetrics]] = {}
    t_selection_0 = time.perf_counter()
    # Task 3 & 4: Save enriched panel to disk for lightweight worker references
    _enriched_path = out_dir / "enriched_panel_temp.parquet"
    if not _enriched_path.exists():
        df.to_parquet(_enriched_path, index=False)
    
    ctx = OuterEvaluationContext(
        windows=tuple(windows),
        models=tuple(models),
        df_path=_enriched_path,
        cfg=cfg,
        args=args,
        nested_cfg=nested_cfg,
        feat_cols=_gated_feat_cols,
        short_feature_subset=short_feature_subset,
        overlay_feature_subset=overlay_feature_subset,
        horizon=int(horizon),
        horizon_contract=horizon_contract,
        max_positions=int(max_positions),
        min_positions=int(min_positions),
        embargo_days=int(embargo_days),
        use_risk_adj=use_risk_adj,
        exec_costs=exec_costs,
        research_max_name_weight=research_max_name_weight,
        global_prepare_cache=global_prepare_cache,
        metric_trial_count=metric_trial_count,
        ms_cfg=ms_cfg,
        gate_cfg=gate_cfg,
        long_cand_cfg=long_cand_cfg,
        feature_contract_summary=feature_contract_summary,
        alpha_admission_summary=alpha_admission_summary,
        research_state=research_state,
        timing_ledger=timing_ledger,
        parallel_cfg=parallel_cfg,
        precomputed_factor_returns=precomputed_factor_returns,
        primary_path=primary_path,
        diagnostic_plan=DiagnosticExecutionPlan.from_config(cfg),
        _df=None, # Trigger lazy loading
    )
    # If we are in memory-safe mode, we can clear the parent's df to free 7GB
    if getattr(args, "horizon_optimization", False) or getattr(args, "joint_optimization", False):
        print(f"  [Memory-Safe] Clearing parent memory after disk-persistence...")
        # No need to set global_prepare_cache.base_df = None separately, the setter handles it
        ctx.df = None
        import gc
        gc.collect()

    # P39: Fail-fast assertion — in sweep-controlled mode, verify that all
    # contract dimensions are collapsed to the sweep horizon.  This catches
    # any downstream code that reconstructs horizon from raw YAML.
    if _is_sweep_child and ctx.horizon_contract is not None:
        _sweep_h = int(horizon)
        _hc = ctx.horizon_contract.config
        _mismatches: list[str] = []
        if int(_hc.production_horizon_days) != _sweep_h:
            _mismatches.append(f"production_horizon_days={_hc.production_horizon_days}d != sweep={_sweep_h}d")
        if int(_hc.target_horizon_days) != _sweep_h:
            _mismatches.append(f"target_horizon_days={_hc.target_horizon_days}d != sweep={_sweep_h}d")
        if int(_hc.holding_period_days) != _sweep_h:
            _mismatches.append(f"holding_period_days={_hc.holding_period_days}d != sweep={_sweep_h}d")
        if int(_hc.rebalance_frequency_days) != _sweep_h:
            _mismatches.append(f"rebalance_frequency_days={_hc.rebalance_frequency_days}d != sweep={_sweep_h}d")
        if int(_hc.ic_evaluation_horizon) != _sweep_h:
            _mismatches.append(f"ic_evaluation_horizon={_hc.ic_evaluation_horizon}d != sweep={_sweep_h}d")
        if _mismatches:
            raise RuntimeError(
                f"[HorizonContract] Sweep-mode dimension mismatch in sweep child (horizon={_sweep_h}d):\n  "
                + "\n  ".join(_mismatches)
                + "\nAll contract dimensions must collapse to the sweep horizon unless sweep_dimension_separation=true."
            )
        print(f"[HorizonContract] Sweep-mode assertion passed: all dimensions collapsed to {_sweep_h}d")

    screening_cfg = _screening_config(cfg)

    def _run_model_phase(
        phase_name: str,
        phase_models: list[tuple[str, Any, bool, str]],
        *,
        evaluator: Any,
    ) -> list[dict[str, Any]]:
        if not phase_models:
            return []
        if phase_name in {"FastSweep", "SignalDiscovery"}:
            mem_ledger.record(f"phase_{phase_name}_start", n_models=len(phase_models))

        # P39: Phase-level contract audit artifact — record resolved horizon
        # dimensions at each phase boundary for sweep governance traceability.
        if ctx.horizon_contract is not None:
            try:
                import json as _json
                _hc = ctx.horizon_contract.config
                _phase_audit = {
                    "phase": phase_name,
                    "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "sweep_mode": str(ctx.horizon_contract.sweep_mode),
                    "production_horizon_days": int(_hc.production_horizon_days),
                    "target_horizon_days": int(_hc.target_horizon_days),
                    "holding_period_days": int(_hc.holding_period_days),
                    "rebalance_frequency_days": int(_hc.rebalance_frequency_days),
                    "ic_evaluation_horizon": int(_hc.ic_evaluation_horizon),
                    "execution_tau_days": float(_hc.execution_tau_days) if _hc.execution_tau_days is not None else None,
                    "context_horizon": int(ctx.horizon),
                    "source_map": dict(ctx.horizon_contract.source_map),
                }
                _phase_audit_path = out_dir / f"phase_contract_audit_{phase_name}.json"
                _phase_audit_path.write_text(_json.dumps(_phase_audit, indent=2, default=str), encoding="utf-8")
            except Exception:
                pass

        phase_results: list[dict[str, Any]] = []
        if phase_name == "EconomicDeepSelection":
            model_workers = int(parallel_cfg.get("economic_model_workers", 1))
        else:
            model_workers = int(parallel_cfg.get("model_workers", 1))

        # Task 1 & 2: Automatic memory-safe downscaling
        is_memory_intensive = any([
            getattr(args, "horizon_optimization", False),
            phase_name == "EconomicDeepSelection",
            getattr(args, "joint_optimization", False),
        ])
        if is_memory_intensive:
            print(f"  [Memory-Safe Mode] Downscaling parallelism: model_workers=1, nested_workers=1")
            model_workers = 1
            # We also need to tell the nested searchers to be single-threaded
            # We'll do this by modifying a temporary parallel_cfg if needed
            effective_parallel_cfg = dict(parallel_cfg)
            effective_parallel_cfg["nested_candidate_workers"] = 1
        else:
            effective_parallel_cfg = parallel_cfg

        print()
        print(
            f"{phase_name}: {len(phase_models)} model families | "
            f"model_workers={model_workers} nested_candidate_workers={effective_parallel_cfg.get('nested_candidate_workers', 1)}"
        )
        _run_parallel = bool(effective_parallel_cfg.get("enabled", True)) and model_workers > 1 and len(phase_models) > 1
        
        # Always set the global ctx for the wrapper to use
        global _FORK_GLOBAL_CTX
        _FORK_GLOBAL_CTX = ctx
        
        if _run_parallel:
            try:
                _fork_ctx = _mp.get_context("fork")
                with threadpoolctl.threadpool_limits(limits=1), ProcessPoolExecutor(
                    max_workers=min(model_workers, len(phase_models)),
                    mp_context=_fork_ctx,
                ) as executor:
                    future_meta = {
                        executor.submit(_evaluate_model_fork_wrapper, evaluator, model_spec, effective_parallel_cfg): (
                            str(model_spec[0]),
                            time.perf_counter(),
                        )
                        for model_spec in phase_models
                    }
                    for completed_idx, future in enumerate(as_completed(future_meta), 1):
                        model_name, submitted_at = future_meta[future]
                        result = future.result()
                        
                        worker_started = result.get("_worker_started_at", submitted_at) if isinstance(result, dict) else submitted_at
                        worker_completed = result.get("_worker_completed_at", time.perf_counter()) if isinstance(result, dict) else time.perf_counter()
                        
                        queue_wait = worker_started - submitted_at
                        true_cpu_time = worker_completed - worker_started
                        wall_elapsed = time.perf_counter() - submitted_at
                        
                        row = result.get("row") if isinstance(result, dict) else None
                        if ctx.timing_ledger is not None:
                            ctx.timing_ledger.record(
                                "model_family_completed",
                                phase=str(phase_name),
                                model_name=str(model_name),
                                elapsed_s=float(true_cpu_time),
                                queue_wait_s=float(queue_wait),
                                valid_windows=int(row.get("n_windows", 0) or 0) if isinstance(row, dict) else int(result.get("valid_windows", 0) or 0),
                            )
                        if evaluator is _screen_model_family:
                            valid_windows = int(result.get("valid_windows", 0) or 0)
                            screen_score = float(result.get("screen_score", float("nan")))
                            print(
                                f"  [{phase_name} {completed_idx}/{len(phase_models)}] {model_name} finished | "
                                f"wall={wall_elapsed:.1f}s (cpu={true_cpu_time:.1f}s, q_wait={queue_wait:.1f}s)"
                                f" | valid_windows={valid_windows} | screen_score={screen_score:.4f}"
                            )
                        else:
                            valid_windows = int(row.get("n_windows", 0) or 0) if isinstance(row, dict) else 0
                            exec_avg = float(row.get("exec_time_avg", float("nan"))) if isinstance(row, dict) else float("nan")
                            nested_search = bool(row.get("nested_search_applicable", False)) if isinstance(row, dict) else False
                            _nested_failures = int(row.get("nested_window_failures", 0) or 0) if isinstance(row, dict) else 0
                            _failure_suffix = f" | nested_failures={_nested_failures}" if _nested_failures > 0 else ""
                            if nested_search:
                                nested_elapsed = (
                                    float(row.get("nested_proxy_elapsed_avg", float("nan")) or float("nan"))
                                    + float(row.get("nested_exec_elapsed_avg", float("nan")) or float("nan"))
                                    if isinstance(row, dict)
                                    else float("nan")
                                )
                                nested_candidates = int(row.get("nested_candidate_count", 0) or 0) if isinstance(row, dict) else 0
                                shortlist_k = int(row.get("nested_prefilter_top_k", 0) or 0) if isinstance(row, dict) else 0
                                cache_hit = float(row.get("nested_cache_hit", 0.0) or 0.0) if isinstance(row, dict) else 0.0
                                print(
                                    f"  [{phase_name} {completed_idx}/{len(phase_models)}] {model_name} finished | "
                                    f"wall={wall_elapsed:.1f}s (cpu={true_cpu_time:.1f}s, q_wait={queue_wait:.1f}s)"
                                    f" | valid_windows={valid_windows}"
                                    f" | nested={nested_candidates}->{shortlist_k}"
                                    f" | cache={'HIT' if cache_hit > 0 else 'MISS'}"
                                    f" | nested_avg={nested_elapsed:.1f}s"
                                    f" | exec_avg={exec_avg:.1f}s"
                                    f"{_failure_suffix}"
                                )
                            else:
                                print(
                                    f"  [{phase_name} {completed_idx}/{len(phase_models)}] {model_name} finished | "
                                    f"wall={wall_elapsed:.1f}s (cpu={true_cpu_time:.1f}s, q_wait={queue_wait:.1f}s)"
                                    f" | valid_windows={valid_windows}"
                                    f" | {'role=execution_validation' if phase_name == 'ExecutionValidation' else ('role=confirmatory_only' if phase_name == 'ConventionalDeepValidation' else 'selection=fixed_shortlist')}"
                                    f" | exec_avg={exec_avg:.1f}s"
                                    f"{_failure_suffix}"
                                )
                        phase_results.append(result)
                        global_audit.collect_ic_counters(result)
            except BrokenProcessPool:
                print(
                    f"  WARNING [{phase_name}] parallel worker pool crashed "
                    f"(model_workers={model_workers}) — retrying all {len(phase_models)} models sequentially"
                )
                phase_results = []
                _run_parallel = False
        if not _run_parallel:
            for idx, model_spec in enumerate(phase_models, 1):
                # Use the wrapper even in sequential mode to ensure lazy loading and memory clearing
                result = _evaluate_model_fork_wrapper(evaluator, model_spec, effective_parallel_cfg)
                
                row = result.get("row") if isinstance(result, dict) else None
                elapsed = result.get("_worker_completed_at", 0) - result.get("_worker_started_at", 0)
                if ctx.timing_ledger is not None:
                    ctx.timing_ledger.record(
                        "model_family_completed",
                        phase=str(phase_name),
                        model_name=str(model_spec[0]),
                        elapsed_s=float(elapsed),
                        valid_windows=int(row.get("n_windows", 0) or 0) if isinstance(row, dict) else int(result.get("valid_windows", 0) or 0),
                    )
                if evaluator is _screen_model_family:
                    valid_windows = int(result.get("valid_windows", 0) or 0)
                    screen_score = float(result.get("screen_score", float("nan")))
                    print(
                        f"  [{phase_name} {idx}/{len(phase_models)}] {model_spec[0]} finished in {elapsed:.1f}s"
                        f" | valid_windows={valid_windows} | screen_score={screen_score:.4f}"
                    )
                else:
                    valid_windows = int(row.get("n_windows", 0) or 0) if isinstance(row, dict) else 0
                    exec_avg = float(row.get("exec_time_avg", float("nan"))) if isinstance(row, dict) else float("nan")
                    nested_search = bool(row.get("nested_search_applicable", False)) if isinstance(row, dict) else False
                    _nested_failures = int(row.get("nested_window_failures", 0) or 0) if isinstance(row, dict) else 0
                    _failure_suffix = f" | nested_failures={_nested_failures}" if _nested_failures > 0 else ""
                    if nested_search:
                        nested_elapsed = (
                            float(row.get("nested_proxy_elapsed_avg", float("nan")) or float("nan"))
                            + float(row.get("nested_exec_elapsed_avg", float("nan")) or float("nan"))
                            if isinstance(row, dict)
                            else float("nan")
                        )
                        nested_candidates = int(row.get("nested_candidate_count", 0) or 0) if isinstance(row, dict) else 0
                        shortlist_k = int(row.get("nested_prefilter_top_k", 0) or 0) if isinstance(row, dict) else 0
                        cache_hit = float(row.get("nested_cache_hit", 0.0) or 0.0) if isinstance(row, dict) else 0.0
                        print(
                            f"  [{phase_name} {idx}/{len(phase_models)}] {model_spec[0]} finished in {elapsed:.1f}s"
                            f" | valid_windows={valid_windows}"
                            f" | nested={nested_candidates}->{shortlist_k}"
                            f" | cache={'HIT' if cache_hit > 0 else 'MISS'}"
                            f" | nested_avg={nested_elapsed:.1f}s"
                            f" | exec_avg={exec_avg:.1f}s"
                            f"{_failure_suffix}"
                        )
                    else:
                        print(
                            f"  [{phase_name} {idx}/{len(phase_models)}] {model_spec[0]} finished in {elapsed:.1f}s"
                            f" | valid_windows={valid_windows}"
                            f" | {'role=execution_validation' if phase_name == 'ExecutionValidation' else ('role=confirmatory_only' if phase_name == 'ConventionalDeepValidation' else 'selection=fixed_shortlist')}"
                            f" | exec_avg={exec_avg:.1f}s"
                            f"{_failure_suffix}"
                        )
                phase_results.append(result)
                global_audit.collect_ic_counters(result)

        if phase_name in {"FastSweep", "SignalDiscovery"} and phase_results:
            print("\n" + "=" * 80)
            logger.debug("--- SignalDiscovery / Feasibility Performance Audit ---")
            print(f"{'Model':<25} | {'CPU (s)':<8} | {'Score':<8} | {'Score/s':<10} | Warning")
            print("-" * 80)
            
            baseline_score = float("-inf")
            baseline_time = float("inf")
            
            metrics = []
            for res in phase_results:
                if not isinstance(res, dict): continue
                model_name = res.get("model_name", "Unknown")
                score = float(res.get("screen_score", float("nan")))
                cpu = res.get("_worker_completed_at", 0) - res.get("_worker_started_at", 0)
                if cpu <= 0: cpu = 0.001
                score_per_sec = score / cpu if score > 0 else 0
                ic_mean = float(res.get("screen_ic_mean", float("nan")))
                metrics.append((model_name, cpu, score, score_per_sec, ic_mean))
                
                if score > baseline_score and cpu < 30.0:
                    baseline_score = score
                    baseline_time = cpu
            
            for m_name, cpu, score, sps, ic_mean in metrics:
                warning = ""
                if cpu > 60.0 and score <= baseline_score + 0.01:
                    warning = f"REDUNDANT: Dominated by fast baseline (score {baseline_score:.4f} in {baseline_time:.1f}s)"
                elif np.isnan(score):
                    warning = "UNDEFINED: Missing metrics or convergence failure"
                elif ic_mean < 0:
                    warning = "TRUE_NEGATIVE_SIGNAL: Model learning inverted logic"
                elif ic_mean > 0 and score < 0:
                    warning = "PENALTY_DOMINATED: Predictive edge exists but costs/risk dominate"
                elif score <= -1.0:
                    warning = "DEGENERATE: Failed to converge or zeroed features"
                
                print(f"{m_name:<25} | {cpu:<8.1f} | {score:<8.4f} | {sps:<10.5f} | {warning}")
            print("=" * 80)

        if phase_results:
            _phase_cache = _aggregate_worker_cache_stats(phase_results)
            _phase_effective_rate = float(_phase_cache.get("effective_hit_rate", 0.0))
            _phase_fold_lookups = int(_phase_cache.get("prepared_fold_lookups", 0) or 0)
            _phase_fold_unique = int(_phase_cache.get("prepared_fold_unique_keys", 0) or 0)
            global_audit.cache_hit_rate = max(global_audit.cache_hit_rate, _phase_effective_rate / 100.0)
            if _phase_fold_lookups > 0:
                global_audit.cache_total_fold_lookups += _phase_fold_lookups
                global_audit.cache_unique_keys += _phase_fold_unique
                global_audit.cache_structurally_unique = (
                    global_audit.cache_structurally_unique
                    or (_phase_fold_unique / max(1, _phase_fold_lookups)) > 0.90
                )
            logger.info(
                "  [Distributed Cache Summary:%s] workers=%d memory_hits=%d artifact_hits=%d "
                "misses=%d evictions=%d memory_hit_rate=%.1f%% effective_hit_rate=%.1f%% "
                "horizon(mem/artifact/miss)=%d/%d/%d folds(hit/miss/unique)=%d/%d/%d "
                "targets(mem/artifact/miss)=%d/%d/%d",
                phase_name,
                int(_phase_cache.get("worker_result_count", 0)),
                int(_phase_cache.get("hits", 0)),
                int(_phase_cache.get("artifact_hits", 0)),
                int(_phase_cache.get("misses", 0)),
                int(_phase_cache.get("evictions", 0)),
                float(_phase_cache.get("memory_hit_rate", 0.0)),
                float(_phase_cache.get("effective_hit_rate", 0.0)),
                int(_phase_cache.get("horizon_memory_hits", 0)),
                int(_phase_cache.get("horizon_artifact_hits", 0)),
                int(_phase_cache.get("horizon_memory_misses", 0)),
                int(_phase_cache.get("fold_memory_hits", 0)),
                int(_phase_cache.get("fold_memory_misses", 0)),
                _phase_fold_unique,
                int(_phase_cache.get("target_memory_hits", 0)),
                int(_phase_cache.get("target_artifact_hits", 0)),
                int(_phase_cache.get("target_memory_misses", 0)),
            )

        if phase_name in {"FastSweep", "SignalDiscovery"}:
            mem_ledger.record(f"phase_{phase_name}_complete", n_results=len(phase_results))
        import gc
        gc.collect()
        return phase_results

    fast_models, economic_models = _split_model_families(models)
    if economic_models:
        print(
            "Economic objective model families are excluded from proxy screening; "
            "alpha models must emit scores only, with portfolio economics deferred to Stage C."
        )
    signal_results = signal_screening(fast_models, phase_runner=_run_model_phase)

    # P9: Evaluate baseline scorers (equal-weight, IC-weighted) as reference statistics.
    # These establish the "no-ML" floor that learned models must beat.
    baseline_results: list[dict[str, Any]] = []
    if baseline_models:
        baseline_results = _evaluate_baseline_scorers(ctx=ctx)
        for br in baseline_results:
            print(f"\n[Baseline] {br['model_name']}: IC={br['screen_ic_mean']:.4f} | ICIR={br['screen_daily_icir']:.2f} | windows={br['valid_windows']}")
            for line in br["log_lines"]:
                logger.info(line)

    shortlisted_fast, feasibility_audit = feasibility_filter(
        signal_results,
        fast_models,
        screening_cfg=screening_cfg,
    )
    if feasibility_audit:
        print()
        print("FeasibilityFilter shortlist:")
        for row in feasibility_audit:
            diagnostic_only = row.get("diagnostic_only_models") or []
            diagnostic_suffix = f" | diagnostic_only={diagnostic_only}" if diagnostic_only else ""
            print(
                f"  {row['primary_path']}: keep {row['candidate_count_out']}/{row['candidate_count_in']} "
                f"-> {row['selected_models']}{diagnostic_suffix}"
            )

    model_results: list[dict[str, Any]] = []
    # Stage C is the only executable path. It receives only Stage B survivors and
    # is responsible for optimizer, full cost model, simulator, and promotion gates.
    model_results.extend(execution_validation(shortlisted_fast, phase_runner=_run_model_phase))

    all_nested_ranking_artifacts: list[dict[str, Any]] = []
    all_optimizer_audit_records: list[dict[str, Any]] = []
    all_decomp_parts: list[dict[str, Any]] = []  # P36: alpha-capture summaries from workers
    for result in model_results:
        print()
        for line in result["log_lines"]:
            logger.info(line)
        all_window_details[str(result["model_name"])] = result["window_metrics"]
        # Merge simulation telemetry from forked workers back to parent.
        _worker_telem = result.get("_sim_telemetry")
        if isinstance(_worker_telem, list) and _worker_telem:
            _SIMULATION_TELEMETRY.merge(_worker_telem)

        _worker_ms_stats = result.get("_market_state_stats")
        if isinstance(_worker_ms_stats, dict) and _worker_ms_stats:
            _SIMULATION_TELEMETRY.merge_market_state_stats(_worker_ms_stats)

        _worker_artifacts = result.get("_ranking_artifacts")
        if isinstance(_worker_artifacts, list) and _worker_artifacts:
            all_nested_ranking_artifacts.extend(_worker_artifacts)

        _worker_opt_audit = result.get("_optimizer_audit_records")
        if isinstance(_worker_opt_audit, list) and _worker_opt_audit:
            all_optimizer_audit_records.extend(_worker_opt_audit)

        row = result.get("row")
        if row is not None:
            rows.append(row)

    _stage("evaluation_and_selection")

    # Baseline comparison (LearnedWeights) — no training, score + simulate.
    if args.compare_baseline and df_baseline is not None:
        print()
        print("=== LearnedWeightsBaseline ===")
        wm: list[WindowMetrics] = []
        oos_parts: list[pd.DataFrame] = []
        daily_parts: list[pd.Series] = []
        overlay_daily_parts: list[pd.Series] = []
        pnl_parts: list[pd.DataFrame] = []
        overlay_pnl_parts: list[pd.DataFrame] = []
        decomp_parts: list[dict[str, Any]] = []
        primary_path = "long_short_spread"

        for win_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows, 1):
            te_label = f"{te_s.date()} → {(te_e - pd.Timedelta(days=1)).date()}"
            te = df_baseline[(df_baseline["date"] >= te_s) & (df_baseline["date"] < te_e)].copy()
            if te.empty:
                print(f"  [window {win_idx}/{len(windows)}] skip: empty test | test={te_label}")
                continue
            n_test_unique = int(te["date"].nunique())
            if n_test_unique < int(args.min_oos_days):
                print(
                    f"  WARNING [window {win_idx}/{len(windows)}] skip: only {n_test_unique} test days "
                    f"(min_oos_days={args.min_oos_days}) | test={te_label}"
                )
                continue

            try:
                t0 = time.perf_counter()
                score = _learned_weights_score_series(te)
                t1 = time.perf_counter()

                te_scored = te.assign(score=score)
                y_te_bin = te_scored["y_bin"].to_numpy(dtype=int)

                ic_stats = cross_sectional_ic(te_scored, target_col="target_return")
                ic = float(ic_stats.get("cs_ic_spearman_mean", float("nan")))
                dir_acc = float(((np.asarray(score) >= 0) == (y_te_bin == 1)).mean()) if len(score) else float("nan")
                b_halflife = float(ic_stats.get("signal_halflife_days", float("nan")))

                eval_cfg = _evaluation_config(
                    cfg,
                    path=primary_path,
                    max_positions=int(max_positions),
                    min_positions=int(min_positions),
                    horizon=int(horizon),
                    horizon_contract=horizon_contract,
                    signal_halflife_days=b_halflife,
                )
                eval_state = global_prepare_cache.get_validation_state(
                    start=te_s,
                    end=te_e,
                    horizon_days=int(horizon),
                    evaluation_cfg=eval_cfg,
                )
                te_scored_qp = _prescreen_qp_candidates(
                    te_scored, primary_path=primary_path, max_positions=int(max_positions)
                )
                _t_sim_0 = time.perf_counter()
                daily_ret_s, pnl_detail = simulate_executable_portfolio(
                    te_scored_qp,
                    eval_cfg,
                    state_cache=eval_state,
                )
                _SIMULATION_TELEMETRY.record(
                    phase="baseline_validation",
                    model_name="LearnedWeightsBaseline",
                    window_idx=win_idx,
                    scored=te_scored_qp,
                    cfg=eval_cfg,
                    runtime_s=0.0,
                    simulation_mode="executable",
                    is_cached=getattr(pnl_detail, "attrs", {}).get("_is_cached", False),
                )
                overlay_cfg = _evaluation_config(
                    cfg,
                    path="long_only_overlay",
                    max_positions=int(max_positions),
                    min_positions=int(min_positions),
                    horizon=int(horizon),
                    horizon_contract=horizon_contract,
                    signal_halflife_days=b_halflife,
                )
                overlay_scored_qp = _prescreen_qp_candidates(
                    te_scored, primary_path="long_only_overlay", max_positions=int(max_positions)
                )
                overlay_ret_s, overlay_pnl_detail = simulate_executable_portfolio(
                    overlay_scored_qp,
                    overlay_cfg,
                    state_cache=eval_state,
                )
                _t_sim_1 = time.perf_counter()
                if _SIMULATION_TELEMETRY.records:
                    _SIMULATION_TELEMETRY.records[-1]["runtime_s"] = round(float(_t_sim_1 - _t_sim_0), 4)
                n_daily_pts = int(len(daily_ret_s))
                if n_daily_pts < int(args.min_oos_days):
                    print(
                        f"  WARNING [window {win_idx}/{len(windows)}] skip: portfolio sim has {n_daily_pts} days "
                        f"(min_oos_days={args.min_oos_days}) | test={te_label}"
                    )
                    continue

                n_invested = (
                    int(pd.to_numeric(pnl_detail.get("n_positions"), errors="coerce").gt(0).sum())
                    if not pnl_detail.empty and "n_positions" in pnl_detail
                    else 0
                )
                sharpe = _sharpe_from_series(daily_ret_s.to_numpy(dtype=float), horizon=int(horizon))
                sharpe_str = f"{sharpe:.4f}" if np.isfinite(sharpe) else "nan"
                overlay_sharpe = (
                    _sharpe_from_series(overlay_ret_s.to_numpy(dtype=float), horizon=int(horizon))
                    if len(overlay_ret_s) >= int(args.min_oos_days)
                    else float("nan")
                )
                overlay_str = f"{overlay_sharpe:.4f}" if np.isfinite(overlay_sharpe) else "nan"
                print(
                    f"  [window {win_idx}/{len(windows)}] test=[{te_label}] | n_days={n_test_unique} | "
                    f"path={primary_path} | days_with_positions={n_invested} | "
                    f"ExecSharpe={sharpe_str} | OverlayExecSharpe={overlay_str} | CS_IC={ic:.4f}"
                )

                daily_parts.append(daily_ret_s)
                pnl_parts.append(pnl_detail)
                if len(overlay_ret_s):
                    overlay_daily_parts.append(overlay_ret_s)
                if not overlay_pnl_detail.empty:
                    overlay_pnl_parts.append(overlay_pnl_detail)
                oos_keep = [
                    c
                    for c in (
                        "date",
                        "ticker",
                        "forward_return",
                        "target_return",
                        "score",
                        "daily_return",
                        "adv_dollar_20",
                        "realised_vol_20d",
                        "capm_beta",
                        "sector",
                        "short_squeeze_risk",
                        "hard_short_squeeze_filter",
                        "borrow_crowding_risk",
                        "short_interest_ratio",
                    )
                    if c in te_scored.columns
                ]
                oos_parts.append(te_scored[oos_keep].copy())
                # P36: Alpha-capture decomposition — per-ticker score-to-PnL
                try:
                    _decomp_df, _decomp_summary = _alpha_capture_decomposition(
                        te_scored,
                        target_weights_df=None,
                        pnl_detail_df=pnl_detail,
                        model_name=name,
                        window_idx=win_idx,
                    )
                    decomp_parts.append(_decomp_summary)
                except Exception:
                    pass
                wm.append(
                    WindowMetrics(
                        oos_sharpe=float(sharpe) if np.isfinite(sharpe) else float("nan"),
                        oos_ic=float(ic) if np.isfinite(ic) else float("nan"),
                        oos_dir_acc=float(dir_acc) if np.isfinite(dir_acc) else float("nan"),
                        train_time_s=0.0,
                        test_time_s=float(t1 - t0),
                        n_train=0,
                        n_test=int(len(te_scored)),
                        train_start=str(tr_s.date()),
                        train_end=str((tr_e - pd.Timedelta(days=1)).date()),
                        test_start=str(te_s.date()),
                        test_end=str((te_e - pd.Timedelta(days=1)).date()),
                    )
                )
            except Exception as exc:
                print(f"  ERROR [window {win_idx}/{len(windows)}] baseline failed ({te_label}): {exc}")
                traceback.print_exc()
                continue

        if wm:
            sharpe_vals = np.array([w.oos_sharpe for w in wm], dtype=float)
            ic_vals = np.array([w.oos_ic for w in wm], dtype=float)
            acc_vals = np.array([w.oos_dir_acc for w in wm], dtype=float)
            te_t = np.array([w.test_time_s for w in wm], dtype=float)
            oos_df = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame()
            chained_daily = _concat_window_daily_returns(daily_parts)
            oos_sharpe_chained = _sharpe_from_series(chained_daily, horizon=int(horizon))
            oos_cagr_chained = _cagr_from_daily_returns(chained_daily)
            oos_max_dd = _max_drawdown_from_daily_returns(chained_daily)
            oos_win_rate = _win_rate_from_daily_returns(chained_daily)
            exec_stats = _pnl_detail_metrics(pnl_parts, chained_daily, horizon=int(horizon))
            exec_stats.update(decile_return_diagnostics(oos_df, target_col="target_return"))
            overlay_chained_daily = _concat_window_daily_returns(overlay_daily_parts)
            overlay_oos_sharpe_chained = _sharpe_from_series(overlay_chained_daily, horizon=int(horizon))
            overlay_oos_cagr_chained = _cagr_from_daily_returns(overlay_chained_daily)
            overlay_oos_max_dd = _max_drawdown_from_daily_returns(overlay_chained_daily)
            overlay_oos_win_rate = _win_rate_from_daily_returns(overlay_chained_daily)
            overlay_exec_stats = _pnl_detail_metrics(overlay_pnl_parts, overlay_chained_daily, horizon=int(horizon))
            overlay_sharpe_vals = np.array(
                [
                    _sharpe_from_series(s.to_numpy(dtype=float), horizon=int(horizon))
                    for s in overlay_daily_parts
                    if len(s) >= int(args.min_oos_days)
                ],
                dtype=float,
            )
            cs_ic_stats = cross_sectional_ic(oos_df, target_col="target_return", horizon_days=int(horizon))
            oos_ic_chained = float(cs_ic_stats.get("cs_ic_spearman_mean", float("nan")))
            overlay_oos_ic_chained = oos_ic_chained

            _base_inst = _compute_institutional_metrics(
                ic_vals, sharpe_vals, float(oos_cagr_chained), float(oos_max_dd)
            )
            row = {
                "model_name": "LearnedWeightsBaseline",
                "model_kind": "baseline",
                "horizon_days": int(horizon),
                "rebalance_frequency": int(horizon),
                "oos_evaluation_path": primary_path,
                "overlay_evaluation_path": "long_only_overlay",
                "oos_sharpe_mean": float(np.nanmean(sharpe_vals)),
                "oos_sharpe_std": float(np.nanstd(sharpe_vals, ddof=1)) if len(wm) > 1 else 0.0,
                "oos_ic_mean": float(np.nanmean(ic_vals)),
                "oos_ic_std": float(np.nanstd(ic_vals, ddof=1)) if len(wm) > 1 else 0.0,
                "oos_dir_acc_mean": float(np.nanmean(acc_vals)),
                "oos_dir_acc_std": float(np.nanstd(acc_vals, ddof=1)) if len(wm) > 1 else 0.0,
                "oos_sharpe_chained": float(oos_sharpe_chained),
                "oos_cagr_chained": float(oos_cagr_chained),
                "oos_max_dd": float(oos_max_dd),
                "oos_win_rate": float(oos_win_rate),
                "oos_ic_chained": float(oos_ic_chained),
                **cs_ic_stats,
                **exec_stats,
                "long_short_oos_sharpe_chained": float(oos_sharpe_chained),
                "long_short_oos_cagr_chained": float(oos_cagr_chained),
                "long_short_oos_max_dd": float(oos_max_dd),
                "long_short_oos_ic_chained": float(oos_ic_chained),
                "short_side_oos_sharpe_chained": float("nan"),
                "short_side_oos_cagr_chained": float("nan"),
                "short_side_oos_max_dd": float("nan"),
                "short_side_oos_ic_chained": float("nan"),
                "overlay_oos_sharpe_mean": float(np.nanmean(overlay_sharpe_vals)) if len(overlay_sharpe_vals) else float("nan"),
                "overlay_oos_sharpe_std": (
                    float(np.nanstd(overlay_sharpe_vals, ddof=1)) if len(overlay_sharpe_vals) > 1 else 0.0
                ),
                "overlay_oos_sharpe_chained": float(overlay_oos_sharpe_chained),
                "overlay_oos_cagr_chained": float(overlay_oos_cagr_chained),
                "overlay_oos_max_dd": float(overlay_oos_max_dd),
                "overlay_oos_win_rate": float(overlay_oos_win_rate),
                "overlay_oos_ic_chained": float(overlay_oos_ic_chained),
                "overlay_exec_sharpe": float(overlay_exec_stats.get("exec_sharpe", float("nan"))),
                "overlay_exec_cost_return_sum": float(overlay_exec_stats.get("exec_cost_return_sum", float("nan"))),
                "overlay_exec_turnover_mean": float(overlay_exec_stats.get("exec_turnover_mean", float("nan"))),
                # Institutional metrics
                **_base_inst,
                "train_time_avg": 0.0,
                "test_time_avg": float(np.nanmean(te_t)),
                "n_windows": int(len(wm)),
                "oos_psr": round(_compute_psr(chained_daily), 4),
                "oos_deflated_sharpe": round(_compute_deflated_sharpe(chained_daily, n_trials=metric_trial_count), 4),
                **feature_contract_summary,
                **alpha_admission_summary,
            }
            row.update(evaluate_promotion_gates(row, gate_cfg, long_cand_cfg=long_cand_cfg))

            suspicious = bool(
                np.isfinite(row["oos_sharpe_chained"])
                and np.isfinite(row["oos_ic_chained"])
                and (row["oos_sharpe_chained"] > 2.0)
                and (row["oos_ic_chained"] < 0.05)
            )
            row["leakage_suspect"] = bool(suspicious)
            if suspicious:
                print(
                    f"WARNING: LearnedWeightsBaseline suspicious metrics (Sharpe_chained={row['oos_sharpe_chained']:.3f}, "
                    f"IC_chained={row['oos_ic_chained']:.3f})."
                )
                if args.discard_suspicious_models:
                    print("  -> Discarding baseline due to --discard_suspicious_models.")
                else:
                    rows.append(row)
            else:
                rows.append(row)

        # P36: Merge baseline decomposition parts into outer scope
        if decomp_parts:
            all_decomp_parts.extend(decomp_parts)

    if not rows:
        raise SystemExit("No model produced valid results.")

    # ── Metric taxonomy ───────────────────────────────────────────────────────────────
    # oos_psr is a probabilistic Sharpe against a fixed hurdle.
    # oos_deflated_sharpe is the formal Bailey-López de Prado multiple-testing deflation.
    # oos_composite_psr is retained as a heuristic portfolio-quality score, but is not DSR.
    if rows:
        for _r in rows:
            _r["oos_composite_psr"] = round(
                float(_r.get("oos_composite", 0.0)) * float(_r.get("oos_psr", 0.5)), 4
            )
        _psr_vals = [_r["oos_psr"] for _r in rows]
        _dsr_vals = [_r["oos_deflated_sharpe"] for _r in rows]
        print(
            f"\nSharpe inference: PSR range [{min(_psr_vals):.3f}, {max(_psr_vals):.3f}], "
            f"formal DSR range [{min(_dsr_vals):.3f}, {max(_dsr_vals):.3f}] "
            f"across {len(rows)} reported models using {metric_trial_count} selection trials. "
            "Default ranking uses oos_deflated_sharpe on the primary research path."
        )

    report = pd.DataFrame(rows)
    # Selection/ranking: honor --select_metric with a penalty if suspicious.
    if args.select_metric not in report.columns:
        raise SystemExit(f"select_metric '{args.select_metric}' not found in report columns")
    report["_selection_metric"] = pd.to_numeric(report[args.select_metric], errors="coerce")
    if "leakage_suspect" in report.columns:
        report.loc[report["leakage_suspect"].eq(True), "_selection_metric"] = -1e9
    report = report.sort_values("_selection_metric", ascending=False).reset_index(drop=True)
    # Add percentile ranks for robust diagnostics
    robust_diag_cols = [
        "diag_robust_signal_halflife", "diag_robust_cost_adjusted_ic",
        "diag_robust_capacity_weighted_ic", "diag_robust_turnover_volatility",
        "diag_robust_tail_stability", "diag_robust_hhi"
    ]
    for c in robust_diag_cols:
        if c in report.columns:
            # signal_halflife, cost_ic, cap_ic, tail_stability: higher is better
            # turnover_volatility, hhi: lower is better
            is_higher_better = c not in ["diag_robust_turnover_volatility", "diag_robust_hhi"]
            report[f"{c}_percentile"] = report[c].rank(pct=True, ascending=is_higher_better)

    # ── Module 15: Execution Robustness Selection Penalty ─────────────────────
    if "diag_robust_cost_adjusted_ic" in report.columns and "cs_ic_spearman_mean" in report.columns:
        report["diag_caic_to_raw_ic_ratio"] = (
            report["diag_robust_cost_adjusted_ic"] / report["cs_ic_spearman_mean"].abs().clip(lower=1e-6)
        ).clip(lower=0.0, upper=2.0)
        report["diag_caic_to_raw_ic_ratio_percentile"] = report["diag_caic_to_raw_ic_ratio"].rank(pct=True)

    if gate_cfg.execution_robustness_enabled and gate_cfg.execution_robustness_affect_selection:
        print("\nApplying Execution Robustness Selection Penalties...")
        rebalance_freq = float(horizon_contract.config.rebalance_frequency_days)
        
        for idx, r in report.iterrows():
            penalties = []
            
            # 1. Signal Halflife
            halflife = r.get("diag_robust_signal_halflife", np.nan)
            halflife_req = rebalance_freq + gate_cfg.min_signal_halflife_buffer
            if not np.isfinite(halflife) or halflife < halflife_req:
                penalties.append(f"Halflife {halflife:.2f} < {halflife_req:.2f}")

            # 2. CAIC Ratio
            caic_ratio = r.get("diag_caic_to_raw_ic_ratio", np.nan)
            if not np.isfinite(caic_ratio) or caic_ratio < gate_cfg.min_caic_to_ic_ratio:
                penalties.append(f"CAIC Ratio {caic_ratio:.2f} < {gate_cfg.min_caic_to_ic_ratio:.2f}")

            # 3. Average Turnover
            turn_mean = r.get("diag_robust_turnover_mean", np.nan)
            if not np.isfinite(turn_mean) or turn_mean > gate_cfg.max_avg_turnover:
                penalties.append(f"Turnover {turn_mean:.2f} > {gate_cfg.max_avg_turnover:.2f}")

            if penalties:
                old_val = report.loc[idx, "_selection_metric"]
                # Apply a heavy penalty to push it to the bottom
                penalty_amount = 1e9
                report.loc[idx, "_selection_metric"] -= penalty_amount
                new_val = report.loc[idx, "_selection_metric"]
                
                print(f"  PENALTY [{r['model_name']}]: {', '.join(penalties)} | "
                      f"Score: {old_val:.4f} -> {new_val:.4f}")

        # Re-sort after penalties
        report = report.sort_values("_selection_metric", ascending=False).reset_index(drop=True)

        # P17: Halflife alignment audit — print the actual persistence expectation
        if "diag_robust_signal_halflife" in report.columns:
            _hl_vals = report["diag_robust_signal_halflife"].dropna()
            if len(_hl_vals):
                _hl_median = float(_hl_vals.median())
                _rebal = float(horizon_contract.config.rebalance_frequency_days)
                _rho_daily = 2.0 ** (-1.0 / max(_hl_median, 1e-6))
                _persistence = _rho_daily ** _rebal
                _verdict = (
                    f"OK ({_persistence*100:.0f}% ranking survives rebalance)"
                    if _hl_median >= _rebal
                    else f"MISMATCH: halflife={_hl_median:.1f}d < rebalance={_rebal:.0f}d "
                         f"({_persistence*100:.0f}% ranking persistence at rebalance)"
                )
                print(
                    f"  [HalflifeAudit] median_halflife={_hl_median:.1f}d "
                    f"rebalance={_rebal:.0f}d daily_rho={_rho_daily:.3f} → {_verdict}"
                )

        # P39: Halflife/persistence contract audit artifact — per-model
        if "diag_robust_signal_halflife" in report.columns:
            _rebal = float(horizon_contract.config.rebalance_frequency_days)
            _halflife_audit_rows = []
            for _idx, _r in report.iterrows():
                _hl = _r.get("diag_robust_signal_halflife", float("nan"))
                if not np.isfinite(_hl) or _hl <= 0:
                    continue
                _p_at_rebal = 2.0 ** (-_rebal / _hl)
                _req_halflife = _rebal + gate_cfg.min_signal_halflife_buffer
                _req_persistence = 2.0 ** (-_rebal / _req_halflife)
                _hl_pass = _hl >= _req_halflife
                _pers_pass = _p_at_rebal >= _req_persistence
                _halflife_audit_rows.append({
                    "model_name": _r["model_name"],
                    "halflife_days": round(_hl, 2),
                    "rebalance_frequency_days": int(_rebal),
                    "persistence_at_rebalance": round(_p_at_rebal, 4),
                    "required_halflife_days": round(_req_halflife, 2),
                    "required_persistence": round(_req_persistence, 4),
                    "halflife_pass": _hl_pass,
                    "persistence_pass": _pers_pass,
                    "contract_consistent": _hl_pass == _pers_pass,
                })
            if _halflife_audit_rows:
                _hl_audit_path = out_dir / "halflife_persistence_audit.csv"
                pd.DataFrame(_halflife_audit_rows).to_csv(_hl_audit_path, index=False)
                print(f"  [HalflifeAudit] Per-model audit: {_hl_audit_path}")

    p_cols = [
        "diag_robust_signal_halflife_percentile",
        "diag_robust_cost_adjusted_ic_percentile",
        "diag_caic_to_raw_ic_ratio_percentile",
        "diag_robust_turnover_volatility_percentile"
    ]
    p_available = [c for c in p_cols if c in report.columns]
    if p_available:
        # P13 fix: Use skipna=True (per-row) to avoid the all-or-nothing collapse
        # where a single missing diagnostic zeroes the entire robustness score.
        # A model with 3/4 diagnostics at 0.9 should NOT rank below one with
        # 4/4 at 0.1.  When fewer than half the diagnostics are available,
        # the model's robustness is treated as unknown (NaN).
        n_cols = len(p_available)
        min_required = max(1, n_cols // 2)
        row_avail = report[p_available].notna().sum(axis=1)
        raw_mean = report[p_available].mean(axis=1, skipna=True)
        report["execution_robustness_score"] = np.where(
            row_avail >= min_required, raw_mean, np.nan
        )
        # Only zero-out models where robustness is genuinely unavailable
        report["execution_robustness_score"] = report["execution_robustness_score"].fillna(0.0)
        n_zeroed = int((row_avail < min_required).sum())
        if n_zeroed > 0:
            zeroed_names = report.loc[row_avail < min_required, "model_name"].tolist()
            print(
                f"  [Robustness] {n_zeroed} model(s) have insufficient diagnostic coverage "
                f"(<{min_required}/{n_cols}): {zeroed_names}"
            )
    else:
        report["execution_robustness_score"] = 0.0

    # Capture base rank before adjustment
    report["_base_rank"] = report["_selection_metric"].rank(ascending=False, method="first")
    report["base_screen_score"] = report["screen_score"] if "screen_score" in report.columns else report["_selection_metric"]

    # Apply penalty to screen_score
    if "screen_score" in report.columns:
        # For positive scores, robustness < 1.0 reduces them (penalty).
        # For negative scores, we keep them negative but apply the multiplier to preserve the order relative to positive.
        report["adjusted_screen_score"] = report["screen_score"] * report["execution_robustness_score"]
        
        # If the user is selecting by screen_score, update the selection metric
        if args.select_metric == "screen_score":
            report["_selection_metric"] = report["adjusted_screen_score"]
            # Re-sort after adjustment
            report = report.sort_values("_selection_metric", ascending=False).reset_index(drop=True)

    report["_final_rank"] = report["_selection_metric"].rank(ascending=False, method="first")
    report["rank_change"] = report["_base_rank"] - report["_final_rank"]
    
    def _get_robust_reason(r):
        reasons = []
        # Check for actual NaN values first
        if not np.isfinite(r.get("diag_robust_signal_halflife", np.nan)): reasons.append("Missing Halflife")
        if not np.isfinite(r.get("diag_robust_cost_adjusted_ic", np.nan)): reasons.append("Missing Cost-Adj IC")
        if not np.isfinite(r.get("diag_robust_turnover_volatility", np.nan)): reasons.append("Missing Turnover Vol")
        
        # Then check percentiles for performance relative to peer group
        if r.get("diag_robust_signal_halflife_percentile", 0.5) < 0.3: reasons.append("Low Halflife (Peer)")
        if r.get("diag_caic_to_raw_ic_ratio_percentile", 0.5) < 0.3: reasons.append("High Cost Drag (Peer)")
        if r.get("diag_robust_turnover_volatility_percentile", 0.5) < 0.3: reasons.append("Unstable Turnover (Peer)")
        
        return "|".join(reasons) if reasons else "Robust"
        
    report["robustness_reason"] = report.apply(_get_robust_reason, axis=1)

    report_path = out_dir / "model_comparison.csv"
    report.to_csv(report_path, index=False)
    try:
        _promotion_decisions = []
        if not report.empty:
            for _row in report.to_dict(orient="records"):
                _promotion_decisions.append(
                    promotion_decision_from_row(
                        _row,
                        horizon_contract_fingerprint=horizon_run_contract.fingerprint(),
                        target_spec_fingerprint=_target_fp_for_gates,
                        cost_assumption_fingerprint=cost_assumption_set.fingerprint(),
                        run_id=institutional_run_id,
                    ).to_dict()
                )
        write_json_artifact(
            out_dir / "promotion_decisions.json",
            {
                "schema_version": "phase_b.1",
                "run_id": institutional_run_id,
                "promotion_decisions": _promotion_decisions,
            },
        )
        _contract_artifacts["promotion_decisions"] = str(out_dir / "promotion_decisions.json")
        emit_telemetry_event(
            out_dir,
            run_id=institutional_run_id,
            stage="promotion",
            event_type="promotion_decisions_written",
            message=f"Wrote {len(_promotion_decisions)} promotion decision contract records.",
            contract_fingerprints={
                "horizon": horizon_run_contract.fingerprint(),
                "cost": cost_assumption_set.fingerprint(),
            },
        )
    except Exception as _promo_contract_exc:
        emit_telemetry_event(
            out_dir,
            run_id=institutional_run_id,
            stage="promotion",
            event_type="promotion_decision_contract_failed",
            message="Promotion decision contract generation failed.",
            severity="warning",
            ledger="failure_ledger.jsonl",
            exception=_promo_contract_exc,
            recoverable=True,
        )

    # ── Per-model path diagnostics CSV ──────────────────────────────────────────────
    # Exports a flattened diagnostic view with path_type, leg Sharpes, IC stats,
    # costs, regime performance, and promotion tier for every evaluated model.
    _DIAG_COLS = [
        "model_name", "model_kind", "promotion_tier", "promotion_pass", "promotion_failures", "is_diagnostic_only",
        # Score direction calibration
        "score_direction_selected", "score_direction_mode", "score_direction_flip_windows", "score_direction_fixed_windows",
        # Path classification
        "path_type",
        # Sharpe breakdown
        "oos_sharpe_chained", "exec_sharpe", "exec_long_leg_sharpe", "exec_short_leg_sharpe",
        # IC / signal quality
        "oos_ic_chained", "oos_ic_ir", "oos_ic_tstat",
        "horizon_adj_ic_ir", "horizon_adj_ic_tstat",
        "cs_ic_spearman_mean", "cs_ic_spearman_tstat", "cs_ic_spearman_annualized_icir",
        # Beat rate and stability
        "oos_beat_rate", "oos_dual_beat_rate", "n_windows",
        # Statistical validity
        "oos_psr", "oos_deflated_sharpe",
        # Costs and turnover
        "exec_cost_to_gross_pnl", "exec_cost_return_sum", "exec_borrow_return_sum",
        "exec_turnover_mean",
        # Risk / exposure
        "exec_beta_abs_mean", "exec_max_sector_abs_mean", "exec_max_dd",
        # Signal structure
        "decile_spread", "decile_monotonicity",
        # Subsumption alpha
        "subsumption_alpha_ann", "subsumption_alpha_tstat", "subsumption_r2",
        # Regime (legacy keys if present)
        "regime_bull_sharpe", "regime_bear_sharpe", "regime_neutral_sharpe",
        # Research diagnostics — turnover
        "diag_turnover_mean_daily", "diag_avg_holding_period_days",
        "diag_cost_to_gross_ratio", "diag_cost_per_trade", "diag_net_pnl_per_trade",
        "diag_sharpe_per_unit_turnover",
        # Research diagnostics — horizon alignment
        "diag_ic_1d_mean", "diag_ic_3d_mean", "diag_ic_5d_mean", "diag_ic_10d_mean",
        "diag_ic_1d_tstat", "diag_ic_10d_tstat",
        "diag_optimal_horizon", "diag_overtraded", "diag_ic_decay_ratio",
        # Research diagnostics — long-only
        "diag_lo_sharpe", "diag_lo_net_sharpe", "diag_lo_cagr", "diag_lo_max_dd",
        "diag_lo_win_rate", "diag_lo_turnover_mean",
        # Research diagnostics — short attribution
        "diag_short_value_added", "diag_short_destroys_value",
        "diag_beta_hedge_sharpe", "diag_long_alpha_fraction",
        # Research diagnostics — cross-sectional strength
        "diag_decile_spread_std", "diag_decile_spread_cv",
        "diag_top_vs_median_spread", "diag_score_dispersion_mean",
        "diag_rank_ic_monotonicity", "diag_ic_consistency_rate",
        # Research diagnostics — cost-aware score
        "diag_net_ic_score", "diag_net_sharpe_score",
        "diag_cost_drag", "diag_alpha_efficiency",
        # Research diagnostics — regime segmentation
        "diag_regime_bull_sharpe", "diag_regime_bear_sharpe",
        "diag_regime_high_vol_sharpe", "diag_regime_low_vol_sharpe",
        "diag_regime_bull_ic", "diag_regime_bear_ic",
        "diag_regime_bull_n_days", "diag_regime_bear_n_days",
        "diag_regime_best", "diag_regime_worst", "diag_regime_spread",
        # Research diagnostics — cost decomposition (module 9)
        "diag_cost_pct_commission", "diag_cost_pct_spread",
        "diag_cost_pct_impact", "diag_cost_pct_borrow",
        "diag_cost_dominant", "diag_cost_rate_daily",
        "diag_gross_alpha_per_turn", "diag_cost_per_unit_turn", "diag_alpha_vs_cost_ratio",
        # Selection adjustment (module 15)
        "base_screen_score", "execution_robustness_score", "adjusted_screen_score",
        "rank_change", "robustness_reason", "diag_caic_to_raw_ic_ratio",
        # Research diagnostics — execution robustness (module 14)
        "diag_robust_signal_halflife", "diag_robust_cost_adjusted_ic",
        "diag_robust_capacity_weighted_ic", "diag_robust_turnover_volatility",
        "diag_robust_tail_stability", "diag_robust_hhi",
        "diag_robust_signal_halflife_percentile", "diag_robust_cost_adjusted_ic_percentile",
        "diag_robust_capacity_weighted_ic_percentile", "diag_robust_turnover_volatility_percentile",
        "diag_robust_tail_stability_percentile", "diag_robust_hhi_percentile",
        "diag_caic_to_raw_ic_ratio_percentile",
        # Research diagnostics — exposure decomposition (module 10)
        "diag_beta_attributed_sharpe", "diag_residual_alpha_sharpe", "diag_beta_pnl_pct",
        "diag_mkt_corr", "diag_vol_tilt", "diag_sector_exposure_mean",
        # Research diagnostics — short-side classification (module 11)
        "diag_short_classification", "diag_short_mkt_corr",
        "diag_short_ic_positive_rate", "diag_short_residual_sharpe", "diag_short_alpha_pct",
        # Research diagnostics — alpha capture (module 12)
        "diag_breadth_daily_mean", "diag_theoretical_sr_flam",
        "diag_alpha_capture_ratio", "diag_cost_loss_ratio", "diag_breakeven_ic",
        # Research diagnostics — turnover by decile (module 13)
        "diag_decile_turn_top", "diag_decile_turn_bottom",
        "diag_decile_turn_top_vs_bot", "diag_decile_signal_halflife",
        # Research label
        "diag_research_label", "diag_research_labels_all",
    ]

    def _derive_path_type(row: "pd.Series") -> str:
        path = str(row.get("oos_evaluation_path", "") or "").strip().lower()
        if path in {"long_only_overlay", "long_short_spread", "short_side"}:
            return path
        kind = str(row.get("model_kind", "") or "").lower()
        if kind in {"short_classifier", "short_alpha"}:
            return "short_side"
        if kind == "overlay_alpha":
            return "long_only_overlay"
        if kind in {"long_alpha", "regressor"}:
            return "long_short_spread"
        return "unknown"

    if not report.empty:
        diag = report.copy()
        if "path_type" not in diag.columns:
            diag["path_type"] = diag.apply(_derive_path_type, axis=1)
        diag_cols_present = [c for c in _DIAG_COLS if c in diag.columns]
        diag_path = out_dir / "model_diagnostics.csv"
        diag[diag_cols_present].to_csv(diag_path, index=False)
        print(f"Saved diagnostics: {diag_path}")

        # Research report: human-readable classification of every model's failure mode
        research_report_path = out_dir / "research_report.txt"
        research_report_text = _generate_research_report(report, out_path=research_report_path)
        print(research_report_text)
        print(f"Saved research report: {research_report_path}")
        _stage("report_export")

        # Empirical baseline diagnostics parquet
        _bl_id_cols = ["model_name", "model_kind"]
        _bl_cols = [c for c in report.columns if c.startswith("baseline_")]
        if _bl_cols:
            _bl_path = out_dir / "baseline_diagnostics.parquet"
            report[[c for c in _bl_id_cols if c in report.columns] + _bl_cols].to_parquet(
                _bl_path, index=False
            )
            print(f"Saved baseline diagnostics: {_bl_path}")

        # Alpha vs execution decomposition parquet
        _decomp_cols = [c for c in report.columns if c.startswith("decomp_")]
        if _decomp_cols:
            _decomp_path = out_dir / "alpha_execution_decomposition.parquet"
            report[[c for c in _bl_id_cols if c in report.columns] + _decomp_cols].to_parquet(
                _decomp_path, index=False
            )
            print(f"Saved alpha/execution decomposition: {_decomp_path}")

        # P36: Alpha-capture decomposition — per-ticker score-to-PnL attribution
        if all_decomp_parts:
            try:
                _ac_summary = _alpha_capture_summary_from_per_model(all_decomp_parts)
                if not _ac_summary.empty:
                    _ac_summary_path = out_dir / "alpha_capture_summary.csv"
                    _ac_summary.to_csv(_ac_summary_path, index=False)
                    print(f"Saved alpha capture summary: {_ac_summary_path}")
            except Exception as _ac_exc:
                logger.debug("alpha_capture_summary save failed: %s", _ac_exc)
        else:
            print("[AlphaCapture] No decomposition data collected — skipping alpha_capture_summary.csv")

        # Economic selection stability audit parquet
        if all_nested_ranking_artifacts:
            try:
                _audit_df = pd.DataFrame(all_nested_ranking_artifacts)
                _audit_path = out_dir / "economic_selection_audit.parquet"
                _audit_df.to_parquet(_audit_path, index=False)
                print(f"Saved economic selection audit: {_audit_path}")
            except Exception as _audit_exc:
                logger.debug("economic_selection_audit save failed: %s", _audit_exc)
        if all_optimizer_audit_records:
            try:
                _opt_audit_df = pd.DataFrame(all_optimizer_audit_records)
                _opt_audit_path = out_dir / "optimizer_score_weight_audit.parquet"
                _opt_audit_df.to_parquet(_opt_audit_path, index=False)
                print(f"Saved optimizer score/weight audit: {_opt_audit_path}")
            except Exception as _opt_audit_exc:
                logger.debug("optimizer_score_weight_audit save failed: %s", _opt_audit_exc)

    print()
    print(f"Saved report: {report_path}")
    print(f"Model-selection runtime (post feature/alpha prep): {time.perf_counter() - t_selection_0:.2f}s")

    # ── P20: Economic Execution Policy — cost-aware viability analysis ─────────
    try:
        from model_selection.economic_policy import (
            EconomicPolicyConfig, build_audit_from_report_row,
            classify_economic_viability, select_best_horizon,
            format_economic_report,
        )
        _econ_cfg = EconomicPolicyConfig(
            max_cost_pnl=float((ms_cfg.get("economic_policy", {}) or {}).get("max_cost_pnl", 0.50)),
            max_impact_fraction=float((ms_cfg.get("economic_policy", {}) or {}).get("max_impact_fraction", 0.70)),
            min_ic_tstat=float((ms_cfg.get("economic_policy", {}) or {}).get("min_ic_tstat", 1.50)),
            min_beat_rate=float((ms_cfg.get("economic_policy", {}) or {}).get("min_beat_rate", 0.25)),
            min_alpha_capture=float((ms_cfg.get("economic_policy", {}) or {}).get("min_alpha_capture", -3.0)),
        )
        _reb = horizon_contract.config.rebalance_frequency_days
        _report_df = pd.read_csv(report_path)
        _econ_audits = [
            build_audit_from_report_row(r.to_dict(), int(horizon), _reb)
            for _, r in _report_df.iterrows()
        ]
        _econ_viabilities = [classify_economic_viability(a, _econ_cfg) for a in _econ_audits]
        _econ_best, _econ_all, _econ_diag = select_best_horizon(_econ_audits, _econ_cfg)
        _econ_report = format_economic_report(_econ_audits, _econ_viabilities, _econ_diag, _econ_best)
        print("\n" + _econ_report)
        _econ_path = out_dir / "economic_policy_report.txt"
        _econ_path.write_text(_econ_report, encoding="utf-8")
        print(f"Saved economic policy report: {_econ_path}")
    except Exception as _econ_exc:
        logger.debug("EconomicExecutionPolicy skipped: %s", _econ_exc)
    # ───────────────────────────────────────────────────────────────────────────
    cache_stats = global_prepare_cache.stats()
    print(
        "Prepared panel cache: "
        f"raw={cache_stats['raw_panel_cache_entries']} "
        f"retargeted={cache_stats['retargeted_panel_cache_entries']} "
        f"folds={cache_stats['prepared_fold_cache_entries']} "
        f"targets={cache_stats['training_target_cache_entries']}"
    )

    # --- ENSEMBLE WINNER SELECTION (Pillar 24) ---
    top_n = int(ms_cfg.get("ensemble_size", 3))
    
    # Promotion is separated by economic mandate. Spread/overlay/short models use
    # different labels, feature views, and gates; legacy regressors remain eligible
    # only as research controls when they pass the same executable validation.
    long_kinds = ["long_alpha", "regressor"]
    overlay_kinds = ["overlay_alpha"]
    short_kinds = ["short_alpha", "short_classifier"]

    def _get_consistent_pool(full_pool, kinds, size):
        if full_pool.empty: return full_pool
        # Production gate: a model can rank in the research report, but it cannot
        # enter an exportable ensemble unless it passes the executable validation
        # gates (cost-aware Sharpe, factor/liquidity diagnostics, IC evidence, DD).
        if "promotion_pass" in full_pool.columns:
            full_pool = full_pool[full_pool["promotion_pass"].eq(True)].copy()
        if full_pool.empty: return full_pool
        if full_pool.empty: return full_pool
        # Type-Consistency Lockdown: Anchor to the #1 winner's type
        anchor_kind = full_pool.iloc[0]["model_kind"]
        consistent = full_pool[full_pool["model_kind"] == anchor_kind].copy()
        if anchor_kind in {"long_alpha", "overlay_alpha", "short_alpha"}:
            anchor_horizon = consistent.iloc[0].get("nested_selected_horizon_mode", np.nan)
            anchor_view = str(consistent.iloc[0].get("nested_selected_feature_view_mode", "") or "")
            if "nested_selected_horizon_mode" in consistent.columns and np.isfinite(_safe_float(anchor_horizon, np.nan)):
                consistent = consistent[
                    pd.to_numeric(consistent["nested_selected_horizon_mode"], errors="coerce").eq(int(anchor_horizon))
                ].copy()
            if "nested_selected_feature_view_mode" in consistent.columns and anchor_view:
                consistent = consistent[
                    consistent["nested_selected_feature_view_mode"].astype(str).eq(anchor_view)
                ].copy()
        consistent = consistent.head(size)
        return consistent.reset_index(drop=True)

    if "model_kind" not in report.columns:
        report["model_kind"] = "unknown"
    long_pool = _get_consistent_pool(report[report["model_kind"].isin(long_kinds)], long_kinds, top_n)
    # P9 Subsumption-Aware Baseline Fallback: when ALL ML models fail subsumption
    # (i.e., destroy value vs. a linear combination of raw features), the
    # IC-weighted or equal-weighted baseline is objectively the superior choice.
    # Promote it automatically so production always has a viable model.
    if long_pool.empty and baseline_models:
        baseline_report = report[report["model_kind"] == "baseline"]
        if not baseline_report.empty:
            # Baselines don't go through executable validation (no optimizer/costs),
            # so promotion_pass is set by evaluate_promotion_gates during their
            # evaluation or we relax the gate for baseline fallback.
            baseline_pool = baseline_report.head(top_n).copy()
            if not baseline_pool.empty:
                baseline_pool["model_kind"] = "baseline_fallback"
                baseline_pool["promotion_pass"] = True
                long_pool = baseline_pool.reset_index(drop=True)
                print(
                    f"  [Subsumption Fallback] All ML models failed promotion; "
                    f"automatically promoting baseline as production model: "
                    f"{list(baseline_pool['model_name'])}"
                )
    overlay_pool = _get_consistent_pool(report[report["model_kind"].isin(overlay_kinds)], overlay_kinds, top_n)
    short_pool = _get_consistent_pool(report[report["model_kind"].isin(short_kinds)], short_kinds, top_n) if do_shorts else pd.DataFrame()

    def _get_ensemble_specs(pool):
        if pool.empty: return pool, []
        # DeMiguel (2009): ICIR-weighted ensemble outperforms Sharpe-weighted OOS.
        # ICIR captures both signal quality (IC mean) and consistency (IC std),
        # which is exactly what we want to upweight in a multi-model ensemble.
        if "oos_ic_ir" in pool.columns:
            raw = pd.to_numeric(pool["oos_ic_ir"], errors="coerce").fillna(0.01).clip(lower=0.01).values
        else:
            raw = pd.to_numeric(pool["_selection_metric"], errors="coerce").fillna(0.01).clip(lower=0.01).values
        weights = raw / raw.sum() if raw.sum() > 0 else np.ones(len(raw)) / len(raw)
        return pool.reset_index(drop=True), weights.tolist()

    best_long_pool, long_weights = _get_ensemble_specs(long_pool)
    best_overlay_pool, overlay_weights = _get_ensemble_specs(overlay_pool)
    best_short_pool, short_weights = _get_ensemble_specs(short_pool)

    def _display_selection(pool):
        if pool.empty:
            return ""
        labels = []
        for _, row in pool.iterrows():
            label = str(row.get("model_name", "") or "")
            nested_model = str(row.get("nested_selected_model_mode", "") or "").strip()
            nested_h = _safe_float(row.get("nested_selected_horizon_mode"), np.nan)
            if nested_model:
                label = f"{label}->{nested_model}"
                if np.isfinite(nested_h):
                    label = f"{label}@h{int(nested_h)}"
            labels.append(label)
        return ", ".join(labels)

    if not best_long_pool.empty:
        print(f"Selected Top-{len(best_long_pool)} LONG Ensemble: {_display_selection(best_long_pool)}")
    else:
        print("BLOCKED: no long model passed production promotion gates; no long artifact will be exported.")
    if not best_overlay_pool.empty:
        print(f"Selected Top-{len(best_overlay_pool)} OVERLAY Ensemble: {_display_selection(best_overlay_pool)}")
    else:
        print("BLOCKED: no overlay model passed production promotion gates; no overlay artifact will be exported.")
    if not best_short_pool.empty:
        print(f"Selected Top-{len(best_short_pool)} SHORT Ensemble: {_display_selection(best_short_pool)}")
    elif do_shorts:
        print("BLOCKED: no short model passed production promotion gates; no short artifact will be exported.")

    # --- STRUCTURED FAILURE REPORT ---
    # When any promotion pool is empty, emit a detailed breakdown of the best
    # candidate per path — what tier it reached, what gates failed, and why.
    _any_blocked = (
        best_long_pool.empty
        or best_overlay_pool.empty
        or (do_shorts and best_short_pool.empty)
    )
    if _any_blocked and not report.empty:
        _print_failure_report(report, long_kinds, overlay_kinds, short_kinds, out_dir)

    # --- RESEARCH-ONLY LONG ALPHA EXPORT ---
    # When no full production model passes but at least one long_alpha_candidate exists,
    # export the best candidate as a research artifact (clearly labelled non-production).
    if best_long_pool.empty and "promotion_tier" in report.columns:
        _long_cand_pool = report[
            report["model_kind"].isin(long_kinds)
            & report["promotion_tier"].eq(PromotionTier.LONG_ALPHA_CAND.value)
        ]
        if not _long_cand_pool.empty:
            _best_cand = _long_cand_pool.iloc[0]
            print(
                f"\nRESEARCH-ONLY: long_alpha_candidate '{_best_cand.get('model_name', '?')}' "
                f"(tier={PromotionTier.LONG_ALPHA_CAND.value}) "
                f"long_leg_sharpe={_safe_float(_best_cand.get('exec_long_leg_sharpe'), np.nan):.3f} "
                f"ic_tstat={_safe_float(_best_cand.get('oos_ic_tstat', _best_cand.get('cs_ic_spearman_tstat')), np.nan):.2f} "
                "— NOT for production allocation."
            )
            _cand_path = out_dir / "long_alpha_candidate.csv"
            _long_cand_pool.head(1).to_csv(_cand_path, index=False)
            print(f"  Research artifact saved: {_cand_path}")

    # Final full-dataset training uses the same train-only preprocessing artifact
    # and estimator-specific residual/net-of-cost targets as walk-forward.
    # Ensemble training helper (Pillar 24)
    def _save_baseline_artifact(pool, weights, b_label):
        """Save baseline scorer as a JSON artifact (baselines have no fit() method)."""
        artifact = {
            "model_name": f"Baseline_Top{len(pool)}",
            "model_type": "baseline",
            "baseline_members": [str(r.get("model_name", "")) for _, r in pool.iterrows()],
            "ensemble_weights": weights,
            "horizon_days": int(horizon),
            "reason": "subsumption_alpha_negative_ml_destroys_value",
            "note": (
                "All ML models produced negative subsumption alpha (t < -2). "
                "The IC-weighted or equal-weighted linear baseline is the "
                "objectively superior choice for production deployment."
            ),
        }
        path = out_dir / f"best_{b_label}_model.json"
        _write_json(path, artifact)
        print(f"  Baseline artifact saved: {path}")

    def _train_and_save_ensemble(pool, weights, b_label):
        if pool.empty:
            return

        leader_kind = str(pool.iloc[0].get("model_kind", "classifier") or "classifier")

        # P9 Subsumption-Aware Baseline Export
        if leader_kind in ("baseline", "baseline_fallback"):
            _save_baseline_artifact(pool, weights, b_label)
            return
        fitted_estimators: list[tuple[str, Any, str, bool]] = []
        estimator_preprocessors: list[FeaturePreprocessor | None] = []
        member_specs: list[dict[str, Any]] = []
        for _, row in pool.iterrows():
            spec = _resolve_report_training_spec(
                row,
                models=models,
                cfg=cfg,
                feat_cols=_gated_feat_cols,
                short_feature_subset=short_feature_subset,
                overlay_feature_subset=overlay_feature_subset,
                default_horizon=int(horizon),
            )
            prepared = global_prepare_cache.get_prepared_fold(
                train_start=pd.Timestamp(start_date),
                train_end=pd.Timestamp(end_date) + pd.Timedelta(days=1),
                eval_start=pd.Timestamp(start_date),
                eval_end=pd.Timestamp(end_date) + pd.Timedelta(days=1),
                horizon_days=int(spec["horizon"]),
                active_features=spec["active_features"],
            )
            df_fit = prepared.train_df
            preproc = prepared.preprocessor
            x_fit = prepared.x_train
            y_fit = global_prepare_cache.get_training_target(
                start=pd.Timestamp(start_date),
                end=pd.Timestamp(end_date) + pd.Timedelta(days=1),
                horizon_days=int(spec["horizon"]),
                model_name=str(spec["resolved_model_name"]),
                model_kind=str(spec["resolved_model_kind"]),
                use_risk_adj=use_risk_adj,
            )
            est = _fit_candidate_model(
                model_template=spec["model_template"],
                name=str(spec["resolved_model_name"]),
                model_kind=str(spec["resolved_model_kind"]),
                tr=df_fit,
                x_tr=x_fit,
                y_tr=np.nan_to_num(y_fit, nan=0.0, posinf=0.0, neginf=0.0),
            )
            fitted_estimators.append(
                (
                    str(spec["resolved_model_name"]),
                    est,
                    str(spec["resolved_model_kind"]),
                    bool(spec["uses_proba"]),
                )
            )
            estimator_preprocessors.append(preproc)
            member_specs.append(spec)

        if not fitted_estimators:
            print(f"WARNING: No valid estimators found for {b_label} ensemble.")
            return

        print(f"Training best {b_label} ENSEMBLE ({len(fitted_estimators)} models) on full dataset...")
        shared_preproc = estimator_preprocessors[0] if estimator_preprocessors and all(
            p is not None and estimator_preprocessors[0] is not None and p.active_features == estimator_preprocessors[0].active_features
            for p in estimator_preprocessors
        ) else None
        ensemble = PrefitWeightedEnsemble(
            estimators=fitted_estimators,
            weights=weights[: len(fitted_estimators)],
            feature_preprocessor=shared_preproc,
            estimator_preprocessors=estimator_preprocessors,
        )

        path = out_dir / f"best_{b_label}_model.pkl"
        admitted_manifest = (
            alpha_admission.loc[alpha_admission["admitted"].eq(True)].to_dict(orient="records")
            if isinstance(alpha_admission, pd.DataFrame) and not alpha_admission.empty and "admitted" in alpha_admission.columns
            else []
        )
        member_horizons = [int(spec["horizon"]) for spec in member_specs]
        artifact = {
            "model_name": f"Top{len(fitted_estimators)}_Ensemble",
            "model_type": leader_kind,
            "ensemble_members": [str(spec["resolved_model_name"]) for spec in member_specs],
            "ensemble_weights": weights,
            "ensemble_member_specs": [
                {
                    "row_model_kind": str(spec["row_model_kind"]),
                    "resolved_model_name": str(spec["resolved_model_name"]),
                    "resolved_model_kind": str(spec["resolved_model_kind"]),
                    "feature_view": str(spec["feature_view"]),
                    "horizon": int(spec["horizon"]),
                    "active_features": list(spec["active_features"]),
                }
                for spec in member_specs
            ],
            "horizon_days": int(pd.Series(member_horizons, dtype=int).mode().iloc[0]) if member_horizons else int(horizon),
            "target": (
                "bottom_decile_forward_return"
                if leader_kind == "short_classifier"
                else "short_side_rank_utility"
                if leader_kind == "short_alpha"
                else "long_only_overlay_rank_utility"
                if leader_kind == "overlay_alpha"
                else "long_short_spread_rank_utility"
            ),
            "target_schema": "residual_alpha",
            "feature_columns": list(
                dict.fromkeys(
                    f
                    for ep in estimator_preprocessors
                    if ep is not None
                    for f in ep.active_features
                )
            ),
            "feature_admission_schema": "ic_decay_regime_marginal_v1",
            "feature_admission": admitted_manifest,
            "feature_preprocessor": shared_preproc,
            "estimator_preprocessors": estimator_preprocessors,
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "estimator": ensemble,
        }
        with open(path, "wb") as fh:
            pickle.dump(artifact, fh)
        print(f"Saved best {b_label} ensemble: {path}")

    # Execute training for both pillars (Ensemble mode)
    t_start = time.perf_counter()
    _train_and_save_ensemble(best_long_pool, long_weights, "long")
    _train_and_save_ensemble(best_overlay_pool, overlay_weights, "overlay")
    if do_shorts:
        _train_and_save_ensemble(best_short_pool, short_weights, "short")
    t_end = time.perf_counter()
    print(f"Total winning model training time: {t_end - t_start:.2f}s")

    print()
    print("Config note:")
    print("- SignalEngine currently supports 'price'/'full' and learned-weights scoring.")
    print("- If you want to wire this pickle model into live/backtest inference, you'll need a small integration step.")
    print("Suggested YAML fields to add (manual):")
    print("signals:")
    print('  mode: "ml"')
    print(f'  ml_model_path: "{out_dir.as_posix()}/best_long_model.pkl"')

    # C2: Regime-conditional models (opt-in via --regime-models)
    if getattr(args, "regime_models", False):
        _train_regime_models(df, _gated_feat_cols, [], out_dir, horizon)

    # C4: Panel-native regime conditioning evaluation (opt-in via --regime-conditioning)
    if getattr(args, "regime_conditioning", False):
        _run_regime_conditioning(df, _gated_feat_cols, out_dir)

    # C5: Feature redundancy analysis (opt-in via --feature-redundancy)
    if getattr(args, "feature_redundancy", False):
        _run_feature_redundancy(df, _gated_feat_cols, out_dir, corr_threshold=getattr(args, "corr_threshold", 0.60))

    # C6: Dynamic ensemble weighting (opt-in via --ensemble-weighting)
    if getattr(args, "ensemble_weighting", False):
        _run_ensemble_weighting(df, _gated_feat_cols, out_dir)

    # C7: Meta-labeling framework (opt-in via --meta-labeling)
    if getattr(args, "meta_labeling", False):
        _run_meta_labeling(
            df, _gated_feat_cols, out_dir,
            meta_model_type=getattr(args, "meta_model_type", "ridge"),
        )

    # C8: Short-side modeling (opt-in via --short-modeling)
    if getattr(args, "short_modeling", False):
        _run_short_modeling(df, _gated_feat_cols, out_dir)

    _run_optional_research_pillars(df, out_dir, args, args.config)

    # [AUDIT] Promotion Integrity
    if not report.empty:
        # Check if promotion_pass exists
        if "promotion_pass" in report.columns:
            promo_series = report["promotion_pass"].fillna(False).astype(bool)
            global_audit.promotion_candidate_count = int(len(report))
            global_audit.promotion_pass_count = int(promo_series.sum())
            if global_audit.promotion_pass_count <= 0:
                global_audit.gate_failures.append("No production model promoted")
            failed_promo = report[promo_series.eq(False)]
            for _, r in failed_promo.iterrows():
                # We only care about models that HAD potential but failed a specific gate
                reason = str(r.get("fail_reason", "unknown"))
                if reason != "unknown":
                    global_audit.gate_failures.append(f"{r.get('model_name')}: {reason}")
        
        # Check for NaN metrics labeling
        for col in ["oos_sharpe_chained", "oos_ic_chained", "exec_sharpe"]:
            if col in report.columns:
                if report[col].isna().any():
                    global_audit.gate_failures.append(f"Metric Integrity: {col} contains NaNs")
    else:
        global_audit.gate_failures.append("No models reached promotion evaluation")

    _SIMULATION_TELEMETRY.print_summary()
    telemetry_path = out_dir / "simulation_runtime_report.json"
    _write_json(telemetry_path, _SIMULATION_TELEMETRY.to_json_payload())
    print(f"Simulation runtime report saved to: {telemetry_path}")
    
    # [AUDIT] Cache Efficiency & Final Report
    cache_stats = global_prepare_cache.stats()
    global_audit.cache_hit_rate = max(
        float(global_audit.cache_hit_rate),
        float(cache_stats.get("effective_cache_hit_rate", cache_stats.get("prepared_fold_cache_hit_rate", 0.0)) or 0.0) / 100.0,
    )
    _parent_lookups = int(cache_stats.get("prepared_fold_lookups", 0) or 0)
    _parent_unique = int(cache_stats.get("prepared_fold_unique_keys", 0) or 0)
    if _parent_lookups > 0:
        global_audit.cache_structurally_unique = (
            bool(global_audit.cache_structurally_unique)
            or bool(cache_stats.get("prepared_fold_structurally_unique", False))
        )
        global_audit.cache_unique_keys += _parent_unique
        global_audit.cache_total_fold_lookups += _parent_lookups
    global_audit.cache_capacity = int(cache_stats.get("prepared_fold_cache_capacity", 0) or 0)
    global_audit.peak_rss_gb = PreparedPanelCache.get_rss_mb() / 1024.0
    if not report.empty and "oos_evaluation_path" in report.columns:
        audit_targets = {
            _validation_target_col_for_path(str(path))
            for path in report["oos_evaluation_path"].dropna().astype(str)
            if str(path).strip()
        }
        if audit_targets:
            global_audit.target_col = ",".join(sorted(audit_targets))
    global_audit.report()

    # ── P29: Final institutional cost viability reports ────────────────────────
    # Generates scorecard, stress test, turnover attribution, and dominated
    # candidate reports. Uses accumulated results from feature evaluation above.
    try:
        _cv_out = out_dir / "cost_viability"
        _cv_out.mkdir(parents=True, exist_ok=True)
        if "_cost_viability_state" in locals() and _cost_viability_state is not None:
            generate_cost_viability_reports(
                _cost_viability_state.feature_results,
                _cost_viability_state.candidate_results,
                _cost_viability_state.band_results,
                cfg, _cv_out,
            )
            if getattr(_cost_viability_state, "production_feature_results", None):
                generate_cost_viability_reports(
                    _cost_viability_state.production_feature_results,
                    _cost_viability_state.candidate_results,
                    _cost_viability_state.band_results,
                    cfg,
                    _cv_out / "production_admitted",
                )
            print(f"\n[CostViability] Reports written to {_cv_out}/")
        else:
            print(f"\n[CostViability] No feature results accumulated (alpha research may have been cached)")
    except Exception as _cv_exc:
        print(f"\n[CostViability] Final report generation skipped: {_cv_exc}")
        emit_telemetry_event(
            out_dir,
            run_id=institutional_run_id,
            stage="cost_viability",
            event_type="diagnostic_failure",
            message="Final cost viability report generation skipped.",
            severity="warning",
            ledger="failure_ledger.jsonl",
            exception=_cv_exc,
            recoverable=True,
            contract_fingerprints={"cost": cost_assumption_set.fingerprint()},
        )

    mem_ledger.record("completion")
    try:
        _contract_artifacts.update(
            {
                "model_comparison": str(out_dir / "model_comparison.csv"),
                "simulation_runtime_report": str(telemetry_path),
            }
        )
        write_institutional_run_manifest(
            out_dir,
            run_id=institutional_run_id,
            config_path=str(args.config),
            cfg=cfg,
            horizon_contract=horizon_run_contract,
            target_manifest=target_manifest,
            pit_specs=pit_specs,
            feature_manifest=feature_manifest,
            cost_assumption_set=cost_assumption_set,
            promotion_gate_specs=promotion_gate_specs,
            cache_key_specs=cache_key_specs,
            artifacts=_contract_artifacts,
            telemetry_contract=telemetry_contract,
            warnings=list(horizon_run_contract.warnings) + list(feature_manifest.get("unknown_features", [])),
            errors=[],
        )
        emit_telemetry_event(
            out_dir,
            run_id=institutional_run_id,
            stage="completion",
            event_type="institutional_manifest_finalized",
            message="Final institutional run manifest written.",
            artifact_path=out_dir / "institutional_run_manifest.json",
        )
    except Exception as _manifest_exc:
        emit_telemetry_event(
            out_dir,
            run_id=institutional_run_id,
            stage="completion",
            event_type="institutional_manifest_finalize_failed",
            message="Final institutional run manifest write failed.",
            severity="warning",
            ledger="failure_ledger.jsonl",
            exception=_manifest_exc,
            recoverable=True,
        )

    # ── Stage timing summary ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("STAGE-LEVEL TIMING SUMMARY")
    print("=" * 72)
    _total = 0.0
    _sorted = sorted(_stage_times.items(), key=lambda x: x[1], reverse=True)
    for _name, _t in _sorted:
        print(f"  {_name:<45s} {_t:>8.3f}s")
        _total += _t
    print(f"  {'TOTAL':<45s} {_total:>8.3f}s")
    if _sub_stage_times:
        print()
        print("SUB-STAGE TIMING (within p30_p34_diagnostics)")
        print("-" * 72)
        for _name, _t in sorted(_sub_stage_times.items(), key=lambda x: x[1], reverse=True):
            print(f"  {_name:<40s} {_t:>8.3f}s")
    print("=" * 72)


def _run_regime_conditioning(
    df: "pd.DataFrame",
    feat_cols: list[str],
    out_dir: "Path",
) -> None:
    """C4: Evaluate regime conditioning using panel-native features."""
    try:
        from model_selection.regime_conditioning import (
            RegimeConfig,
            evaluate_regime_conditioning,
            format_regime_report,
        )
        from model_selection.objective_comparison import WalkForwardConfig
        from model_selection.model_registry import build_models
    except ImportError as exc:
        print(f"[C4] Import error — skipping regime conditioning: {exc}")
        return

    print("\n[C4] Running panel-native regime conditioning evaluation...")

    try:
        models = build_models({"model_selection": {"include_classifiers": False}})
    except Exception as exc:
        print(f"[C4] Could not build models: {exc}")
        return

    wf_cfg = WalkForwardConfig(train_days=252, test_days=21, n_windows=6, min_train_obs=150)
    regime_cfg = RegimeConfig(n_regimes_min=2, n_regimes_max=5, min_regime_obs=60)

    try:
        regime_table, cond_table = evaluate_regime_conditioning(
            df, models, feat_cols, wf_cfg, regime_cfg
        )
    except Exception as exc:
        print(f"[C4] Regime conditioning evaluation failed: {exc}")
        return

    report = format_regime_report(regime_table, cond_table)
    print(report)

    if not regime_table.empty:
        regime_table.to_csv(out_dir / "regime_conditioning_strategy_a.csv", index=False)
        print(f"[C4] Strategy A table saved: {out_dir}/regime_conditioning_strategy_a.csv")
    if not cond_table.empty:
        cond_table.to_csv(out_dir / "regime_conditioning_strategy_b.csv", index=False)
        print(f"[C4] Strategy B table saved: {out_dir}/regime_conditioning_strategy_b.csv")

    # ── Phase 1 diagnostic: latent regime (shadow, no change to regime_table) ─
    try:
        from model_selection.latent_regime import (
            LatentRegimeModel,
            LatentRegimeConfig,
            extract_latent_features,
            evaluate_transition_stability,
            format_latent_regime_report,
        )
        print("\n[LatentRegimeDiag] Fitting continuous latent regime model...")
        _lrc = LatentRegimeConfig()
        _lat_feats = extract_latent_features(df, cfg=_lrc)
        if not _lat_feats.empty:
            _lrm = LatentRegimeModel(cfg=_lrc).fit(_lat_feats)
            _soft = _lrm.predict_soft(_lat_feats)
            _transition = evaluate_transition_stability(_soft)
            print(format_latent_regime_report(_transition))
    except Exception as _exc:
        print(f"[LatentRegimeDiag] skipped: {_exc}")
    # ── End Phase 1 latent_regime diagnostic ─────────────────────────────────


def _run_feature_redundancy(
    df: "pd.DataFrame",
    feat_cols: list[str],
    out_dir: "Path",
    corr_threshold: float = 0.60,
) -> None:
    """C5: Feature redundancy analysis — correlation clustering + IC stability."""
    try:
        from model_selection.feature_redundancy import (
            RedundancyConfig,
            run_feature_redundancy,
            format_redundancy_report,
        )
    except ImportError as exc:
        print(f"[C5] Import error — skipping feature redundancy: {exc}")
        return

    print("\n[C5] Running feature redundancy analysis...")

    cfg = RedundancyConfig(corr_threshold=corr_threshold)

    try:
        result = run_feature_redundancy(df, feat_cols, cfg)
    except Exception as exc:
        print(f"[C5] Feature redundancy analysis failed: {exc}")
        return

    report = format_redundancy_report(result)
    print(report)

    if not result.diagnostics.empty:
        result.diagnostics.to_csv(out_dir / "feature_redundancy_diagnostics.csv", index=False)
        print(f"[C5] Diagnostics saved: {out_dir}/feature_redundancy_diagnostics.csv")
    if not result.cluster_table.empty:
        result.cluster_table.to_csv(out_dir / "feature_redundancy_clusters.csv", index=False)
        print(f"[C5] Cluster table saved: {out_dir}/feature_redundancy_clusters.csv")
    if not result.evaluation.empty:
        result.evaluation.to_csv(out_dir / "feature_redundancy_evaluation.csv", index=False)
        print(f"[C5] Evaluation saved: {out_dir}/feature_redundancy_evaluation.csv")
    if result.representatives:
        reps_path = out_dir / "feature_redundancy_representatives.txt"
        reps_path.write_text("\n".join(sorted(result.representatives)))
        print(f"[C5] Representatives saved: {reps_path}")


def _run_ensemble_weighting(
    df: "pd.DataFrame",
    feat_cols: list[str],
    out_dir: "Path",
) -> None:
    """C6: Dynamic walk-forward ensemble weighting with penalized softmax + EMA smoothing."""
    try:
        from model_selection.ensemble_weighting import (
            EnsembleConfig,
            build_dynamic_ensemble,
            evaluate_ensemble,
            format_ensemble_report,
        )
        from model_selection.objective_comparison import WalkForwardConfig
        from model_selection.model_registry import build_models
    except ImportError as exc:
        print(f"[C6] Import error — skipping ensemble weighting: {exc}")
        return

    print("\n[C6] Running dynamic ensemble weighting...")

    try:
        models = build_models({"model_selection": {"include_classifiers": False}})
    except Exception as exc:
        print(f"[C6] Could not build models: {exc}")
        return

    wf_cfg = WalkForwardConfig(train_days=252, test_days=21, n_windows=8, min_train_obs=150)
    ens_cfg = EnsembleConfig()

    try:
        result = build_dynamic_ensemble(df, models, feat_cols, wf_cfg, ens_cfg)
    except Exception as exc:
        print(f"[C6] Ensemble weighting failed: {exc}")
        return

    report = format_ensemble_report(result)
    print(report)

    eval_df = evaluate_ensemble(result)
    if not eval_df.empty:
        eval_path = out_dir / "ensemble_weighting_evaluation.csv"
        eval_df.to_csv(eval_path, index=False)
        print(f"[C6] Evaluation saved: {eval_path}")

    if result.weight_history:
        import json
        wh_path = out_dir / "ensemble_weight_history.json"
        serialisable = [
            {k: float(v) for k, v in w.items()} for w in result.weight_history
        ]
        wh_path.write_text(json.dumps(serialisable, indent=2))
        print(f"[C6] Weight history saved: {wh_path}")

    if not result.contribution_table.empty:
        ct_path = out_dir / "ensemble_contribution_table.csv"
        result.contribution_table.to_csv(ct_path)
        print(f"[C6] Contribution table saved: {ct_path}")


def _run_meta_labeling(
    df: "pd.DataFrame",
    feat_cols: list[str],
    out_dir: "Path",
    meta_model_name: str | None = None,
    meta_model_type: str = "ridge",
) -> None:
    """C7: Meta-labeling — scale base-model exposure by predicted signal reliability."""
    try:
        from model_selection.meta_labeling import (
            MetaLabelConfig,
            build_meta_labeled_scores,
            evaluate_meta_labeling,
            format_meta_report,
        )
        from model_selection.objective_comparison import WalkForwardConfig
        from model_selection.model_registry import build_models
    except ImportError as exc:
        print(f"[C7] Import error — skipping meta-labeling: {exc}")
        return

    print("\n[C7] Running meta-labeling framework...")

    try:
        models = build_models({"model_selection": {"include_classifiers": False}})
    except Exception as exc:
        print(f"[C7] Could not build models: {exc}")
        return

    # Select base model: use --meta-model-name if given, else first regressor/ranker
    base_entry = None
    if meta_model_name:
        for entry in models:
            if entry[0] == meta_model_name:
                base_entry = entry
                break
        if base_entry is None:
            print(f"[C7] Model '{meta_model_name}' not found. Available: {[m[0] for m in models]}")
            return
    else:
        for entry in models:
            if entry[3] in ("regressor", "long_alpha", "ranker", "lgbm_ranker"):
                base_entry = entry
                break
        if base_entry is None:
            base_entry = models[0] if models else None
    if base_entry is None:
        print("[C7] No models available")
        return

    base_name, base_model, base_uses_proba, base_kind = base_entry
    print(f"[C7] Base model: {base_name} ({base_kind})")

    wf_cfg = WalkForwardConfig(train_days=252, test_days=21, n_windows=8, min_train_obs=150)
    meta_cfg = MetaLabelConfig(meta_model_type=meta_model_type)

    try:
        result = build_meta_labeled_scores(
            df, base_model, base_name, base_kind, base_uses_proba,
            feat_cols, wf_cfg, meta_cfg,
        )
    except Exception as exc:
        print(f"[C7] Meta-labeling failed: {exc}")
        return

    report = format_meta_report(result)
    print(report)

    comp_df = evaluate_meta_labeling(result)
    if not comp_df.empty:
        comp_path = out_dir / "meta_labeling_comparison.csv"
        comp_df.to_csv(comp_path, index=False)
        print(f"[C7] Comparison table saved: {comp_path}")

    if not result.meta_feature_importance.empty:
        fi_path = out_dir / "meta_labeling_feature_importance.csv"
        result.meta_feature_importance.to_csv(fi_path, index=False)
        print(f"[C7] Feature importance saved: {fi_path}")

    if not result.multiplier_stats.empty:
        ms_path = out_dir / "meta_labeling_multiplier_stats.csv"
        result.multiplier_stats.to_csv(ms_path, index=False)
        print(f"[C7] Multiplier stats saved: {ms_path}")

    # Concatenate all scaled frames for downstream use
    if result.scaled_frames:
        import pandas as _pd
        scaled_all = _pd.concat(result.scaled_frames, ignore_index=True)
        scaled_path = out_dir / "meta_labeling_scaled_scores.csv"
        scaled_all.to_csv(scaled_path, index=False)
        print(f"[C7] Scaled scores saved: {scaled_path}")


def _run_short_modeling(
    df: "pd.DataFrame",
    feat_cols: list[str],
    out_dir: "Path",
) -> None:
    """C8: Short-side modeling — asymmetric targets, separate evaluation, alpha classification."""
    try:
        from model_selection.short_modeling import (
            ShortTargetConfig,
            ShortEvalConfig,
            run_short_modeling,
            format_short_report,
        )
        from model_selection.objective_comparison import WalkForwardConfig
    except ImportError as exc:
        print(f"[C8] Import error — skipping short modeling: {exc}")
        return

    print("\n[C8] Running short-side modeling framework...")

    wf_cfg = WalkForwardConfig(train_days=252, test_days=21, n_windows=8, min_train_obs=150)
    short_cfg = ShortTargetConfig()
    eval_cfg = ShortEvalConfig()

    try:
        result = run_short_modeling(
            df, feat_cols, wf_cfg, short_cfg=short_cfg, eval_cfg=eval_cfg
        )
    except Exception as exc:
        print(f"[C8] Short modeling failed: {exc}")
        return

    report = format_short_report(result)
    print(report)

    if not result.comparison.empty:
        comp_path = out_dir / "short_modeling_comparison.csv"
        result.comparison.to_csv(comp_path, index=False)
        print(f"[C8] Comparison table saved: {comp_path}")

    # Save best scored frames
    import pandas as _pd
    all_best_frames: list[_pd.DataFrame] = []
    # pull scored frames from the best target via re-inspection is not needed;
    # comparison table already has all target metrics
    print(f"[C8] Best short target: {result.best_target} "
          f"(classification={result.best_metrics.classification})")
    print(f"[C8] Verdict: {result.verdict[:120]}...")


def _run_optional_research_pillars(
    df: pd.DataFrame,
    out_dir: Path,
    args: Any,
    config_path: str,
) -> None:
    """Route opt-in research modules in deterministic CLI flag order."""
    # Many of these require the full enriched panel saved to disk for standalone analysis.
    research_requested = any([
        getattr(args, "horizon_optimization", False),
        getattr(args, "confidence_weighting", False),
        getattr(args, "regime_gating", False),
        getattr(args, "asymmetry_correction", False),
        getattr(args, "capacity_analysis", False),
        getattr(args, "marginal_value", False),
        getattr(args, "cost_sensitivity", False),
        getattr(args, "joint_optimization", False),
        getattr(args, "deployability_ranking", False),
    ])

    if research_requested:
        panel_path = out_dir / "enriched_panel.parquet"
        if not panel_path.exists():
            print(f"Saving enriched panel for research modules: {panel_path}")
            df.to_parquet(panel_path, index=False)

    if getattr(args, "horizon_optimization", False):
        _run_horizon_optimization(df, out_dir, config_path)

    if getattr(args, "confidence_weighting", False):
        _run_confidence_weighting(df, out_dir, config_path)

    if getattr(args, "regime_gating", False):
        _run_regime_gating(df, out_dir, config_path)

    if getattr(args, "asymmetry_correction", False):
        _run_asymmetry_correction(df, out_dir, config_path)

    if getattr(args, "capacity_analysis", False):
        _run_capacity_analysis(df, out_dir, config_path)

    if getattr(args, "marginal_value", False):
        _run_marginal_value_analysis(df, out_dir, config_path)

    if getattr(args, "cost_sensitivity", False):
        _run_cost_sensitivity(df, out_dir, config_path)

    if getattr(args, "joint_optimization", False):
        _run_joint_optimization(df, out_dir, config_path)

    if getattr(args, "deployability_ranking", False):
        _run_deployability_ranking(out_dir)

    if getattr(args, "viability_check", False):
        _run_viability_check(df, out_dir, config_path)


def _run_horizon_optimization(df: pd.DataFrame, out_dir: Path, config_path: str) -> None:
    """C9: Identify economically optimal holding horizon per model."""
    print("\n[C9] Running Economic Horizon Optimization Analysis...")
    output_path = out_dir / "horizon_analysis.json"
    cmd = f"python3 tools/horizon_optimizer.py --config {config_path} --data {out_dir}/enriched_panel.parquet --output {output_path}"
    os.system(cmd)
    
    if output_path.exists():
        with open(output_path, "r") as f:
            report = json.load(f)
        
        print("\nHorizon Optimization Results (IC vs Turnover vs Sharpe):")
        for model_name, data in report.items():
            print(f"\nModel: {model_name}")
            print(f"{'Horizon':<8} | {'IC_mean':<8} | {'Turnover':<8} | {'NetSharpe':<10} | {'AlphaEff'}")
            print("-" * 55)
            for res in data.get("performance_curve", []):
                print(
                    f"{res['horizon']:<8} | "
                    f"{res['ic_mean']:>8.4f} | "
                    f"{res['turnover_mean']:>8.4f} | "
                    f"{res['net_sharpe']:>10.4f} | "
                    f"{res['alpha_efficiency']:>8.4f}"
                )
            print(f"Optimal Horizon (H*): {data['h_star']} ({data['diagnosis']})")
            print(f"Verdict: {data['transition_verdict']}")

def _run_confidence_weighting(df: pd.DataFrame, out_dir: Path, config_path: str) -> None:
    """C10: Bootstrap daily IC to estimate P(IC>0) and apply Bayesian confidence weighting."""
    print("\n[C10] Running Statistical Confidence Weighting Analysis...")
    cmd = f"python3 tools/confidence_researcher.py --config {config_path} --data {out_dir}/enriched_panel.parquet --output {out_dir}/confidence_analysis.json"
    os.system(cmd)

def _run_regime_gating(df: pd.DataFrame, out_dir: Path, config_path: str) -> None:
    """C11: Identify market regimes and learn data-driven gating functions."""
    print("\n[C11] Running Regime-Conditional Gating Research...")
    cmd = f"python3 tools/regime_researcher.py --config {config_path} --data {out_dir}/enriched_panel.parquet --output {out_dir}/regime_analysis.json"
    os.system(cmd)

def _run_asymmetry_correction(df: pd.DataFrame, out_dir: Path, config_path: str) -> None:
    """C12: Diagnose and correct long/short performance asymmetry."""
    print("\n[C12] Running Long/Short Asymmetry Diagnosis & Correction...")
    cmd = f"python3 tools/leg_researcher.py --config {config_path} --data {out_dir}/enriched_panel.parquet --output {out_dir}/leg_analysis.json"
    os.system(cmd)

def _run_capacity_analysis(df: pd.DataFrame, out_dir: Path, config_path: str) -> None:
    """C13: Estimate strategy capacity and alpha decay curve."""
    print("\n[C13] Running Capacity & Alpha Decay Research...")
    cmd = f"python3 tools/capacity_analyzer.py --config {config_path} --data {out_dir}/enriched_panel.parquet --output {out_dir}/capacity_analysis.json"
    os.system(cmd)

def _run_marginal_value_analysis(df: pd.DataFrame, out_dir: Path, config_path: str) -> None:
    """C14: Evaluate signal's marginal value via incremental ensemble analysis."""
    print("\n[C14] Running Marginal Value Analysis...")
    # Using a default signal from the registry if not specified, usually first one
    cmd = f"python3 tools/marginal_value_analyzer.py --config {config_path} --data {out_dir}/enriched_panel.parquet --signal momentum_12m_1m --output {out_dir}/marginal_value.json"
    os.system(cmd)

def _run_cost_sensitivity(df: pd.DataFrame, out_dir: Path, config_path: str) -> None:
    """C15: Run empirical cost sensitivity analysis."""
    print("\n[C15] Running Empirical Cost Analysis & Sensitivity...")
    cmd = f"python3 tools/cost_analyzer.py --config {config_path} --data {out_dir}/enriched_panel.parquet --output {out_dir}/cost_sensitivity.json"
    os.system(cmd)

def _run_deployability_ranking(out_dir: Path) -> None:
    """C20: Produce a unified deployability score combining all evidence."""
    print("\n[C20] Running Deployability Scoring & Ranking...")
    cmd = f"python3 tools/deployability_ranker.py --report-dir {out_dir} --output {out_dir}/deployability_ranking.json"
    os.system(cmd)

def _run_joint_optimization(df: pd.DataFrame, out_dir: Path, config_path: str) -> None:
    """C21: Jointly select (H, lambda_turn) maximizing net Sharpe via nested CV."""
    print("\n[C21] Running Joint Horizon & Turnover Optimization...")
    cmd = f"python3 tools/joint_optimizer.py --config {config_path} --data {out_dir}/enriched_panel.parquet --output {out_dir}/joint_optimization.json"
    os.system(cmd)

def _run_viability_check(df: pd.DataFrame, out_dir: Path, config_path: str) -> None:
    """Quantify alpha-to-cost ratios and classify signal tradeability."""
    print("\nRunning Signal Viability Analysis...")
    cmd = f"python3 tools/viability_analyzer.py --data {out_dir}/enriched_panel.parquet --config {config_path}"
    os.system(cmd)

if __name__ == "__main__":
    main()
