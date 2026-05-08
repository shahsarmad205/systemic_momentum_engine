#!/usr/bin/env python3
"""Thin research pipeline orchestrator.

Responsibility: Parse args, load config, call stages in order, write manifest.
No feature math, target math, cost math, promotion gate math, or cache internals.
All assumptions come from ResearchContract.
"""
from __future__ import annotations

import logging

from research_pipeline.contract import ResearchContract
from research_pipeline.artifact_manifest import create_manifest
from research_pipeline.cli import parse_pipeline_args, cli_overrides, contract_hash, load_universe, build_cost_candidates


def main() -> None:
    args = parse_pipeline_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    contract = ResearchContract.from_yaml(args.config, cli_overrides=cli_overrides(args))
    logging.info("Contract: horizon=%dd, %s to %s", contract.horizon.target_horizon_days,
                 contract.start_date, contract.end_date)

    manifest = create_manifest()
    manifest.contract_hash = contract_hash(contract)

    tickers = load_universe(contract)
    if contract.limit_tickers > 0:
        tickers = tickers[: contract.limit_tickers]

    from research_pipeline.data_loader import load_feature_matrix
    df = load_feature_matrix(contract, tickers)
    manifest.add_artifact("feature_matrix", "output/models/feature_matrix.parquet", "data_loader", len(df))

    from research_pipeline.schema_validation import validate_schema
    from research_pipeline.report_writer import write_schema_validation_report
    cols = [c for c in df.columns if c not in {"date", "ticker"}]
    rpt = validate_schema(df, cols)
    p = "output/research_pipeline/schema_validation_report.json"
    write_schema_validation_report(rpt, p)
    manifest.add_artifact("schema_validation", p, "schema_validation")

    from research_pipeline.feature_registry import build_feature_registry
    from research_pipeline.report_writer import write_feature_registry_report
    reg = build_feature_registry(cols, contract.horizon.target_horizon_days,
                                  contract.horizon.horizon_alignment_multiplier)
    p = "output/research_pipeline/feature_registry_report.csv"
    write_feature_registry_report(reg, p)
    manifest.add_artifact("feature_registry", p, "feature_registry", reg.n_families)

    from research_pipeline.target_builder import build_targets
    from research_pipeline.report_writer import write_target_build_report
    df, rpt = build_targets(df, contract)
    p = "output/research_pipeline/target_build_report.json"
    write_target_build_report(rpt, p)
    manifest.add_artifact("target_build", p, "target_builder")

    from research_pipeline.feature_admission import run_feature_admission
    from research_pipeline.report_writer import write_feature_admission_report
    active, results = run_feature_admission(df, cols, contract)
    p = "output/research_pipeline/feature_admission_report.csv"
    write_feature_admission_report(results, p)
    manifest.add_artifact("feature_admission", p, "feature_admission", len(results))

    from research_pipeline.horizon_diagnostics import compute_horizon_diagnostics
    from research_pipeline.report_writer import write_horizon_diagnostics
    diags = compute_horizon_diagnostics(df, active, contract)
    p = "output/research_pipeline/horizon_diagnostics.csv"
    write_horizon_diagnostics(diags, p)
    manifest.add_artifact("horizon_diagnostics", p, "horizon_diagnostics", len(diags))

    from research_pipeline.candidate_factory import build_model_specs, build_candidate_pool
    from research_pipeline.report_writer import write_candidate_factory_report
    specs = build_model_specs(contract)
    candidates = build_candidate_pool(specs, active, contract)
    p = "output/research_pipeline/candidate_factory_report.csv"
    write_candidate_factory_report(candidates, p)
    manifest.add_artifact("candidate_factory", p, "candidate_factory", len(candidates))

    from research_pipeline.cost_viability_adapter import batch_evaluate_cost_viability
    from research_pipeline.report_writer import write_cost_viability_scorecard
    cost_cands = build_cost_candidates(candidates, contract)
    cost_results = batch_evaluate_cost_viability(cost_cands, contract)
    p = "output/research_pipeline/cost_viability_scorecard.csv"
    write_cost_viability_scorecard(cost_results, p)
    manifest.add_artifact("cost_viability", p, "cost_viability", len(cost_results))

    manifest.complete()
    manifest.write("output/research_pipeline/artifact_manifest.json")
    logging.info("Pipeline complete: %d artifacts in %.1fs", len(manifest.artifacts), manifest.duration_seconds)

    if any(a.status == "error" for a in manifest.artifacts):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
