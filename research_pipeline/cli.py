"""Pipeline CLI helpers — arg parsing, contract overrides, universe loading."""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

from research_pipeline.contract import ResearchContract


def parse_pipeline_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the research pipeline."""
    parser = argparse.ArgumentParser(description="Research pipeline orchestrator")
    parser.add_argument("--config", type=str, default="backtest_config.yaml")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--limit-tickers", type=int, default=0)
    parser.add_argument("--risk-adj-target", action="store_true", dest="risk_adj_target")
    parser.add_argument("--debug-validation", action="store_true", dest="debug_validation")
    return parser.parse_args(argv)


def cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Build CLI override dict for ResearchContract."""
    o: dict[str, Any] = {}
    if args.horizon is not None:
        o["model_selection"] = {"lookahead_horizon_days": args.horizon}
    if args.limit_tickers:
        o["limit_tickers"] = args.limit_tickers
    if args.risk_adj_target:
        o["risk_adj_target"] = True
    if args.debug_validation:
        o["debug_validation"] = True
    return o


def contract_hash(contract: ResearchContract) -> str:
    """Stable hash of contract for cache invalidation."""
    payload = json.dumps({
        "horizon": contract.horizon.target_horizon_days,
        "start": contract.start_date,
        "end": contract.end_date,
        "features": contract.feature_subset,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_universe(contract: ResearchContract) -> list[str]:
    """Load ticker universe from config."""
    from utils.universe import load_universe
    return load_universe(contract.raw_config)


def build_cost_candidates(candidates, contract) -> list[dict]:
    """Build cost candidate dicts from candidate specs."""
    from model_selection.research_contract import FEATURE_SPECS
    out = []
    for c in candidates:
        spec = FEATURE_SPECS.get(c.active_features[0]) if c.active_features else None
        out.append({
            "candidate_id": c.candidate_id,
            "feature": c.active_features[0] if c.active_features else "unknown",
            "family": spec.family if spec else "unknown",
            "horizon": c.horizon,
            "ic": 0.0,
            "sigma_annual": 0.20,
            "halflife": 0.0,
            "expected_turnover": 0.10,
            "adv_usd": contract.cost_viability.default_adv_usd,
            "daily_vol": contract.cost_viability.default_daily_vol,
        })
    return out
