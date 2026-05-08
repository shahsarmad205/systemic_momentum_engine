from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import FrameType, SimpleNamespace
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANDIDATE_NAMES = {
    "_model_filename",
    "_count_invested_days",
    "_chained_oos_metrics",
    "_is_economic_candidate_pool",
    "orientation_policy_config",
    "_regime_stability",
    "_pairwise_signal_correlation",
    "_run_optional_research_pillars",
    "_run_horizon_optimization",
    "_run_confidence_weighting",
    "_test_portfolio_simulation_logic",
}


def _repo_relative(path: str) -> str | None:
    try:
        resolved = Path(path).resolve()
        return str(resolved.relative_to(ROOT))
    except Exception:
        return None


def _trace_callable(path_name: str, fn: Callable[[], Any]) -> dict[str, Any]:
    calls: Counter[tuple[str, str, int, str]] = Counter()
    callers: dict[tuple[str, str, int, str], Counter[str]] = defaultdict(Counter)
    started = time.perf_counter()
    status = "ok"
    error = ""
    exit_code: int | None = None

    def profiler(frame: FrameType, event: str, arg: Any) -> Callable[..., Any] | None:
        if event != "call":
            return profiler
        rel = _repo_relative(frame.f_code.co_filename)
        if rel is None or rel.startswith("dev_tools/trace_run_model_selection.py"):
            return profiler
        module = str(frame.f_globals.get("__name__", ""))
        func = frame.f_code.co_name
        key = (rel, func, int(frame.f_code.co_firstlineno), module)
        calls[key] += 1
        parent = frame.f_back
        if parent is not None:
            parent_rel = _repo_relative(parent.f_code.co_filename)
            if parent_rel is not None:
                parent_module = str(parent.f_globals.get("__name__", ""))
                callers[key][f"{parent_module}.{parent.f_code.co_name}:{parent.f_code.co_firstlineno}"] += 1
        return profiler

    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        fn()
    except SystemExit as exc:
        status = "system_exit"
        exit_code = int(exc.code) if isinstance(exc.code, int) else 0
    except Exception as exc:  # trace harness should record blocked paths.
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        sys.setprofile(previous)

    records = [
        {
            "file": file,
            "function": func,
            "line": line,
            "module": module,
            "calls": count,
            "callers": [
                {"caller": caller, "calls": c}
                for caller, c in caller_counts.most_common(5)
            ],
        }
        for (file, func, line, module), count in calls.most_common()
        for caller_counts in [callers.get((file, func, line, module), Counter())]
    ]
    modules = sorted({record["module"] for record in records if record["module"]})
    candidate_hits = {
        name: sum(record["calls"] for record in records if record["function"] == name)
        for name in sorted(CANDIDATE_NAMES)
    }
    return {
        "path": path_name,
        "status": status,
        "exit_code": exit_code,
        "error": error,
        "duration_s": round(time.perf_counter() - started, 6),
        "modules_reached": modules,
        "candidate_hits": candidate_hits,
        "calls": records,
    }


def _path_cli_help() -> None:
    import run_model_selection as rms

    old_argv = sys.argv[:]
    try:
        sys.argv = ["run_model_selection.py", "--help"]
        rms.main()
    finally:
        sys.argv = old_argv


def _path_sim_smoke() -> None:
    import run_model_selection as rms

    old_argv = sys.argv[:]
    try:
        sys.argv = ["run_model_selection.py", "--run_sim_test"]
        rms.main()
    finally:
        sys.argv = old_argv


def _path_research_routes() -> None:
    import pandas as pd
    import run_model_selection as rms

    old_system = rms.os.system
    try:
        rms.os.system = lambda _cmd: 0
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            (out_dir / "enriched_panel.parquet").write_text("trace-placeholder", encoding="utf-8")
            args = SimpleNamespace(
                horizon_optimization=True,
                confidence_weighting=True,
                regime_gating=False,
                asymmetry_correction=False,
                capacity_analysis=False,
                marginal_value=False,
                cost_sensitivity=False,
                joint_optimization=False,
                deployability_ranking=False,
                viability_check=False,
            )
            rms._run_optional_research_pillars(pd.DataFrame({"x": [1.0]}), out_dir, args, "tests/mock_backtest_config.yaml")
    finally:
        rms.os.system = old_system


def _synthetic_panel() -> Any:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(13)
    dates = pd.bdate_range("2020-01-01", periods=50)
    tickers = [f"T{i:02d}" for i in range(12)]
    rows: list[dict[str, Any]] = []
    for i, dt in enumerate(dates):
        for j, ticker in enumerate(tickers):
            signal = float(j - len(tickers) / 2) + rng.normal(0.0, 0.1)
            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "sector": "Tech" if j % 2 else "Industrials",
                    "sector_asof": "Tech" if j % 2 else "Industrials",
                    "regime_label": "Bull" if i < len(dates) // 2 else "Normal",
                    "daily_return": 0.0005 * signal + rng.normal(0.0, 0.001),
                    "forward_return": 0.002 * signal + rng.normal(0.0, 0.002),
                    "alpha_signal": signal,
                    "adv_dollar_20": 50_000_000.0,
                    "realised_vol_20d": 0.02,
                    "capm_beta": 1.0,
                    "market_cap": 1_000_000_000.0 + j,
                    "rolling_vol_20": 0.20 + 0.01 * (j % 3),
                }
            )
    return pd.DataFrame(rows)


def _path_alpha_research() -> None:
    from model_selection.alpha_research import AlphaAdmissionConfig, run_alpha_research
    from model_selection.training import TargetConfig
    from model_selection.validation import ExecutionCostConfig

    run_alpha_research(
        _synthetic_panel(),
        ["alpha_signal"],
        cfg=AlphaAdmissionConfig(
            horizons=(1, 2),
            production_horizon=1,
            min_coverage=0.20,
            min_abs_ic=0.0,
            min_ic_tstat=-99.0,
            min_monotonicity=0.0,
            min_regime_stability=0.0,
            min_marginal_abs_ic=0.0,
        ),
        target_cfg=TargetConfig(horizon_days=1),
        costs=ExecutionCostConfig(),
        max_name_weight=0.10,
    )


def _path_diagnostics_engines() -> None:
    from model_selection.conditional_alpha_engine import ConditionalAlphaEngine
    from model_selection.cost_viability_wiring import evaluate_feature_cost_viability
    from model_selection.feature_diversity_wiring import evaluate_feature_diversity
    from model_selection.ic_diagnostics_engine import ICDiagnosticsEngine
    from model_selection.pit_condition_engine import PITConditionEngine

    import pandas as pd

    panel = _synthetic_panel()
    features = ["alpha_signal"]
    cfg = {
        "model_selection": {
            "ic_diagnostics": {
                "horizons": [1],
                "min_dates_for_ic": 5,
                "min_breadth_for_ic": 3,
                "conditional_conditions": ["regime", "sector"],
            },
            "conditional_alpha": {
                "horizons": [1],
                "condition_types": ["regime", "sector"],
                "pit_regime_min_obs": 5,
            },
            "pit_conditions": {
                "condition_types": ["regime", "sector"],
                "regime_min_obs": 5,
                "expanding_min_obs": 5,
                "min_breadth_per_date": 3,
            },
            "feature_diversity": {"horizons": [1]},
        }
    }
    ICDiagnosticsEngine(cfg).run_full_diagnostics(panel, features, horizons=[1])
    ConditionalAlphaEngine(cfg).run_full_validation(panel, features, horizons=[1])
    PITConditionEngine(cfg).run_full_pit_validation(panel, features, horizons=[1])
    evaluate_feature_diversity(panel, features, cfg, horizons=[1])
    admission = pd.DataFrame(
        [{"feature": "alpha_signal", "admitted": True, "mean_ic": 0.02, "ic_tstat": 2.0, "signal_halflife_days": 5.0, "turnover_mean": 0.1}]
    )
    decay = pd.DataFrame([{"feature": "alpha_signal", "horizon_days": 1, "mean_ic": 0.02}])
    evaluate_feature_cost_viability(admission, decay, panel, cfg, horizon=1)


def _path_full_mock_run() -> None:
    import run_model_selection as rms

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "run_model_selection.py",
            "--config",
            "tests/mock_backtest_config.yaml",
            "--limit_tickers",
            "3",
            "--horizon",
            "5",
            "--min_oos_days",
            "1",
            "--min_test_days",
            "1",
            "--simplified",
            "--log-level",
            "ERROR",
        ]
        rms.main()
    finally:
        sys.argv = old_argv


PATHS: dict[str, Callable[[], Any]] = {
    "cli_help": _path_cli_help,
    "sim_smoke": _path_sim_smoke,
    "research_routes": _path_research_routes,
    "alpha_research": _path_alpha_research,
    "diagnostics_engines": _path_diagnostics_engines,
    "full_mock_run": _path_full_mock_run,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace representative run_model_selection paths.")
    parser.add_argument("--path", action="append", choices=sorted(PATHS), help="Path to trace; repeatable. Defaults to all.")
    parser.add_argument("--output", type=Path, default=Path("audit/runtime_trace_phase_a2.json"))
    args = parser.parse_args()

    selected = args.path or list(PATHS)
    results = [_trace_callable(name, PATHS[name]) for name in selected]
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": str(ROOT),
        "paths": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
