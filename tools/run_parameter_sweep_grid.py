from __future__ import annotations

"""
Parameter sweep runner (grid search).

What it does:
1) Generates all parameter combinations with itertools.product
2) Calls run_backtest_with_config(params) for each combination
3) Collects key metrics into a pandas DataFrame
4) Saves results to CSV

Replace `run_backtest_with_config` with your real backtest invocation.
"""

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# -----------------------------
# 1) Parameter grid definition
# -----------------------------
PARAM_GRID: dict[str, list[Any]] = {
    "high_conviction_threshold": [0.50, 0.55, 0.60, 0.65],
    "portfolio_top_k": [4, 5, 6],
    "crisis_portfolio_top_k": [3, 4, 5],
    "normal_exposure": [0.8, 1.0],
    "crisis_exposure": [0.20, 0.25, 0.30],
    "vol_kill_switch_cut_factor": [0.4, 0.5, 0.6],
}


def _valid_combo(params: dict[str, Any]) -> bool:
    """Optional constraints to skip invalid/illogical combinations."""
    if params["crisis_portfolio_top_k"] > params["portfolio_top_k"]:
        return False
    if params["crisis_exposure"] > params["normal_exposure"]:
        return False
    return True


def _run_id(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


# ----------------------------------------------------
# 2) Placeholder backtest function (replace this part)
# ----------------------------------------------------
def _set_param_in_config(raw: dict[str, Any], key: str, value: Any) -> None:
    """
    Set key in first known section that already has it; otherwise set under backtest.
    """
    preferred_sections = ["backtest", "risk", "sectors", "execution", "signals"]
    for sec in preferred_sections:
        node = raw.get(sec)
        if isinstance(node, dict) and key in node:
            node[key] = value
            return
    raw.setdefault("backtest", {})
    raw["backtest"][key] = value


def _extract_metric(pattern: str, text: str) -> float | None:
    m = re.findall(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    try:
        return float(m[-1])
    except Exception:
        return None


def _compute_crisis_sharpe(daily_equity_csv: Path) -> float | None:
    if not daily_equity_csv.exists():
        return None
    e = pd.read_csv(daily_equity_csv)
    if "equity" not in e.columns or "regime" not in e.columns:
        return None
    e["ret"] = pd.to_numeric(e["equity"], errors="coerce").pct_change()
    r = e.loc[e["regime"] == "Crisis", "ret"].dropna()
    if len(r) < 2:
        return 0.0
    sd = float(r.std())
    if not np.isfinite(sd) or sd <= 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(252.0))


def run_backtest_with_config(params: dict[str, Any]) -> dict[str, Any]:
    """
    Temporarily override backtest_config.yaml, run run_backtest.py, then restore config.
    Returns metrics dict suitable for a sweep loop.
    """
    config_path = Path("backtest_config.yaml")
    original_text = config_path.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(original_text) or {}
        for k, v in params.items():
            _set_param_in_config(raw, k, v)
        config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

        proc = subprocess.run(
            ["python", "run_backtest.py"],
            capture_output=True,
            text=True,
            check=False,
        )
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            return {"status": "failed", "error": f"run_backtest.py exited {proc.returncode}"}

        net_sharpe = _extract_metric(r"Net Sharpe Ratio\s*:\s*([-\d\.]+)", combined)
        cagr_pct = _extract_metric(r"CAGR\s*:\s*([-\d\.]+)%", combined)
        max_dd_pct = _extract_metric(r"Max Drawdown\s*:\s*([-\d\.]+)%", combined)

        trades_csv = Path("output/backtests/trades.csv")
        trades_count = 0
        if trades_csv.exists():
            t = pd.read_csv(trades_csv)
            trades_count = int(len(t))

        crisis_sharpe = _compute_crisis_sharpe(Path("output/backtests/daily_equity.csv"))
        return {
            "net_sharpe": net_sharpe,
            "cagr": (cagr_pct / 100.0) if cagr_pct is not None else None,
            "max_drawdown": (max_dd_pct / 100.0) if max_dd_pct is not None else None,
            "crisis_sharpe": crisis_sharpe,
            "trades_count": trades_count,
            "status": "ok",
        }
    finally:
        config_path.write_text(original_text, encoding="utf-8")


def _timeout_worker(queue: mp.Queue, p: dict[str, Any]) -> None:
    """
    Top-level worker required for spawn-based multiprocessing (macOS/Windows).
    """
    try:
        queue.put(run_backtest_with_config(p))
    except Exception as exc:  # pragma: no cover
        queue.put({"status": "failed", "error": str(exc)})


def generate_param_combinations(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(param_grid.keys())
    vals = [param_grid[k] for k in keys]
    combos = []
    for tup in itertools.product(*vals):
        p = dict(zip(keys, tup, strict=False))
        if _valid_combo(p):
            combos.append(p)
    return combos


def run_sweep(
    output_csv: str,
    max_combinations: int | None = None,
    per_run_timeout_sec: int | None = None,
) -> pd.DataFrame:
    combos = generate_param_combinations(PARAM_GRID)
    if max_combinations is not None:
        combos = combos[: max(0, int(max_combinations))]

    rows: list[dict[str, Any]] = []
    try:
        for i, params in enumerate(combos, start=1):
            rid = _run_id(params)
            try:
                if per_run_timeout_sec is None:
                    metrics = run_backtest_with_config(params)
                else:
                    q: mp.Queue = mp.Queue()
                    proc = mp.Process(target=_timeout_worker, args=(q, params))
                    proc.start()
                    proc.join(timeout=float(per_run_timeout_sec))
                    if proc.is_alive():
                        proc.terminate()
                        proc.join()
                        metrics = {
                            "status": "failed",
                            "error": f"timeout after {per_run_timeout_sec}s",
                        }
                    elif not q.empty():
                        metrics = q.get()
                    else:
                        metrics = {"status": "failed", "error": "worker exited without metrics"}

                row = {"run_id": rid, **params, **metrics}
            except Exception as exc:  # pragma: no cover
                row = {"run_id": rid, **params, "status": "failed", "error": str(exc)}
            rows.append(row)
            if i % 25 == 0 or i == len(combos):
                print(f"[{i}/{len(combos)}] completed")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial results...")

    df = pd.DataFrame(rows)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved sweep results: {out} ({len(df)} rows)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parameter grid sweep and save CSV results.")
    parser.add_argument(
        "--output-csv",
        default="output/sweeps/parameter_sweep_results.csv",
        help="Path to output CSV",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs",
    )
    parser.add_argument(
        "--per-run-timeout-sec",
        type=int,
        default=None,
        help="Optional timeout per backtest run; timed-out combos are marked failed",
    )
    args = parser.parse_args()

    run_sweep(
        output_csv=args.output_csv,
        max_combinations=args.max_combinations,
        per_run_timeout_sec=args.per_run_timeout_sec,
    )


if __name__ == "__main__":
    main()

