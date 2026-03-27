#!/usr/bin/env python3
"""
Compare ridge / gbr / xgb weight learning + backtests, pick best by net Sharpe,
copy winner to output/learned_weights.json (and matching *_scaler.json).

Run from repo root (directory containing run_weight_learning.py):
    python scripts/compare_learned_models.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ("ridge", "gbr", "xgb")


def _run(cmd: list[str], *, cwd: Path, timeout: int | None = 14_400) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def _parse_train_ic(stdout: str) -> float | None:
    m = re.search(r"Train IC\s*:\s*([-\d.eE+]+)", stdout)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _parse_wf_ic(stdout: str) -> float | None:
    block = re.search(
        r"Average across folds:\s*\n(.*?)(?:\n\n|\n  Weight stability)",
        stdout,
        re.DOTALL,
    )
    if not block:
        return None
    sub = block.group(1)
    m = re.search(r"Information Coeff\.\s*:\s*([^\s]+)", sub)
    if not m:
        return None
    val = m.group(1).strip()
    if val.upper() == "N/A":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _parse_backtest_metrics(stdout: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "net_sharpe": None,
        "cagr": None,
        "max_dd": None,
        "win_rate": None,
        "total_return": None,
    }
    sm = re.search(
        r"BACKTEST RESULTS SUMMARY\s*\n=+\s*\n(.*?)\n=+",
        stdout,
        re.DOTALL,
    )
    blob = sm.group(1) if sm else stdout
    m = re.search(r"Net Sharpe Ratio\s*:\s*([-\d.]+)", blob)
    if m:
        out["net_sharpe"] = float(m.group(1))
    m = re.search(r"CAGR\s*:\s*([-\d.]+)%", blob)
    if m:
        out["cagr"] = float(m.group(1)) / 100.0
    m = re.search(r"Max Drawdown\s*:\s*([-\d.]+)%", blob)
    if m:
        out["max_dd"] = float(m.group(1)) / 100.0
    m = re.search(r"Win Rate\s*:\s*([-\d.]+)%", blob)
    if m:
        out["win_rate"] = float(m.group(1)) / 100.0
    m = re.search(r"Total Return\s*:\s*([-\d.]+)%", blob)
    if m:
        out["total_return"] = float(m.group(1)) / 100.0
    return out


def main() -> int:
    os.chdir(ROOT)
    py = sys.executable
    rows: list[dict] = []

    for model in MODELS:
        wpath = ROOT / "output" / f"learned_weights_{model}.json"
        spath = ROOT / "output" / f"learned_weights_{model}_scaler.json"
        print(f"\n{'=' * 60}\n  Training: {model}\n{'=' * 60}\n", flush=True)
        r = _run(
            [
                py,
                "run_weight_learning.py",
                "--model",
                model,
                "--target-type",
                "regression",
                "--output",
                str(wpath),
            ],
            cwd=ROOT,
        )
        print(r.stdout, end="" if r.stdout.endswith("\n") else r.stdout + "\n", flush=True)
        if r.stderr:
            print(r.stderr, file=sys.stderr, flush=True)

        train_ic = _parse_train_ic(r.stdout)
        wf_ic = _parse_wf_ic(r.stdout)

        if r.returncode != 0 or not wpath.is_file():
            print(f"  [SKIP] {model} failed or weights missing (code={r.returncode})", flush=True)
            rows.append(
                {
                    "model": model,
                    "wf_ic": wf_ic,
                    "train_ic": train_ic,
                    "net_sharpe": None,
                    "cagr": None,
                    "max_dd": None,
                    "error": f"exit {r.returncode}",
                }
            )
            continue

        print(f"\n  Backtesting with {wpath.name}…\n", flush=True)
        br = _run(
            [
                py,
                "run_backtest.py",
                "--learned-weights",
                str(wpath),
            ],
            cwd=ROOT,
        )
        print(br.stdout, end="" if br.stdout.endswith("\n") else br.stdout + "\n", flush=True)
        if br.stderr:
            print(br.stderr, file=sys.stderr, flush=True)

        metrics = _parse_backtest_metrics(br.stdout) if br.returncode == 0 else {}
        # Save per-model backtest artifacts
        tag = f"_{model}"
        for src_name, dst_name in (
            ("output/backtests/trades.csv", f"output/backtests/trades{tag}.csv"),
            ("output/backtests/daily_equity.csv", f"output/backtests/daily_equity{tag}.csv"),
            ("output/backtests/latest_summary.txt", f"output/backtests/latest_summary{tag}.txt"),
        ):
            src = ROOT / src_name
            dst = ROOT / dst_name
            if src.is_file():
                shutil.copy2(src, dst)

        rows.append(
            {
                "model": model,
                "wf_ic": wf_ic,
                "train_ic": train_ic,
                "net_sharpe": metrics.get("net_sharpe"),
                "cagr": metrics.get("cagr"),
                "max_dd": metrics.get("max_dd"),
                "win_rate": metrics.get("win_rate"),
                "total_return": metrics.get("total_return"),
                "weights": str(wpath),
                "scaler": str(spath) if spath.is_file() else "",
                "error": None if br.returncode == 0 else f"backtest exit {br.returncode}",
            }
        )

    # Pick best by net Sharpe (fallback: WF IC, then train IC)
    valid = [x for x in rows if x.get("net_sharpe") is not None]
    if not valid:
        print("\n[ERROR] No successful backtests; cannot select best model.\n")
        return 1

    def sort_key(x: dict):
        ns = x.get("net_sharpe")
        wf = x.get("wf_ic")
        tr = x.get("train_ic")
        return (
            ns if ns is not None else float("-inf"),
            wf if wf is not None else float("-inf"),
            tr if tr is not None else float("-inf"),
        )

    best = max(valid, key=sort_key)
    best_m = best["model"]
    src_w = ROOT / "output" / f"learned_weights_{best_m}.json"
    src_s = ROOT / "output" / f"learned_weights_{best_m}_scaler.json"
    dst_w = ROOT / "output" / "learned_weights.json"
    dst_s = ROOT / "output" / "learned_weights_scaler.json"

    shutil.copy2(src_w, dst_w)
    if src_s.is_file():
        shutil.copy2(src_s, dst_s)

    # Optional: mirror to Quant-project/output if present (parent of trend_signal_engine)
    parent_out = ROOT.parent / "output"
    if parent_out.is_dir():
        shutil.copy2(dst_w, parent_out / "learned_weights.json")
        if dst_s.is_file():
            shutil.copy2(dst_s, parent_out / "learned_weights_scaler.json")

    out_dir = ROOT / "output" / "learning"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "model_comparison_phase2.md"

    lines = [
        "# Model comparison (ridge / gbr / xgb)",
        "",
        "| Model | WF IC | Train IC | Net Sharpe | CAGR | MaxDD |",
        "|-------|-------|----------|------------|------|-------|",
    ]
    for x in rows:
        def fmt(v: float | None, pct: bool = False) -> str:
            if v is None:
                return ""
            if pct:
                return f"{v:.2%}"
            return f"{v:.4f}" if abs(v) < 100 else f"{v:.2f}"

        lines.append(
            f"| {x['model']} | {fmt(x.get('wf_ic'))} | {fmt(x.get('train_ic'))} | "
            f"{fmt(x.get('net_sharpe'))} | {fmt(x.get('cagr'), True)} | {fmt(x.get('max_dd'), True)} |"
        )
    lines.append(f"\n**Selected (best net Sharpe): `{best_m}`** → `output/learned_weights.json`\n")
    report.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print("  COMPARISON TABLE")
    print("=" * 60)
    print("\n".join(lines))
    print(f"\n  Best model: {best_m}  →  copied to output/learned_weights.json")
    print(f"  Report: {report}")
    print("=" * 60 + "\n")

    # Final backtest with canonical weights path (refresh trades.csv)
    print("  Running final backtest with output/learned_weights.json …\n", flush=True)
    fr = _run([py, "run_backtest.py"], cwd=ROOT)
    print(fr.stdout, flush=True)
    return 0 if fr.returncode == 0 else fr.returncode


if __name__ == "__main__":
    raise SystemExit(main())
