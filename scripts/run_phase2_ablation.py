#!/usr/bin/env python3
"""
Phase 2 feature ablation (Ridge) + optional Ridge vs GBR on the best step.

Uses TSE_ABLATION_STEP (see agents/weight_learning_agent/feature_flags.py).

Usage (from trend_signal_engine/):
    python scripts/run_phase2_ablation.py
    python scripts/run_phase2_ablation.py --steps 0,6          # smoke: baseline + full only
    python scripts/run_phase2_ablation.py --compare-models     # after table, train ridge+gbr on best step

Outputs:
    output/learning/phase2_ablation_table.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Allow `python scripts/run_phase2_ablation.py` (repo root must be on path for `agents`)
_root_str = str(ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

from agents.weight_learning_agent.feature_flags import describe_ablation_step


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(f"\n  $ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _parse_wf_ic(text: str) -> float | None:
    m = re.search(r"WF mean IC\s*:\s*([-\d.]+|N/A)", text)
    if not m or m.group(1) == "N/A":
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _parse_bt(path: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {"net_sharpe": None, "win_rate": None, "cagr": None}
    if not path.is_file():
        return out
    t = path.read_text(encoding="utf-8")
    m = re.search(r"Net Sharpe Ratio\s*:\s*([-\d.]+)", t)
    if m:
        out["net_sharpe"] = float(m.group(1))
    m = re.search(r"Win Rate\s*:\s*([-\d.]+)%", t)
    if m:
        out["win_rate"] = float(m.group(1)) / 100.0
    m = re.search(r"CAGR\s*:\s*([-\d.]+)%", t)
    if m:
        out["cagr"] = float(m.group(1)) / 100.0
    return out


def _n_active_features() -> int:
    p = ROOT / "output" / "learned_weights_scaler.json"
    if not p.is_file():
        return 0
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return len(d.get("active_features") or [])
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--steps",
        default="0,1,2,3,4,5,6",
        help="Comma-separated ablation steps (default: full ladder)",
    )
    ap.add_argument(
        "--compare-models",
        action="store_true",
        help="After ablation, train Ridge and GBR on the best Net Sharpe step and print comparison",
    )
    args = ap.parse_args()
    steps = [int(x.strip()) for x in args.steps.split(",") if x.strip()]

    os.chdir(ROOT)
    py = sys.executable
    prev_backup = os.environ.get("TSE_ABLATION_STEP")

    rows: list[dict] = []
    prev_sharpe: float | None = None
    best_step = 0
    best_sharpe = -1e9

    try:
        for step in steps:
            step = max(0, min(6, step))
            os.environ["TSE_ABLATION_STEP"] = str(step)

            label = describe_ablation_step(step)
            print(f"\n{'=' * 70}\n  ABLATION STEP {step}: {label}\n{'=' * 70}", flush=True)

            _run(
                [py, "run_weight_learning.py", "--model", "ridge", "--target-type", "regression"],
                cwd=ROOT,
            )
            summ_path = ROOT / "output" / "learning" / "latest_learning_summary.txt"
            wf_ic = _parse_wf_ic(summ_path.read_text(encoding="utf-8")) if summ_path.is_file() else None

            wpath = ROOT / "output" / "learned_weights.json"
            train_ic = None
            if wpath.is_file():
                train_ic = float(json.loads(wpath.read_text(encoding="utf-8")).get("ic") or 0.0)

            _run([py, "run_backtest.py"], cwd=ROOT)
            bt = _parse_bt(ROOT / "output" / "backtests" / "latest_summary.txt")
            ns = bt["net_sharpe"]
            nfeat = _n_active_features()

            keep = "—"
            if ns is not None:
                if ns > best_sharpe + 1e-9:
                    best_sharpe = ns
                    best_step = step
                if prev_sharpe is None:
                    keep = "baseline"
                elif ns + 1e-9 >= prev_sharpe:
                    keep = "KEEP"
                else:
                    keep = "REGRESS"
                prev_sharpe = ns

            rows.append(
                {
                    "step": step,
                    "label": label,
                    "nfeat": nfeat,
                    "wf_ic": wf_ic,
                    "train_ic": train_ic,
                    "net_sharpe": ns,
                    "win_rate": bt["win_rate"],
                    "cagr": bt["cagr"],
                    "keep": keep,
                }
            )
    finally:
        if prev_backup is None:
            os.environ.pop("TSE_ABLATION_STEP", None)
        else:
            os.environ["TSE_ABLATION_STEP"] = prev_backup

    # Write markdown table
    out_md = ROOT / "output" / "learning" / "phase2_ablation_table.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 2 Ridge ablation",
        "",
        "| Step | Group | n active | WF IC | Train IC | Net Sharpe | Win% | CAGR | vs prev |",
        "|------|-------|----------|-------|----------|------------|------|------|---------|",
    ]
    for r in rows:
        wf = f"{r['wf_ic']:.4f}" if r["wf_ic"] is not None else ""
        tr = f"{r['train_ic']:.4f}" if r["train_ic"] is not None else ""
        ns = f"{r['net_sharpe']:.3f}" if r["net_sharpe"] is not None else ""
        wr = f"{r['win_rate']*100:.1f}%" if r["win_rate"] is not None else ""
        cg = f"{r['cagr']*100:.2f}%" if r["cagr"] is not None else ""
        lines.append(
            f"| {r['step']} | {r['label'][:40]} | {r['nfeat']} | {wf} | {tr} | {ns} | {wr} | {cg} | {r['keep']} |"
        )
    lines.append("")
    lines.append(f"**Best Net Sharpe step:** `{best_step}` (sharpe={best_sharpe:.3f})")
    lines.append("")
    lines.append("Re-train with that step: `export TSE_ABLATION_STEP=" + str(best_step) + "` then `python run_weight_learning.py --model ridge ...`")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines) + "\n")
    print(f"  Wrote → {out_md}")

    if args.compare_models:
        os.environ["TSE_ABLATION_STEP"] = str(best_step)
        try:
            print("\n--- Ridge (best step) ---\n", flush=True)
            _run([py, "run_weight_learning.py", "--model", "ridge", "--target-type", "regression"], cwd=ROOT)
            ridge_train_ic = None
            ridge_wf_ic = None
            sp = ROOT / "output" / "learning" / "latest_learning_summary.txt"
            if sp.is_file():
                ridge_wf_ic = _parse_wf_ic(sp.read_text(encoding="utf-8"))
            wp = ROOT / "output" / "learned_weights.json"
            if wp.is_file():
                ridge_train_ic = float(json.loads(wp.read_text(encoding="utf-8")).get("ic") or 0.0)
            path_ridge = ROOT / "output" / "learned_weights_ablation_ridge.json"
            path_gbr = ROOT / "output" / "learned_weights_ablation_gbr.json"

            shutil.copy2(wp, path_ridge)
            shutil.copy2(
                ROOT / "output" / "learned_weights_scaler.json",
                path_ridge.with_name(path_ridge.stem + "_scaler.json"),
            )

            _run([py, "run_backtest.py", "--learned-weights", str(path_ridge)], cwd=ROOT)
            bt_r = _parse_bt(ROOT / "output" / "backtests" / "latest_summary.txt")

            print("\n--- GBR (best step) ---\n", flush=True)
            _run([py, "run_weight_learning.py", "--model", "gbr", "--target-type", "regression"], cwd=ROOT)
            gbr_train = None
            if wp.is_file():
                gbr_train = float(json.loads(wp.read_text(encoding="utf-8")).get("ic") or 0.0)
            sp = ROOT / "output" / "learning" / "latest_learning_summary.txt"
            wf_g = _parse_wf_ic(sp.read_text(encoding="utf-8")) if sp.is_file() else None
            shutil.copy2(wp, path_gbr)
            shutil.copy2(
                ROOT / "output" / "learned_weights_scaler.json",
                path_gbr.with_name(path_gbr.stem + "_scaler.json"),
            )
            _run([py, "run_backtest.py", "--learned-weights", str(path_gbr)], cwd=ROOT)
            bt_g = _parse_bt(ROOT / "output" / "backtests" / "latest_summary.txt")

            gap_r = (
                (ridge_train_ic or 0) - (ridge_wf_ic or 0)
                if ridge_train_ic is not None and ridge_wf_ic is not None
                else None
            )
            gap_g = (
                (gbr_train or 0) - (wf_g or 0) if gbr_train is not None and wf_g is not None else None
            )

            print("\n=== STEP 3: Ridge vs GBR (best ablation step) ===\n")
            print(
                f"  Ridge: train IC={ridge_train_ic}  WF IC={ridge_wf_ic}  gap={gap_r}  Net Sharpe={bt_r['net_sharpe']}"
            )
            print(f"  GBR:   train IC={gbr_train}  WF IC={wf_g}  gap={gap_g}  Net Sharpe={bt_g['net_sharpe']}")
            print(
                "\n  Pick: higher WF IC, IC gap < 0.10, higher Net Sharpe — then copy winner to output/learned_weights.json\n"
            )
        finally:
            if prev_backup is None:
                os.environ.pop("TSE_ABLATION_STEP", None)
            else:
                os.environ["TSE_ABLATION_STEP"] = prev_backup

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
