from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def select_top_configs(
    csv_path: str,
    *,
    min_net_sharpe: float = 1.4,
    min_crisis_sharpe: float = 0.5,
    max_drawdown_abs: float = 0.07,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Select sweep rows satisfying:
      - net_sharpe > min_net_sharpe
      - crisis_sharpe > min_crisis_sharpe
      - |max_drawdown| < max_drawdown_abs

    Note: max_drawdown in this project CSVs is typically stored as negative decimal,
    e.g., -0.068 for -6.8%.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        sweep_dir = Path("output/sweeps")
        available = sorted([p.name for p in sweep_dir.glob("*.csv")]) if sweep_dir.exists() else []
        msg = f"Sweep CSV not found: {csv_file}"
        if available:
            msg += f". Available CSVs in output/sweeps: {available}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(csv_file)

    needed = ["net_sharpe", "crisis_sharpe", "max_drawdown"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
        if df.empty:
            raise ValueError(
                "No successful rows found (status == 'ok'). "
                "Check the sweep CSV 'error' column for failures."
            )

    filtered = df[
        (df["net_sharpe"] > min_net_sharpe)
        & (df["crisis_sharpe"] > min_crisis_sharpe)
        & (df["max_drawdown"].abs() < max_drawdown_abs)
    ].copy()

    filtered = filtered.sort_values("net_sharpe", ascending=False).head(top_n)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter parameter sweep CSV and show top configurations."
    )
    parser.add_argument(
        "--csv",
        default="output/sweeps/parameter_sweep_results.csv",
        help="Path to sweep CSV",
    )
    parser.add_argument("--min-net-sharpe", type=float, default=1.4)
    parser.add_argument("--min-crisis-sharpe", type=float, default=0.5)
    parser.add_argument(
        "--max-drawdown-pct",
        type=float,
        default=7.0,
        help="Max allowed drawdown in percent points (absolute), e.g. 7 for 7%%",
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to save filtered top configs",
    )
    args = parser.parse_args()

    top = select_top_configs(
        args.csv,
        min_net_sharpe=args.min_net_sharpe,
        min_crisis_sharpe=args.min_crisis_sharpe,
        max_drawdown_abs=args.max_drawdown_pct / 100.0,
        top_n=args.top_n,
    )

    if top.empty:
        print("No configurations matched the constraints.")
        return

    cols = [c for c in top.columns if c not in {"error"}]
    print(top[cols].to_string(index=False))

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        top.to_csv(out, index=False)
        print(f"\nSaved filtered configs: {out}")


if __name__ == "__main__":
    main()
