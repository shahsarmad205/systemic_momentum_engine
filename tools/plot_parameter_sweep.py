from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _prep_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _save_heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    title: str,
    output_path: Path,
    cmap: str = "viridis",
) -> None:
    pivot = df.pivot_table(index=y, columns=x, values=z, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_3d_surface(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    title: str,
    output_path: Path,
) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pivot = df.pivot_table(index=y, columns=x, values=z, aggfunc="mean").sort_index()
    if pivot.empty:
        return

    xx = pivot.columns.to_numpy(dtype=float)
    yy = pivot.index.to_numpy(dtype=float)
    X, Y = np.meshgrid(xx, yy)
    Z = pivot.to_numpy(dtype=float)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", linewidth=0.3, alpha=0.95)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_plots(csv_path: str, output_dir: str, with_3d: bool = True) -> list[Path]:
    df = pd.read_csv(csv_path)
    required = [
        "high_conviction_threshold",
        "portfolio_top_k",
        "net_sharpe",
        "crisis_exposure",
        "vol_kill_switch_cut_factor",
        "crisis_sharpe",
        "normal_exposure",
        "max_drawdown",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in sweep CSV: {missing}")

    df = _prep_numeric(df, required)
    out_dir = Path(output_dir)
    saved: list[Path] = []

    # 1) Net Sharpe vs High Conviction Threshold vs Top-K
    p1 = out_dir / "heatmap_net_sharpe_hct_vs_topk.png"
    _save_heatmap(
        df,
        x="portfolio_top_k",
        y="high_conviction_threshold",
        z="net_sharpe",
        title="Net Sharpe vs High Conviction Threshold vs Top-K",
        output_path=p1,
        cmap="YlGnBu",
    )
    saved.append(p1)
    if with_3d:
        p1_3d = out_dir / "surface_net_sharpe_hct_vs_topk.png"
        _save_3d_surface(
            df,
            x="portfolio_top_k",
            y="high_conviction_threshold",
            z="net_sharpe",
            title="3D Net Sharpe Surface",
            output_path=p1_3d,
        )
        saved.append(p1_3d)

    # 2) Crisis Sharpe vs Crisis Exposure vs Vol Kill Switch Cut Factor
    p2 = out_dir / "heatmap_crisis_sharpe_crisis_exposure_vs_killcut.png"
    _save_heatmap(
        df,
        x="vol_kill_switch_cut_factor",
        y="crisis_exposure",
        z="crisis_sharpe",
        title="Crisis Sharpe vs Crisis Exposure vs Vol Kill Cut",
        output_path=p2,
        cmap="magma",
    )
    saved.append(p2)
    if with_3d:
        p2_3d = out_dir / "surface_crisis_sharpe_crisis_exposure_vs_killcut.png"
        _save_3d_surface(
            df,
            x="vol_kill_switch_cut_factor",
            y="crisis_exposure",
            z="crisis_sharpe",
            title="3D Crisis Sharpe Surface",
            output_path=p2_3d,
        )
        saved.append(p2_3d)

    # 3) Max Drawdown vs Normal Exposure vs Top-K
    p3 = out_dir / "heatmap_max_drawdown_normal_exposure_vs_topk.png"
    _save_heatmap(
        df,
        x="portfolio_top_k",
        y="normal_exposure",
        z="max_drawdown",
        title="Max Drawdown vs Normal Exposure vs Top-K",
        output_path=p3,
        cmap="coolwarm_r",
    )
    saved.append(p3)
    if with_3d:
        p3_3d = out_dir / "surface_max_drawdown_normal_exposure_vs_topk.png"
        _save_3d_surface(
            df,
            x="portfolio_top_k",
            y="normal_exposure",
            z="max_drawdown",
            title="3D Max Drawdown Surface",
            output_path=p3_3d,
        )
        saved.append(p3_3d)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot parameter sweep heatmaps and 3D surfaces.")
    parser.add_argument(
        "--csv",
        type=str,
        default="output/sweeps/parameter_sweep_results_sample.csv",
        help="Path to parameter sweep CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/sweeps/plots",
        help="Directory to save generated plots",
    )
    parser.add_argument(
        "--no-3d",
        action="store_true",
        help="Disable 3D surface plot generation",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    saved = generate_plots(csv_path=args.csv, output_dir=args.output_dir, with_3d=not args.no_3d)
    for path in saved:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
