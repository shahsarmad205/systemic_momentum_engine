from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from utils.vol_sizing import compute_realized_vol_annualized, compute_vol_target_scaling_factor
except ModuleNotFoundError:  # pragma: no cover
    from trend_signal_engine.utils.vol_sizing import (  # type: ignore[no-redef]
        compute_realized_vol_annualized,
        compute_vol_target_scaling_factor,
    )


def _simulate_regime_returns(n: int, daily_sigma: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(loc=0.0, scale=daily_sigma, size=n), dtype=float)


def run_simulation(
    target_vol: float = 0.15,
    window: int = 20,
    annualization: int = 252,
    min_vol_floor: float = 0.05,
    max_scale_cap: float = 3.0,
    save_plot: str | None = None,
) -> dict[str, float]:
    # Synthetic regimes
    low = _simulate_regime_returns(300, daily_sigma=0.005, seed=42)
    high = _simulate_regime_returns(300, daily_sigma=0.020, seed=43)
    shift = pd.concat(
        [
            _simulate_regime_returns(150, daily_sigma=0.006, seed=44),
            _simulate_regime_returns(150, daily_sigma=0.020, seed=45),
        ],
        ignore_index=True,
    )
    returns = pd.concat([low, high, shift], ignore_index=True)

    realized = compute_realized_vol_annualized(returns, window=window, annualization=annualization)
    scaler = realized.apply(
        lambda v: compute_vol_target_scaling_factor(
            v,
            target_vol=target_vol,
            min_vol_floor=min_vol_floor,
            max_scale_cap=max_scale_cap,
        )
    )
    targeted_returns = returns * scaler
    targeted_vol = compute_realized_vol_annualized(targeted_returns, window=window, annualization=annualization)

    obs = targeted_vol.dropna()
    out = {
        "target_vol": float(target_vol),
        "realized_targeted_vol_mean": float(obs.mean()),
        "realized_targeted_vol_median": float(obs.median()),
        "realized_targeted_vol_std": float(obs.std()),
        "scale_min": float(scaler.min()),
        "scale_max": float(scaler.max()),
        "scale_mean": float(scaler.mean()),
    }

    print("=== Vol Targeting Diagnostic (Synthetic) ===")
    print(f"Target annual vol              : {target_vol:.3f}")
    print(f"Mean realized vol (targeted)   : {out['realized_targeted_vol_mean']:.3f}")
    print(f"Median realized vol (targeted) : {out['realized_targeted_vol_median']:.3f}")
    print(f"Std realized vol (targeted)    : {out['realized_targeted_vol_std']:.3f}")
    print(f"Scaling factor min/max/mean    : {out['scale_min']:.3f} / {out['scale_max']:.3f} / {out['scale_mean']:.3f}")

    if save_plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        ax[0].plot(realized.values, label="Raw realized vol", alpha=0.8)
        ax[0].plot(targeted_vol.values, label="Targeted realized vol", alpha=0.9)
        ax[0].axhline(target_vol, color="black", linestyle="--", linewidth=1, label="Target")
        ax[0].set_title("Rolling annualized volatility")
        ax[0].legend()

        ax[1].plot(scaler.values, color="tab:purple", label="Scaling factor")
        ax[1].set_title("Vol targeting scaling factor")
        ax[1].legend()
        fig.tight_layout()

        out_path = Path(save_plot)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved diagnostic plot: {out_path}")

    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simulate volatility targeting diagnostics.")
    p.add_argument("--target-vol", type=float, default=0.15)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--annualization", type=int, default=252)
    p.add_argument("--min-vol-floor", type=float, default=0.05)
    p.add_argument("--max-scale-cap", type=float, default=3.0)
    p.add_argument("--save-plot", type=str, default="output/backtests/vol_targeting_diagnostic.png")
    args = p.parse_args()

    run_simulation(
        target_vol=args.target_vol,
        window=args.window,
        annualization=args.annualization,
        min_vol_floor=args.min_vol_floor,
        max_scale_cap=args.max_scale_cap,
        save_plot=args.save_plot,
    )

