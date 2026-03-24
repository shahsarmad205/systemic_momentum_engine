from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.position_sizing import compose_position_size


def _run_regime(name: str, vol_scaling: float, regime_scaling: float, n: int = 200) -> pd.DataFrame:
    equity = 100_000.0
    rows = []
    for i in range(n):
        # bounded weights in [2%, 18%]
        w = 0.02 + 0.16 * (0.5 + 0.5 * np.sin(i / 12.0))
        size = compose_position_size(
            equity=equity,
            weight=w,
            vol_scaling=vol_scaling,
            regime_scaling=regime_scaling,
            max_single_position_pct=0.12,
            min_single_position_pct=0.01,
            long_only=True,
        )
        rows.append({"regime": name, "t": i, "weight": w, "position_size": size})
    return pd.DataFrame(rows)


def main(save_plot: str) -> None:
    low = _run_regime("LowVol/Bull", vol_scaling=1.4, regime_scaling=1.0, n=200)
    high = _run_regime("HighVol/Bear", vol_scaling=0.45, regime_scaling=0.6, n=200)

    # Regime shift: first half low-vol bull, second half high-vol crisis
    shift_a = _run_regime("Shift-Low", vol_scaling=1.2, regime_scaling=1.0, n=150)
    shift_b = _run_regime("Shift-High", vol_scaling=0.35, regime_scaling=0.3, n=150)
    shift = pd.concat([shift_a, shift_b], ignore_index=True)
    shift["regime"] = "RegimeShift"
    shift["t"] = np.arange(len(shift))

    all_df = pd.concat([low, high, shift], ignore_index=True)

    print("=== Position Sizing Stress Diagnostics ===")
    for reg, g in all_df.groupby("regime"):
        print(
            f"{reg:12s} | mean={g['position_size'].mean():8.2f} "
            f"max={g['position_size'].max():8.2f} "
            f"cap_violations={(g['position_size'] > 12000).sum():3d} "
            f"zeros={(g['position_size'] <= 0).sum():3d}"
        )

    p = Path(save_plot)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    for reg in ["LowVol/Bull", "HighVol/Bear"]:
        g = all_df[all_df["regime"] == reg]
        ax[0].plot(g["t"].to_numpy(), g["position_size"].to_numpy(), label=reg)
    ax[0].axhline(12000, color="black", linestyle="--", linewidth=1, label="12% cap")
    ax[0].set_title("Low vs High Vol regimes")
    ax[0].legend()

    g = all_df[all_df["regime"] == "RegimeShift"]
    ax[1].plot(g["t"].to_numpy(), g["position_size"].to_numpy(), color="tab:purple", label="RegimeShift")
    ax[1].axvline(150, color="red", linestyle="--", linewidth=1, label="Shift point")
    ax[1].axhline(12000, color="black", linestyle="--", linewidth=1)
    ax[1].set_title("Regime shift stress test")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(p, dpi=150)
    print(f"Saved plot: {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress-test multiplicative position sizing.")
    parser.add_argument(
        "--save-plot",
        default="output/backtests/position_sizing_stress.png",
        help="Output PNG path",
    )
    args = parser.parse_args()
    main(args.save_plot)

