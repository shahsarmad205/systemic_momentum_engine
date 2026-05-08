from __future__ import annotations

import numpy as np
import pandas as pd


def chained_oos_metrics_reference(
    oos: pd.DataFrame,
    *,
    max_positions: int = 10,
    min_positions: int = 1,
    horizon: int = 1,
    evaluation_path: str = "long_only_overlay",
) -> tuple[float, float, float]:
    """Quarantined reference copy of run_model_selection._chained_oos_metrics."""
    raise RuntimeError(
        "This is quarantined reference material only. Use the active "
        "run_model_selection evaluation/report path instead."
    )


def regime_stability_reference(df: pd.DataFrame, feature: str, target: str, *, sign: int) -> float:
    """Quarantined reference copy of alpha_research._regime_stability."""
    if "regime_label" not in df.columns:
        return 1.0
    regimes = [r for r in sorted(df["regime_label"].dropna().astype(str).unique()) if r]
    if not regimes:
        return 1.0
    positives = 0
    tested = 0
    for regime in regimes:
        g = df[df["regime_label"].astype(str) == str(regime)]
        vals = []
        for _, daily in g.groupby("date", sort=False):
            if daily[feature].nunique(dropna=True) < 2 or daily[target].nunique(dropna=True) < 2:
                continue
            vals.append(float(daily[feature].corr(daily[target], method="spearman")))
        if len(vals) < 5:
            continue
        tested += 1
        if float(sign) * float(np.nanmean(vals)) > 0.0:
            positives += 1
    return float(positives / tested) if tested else 1.0
