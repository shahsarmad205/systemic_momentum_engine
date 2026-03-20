"""
Feature correlation auditing and redundancy reduction utilities.

These helpers compute correlation structure across a feature matrix,
suggest which features to drop (while always preserving protected
columns), and optionally render a correlation heatmap for inspection.
"""

from __future__ import annotations

import logging

from typing import Dict, List

import numpy as np
import pandas as pd

logging.getLogger("matplotlib").setLevel(logging.WARNING)


# Columns that must never be dropped automatically, even if highly correlated.
PROTECTED_COLUMNS: List[str] = [
    "daily_return",
    "momentum_3m",
    "momentum_6m",
    "ma_50",
    "ma_200",
    "ma_crossover_signal",
    "rolling_vol_20",
    "relative_volume",
    "volume_zscore",
    "trend_score",
    "adjusted_score",
]


def compute_feature_correlation_report(
    feature_matrix: pd.DataFrame,
    threshold: float = 0.85,
) -> pd.DataFrame:
    """
    Compute pairwise feature correlations and recommendations.

    Financial intuition
    -------------------
    - Highly correlated features (|ρ| > threshold) add little new
      information but increase estimation noise for models like Ridge.
    - We prefer to keep a canonical representative (especially if it
      is one of the PROTECTED_COLUMNS) and consider dropping the
      redundant one.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Feature DataFrame; non-numeric columns are ignored.
    threshold : float, default 0.85
        Absolute correlation above which a pair is flagged.

    Returns
    -------
    pd.DataFrame
        Columns:
            feature_a, feature_b, correlation, recommendation
        Recommendation is one of:
            'drop_b'     — prefer feature_a (usually protected); drop b.
            'drop_a'     — prefer feature_b; drop a.
            'keep_both'  — both protected; flagged for manual review.
            'manual_review' — suspicious but < 0.95 absolute corr.
    """
    num = feature_matrix.select_dtypes(include=[np.number])
    num = num.dropna(axis=1, how="all")
    cols = list(num.columns)
    if len(cols) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "correlation", "recommendation"])

    corr = num.corr()
    records = []

    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            c = corr.loc[a, b]
            if pd.isna(c):
                continue
            ac = abs(float(c))
            if ac <= threshold:
                continue

            if ac >= 0.95:
                a_prot = a in PROTECTED_COLUMNS
                b_prot = b in PROTECTED_COLUMNS
                if a_prot and not b_prot:
                    rec = "drop_b"
                elif b_prot and not a_prot:
                    rec = "drop_a"
                elif not a_prot and not b_prot:
                    rec = "drop_b"  # arbitrary but deterministic
                else:
                    rec = "keep_both"
            else:
                rec = "manual_review"

            records.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "correlation": float(c),
                    "recommendation": rec,
                }
            )

    return pd.DataFrame.from_records(
        records,
        columns=["feature_a", "feature_b", "correlation", "recommendation"],
    )


def get_low_redundancy_features(
    feature_matrix: pd.DataFrame,
    protected_cols: List[str],
    threshold: float = 0.85,
) -> List[str]:
    """
    Return a list of feature columns with redundant ones removed.

    Logic
    -----
    - Runs ``compute_feature_correlation_report`` to find high-corr pairs.
    - For each high-corr pair, if one column is not protected, mark it
      for dropping; never drop any column in ``protected_cols`` or
      ``PROTECTED_COLUMNS``.
    - Returns a sorted list of columns to keep.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Input feature matrix.
    protected_cols : list[str]
        Additional columns that must never be dropped.
    threshold : float, default 0.85
        Correlation threshold passed through to the report.

    Returns
    -------
    list[str]
        Column names to keep.
    """
    report = compute_feature_correlation_report(feature_matrix, threshold=threshold)
    all_protected = set(PROTECTED_COLUMNS) | set(protected_cols or [])

    to_drop: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    # Process pairs grouped by unordered (a,b) key to avoid double decisions.
    by_key: Dict[tuple, Dict] = {}
    for _, row in report.iterrows():
        a = row["feature_a"]
        b = row["feature_b"]
        ac = abs(float(row["correlation"]))
        if ac <= threshold:
            continue
        key = tuple(sorted((a, b)))
        by_key.setdefault(key, row)

    for (a, b), row in by_key.items():
        ac = abs(float(row["correlation"]))
        if ac <= threshold:
            continue

        if a in to_drop or b in to_drop:
            continue

        a_prot = a in all_protected
        b_prot = b in all_protected

        if a_prot and b_prot:
            # Both protected: keep; flag only in the report.
            continue
        if (a_prot and not b_prot) or (b_prot and not a_prot):
            # Correlation with a protected feature is for diagnostics only;
            # do not auto-drop the non-protected side here.
            continue
        else:
            # Neither protected: drop the feature with the smaller variance
            var_a = float(feature_matrix[a].var())
            var_b = float(feature_matrix[b].var())
            drop = b if var_b <= var_a else a
            to_drop.add(drop)

    keep = [c for c in feature_matrix.columns if c not in to_drop]
    return sorted(keep)


def plot_correlation_heatmap(
    feature_matrix: pd.DataFrame,
    output_path: str = "output/feature_correlation.png",
) -> None:
    """
    Plot a correlation heatmap of numeric features and save to disk.

    Cells with |ρ| > 0.85 are annotated in red for quick inspection.
    Uses seaborn when available; otherwise falls back to pure
    matplotlib. Does not call ``plt.show()``.
    """
    import warnings

    num = feature_matrix.select_dtypes(include=[np.number])
    num = num.dropna(axis=1, how="all")
    if num.empty:
        warnings.warn("plot_correlation_heatmap: no numeric features to plot.", UserWarning)
        return

    corr = num.corr()

    try:
        import matplotlib.pyplot as plt
    except Exception:
        warnings.warn("plot_correlation_heatmap: matplotlib not available; skipping plot.", UserWarning)
        return

    try:
        import seaborn as sns
        use_sns = True
    except Exception:
        use_sns = False

    fig, ax = plt.subplots(figsize=(10, 8))

    if use_sns:
        sns.heatmap(corr, cmap="coolwarm", center=0.0, ax=ax)
    else:
        cax = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(cax, ax=ax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)

    n = corr.shape[0]
    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            if abs(val) > 0.85 and i != j:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=6,
                )

    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

