"""
Feature Redundancy Analysis
===========================
Diagnoses cross-feature overlap, clusters correlated features, and produces a
redundancy-reduced feature set via residualization / orthogonalization.

Pipeline
--------
1. Cross-sectional pairwise Spearman correlation (pooled across dates)
2. IC, IC-stability (CV), turnover contribution, signal-decay half-life
3. Hierarchical clustering on |correlation| → feature families / clusters
4. Redundancy scoring per cluster: retain the highest-IC representative,
   flag others as redundant with justification
5. Optional: residualize each retained feature against other cluster
   representatives (Gram-Schmidt orthogonalization cross-sectionally)
6. Evaluate IC change, IC stability change, decile spread, model robustness
   (walk-forward Spearman IC with and without redundant features)

Constraints enforced
--------------------
- DO NOT change models or execution
- Features are never hard-dropped without an audit entry justifying it
- All decisions are data-driven; no hardcoded cutoffs on feature names

Usage
-----
    from model_selection.feature_redundancy import (
        RedundancyConfig, run_feature_redundancy,
        format_redundancy_report,
    )

    cfg = RedundancyConfig()
    result = run_feature_redundancy(panel_df, feature_cols, cfg)
    print(format_redundancy_report(result))
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

from model_selection.research_contract import (
    FEATURE_SPECS,
    _daily_rank_ic,
    _monotonicity,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RedundancyConfig:
    # Clustering
    corr_threshold: float = 0.60          # |Spearman| above this → same cluster
    linkage_method: str = "average"       # scipy linkage method

    # IC / stability
    min_dates_for_ic: int = 20            # skip features with fewer valid dates
    ic_stability_cv_threshold: float = 2.0  # IC CV above this → unstable
    ic_abs_threshold: float = 0.005       # |IC| below this → weak alpha source

    # Residualization
    orthogonalize: bool = True            # residualize within cluster
    orth_ridge: float = 0.01              # ridge regularization for orth regression

    # Signal decay
    decay_horizons: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    half_life_target: str = "target_return"

    # Turnover contribution
    top_q_for_turnover: float = 0.20      # fraction considered "top bucket"

    # Evaluation
    eval_target_col: str = "target_return"
    eval_raw_col: str = "raw_5d"
    eval_holding_days: int = 5
    min_obs_for_eval: int = 50


# ---------------------------------------------------------------------------
# Step 1 — cross-sectional pairwise Spearman correlation
# ---------------------------------------------------------------------------


def compute_cs_correlation_matrix(
    df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """
    Pool cross-sectional rank correlations across all dates.

    For each date compute the full pairwise Spearman rank correlation matrix
    on the cross-section, then average over dates (equal-weight).  This
    removes time-series autocorrelation and gives a pure cross-sectional
    redundancy estimate.
    """
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        return pd.DataFrame()

    work = df[["date"] + cols].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    # Vectorised per-date ranking (avoids repeated rank() inside the loop)
    ranked_all = work.groupby("date", sort=False)[cols].rank(
        method="average", na_option="keep"
    )
    ranked_all.insert(0, "date", work["date"].values)

    n = len(cols)
    acc = np.zeros((n, n), dtype=float)
    count = 0

    for _, g in ranked_all.groupby("date", sort=False):
        sub = g[cols].to_numpy(dtype=float)
        # Drop rows with any nan
        valid_rows = np.isfinite(sub).all(axis=1)
        sub = sub[valid_rows]
        if len(sub) < 10:
            continue
        # Identify columns with ≥2 unique values
        valid_col_mask = np.array([len(np.unique(sub[:, j])) >= 2 for j in range(n)])
        valid_col_idx = np.where(valid_col_mask)[0]
        if len(valid_col_idx) < 2:
            continue
        # Pearson correlation of pre-ranked data == Spearman correlation
        corr = np.corrcoef(sub[:, valid_col_idx].T)
        ij = np.ix_(valid_col_idx, valid_col_idx)
        finite_mask = np.isfinite(corr)
        acc[ij] += np.where(finite_mask, corr, 0.0)
        count += 1

    if count == 0:
        return pd.DataFrame(np.eye(n), index=cols, columns=cols)

    avg = acc / count
    np.fill_diagonal(avg, 1.0)
    return pd.DataFrame(avg, index=cols, columns=cols)


# ---------------------------------------------------------------------------
# Step 2 — per-feature IC, IC stability, turnover contribution, signal decay
# ---------------------------------------------------------------------------


def _ic_series_for_feature(
    df: pd.DataFrame, feature: str, target_col: str
) -> np.ndarray:
    vals: list[float] = []
    for _, g in df[["date", feature, target_col]].dropna().groupby("date", sort=True):
        if len(g) < 8 or g[feature].nunique() < 2 or g[target_col].nunique() < 2:
            continue
        c = g[feature].corr(g[target_col], method="spearman")
        if np.isfinite(c):
            vals.append(float(c))
    return np.array(vals, dtype=float)


def _signal_decay_half_life(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    horizons: list[int],
) -> float:
    """
    Compute the IC at each horizon (shift feature vs forward-return target),
    then fit an exponential decay to find the half-life in days.
    Returns nan if too few valid horizons.
    """
    ic_by_horizon: list[tuple[int, float]] = []
    for h in horizons:
        col_name = f"_fwd_{h}"
        tmp = df[["date", "ticker", feature, target_col]].copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        tmp = tmp.sort_values(["ticker", "date"])
        tmp[col_name] = (
            pd.to_numeric(tmp[target_col], errors="coerce")
            .groupby(tmp["ticker"])
            .shift(-h)
        )
        ics = _ic_series_for_feature(tmp.dropna(subset=[col_name]), feature, col_name)
        if len(ics) >= 5:
            ic_by_horizon.append((h, float(np.mean(ics))))

    if len(ic_by_horizon) < 3:
        return float("nan")

    hs = np.array([x[0] for x in ic_by_horizon], dtype=float)
    ics = np.array([abs(x[1]) for x in ic_by_horizon], dtype=float)
    ic0 = ics[0]
    if ic0 < 1e-8:
        return float("nan")
    half_target = ic0 / 2.0

    # Linear interpolation to find half-life
    for i in range(len(ics) - 1):
        if ics[i] >= half_target >= ics[i + 1]:
            frac = (ics[i] - half_target) / max(ics[i] - ics[i + 1], 1e-10)
            return float(hs[i] + frac * (hs[i + 1] - hs[i]))
    return float(hs[-1])  # still above half at longest horizon


def _top_q_turnover(df: pd.DataFrame, feature: str, top_q: float = 0.20) -> float:
    """
    Fraction of top-q bucket that changes per date (Jaccard complement).
    High turnover = feature requires expensive rebalancing.
    """
    dates = sorted(df["date"].dropna().unique())
    if len(dates) < 2:
        return float("nan")
    turnovers = []
    prev_top: set[str] = set()
    for d in dates:
        g = df[df["date"] == d]
        s = pd.to_numeric(g[feature], errors="coerce")
        if s.notna().sum() < 5:
            continue
        threshold = s.quantile(1.0 - top_q)
        cur_top = set(g.loc[s >= threshold, "ticker"].dropna().astype(str).tolist())
        if prev_top:
            union = prev_top | cur_top
            if union:
                turnovers.append(1.0 - len(prev_top & cur_top) / len(union))
        prev_top = cur_top
    return float(np.mean(turnovers)) if turnovers else float("nan")


def compute_feature_diagnostics(
    df: pd.DataFrame,
    feature_cols: list[str],
    cfg: RedundancyConfig,
) -> pd.DataFrame:
    """
    Compute per-feature: IC, IC std, IC CV, IC t-stat, turnover contribution,
    signal-decay half-life, decile spread, monotonicity.
    """
    target_col = cfg.eval_target_col
    if target_col not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    cols = [c for c in feature_cols if c in work.columns]

    rows = []
    for feat in cols:
        spec = FEATURE_SPECS.get(feat)
        ics = _ic_series_for_feature(work, feat, target_col)
        if len(ics) < cfg.min_dates_for_ic:
            ic_mean = ic_std = ic_cv = ic_tstat = float("nan")
        else:
            ic_mean = float(np.mean(ics))
            ic_std = float(np.std(ics, ddof=1))
            ic_cv = ic_std / abs(ic_mean) if abs(ic_mean) > 1e-10 else float("nan")
            ic_tstat = float(ic_mean / (ic_std / math.sqrt(len(ics)))) if ic_std > 1e-10 else float("nan")

        spread, mono = _monotonicity(work, feat, target_col)
        turnover = _top_q_turnover(work, feat, top_q=cfg.top_q_for_turnover)
        half_life = _signal_decay_half_life(work, feat, target_col, cfg.decay_horizons)

        rows.append(
            {
                "feature": feat,
                "family": spec.family if spec else "unknown",
                "expected_sign": int(spec.expected_sign) if spec else 0,
                "ic_mean": round(ic_mean, 5) if math.isfinite(ic_mean) else float("nan"),
                "ic_std": round(ic_std, 5) if math.isfinite(ic_std) else float("nan"),
                "ic_cv": round(ic_cv, 3) if math.isfinite(ic_cv) else float("nan"),
                "ic_tstat": round(ic_tstat, 3) if math.isfinite(ic_tstat) else float("nan"),
                "decile_spread": round(spread, 5) if math.isfinite(spread) else float("nan"),
                "monotonicity": round(mono, 3) if math.isfinite(mono) else float("nan"),
                "turnover": round(turnover, 3) if math.isfinite(turnover) else float("nan"),
                "half_life_days": round(half_life, 1) if math.isfinite(half_life) else float("nan"),
                "n_ic_days": len(ics),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3 — hierarchical clustering on |correlation|
# ---------------------------------------------------------------------------


def cluster_features_by_correlation(
    corr_matrix: pd.DataFrame, cfg: RedundancyConfig
) -> dict[int, list[str]]:
    """
    Run agglomerative clustering on 1 - |Spearman| distance matrix.
    Returns {cluster_id: [feature_name, ...]} dict.
    """
    if corr_matrix.empty or len(corr_matrix) < 2:
        cols = list(corr_matrix.columns)
        return {i: [c] for i, c in enumerate(cols)}

    cols = list(corr_matrix.columns)
    dist = 1.0 - corr_matrix.abs().to_numpy(dtype=float)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 1.0)

    condensed = squareform(dist)
    Z = linkage(condensed, method=cfg.linkage_method)
    cut_distance = 1.0 - cfg.corr_threshold
    labels = fcluster(Z, cut_distance, criterion="distance")

    clusters: dict[int, list[str]] = {}
    for col, cid in zip(cols, labels):
        clusters.setdefault(int(cid), []).append(col)
    return clusters


# ---------------------------------------------------------------------------
# Step 4 — redundancy scoring: pick representative per cluster
# ---------------------------------------------------------------------------


def select_cluster_representatives(
    clusters: dict[int, list[str]],
    diagnostics: pd.DataFrame,
    cfg: RedundancyConfig,
) -> pd.DataFrame:
    """
    For each cluster:
    - rank features by |IC t-stat| (primary) × monotonicity (secondary)
    - select the highest-scoring feature as the cluster representative
    - flag others as redundant with justification

    Returns a DataFrame with columns:
        cluster_id, feature, role (representative|redundant),
        justification, ic_mean, ic_tstat, ic_cv, family
    """
    if diagnostics.empty:
        return pd.DataFrame()

    diag = diagnostics.set_index("feature")
    rows = []

    for cid, feats in clusters.items():
        if not feats:
            continue

        scores = []
        for f in feats:
            if f not in diag.index:
                scores.append((0.0, f))
                continue
            row = diag.loc[f]
            ic_t = abs(float(row.get("ic_tstat", 0) or 0))
            mono = float(row.get("monotonicity", 0.5) or 0.5)
            ic_abs = abs(float(row.get("ic_mean", 0) or 0))
            scores.append((ic_t * mono * (1.0 + ic_abs), f))

        scores.sort(reverse=True)
        representative = scores[0][1]

        for _, feat in scores:
            if feat not in diag.index:
                rows.append(
                    {
                        "cluster_id": cid,
                        "feature": feat,
                        "role": "redundant" if feat != representative else "representative",
                        "justification": "not in diagnostics",
                        "ic_mean": float("nan"),
                        "ic_tstat": float("nan"),
                        "ic_cv": float("nan"),
                        "family": FEATURE_SPECS[feat].family if feat in FEATURE_SPECS else "unknown",
                    }
                )
                continue

            r = diag.loc[feat]
            ic_t = float(r.get("ic_tstat", float("nan")))
            ic_mean = float(r.get("ic_mean", float("nan")))
            ic_cv = float(r.get("ic_cv", float("nan")))
            family = str(r.get("family", "unknown"))

            if feat == representative:
                role = "representative"
                justification = f"Highest IC·mono score in cluster (|IC_t|={abs(ic_t):.2f})"
            else:
                rep_row = diag.loc[representative]
                rep_ic_t = abs(float(rep_row.get("ic_tstat", 0) or 0))
                ic_ratio = abs(ic_t) / max(rep_ic_t, 1e-8)
                # Compute pairwise corr with representative
                justification = (
                    f"Redundant with '{representative}': "
                    f"|IC_t_ratio|={ic_ratio:.2f}, "
                    f"same cluster (|corr|>{cfg.corr_threshold:.2f}). "
                )
                if abs(ic_mean) < cfg.ic_abs_threshold:
                    justification += f"Also: |IC|={abs(ic_mean):.4f} < {cfg.ic_abs_threshold} threshold."
                elif math.isfinite(ic_cv) and ic_cv > cfg.ic_stability_cv_threshold:
                    justification += f"Also: IC_CV={ic_cv:.2f} > {cfg.ic_stability_cv_threshold} (unstable)."
                role = "redundant"

            rows.append(
                {
                    "cluster_id": cid,
                    "feature": feat,
                    "role": role,
                    "justification": justification,
                    "ic_mean": ic_mean,
                    "ic_tstat": ic_t,
                    "ic_cv": ic_cv,
                    "family": family,
                }
            )

    return pd.DataFrame(rows).sort_values(["cluster_id", "role"], ascending=[True, True]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 5 — within-cluster Gram-Schmidt orthogonalization
# ---------------------------------------------------------------------------


def _ols_residualize(y: np.ndarray, X: np.ndarray, ridge: float = 0.01) -> np.ndarray:
    """Ridge OLS residualize y against columns of X, cross-sectionally."""
    if X.shape[1] == 0:
        return y.copy()
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    try:
        coef = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
    return y - X @ coef


def orthogonalize_within_cluster(
    df: pd.DataFrame,
    cluster_features: list[str],
    representative: str,
    cfg: RedundancyConfig,
) -> pd.DataFrame:
    """
    For each non-representative feature in the cluster, residualize it against
    all features that come earlier in the ordering (Gram-Schmidt cross-sectionally
    per date).

    Returns df with orthogonalized columns appended as `<feature>_orth`.
    """
    out = df.copy()
    ordered = [representative] + [f for f in cluster_features if f != representative]

    for i, feat in enumerate(ordered):
        if i == 0 or feat not in out.columns:
            continue
        orth_col = f"{feat}_orth"
        predecessors = [f for f in ordered[:i] if f in out.columns]

        result_vals = out[feat].copy().to_numpy(dtype=float)

        for _, g in out.groupby("date", sort=False):
            idx = g.index
            y = pd.to_numeric(g[feat], errors="coerce").to_numpy(dtype=float)
            X_cols = [c for c in predecessors if c in g.columns]
            if not X_cols:
                continue
            Xmat = g[X_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(y) & np.all(np.isfinite(Xmat), axis=1)
            if valid.sum() < 10:
                continue
            Xmat_v = Xmat[valid]
            # Demean each column
            Xmat_v = Xmat_v - Xmat_v.mean(axis=0)
            y_v = y[valid] - np.nanmean(y[valid])
            resid = _ols_residualize(y_v, Xmat_v, cfg.orth_ridge)
            result_vals[idx[valid]] = resid

        out[orth_col] = result_vals
    return out


# ---------------------------------------------------------------------------
# Step 6 — evaluate impact of redundancy reduction
# ---------------------------------------------------------------------------


def evaluate_redundancy_reduction(
    df: pd.DataFrame,
    full_feature_cols: list[str],
    reduced_feature_cols: list[str],
    cfg: RedundancyConfig,
) -> pd.DataFrame:
    """
    Compare IC, IC stability, decile spread, and turnover for:
    - full feature set
    - redundancy-reduced feature set (representatives only)
    - orthogonalized feature set (if orth columns present)

    Returns a DataFrame with one row per (feature_set × metric) summary.
    """
    target_col = cfg.eval_target_col
    if target_col not in df.columns:
        return pd.DataFrame()

    sets: dict[str, list[str]] = {
        "full": [c for c in full_feature_cols if c in df.columns],
        "reduced": [c for c in reduced_feature_cols if c in df.columns],
    }
    orth_cols = [c for c in df.columns if c.endswith("_orth")]
    if orth_cols:
        sets["orthogonalized"] = orth_cols

    rows = []
    for set_name, feat_list in sets.items():
        if not feat_list:
            continue
        ic_means = []
        ic_cvs = []
        spreads = []
        monos = []
        turnovers = []
        for feat in feat_list:
            ics = _ic_series_for_feature(df, feat, target_col)
            if len(ics) < cfg.min_obs_for_eval:
                continue
            ic_m = float(np.mean(ics))
            ic_s = float(np.std(ics, ddof=1))
            ic_cv = ic_s / abs(ic_m) if abs(ic_m) > 1e-10 else float("nan")
            ic_means.append(ic_m)
            ic_cvs.append(ic_cv)
            spread, mono = _monotonicity(df, feat, target_col)
            spreads.append(spread)
            monos.append(mono)
            turnovers.append(_top_q_turnover(df, feat, cfg.top_q_for_turnover))

        rows.append(
            {
                "feature_set": set_name,
                "n_features": len(feat_list),
                "mean_ic": round(float(np.nanmean(ic_means)), 5) if ic_means else float("nan"),
                "mean_ic_cv": round(float(np.nanmean([x for x in ic_cvs if math.isfinite(x)])), 3)
                if ic_cvs else float("nan"),
                "mean_decile_spread": round(float(np.nanmean(spreads)), 5) if spreads else float("nan"),
                "mean_monotonicity": round(float(np.nanmean(monos)), 3) if monos else float("nan"),
                "mean_turnover": round(float(np.nanmean([x for x in turnovers if math.isfinite(x)])), 3)
                if turnovers else float("nan"),
                "ic_positive_rate": round(float(np.mean([x > 0 for x in ic_means])), 3) if ic_means else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


@dataclass
class RedundancyResult:
    corr_matrix: pd.DataFrame
    diagnostics: pd.DataFrame
    clusters: dict[int, list[str]]
    cluster_table: pd.DataFrame
    evaluation: pd.DataFrame
    representatives: list[str]
    redundant: list[str]
    cfg: RedundancyConfig


def run_feature_redundancy(
    df: pd.DataFrame,
    feature_cols: list[str],
    cfg: RedundancyConfig | None = None,
) -> RedundancyResult:
    """
    Full redundancy analysis pipeline.

    Parameters
    ----------
    df          : panel DataFrame with date/ticker and feature + target columns
    feature_cols: candidate alpha feature names
    cfg         : RedundancyConfig (default values used if None)

    Returns
    -------
    RedundancyResult with all intermediate outputs
    """
    if cfg is None:
        cfg = RedundancyConfig()

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values(["date", "ticker"]).reset_index(drop=True)

    active_cols = [c for c in feature_cols if c in work.columns]
    if not active_cols:
        empty = pd.DataFrame()
        return RedundancyResult(empty, empty, {}, empty, empty, [], [], cfg)

    # 1. Cross-sectional correlation matrix
    corr_matrix = compute_cs_correlation_matrix(work, active_cols)

    # 2. Per-feature diagnostics
    diagnostics = compute_feature_diagnostics(work, active_cols, cfg)

    # 3. Cluster features
    clusters = cluster_features_by_correlation(corr_matrix, cfg)

    # 4. Select representatives and flag redundant features
    cluster_table = select_cluster_representatives(clusters, diagnostics, cfg)

    representatives = (
        cluster_table.loc[cluster_table["role"] == "representative", "feature"].tolist()
    )
    redundant = (
        cluster_table.loc[cluster_table["role"] == "redundant", "feature"].tolist()
    )

    # 5. Orthogonalize within clusters (if enabled)
    if cfg.orthogonalize:
        for cid, feats in clusters.items():
            if len(feats) < 2:
                continue
            rep_candidates = [f for f in representatives if f in feats]
            rep = rep_candidates[0] if rep_candidates else feats[0]
            work = orthogonalize_within_cluster(work, feats, rep, cfg)

    # 6. Evaluate impact
    evaluation = evaluate_redundancy_reduction(work, active_cols, representatives, cfg)

    return RedundancyResult(
        corr_matrix=corr_matrix,
        diagnostics=diagnostics,
        clusters=clusters,
        cluster_table=cluster_table,
        evaluation=evaluation,
        representatives=representatives,
        redundant=redundant,
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_W = 28
_NW = 9


def format_redundancy_report(result: RedundancyResult) -> str:
    lines: list[str] = []

    # ── Cluster overview ─────────────────────────────────────────────────────
    lines.append("=" * 80)
    lines.append("FEATURE REDUNDANCY ANALYSIS")
    lines.append("=" * 80)
    lines.append(
        f"  Features analyzed: {len(result.diagnostics)} | "
        f"Clusters: {len(result.clusters)} | "
        f"Representatives: {len(result.representatives)} | "
        f"Redundant: {len(result.redundant)}"
    )
    lines.append(f"  Correlation threshold: |r|>{result.cfg.corr_threshold:.2f}  "
                 f"Linkage: {result.cfg.linkage_method}")
    lines.append("")

    # ── Per-cluster detail ───────────────────────────────────────────────────
    lines.append("CLUSTER DETAIL")
    lines.append("-" * 80)
    if result.cluster_table.empty:
        lines.append("  [No clusters — insufficient feature data]")
    else:
        for cid, sub in result.cluster_table.groupby("cluster_id"):
            n = len(sub)
            reps = sub[sub["role"] == "representative"]["feature"].tolist()
            lines.append(f"\n  Cluster {cid}  ({n} features, rep={reps})")
            hdr = f"    {'Feature':<{_W}} {'Role':<14} {'IC_mean':>{_NW}} {'IC_tstat':>{_NW}} {'IC_CV':>{_NW}} {'Family'}"
            lines.append(hdr)
            lines.append("    " + "-" * (len(hdr) - 4))
            for _, row in sub.iterrows():
                ic_m = float(row.get("ic_mean", float("nan")))
                ic_t = float(row.get("ic_tstat", float("nan")))
                ic_cv = float(row.get("ic_cv", float("nan")))
                lines.append(
                    f"    {str(row['feature']):<{_W}} "
                    f"{'*** rep ***' if row['role'] == 'representative' else 'redundant':<14} "
                    f"{ic_m:>{_NW}.4f} "
                    f"{ic_t:>{_NW}.3f} "
                    f"{ic_cv:>{_NW}.3f} "
                    f"{row.get('family', '')}"
                )
            # Show justifications for redundant features
            for _, row in sub[sub["role"] == "redundant"].iterrows():
                lines.append(f"      ↳ {row['feature']}: {row.get('justification', '')}")

    # ── Full feature diagnostics ─────────────────────────────────────────────
    lines.append("")
    lines.append("FEATURE DIAGNOSTICS (all features, sorted by |IC t-stat|)")
    lines.append("-" * 80)
    if result.diagnostics.empty:
        lines.append("  [No diagnostics]")
    else:
        diag = result.diagnostics.copy()
        diag["_sort"] = diag["ic_tstat"].abs()
        diag = diag.sort_values("_sort", ascending=False).drop(columns=["_sort"])
        hdr = (
            f"  {'Feature':<{_W}} {'Family':<22} "
            f"{'IC_mean':>{_NW}} {'IC_tstat':>{_NW}} {'IC_CV':>{_NW}} "
            f"{'Mono':>{_NW}} {'Turnover':>{_NW}} {'HalfLife':>{_NW}}"
        )
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))
        for _, row in diag.iterrows():
            lines.append(
                f"  {str(row['feature']):<{_W}} {str(row.get('family','')):<22} "
                f"{float(row.get('ic_mean', float('nan'))):>{_NW}.4f} "
                f"{float(row.get('ic_tstat', float('nan'))):>{_NW}.3f} "
                f"{float(row.get('ic_cv', float('nan'))):>{_NW}.3f} "
                f"{float(row.get('monotonicity', float('nan'))):>{_NW}.3f} "
                f"{float(row.get('turnover', float('nan'))):>{_NW}.3f} "
                f"{float(row.get('half_life_days', float('nan'))):>{_NW}.1f}"
            )

    # ── Evaluation table ─────────────────────────────────────────────────────
    lines.append("")
    lines.append("IMPACT EVALUATION  (full vs reduced vs orthogonalized)")
    lines.append("-" * 80)
    if result.evaluation.empty:
        lines.append("  [No evaluation data]")
    else:
        ev = result.evaluation
        hdr_e = (
            f"  {'Feature Set':<20} {'N':>5} "
            f"{'mean_IC':>{_NW}} {'mean_IC_CV':>{_NW}} "
            f"{'Spread':>{_NW}} {'Mono':>{_NW}} "
            f"{'Turnover':>{_NW}} {'IC_pos_rate':>{_NW}}"
        )
        lines.append(hdr_e)
        lines.append("  " + "-" * (len(hdr_e) - 2))
        for _, row in ev.iterrows():
            lines.append(
                f"  {str(row.get('feature_set','')):<20} "
                f"{int(row.get('n_features', 0)):>5} "
                f"{float(row.get('mean_ic', float('nan'))):>{_NW}.4f} "
                f"{float(row.get('mean_ic_cv', float('nan'))):>{_NW}.3f} "
                f"{float(row.get('mean_decile_spread', float('nan'))):>{_NW}.4f} "
                f"{float(row.get('mean_monotonicity', float('nan'))):>{_NW}.3f} "
                f"{float(row.get('mean_turnover', float('nan'))):>{_NW}.3f} "
                f"{float(row.get('ic_positive_rate', float('nan'))):>{_NW}.3f}"
            )

    # ── Selected independent features ────────────────────────────────────────
    lines.append("")
    lines.append("SELECTED INDEPENDENT FEATURES  (cluster representatives)")
    lines.append("-" * 80)
    for f in sorted(result.representatives):
        spec = FEATURE_SPECS.get(f)
        family = spec.family if spec else "unknown"
        lines.append(f"  {f:<{_W}}  [{family}]")

    # ── Redundant features with justification ────────────────────────────────
    lines.append("")
    lines.append("REDUNDANT FEATURES  (data-justified, not hard-dropped)")
    lines.append("-" * 80)
    if not result.redundant:
        lines.append("  None — all features are independent at the given threshold.")
    else:
        ct = result.cluster_table.set_index("feature")
        for f in sorted(result.redundant):
            just = ct.loc[f, "justification"] if f in ct.index else "see cluster table"
            lines.append(f"  {f:<{_W}}  {just}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 80)
    lines.append("VERDICT")
    lines.append("=" * 80)
    lines.append(_redundancy_verdict(result))

    return "\n".join(lines)


def _redundancy_verdict(result: RedundancyResult) -> str:
    n_total = len(result.diagnostics)
    n_rep = len(result.representatives)
    n_red = len(result.redundant)

    if n_total == 0:
        return "  Insufficient data."

    red_pct = n_red / n_total * 100

    ev = result.evaluation
    parts = [
        f"  {n_total} features analyzed → {n_rep} independent representatives, "
        f"{n_red} flagged redundant ({red_pct:.0f}%)."
    ]

    if not ev.empty and "feature_set" in ev.columns:
        full_row = ev[ev["feature_set"] == "full"]
        red_row = ev[ev["feature_set"] == "reduced"]
        if not full_row.empty and not red_row.empty:
            ic_lift = float(red_row["mean_ic"].values[0]) - float(full_row["mean_ic"].values[0])
            cv_lift = float(full_row["mean_ic_cv"].values[0]) - float(red_row["mean_ic_cv"].values[0])
            to_lift = float(full_row["mean_turnover"].values[0]) - float(red_row["mean_turnover"].values[0])
            parts.append(
                f"  Reduction impact:  IC change {ic_lift:+.5f},  "
                f"IC_CV change {-cv_lift:+.3f},  "
                f"turnover change {to_lift:+.3f}."
            )
            if ic_lift >= 0 and cv_lift >= 0:
                parts.append(
                    "  RECOMMEND: use reduced set — same or better IC with lower instability."
                )
            elif ic_lift >= -0.001:
                parts.append(
                    "  MARGINAL: negligible IC loss (<0.001) from redundancy removal — "
                    "reduced set is acceptable."
                )
            else:
                parts.append(
                    "  CAUTION: IC declines with reduced set — some 'redundant' features "
                    "carry incremental information. Review cluster memberships."
                )

    if result.cfg.orthogonalize:
        orth_row = ev[ev["feature_set"] == "orthogonalized"] if not ev.empty else pd.DataFrame()
        if not orth_row.empty and not full_row.empty:
            o_cv = float(orth_row["mean_ic_cv"].values[0])
            f_cv = float(full_row["mean_ic_cv"].values[0])
            if o_cv < f_cv:
                parts.append(
                    f"  Orthogonalization reduces mean IC_CV from {f_cv:.3f} → {o_cv:.3f}: "
                    f"signal is more stable after within-cluster residualization."
                )

    return "\n".join(parts)
