"""
Regime Conditioning
===================
Panel-native market regime detection and regime-conditional model evaluation.

Regimes are derived algorithmically from cross-sectional panel statistics —
no external downloads (SPY/VIX), no hardcoded thresholds:

    cs_vol          : mean realized vol across the cross-section (proxy for market vol)
    mkt_trend       : 21d rolling equal-weighted market return (trend momentum)
    mkt_drawdown    : running drawdown from rolling peak (stress measure)
    cs_dispersion   : cross-sectional std of daily returns (regime-shift sensor)
    liq_change      : 20d change in log median ADV (liquidity stress)

A GaussianMixture with BIC-selected n_components (2–6) assigns each date a
regime label.  Semantic tags (Low-Vol-Trend, High-Vol-Stress, …) are derived
from the component's feature centroid — not hardcoded.

Two conditioning strategies are supported:

    Strategy A — per-regime separate models:  each regime gets its own
        walk-forward trained on regime-filtered history.  Produces a
        regime × model performance table.

    Strategy B — regime as feature:  one-hot-encoded regime dummies are
        appended to the feature matrix.  Existing walk-forward runs as-is.

Usage
-----
    from model_selection.regime_conditioning import (
        RegimeConfig, PanelRegimeDetector, evaluate_regime_conditioning,
        format_regime_report,
    )
    from model_selection.objective_comparison import WalkForwardConfig
    from model_selection.model_registry import build_models

    regime_cfg = RegimeConfig()
    wf_cfg = WalkForwardConfig()
    models = build_models(cfg)

    regime_table, cond_table = evaluate_regime_conditioning(
        panel_df, models, feature_cols, wf_cfg, regime_cfg
    )
    print(format_regime_report(regime_table, cond_table))
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from model_selection.objective_comparison import (
    WalkForwardConfig,
    _WindowPrep,
    _aggregate_window_metrics,
    _daily_rank_ic,
    _decile_metrics,
    _fit_one_window,
    _long_quintile_sharpe,
    _walk_forward_dates,
    prepare_window,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RegimeConfig:
    n_regimes_min: int = 2
    n_regimes_max: int = 6
    cluster_method: str = "gmm"      # "gmm" or "kmeans"
    smoothing_days: int = 5          # majority-vote smoothing window
    min_regime_obs: int = 40         # minimum OOS rows to include a regime in eval
    random_state: int = 42

    # Panel column names used to derive daily regime features
    return_col: str = "daily_return"
    adv_col: str = "adv_dollar_20"
    vol_col: str = "realised_vol_20d"


# ---------------------------------------------------------------------------
# Daily feature extraction
# ---------------------------------------------------------------------------


def _compute_daily_regime_features(
    df: pd.DataFrame, cfg: RegimeConfig
) -> pd.DataFrame:
    """
    Collapse the cross-sectional panel into one row per date with five
    market-level regime features.  All inputs come from existing panel columns.
    """
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")

    agg: dict[str, pd.Series] = {}

    # Cross-sectional dispersion and equal-weighted market return
    if cfg.return_col in work.columns:
        r = work[cfg.return_col].replace([np.inf, -np.inf], np.nan)
        work["_r"] = r
        g = work.groupby("date")["_r"]
        agg["cs_dispersion"] = g.std()
        agg["mkt_return"] = g.mean()

    # Cross-sectional mean realized vol
    if cfg.vol_col in work.columns:
        v = work[cfg.vol_col].replace([np.inf, -np.inf], np.nan)
        work["_v"] = v
        agg["cs_vol"] = work.groupby("date")["_v"].mean()

    # Log median ADV (proxy for liquidity)
    if cfg.adv_col in work.columns:
        a = work[cfg.adv_col].replace([np.inf, -np.inf], np.nan)
        work["_a"] = a.apply(lambda x: math.log(max(x, 1e-10)) if pd.notna(x) else np.nan)
        agg["log_adv"] = work.groupby("date")["_a"].median()

    if not agg:
        raise ValueError("No usable panel columns found for regime feature extraction.")

    daily = pd.DataFrame(agg).sort_index()
    daily.index = pd.to_datetime(daily.index)

    # Derived rolling features
    if "mkt_return" in daily.columns:
        daily["mkt_trend"] = daily["mkt_return"].rolling(21, min_periods=10).sum()
        cum = daily["mkt_return"].cumsum().ffill()
        peak = cum.rolling(252, min_periods=21).max()
        daily["mkt_drawdown"] = (cum - peak).clip(upper=0.0)
        daily = daily.drop(columns=["mkt_return"])

    if "log_adv" in daily.columns:
        daily["liq_change"] = daily["log_adv"].diff(20)
        daily = daily.drop(columns=["log_adv"])

    return daily.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Regime Detector
# ---------------------------------------------------------------------------


class PanelRegimeDetector:
    """
    Fit a GaussianMixture (or KMeans) on daily market features derived from the
    cross-sectional panel and assign semantic labels based on centroids.
    """

    def __init__(self, cfg: RegimeConfig | None = None) -> None:
        self.cfg = cfg or RegimeConfig()
        self._scaler: StandardScaler | None = None
        self._cluster_model: Any = None
        self._n_components: int = 2
        self._feature_cols: list[str] = []
        self._label_map: dict[int, str] = {}

    def fit(self, daily_features: pd.DataFrame) -> "PanelRegimeDetector":
        feat = daily_features.dropna(how="any")
        if len(feat) < max(self.cfg.n_regimes_max * 10, 40):
            raise ValueError(
                f"Insufficient daily observations ({len(feat)}) for regime fitting."
            )

        self._feature_cols = [c for c in feat.columns if feat[c].dtype.kind in "fd"]
        X = feat[self._feature_cols].to_numpy(dtype=float)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        self._scaler = scaler

        if self.cfg.cluster_method == "gmm":
            best_bic = np.inf
            best_model = None
            best_k = self.cfg.n_regimes_min
            for k in range(self.cfg.n_regimes_min, self.cfg.n_regimes_max + 1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gm = GaussianMixture(
                        n_components=k,
                        covariance_type="full",
                        n_init=3,
                        max_iter=200,
                        random_state=self.cfg.random_state,
                    )
                    gm.fit(Xs)
                bic = gm.bic(Xs)
                if bic < best_bic:
                    best_bic = bic
                    best_model = gm
                    best_k = k
            self._cluster_model = best_model
            self._n_components = best_k
        else:
            from sklearn.cluster import KMeans

            best_inertia = np.inf
            best_model = None
            best_k = self.cfg.n_regimes_min
            for k in range(self.cfg.n_regimes_min, self.cfg.n_regimes_max + 1):
                km = KMeans(
                    n_clusters=k,
                    n_init=5,
                    max_iter=300,
                    random_state=self.cfg.random_state,
                )
                km.fit(Xs)
                if km.inertia_ < best_inertia:
                    best_inertia = km.inertia_
                    best_model = km
                    best_k = k
            self._cluster_model = best_model
            self._n_components = best_k

        # Build semantic labels from centroids
        if self.cfg.cluster_method == "gmm":
            centroids_scaled = self._cluster_model.means_
        else:
            centroids_scaled = self._cluster_model.cluster_centers_

        centroids = self._scaler.inverse_transform(centroids_scaled)
        centroid_df = pd.DataFrame(centroids, columns=self._feature_cols)

        vol_col = "cs_vol" if "cs_vol" in centroid_df.columns else centroid_df.columns[0]
        trend_col = "mkt_trend" if "mkt_trend" in centroid_df.columns else None

        n = len(centroid_df)
        vol_rank = centroid_df[vol_col].rank().astype(int)
        trend_rank = (
            centroid_df[trend_col].rank().astype(int) if trend_col else pd.Series(range(n))
        )
        mid = n // 2

        self._label_map = {}
        for i in range(n):
            high_vol = vol_rank.iloc[i] > mid
            if trend_col:
                high_trend = trend_rank.iloc[i] > mid
                if high_vol and high_trend:
                    tag = "HighVol_Trend"
                elif high_vol:
                    tag = "HighVol_Stress"
                elif high_trend:
                    tag = "LowVol_Trend"
                else:
                    tag = "LowVol_Flat"
            else:
                tag = "HighVol" if high_vol else "LowVol"
            self._label_map[i] = f"R{i}_{tag}"

        return self

    def predict(self, daily_features: pd.DataFrame) -> pd.Series:
        """Return pd.Series of string regime labels indexed by date."""
        if self._cluster_model is None or self._scaler is None:
            raise RuntimeError("PanelRegimeDetector must be fitted first.")

        feat = daily_features.copy()
        missing = set(self._feature_cols) - set(feat.columns)
        if missing:
            raise ValueError(f"Missing regime features: {missing}")

        valid_mask = feat[self._feature_cols].notna().all(axis=1)
        X = feat.loc[valid_mask, self._feature_cols].to_numpy(dtype=float)
        Xs = self._scaler.transform(X)
        raw = self._cluster_model.predict(Xs)

        labels = pd.Series(
            [self._label_map[int(r)] for r in raw],
            index=feat.index[valid_mask],
            name="regime_label",
            dtype="object",
        )

        # Fill NaN dates (from invalid rows) with nearest valid label
        full = labels.reindex(feat.index).ffill().bfill()

        # Majority-vote smoothing over smoothing_days window
        if self.cfg.smoothing_days > 1:
            full = _string_mode_smooth(full, self.cfg.smoothing_days)

        return full

    def attach_to_panel(
        self, df: pd.DataFrame, daily_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge regime labels into the panel df on the date column."""
        labels = self.predict(daily_features)
        labels_df = (
            labels.rename("regime_label")
            .reset_index()
            .rename(columns={"index": "date"})
        )
        labels_df["date"] = pd.to_datetime(labels_df["date"])

        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        if "regime_label" in out.columns:
            out = out.drop(columns=["regime_label"])
        return out.merge(labels_df, on="date", how="left")

    @property
    def n_regimes(self) -> int:
        return self._n_components

    @property
    def label_map(self) -> dict[int, str]:
        return dict(self._label_map)


def _string_mode_smooth(s: pd.Series, window: int) -> pd.Series:
    """Rolling majority-vote smoothing for a string Series (O(n·w))."""
    arr = s.to_numpy(dtype=object)
    out = arr.copy()
    n = len(arr)
    for i in range(n):
        lo = max(0, i - window + 1)
        window_vals = arr[lo : i + 1]
        unique, counts = np.unique(window_vals, return_counts=True)
        out[i] = unique[int(counts.argmax())]
    return pd.Series(out, index=s.index, name=s.name)


# ---------------------------------------------------------------------------
# Strategy A: per-regime walk-forward
# ---------------------------------------------------------------------------


def _regime_walk_forward(
    regime_label: str,
    regime_df: pd.DataFrame,
    models: list[tuple[str, Any, bool, str]],
    feature_cols: list[str],
    cfg: WalkForwardConfig,
) -> list[dict]:
    """
    Run a walk-forward within a single regime slice and return per-model metrics.
    models is a list of (name, estimator, uses_proba, kind) tuples.
    """
    actual_feat = [c for c in feature_cols if c in regime_df.columns]
    if not actual_feat:
        return []

    windows = _walk_forward_dates(regime_df, cfg)
    if not windows:
        return []

    per_model: dict[str, list[pd.DataFrame]] = {name: [] for name, _, _, _ in models}

    for train_start, train_end, test_start, test_end in windows:
        train_df = regime_df[
            (regime_df["date"] >= train_start) & (regime_df["date"] <= train_end)
        ]
        test_df = regime_df[
            (regime_df["date"] >= test_start) & (regime_df["date"] <= test_end)
        ]
        if len(train_df) < cfg.min_train_obs or test_df.empty:
            continue
        window_prep: _WindowPrep | None = prepare_window(train_df, test_df, actual_feat, cfg)
        for name, model, _uses_proba, kind in models:
            if kind == "short_classifier":
                continue
            scored = _fit_one_window(
                name, model, kind, train_df, test_df, actual_feat, cfg.target_col, cfg,
                _precomputed=window_prep,
            )
            if scored is not None and not scored.empty:
                per_model[name].append(scored)

    rows = []
    for name, frames in per_model.items():
        if not frames:
            continue
        m = _aggregate_window_metrics(frames, cfg)
        m["regime"] = regime_label
        m["model"] = name
        m["n_obs"] = len(regime_df)
        rows.append(m)
    return rows


# ---------------------------------------------------------------------------
# Strategy B: regime dummies as features
# ---------------------------------------------------------------------------


def add_regime_dummies(
    df: pd.DataFrame, regime_col: str = "regime_label"
) -> pd.DataFrame:
    """
    Append one-hot encoded regime dummies to df.
    Column names: regime_<label>.  Picked up automatically as model features.
    """
    if regime_col not in df.columns:
        return df
    dummies = pd.get_dummies(
        df[regime_col], prefix="regime", drop_first=False, dtype=float
    )
    existing = [c for c in dummies.columns if c in df.columns]
    return pd.concat([df.drop(columns=existing), dummies], axis=1)


def _strategy_b_walk_forward(
    df: pd.DataFrame,
    models: list[tuple[str, Any, bool, str]],
    feature_cols: list[str],
    cfg: WalkForwardConfig,
    tag: str,
) -> list[dict]:
    actual_feat = [c for c in feature_cols if c in df.columns]
    windows = _walk_forward_dates(df, cfg)

    per_model: dict[str, list[pd.DataFrame]] = {name: [] for name, _, _, _ in models}

    for train_start, train_end, test_start, test_end in windows:
        train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
        if len(train_df) < cfg.min_train_obs or test_df.empty:
            continue
        window_prep_b: _WindowPrep | None = prepare_window(train_df, test_df, actual_feat, cfg)
        for name, model, _uses_proba, kind in models:
            if kind == "short_classifier":
                continue
            scored = _fit_one_window(
                name, model, kind, train_df, test_df, actual_feat, cfg.target_col, cfg,
                _precomputed=window_prep_b,
            )
            if scored is not None and not scored.empty:
                per_model[name].append(scored)

    rows = []
    for name, frames in per_model.items():
        if not frames:
            continue
        m = _aggregate_window_metrics(frames, cfg)
        m["strategy"] = tag
        m["model"] = name
        rows.append(m)
    return rows


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------


def evaluate_regime_conditioning(
    df: pd.DataFrame,
    models: list[tuple[str, Any, bool, str]],
    feature_cols: list[str],
    wf_cfg: WalkForwardConfig | None = None,
    regime_cfg: RegimeConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full regime-conditioning evaluation.

    Parameters
    ----------
    df          : panel DataFrame with date/ticker index and all feature + target columns
    models      : list of (name, estimator, uses_proba, kind) from build_models()
    feature_cols: list of alpha feature column names
    wf_cfg      : walk-forward configuration (default WalkForwardConfig())
    regime_cfg  : regime detector configuration (default RegimeConfig())

    Returns
    -------
    regime_table : per-regime × per-model performance table (Strategy A)
    cond_table   : Strategy B (regime dummies) vs baseline comparison
    """
    if wf_cfg is None:
        wf_cfg = WalkForwardConfig()
    if regime_cfg is None:
        regime_cfg = RegimeConfig()

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values(["date", "ticker"]).reset_index(drop=True)

    # Fit regime detector on daily panel-derived features
    try:
        daily = _compute_daily_regime_features(work, regime_cfg)
        detector = PanelRegimeDetector(regime_cfg)
        detector.fit(daily)
    except (ValueError, RuntimeError) as exc:
        warnings.warn(f"Regime fitting failed: {exc}", stacklevel=2)
        return pd.DataFrame(), pd.DataFrame()

    work = detector.attach_to_panel(work, daily)

    # ---- Strategy A: per-regime walk-forward ----
    regime_rows: list[dict] = []
    regimes_found = work["regime_label"].dropna().unique()

    for rlabel in sorted(regimes_found):
        r_df = work[work["regime_label"] == rlabel].copy()
        if len(r_df) < regime_cfg.min_regime_obs:
            continue
        rows = _regime_walk_forward(rlabel, r_df, models, feature_cols, wf_cfg)
        regime_rows.extend(rows)

    regime_table = pd.DataFrame(regime_rows)
    if not regime_table.empty:
        regime_table = regime_table.sort_values(
            ["regime", "ic_mean"], ascending=[True, False]
        ).reset_index(drop=True)

    # ---- Strategy B: regime dummies as features ----
    baseline_rows = _strategy_b_walk_forward(work, models, feature_cols, wf_cfg, tag="baseline")

    df_b = add_regime_dummies(work, "regime_label")
    dummy_cols = [
        c for c in df_b.columns
        if c.startswith("regime_") and c != "regime_label"
    ]
    aug_features = feature_cols + [c for c in dummy_cols if c not in feature_cols]
    regime_dummy_rows = _strategy_b_walk_forward(
        df_b, models, aug_features, wf_cfg, tag="regime_dummies"
    )

    cond_table = pd.DataFrame(baseline_rows + regime_dummy_rows)
    if not cond_table.empty:
        cond_table = cond_table.sort_values(
            ["strategy", "ic_mean"], ascending=[True, False]
        ).reset_index(drop=True)

    return regime_table, cond_table


# ---------------------------------------------------------------------------
# IC stability per regime
# ---------------------------------------------------------------------------


def compute_regime_ic_stability(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    regime_col: str = "regime_label",
) -> pd.DataFrame:
    """
    For each regime compute IC mean, IC std, and CV (std / |mean|).
    Lower CV → more stable signal across market conditions.
    """
    rows = []
    for rlabel, g in df.groupby(regime_col):
        ics = _daily_rank_ic(g, score_col, target_col)
        if len(ics) < 5:
            continue
        ic_arr = np.array(ics, dtype=float)
        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr, ddof=1))
        ic_cv = ic_std / abs(ic_mean) if abs(ic_mean) > 1e-10 else float("nan")
        ic_ir = ic_mean / ic_std if ic_std > 1e-10 else float("nan")
        spread, mono = _decile_metrics(g, score_col, target_col)
        sharpe = _long_quintile_sharpe(g, score_col, target_col, holding_days=5)
        rows.append(
            {
                "regime": rlabel,
                "n_obs": len(g),
                "ic_mean": round(ic_mean, 5),
                "ic_std": round(ic_std, 5),
                "ic_cv": round(ic_cv, 3) if math.isfinite(ic_cv) else float("nan"),
                "ic_ir": round(ic_ir, 3) if math.isfinite(ic_ir) else float("nan"),
                "decile_spread": round(spread, 5) if math.isfinite(spread) else float("nan"),
                "monotonicity": round(mono, 3) if math.isfinite(mono) else float("nan"),
                "long_sharpe": round(sharpe, 3) if math.isfinite(sharpe) else float("nan"),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ic_cv").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_COL_W = 20
_NUM_W = 10


def format_regime_report(
    regime_table: pd.DataFrame,
    cond_table: pd.DataFrame,
) -> str:
    lines: list[str] = []

    # ── Strategy A: per-regime table ────────────────────────────────────────
    lines.append("=" * 90)
    lines.append("STRATEGY A — PER-REGIME MODELS  (separate walk-forward per regime slice)")
    lines.append("=" * 90)

    if regime_table.empty:
        lines.append("  [No results — insufficient per-regime observations]")
    else:
        num_cols = [
            c for c in [
                "ic_mean", "ic_tstat", "ic_ir", "long_sharpe",
                "monotonicity", "decile_spread", "turnover", "alpha_cap_proxy",
            ]
            if c in regime_table.columns
        ]
        header = f"{'Regime':<24} {'Model':<28}" + "".join(f"{c:>{_NUM_W}}" for c in num_cols)
        lines.append(header)
        lines.append("-" * len(header))
        for _, row in regime_table.iterrows():
            vals = "".join(
                f"{float(row.get(c, float('nan'))):>{_NUM_W}.4f}" for c in num_cols
            )
            lines.append(f"{str(row.get('regime', '')):<24} {str(row.get('model', '')):<28}{vals}")

    # ── Strategy B: regime-as-feature comparison ─────────────────────────────
    lines.append("")
    lines.append("=" * 90)
    lines.append("STRATEGY B — REGIME DUMMIES AS FEATURES  (appended to feature matrix)")
    lines.append("=" * 90)

    if cond_table.empty:
        lines.append("  [No results]")
    else:
        num_cols_b = [
            c for c in [
                "ic_mean", "ic_tstat", "ic_ir", "long_sharpe",
                "monotonicity", "turnover", "alpha_cap_proxy",
            ]
            if c in cond_table.columns
        ]
        strategies = (
            cond_table["strategy"].unique()
            if "strategy" in cond_table.columns
            else []
        )
        for strat in strategies:
            sub = cond_table[cond_table["strategy"] == strat]
            lines.append(f"\n  [{strat}]")
            hdr = f"  {'Model':<32}" + "".join(f"{c:>{_NUM_W}}" for c in num_cols_b)
            lines.append(hdr)
            lines.append("  " + "-" * (len(hdr) - 2))
            for _, row in sub.iterrows():
                vals = "".join(
                    f"{float(row.get(c, float('nan'))):>{_NUM_W}.4f}" for c in num_cols_b
                )
                lines.append(f"  {str(row.get('model', '')):<32}{vals}")

        # Δ section when both strategies are present
        if set(strategies) == {"baseline", "regime_dummies"}:
            lines.append("\n  LIFT: regime_dummies vs baseline (mean across models)")
            base = cond_table[cond_table["strategy"] == "baseline"].set_index("model")
            aug = cond_table[cond_table["strategy"] == "regime_dummies"].set_index("model")
            common_models = base.index.intersection(aug.index)
            for col in num_cols_b:
                if col in base.columns and col in aug.columns:
                    lifts = aug.loc[common_models, col] - base.loc[common_models, col]
                    n_imp = int((lifts > 0).sum())
                    lines.append(
                        f"  {col:<30} mean_lift={float(lifts.mean()):+.4f}  "
                        f"n_improve={n_imp}/{len(lifts)}"
                    )

    # ── Verdict ─────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 90)
    lines.append("VERDICT")
    lines.append("=" * 90)
    lines.append(_regime_verdict(regime_table, cond_table))

    return "\n".join(lines)


def _regime_verdict(
    regime_table: pd.DataFrame, cond_table: pd.DataFrame
) -> str:
    parts: list[str] = []

    # Strategy A verdict: IC spread across regimes
    if not regime_table.empty and "ic_mean" in regime_table.columns:
        ic_spread = float(
            regime_table["ic_mean"].max() - regime_table["ic_mean"].min()
        )
        n_reg = regime_table["regime"].nunique() if "regime" in regime_table.columns else 0
        parts.append(
            f"Strategy A: {n_reg} regimes detected.  "
            f"IC spread across regimes = {ic_spread:.4f} "
            f"({'high — strong regime dependence' if ic_spread > 0.02 else 'moderate'})."
        )

    # Strategy B verdict: lift from regime dummies
    if (
        not cond_table.empty
        and "strategy" in cond_table.columns
        and "ic_mean" in cond_table.columns
    ):
        base = cond_table[cond_table["strategy"] == "baseline"]
        aug = cond_table[cond_table["strategy"] == "regime_dummies"]
        if not base.empty and not aug.empty:
            b_ic = float(base["ic_mean"].mean())
            a_ic = float(aug["ic_mean"].mean())
            lift = a_ic - b_ic
            common = base.set_index("model").index.intersection(
                aug.set_index("model").index
            )
            if len(common):
                improve_pct = int(
                    (
                        aug.set_index("model").loc[common, "ic_mean"]
                        - base.set_index("model").loc[common, "ic_mean"]
                        > 0
                    ).mean()
                    * 100
                )
            else:
                improve_pct = 0
            parts.append(
                f"Strategy B: regime dummies shift mean IC by {lift:+.4f} "
                f"({improve_pct}% of models improved)."
            )
            if lift > 0.002:
                parts.append(
                    "RECOMMEND: include regime dummies in feature matrix — "
                    "measurable IC lift with no architectural change."
                )
            elif lift < -0.002:
                parts.append(
                    "WARNING: regime dummies hurt IC — omit or investigate "
                    "collinearity with existing features."
                )
            else:
                parts.append(
                    "NEUTRAL: regime dummies have marginal impact (<0.2 pp IC lift).  "
                    "Consider Strategy A (per-regime models) if per-regime IC spread is high."
                )

    if not parts:
        return "  Insufficient data for verdict."
    return "\n".join(f"  {p}" for p in parts)
