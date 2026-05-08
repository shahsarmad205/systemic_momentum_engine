"""
Meta-labeling framework for signal reliability scaling.

Base model outputs a continuous score. Meta-model predicts the reliability of
that score from six meta-features (score dispersion, recent IC, recent alpha
capture proxy, market volatility, liquidity, model uncertainty).

Scaled exposure = base_score × meta_multiplier, where the multiplier is
learned walk-forward from the distribution of past predictions. No thresholds
are hardcoded — quantile mapping is fit from training data each window.

Architecture
------------
  walk-forward window t:
    1. Fit base model on train_t  →  base scored frame
    2. Train meta-model on windows 1..t-1 (X = meta-features, y = daily IC)
    3. Predict meta-multiplier for window t
    4. Scaled score = base_score × meta_multiplier  (clipped to [min, max])

  Window 1 receives full exposure (max_multiplier) — no prior history.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class MetaLabelConfig:
    target_col: str = "target_return"
    return_col: str = "daily_return"
    adv_col: str = "adv_dollar_20"
    vol_col: str = "realised_vol_20d"
    meta_model_type: str = "ridge"   # "ridge" | "lgbm" | "rf"
    ic_ema_span_short: int = 10      # days for short EMA of IC history
    ic_ema_span_long: int = 25       # days for long EMA of IC history
    min_train_dates: int = 20        # min daily dates to fit meta-model
    min_multiplier: float = 0.0
    max_multiplier: float = 1.0
    multiplier_q_low: float = 0.10   # training pred quantile → min_multiplier
    multiplier_q_high: float = 0.90  # training pred quantile → max_multiplier
    turnover_top_q: float = 0.20     # top-q bucket for turnover metric


# ── Meta-feature computation ──────────────────────────────────────────────────

def _daily_ic_series(scored_df: pd.DataFrame, target_col: str) -> pd.Series:
    """Rank IC per date from a scored frame. Returns Series indexed by date."""
    ics: dict[pd.Timestamp, float] = {}
    for dt, g in (
        scored_df.dropna(subset=["score", target_col]).groupby("date", sort=True)
    ):
        if len(g) < 8 or g["score"].nunique() < 2 or g[target_col].nunique() < 2:
            continue
        ics[dt] = float(g["score"].rank().corr(g[target_col].rank()))
    return pd.Series(ics, name="ic").sort_index()


def _compute_daily_meta_features(
    scored_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    cfg: MetaLabelConfig,
) -> pd.DataFrame:
    """
    Compute daily meta-features for the meta-model.

    Features (all derived from scored_df + panel_df, no alpha signals):
      score_cs_std   — cross-sectional score dispersion (signal strength proxy)
      score_cs_iqr   — cross-sectional IQR (model uncertainty proxy)
      score_cs_skew  — cross-sectional skewness (directional bias)
      ic_ema_short   — short EMA of historical IC (lagged 1 day)
      ic_ema_long    — long EMA of historical IC (lagged 1 day)
      mkt_vol        — cross-sectional mean of realised_vol
      liq_proxy      — log(median ADV) — liquidity regime
      cs_ret_std     — cross-sectional std of daily returns (dispersion)
    """
    scored_df = scored_df.copy()
    scored_df["date"] = pd.to_datetime(scored_df["date"], errors="coerce")

    # Per-date score distribution statistics
    def _iqr(x: pd.Series) -> float:
        return float(x.quantile(0.75) - x.quantile(0.25))

    cs = (
        scored_df.groupby("date", sort=True)["score"]
        .agg(score_cs_std=("std"), score_cs_skew=("skew"))
    )
    cs["score_cs_iqr"] = scored_df.groupby("date", sort=True)["score"].apply(_iqr)

    # IC EMA features — shift(1) to avoid lookahead: ic[d] is computed after d+1
    ic_series = _daily_ic_series(scored_df, cfg.target_col)
    cs["ic_ema_short"] = (
        ic_series.ewm(span=cfg.ic_ema_span_short, min_periods=3).mean().shift(1)
    )
    cs["ic_ema_long"] = (
        ic_series.ewm(span=cfg.ic_ema_span_long, min_periods=5).mean().shift(1)
    )

    # Recent alpha capture proxy: top-q return / cross-section return (lagged 1 day)
    def _ac_per_date(g: pd.DataFrame) -> float:
        if len(g) < 10:
            return float("nan")
        threshold = g["score"].quantile(0.80)
        top_ret = float(g.loc[g["score"] >= threshold, cfg.target_col].mean())
        all_ret = float(g[cfg.target_col].mean())
        return top_ret / abs(all_ret) if abs(all_ret) > 1e-8 else float("nan")

    ac_series = (
        scored_df.dropna(subset=["score", cfg.target_col])
        .groupby("date", sort=True)
        .apply(_ac_per_date)
        .rename("ac_proxy")
        .dropna()
    )
    cs["ac_ema"] = ac_series.ewm(span=cfg.ic_ema_span_short, min_periods=3).mean().shift(1)

    # Market context from panel (daily aggregates)
    panel_df = panel_df.copy()
    panel_df["date"] = pd.to_datetime(panel_df["date"], errors="coerce")
    mkt_parts: list[pd.Series | pd.DataFrame] = []

    if cfg.vol_col in panel_df.columns:
        mkt_parts.append(
            panel_df.groupby("date")[cfg.vol_col]
            .agg(mkt_vol="mean", mkt_vol_std="std")
        )

    if cfg.adv_col in panel_df.columns:
        adv = panel_df.groupby("date")[cfg.adv_col].median()
        mkt_parts.append(np.log(adv.clip(lower=1e3)).rename("liq_proxy"))

    if cfg.return_col in panel_df.columns:
        mkt_parts.append(
            panel_df.groupby("date")[cfg.return_col].std().rename("cs_ret_std")
        )

    if mkt_parts:
        mkt = pd.concat(mkt_parts, axis=1)
        cs = cs.join(mkt, how="left")

    return cs.ffill().fillna(0.0)


# ── Meta training data construction ──────────────────────────────────────────

def _build_meta_train_data(
    scored_frames: list[pd.DataFrame],
    panel_df: pd.DataFrame,
    cfg: MetaLabelConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build (X_meta, y_meta) from accumulated scored frames.

    y = daily IC (rank corr of score vs target per date)
    X = contemporaneous meta-features (IC EMA features are lagged internally)
    """
    if not scored_frames:
        return pd.DataFrame(), pd.Series(dtype=float)

    combined = pd.concat(scored_frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")

    X = _compute_daily_meta_features(combined, panel_df, cfg)
    y = _daily_ic_series(combined, cfg.target_col)

    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]

    mask = y.notna() & X.notna().all(axis=1)
    return X[mask], y[mask]


# ── Meta-model fitting and prediction ─────────────────────────────────────────

def _fit_meta_model(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: MetaLabelConfig,
) -> tuple[Any | None, Any | None, float, float]:
    """
    Fit meta-model predicting daily IC from meta-features.

    Returns (model, scaler, q_low_pred, q_high_pred).
    q_low/q_high are quantiles of training-set predictions used to map
    raw predictions to [min_multiplier, max_multiplier] without hardcoding.
    """
    from sklearn.preprocessing import StandardScaler

    if len(X) < cfg.min_train_dates:
        return None, None, 0.0, 1.0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    if cfg.meta_model_type == "lgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(
            n_estimators=60, max_depth=3, learning_rate=0.05, n_jobs=1, verbose=-1
        )
    elif cfg.meta_model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=60, max_depth=4, n_jobs=1)
    else:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)

    model.fit(X_scaled, y.values)
    preds = model.predict(X_scaled)
    q_low  = float(np.nanpercentile(preds, cfg.multiplier_q_low * 100))
    q_high = float(np.nanpercentile(preds, cfg.multiplier_q_high * 100))

    return model, scaler, q_low, q_high


def _predict_multiplier(
    model: Any | None,
    scaler: Any | None,
    X_test: pd.DataFrame,
    q_low: float,
    q_high: float,
    cfg: MetaLabelConfig,
) -> pd.Series:
    """Map meta-model predictions to exposure multiplier ∈ [min_mult, max_mult]."""
    if model is None or X_test.empty:
        return pd.Series(cfg.max_multiplier, index=X_test.index)

    X_scaled = scaler.transform(X_test.fillna(0.0).values)
    raw_pred = model.predict(X_scaled)

    span = q_high - q_low if abs(q_high - q_low) > 1e-9 else 1.0
    mult = cfg.min_multiplier + (
        (np.clip(raw_pred, q_low, q_high) - q_low) / span
        * (cfg.max_multiplier - cfg.min_multiplier)
    )
    return pd.Series(mult, index=X_test.index)


def _extract_feature_importance(
    model: Any, feature_names: list[str], window: int
) -> list[dict]:
    rows: list[dict] = []
    if hasattr(model, "coef_"):
        for name, coef in zip(feature_names, model.coef_):
            rows.append({"feature": name, "importance": abs(float(coef)), "window": window})
    elif hasattr(model, "feature_importances_"):
        for name, imp in zip(feature_names, model.feature_importances_):
            rows.append({"feature": name, "importance": float(imp), "window": window})
    return rows


# ── Walk-forward meta-labeling pipeline ──────────────────────────────────────

@dataclass
class MetaLabelResult:
    base_frames: list[pd.DataFrame]
    scaled_frames: list[pd.DataFrame]
    meta_feature_importance: pd.DataFrame
    base_metrics: dict[str, float]
    meta_metrics: dict[str, float]
    comparison: pd.DataFrame
    multiplier_stats: pd.DataFrame


def build_meta_labeled_scores(
    df: pd.DataFrame,
    base_model: Any,
    base_model_name: str,
    base_model_kind: str,
    base_uses_proba: bool,
    feature_cols: list[str],
    wf_cfg: Any,
    meta_cfg: MetaLabelConfig,
) -> MetaLabelResult:
    """
    Walk-forward meta-labeling pipeline.

    For each window t:
      1. Fit base model on train_t, score test_t  →  base scored frame
      2. Train meta-model on windows 1..t-1
      3. Compute meta-multiplier for test_t from combined history
      4. Scaled score = base_score × meta_multiplier

    Window 1: no prior history, multiplier = max_multiplier (full exposure).
    """
    from model_selection.objective_comparison import (
        _fit_one_window,
        _walk_forward_dates,
    )

    windows = _walk_forward_dates(df, wf_cfg)
    if not windows:
        raise ValueError("No walk-forward windows produced by _walk_forward_dates()")

    base_frames: list[pd.DataFrame] = []
    scaled_frames: list[pd.DataFrame] = []
    mult_stats_rows: list[dict] = []
    all_feat_importance: list[dict] = []

    # Incremental buffer: replaces O(windows²) pd.concat(base_frames + [base_frame])
    accumulated_hist: pd.DataFrame = pd.DataFrame()

    last_model: Any = None
    last_scaler: Any = None
    last_q_low = 0.0
    last_q_high = 1.0

    for w_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
        train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
        test_df  = df[(df["date"] >= test_start)  & (df["date"] <= test_end)]

        base_frame = _fit_one_window(
            base_model_name, base_model, base_model_kind,
            train_df, test_df, feature_cols,
            meta_cfg.target_col, wf_cfg,
        )
        if base_frame is None or base_frame.empty:
            continue

        base_frame["date"] = pd.to_datetime(base_frame["date"], errors="coerce")
        test_dates = base_frame["date"].dropna().unique()

        if accumulated_hist.empty:
            # No history: full exposure, no scaling
            multiplier = pd.Series(
                meta_cfg.max_multiplier,
                index=pd.to_datetime(test_dates),
            )
        else:
            # Train meta-model on accumulated history (all prior windows)
            X_tr, y_tr = _build_meta_train_data([accumulated_hist], df, meta_cfg)
            if not X_tr.empty:
                last_model, last_scaler, last_q_low, last_q_high = _fit_meta_model(
                    X_tr, y_tr, meta_cfg
                )
                if last_model is not None:
                    fi = _extract_feature_importance(last_model, X_tr.columns.tolist(), w_idx)
                    all_feat_importance.extend(fi)

            # Compute test meta-features from history + current frame (IC EMA lagged internally)
            combined_for_features = pd.concat([accumulated_hist, base_frame], ignore_index=True)
            X_test_all = _compute_daily_meta_features(combined_for_features, df, meta_cfg)
            X_test = X_test_all.loc[X_test_all.index.isin(pd.to_datetime(test_dates))]

            multiplier = _predict_multiplier(
                last_model, last_scaler, X_test,
                last_q_low, last_q_high, meta_cfg,
            )

        # Apply scaling
        scaled_frame = base_frame.copy()
        scaled_frame["meta_multiplier"] = (
            scaled_frame["date"].map(multiplier).fillna(meta_cfg.max_multiplier)
        )
        scaled_frame["base_score"] = scaled_frame["score"].copy()
        scaled_frame["score"] = scaled_frame["score"] * scaled_frame["meta_multiplier"]

        # Append to incremental buffer AFTER computing test features (OOS integrity)
        accumulated_hist = (
            pd.concat([accumulated_hist, base_frame], ignore_index=True)
            if not accumulated_hist.empty
            else base_frame.copy()
        )
        base_frames.append(base_frame)
        scaled_frames.append(scaled_frame)

        mv = scaled_frame["meta_multiplier"]
        mult_stats_rows.append({
            "window":    w_idx + 1,
            "mult_mean": round(float(mv.mean()), 4),
            "mult_std":  round(float(mv.std()),  4),
            "mult_min":  round(float(mv.min()),  4),
            "mult_max":  round(float(mv.max()),  4),
        })

    base_metrics  = _aggregate_oos_metrics(base_frames,  "score",      meta_cfg)
    meta_metrics  = _aggregate_oos_metrics(scaled_frames, "score",      meta_cfg)
    feat_imp_df   = _summarise_importance(all_feat_importance)
    comparison    = _build_comparison_table(base_metrics, meta_metrics)
    mult_stats_df = pd.DataFrame(mult_stats_rows)

    return MetaLabelResult(
        base_frames=base_frames,
        scaled_frames=scaled_frames,
        meta_feature_importance=feat_imp_df,
        base_metrics=base_metrics,
        meta_metrics=meta_metrics,
        comparison=comparison,
        multiplier_stats=mult_stats_df,
    )


def _summarise_importance(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame(rows)
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )


# ── OOS metric helpers ────────────────────────────────────────────────────────

def _aggregate_oos_metrics(
    frames: list[pd.DataFrame],
    score_col: str,
    cfg: MetaLabelConfig,
) -> dict[str, float]:
    nan = float("nan")
    blank: dict[str, float] = {
        k: nan for k in ("ic_mean", "ic_tstat", "sharpe", "max_drawdown", "turnover", "alpha_capture")
    }
    if not frames:
        return blank

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined[score_col] = pd.to_numeric(combined[score_col], errors="coerce")

    ic_s = _daily_ic_series(combined.rename(columns={score_col: "score"}), cfg.target_col)
    ic_mean  = float(ic_s.mean()) if not ic_s.empty else nan
    ic_tstat = (
        float(ic_s.mean() / (ic_s.std() / math.sqrt(len(ic_s))))
        if len(ic_s) > 1 else nan
    )
    sharpe       = _long_sharpe(combined, score_col, cfg.target_col)
    max_dd       = _max_drawdown(combined, score_col, cfg.target_col)
    turnover     = _top_q_turnover(combined, score_col, cfg.turnover_top_q)
    alpha_cap    = _alpha_capture(combined, score_col, cfg.target_col)

    return {
        "ic_mean": ic_mean, "ic_tstat": ic_tstat,
        "sharpe": sharpe, "max_drawdown": max_dd,
        "turnover": turnover, "alpha_capture": alpha_cap,
    }


def _long_sharpe(
    df: pd.DataFrame, score_col: str, target_col: str
) -> float:
    if target_col not in df.columns:
        return float("nan")
    pnl: list[float] = []
    for _, g in df.dropna(subset=[score_col, target_col]).groupby("date", sort=True):
        if len(g) < 5:
            continue
        top = g.loc[g[score_col] >= g[score_col].quantile(0.80), target_col]
        if len(top):
            pnl.append(float(top.mean()))
    if len(pnl) < 5:
        return float("nan")
    arr = np.array(pnl)
    std = arr.std()
    return float("nan") if std < 1e-10 else float(arr.mean() / std * math.sqrt(252))


def _max_drawdown(
    df: pd.DataFrame, score_col: str, target_col: str
) -> float:
    if target_col not in df.columns:
        return float("nan")
    pnl_by_dt: dict[Any, float] = {}
    for dt, g in df.dropna(subset=[score_col, target_col]).groupby("date", sort=True):
        if len(g) < 5:
            continue
        top = g.loc[g[score_col] >= g[score_col].quantile(0.80), target_col]
        if len(top):
            pnl_by_dt[dt] = float(top.mean())
    if len(pnl_by_dt) < 5:
        return float("nan")
    cum = np.cumsum(list(pnl_by_dt.values()))
    dd  = cum - np.maximum.accumulate(cum)
    return float(dd.min())


def _top_q_turnover(
    df: pd.DataFrame, score_col: str, top_q: float
) -> float:
    dates = sorted(df["date"].dropna().unique())
    if len(dates) < 2:
        return float("nan")
    turnovers: list[float] = []
    prev: set[str] = set()
    for dt in dates:
        g = df.loc[df["date"] == dt, ["ticker", score_col]].dropna()
        if len(g) < 5:
            continue
        cur = set(
            g.loc[g[score_col] >= g[score_col].quantile(1 - top_q), "ticker"].astype(str)
        )
        if prev:
            union = prev | cur
            if union:
                turnovers.append(1 - len(prev & cur) / len(union))
        prev = cur
    return float(np.mean(turnovers)) if turnovers else float("nan")


def _alpha_capture(
    df: pd.DataFrame, score_col: str, target_col: str
) -> float:
    if target_col not in df.columns:
        return float("nan")
    ratios: list[float] = []
    for _, g in df.dropna(subset=[score_col, target_col]).groupby("date", sort=True):
        if len(g) < 10:
            continue
        top_ret = float(g.loc[g[score_col] >= g[score_col].quantile(0.80), target_col].mean())
        all_ret = float(g[target_col].mean())
        if abs(all_ret) > 1e-8:
            ratios.append(top_ret / abs(all_ret))
    return float(np.mean(ratios)) if ratios else float("nan")


# ── Comparison table ──────────────────────────────────────────────────────────

def _build_comparison_table(
    base: dict[str, float], meta: dict[str, float]
) -> pd.DataFrame:
    metrics = [
        ("ic_mean",       "higher"),
        ("ic_tstat",      "higher"),
        ("sharpe",        "higher"),
        ("max_drawdown",  "higher"),   # less negative = better
        ("turnover",      "lower"),
        ("alpha_capture", "higher"),
    ]
    rows: list[dict] = []
    for m, direction in metrics:
        b = base.get(m, float("nan"))
        a = meta.get(m, float("nan"))
        if np.isfinite(b) and np.isfinite(a) and abs(b) > 1e-10:
            if direction == "lower":
                delta_pct = (b - a) / abs(b) * 100
            else:
                delta_pct = (a - b) / abs(b) * 100
        else:
            delta_pct = float("nan")
        rows.append({
            "metric":    m,
            "direction": direction,
            "base":      round(b, 4),
            "meta":      round(a, 4),
            "delta_pct": round(delta_pct, 1),
        })
    return pd.DataFrame(rows)


# ── Evaluation & reporting ────────────────────────────────────────────────────

def evaluate_meta_labeling(result: MetaLabelResult) -> pd.DataFrame:
    """Return the before/after comparison DataFrame."""
    return result.comparison


def format_meta_report(result: MetaLabelResult) -> str:
    lines: list[str] = ["=" * 62]
    lines.append("META-LABELING FRAMEWORK — SIGNAL RELIABILITY SCALING")
    lines.append("=" * 62)

    lines.append("\n── Before vs After ──────────────────────────────────────────")
    if not result.comparison.empty:
        lines.append(f"  {'Metric':<20}  {'Base':>10}  {'Meta':>10}  {'Δ%':>8}  {'Better?':>7}")
        lines.append("  " + "─" * 56)
        for _, row in result.comparison.iterrows():
            improved = _metric_improved(row["metric"], row["direction"], row["base"], row["meta"])
            flag = "  YES" if improved else "   —"
            dp = f"{row['delta_pct']:>7.1f}%" if np.isfinite(row["delta_pct"]) else "    n/a"
            lines.append(
                f"  {str(row['metric']):<20}  {row['base']:>10.4f}  {row['meta']:>10.4f}"
                f"  {dp}  {flag}"
            )

    lines.append("\n── Meta-Model Feature Importance ────────────────────────────")
    if not result.meta_feature_importance.empty:
        fi = result.meta_feature_importance
        max_imp = fi["importance"].max()
        for _, row in fi.head(10).iterrows():
            bar = "█" * max(1, int(row["importance"] / max_imp * 20))
            lines.append(f"  {str(row['feature']):<22}  {row['importance']:.4f}  {bar}")
    else:
        lines.append("  (no feature importance — only 1 window used, no meta-model trained)")

    lines.append("\n── Multiplier Statistics per Window ─────────────────────────")
    if not result.multiplier_stats.empty:
        for _, row in result.multiplier_stats.iterrows():
            lines.append(
                f"  Window {int(row['window']):<2}: "
                f"mean={row['mult_mean']:.3f}  std={row['mult_std']:.3f}  "
                f"[{row['mult_min']:.3f}, {row['mult_max']:.3f}]"
            )

    lines.append("\n── Verdict ──────────────────────────────────────────────────")
    lines.append(_meta_verdict(result))
    lines.append("=" * 62)
    return "\n".join(lines)


def _metric_improved(metric: str, direction: str, base: float, meta: float) -> bool:
    if not (np.isfinite(base) and np.isfinite(meta)):
        return False
    return meta < base if direction == "lower" else meta > base


def _meta_verdict(result: MetaLabelResult) -> str:
    improvements = total = 0
    for _, row in result.comparison.iterrows():
        b, a = row["base"], row["meta"]
        if not (np.isfinite(b) and np.isfinite(a)):
            continue
        total += 1
        if _metric_improved(row["metric"], row["direction"], b, a):
            improvements += 1
    if total == 0:
        return "  Verdict: Insufficient data"
    ratio = improvements / total
    if ratio >= 0.67:
        return (
            f"  Verdict: POSITIVE ({improvements}/{total} metrics improved) — "
            "meta-scaling adds value; reduce exposure when signal unreliable"
        )
    if ratio >= 0.50:
        return (
            f"  Verdict: MIXED ({improvements}/{total} metrics improved) — "
            "conditional value; inspect feature importance for calibration"
        )
    return (
        f"  Verdict: NEGATIVE ({improvements}/{total} metrics improved) — "
        "base model preferred; meta-features may lack regime variation"
    )
