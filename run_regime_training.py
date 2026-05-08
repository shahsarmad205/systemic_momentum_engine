#!/usr/bin/env python3
"""
Regime-Conditional Long Model Trainer
======================================
Trains a separate XGBRegressor + Ridge for each market regime:
    Bull     — SPY above 200-day MA, low volatility (≈60-65% of days)
    Bear     — SPY below 200-day MA                 (≈10-12% of days)
    Crisis   — High-vol regime (VIX spike)           (≈15-20% of days)
    Sideways — Default / transitional                (≈10-15% of days)

Why regime-conditional training beats one global model:
  - Feature importance shifts by regime: momentum matters in Bull,
    defensive quality matters in Bear, vol signals matter in Crisis
  - A global model averages across regimes → suboptimal in each
  - IC improvement expected: +0.01-0.03 per regime (compounded effect)

Taxonomy mapping (feature_builder → backtester canonical):
    HighVol → Crisis
    Normal  → Sideways
    Bull    → Bull
    Bear    → Bear

Output:
    output/models/regime_bull_long.pkl
    output/models/regime_bear_long.pkl
    output/models/regime_crisis_long.pkl
    output/models/regime_sideways_long.pkl

Usage:
    python run_regime_training.py
    python run_regime_training.py --limit-tickers 100   # fast dev run
    python run_regime_training.py --config backtest_config.yaml
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Taxonomy: feature_builder labels → canonical backtester labels
_FB_TO_CANONICAL = {
    "Bull": "Bull",
    "Bear": "Bear",
    "HighVol": "Crisis",
    "Normal": "Sideways",
    "Crisis": "Crisis",
    "Sideways": "Sideways",
}

CANONICAL_REGIMES = ["Bull", "Bear", "Crisis", "Sideways"]
MIN_TRAIN_ROWS = 2_000   # below this → skip regime model, global model used as fallback


# ---------------------------------------------------------------------------
# Helpers (reuse patterns from run_model_selection.py)
# ---------------------------------------------------------------------------

def _sanitize(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.clip(
            np.where(np.isfinite(X), X, 0.0),
            -10.0, 10.0
        ),
        nan=0.0, posinf=0.0, neginf=0.0,
    ).copy()


def _safe_ic(pred: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() < 10:
        return float("nan")
    try:
        return float(np.corrcoef(pred[mask], target[mask])[0, 1])
    except Exception:
        return float("nan")


def _oos_sharpe(pred: np.ndarray, target: np.ndarray, dates: pd.Series) -> float:
    """Daily long-minus-short portfolio Sharpe from cross-sectional scores."""
    df = pd.DataFrame({"date": dates, "score": pred, "ret": target})
    daily = []
    for dt, g in df.groupby("date"):
        if len(g) < 6:
            continue
        med = g["score"].median()
        long_ret = g.loc[g["score"] >= med, "ret"].mean()
        short_ret = g.loc[g["score"] < med, "ret"].mean()
        daily.append(long_ret - short_ret)
    if len(daily) < 20:
        return float("nan")
    s = pd.Series(daily)
    return float((s.mean() / s.std()) * np.sqrt(252)) if s.std() > 0 else float("nan")


def _drop_singular(X: np.ndarray, feat_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    stds = X.std(axis=0)
    mask = stds > 1e-6
    return X[:, mask], [f for f, m in zip(feat_cols, mask) if m]


def _recency_weights(dates: pd.Series, half_life_years: float = 4.0) -> np.ndarray:
    """Exponential decay weights — recent samples count more."""
    t_max = dates.max()
    age_years = (t_max - dates).dt.days / 365.25
    lam = np.log(2) / half_life_years
    w = np.exp(-lam * age_years.values)
    return (w / w.sum() * len(w)).astype(float)   # normalise so mean ≈ 1


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_xgb_regressor(n_rows: int) -> object:
    from xgboost import XGBRegressor
    # Fewer trees for data-scarce regimes to reduce overfit
    n_est = 300 if n_rows > 20_000 else 150
    return XGBRegressor(
        n_estimators=n_est,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        eval_metric="mae",
        verbosity=0,
        random_state=42,
        tree_method="hist",
    )


def _build_ridge() -> object:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])


# ---------------------------------------------------------------------------
# Walk-forward evaluation for one regime's dataset
# ---------------------------------------------------------------------------

def _walk_forward_regime(
    rdf: pd.DataFrame,
    feat_cols: list[str],
    regime_name: str,
    train_years: float = 8.0,
    test_years: float = 1.5,
    step_years: float = 1.5,
) -> dict[str, dict]:
    """
    Walk-forward validate XGBRegressor and Ridge on `rdf` (regime-filtered rows).
    Returns per-model metrics dict: {name: {oos_ic, oos_sharpe, windows}}.
    """
    dates = pd.to_datetime(rdf["date"])
    date_min = dates.min()
    date_max = dates.max()

    # Build time windows from the regime's own date range
    # Use Timedelta (days) to avoid pd.DateOffset rejecting float years
    train_days = int(train_years * 365.25)
    test_days  = int(test_years  * 365.25)
    step_days  = int(step_years  * 365.25)

    windows = []
    cursor = date_min
    while True:
        tr_end = cursor + pd.Timedelta(days=train_days)
        te_end = tr_end + pd.Timedelta(days=test_days)
        if te_end > date_max:
            break
        windows.append((cursor, tr_end, tr_end, te_end))
        cursor += pd.Timedelta(days=step_days)

    # If data span < train+test, fall back to a single 70/30 split
    if not windows:
        split = int(len(rdf) * 0.70)
        if split < 500 or (len(rdf) - split) < 200:
            return {}
        sorted_df = rdf.sort_values("date")
        split_date = pd.to_datetime(sorted_df["date"].iloc[split])
        windows = [(date_min, split_date, split_date, date_max)]

    candidates = [("XGB", _build_xgb_regressor(len(rdf))), ("Ridge", _build_ridge())]
    results: dict[str, dict] = {}

    for model_name, model in candidates:
        ic_list, sharpe_list = [], []
        for (tr_s, tr_e, te_s, te_e) in windows:
            embargo = pd.Timedelta(days=7)
            tr = rdf[(rdf["date"] >= tr_s) & (rdf["date"] < tr_e - embargo)]
            te = rdf[(rdf["date"] >= te_s) & (rdf["date"] < te_e)]
            if len(tr) < 500 or len(te) < 100:
                continue

            X_tr = _sanitize(tr[feat_cols].values)
            y_tr = tr["forward_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.3, 0.3).values.astype(float)
            y_te = te["forward_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(float)

            X_tr_clean, active = _drop_singular(X_tr, feat_cols)
            X_te_clean = _sanitize(te[active].values)

            w_tr = _recency_weights(pd.to_datetime(tr["date"]))

            try:
                if model_name == "XGB":
                    model.fit(X_tr_clean, y_tr, sample_weight=w_tr)
                else:
                    model.fit(X_tr_clean, y_tr, **{"ridge__sample_weight": w_tr} if hasattr(model, "steps") else {})
            except TypeError:
                model.fit(X_tr_clean, y_tr)

            pred = np.asarray(model.predict(X_te_clean), dtype=float)
            ic_list.append(_safe_ic(pred, y_te))
            sharpe_list.append(_oos_sharpe(pred, y_te, te["date"]))

        if not ic_list:
            continue

        results[model_name] = {
            "oos_ic": float(np.nanmean(ic_list)),
            "oos_sharpe": float(np.nanmean(sharpe_list)),
            "windows": len(ic_list),
        }
        ic_str = " | ".join(f"IC={v:.4f}" for v in ic_list)
        sh_str = " | ".join(f"Sh={v:.3f}" for v in sharpe_list if np.isfinite(v))
        print(f"    {model_name}: {ic_str}  {sh_str}")

    return results


# ---------------------------------------------------------------------------
# Train final model on all regime data with recency weighting
# ---------------------------------------------------------------------------

def _train_final_model(
    rdf: pd.DataFrame,
    feat_cols: list[str],
    model_name: str,
) -> tuple[object, list[str]]:
    X = _sanitize(rdf[feat_cols].values)
    y = rdf["forward_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.3, 0.3).values.astype(float)
    X, active = _drop_singular(X, feat_cols)
    w = _recency_weights(pd.to_datetime(rdf["date"]))

    if model_name == "XGB":
        model = _build_xgb_regressor(len(rdf))
        model.fit(X, y, sample_weight=w)
    else:
        model = _build_ridge()
        try:
            model.fit(X, y, ridge__sample_weight=w)
        except TypeError:
            model.fit(X, y)

    return model, active


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Regime-conditional long model trainer")
    ap.add_argument("--config", default="backtest_config.yaml")
    ap.add_argument("--limit-tickers", type=int, default=0,
                    help="Cap universe size for dev runs (0 = full universe)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        sys.exit(f"Config not found: {cfg_path}")

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    research = cfg.get("research", {})
    start_date = str(research.get("start_date", "2008-01-01"))
    end_date   = str(research.get("end_date",   "2022-12-31"))
    holding_period = int(research.get("holding_period_days", 5))
    feature_sel = cfg.get("feature_selection", {})
    feature_subset = [str(c).strip() for c in (feature_sel.get("feature_subset") or []) if str(c).strip()]

    print("=" * 60)
    print("  REGIME-CONDITIONAL LONG MODEL TRAINER")
    print("=" * 60)
    print(f"  Period : {start_date} → {end_date}")
    print(f"  Horizon: {holding_period}d")
    print()

    # Build feature matrix
    from utils.universe import load_universe
    from agents.weight_learning_agent.feature_builder import build_feature_matrix

    universe = load_universe(cfg)
    if args.limit_tickers > 0:
        universe = universe[: args.limit_tickers]
        print(f"  [dev] limited to {len(universe)} tickers")

    print(f"  Building feature matrix for {len(universe)} tickers…")
    df = build_feature_matrix(
        tickers=universe,
        start_date=start_date,
        end_date=end_date,
        holding_period=holding_period,
        feature_subset=feature_subset or None,
    )
    print(f"  Raw matrix: {len(df):,} rows  |  {df.shape[1]} columns")

    # Map feature_builder taxonomy → canonical backtester taxonomy
    if "regime_label" in df.columns:
        df["regime_label"] = df["regime_label"].map(_FB_TO_CANONICAL).fillna("Sideways")
    else:
        df["regime_label"] = "Sideways"

    print("\n  Regime distribution:")
    for reg in CANONICAL_REGIMES:
        n = (df["regime_label"] == reg).sum()
        pct = 100 * n / len(df)
        print(f"    {reg:<10}: {n:>8,} rows  ({pct:.1f}%)")

    # Identify feature columns (exclude identifiers and targets)
    base_exclude = {
        "date", "ticker", "sector", "direction",
        "regime_label", "y_bin",
        "forward_return", "forward_return_risk_adj",
        "forward_return_excess", "spy_forward_5d",
        "is_high_vol_regime",
    }
    feat_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in base_exclude and "forward" not in c.lower()
    ]
    if feature_subset:
        feat_cols = [c for c in feature_subset if c in feat_cols]
    if not feat_cols:
        sys.exit("No feature columns found.")

    leaked = [c for c in feat_cols if "forward" in c.lower()]
    if leaked:
        sys.exit(f"Leakage detected: {leaked}")

    print(f"\n  Feature columns: {len(feat_cols)}")

    out_dir = Path("output/models")
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, str] = {}

    for regime in CANONICAL_REGIMES:
        rdf = df[df["regime_label"] == regime].copy()
        n = len(rdf)
        print(f"\n{'─'*60}")
        print(f"  [{regime.upper()}]  n={n:,} rows")

        if n < MIN_TRAIN_ROWS:
            print(f"    ✗ Skipping — {n} < {MIN_TRAIN_ROWS} minimum (global model will be used as fallback)")
            continue

        # Walk-forward evaluation
        print(f"  Walk-forward OOS validation:")
        metrics = _walk_forward_regime(rdf, feat_cols, regime)
        if not metrics:
            print("    ✗ No valid windows — skipping")
            continue

        # Pick best model by OOS IC
        best_name = max(metrics, key=lambda k: metrics[k]["oos_ic"] if np.isfinite(metrics[k]["oos_ic"]) else -999)
        best = metrics[best_name]
        print(f"\n  → Best: {best_name}  OOS IC={best['oos_ic']:.4f}  OOS Sharpe={best['oos_sharpe']:.3f}  windows={best['windows']}")

        # Train final model on all regime data
        print(f"  Retraining {best_name} on all {n:,} {regime} rows…")
        estimator, active_feats = _train_final_model(rdf, feat_cols, best_name)

        path = out_dir / f"regime_{regime.lower()}_long.pkl"
        artifact = {
            "model_name": f"{best_name}_{regime}",
            "model_type": "regressor",
            "regime": regime,
            "horizon_days": holding_period,
            "target": "forward_return",
            "feature_columns": active_feats,
            "n_train": n,
            "oos_ic": best["oos_ic"],
            "oos_sharpe": best["oos_sharpe"],
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "estimator": estimator,
        }
        with open(path, "wb") as fh:
            pickle.dump(artifact, fh)

        saved_paths[regime] = str(path)
        print(f"  Saved: {path.name}  ({path.stat().st_size / 1024:.0f} KB)")

    print(f"\n{'='*60}")
    print("  REGIME MODEL TRAINING COMPLETE")
    print(f"{'='*60}")
    for regime, path in saved_paths.items():
        m = _load_meta(path)
        print(f"  {regime:<10}: {Path(path).name}  IC={m.get('oos_ic', float('nan')):.4f}  Sharpe={m.get('oos_sharpe', float('nan')):.3f}")

    print()
    print("  Regime routing is active via ml_regime_models_dir in backtest_config.yaml")
    print("  Next: python run_backtest.py")


def _load_meta(path: str) -> dict:
    try:
        with open(path, "rb") as fh:
            a = pickle.load(fh)
        return {k: v for k, v in a.items() if k != "estimator"}
    except Exception:
        return {}


if __name__ == "__main__":
    main()
