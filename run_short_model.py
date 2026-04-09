"""
Dedicated Short Model Trainer
==============================
Trains a regressor-based short candidate scorer with a fundamentally different
architecture from the long model:

  Long model  → momentum continuation (regressor, 10d horizon, full universe)
  Short model → overextension reversal (regressor, 7d horizon, filtered to overbought stocks)

Key design choices vs the original short_classifier approach:
  1. Regressor target (forward_return): predicts HOW MUCH the stock will decline,
     not just direction.  Lower predicted return = stronger short candidate.
  2. Filtered training universe: only rows where the stock is already overextended
     (high RSI or near 52-week high).  Teaches the model to discriminate within
     the set of genuinely shortable candidates.
  3. 7-day holding horizon: overextension reverts faster than momentum builds.
  4. Dedicated short-alpha features: emphasis on overbought signals, volatility
     exhaustion, and sector relative strength.

Output: output/models/best_short_model.pkl
  Saved as {"estimator": <model>, "feature_columns": [...], "model_type": "short_regressor"}
  Consumed by signals.py via ml_short_model_path config entry.

Usage:
  python run_short_model.py
  python run_short_model.py --horizon 5      # faster reversal (5d)
  python run_short_model.py --limit-tickers 50  # quick smoke-test
"""
from __future__ import annotations

import argparse
import json
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Short-specific feature set — overextension / reversal signals
# ---------------------------------------------------------------------------
SHORT_FEATURES: list[str] = [
    # ── Fundamental deterioration (primary gate — necessary condition) ────
    "f_score",                  # Piotroski composite (0=weakest, 5=strongest); low = short
    "accruals_ratio",           # (Net Income - Op CF) / Assets; HIGH = earnings inflation
    "roa",                      # Return on assets; NEGATIVE = unprofitable
    "delta_roa",                # QoQ ROA change; NEGATIVE = deteriorating profitability
    "delta_leverage",           # QoQ leverage change; POSITIVE = rising debt = stress

    # ── Technical overbought / overextension (timing) ─────────────────────
    "dist_from_52w_high",       # proximity to 52-week high → mean-reversion target
    "rsi_14",                   # RSI level (higher = more overbought)
    "rsi_overbought",           # binary flag: RSI > 70
    "nearness_52w_high",        # George & Hwang 2004 reversal signal

    # ── Volatility exhaustion ─────────────────────────────────────────────
    "vol_expansion",            # recent vol spike → potential exhaustion signal
    "down_up_vol_ratio",        # down-day vs up-day vol ratio (bearish asymmetry)
    "rolling_vol_20",           # baseline realized vol (risk scaling)

    # ── Momentum deceleration ─────────────────────────────────────────────
    "momentum_acceleration",    # ret_5d - ret_10d: deceleration = reversal signal
    "ret_5d",                   # recent run-up
    "ret_20d",                  # medium-term overextension

    # ── Relative / cross-sectional ────────────────────────────────────────
    "sector_relative_20d",      # stock vs sector: relative laggards short better
    "cs_momentum_percentile",   # CS rank: top decile = overextended short candidate
    "capm_residual_vol",        # idiosyncratic risk
]


# ---------------------------------------------------------------------------
# Helpers (thin versions of run_model_selection utilities)
# ---------------------------------------------------------------------------

def _read_config(path: str) -> dict[str, Any]:
    import yaml
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _load_universe(cfg: dict) -> list[str]:
    from utils.universe import load_universe
    return load_universe(cfg)


def _walk_forward_windows(
    dates: pd.Series,
    train_years: float,
    test_years: float,
    step_years: float,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Return (train_start, train_end, test_start, test_end) tuples."""
    dates = pd.to_datetime(dates).sort_values().drop_duplicates()
    global_start = dates.iloc[0]
    global_end = dates.iloc[-1]

    train_delta = pd.DateOffset(years=int(train_years))
    test_delta = pd.DateOffset(years=int(test_years))
    step_delta = pd.DateOffset(years=int(step_years))

    windows = []
    cur = global_start
    while True:
        tr_s = cur
        tr_e = cur + train_delta
        te_s = tr_e
        te_e = te_s + test_delta
        if te_e > global_end + pd.Timedelta(days=1):
            break
        windows.append((tr_s, tr_e, te_s, te_e))
        cur = cur + step_delta
    return windows


def _sharpe(returns: np.ndarray, horizon: int = 7) -> float:
    rets = np.asarray(returns, dtype=float)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 10:
        return float("nan")
    mean_ = float(np.mean(rets))
    std_ = float(np.std(rets, ddof=1))
    if std_ < 1e-12:
        return float("nan")
    scale = np.sqrt(252 / horizon)
    return float(mean_ / std_ * scale)


def _ic(score: np.ndarray, fwd: np.ndarray) -> float:
    mask = np.isfinite(score) & np.isfinite(fwd)
    if mask.sum() < 5:
        return float("nan")
    from scipy.stats import pearsonr
    corr, _ = pearsonr(score[mask], fwd[mask])
    return float(corr)


def _portfolio_daily_returns(
    te: pd.DataFrame,
    max_pos: int,
    horizon: int,
) -> pd.Series:
    """
    Simple long-short portfolio simulation for OOS eval.
    Shorts: bottom `max_pos` by score (lowest predicted return).
    Returns a daily return series.
    """
    parts: list[pd.Series] = []
    for date, grp in te.groupby("date"):
        grp_s = grp.dropna(subset=["score", "forward_return"])
        if grp_s.empty:
            continue
        # Short candidates: lowest predicted return = strongest short signal
        candidates = grp_s.nsmallest(max_pos, "score")
        # Expected P&L: short = -(actual return)
        short_rets = -candidates["forward_return"].values
        if len(short_rets) == 0:
            continue
        parts.append(pd.Series([float(np.mean(short_rets))], index=[date]))
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).sort_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train dedicated short model (overextension regressor)")
    parser.add_argument("--config", default="backtest_config.yaml")
    parser.add_argument("--horizon", type=int, default=7, help="Forward return horizon in trading days")
    parser.add_argument("--limit-tickers", type=int, default=0)
    parser.add_argument("--train-years", type=float, default=12.0)
    parser.add_argument("--test-years", type=float, default=1.0)
    parser.add_argument("--step-years", type=float, default=1.0)
    parser.add_argument(
        "--overbought-filter",
        type=float,
        default=0.0,
        help=(
            "Filter training data to rows where dist_from_52w_high or rsi_14 (z-scored) "
            "exceed this percentile (0=disabled, 0.6=top 40%% most overbought). "
            "Higher values = more targeted but less training data."
        ),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SHORT MODEL TRAINER  (overextension regressor)")
    print("=" * 60)
    print(f"Config  : {args.config}")
    print(f"Horizon : {args.horizon} trading days")
    print(f"Features: {len(SHORT_FEATURES)} short-specific features")
    print(f"Target  : forward_return (negative = price decline = good short)")
    print(f"Filter  : overbought_filter={args.overbought_filter}")

    cfg = _read_config(args.config)
    bt = cfg.get("backtest", {}) or {}
    research = cfg.get("research", {}) or {}

    start_date = str(bt.get("start_date", "2007-01-01"))
    end_date = str(bt.get("end_date", "2022-12-31"))
    train_years = float(research.get("train_years", args.train_years))
    test_years = float(research.get("test_years", args.test_years))
    step_years = float(research.get("step_years", test_years))

    tickers = _load_universe(cfg)
    if int(args.limit_tickers or 0) > 0:
        tickers = tickers[: int(args.limit_tickers)]

    print(f"Universe: {len(tickers)} tickers  |  Window: {start_date} → {end_date}")

    # ------------------------------------------------------------------
    # 1. Build feature matrix
    # ------------------------------------------------------------------
    print("\nBuilding feature matrix …")
    from agents.weight_learning_agent.feature_builder import build_feature_matrix

    df = build_feature_matrix(
        tickers,
        start_date=start_date,
        end_date=end_date,
        holding_period=args.horizon,
        feature_subset=SHORT_FEATURES,
    )
    if df is None or df.empty:
        raise SystemExit("Feature matrix is empty — check data cache and tickers.")

    print(f"Raw matrix: {len(df):,} rows  |  {df.shape[1]} columns")

    # Ensure all short features exist (fill missing with 0)
    for f in SHORT_FEATURES:
        if f not in df.columns:
            print(f"  WARNING: feature '{f}' missing from matrix — filling with 0.0")
            df[f] = 0.0

    # ------------------------------------------------------------------
    # 2. Filter to overextended / overbought candidates
    #    Only train on stocks that are plausible short candidates.
    #    This teaches the model to discriminate WITHIN the shortable universe
    #    rather than learning trivial "this stock is not overbought at all".
    # ------------------------------------------------------------------
    if args.overbought_filter > 0:
        # Use raw dist_from_52w_high and rsi_14 before z-scoring
        # (feature_builder already returns these as lagged per-ticker values)
        mask_52w = pd.Series(False, index=df.index)
        mask_rsi = pd.Series(False, index=df.index)

        if "dist_from_52w_high" in df.columns:
            thr_52w = float(df["dist_from_52w_high"].quantile(args.overbought_filter))
            mask_52w = df["dist_from_52w_high"] >= thr_52w

        if "rsi_14" in df.columns:
            thr_rsi = float(df["rsi_14"].quantile(args.overbought_filter))
            mask_rsi = df["rsi_14"] >= thr_rsi

        overbought_mask = mask_52w | mask_rsi
        n_before = len(df)
        df = df[overbought_mask].copy()
        print(f"Overbought filter (>{args.overbought_filter:.0%} pct): "
              f"{n_before:,} → {len(df):,} rows ({len(df)/n_before:.1%} retained)")

    # ------------------------------------------------------------------
    # 3. Build models to evaluate
    # ------------------------------------------------------------------
    def _build_models():
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler

        models = []

        # Ridge regressor (fast, stable, good baseline)
        models.append((
            "ShortRidge",
            Pipeline([
                ("scaler", RobustScaler()),
                ("model", Ridge(alpha=1.0)),
            ]),
        ))

        # XGBRegressor (main candidate — can capture non-linear overbought patterns)
        try:
            from xgboost import XGBRegressor
            models.append((
                "ShortXGB",
                XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="rmse",
                ),
            ))
        except ImportError:
            print("  NOTE: xgboost not available — using Ridge only")

        return models

    models = _build_models()

    # ------------------------------------------------------------------
    # 4. Walk-forward evaluation
    # ------------------------------------------------------------------
    windows = _walk_forward_windows(
        df["date"], train_years=train_years, test_years=test_years, step_years=step_years
    )
    if len(windows) < 2:
        print("WARNING: fewer than 2 walk-forward windows — widening train ratio")
        windows = _walk_forward_windows(df["date"], train_years=8, test_years=1, step_years=1)

    print(f"\nWalk-forward: {len(windows)} windows  "
          f"(train={train_years}y  test={test_years}y  step={step_years}y)")
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows, 1):
        print(f"  {i:02d}  train=[{tr_s.date()} → {(tr_e - pd.Timedelta(days=1)).date()}]  "
              f"test=[{te_s.date()} → {(te_e - pd.Timedelta(days=1)).date()}]")

    embargo_days = max(5, 2 * args.horizon)
    max_pos = 5  # short book depth for OOS simulation

    feat_cols = SHORT_FEATURES

    results: list[dict] = []
    for name, model in models:
        print(f"\n=== {name} ===")
        wm_sharpe: list[float] = []
        wm_ic: list[float] = []

        for win_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows, 1):
            purge_cutoff = te_s - pd.Timedelta(days=embargo_days)
            tr = df[(df["date"] >= tr_s) & (df["date"] < min(tr_e, purge_cutoff))].copy()
            te = df[(df["date"] >= te_s) & (df["date"] < te_e)].copy()

            if tr.empty or te.empty:
                continue
            if te["date"].nunique() < 20:
                continue

            X_tr = (
                tr[feat_cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .clip(-10.0, 10.0)
                .values
            )
            y_tr = (
                tr["forward_return"]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .clip(-0.3, 0.3)
                .values.astype(float)
            )

            X_te = (
                te[feat_cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .clip(-10.0, 10.0)
                .values
            )
            y_te = (
                te["forward_return"]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .values.astype(float)
            )

            X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
            X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

            try:
                from sklearn.base import clone
                win_model = clone(model)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    win_model.fit(X_tr, y_tr)

                with np.errstate(all="ignore"):
                    score = win_model.predict(X_te).astype(float)

                ic_val = _ic(score, y_te)
                te_scored = te.assign(score=score)
                daily_rets = _portfolio_daily_returns(te_scored, max_pos=max_pos, horizon=args.horizon)
                sharpe_val = _sharpe(daily_rets.values, horizon=args.horizon)

                te_label = f"{te_s.date()}→{(te_e - pd.Timedelta(days=1)).date()}"
                print(f"  [{win_idx:02d}] test={te_label}  "
                      f"IC={ic_val:+.4f}  Sharpe={sharpe_val:.3f}  n_rows={len(te)}")

                if np.isfinite(ic_val):
                    wm_ic.append(ic_val)
                if np.isfinite(sharpe_val):
                    wm_sharpe.append(sharpe_val)

            except Exception as exc:
                print(f"  [{win_idx:02d}] FAILED: {exc}")
                traceback.print_exc()
                continue

        mean_ic = float(np.nanmean(wm_ic)) if wm_ic else float("nan")
        mean_sharpe = float(np.nanmean(wm_sharpe)) if wm_sharpe else float("nan")
        print(f"  → mean IC={mean_ic:+.4f}  mean Sharpe={mean_sharpe:.3f}  windows={len(wm_sharpe)}")
        results.append({
            "name": name,
            "model": model,
            "mean_ic": mean_ic,
            "mean_sharpe": mean_sharpe,
            "n_windows": len(wm_sharpe),
        })

    # ------------------------------------------------------------------
    # 5. Select best model by mean OOS Sharpe (IC as tiebreaker)
    # ------------------------------------------------------------------
    valid = [r for r in results if np.isfinite(r["mean_sharpe"])]
    if not valid:
        # Fallback: select by IC
        valid = [r for r in results if np.isfinite(r["mean_ic"])]
    if not valid:
        raise SystemExit("No valid OOS results — check data and features.")

    best = max(valid, key=lambda r: (r["mean_sharpe"], r["mean_ic"]))
    print(f"\nBest model: {best['name']} | OOS Sharpe={best['mean_sharpe']:.3f} | IC={best['mean_ic']:+.4f}")

    # ------------------------------------------------------------------
    # 6. Retrain best model on FULL dataset and save
    # ------------------------------------------------------------------
    print("\nRetraining on full dataset …")
    final_model = best["model"]
    from sklearn.base import clone
    final_model = clone(final_model)

    X_full = (
        df[feat_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-10.0, 10.0)
        .values
    )
    y_full = (
        df["forward_return"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-0.3, 0.3)
        .values.astype(float)
    )
    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(X_full, y_full)

    out_dir = Path("output/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best_short_model.pkl"

    artifact = {
        "estimator": final_model,
        "feature_columns": feat_cols,
        "model_type": "short_regressor",   # consumed by ensemble_scoring._predict_model
        "model_name": best["name"],
        "horizon_days": args.horizon,
        "oos_sharpe": best["mean_sharpe"],
        "oos_ic": best["mean_ic"],
    }

    import joblib
    joblib.dump(artifact, out_path)
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")

    # Save meta JSON for inspection
    meta_path = out_dir / "best_short_model.meta.json"
    meta = {
        "model_name": best["name"],
        "model_type": "short_regressor",
        "horizon_days": args.horizon,
        "oos_sharpe": float(best["mean_sharpe"]) if np.isfinite(best["mean_sharpe"]) else None,
        "oos_ic": float(best["mean_ic"]) if np.isfinite(best["mean_ic"]) else None,
        "feature_columns": feat_cols,
        "overbought_filter": args.overbought_filter,
        "train_rows": int(len(df)),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Meta  : {meta_path}")

    print("\n" + "=" * 60)
    print("SHORT MODEL TRAINING COMPLETE")
    print(f"  Model      : {best['name']}")
    print(f"  Type       : short_regressor  (lower score = stronger short)")
    print(f"  Horizon    : {args.horizon}d")
    print(f"  OOS Sharpe : {best['mean_sharpe']:.3f}")
    print(f"  OOS IC     : {best['mean_ic']:+.4f}")
    print(f"  Features   : {len(feat_cols)}")
    print(f"  Output     : {out_path}")
    print("=" * 60)
    print()
    print("Next: run the backtest to validate short P&L improvement")
    print("  python run_backtest.py")


if __name__ == "__main__":
    main()
