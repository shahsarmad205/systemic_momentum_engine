#!/usr/bin/env python3
"""
Walk-forward validation: Ridge vs GBR vs XGB vs StackedEnsemble (OOS only).

Analysis-only script; does not change training or backtest pipeline.

Run from trend_signal_engine root:
    python analysis/ensemble_validation.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, ttest_1samp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.weight_learning_agent import build_feature_matrix  # noqa: E402
from agents.weight_learning_agent.ensemble_model import StackedEnsemble  # noqa: E402
from agents.weight_learning_agent.weight_model import (  # noqa: E402
    COMPOUND_AND_PRICE_FEATURES,
    TARGET,
)

try:
    from xgboost import XGBRegressor
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install xgboost: pip install xgboost") from e


def _load_tickers_and_dates() -> tuple[list[str], str, str]:
    cfg_path = ROOT / "backtest_config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    tickers = list(c.get("tickers") or [])
    return tickers, "2013-01-01", "2024-01-01"


def _select_features_ic(train_df: pd.DataFrame, y_ic: np.ndarray) -> list[str]:
    """Match WeightLearner walk-forward IC feature selection."""
    candidate_features = [
        c
        for c in COMPOUND_AND_PRICE_FEATURES
        if c in train_df.columns and train_df[c].std() > 1e-12
    ]

    def _select_by_ic(threshold: float) -> list[str]:
        feats: list[str] = []
        for col in candidate_features:
            x = train_df[col].values.astype(float)
            mask = np.isfinite(x) & np.isfinite(y_ic)
            if mask.sum() <= 2:
                continue
            try:
                rho, _ = spearmanr(x[mask], y_ic[mask])
                if np.isfinite(rho) and abs(float(rho)) > threshold:
                    feats.append(col)
            except Exception:
                continue
        return feats

    selected = _select_by_ic(0.02)
    if len(selected) < 5:
        relaxed = _select_by_ic(0.01)
        if len(relaxed) >= 3:
            selected = relaxed
        else:
            selected = candidate_features
    return selected


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """IC (Spearman), directional accuracy, AUC (up/down vs score)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    yt, yp = y_true[m], y_pred[m]
    ic_val, _ = spearmanr(yp, yt)
    ic_val = float(ic_val) if np.isfinite(ic_val) else float("nan")
    dir_acc = float((np.sign(yp) == np.sign(yt)).mean())
    y_bin = (yt > 0).astype(int)
    if len(np.unique(y_bin)) < 2:
        auc = float("nan")
    else:
        try:
            auc = float(roc_auc_score(y_bin, yp))
        except ValueError:
            auc = float("nan")
    return ic_val, dir_acc, auc


def _best_name(ridge_ic: float, gbr_ic: float, xgb_ic: float, ens_ic: float) -> str:
    d = {
        "Ridge": ridge_ic,
        "GBR": gbr_ic,
        "XGB": xgb_ic,
        "Ensemble": ens_ic,
    }
    best = max(d, key=lambda k: d[k] if np.isfinite(d[k]) else -np.inf)
    return best


def main() -> None:
    tickers, start_date, end_date = _load_tickers_and_dates()
    holding_period = 5
    n_splits = 5
    EMBARGO_DAYS = 5

    print("Loading feature matrix (same as weight learning)…")
    df = build_feature_matrix(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        holding_period=holding_period,
    )
    if df.empty or TARGET not in df.columns:
        raise SystemExit("Empty feature matrix or missing target.")

    dates = np.sort(df["date"].unique())
    split_size = len(dates) // n_splits

    rows: list[dict] = []
    blend_history: list[dict[str, float]] = []

    for k in range(1, n_splits):
        cutoff_idx = k * split_size
        test_end_idx = min((k + 1) * split_size, len(dates) - 1)
        test_start = dates[cutoff_idx]
        test_end = dates[test_end_idx]
        train_end_idx = max(0, cutoff_idx - EMBARGO_DAYS)
        train_end = dates[train_end_idx]
        train_df = df[df["date"] < train_end].copy()
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

        if len(train_df) < 100 or len(test_df) < 20:
            print(f"  [Skip fold {k}: insufficient rows]")
            continue

        y_train_ic = train_df["forward_return"].astype(float).values
        selected = _select_features_ic(train_df, y_train_ic)

        sub_train = train_df.dropna(subset=selected + [TARGET, "date"])
        sub_test = test_df.dropna(subset=selected + [TARGET])

        if len(sub_train) < 50 or len(sub_test) < 10:
            print(f"  [Skip fold {k}: NaN drop too aggressive]")
            continue

        X_tr = sub_train[selected].values.astype(np.float64)
        y_tr = sub_train[TARGET].values.astype(np.float64)
        X_te = sub_test[selected].values.astype(np.float64)
        y_te = sub_test[TARGET].values.astype(np.float64)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # --- Single models (full train fold) ---
        ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0])
        ridge.fit(X_tr_s, y_tr)
        pred_ridge = ridge.predict(X_te_s)

        gbr = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=50,
            random_state=42,
        )
        gbr.fit(X_tr_s, y_tr)
        pred_gbr = gbr.predict(X_te_s)

        xgb = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            min_child_weight=50,
            objective="reg:squarederror",
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
            eval_metric="rmse",
        )
        xgb.fit(X_tr_s, y_tr)
        pred_xgb = xgb.predict(X_te_s)

        # --- Ensemble: inner time-ordered 80/20 on train for blend; refit on full train ---
        order = np.argsort(pd.to_datetime(sub_train["date"]).values)
        n_in = len(order)
        split_i = int(0.8 * n_in)
        if n_in - split_i < 10:
            split_i = max(1, n_in - 10)
        tr_idx = order[:split_i]
        va_idx = order[split_i:]
        stack = StackedEnsemble()
        with contextlib.redirect_stdout(io.StringIO()):
            stack.fit(X_tr_s[tr_idx], y_tr[tr_idx], X_tr_s[va_idx], y_tr[va_idx])
        pred_ens = stack.predict(X_te_s)
        blend = dict(stack.blend_weights)
        blend_history.append(blend)

        ic_r, dir_r, auc_r = _metrics(y_te, pred_ridge)
        ic_g, dir_g, auc_g = _metrics(y_te, pred_gbr)
        ic_x, dir_x, auc_x = _metrics(y_te, pred_xgb)
        ic_e, dir_e, auc_e = _metrics(y_te, pred_ens)

        best = _best_name(ic_r, ic_g, ic_x, ic_e)
        rows.append(
            {
                "fold": k,
                "ridge_ic": ic_r,
                "gbr_ic": ic_g,
                "xgb_ic": ic_x,
                "ens_ic": ic_e,
                "best": best,
                "ridge_dir": dir_r,
                "gbr_dir": dir_g,
                "xgb_dir": dir_x,
                "ens_dir": dir_e,
                "ridge_auc": auc_r,
                "gbr_auc": auc_g,
                "xgb_auc": auc_x,
                "ens_auc": auc_e,
                "blend": blend,
            }
        )

        print(
            f"  Fold {k}: Ridge IC={ic_r:.4f}  GBR={ic_g:.4f}  XGB={ic_x:.4f}  "
            f"Ensemble={ic_e:.4f}  best={best}  blend={blend}"
        )

    if not rows:
        raise SystemExit("No valid walk-forward folds.")

    # --- Tables: IC ---
    print("\n" + "=" * 72)
    print("OUT-OF-SAMPLE IC (Spearman vs forward_return on test fold)")
    print("=" * 72)
    hdr = f"{'Fold':^6} | {'Ridge IC':^10} | {'GBR IC':^10} | {'XGB IC':^10} | {'Ensemble IC':^12} | {'Best':^10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['fold']:^6} | {r['ridge_ic']:10.4f} | {r['gbr_ic']:10.4f} | "
            f"{r['xgb_ic']:10.4f} | {r['ens_ic']:12.4f} | {r['best']:^10}"
        )
    avg_r = np.nanmean([r["ridge_ic"] for r in rows])
    avg_g = np.nanmean([r["gbr_ic"] for r in rows])
    avg_x = np.nanmean([r["xgb_ic"] for r in rows])
    avg_e = np.nanmean([r["ens_ic"] for r in rows])
    print(
        f"{'Avg':^6} | {avg_r:10.4f} | {avg_g:10.4f} | {avg_x:10.4f} | {avg_e:12.4f} | {'':^10}"
    )

    # Directional accuracy & AUC tables (compact)
    print("\n" + "=" * 72)
    print("Directional accuracy (test fold)")
    print("=" * 72)
    for r in rows:
        print(
            f"  Fold {r['fold']}: Ridge={r['ridge_dir']:.4f}  GBR={r['gbr_dir']:.4f}  "
            f"XGB={r['xgb_dir']:.4f}  Ensemble={r['ens_dir']:.4f}"
        )
    print(
        f"  Avg        : Ridge={np.nanmean([r['ridge_dir'] for r in rows]):.4f}  "
        f"GBR={np.nanmean([r['gbr_dir'] for r in rows]):.4f}  "
        f"XGB={np.nanmean([r['xgb_dir'] for r in rows]):.4f}  "
        f"Ensemble={np.nanmean([r['ens_dir'] for r in rows]):.4f}"
    )

    print("\n" + "=" * 72)
    print("AUC (up/down vs predicted score, test fold)")
    print("=" * 72)
    for r in rows:
        print(
            f"  Fold {r['fold']}: Ridge={r['ridge_auc']:.4f}  GBR={r['gbr_auc']:.4f}  "
            f"XGB={r['xgb_auc']:.4f}  Ensemble={r['ens_auc']:.4f}"
        )

    # Blend stability
    print("\n" + "=" * 72)
    print("Blend weights per fold (StackedEnsemble optimizer)")
    print("=" * 72)
    for r in rows:
        b = r["blend"]
        print(f"  Fold {r['fold']}: {b}")
    if blend_history:
        w_ridge = np.array([b.get("ridge", np.nan) for b in blend_history])
        w_gbr = np.array([b.get("gbr", np.nan) for b in blend_history])
        w_xgb = np.array([b.get("xgb", np.nan) for b in blend_history])
        print("\n  Std dev across folds:  "
              f"ridge={np.nanstd(w_ridge):.4f}  gbr={np.nanstd(w_gbr):.4f}  xgb={np.nanstd(w_xgb):.4f}")
        print("  (Low variance ⇒ stable blend; high variance ⇒ unstable ensemble.)")

    # --- Statistical test: ensemble vs best single model per fold ---
    diffs = []
    best_single_ics = []
    ens_ics = []
    for r in rows:
        singles = np.array([r["ridge_ic"], r["gbr_ic"], r["xgb_ic"]], dtype=float)
        best_single = float(np.nanmax(singles))
        ens_ics.append(r["ens_ic"])
        best_single_ics.append(best_single)
        diffs.append(r["ens_ic"] - best_single)

    diffs = np.array(diffs, dtype=float)
    ens_mean = float(np.mean(ens_ics))
    # Per-fold champion (max of the three singles) — conservative OOS benchmark.
    mean_max_single_ic = float(np.mean(best_single_ics))
    # Best “headline” single model by average IC across folds (pick-one-model).
    best_column_avg_ic = max(avg_r, avg_g, avg_x)

    # Paired one-sided t-test: mean(ensemble − per-fold max single) > 0
    try:
        t_stat, p_two = ttest_1samp(diffs, popmean=0.0, alternative="greater")
    except TypeError:
        # older scipy
        t_stat, p_two = ttest_1samp(diffs, popmean=0.0)
        if t_stat < 0:
            p_one = 1.0 - p_two / 2.0
        else:
            p_one = p_two / 2.0
        p_val = p_one
    else:
        p_val = float(p_two)

    print("\n" + "=" * 72)
    print("Statistical test (paired across folds)")
    print("=" * 72)
    print("  Difference per fold: Ensemble IC − max(Ridge, GBR, XGB) IC on that fold")
    print(f"  Mean ensemble IC:              {ens_mean:.4f}")
    print(f"  Mean per-fold max(singles) IC: {mean_max_single_ic:.4f}")
    print(f"  Best single by avg column IC:  {best_column_avg_ic:.4f}  (max of Ridge/GBR/XGB Avg row)")
    print(f"  Mean paired difference:        {float(np.mean(diffs)):.4f}")
    print(f"  t-stat (H0: mean diff ≤ 0):    {float(t_stat):.4f}")
    print(f"  p-value (one-sided, greater):  {p_val:.4f}")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    # User rule: ensemble_mean_ic > best_single_ic (we use best column-average IC) AND p<0.10
    if ens_mean > best_column_avg_ic and p_val < 0.10:
        print("ENSEMBLE VALIDATED — use as production model")
    elif ens_mean > best_column_avg_ic:
        print("ENSEMBLE MARGINALLY BETTER — use with caution")
    else:
        print("ENSEMBLE NOT BETTER — keep Ridge as production")

    print("\nRun backtest to confirm no regression:")
    print('  python run_backtest.py 2>&1 | grep "Net Sharpe"')


if __name__ == "__main__":
    main()
