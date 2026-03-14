"""
Weight Learning Runner
========================
Trains the weight-learning model on historical data, validates via
walk-forward splits, saves weights, and optionally compares learned
vs rule-based systems in a backtest.

Usage:
    python run_weight_learning.py                     # Ridge, default tickers
    python run_weight_learning.py --model gbr         # Gradient Boosting
    python run_weight_learning.py --compare           # + backtest comparison
    python run_weight_learning.py --tickers AAPL MSFT # custom tickers
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SEP = "=" * 65
OUTPUT_DIR = "output"
WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "learned_weights.json")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Trend Signal Engine — Weight Learning",
    )
    p.add_argument("--model", choices=["ridge", "gbr", "rf", "logistic", "xgb"],
                   default="ridge", help="Model type (default: ridge)")
    p.add_argument("--alpha", type=float, default=0.01,
                   help="Ridge/Logistic regularisation (default 0.01 for non-zero weights)")
    p.add_argument("--decay", type=float, default=0.001,
                   help="Time-decay lambda (0 = uniform)")
    p.add_argument("--start", default="2013-01-01",
                   help="Training start date (default 2013-01-01 for ~10y history)")
    p.add_argument("--end", default="2024-01-01",
                   help="Training end date")
    p.add_argument("--holding-period", "--holding", type=int, default=5, dest="holding_period",
                   help="Forward return holding period in trading days (default 5)")
    p.add_argument("--target-type", choices=["regression", "classification"], default="regression",
                   dest="target_type",
                   help="Target task: regression (default, numeric) or classification (direction).")
    p.add_argument("--return-target", choices=["raw", "excess", "sharpe_scaled"], default="raw",
                   dest="return_target",
                   help="Return target variant: raw forward_return (default), "
                        "excess vs SPY, or sharpe_scaled (forward_return / 20d vol).")
    p.add_argument("--splits", type=int, default=5,
                   help="Walk-forward CV splits")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Override tickers (space-separated)")
    p.add_argument("--output", default=WEIGHTS_PATH,
                   help="Path to save learned weights JSON")
    p.add_argument("--compare", action="store_true",
                   help="Run side-by-side backtest: rule-based vs learned")
    p.add_argument("--validate", action="store_true",
                   help="Run backtest with learned weights and require total_trades > 0")
    p.add_argument("--regime", action="store_true",
                   help="Train regime-specific weights (Bull, Bear, HighVol, Normal)")
    p.add_argument("--config", default=None,
                   help="Optional YAML file with default weight-learning settings")
    p.add_argument("--tune", action="store_true",
                   help="Enable hyperparameter tuning inside each training window")
    return p.parse_args()


# ------------------------------------------------------------------
# Ticker resolution (uses central config.DEV_MODE / get_effective_tickers)
# ------------------------------------------------------------------

def _resolve_tickers(args):
    from config import get_effective_tickers
    try:
        from main import TICKERS
        fallback = list(TICKERS)
    except Exception:
        fallback = [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META", "TSLA",
            "JPM", "V", "JNJ", "WMT", "PG", "XOM", "UNH",
            "BAC", "ABBV", "PFE", "KO", "PEP", "MRK", "AVGO",
            "COST", "TMO", "CSCO", "MCD", "NKE", "ADBE",
            "CRM", "AMD", "INTC", "ORCL", "IBM", "GS", "CAT",
            "SPY", "QQQ", "IWM", "DIA", "XLK", "VTI",
        ]
    return get_effective_tickers(args.tickers or [], fallback)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()
    tickers = _resolve_tickers(args)

    holding_days = getattr(args, "holding_period", getattr(args, "holding", 20))
    target_type = getattr(args, "target_type", "classification")
    return_target = getattr(args, "return_target", "raw")

    print(f"\n{SEP}")
    print("  Trend Signal Engine — Weight Learning")
    print(SEP)
    print(f"  Model         : {args.model}")
    print(f"  Alpha         : {args.alpha}")
    print(f"  Time Decay λ  : {args.decay}")
    print(f"  Period        : {args.start}  →  {args.end}")
    print(f"  Holding       : {holding_days} trading days")
    print(f"  Target task   : {target_type}")
    print(f"  Return target : {return_target}")
    print(f"  Tickers       : {len(tickers)}")
    print(f"  Walk-Forward  : {args.splits} splits")
    print(f"  Output        : {args.output}")
    print(SEP)
    print()

    # ── Phase 1: Build feature matrix ──────────────────────────────
    print("Phase 1: Building feature matrix…\n")
    from agents.weight_learning_agent import build_feature_matrix

    features_df = build_feature_matrix(
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        holding_period=holding_days,
    )

    if features_df.empty:
        print("\n[ERROR] No features produced. Aborting.")
        sys.exit(1)

    n_tickers = features_df["ticker"].nunique()
    print(f"\n  Feature matrix: {len(features_df):,} rows × "
          f"{features_df.shape[1]} cols from {n_tickers} tickers")
    print(f"  Date range   : {features_df['date'].min()} → {features_df['date'].max()}")
    print(f"  Forward return: mean={features_df['forward_return'].mean():.4f}  "
          f"std={features_df['forward_return'].std():.4f}")
    print()

    # ── Phase 2: Train model (single or regime-specific) ─────────────
    from agents.weight_learning_agent import WeightLearner
    from agents.weight_learning_agent.weight_model import save_regime_weights, TARGET

    REGIME_ORDER = ["bull", "bear", "high_vol", "normal"]
    MIN_SAMPLES_PER_REGIME = 300

    if args.regime:
        print("Phase 2: Regime-specific weight learning…\n")
        from agents.weight_learning_agent.regime_detection import detect_regimes

        regime_series = detect_regimes(args.start, args.end)
        if regime_series.empty:
            print("  [WARN] No regime labels; falling back to single model.")
            features_df["regime"] = "normal"
        else:
            # Map date -> regime (lowercase, HighVol -> high_vol)
            def to_key(r):
                if r == "HighVol":
                    return "high_vol"
                return r.lower() if isinstance(r, str) else "normal"
            features_df["regime"] = features_df["date"].map(
                lambda d: to_key(regime_series.get(pd.Timestamp(d), "Normal"))
            )
            features_df["regime"] = features_df["regime"].fillna("normal")

        regime_models = {}
        active_features = []
        model_type = args.model
        print(f"[DEBUG] Using model type: {model_type} (regime-specific)")
        learner = WeightLearner(
            model_type=model_type,
            alpha=args.alpha,
            time_decay_lambda=args.decay,
            target_type=target_type,
            return_target_type=return_target,
        )
        learner.tune = args.tune

        for reg in REGIME_ORDER:
            subset = features_df[features_df["regime"] == reg]
            n = len(subset)
            print(f"  Regime {reg:<10} : {n:,} samples", end="")
            if n < MIN_SAMPLES_PER_REGIME:
                print(f"  (skip, need >={MIN_SAMPLES_PER_REGIME})")
                continue
            reg_learner = WeightLearner(
                model_type=args.model,
                alpha=args.alpha,
                time_decay_lambda=args.decay,
                target_type=target_type,
                return_target_type=return_target,
            )
            reg_learner.tune = args.tune
            w = reg_learner.fit(subset)
            regime_models[reg] = w
            if not active_features:
                active_features = reg_learner.active_features
            print(f"  → IC={w.ic:.4f}  Dir={w.directional_accuracy:.1%}  n_samples={w.n_samples:,}")

        if not regime_models:
            print("  [WARN] No regime had enough samples; training single global model.")
            learner = WeightLearner(
                model_type=args.model,
                alpha=args.alpha,
                time_decay_lambda=args.decay,
                target_type=target_type,
                return_target_type=return_target,
            )
            learner.tune = args.tune
            weights = learner.fit(features_df)
            regime_models = None
            active_features = None
        else:
            weights = next(iter(regime_models.values()))
            print(f"\n  Regime models trained: {list(regime_models.keys())}")
    else:
        print("Phase 2: Training model…\n")
        model_type = args.model
        print(f"[DEBUG] Using model type: {model_type}")
        learner = WeightLearner(
            model_type=model_type,
            alpha=args.alpha,
            time_decay_lambda=args.decay,
            target_type=target_type,
            return_target_type=return_target,
        )
        learner.tune = args.tune
        weights = learner.fit(features_df)
        regime_models = None
        active_features = None

    # Log weights (single or first regime)
    print(f"\n  Learned Weights (verify non-zero):")
    w_names = [
        "w_trend", "w_regional", "w_global", "w_social",
        "w_ret_5d", "w_ret_10d", "w_vol_10", "w_vol", "w_rel_vol",
        "w_vol_zscore", "w_corr_market",
    ]
    for name in w_names:
        val = getattr(weights, name, 0.0)
        print(f"    {name:<14} = {val:+.6f}")
    print(f"    {'intercept':<14} = {weights.intercept:+.8f}")
    print(f"    {'score_scale':<14} = {getattr(weights, 'score_scale', 1.0):.4f}")
    non_zero = [n for n in w_names if abs(getattr(weights, n, 0)) > 1e-8]
    print(f"  Non-zero weights: {non_zero or 'none (check alpha/features)'}")
    print()
    print(f"  Train Metrics (return_target={return_target}):")
    target_type_eff = getattr(weights, "target_type", target_type)
    if target_type_eff == "regression":
        # Regression focus: R², MAE, IC (Dir. Acc. still shown but secondary)
        print(f"    R²                    : {weights.r2:.4f}")
        print(f"    MAE                   : {weights.mae:.6f}")
        print(f"    Information Coeff.    : {weights.ic:.4f}")
        print(f"    Directional Accuracy  : {weights.directional_accuracy:.1%}")
    else:
        # Classification focus: Dir. Acc., IC, AUC
        print(f"    Directional Accuracy  : {weights.directional_accuracy:.1%}")
        print(f"    Information Coeff.    : {weights.ic:.4f}")
        print(f"    AUC (train)           : {getattr(weights, 'auc_score', 0.0):.4f}")
    ess = learner._estimate_ess(features_df[TARGET])
    print(f"    Effective sample size : {ess:,} (est.)")
    print()

    # Validate predictive power
    print("  Validation — Predictive power:")
    if target_type_eff == "regression":
        ic_ok = weights.ic > 0
        r2_ok = weights.r2 > 0
        print(f"    R² > 0                  : {weights.r2:.4f}  {'OK' if r2_ok else 'WARN (<=0)'}")
        print(f"    IC > 0                  : {weights.ic:.4f}  {'OK' if ic_ok else 'WARN (<=0)'}")
        if not (ic_ok and r2_ok):
            print("    [WARN] Model may not be better than random. Consider more features or lower alpha.")
    else:
        ic_ok = weights.ic > 0
        acc_ok = weights.directional_accuracy > 0.5
        print(f"    IC > 0                  : {weights.ic:.4f}  {'OK' if ic_ok else 'WARN (<=0)'}")
        print(f"    Directional acc. > 50%  : {weights.directional_accuracy:.1%}  {'OK' if acc_ok else 'WARN (<=50%)'}")
        if not ic_ok or not acc_ok:
            print("    [WARN] Model may not be better than random. Consider more features or lower alpha.")
    print()

    # ── Phase 3: Walk-forward validation (single model only) ─────────
    if regime_models is None:
        print(f"Phase 3: Walk-forward validation ({args.splits} splits)…\n")
        wf_results = learner.walk_forward_validate(features_df, n_splits=args.splits)
    else:
        wf_results = []
        print("Phase 3: Walk-forward skipped (regime-specific models).\n")

    if wf_results:
        def _safe_mean(key):
            vals = [r[key] for r in wf_results if r.get(key) is not None]
            return np.mean(vals) if vals else None

        avg_r2 = _safe_mean("r2")
        avg_mae = _safe_mean("mae")
        avg_dir = np.mean([r["directional_accuracy"] for r in wf_results])
        avg_ic = _safe_mean("ic")
        avg_auc = _safe_mean("auc")

        def _fmt(val, fmt_spec=".4f"):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "N/A"
            return format(val, fmt_spec)

        for r in wf_results:
            r2_s = _fmt(r["r2"], "+.4f")
            mae_s = _fmt(r["mae"], ".6f")
            ic_s = _fmt(r["ic"], "+.4f")
            auc_s = _fmt(r.get("auc"), ".4f")
            print(f"  Split {r['split']}: train→{r['train_end']}  "
                  f"test→{r['test_end']}  "
                  f"R²={r2_s}  MAE={mae_s}  "
                  f"Dir={r['directional_accuracy']:.1%}  IC={ic_s}  AUC={auc_s}  "
                  f"(n_train={r['n_train']:,}  n_test={r['n_test']:,})")

        print(f"\n  Average across folds:")
        print(f"    R²                    : {_fmt(avg_r2)}")
        print(f"    MAE                   : {_fmt(avg_mae, '.6f')}")
        print(f"    Directional Accuracy  : {avg_dir:.1%}")
        print(f"    Information Coeff.    : {_fmt(avg_ic)}")
        if avg_auc is not None:
            print(f"    AUC                   : {avg_auc:.4f}")

        # Weight stability across folds
        print(f"\n  Weight stability across folds:")
        w_trends = [r["weights"]["w_trend"] for r in wf_results]
        print(f"    w_trend: mean={np.mean(w_trends):+.4f}  "
              f"std={np.std(w_trends):.4f}  "
              f"range=[{min(w_trends):+.4f}, {max(w_trends):+.4f}]")

        # Optional: compare IC across return target variants on the same feature set.
        print("\n  IC comparison across return targets (train set):")
        comparison_rows = []
        for rt_mode in ["raw", "excess", "sharpe_scaled"]:
            comp_learner = WeightLearner(
                model_type=model_type,
                alpha=args.alpha,
                time_decay_lambda=args.decay,
                target_type=target_type,
                return_target_type=rt_mode,
            )
            comp_learner.tune = args.tune
            comp_weights = comp_learner.fit(features_df)
            comparison_rows.append(
                (rt_mode, comp_weights.ic, comp_weights.r2, comp_weights.mae)
            )

        print(f"    {'return_target':<14} {'IC':>8} {'R²':>8} {'MAE':>12}")
        print(f"    {'-'*14} {'-'*8} {'-'*8} {'-'*12}")
        for rt_mode, ic_val, r2_val, mae_val in comparison_rows:
            ic_s = "N/A" if ic_val is None else f"{ic_val:+.4f}"
            r2_s = "N/A" if r2_val is None else f"{r2_val:+.4f}"
            mae_s = "N/A" if mae_val is None else f"{mae_val:.6f}"
            print(f"    {rt_mode:<14} {ic_s:>8} {r2_s:>8} {mae_s:>12}")

        # If XGBoost is selected, compare OOS IC vs Ridge and GBR on the same feature set.
        if model_type == "xgb":
            print("\n  OOS IC comparison across models (walk-forward):")
            ic_rows = []
            for mtype in ["ridge", "gbr", "xgb"]:
                if mtype == "xgb":
                    # reuse existing wf_results
                    ics = [r.get("ic") for r in wf_results if r.get("ic") is not None]
                else:
                    tmp_learner = WeightLearner(
                        model_type=mtype,
                        alpha=args.alpha,
                        time_decay_lambda=args.decay,
                        target_type=target_type,
                        return_target_type=return_target,
                    )
                    tmp_learner.tune = args.tune
                    tmp_res = tmp_learner.walk_forward_validate(features_df, n_splits=args.splits)
                    ics = [r.get("ic") for r in tmp_res if r.get("ic") is not None]
                ics = [v for v in ics if v is not None and not (isinstance(v, float) and np.isnan(v))]
                mean_ic = float(np.mean(ics)) if ics else float("nan")
                ic_rows.append((mtype, mean_ic))

            print(f"    {'model':<8} {'mean_IC':>10}")
            print(f"    {'-'*8} {'-'*10}")
            for mtype, mic in ic_rows:
                mic_s = "N/A" if np.isnan(mic) else f"{mic:+.4f}"
                print(f"    {mtype:<8} {mic_s:>10}")
    else:
        if regime_models is None:
            print("  [WARN] Not enough data for walk-forward splits.")

    print()

    # ── Phase 4: Save weights (and scaler for single model) ──────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if regime_models is not None:
        save_regime_weights(args.output, regime_models, active_features or [], default_regime="normal")
        print(f"Phase 4: Saved regime-specific weights → {args.output}")
        print(f"         Regimes: {list(regime_models.keys())}\n")
    else:
        weights.save(args.output)
        scaler_path = args.output.replace(".json", "_scaler.json")
        learner.save_scaler(scaler_path)
        print(f"Phase 4: Saved learned weights → {args.output}")
        print(f"         Saved scaler (same scaling at inference) → {scaler_path}\n")

    # ── Phase 5 (optional): Backtest comparison ───────────────────
    if args.compare:
        _run_comparison(args, tickers)

    # ── Phase 6 (optional): Validate signals and trades ───────────
    if args.validate:
        _run_validate(args, tickers)

    # ------------------------------------------------------------------
    # Summary block for weight-learning run
    # ------------------------------------------------------------------
    train_ic = getattr(weights, "ic", np.nan)
    train_r2 = getattr(weights, "r2", np.nan)
    train_mae = getattr(weights, "mae", np.nan)
    train_dir = getattr(weights, "directional_accuracy", np.nan)

    wf_ic = wf_r2 = wf_dir = wf_auc = np.nan
    if wf_results:
        ics = [r.get("ic") for r in wf_results if r.get("ic") is not None]
        r2s = [r.get("r2") for r in wf_results if r.get("r2") is not None]
        dirs = [r.get("directional_accuracy") for r in wf_results if r.get("directional_accuracy") is not None]
        aucs = [r.get("auc") for r in wf_results if r.get("auc") is not None]
        if ics:
            wf_ic = float(np.mean(ics))
        if r2s:
            wf_r2 = float(np.mean(r2s))
        if dirs:
            wf_dir = float(np.mean(dirs))
        if aucs:
            wf_auc = float(np.mean(aucs))

    summary_block = (
        "\n" + "=" * 60 + "\n"
        "  WEIGHT LEARNING SUMMARY\n"
        + "=" * 60 + "\n"
        f"  Model              : {model_type}\n"
        f"  Target task        : {target_type_eff}\n"
        f"  Return target      : {return_target}\n"
        f"  Train IC           : {train_ic:.4f}\n"
        f"  Train R²           : {train_r2:.4f}\n"
        f"  Train MAE          : {train_mae:.6f}\n"
        f"  Train Dir. Acc.    : {train_dir:.1%}\n"
        f"  WF mean IC         : {wf_ic:.4f}\n"
        f"  WF mean R²         : {wf_r2:.4f}\n"
        f"  WF mean Dir. Acc.  : {wf_dir:.1%}\n"
        f"  WF mean AUC        : {wf_auc:.4f}\n"
        + "=" * 60 + "\n"
    )

    print(summary_block)

    out_dir = Path(OUTPUT_DIR) / "learning"
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_summary_path = out_dir / "latest_learning_summary.txt"
    with latest_summary_path.open("w") as fh:
        fh.write(summary_block)

    print(f"{SEP}")
    print("  Weight learning complete.")
    print(SEP)

    # Experiment snapshot for reproducibility: save args/config to a timestamped folder.
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(OUTPUT_DIR) / "experiments" / f"weight_learning_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save CLI arguments as YAML config snapshot
    config_snapshot_path = exp_dir / "config.yaml"
    with config_snapshot_path.open("w") as fh:
        yaml.safe_dump(
            {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            fh,
            sort_keys=True,
        )

    # If an explicit input config file was used, copy it alongside the snapshot
    if getattr(args, "config", None):
        cfg_path = Path(args.config)
        if cfg_path.is_file():
            dest_name = cfg_path.name or "input_config.yaml"
            dest_path = exp_dir / dest_name
            if dest_path.resolve() != cfg_path.resolve():
                import shutil

                shutil.copy2(str(cfg_path), str(dest_path))


# ------------------------------------------------------------------
# Backtest comparison: rule-based vs learned
# ------------------------------------------------------------------

def _run_validate(args, tickers):
    """Run backtest with learned weights and verify signals produce trades."""
    print(f"{SEP}")
    print("  Phase 6: Validate signal generation (backtest with learned weights)")
    print(SEP)
    print()

    from backtesting import Backtester, load_config

    from config import apply_dev_mode
    config_path = "backtest_config.yaml"
    base_config = load_config(config_path)
    apply_dev_mode(base_config)
    base_config.tickers = tickers
    base_config.start_date = args.start
    base_config.end_date = args.end
    base_config.signal_mode = "learned"
    base_config.learned_weights_path = args.output

    bt = Backtester(base_config)
    result = bt.run(tickers)
    total_trades = result.metrics["total_trades"]

    print(f"\n  Validation — Signal generation:")
    print(f"    Total trades : {total_trades:,}")
    if total_trades > 0:
        print(f"    OK: Signals generated and positions taken.")
    else:
        print(f"    FAIL: No trades. Check learned weights and score_scale.")
        sys.exit(1)
    print()


def _run_comparison(args, tickers):
    print(f"{SEP}")
    print("  Phase 5: Backtest Comparison — Rule-Based vs Learned")
    print(SEP)
    print()

    from backtesting import Backtester, load_config

    from config import apply_dev_mode
    config_path = "backtest_config.yaml"
    base_config = load_config(config_path)
    apply_dev_mode(base_config)
    base_config.tickers = tickers
    base_config.start_date = args.start
    base_config.end_date = args.end

    # Rule-based run
    print("─── Rule-Based (default weights) ───\n")
    rb_config = _clone_config(base_config)
    rb_config.signal_mode = "price"
    rb_bt = Backtester(rb_config)
    rb_result = rb_bt.run(tickers)

    # Learned-weight run
    print("\n─── Learned Weights ───\n")
    lw_config = _clone_config(base_config)
    lw_config.signal_mode = "learned"
    lw_config.learned_weights_path = args.output
    lw_bt = Backtester(lw_config)
    lw_result = lw_bt.run(tickers)

    # Side-by-side comparison
    print(f"\n{SEP}")
    print("  Head-to-Head Comparison")
    print(SEP)

    rb_m = rb_result.metrics
    lw_m = lw_result.metrics

    rows = [
        ("Total Trades",      f"{rb_m['total_trades']:,}",        f"{lw_m['total_trades']:,}"),
        ("Win Rate",           f"{rb_m['win_rate']:.1%}",          f"{lw_m['win_rate']:.1%}"),
        ("Average Return",     f"{rb_m['average_return']:.2%}",    f"{lw_m['average_return']:.2%}"),
        ("Profit Factor",      f"{rb_m['profit_factor']:.2f}",     f"{lw_m['profit_factor']:.2f}"),
        ("Sharpe Ratio",       f"{rb_m['sharpe_ratio']:.2f}",      f"{lw_m['sharpe_ratio']:.2f}"),
        ("Sortino Ratio",      f"{rb_m['sortino_ratio']:.2f}",     f"{lw_m['sortino_ratio']:.2f}"),
        ("Max Drawdown",       f"{rb_m['max_drawdown']:.2%}",      f"{lw_m['max_drawdown']:.2%}"),
        ("Signal Accuracy",    f"{rb_m['signal_accuracy']:.1%}",   f"{lw_m['signal_accuracy']:.1%}"),
        ("Info. Coefficient",  f"{rb_m['information_coefficient']:.4f}",
                               f"{lw_m['information_coefficient']:.4f}"),
    ]

    print(f"\n  {'Metric':<22} {'Rule-Based':>14} {'Learned':>14}")
    print(f"  {'─'*22} {'─'*14} {'─'*14}")
    for label, rb_val, lw_val in rows:
        print(f"  {label:<22} {rb_val:>14} {lw_val:>14}")
    print()


def _clone_config(cfg):
    """Shallow-copy a BacktestConfig for independent mutation."""
    from dataclasses import fields
    from backtesting.config import BacktestConfig
    kwargs = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    return BacktestConfig(**kwargs)


if __name__ == "__main__":
    main()
