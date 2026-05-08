import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

def run_asymmetry_audit():
    print("Loading enriched panel...")
    df = pd.read_parquet("output/models/enriched_panel.parquet")

    # Standard walk-forward setup
    tr_s = pd.Timestamp("2021-01-01")
    tr_e = pd.Timestamp("2022-01-01")
    ev_s = pd.Timestamp("2022-01-01")
    ev_e = pd.Timestamp("2022-06-01")

    from model_selection.preparation import PreparedPanelCache
    from model_selection.training import TargetConfig
    from model_selection.validation import ExecutionCostConfig

    target_cfg = TargetConfig(horizon_days=5)
    costs = ExecutionCostConfig()

    cache = PreparedPanelCache(
        df,
        target_cfg=target_cfg,
        costs=costs,
        max_name_weight=0.1,
        winsor_q=0.01
    )

    active_feats = ['ret_10d', 'days_to_cover', 'momentum_acceleration', 'cs_momentum_percentile', 'nearness_52w_high']

    print("Preparing fold...")
    prepared = cache.get_prepared_fold(
        train_start=tr_s, train_end=tr_e,
        eval_start=ev_s, eval_end=ev_e,
        horizon_days=5, active_features=active_feats
    )

    from model_selection.training import make_training_target
    y_tr = make_training_target(
        prepared.train_df,
        model_name="RidgeLogistic",
        model_kind="classifier",
        use_risk_adj=False
    )

    from run_model_selection import _fit_candidate_model, _score_model_predictions
    model = Pipeline([
        ("scaler", RobustScaler()),
        ("model", LogisticRegression(penalty="l2", C=0.01, max_iter=1000))
    ])

    print("Training model...")
    fit_model = _fit_candidate_model(
        model_template=model,
        name="RidgeLogistic",
        model_kind="classifier",
        tr=prepared.train_df,
        x_tr=prepared.x_train,
        y_tr=y_tr
    )

    print("Scoring evaluation set...")
    scores = _score_model_predictions(
        fit_model,
        prepared.x_eval,
        model_kind="classifier",
        uses_proba=True
    )

    eval_df = prepared.eval_df.copy()
    eval_df["score"] = scores
    
    dates = sorted(eval_df["date"].unique())
    daily_results = []

    print("Analyzing leg performance per date...")
    for dt in dates:
        day = eval_df[eval_df["date"] == dt].copy()
        if len(day) < 100: continue
        
        # Calculate deciles
        day["rank"] = day["score"].rank(pct=True)
        long_leg = day[day["rank"] > 0.9]
        short_leg = day[day["rank"] <= 0.1]
        
        # Realized returns
        # Note: target_return is the forward return
        y_long = long_leg["target_return"].values
        y_short = short_leg["target_return"].values
        
        # Daily Stats
        row = {
            "date": dt,
            "long_ret": np.mean(y_long),
            "short_ret": np.mean(y_short),
            "long_ic": np.corrcoef(long_leg["score"], long_leg["target_return"])[0, 1],
            "short_ic": np.corrcoef(short_leg["score"], short_leg["target_return"])[0, 1],
            "long_hit": np.mean(y_long > 0),
            "short_hit": np.mean(y_short < 0) # Short hit is stock going DOWN
        }
        daily_results.append(row)

    results_df = pd.DataFrame(daily_results).dropna()
    
    print("\n" + "="*60)
    print("      LONG-SHORT ASYMMETRY DIAGNOSIS")
    print("="*60)
    
    def _report_leg(name, rets, ics, hits):
        mu = rets.mean()
        std = rets.std()
        sharpe = (mu / std * np.sqrt(252)) if std > 0 else 0.0
        
        print(f"[{name.upper()} LEG]")
        print(f"  Mean Return  : {mu:>10.6f}")
        print(f"  Mean IC      : {ics.mean():>10.4f}")
        print(f"  Sharpe Ratio : {sharpe:>10.2f}")
        print(f"  Hit Rate     : {hits.mean():>10.1%}")
        return sharpe

    # Long leg: raw return
    l_sharpe = _report_leg("Long", results_df["long_ret"], results_df["long_ic"], results_df["long_hit"])
    print("-" * 30)
    # Short leg: we want return to be negative. Sharpe of -ret.
    s_sharpe = _report_leg("Short", -results_df["short_ret"], results_df["short_ic"], results_df["short_hit"])
    
    print("\nStatistical Validity of Short Side:")
    if s_sharpe > 0.5:
        print("  VALID: Short side provides significant independent alpha.")
        rec = "a) long-short"
    elif s_sharpe > 0:
        print("  WEAK: Short side is marginally profitable but noisy.")
        rec = "a) long-short (with caution)"
    elif s_sharpe > -0.5:
        print("  NOISE: Short side has no predictive power; it is essentially random.")
        rec = "b) long-only"
    else:
        print("  INVERTED: Short candidates consistently outperform. Shorting them loses money.")
        rec = "c) inverted short"

    print(f"\nFinal Recommendation: {rec}")
    print("="*60)

if __name__ == "__main__":
    run_asymmetry_audit()
