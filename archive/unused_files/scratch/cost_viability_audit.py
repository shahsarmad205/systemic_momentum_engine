import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

def run_cost_viability_audit():
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
    
    # We need target_expected_cost and target_return
    # Note: target_expected_cost is in total decimal return (e.g. 0.0005 for 5 bps)
    eval_df["target_return"] = pd.to_numeric(eval_df["target_return"], errors="coerce").fillna(0.0)
    eval_df["target_expected_cost"] = pd.to_numeric(eval_df["target_expected_cost"], errors="coerce").fillna(0.0)
    
    dates = sorted(eval_df["date"].unique())
    daily_stats = []

    print("Analyzing cost impact on IC per date...")
    for dt in dates:
        day = eval_df[eval_df["date"] == dt].copy()
        if len(day) < 50: continue
        
        # Center scores for sign check
        s = day["score"].values
        s_centered = s - np.mean(s)
        
        r_raw = day["target_return"].values
        # Cost adjustment: if we go long (s_centered > 0), return is reduced by cost.
        # If we go short (s_centered < 0), return (as a long) is r_raw, 
        # but we care about the IC of the score. 
        # CAIC should be Corr(s, r - sign(s)*cost)
        cost = day["target_expected_cost"].values
        r_adj = r_raw - np.sign(s_centered) * cost
        
        raw_ic = np.corrcoef(s, r_raw)[0, 1]
        ca_ic = np.corrcoef(s, r_adj)[0, 1]
        
        daily_stats.append({
            "date": dt,
            "raw_ic": raw_ic,
            "ca_ic": ca_ic
        })

    stats_df = pd.DataFrame(daily_stats).dropna()
    
    mean_raw_ic = stats_df["raw_ic"].mean()
    mean_ca_ic = stats_df["ca_ic"].mean()
    
    # Use magnitudes for ratio to avoid sign confusion if both are negative
    ratio = abs(mean_ca_ic) / abs(mean_raw_ic) if abs(mean_raw_ic) > 1e-9 else 0.0
    decay = 1.0 - ratio

    print("\n" + "="*60)
    print("      COST VIABILITY & ALPHA SURVIVAL AUDIT")
    print("="*60)
    print(f"Mean Raw IC              : {mean_raw_ic:.4f}")
    print(f"Mean Cost-Adjusted IC    : {mean_ca_ic:.4f}")
    print(f"CAIC / Raw IC Ratio      : {ratio:.2%}")
    print(f"Alpha Decay after Costs  : {decay:.2%}")
    print("-" * 60)
    
    print("Diagnostic Conclusion:")
    if ratio < 0.5:
        print("  COST DOMINATED: Trading costs eat more than 50% of the predictive signal.")
        print("  The strategy is likely unviable in production.")
    elif ratio < 0.8:
        print("  MARGINAL: Costs significantly degrade alpha. Requires tight execution or lower turnover.")
    else:
        print("  ROBUST: Alpha strongly survives expected execution costs.")

    print("="*60)

if __name__ == "__main__":
    run_cost_viability_audit()
