import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

def run_stability_audit():
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
    daily_stats = []

    print("Measuring cross-sectional stability...")
    for dt in dates:
        day = eval_df[eval_df["date"] == dt].copy()
        if len(day) < 10: continue
        
        y_score = day["score"].values
        y_ret = day["target_return"].values
        
        var_val = np.var(y_score)
        n_unique = len(np.unique(y_score))
        dispersion = n_unique / len(day)
        
        # Performance
        if var_val > 1e-12:
            ic = np.corrcoef(y_score, y_ret)[0, 1]
            pnl = np.mean((y_score - np.mean(y_score)) * y_ret)
        else:
            ic = 0.0
            pnl = 0.0
            
        daily_stats.append({
            "date": dt,
            "variance": var_val,
            "unique_ratio": dispersion,
            "ic": ic,
            "pnl": pnl
        })

    stats_df = pd.DataFrame(daily_stats).dropna()
    
    # Analyze correlations between stability and performance
    # IC Drop correlation
    # We look at correlation between variance and absolute IC (or just IC)
    corr_var_ic = stats_df["variance"].corr(stats_df["ic"].abs())
    corr_disp_ic = stats_df["unique_ratio"].corr(stats_df["ic"].abs())
    
    print("\n" + "="*60)
    print("      RANKING STABILITY & COLLAPSE AUDIT")
    print("="*60)
    print(f"Mean Score Variance      : {stats_df['variance'].mean():.8f}")
    print(f"Mean Rank Dispersion     : {stats_df['unique_ratio'].mean():.2%}")
    print("-" * 60)
    print(f"Corr(Variance, Abs IC)   : {corr_var_ic:.4f}")
    print(f"Corr(Dispersion, Abs IC) : {corr_disp_ic:.4f}")
    
    collapse_days = (stats_df["variance"] < 1e-10).sum()
    print(f"Collapsed Days (Var ~ 0) : {collapse_days} / {len(stats_df)}")
    
    print("\nRoot Cause Diagnosis:")
    if collapse_days > 0:
        print("  ISSUE DETECTED: Model collapse occurred on certain dates.")
        
        # Check features on those dates
        bad_dates = stats_df[stats_df["variance"] < 1e-10]["date"].head(1).values
        if len(bad_dates) > 0:
            bad_day = eval_df[eval_df["date"] == bad_dates[0]]
            # Count NaNs in features
            feat_cols = [c for c in bad_day.columns if c in active_feats]
            nan_pct = bad_day[feat_cols].isna().mean().mean()
            if nan_pct > 0.5:
                print(f"  Diagnosis: c) DATA SPARSITY ({nan_pct:.1%} NaNs in features on collapsed dates)")
            else:
                # Check feature variance
                feat_vars = bad_day[feat_cols].var().mean()
                if feat_vars < 1e-10:
                    print(f"  Diagnosis: b) FEATURE DEGENERACY (No variance in input features)")
                else:
                    print(f"  Diagnosis: a) MODEL COLLAPSE (Weights saturated or model predicts constant intercept)")
    else:
        print("  STABLE: Cross-sectional ranking remains diverse throughout the window.")

    print("="*60)

if __name__ == "__main__":
    run_stability_audit()
