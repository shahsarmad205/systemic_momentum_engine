import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

def run_horizon_audit():
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
    
    # 1. Estimate Signal Halflife via Rank Autocorrelation
    print("Estimating signal autocorrelation decay...")
    
    # We need a pivot of scores: Date x Ticker
    score_pivot = eval_df.pivot(index="date", columns="ticker", values="score")
    rank_pivot = score_pivot.rank(axis=1, pct=True)
    
    lags = range(1, 11)
    autocorrs = []
    
    for lag in lags:
        # Cross-sectional correlation between t and t+lag
        # Corr(rank_t, rank_{t+lag})
        # We calculate this per-pair of dates and average
        corrs = []
        for i in range(len(rank_pivot) - lag):
            r1 = rank_pivot.iloc[i]
            r2 = rank_pivot.iloc[i+lag]
            valid = r1.notna() & r2.notna()
            if valid.sum() > 50:
                corrs.append(np.corrcoef(r1[valid], r2[valid])[0, 1])
        
        autocorrs.append(np.mean(corrs) if corrs else np.nan)

    # Estimate halflife H: AC(k) = rho^k -> H = ln(0.5)/ln(rho)
    # Fit rho using first few lags
    valid_acs = np.array(autocorrs[:5])
    rho = np.mean(valid_acs[1:] / valid_acs[:-1]) if len(valid_acs) > 1 else valid_acs[0]
    
    if rho > 0 and rho < 1:
        halflife = np.log(0.5) / np.log(rho)
    else:
        halflife = np.inf if rho >= 1 else 0.0

    # 2. Compare with rebalance and holding period
    rebalance_freq = 1.0 # Daily
    holding_period = 5.0 # Days (target horizon)
    exec_delay = 1.0 # Days (t+1 execution)

    # 3. Expected decay before execution
    # AC(1) is the persistence after 1 day.
    decay_before_exec = 1.0 - autocorrs[0]

    print("\n" + "="*60)
    print("      SIGNAL HORIZON CONSISTENCY AUDIT")
    print("="*60)
    print(f"Signal persistence (AC1) : {autocorrs[0]:.4f}")
    print(f"Estimated Halflife       : {halflife:.2f} days")
    print(f"Target Holding Period    : {holding_period:.2f} days")
    print(f"Execution Delay          : {exec_delay:.2f} days")
    print("-" * 60)
    print(f"Decay Before Execution   : {decay_before_exec:.1%}")
    
    survives = halflife > (exec_delay + 1.0)
    severity = holding_period / halflife if halflife > 0 else 10.0
    
    print(f"Survives Execution?      : {'YES' if survives else 'NO'}")
    print(f"Mismatch Severity Score  : {severity:.2f} (Target/Halflife)")
    
    print("\nDiagnostic Conclusion:")
    if severity > 1.5:
        print("CRITICAL MISMATCH: Signal decays significantly faster than the target holding period.")
        print("The portfolio will likely be holding 'stale' alpha for the majority of the horizon.")
    elif severity > 1.0:
        print("MODERATE MISMATCH: Signal halflife is shorter than target horizon.")
    else:
        print("CONSISTENT: Signal persistence supports the target holding period.")

    print("\nRank Autocorrelation Table:")
    for l, ac in zip(lags, autocorrs):
        print(f"  Lag {l:<2}: {ac:.4f}")
    print("="*60)

if __name__ == "__main__":
    run_horizon_audit()
