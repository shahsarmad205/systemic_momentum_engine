import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

def run_sign_audit():
    print("Loading enriched panel...")
    df = pd.read_parquet("output/models/enriched_panel.parquet")

    # Standard walk-forward setup
    tr_s = pd.Timestamp("2021-01-01")
    tr_e = pd.Timestamp("2022-01-01")
    
    from model_selection.preparation import PreparedPanelCache
    from model_selection.training import TargetConfig, make_training_target
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

    # Pick obvious features: Momentum (positive) and maybe something negative
    active_feats = ['ret_10d', 'cs_momentum_percentile', 'nearness_52w_high']
    
    print("Preparing training fold...")
    prepared = cache.get_prepared_fold(
        train_start=tr_s, train_end=tr_e,
        eval_start=tr_e, eval_end=tr_e + pd.Timedelta(days=90),
        horizon_days=5, active_features=active_feats
    )

    y_tr = make_training_target(
        prepared.train_df,
        model_name="RidgeLogistic",
        model_kind="classifier",
        use_risk_adj=False
    )

    print(f"Target 'Up' mean: {y_tr.mean():.4f}")

    model = Pipeline([
        ("scaler", RobustScaler()),
        ("model", LogisticRegression(penalty="l2", C=0.1, solver="liblinear"))
    ])

    print("Fitting RidgeLogistic...")
    model.fit(prepared.x_train, y_tr)
    
    lr = model.named_steps["model"]
    coeffs = lr.coef_[0]
    
    print("\n--- MODEL COEFFICIENTS (SIGN AUDIT) ---")
    for feat, val in zip(active_feats, coeffs):
        print(f"  {feat:<25}: {val:>10.4f}")

    print("\nInterpretation:")
    # For Momentum (ret_10d, cs_momentum_percentile), we expect positive coefficients
    # in a classifier predicting 'Up'.
    momentum_vals = [val for feat, val in zip(active_feats, coeffs) if "momentum" in feat or "ret_" in feat]
    if all(v < 0 for v in momentum_vals):
        print("  SIGN BREAK DETECTED: Momentum features have NEGATIVE coefficients in a long-biased classifier.")
        print("  This proves the model is learning the inverse of the expected economic direction.")
    else:
        print("  ECONOMIC DIRECTION: Momentum features have positive coefficients as expected.")

if __name__ == "__main__":
    run_sign_audit()
