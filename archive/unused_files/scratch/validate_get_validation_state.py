import pandas as pd
import numpy as np
from model_selection.preparation import PreparedPanelCache
from model_selection.training import TargetConfig, retarget_panel_for_horizon
from model_selection.validation import ExecutionCostConfig, EvaluationConfig, ValidationStateCache

def old_get_prepared_fold_logic(df, train_start, train_end, eval_start, eval_end, horizon_days, target_cfg, costs, max_name_weight):
    # This mimics the logic BEFORE the refactor
    mask = (df['date'] >= train_start) & (df['date'] < eval_end)
    df_win = df[mask].copy()
    df_prepared = retarget_panel_for_horizon(
        df_win, 
        horizon_days=horizon_days, 
        target_cfg=target_cfg, 
        costs=costs, 
        max_name_weight=max_name_weight
    )
    
    # Eval slice
    eval_df = df_prepared[df_prepared['date'] >= eval_start]
    return eval_df

def validate_reconstruction():
    print("=== Validation: Reconstruction of Old Pipeline Outputs ===")
    
    # 1. Setup mock data
    n_days = 200
    dates = pd.date_range("2020-01-01", periods=n_days)
    tickers = ["A", "B", "C"]
    data = []
    for d in dates:
        for t in tickers:
            data.append({
                "date": d, "ticker": t, "daily_return": np.random.normal(0, 0.01),
                "adv_dollar_20": 1e7, "realised_vol_20d": 0.02, "capm_beta": 1.0, "sector": "Tech",
                "f1": np.random.normal(), "f2": np.random.normal()
            })
    df = pd.DataFrame(data)
    
    target_cfg = TargetConfig(horizon_days=5)
    costs = ExecutionCostConfig()
    
    # 2. Instantiate new cache
    cache = PreparedPanelCache(
        df,
        target_cfg=target_cfg,
        costs=costs,
        max_name_weight=0.1,
        winsor_q=0.01
    )
    
    # 3. Test Fold
    tr_s, tr_e = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-01")
    ev_s, ev_e = pd.Timestamp("2020-03-01"), pd.Timestamp("2020-04-01")
    horizon = 5
    feats = ["f1", "f2"]
    
    print(f"Comparing fold: {tr_s.date()} -> {ev_e.date()}")
    
    # New Pipeline Output
    prepared = cache.get_prepared_fold(
        train_start=tr_s, train_end=tr_e, 
        eval_start=ev_s, eval_end=ev_e, 
        horizon_days=horizon, active_features=feats
    )
    new_eval_df = prepared.eval_df
    
    # Old Pipeline Output (reconstructed)
    old_eval_df = old_get_prepared_fold_logic(
        df, tr_s, tr_e, ev_s, ev_e, horizon, target_cfg, costs, 0.1
    )
    
    # 4. Comparison
    print(f"Old Eval Shape: {old_eval_df.shape}")
    print(f"New Eval Shape: {new_eval_df.shape}")
    
    # Check targets and NaNs
    nan_old = old_eval_df['target_return'].isna().sum()
    nan_new = new_eval_df['target_return'].isna().sum()
    mu_old = old_eval_df['target_return'].mean()
    mu_new = new_eval_df['target_return'].mean()
    std_old = old_eval_df['target_return'].std()
    std_new = new_eval_df['target_return'].std()
    
    print(f"Old Target NaN: {nan_old} | Mean: {mu_old:.8f} | Std: {std_old:.8f}")
    print(f"New Target NaN: {nan_new} | Mean: {mu_new:.8f} | Std: {std_new:.8f}")
    
    # Detailed check on first 10 rows
    cols_to_check = ['date', 'ticker', 'target_return']
    diff_count = 0
    # Check all rows for exact match (using np.isclose to handle NaNs)
    for i in range(len(old_eval_df)):
        o_row = old_eval_df.iloc[i]
        n_row = new_eval_df.iloc[i]
        v_old = o_row['target_return']
        v_new = n_row['target_return']
        
        if not ((np.isnan(v_old) and np.isnan(v_new)) or np.isclose(v_old, v_new, atol=1e-10)):
            if diff_count < 10:
                print(f"ROW {i} [{o_row['date'].date()} {o_row['ticker']}] VALUE DIFF: {v_old} vs {v_new}")
            diff_count += 1
            
    if diff_count == 0:
        print("\nSUCCESS: Outputs match old pipeline.")
    else:
        print(f"\nFAILURE: Found {diff_count} mismatches.")
        # exit(1) # Don't exit yet, let's see the rest

    # 5. Check get_validation_state (Tuple Return)
    print("\nValidating get_validation_state (Tuple Return)...")
    x_tr, x_ev, y_tr, y_ev, meta = cache.get_validation_state(
        start=pd.Timestamp("2020-03-01"),
        end=pd.Timestamp("2020-04-01"),
        horizon_days=5,
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2020-03-01"),
        active_features=feats
    )
    
    assert x_tr.shape == (180, 2)
    assert x_ev.shape == (93, 2)
    assert len(y_tr) == 180
    assert len(y_ev) == 93
    print(f"SUCCESS: get_validation_state returned tuple with shapes: x_tr={x_tr.shape}, x_ev={x_ev.shape}")
    print(f"         y_tr.mean={y_tr.mean():.6f}, y_ev.mean={y_ev.mean():.6f}")
    print(f"         metadata={meta}")

    # 6. Check get_validation_state (Cache Return)
    print("\nValidating get_validation_state (Cache Return)...")
    from model_selection.validation import EvaluationConfig
    eval_cfg = EvaluationConfig(path="long_short_spread", horizon_days=5)
    state_cache = cache.get_validation_state(
        start=pd.Timestamp("2020-03-01"),
        end=pd.Timestamp("2020-04-01"),
        horizon_days=5,
        evaluation_cfg=eval_cfg
    )
    assert isinstance(state_cache, ValidationStateCache)
    print("SUCCESS: get_validation_state returned ValidationStateCache when evaluation_cfg provided")

if __name__ == "__main__":
    run_simulation_at_scale = None # Placeholder if needed
    validate_reconstruction()
