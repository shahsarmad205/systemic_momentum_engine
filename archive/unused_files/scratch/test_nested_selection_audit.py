import pandas as pd
import numpy as np
import time

def test_nested_audit():
    print("Loading enriched panel...")
    df = pd.read_parquet("output/models/enriched_panel.parquet")
    
    # 1. Setup Fold
    # Training: 2021-01-01 -> 2022-01-01
    # Evaluation (OOS): 2022-01-01 -> 2022-04-01
    tr_s = pd.Timestamp("2021-01-01")
    tr_e = pd.Timestamp("2022-01-01")
    ev_s = pd.Timestamp("2022-01-01")
    ev_e = pd.Timestamp("2022-04-01")
    
    from model_selection.preparation import PreparedPanelCache
    from model_selection.training import TargetConfig
    from model_selection.validation import ExecutionCostConfig, EvaluationConfig, simulate_executable_portfolio
    from run_model_selection import _fit_candidate_model, _score_model_predictions, _build_models
    
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
    
    # 2. Define Candidate Models (varying turnover_penalty)
    # We'll use the DateGroupedEconomicModel from model_registry
    from model_selection.model_registry import DateGroupedEconomicModel
    
    candidates = []
    penalties = [0.0, 0.1, 0.5, 1.0]
    for p in penalties:
        model = DateGroupedEconomicModel(objective="long_short_spread", turnover_penalty=p, cost_penalty=0.1)
        candidates.append({
            "name": f"EconAlpha_p{p}",
            "model": model,
            "penalty": p
        })
    
    # 3. Step 1: Nested Validation (In-fold of the training set)
    # To mimic nested validation, we'll split the training set (2021-2022) into its own walk-forward
    # Let's just take the last 3 months of the training set as the "nested validation" window.
    nes_s = pd.Timestamp("2021-10-01")
    nes_e = pd.Timestamp("2022-01-01")
    nes_tr_e = pd.Timestamp("2021-10-01") # Nested training end
    
    print("Running Nested Validation (last 3m of training)...")
    nested_results = []
    for cand in candidates:
        # Fit on nested training
        # We need targets for the nested window
        y_nes_tr = cache.get_training_target(
            start=tr_s, end=nes_tr_e,
            horizon_days=5, model_name=cand["name"],
            model_kind="long_alpha", use_risk_adj=False
        )
        prep_nes = cache.get_prepared_fold(
            train_start=tr_s, train_end=nes_tr_e,
            eval_start=nes_s, eval_end=nes_e,
            horizon_days=5, active_features=active_feats
        )
        
        m_nes = _fit_candidate_model(
            model_template=cand["model"],
            name=cand["name"],
            model_kind="long_alpha",
            tr=prep_nes.train_df,
            x_tr=prep_nes.x_train,
            y_tr=y_nes_tr
        )
        
        # Score nested validation
        s_nes = _score_model_predictions(m_nes, prep_nes.x_eval, model_kind="long_alpha", uses_proba=False)
        df_nes = prep_nes.eval_df.assign(score=s_nes)
        
        # Simulate nested
        cfg_nes = EvaluationConfig(path="long_short_spread", horizon_days=5)
        st_nes = cache.get_validation_state(start=nes_s, end=nes_e, horizon_days=5, evaluation_cfg=cfg_nes)
        ret_nes, pnl_nes = simulate_executable_portfolio(df_nes, cfg_nes, state_cache=st_nes)
        
        # Compute Sharpe (Proxy for Selection Score)
        mu = pnl_nes['daily_return'].mean()
        std = pnl_nes['daily_return'].std()
        sharpe = (mu / std * np.sqrt(252)) if std > 0 else -1.0
        
        cand["nested_sharpe"] = sharpe
        print(f"  {cand['name']}: Nested Sharpe = {sharpe:.4f}")
        
    # 4. Step 2: OOS Evaluation (The actual evaluation window)
    print("\nRunning OOS Evaluation (actual eval window)...")
    for cand in candidates:
        # Fit on full training
        y_tr = cache.get_training_target(
            start=tr_s, end=tr_e,
            horizon_days=5, model_name=cand["name"],
            model_kind="long_alpha", use_risk_adj=False
        )
        prep_oos = cache.get_prepared_fold(
            train_start=tr_s, train_end=tr_e,
            eval_start=ev_s, eval_end=ev_e,
            horizon_days=5, active_features=active_feats
        )
        
        m_oos = _fit_candidate_model(
            model_template=cand["model"],
            name=cand["name"],
            model_kind="long_alpha",
            tr=prep_oos.train_df,
            x_tr=prep_oos.x_train,
            y_tr=y_tr
        )
        
        # Score OOS
        s_oos = _score_model_predictions(m_oos, prep_oos.x_eval, model_kind="long_alpha", uses_proba=False)
        df_oos = prep_oos.eval_df.assign(score=s_oos)
        
        # Simulate OOS
        cfg_oos = EvaluationConfig(path="long_short_spread", horizon_days=5)
        st_oos = cache.get_validation_state(start=ev_s, end=ev_e, horizon_days=5, evaluation_cfg=cfg_oos)
        ret_oos, pnl_oos = simulate_executable_portfolio(df_oos, cfg_oos, state_cache=st_oos)
        
        mu = pnl_oos['daily_return'].mean()
        std = pnl_oos['daily_return'].std()
        sharpe = (mu / std * np.sqrt(252)) if std > 0 else -1.0
        
        cand["oos_sharpe"] = sharpe
        print(f"  {cand['name']}: OOS Sharpe = {sharpe:.4f}")
        
    # 5. Comparison Table
    print("\n--- NESTED VS OOS PERFORMANCE ---")
    results_df = pd.DataFrame(candidates)
    results_df["diff"] = results_df["oos_sharpe"] - results_df["nested_sharpe"]
    
    # Selection Score (Nested)
    results_df["selection_score"] = results_df["nested_sharpe"]
    results_df["realized_score"] = results_df["oos_sharpe"]
    
    print(results_df[["name", "selection_score", "realized_score", "diff"]])
    
    # Check Ranking
    results_df["nested_rank"] = results_df["selection_score"].rank(ascending=False)
    results_df["oos_rank"] = results_df["realized_score"].rank(ascending=False)
    
    print("\nCheck: Is ranking different OOS vs nested?")
    if (results_df["nested_rank"] != results_df["oos_rank"]).any():
        print("YES -> Ranking shifted OOS. Potential overfitting in selection layer or regime shift.")
        # Identify the winners
        best_nes = results_df.loc[results_df["nested_rank"] == 1, "name"].values[0]
        best_oos = results_df.loc[results_df["oos_rank"] == 1, "name"].values[0]
        print(f"  Nested Winner : {best_nes}")
        print(f"  OOS Winner    : {best_oos}")
    else:
        print("NO -> Ranking is identical. Selection layer is robust.")

if __name__ == "__main__":
    test_nested_audit()
