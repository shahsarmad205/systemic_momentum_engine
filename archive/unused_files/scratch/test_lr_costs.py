import pandas as pd
import numpy as np

def test_costs():
    print("Loading enriched panel...")
    df = pd.read_parquet("output/models/enriched_panel.parquet")
    
    tr_s = pd.Timestamp("2021-01-01")
    tr_e = pd.Timestamp("2022-01-01")
    ev_s = pd.Timestamp("2022-01-01")
    ev_e = pd.Timestamp("2022-06-01")
    
    from model_selection.preparation import PreparedPanelCache
    from model_selection.training import TargetConfig
    from model_selection.validation import ExecutionCostConfig, EvaluationConfig, simulate_executable_portfolio
    
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
    
    y_tr = cache.get_training_target(
        start=tr_s, end=tr_e,
        horizon_days=5, model_name="LogisticRegression",
        model_kind="classifier", use_risk_adj=False
    )
    
    from run_model_selection import _fit_candidate_model, _score_model_predictions
    from sklearn.linear_model import LogisticRegression
    
    print("Training LogisticRegression...")
    model = LogisticRegression(max_iter=1000)
    screen_model = _fit_candidate_model(
        model_template=model,
        name="LogisticRegression",
        model_kind="classifier",
        tr=prepared.train_df,
        x_tr=prepared.x_train,
        y_tr=y_tr
    )
    
    print("Scoring evaluation set...")
    score = _score_model_predictions(
        screen_model,
        prepared.x_eval,
        model_kind="classifier",
        uses_proba=True
    )
    
    eval_df = prepared.eval_df.assign(score=score)
    
    print("Running Executable Simulation...")
    eval_cfg = EvaluationConfig(path="long_short_spread", horizon_days=5)
    # We need a state_cache for simulate_executable_portfolio
    state_cache = cache.get_validation_state(
        start=ev_s, end=ev_e, horizon_days=5, evaluation_cfg=eval_cfg
    )
    
    daily_ret_s, pnl_detail = simulate_executable_portfolio(
        eval_df,
        eval_cfg,
        state_cache=state_cache
    )
    
    if pnl_detail.empty:
        print("FAIL: pnl_detail is empty.")
        return
        
    print("\n--- COST DIAGNOSTICS (Evaluation Window) ---")
    avg_turnover = pnl_detail['turnover'].mean()
    total_turnover = pnl_detail['turnover'].sum()
    
    # Cost return is negative (e.g. -0.0001)
    total_cost = pnl_detail['cost_return'].sum()
    avg_cost = pnl_detail['cost_return'].mean()
    
    # cost_per_trade? Let's say cost per unit turnover
    cost_per_unit_turnover = total_cost / total_turnover if total_turnover > 0 else 0
    
    gross_pnl = pnl_detail['gross_return'].sum()
    net_pnl = pnl_detail['daily_return'].sum()
    
    print(f"avg_turnover    : {avg_turnover:.4f}")
    print(f"total_turnover  : {total_turnover:.4f}")
    print(f"cost_per_unit_to: {cost_per_unit_turnover:.6f}")
    print(f"total_cost      : {total_cost:.6f}")
    print(f"gross_pnl       : {gross_pnl:.6f}")
    print(f"net_pnl         : {net_pnl:.6f}")
    
    print("\nCheck: Is gross_pnl positive but net_pnl negative?")
    if gross_pnl > 0 and net_pnl < 0:
        print("YES -> execution cost killing signal")
    elif gross_pnl <= 0:
        print("NO (Gross PnL is already negative or zero, signal is the primary issue)")
    else:
        print("NO (Net PnL is still positive)")

if __name__ == "__main__":
    test_costs()
