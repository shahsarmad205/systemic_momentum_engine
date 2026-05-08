import pandas as pd
import numpy as np

def test_neutrality():
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
        
    print("\n--- NEUTRALITY AUDIT ---")
    mean_beta = pnl_detail['beta_exposure'].mean()
    max_beta = pnl_detail['beta_exposure'].abs().max()
    
    mean_sector = pnl_detail['max_sector_exposure'].mean()
    max_sector = pnl_detail['max_sector_exposure'].max()
    
    mean_net_exp = pnl_detail['net_exposure'].mean()
    max_net_exp = pnl_detail['net_exposure'].abs().max()
    
    print(f"mean_beta       : {mean_beta:.6f}")
    print(f"max_beta        : {max_beta:.6f}")
    print(f"mean_sector_exp : {mean_sector:.6f}")
    print(f"max_sector_exp  : {max_sector:.6f}")
    print(f"mean_net_exp    : {mean_net_exp:.6f}")
    print(f"max_net_exp     : {max_net_exp:.6f}")
    
    print("\nConclusion:")
    if abs(mean_beta) > 0.1:
        print(f"Beta (|{mean_beta:.3f}|) is NOT ~0 -> portfolio not neutral. Signal may be drowned by market.")
    else:
        print(f"Beta (|{mean_beta:.3f}|) is close to 0 -> portfolio is market neutral.")

if __name__ == "__main__":
    test_neutrality()
