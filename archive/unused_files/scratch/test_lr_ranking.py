import pandas as pd
import numpy as np

def test_lr():
    print("Loading enriched panel...")
    df = pd.read_parquet("output/models/enriched_panel.parquet")
    print(f"Panel loaded: {df.shape}")
    
    tr_s = pd.Timestamp("2021-01-01")
    tr_e = pd.Timestamp("2022-01-01")
    ev_s = pd.Timestamp("2022-01-01")
    ev_e = pd.Timestamp("2022-04-01")
    
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
    
    # Just take top 5 features
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
    
    eval_dates = sorted(eval_df['date'].unique())
    eval_date = eval_dates[len(eval_dates)//2]
    print(f"\n--- EVALUATION DATE: {pd.Timestamp(eval_date).date()} ---")
    
    day_df = eval_df[eval_df['date'] == eval_date].copy()
    day_df = day_df.sort_values("score", ascending=False)
    
    top_10 = day_df.head(10)
    bot_10 = day_df.tail(10)
    
    print("\nTOP 10 (Highest Score):")
    for _, row in top_10.iterrows():
        print(f"{row['ticker']:<6} | Score: {row['score']:.6f} | Realized Ret: {row.get('target_return', np.nan):.6f}")
        
    print("\nBOTTOM 10 (Lowest Score):")
    for _, row in bot_10.iterrows():
        print(f"{row['ticker']:<6} | Score: {row['score']:.6f} | Realized Ret: {row.get('target_return', np.nan):.6f}")
        
    mean_top = top_10['target_return'].mean()
    mean_bot = bot_10['target_return'].mean()
    
    print(f"\nmean_return_top: {mean_top:.6f}")
    print(f"mean_return_bottom: {mean_bot:.6f}")
    
    if mean_top > mean_bot:
        print("Conclusion: mean_return_top > mean_return_bottom (Ranking works as expected)")
    else:
        print("Conclusion: mean_return_top <= mean_return_bottom -> ranking is inverted or broken")

if __name__ == "__main__":
    test_lr()
