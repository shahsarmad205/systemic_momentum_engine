import pandas as pd
import numpy as np

def test_short_lr():
    print("Loading enriched panel...")
    df = pd.read_parquet("output/models/enriched_panel.parquet")
    
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
    
    active_feats = ['ret_10d', 'days_to_cover', 'momentum_acceleration', 'cs_momentum_percentile', 'nearness_52w_high']
    
    print("Preparing fold...")
    prepared = cache.get_prepared_fold(
        train_start=tr_s, train_end=tr_e,
        eval_start=ev_s, eval_end=ev_e,
        horizon_days=5, active_features=active_feats
    )
    
    y_tr = cache.get_training_target(
        start=tr_s, end=tr_e,
        horizon_days=5, model_name="ShortLogistic",
        model_kind="short_classifier", use_risk_adj=False
    )
    
    from run_model_selection import _fit_candidate_model, _score_model_predictions
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    
    print("Training ShortLogistic...")
    model = Pipeline([("scaler", RobustScaler()), ("model", LogisticRegression(C=0.01, max_iter=1000, class_weight="balanced"))])
    
    screen_model = _fit_candidate_model(
        model_template=model,
        name="ShortLogistic",
        model_kind="short_classifier",
        tr=prepared.train_df,
        x_tr=prepared.x_train,
        y_tr=y_tr
    )
    
    print("Scoring evaluation set...")
    score = _score_model_predictions(
        screen_model,
        prepared.x_eval,
        model_kind="short_classifier",
        uses_proba=True
    )
    
    eval_df = prepared.eval_df.assign(score=score)
    
    eval_dates = sorted(eval_df['date'].unique())
    eval_date = eval_dates[len(eval_dates)//2]
    print(f"\n--- EVALUATION DATE: {pd.Timestamp(eval_date).date()} ---")
    
    day_df = eval_df[eval_df['date'] == eval_date].copy()
    
    from model_selection.validation import _get_implied_weights
    weighted_df = _get_implied_weights(day_df, primary_path="short_side")
    
    day_df = pd.merge(day_df, weighted_df[['ticker', '_w']], on='ticker', how='left')
    day_df['_w'] = day_df['_w'].fillna(0.0)
    
    # Sort by score descending to see who gets shorted
    day_df = day_df.sort_values("score", ascending=False)
    
    top_10 = day_df.head(10)
    bot_10 = day_df.tail(10)
    
    print("\nTOP 10 (Highest Score -> Highest Probability of going DOWN):")
    for _, row in top_10.iterrows():
        print(f"{row['ticker']:<6} | Score (Prob): {row['score']:.6f} | Weight: {row['_w']:.6f} | Return: {row['target_return']:.6f}")
        
    print("\nBOTTOM 10 (Lowest Score -> Lowest Probability of going DOWN):")
    for _, row in bot_10.iterrows():
        print(f"{row['ticker']:<6} | Score (Prob): {row['score']:.6f} | Weight: {row['_w']:.6f} | Return: {row['target_return']:.6f}")
        
    shorted_count_top = (top_10['_w'] < -1e-6).sum()
    shorted_count_bot = (bot_10['_w'] < -1e-6).sum()
    
    print(f"\nCheck: Are you shorting highest scores? {'YES' if shorted_count_top > 0 else 'NO'}")
    print(f"Check: Are you shorting lowest scores?  {'YES' if shorted_count_bot > 0 else 'NO'}")
    
    if shorted_count_bot > 0 and shorted_count_top == 0:
        print("\nConclusion: INVERSION DETECTED. You are shorting the LOWEST scores.")
        print("For a short_classifier, the highest score = highest probability of going down.")
        print("But the portfolio construction (short_side) shorts the lowest scores. So you are shorting stocks predicted to go UP.")
    elif shorted_count_top > 0 and shorted_count_bot == 0:
        print("\nConclusion: Logic is correct. You are shorting the HIGHEST scores.")
    else:
        print("\nConclusion: Unclear or mixed.")

if __name__ == "__main__":
    test_short_lr()
