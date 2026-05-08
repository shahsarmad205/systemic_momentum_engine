import pandas as pd
import numpy as np

def test_pnl():
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
    
    # Calculate implied weights
    from model_selection.validation import _get_implied_weights
    weighted_df = _get_implied_weights(day_df, primary_path="long_short_spread")
    
    # Merge weights back with returns
    day_df = pd.merge(day_df, weighted_df[['ticker', '_w']], on='ticker', how='left')
    day_df['_w'] = day_df['_w'].fillna(0.0)
    
    # Calculate contribution
    day_df['contribution'] = day_df['_w'] * day_df['target_return']
    
    # Sort by score
    day_df = day_df.sort_values("score", ascending=False)
    
    top_10 = day_df.head(10)
    bot_10 = day_df.tail(10)
    
    print("\nTOP 10 (Highest Score):")
    for _, row in top_10.iterrows():
        print(f"{row['ticker']:<6} | Weight: {row['_w']:.6f} | Return: {row['target_return']:.6f} | Contribution: {row['contribution']:.6f}")
        
    print("\nBOTTOM 10 (Lowest Score):")
    for _, row in bot_10.iterrows():
        print(f"{row['ticker']:<6} | Weight: {row['_w']:.6f} | Return: {row['target_return']:.6f} | Contribution: {row['contribution']:.6f}")
        
    total_pnl = day_df['contribution'].sum()
    print(f"\ntotal_pnl = {total_pnl:.6f}")
    
    n_total = len(day_df)
    decile_size = max(1, n_total // 10)
    
    top_decile = day_df.head(decile_size)
    bot_decile = day_df.tail(decile_size)
    
    pnl_top_decile = top_decile['contribution'].sum()
    pnl_bottom_decile = bot_decile['contribution'].sum()
    
    print(f"pnl_top_decile = {pnl_top_decile:.6f}")
    print(f"pnl_bottom_decile = {pnl_bottom_decile:.6f}")
    
    if pnl_bottom_decile < 0:
        print("\nCheck: Is bottom decile contributing negatively? YES (negative PnL is BAD for the total portfolio PnL if it was supposed to make money, but wait! If weight is negative and return is negative, contribution is POSITIVE. If bottom decile contribution is negative, it is losing money!)")
    else:
        print("\nCheck: Is bottom decile contributing negatively? NO")

    print("\nGoal Conclusion:")
    if pnl_bottom_decile < 0:
        print("Bottom decile contributes negatively (BAD). Since they are shorts (negative weight), a negative contribution means they had POSITIVE returns. So the model predicted them to go down, but they went up.")
    else:
        print("Bottom decile contributes positively (GOOD). They are shorts, so a positive contribution means they had NEGATIVE returns. The model correctly predicted they would go down.")

if __name__ == "__main__":
    test_pnl()
