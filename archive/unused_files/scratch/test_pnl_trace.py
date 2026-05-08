import pandas as pd
import numpy as np
from pathlib import Path
from model_selection.validation import simulate_executable_portfolio, EvaluationConfig

def test_pnl_trace():
    # Path to cached feature matrix
    path = Path("output/research_state/model_selection_4d25bcce59c984c4/feature_panel_program.parquet")
    if not path.exists():
        print("Feature matrix not found.")
        return
        
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    
    # Pick a small window to trace
    dates = sorted(df["date"].unique())
    test_dates = dates[-20:] # Last 20 days
    df_test = df[df["date"].isin(test_dates)].copy()
    
    # Create a dummy "score" (e.g. forward return itself to see if it captures it)
    df_test["score"] = df_test["forward_return"]
    
    paths = ["long_short_spread", "short_side", "long_only_overlay"]
    
    from model_selection.validation import evaluate_promotion_gates, PromotionGateConfig
    gate_cfg = PromotionGateConfig()
    
    for p in paths:
        cfg = EvaluationConfig(
            path=p,
            max_positions=10,
            min_positions=2,
            horizon_days=5,
            use_optimizer=False # Rank-based for simplicity
        )
        
        print(f"\n--- Tracing PnL for {p} ---")
        returns, pnl_df = simulate_executable_portfolio(df_test, cfg)
        
        # Mock a result row for gate evaluation
        from model_selection.validation import executable_metrics
        metrics = executable_metrics(df_test, cfg)
        metrics["oos_evaluation_path"] = p
        metrics["n_windows"] = 4
        metrics["oos_sharpe_chained"] = metrics.get("exec_sharpe", 0)
        
        print(f"\n--- Evaluating Gates for {p} ---")
        evaluate_promotion_gates(metrics, gate_cfg)
    print("\nTrace complete.")

if __name__ == "__main__":
    test_pnl_trace()
