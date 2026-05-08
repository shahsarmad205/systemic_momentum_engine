import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

def run_decomposition():
    print("Loading enriched panel...")
    df = pd.read_parquet("output/models/enriched_panel.parquet")
    print(f"Panel loaded: {df.shape}")

    # Standard walk-forward setup
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

    # Use a Classifier (RidgeLogistic) for diagnostics
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
        ("model", LogisticRegression(penalty="l2", C=0.01, solver="lbfgs", max_iter=1000))
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

    # Combine results
    eval_df = prepared.eval_df.copy()
    eval_df["score"] = scores
    
    # We need beta and sector
    # Note: cache.get_prepared_fold might have dropped some columns from eval_df to save memory?
    # No, it keeps them. Let's verify.
    required_cols = ["date", "ticker", "capm_beta", "sector", "target_return", "score"]
    work = eval_df[required_cols].copy()
    work["sector"] = work["sector"].fillna("Unknown").astype(str)
    work["capm_beta"] = pd.to_numeric(work["capm_beta"], errors="coerce").fillna(1.0)
    work["target_return"] = pd.to_numeric(work["target_return"], errors="coerce").fillna(0.0)

    dates = sorted(work["date"].unique())
    daily_stats = []

    print("Running decomposition per date...")
    for dt in dates:
        day = work[work["date"] == dt].copy()
        if len(day) < 50: continue

        # Decompose Score: Score ~ Beta + SectorDummies
        y_score = day["score"].values
        y_ret = day["target_return"].values
        
        # Clean data for regression
        day["sector"] = day["sector"].fillna("Unknown")
        sectors = pd.get_dummies(day["sector"], drop_first=True)
        
        X = np.column_stack([
            np.ones(len(day)),
            day["capm_beta"].values,
            sectors.values.astype(float)
        ])
        
        try:
            # Use Moore-Penrose pseudo-inverse for stability
            coeffs = np.linalg.pinv(X) @ y_score
            
            intercept = coeffs[0]
            market_gamma = coeffs[1]
            sector_deltas = coeffs[2:]
            
            s_mkt = market_gamma * day["capm_beta"].values
            s_sec = intercept + (sectors.values.astype(float) @ sector_deltas)
            s_res = y_score - (s_mkt + s_sec)
            
            day["s_market"] = s_mkt
            day["s_sector"] = s_sec
            day["s_residual"] = s_res
            
            var_total = np.var(y_score)
            if var_total < 1e-12: continue

            row = {"date": dt}
            for col in ["s_market", "s_sector", "s_residual"]:
                # Correlation (IC)
                c_val = day[col].values
                if np.std(c_val) > 1e-12:
                    ic = np.corrcoef(c_val, y_ret)[0, 1]
                else:
                    ic = 0.0
                
                # Daily PnL (unit variance score)
                if np.std(c_val) > 1e-12:
                    norm_c = (c_val - np.mean(c_val)) / np.std(c_val)
                else:
                    norm_c = np.zeros_like(c_val)
                pnl = np.mean(norm_c * y_ret)
                
                row[f"{col}_ic"] = ic
                row[f"{col}_pnl"] = pnl
                row[f"{col}_var"] = np.var(c_val) / var_total
                
            daily_stats.append(row)
        except Exception:
            continue

    stats_df = pd.DataFrame(daily_stats).dropna()
    
    print("\n" + "="*60)
    print("      MODEL SIGNAL DECOMPOSITION: RISK VS ALPHA")
    print("="*60)
    
    summary = []
    for comp in ["s_market", "s_sector", "s_residual"]:
        m_ic = stats_df[f"{comp}_ic"].mean()
        abs_ic = stats_df[f"{comp}_ic"].abs().mean()
        
        mu = stats_df[f"{comp}_pnl"].mean()
        std = stats_df[f"{comp}_pnl"].std()
        sharpe = (mu / std * np.sqrt(252)) if std > 1e-9 else 0.0
        
        m_var = stats_df[f"{comp}_var"].mean()
        
        summary.append({
            "Component": comp.replace("s_", "").capitalize(),
            "Mean IC": m_ic,
            "Abs IC": abs_ic,
            "Sharpe": sharpe,
            "Var Explained": m_var
        })
        
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False, formatters={
        "Mean IC": "{:.4f}".format,
        "Abs IC": "{:.4f}".format,
        "Sharpe": "{:.2f}".format,
        "Var Explained": "{:.1%}".format
    }))
    
    # Attribution by Absolute IC (Predictive Power)
    ic_sum = sum(s["Abs IC"] for s in summary)
    print("\nSignal Attribution (Predictive Power via Abs IC):")
    for s in summary:
        attr = s["Abs IC"] / ic_sum if ic_sum > 0 else 0
        print(f"  {s['Component']:<10}: {attr:>6.1%}")

    # Variance Attribution
    print("\nScore Variance Decomposition:")
    for s in summary:
        print(f"  {s['Component']:<10}: {s['Var Explained']:>6.1%}")

    print("="*60)

if __name__ == "__main__":
    run_decomposition()
