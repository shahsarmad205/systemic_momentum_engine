import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from model_selection.preparation import PreparedPanelCache
from model_selection.training import TargetConfig, make_training_target
from model_selection.validation import ExecutionCostConfig, cross_sectional_ic, annualized_sharpe

def compute_monotonicity(scored, target_col="target_return"):
    df = scored.copy()
    df["decile"] = df.groupby("date")["score"].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
    decile_means = df.groupby("decile")[target_col].mean()
    # Correlation of decile index with decile mean return
    if len(decile_means) < 2: return 0.0
    return np.corrcoef(np.arange(len(decile_means)), decile_means.values)[0, 1]

def run_counterfactual_audit():
    print("Loading enriched panel...")
    df = pd.read_parquet("output/models/enriched_panel.parquet")

    tr_s = pd.Timestamp("2021-01-01")
    tr_e = pd.Timestamp("2022-01-01")
    ev_s = pd.Timestamp("2022-01-01")
    ev_e = pd.Timestamp("2022-06-01")
    
    target_cfg = TargetConfig(horizon_days=5)
    costs = ExecutionCostConfig()
    cache = PreparedPanelCache(df, target_cfg=target_cfg, costs=costs, max_name_weight=0.1, winsor_q=0.01)

    active_feats = ['ret_10d', 'cs_momentum_percentile', 'nearness_52w_high', 'days_to_cover', 'momentum_acceleration']
    
    print("Preparing fold...")
    prepared = cache.get_prepared_fold(
        train_start=tr_s, train_end=tr_e,
        eval_start=ev_s, eval_end=ev_e,
        horizon_days=5, active_features=active_feats
    )

    y_tr = make_training_target(prepared.train_df, model_name="RidgeLogistic", model_kind="classifier", use_risk_adj=False)
    
    model = Pipeline([
        ("scaler", RobustScaler()),
        ("model", LogisticRegression(penalty="l2", C=0.1, solver="liblinear"))
    ])

    print("Fitting RidgeLogistic...")
    model.fit(prepared.x_train, y_tr)
    
    print("Scoring evaluation set...")
    # Get scores
    scores = model.predict_proba(prepared.x_eval)[:, 1] - 0.5
    
    eval_df = prepared.eval_df.copy()
    eval_df["target_return"] = pd.to_numeric(eval_df["target_return"], errors="coerce").fillna(0.0)
    eval_df["target_expected_cost"] = pd.to_numeric(eval_df["target_expected_cost"], errors="coerce").fillna(0.0)
    
    def evaluate_score(s_vec, label):
        tmp = eval_df.copy()
        tmp["score"] = s_vec
        
        # IC
        ic_res = cross_sectional_ic(tmp, target_col="target_return")
        ic = ic_res["cs_ic_spearman_mean"]
        
        # Leg Returns
        tmp["rank"] = tmp.groupby("date")["score"].transform(lambda x: x.rank(pct=True))
        long_days = tmp[tmp["rank"] > 0.9].groupby("date")["target_return"].mean()
        short_days = tmp[tmp["rank"] <= 0.1].groupby("date")["target_return"].mean()
        ls_days = long_days - short_days
        
        l_sharpe = annualized_sharpe(long_days)
        s_sharpe = annualized_sharpe(-short_days)
        ls_sharpe = annualized_sharpe(ls_days)
        
        spread = long_days.mean() - short_days.mean()
        mono = compute_monotonicity(tmp)
        
        # CAIC
        r_adj = tmp["target_return"] - np.sign(tmp["score"]) * tmp["target_expected_cost"]
        tmp["r_adj"] = r_adj
        caic_res = cross_sectional_ic(tmp, target_col="r_adj")
        caic = ca_ic = caic_res["cs_ic_spearman_mean"]
        
        return {
            "Label": label,
            "IC": ic,
            "L_Sharpe": l_sharpe,
            "S_Sharpe": s_sharpe,
            "LS_Sharpe": ls_sharpe,
            "Spread": spread,
            "Mono": mono,
            "CAIC": caic
        }

    res_orig = evaluate_score(scores, "ORIGINAL")
    res_flipped = evaluate_score(-scores, "FLIPPED")

    print("\n" + "="*80)
    print("      COUNTERFACTUAL SIGNAL INVERSION AUDIT")
    print("="*80)
    print(f"{'Metric':<20} | {'ORIGINAL':>15} | {'FLIPPED':>15}")
    print("-" * 80)
    for k in res_orig.keys():
        if k == "Label": continue
        v1, v2 = res_orig[k], res_flipped[k]
        print(f"{k:<20} | {v1:>15.4f} | {v2:>15.4f}")
    print("="*80)

if __name__ == "__main__":
    run_counterfactual_audit()
