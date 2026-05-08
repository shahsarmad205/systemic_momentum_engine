import numpy as np
import pandas as pd
from scipy.stats import rankdata
import pytest

# --- OLD PANDAS-HEAVY IMPLEMENTATION (RECONSTRUCTED) ---

def _get_implied_weights_OLD(df, primary_path):
    if df.empty or "score" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["rank"] = work.groupby("date")["score"].rank(pct=True, method="average")
    
    if primary_path == "long_only_overlay":
        work["_w_raw"] = (work["rank"] - 0.5).clip(lower=0.0)
    elif primary_path == "short_side":
        work["_w_raw"] = -(0.5 - work["rank"]).clip(lower=0.0)
    else:  # long_short_spread
        work["_w_raw"] = work["rank"] - 0.5
        
    work["_w_abs_sum"] = work.groupby("date")["_w_raw"].transform(lambda x: x.abs().sum())
    work["_w"] = work["_w_raw"] / work["_w_abs_sum"].replace(0, np.nan)
    return work.dropna(subset=["_w"])

def compute_execution_robustness_OLD(scored, primary_path, target_col="forward_return"):
    out = {}
    
    # Simple Spearman IC mean
    def get_spearman(g):
        if len(g) < 5: return np.nan
        return g["score"].corr(g[target_col], method="spearman")
    
    daily_ic = scored.groupby("date").apply(get_spearman)
    out["cost_adjusted_ic_mean"] = daily_ic.mean()
    
    # Turnover Volatility using Pivot
    w_df = _get_implied_weights_OLD(scored, primary_path=primary_path)
    if not w_df.empty:
        # HHI
        out["hhi_concentration"] = w_df.groupby("date")["_w"].apply(lambda x: (x**2).sum()).mean()
        
        # Turnover
        piv = w_df.pivot_table(index="date", columns="ticker", values="_w").fillna(0.0)
        daily_turnover = piv.diff().abs().sum(axis=1)
        out["turnover_volatility"] = daily_turnover.std()

    # Signal Halflife (Loop)
    dates = sorted(scored["date"].unique())
    rhos = []
    scored["rank"] = scored.groupby("date")["score"].rank(pct=True, method="average")
    for i in range(len(dates)-1):
        d0, d1 = dates[i], dates[i+1]
        v0 = scored[scored["date"] == d0].set_index("ticker")["rank"]
        v1 = scored[scored["date"] == d1].set_index("ticker")["rank"]
        common = v0.index.intersection(v1.index)
        if len(common) > 10:
            rho = np.corrcoef(v0.loc[common], v1.loc[common])[0,1]
            if np.isfinite(rho): rhos.append(rho)
    
    if rhos:
        avg_rho = np.mean(rhos)
        if 0 < avg_rho < 1.0:
            out["signal_halflife_days"] = -np.log(2) / np.log(avg_rho)
        elif avg_rho >= 1.0:
            out["signal_halflife_days"] = 100.0
        else:
            out["signal_halflife_days"] = 0.0
            
    return out

# --- IMPORT NEW IMPLEMENTATION ---
from model_selection.validation import compute_execution_robustness as compute_NUMPY

def test_robustness_parity():
    """Verify exact mathematical parity between Pandas baseline and NumPy refactor."""
    # 1. Create Messy Deterministic Fixture
    np.random.seed(42)
    n_days = 30
    n_tickers = 100
    dates = pd.date_range("2020-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    
    data = []
    for d in dates:
        # Some tickers missing every day
        active = np.random.choice(tickers, size=int(n_tickers * 0.7), replace=False)
        for t in active:
            score = np.random.randn()
            # Add some ties
            if np.random.rand() < 0.1: score = 0.0
            
            fwd_ret = np.random.randn() * 0.02
            # Add NaNs
            if np.random.rand() < 0.05: fwd_ret = np.nan
            
            data.append({
                "date": d,
                "ticker": t,
                "score": score,
                "forward_return": fwd_ret,
                "target_return": fwd_ret
            })
            
    scored = pd.DataFrame(data)
    
    # 2. Run Both Versions
    primary_path = "long_short_spread"
    old_res = compute_execution_robustness_OLD(scored, primary_path)
    new_res = compute_NUMPY(scored, primary_path=primary_path)
    
    # 3. Assert Equality within floating point tolerance
    metrics = [
        "signal_halflife_days",
        "cost_adjusted_ic_mean",
        "turnover_volatility",
        "hhi_concentration"
    ]
    
    for m in metrics:
        v_old = old_res.get(m, np.nan)
        v_new = new_res.get(m, np.nan)
        
        if np.isnan(v_old):
            assert np.isnan(v_new), f"{m} should be NaN"
        else:
            diff = abs(v_old - v_new)
            assert diff < 1e-12, f"{m} mismatch: old={v_old}, new={v_new}, diff={diff}"

if __name__ == "__main__":
    # Run directly for manual confirmation
    test_robustness_parity()
    print("SUCCESS: All metrics match research baseline within float tolerance.")
