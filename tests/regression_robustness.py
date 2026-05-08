import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model_selection.validation import compute_execution_robustness, cross_sectional_ic
from run_model_selection import _proxy_turnover_from_scores

def run_regression():
    # Create deterministic synthetic data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=10)
    tickers = [f"T{i}" for i in range(20)]
    data = []
    for d in dates:
        for t in tickers:
            data.append({
                "date": d,
                "ticker": t,
                "score": np.random.normal(),
                "forward_return": np.random.normal(),
                "target_return_net": np.random.normal()
            })
    scored = pd.DataFrame(data)
    
    # OLD WAY (Explicit calls)
    ic_base = cross_sectional_ic(scored, target_col="forward_return")
    old_ic = float(ic_base.get("cs_ic_spearman_mean", np.nan))
    old_icir = float(ic_base.get("daily_ic_annualized_icir", np.nan))
    old_turnover = _proxy_turnover_from_scores(scored, primary_path="long_short_spread")
    
    old_score = (
        (old_ic if np.isfinite(old_ic) else -1.0)
        + 0.15 * (old_icir if np.isfinite(old_icir) else 0.0)
        - 0.10 * (old_turnover if np.isfinite(old_turnover) else 0.0)
    )
    
    # NEW WAY (compute_execution_robustness)
    stats = compute_execution_robustness(scored, primary_path="long_short_spread")
    new_ic = float(stats["ic_mean"])
    new_icir = float(stats["daily_icir"])
    new_turnover = float(stats["turnover_mean"])
    
    new_score = (
        new_ic
        + 0.15 * (new_icir if np.isfinite(new_icir) else 0.0)
        - 0.10 * (new_turnover if np.isfinite(new_turnover) else 0.0)
    )
    
    print("REGRESSION RESULTS:")
    print(f"IC:       Old={old_ic:10.6f} | New={new_ic:10.6f} | Match={np.isclose(old_ic, new_ic)}")
    print(f"ICIR:     Old={old_icir:10.6f} | New={new_icir:10.6f} | Match={np.isclose(old_icir, new_icir)}")
    print(f"Turnover: Old={old_turnover:10.6f} | New={new_turnover:10.6f} | Match={np.isclose(old_turnover, new_turnover)}")
    print(f"Score:    Old={old_score:10.6f} | New={new_score:10.6f} | Match={np.isclose(old_score, new_score)}")
    
    assert np.isclose(old_ic, new_ic)
    assert np.isclose(old_icir, new_icir)
    assert np.isclose(old_turnover, new_turnover)
    assert np.isclose(old_score, new_score)
    print("\nREGRESSION PASSED: No mathematical changes detected.")

if __name__ == "__main__":
    run_regression()
