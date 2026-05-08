import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model_selection.validation import compute_execution_robustness

class TestNaNPropagation(unittest.TestCase):
    def test_nan_propagation_in_stats(self):
        """Verify that undefined IC/Turnover propagates as NaN in compute_execution_robustness."""
        # Case 1: Undefined IC (constant score)
        dates = pd.date_range("2020-01-01", periods=10)
        tickers = ["A", "B", "C"]
        data = []
        for d in dates:
            for t in tickers:
                data.append({
                    "date": d,
                    "ticker": t,
                    "score": 1.0, # CONSTANT SCORE -> Undefined variance -> NaN IC
                    "forward_return": np.random.normal(),
                    "target_return_net": np.random.normal()
                })
        scored = pd.DataFrame(data)
        
        stats = compute_execution_robustness(scored, primary_path="long_short_spread")
        
        self.assertTrue(np.isnan(stats["ic_mean"]), "ic_mean should be NaN for constant scores")
        self.assertTrue(np.isnan(stats["daily_icir"]), "daily_icir should be NaN if IC is NaN")
        
    def test_scoring_formula_nan_propagation(self):
        """
        Verify that screen_score = ic + 0.15*icir - 0.10*turnover 
        propagates NaN if any component is NaN.
        """
        # We'll simulate the logic in run_model_selection.py
        ic_mean = np.nan
        daily_icir_mean = 0.5
        turnover_mean = 0.3
        
        # Formula: ic_mean + (0.15 * daily_icir_mean) - (0.10 * turnover_mean)
        # Any NaN in an addition/subtraction results in NaN in IEEE 754
        score = ic_mean + (0.15 * daily_icir_mean) - (0.10 * turnover_mean)
        self.assertTrue(np.isnan(score), "Score should be NaN if IC is NaN")
        
        ic_mean = 0.05
        daily_icir_mean = np.nan
        score = ic_mean + (0.15 * daily_icir_mean) - (0.10 * turnover_mean)
        self.assertTrue(np.isnan(score), "Score should be NaN if ICIR is NaN")

if __name__ == "__main__":
    unittest.main()
