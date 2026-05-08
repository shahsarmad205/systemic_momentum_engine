import unittest
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model_selection.validation import compute_execution_robustness

class TestRobustnessSchema(unittest.TestCase):
    def test_schema_contract(self):
        """Verify that compute_execution_robustness returns all expected keys, even if NaNs."""
        # Create a minimal valid scored dataframe
        dates = pd.date_range("2020-01-01", periods=10)
        tickers = ["A", "B", "C"]
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
        
        # Call function
        stats = compute_execution_robustness(scored, primary_path="long_short_spread")
        
        # Required keys audit (Task 1)
        required_keys = [
            "cs_ic_spearman_mean",
            "daily_ic_annualized_icir",
            "ic_mean",
            "daily_icir",
            "turnover_mean",
            "turnover_volatility",
            "signal_halflife_days",
            "cost_adjusted_ic_mean",
            "caic_ratio",
            "capacity_weighted_ic",
            "decile_tail_stability",
            "hhi_concentration",
            "robustness_score",
            "robustness_reason"
        ]
        
        for key in required_keys:
            with self.subTest(key=key):
                self.assertIn(key, stats, f"Missing required key: {key}")
                val = stats[key]
                # Check that it's not None (Task 2)
                self.assertIsNotNone(val, f"Key {key} is None, should be np.nan or finite")
                # Ensure it's a float or int (except reason)
                if key != "robustness_reason":
                    self.assertTrue(isinstance(val, (float, int, np.floating, np.integer)), 
                                  f"Key {key} should be numeric, got {type(val)}")

    def test_empty_df_returns_nans(self):
        """Verify that empty input still returns the full schema with NaNs."""
        scored = pd.DataFrame(columns=["date", "ticker", "score", "forward_return"])
        stats = compute_execution_robustness(scored, primary_path="long_short_spread")
        
        required_keys = ["ic_mean", "daily_icir", "turnover_mean", "signal_halflife_days"]
        for key in required_keys:
            self.assertIn(key, stats)
            self.assertTrue(np.isnan(stats[key]))

    def test_constant_scores_do_not_emit_runtime_warnings(self):
        """Degenerate model scores should fail closed without noisy divide/mean warnings."""
        dates = pd.date_range("2020-01-01", periods=12)
        tickers = [f"T{i}" for i in range(20)]
        scored = pd.DataFrame(
            {
                "date": np.repeat(dates, len(tickers)),
                "ticker": tickers * len(dates),
                "score": 1.0,
                "forward_return": np.random.default_rng(7).normal(size=len(dates) * len(tickers)),
                "adv_dollar_20": 1_000_000.0,
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            stats = compute_execution_robustness(scored, primary_path="long_short_spread")

        self.assertIn("signal_halflife_days", stats)
        self.assertTrue(np.isnan(stats["capacity_weighted_ic"]))

if __name__ == "__main__":
    unittest.main()
