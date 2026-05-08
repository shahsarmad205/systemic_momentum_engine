import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model_selection.alpha_research import (
    AlphaAdmissionConfig,
    run_alpha_research,
    apply_admitted_feature_transforms,
)
from model_selection.training import TargetConfig
from model_selection.validation import ExecutionCostConfig

class TestPolarityInstitutional(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        dates = pd.date_range("2010-01-01", periods=200, freq="D")
        tickers = [f"T{i}" for i in range(35)] # Increase to 35 tickers
        data = []
        for i, d in enumerate(dates):
            for t in tickers:
                # Next day return
                next_ret = np.random.normal(0, 0.01)
                
                # Current date record
                # Signal strength
                sig = 5.0
                
                f_a = next_ret * sig + np.random.normal(0, 0.05)
                f_b = -next_ret * sig + np.random.normal(0, 0.05)
                
                idx = i
                if idx < 100:
                    f_c = next_ret * sig + np.random.normal(0, 0.05)
                else:
                    f_c = -next_ret * sig + np.random.normal(0, 0.05)
                
                f_d = np.random.normal(0, 1.0)
                
                data.append({
                    "date": d,
                    "ticker": t,
                    "daily_return_next": next_ret, # for alignment check
                    "f_a": f_a,
                    "f_b": f_b,
                    "f_c": f_c,
                    "f_d": f_d
                })
        
        self.df = pd.DataFrame(data)
        self.df = self.df.sort_values(["ticker", "date"])
        # We define forward_return as daily_return_next
        self.df["forward_return"] = self.df["daily_return_next"]
        # And daily_return as the shift(1) of daily_return_next
        self.df["daily_return"] = self.df.groupby("ticker")["daily_return_next"].shift(1)
        self.df = self.df.dropna()

    def test_institutional_polarity_flow(self):
        """
        Assert:
        - negative-IC training feature gets direction=-1
        - polarity is estimated using train only
        - transformed train IC becomes positive
        - test data does not affect direction
        """
        # Split into train (first 80 days) and test (remaining)
        cutoff = self.df["date"].unique()[80]
        train_df = self.df[self.df["date"] < cutoff].copy()
        full_df = self.df.copy()
        
        feat_cols = ["f_a", "f_b", "f_c", "f_d"]
        cfg = AlphaAdmissionConfig(
            min_abs_ic=0.01, 
            min_ic_tstat=2.0,
            production_horizon=1,
            allow_inversion=True,
            minimum_admitted_features=1,
            fail_if_below_minimum=False,
            min_marginal_abs_ic=0.0 # Disable marginality check
        )
        target_cfg = TargetConfig(horizon_days=1, residualize=False, net_of_costs=False)
        costs = ExecutionCostConfig()
        
        # 1. Estimate polarity on TRAIN only
        _, _, admission = run_alpha_research(
            train_df,
            feat_cols,
            cfg=cfg,
            target_cfg=target_cfg,
            costs=costs,
            max_name_weight=0.05
        )
        print("\nADMISSION DEBUG:")
        print(admission[["feature", "admitted", "production_ic", "production_ic_tstat", "recommended_action", "reason"]])
        
        # 2. Check admission results
        # f_a should be admitted with sign +1
        # f_b should be admitted with sign -1
        # f_c (in train) should be admitted with sign +1
        # f_d should probably be removed or have low IC
        
        a_row = admission[admission["feature"] == "f_a"].iloc[0]
        b_row = admission[admission["feature"] == "f_b"].iloc[0]
        c_row = admission[admission["feature"] == "f_c"].iloc[0]
        
        self.assertTrue(a_row["admitted"])
        self.assertEqual(int(a_row["transform_sign"]), 1)
        
        self.assertTrue(b_row["admitted"])
        self.assertEqual(int(b_row["transform_sign"]), -1)
        
        self.assertTrue(c_row["admitted"])
        self.assertEqual(int(c_row["transform_sign"]), 1)
        
        # 3. Apply transforms to FULL df
        transformed_df = apply_admitted_feature_transforms(full_df, admission)
        
        # Verify f_b sign was flipped in FULL df
        # Original f_b had negative correlation with forward_return
        # Transformed f_b should have positive correlation in TRAIN
        train_transformed = transformed_df[transformed_df["date"] < cutoff]
        ic_b_train = train_transformed["f_b"].corr(train_transformed["forward_return"], method="spearman")
        self.assertGreater(ic_b_train, 0, "Flipped feature B should have positive IC in training")
        
        # 4. Verify that flipping sign in TEST (for f_c) does not happen if not in admission
        # f_c was admitted with +1 based on train. 
        # In the second half of full_df, f_c has negative IC. 
        # Since transform_sign is +1, it should remain negative in test.
        test_transformed = transformed_df[transformed_df["date"] >= cutoff]
        ic_c_test = test_transformed["f_c"].corr(test_transformed["forward_return"], method="spearman")
        self.assertLess(ic_c_test, 0, "Feature C should have negative IC in test because it flipped and we froze train polarity")

if __name__ == "__main__":
    unittest.main()
