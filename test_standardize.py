import pickle, numpy as np, pandas as pd
from utils.ensemble_scoring import load_ensemble_models, _zscore_standardize, compute_ensemble_score
from backtesting.config import load_config

config = load_config("backtest_config.yaml")
ens_cfg = {'models': config.ensemble_models, 'normalize': False, 'standardize': True, 'clip': False}

models = load_ensemble_models(ens_cfg)
print(f"Loaded {len(models)} models")

# Create some fake features that mimic the inputs
dates = pd.date_range("2020-01-01", periods=1000, freq="D")
features_df = pd.DataFrame({col: np.random.randn(1000) for col in models[0].feature_columns}, index=dates)

# check raw output of model 0
s_raw = None
for m in models:
    from utils.ensemble_scoring import _predict_model
    s = _predict_model(m, features_df, False).astype(float)
    s_std = _zscore_standardize(s)
    print(f"{m.path.split('/')[-1]}: raw_std={s.std():.6f}, zscore_std={s_std.std():.6f}, mean={s_std.mean():.6f}")

out = compute_ensemble_score(features_df, models, normalize=False, standardize=True)
print(f"Ensemble score: std={out.std():.6f}, mean={out.mean():.6f}")
