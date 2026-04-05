
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def audit_probability_density(model_path="output/models/best_long_model.pkl"):
    print(f"--- Auditing Model: {model_path} ---")
    
    # 1. Load Model
    try:
        artifact = joblib.load(model_path)
        if isinstance(artifact, dict) and 'estimator' in artifact:
            model = artifact['estimator']
        else:
            model = artifact
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Simulate Input Data (using 1000 samples of noise for range check)
    # We ideally want real features, but we can check the weight distribution
    if hasattr(model, 'coef_'):
        print(f"Model Coefficients (Mean/Std): {np.mean(model.coef_):.6f} / {np.std(model.coef_):.6f}")
        print(f"Intercept: {model.intercept_}")
    
    # 3. If it's a RandomForest, we check tree votes
    if hasattr(model, 'estimators_'):
        print(f"Number of trees: {len(model.estimators_)}")

    print("\n--- Calibration Advice ---")
    print("If most probabilities are < 0.60, we should consider:")
    print("1. Standardizing features (Z-score) to widen the logit range.")
    print("2. Using CalibratedClassifierCV(method='sigmoid').")
    print("3. Switching to a 'Percentile-based' conviction gate instead of a fixed 0.60.")

if __name__ == "__main__":
    audit_probability_density()
