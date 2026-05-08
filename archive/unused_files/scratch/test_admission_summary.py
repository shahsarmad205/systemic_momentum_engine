import pandas as pd
import numpy as np
from model_selection.alpha_research import summarize_admission

admission = pd.DataFrame([
    {"feature": "f1", "admitted": True, "recommended_action": "admit"},
    {"feature": "f2", "admitted": True, "recommended_action": "invert"},
    {"feature": "f3", "admitted": False, "recommended_action": "remove"},
    {"feature": "f4", "admitted": False, "recommended_action": "remove"},
    {"feature": "f5", "admitted": False, "recommended_action": "move_horizon"},
])

summary = summarize_admission(admission)
n_before = len(admission)
n_after = int(summary.get("alpha_features_admitted", 0))
n_inverted = int(summary.get("alpha_features_inverted", 0))
n_removed = int(summary.get("alpha_features_removed", 0))

print(f"[Feature Admission Summary] before={n_before} | after={n_after} | inverted={n_inverted} | removed={n_removed}")
