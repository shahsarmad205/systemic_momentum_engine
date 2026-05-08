import pandas as pd
import numpy as np

def audit_metrics(csv_path):
    df = pd.read_csv(csv_path)
    report = []
    
    for _, row in df.iterrows():
        model = row["model_name"]
        path = row["oos_evaluation_path"]
        
        # 1. High turnover but near-zero cost
        turnover = row.get("exec_turnover_mean", 0)
        cost = row.get("exec_cost_return_sum", 0)
        if turnover > 0.1 and abs(cost) < 1e-6:
            report.append(f"[Contradiction] {model} ({path}): High turnover ({turnover:.4f}) but near-zero cost ({cost:.6f})")
            
        # 2. Positive Sharpe but gate fail
        sharpe = row.get("exec_sharpe", 0)
        passed = row.get("promotion_pass", True)
        failures = row.get("promotion_failures", "")
        if sharpe > 0.5 and not passed:
            report.append(f"[Check] {model} ({path}): Positive Sharpe ({sharpe:.2f}) but failed gates: {failures}")
            
        # 3. High directional accuracy but strongly negative PnL
        acc = row.get("oos_dir_acc_mean", 0)
        if acc > 0.55 and sharpe < -1.0:
            report.append(f"[Contradiction] {model} ({path}): High accuracy ({acc:.4f}) but very negative Sharpe ({sharpe:.2f})")
            
        # 4. Positive IC but negative decile spread
        ic = row.get("oos_ic_chained", 0)
        spread = row.get("decile_spread", 0)
        if ic > 0.02 and spread < 0:
            report.append(f"[Contradiction] {model} ({path}): Positive IC ({ic:.4f}) but negative decile spread ({spread:.6f})")
            
        # 5. Nested vs OOS consistency
        nested_ic = row.get("nested_ic_mean", np.nan)
        oos_ic = row.get("oos_ic_chained", np.nan)
        if pd.notnull(nested_ic) and pd.notnull(oos_ic):
            diff = abs(nested_ic - oos_ic)
            if diff > 0.1:
                report.append(f"[Consistency] {model} ({path}): Large IC gap (Nested={nested_ic:.4f}, OOS={oos_ic:.4f}, Diff={diff:.4f})")

        # 6. Cost vs POV
        pov = row.get("exec_participation_mean", 0)
        if pov > 0.05 and abs(cost) < 0.0001:
            report.append(f"[Check] {model} ({path}): High POV ({pov:.4f}) but low reported cost ({cost:.6f})")

    return report

if __name__ == "__main__":
    csv_path = "output/models/model_comparison.csv"
    results = audit_metrics(csv_path)
    print("=== Metric Contradiction Report ===")
    for r in results:
        print(r)
    if not results:
        print("No major contradictions found.")
