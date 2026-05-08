import pandas as pd, numpy as np
from scipy.stats import pearsonr
import os

report_path = 'output/backtests/walk_forward_validation_report.csv'
if not os.path.exists(report_path):
    print(f"Error: {report_path} not found.")
    exit(1)

df = pd.read_csv(report_path)
ic = df['oos_information_coefficient']
t_stat = ic.mean() / (ic.std() / np.sqrt(len(ic)))
p, _ = pearsonr(ic, df['oos_sharpe'])

print('Mean IC:', round(ic.mean(), 4))
print('IC t-stat:', round(t_stat, 2))
print('IC-Sharpe correlation:', round(p, 3))
print('% positive IC windows:', round((ic > 0).mean()*100, 1), '%')
