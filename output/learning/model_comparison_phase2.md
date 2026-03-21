# Model comparison (ridge / gbr / xgb)

| Model | WF IC | Train IC | Net Sharpe | CAGR | MaxDD |
|-------|-------|----------|------------|------|-------|
| ridge | 0.0610 | 0.0678 | 1.1220 | 8.73% | -11.66% |
| gbr | 0.0536 | 0.1731 | 1.1220 | 8.73% | -11.66% |
| xgb | 0.0325 | 0.1706 | 1.1220 | 8.73% | -11.66% |

**Selected (best net Sharpe): `ridge`** → `output/learned_weights.json`
