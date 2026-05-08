"""Run Cost Viability Engine on real panel data and produce PM-level report."""
import os
import numpy as np
import pandas as pd
from collections import Counter

from model_selection.cost_viability_engine import (
    CostViabilityEngine, CostStatus,
    generate_scorecard, generate_stress_test_report,
    generate_cost_dominated_report, generate_turnover_attribution_report,
)

panel = pd.read_parquet('output/models/enriched_panel_temp.parquet')
print(f'Panel: {len(panel)} rows, {panel["ticker"].nunique()} tickers, {panel["date"].nunique()} dates')

feature_families = {
    'momentum': ['ret_5d', 'ret_10d', 'ret_20d', 'momentum_12m_skip1', 'nearness_52w_high', 'momentum_acceleration'],
    'short_momentum': ['momentum_1m_skip_eom'],
    'reversal': ['short_term_reversal', 'nearness_52w_low', 'industry_relative_reversal'],
    'trend': ['f_trend'],
    'risk': ['rolling_vol_20', 'capm_residual_vol'],
    'quality': ['quality_score'],
    'residual_alpha': ['capm_alpha'],
    'quality_lowvol': ['low_vol_score'],
    'squeeze_filter': ['short_squeeze_risk'],
    'sector_relative': ['sector_relative_20d', 'sector_relative_60d'],
    'regime': ['vol_ratio_5_20'],
}

engine = CostViabilityEngine()
results = []
stress_results = {}
horizons = [1, 2, 3, 5, 10, 20]

for family, features in feature_families.items():
    for feat in features:
        if feat not in panel.columns:
            continue
        for h in horizons:
            fwd = panel.groupby('date').apply(
                lambda g: g[feat].corr(g['daily_return']) if len(g) > 10 else np.nan
            )
            mean_ic = fwd.dropna().mean()
            if np.isnan(mean_ic) or abs(mean_ic) < 0.001:
                continue

            adv = panel['adv_dollar_20'].median() if 'adv_dollar_20' in panel.columns else 50_000_000
            vol = panel['daily_return'].std() * np.sqrt(252) if 'daily_return' in panel.columns else 0.20
            daily_vol = vol / np.sqrt(252)

            feat_vals = panel[feat].dropna()
            if len(feat_vals) > 100:
                ac1 = feat_vals.autocorr()
                expected_turnover = max(0.01, min(0.80, 1.0 - abs(ac1)))
            else:
                expected_turnover = 0.10

            ranks = panel.groupby('date')[feat].rank(pct=True)
            if len(ranks.dropna()) > 100:
                ac_ranks = ranks.dropna().autocorr()
                if ac_ranks > 0:
                    halflife = -1.0 / np.log2(ac_ranks) if ac_ranks < 1.0 else 20.0
                else:
                    halflife = 1.0
            else:
                halflife = 1.0

            cid = f'{family}_{feat}_h{h}'
            result = engine.evaluate(
                candidate_id=cid, feature=feat, family=family,
                ic=mean_ic, horizon=h, sigma_annual=0.20,
                halflife=halflife, expected_turnover=expected_turnover,
                adv_usd=adv, daily_vol=daily_vol,
                n_dates=panel['date'].nunique(),
                avg_breadth=panel['ticker'].nunique(),
            )
            results.append(result)

            stress = engine.run_stress_test(
                candidate_id=cid, feature=feat, family=family,
                ic=mean_ic, horizon=h, sigma_annual=0.20,
                halflife=halflife, expected_turnover=expected_turnover,
                adv_usd=adv, daily_vol=daily_vol,
            )
            stress_results[cid] = stress

print(f'Evaluated {len(results)} candidates')

# --- No-trade band evaluation ---
from model_selection.cost_viability_engine import NoTradeBandEngine
band_engine = NoTradeBandEngine(config=engine.config)
band_results = []
for r in results:
    br = band_engine.apply(
        candidate_id=r.candidate_id,
        current_weight=0.02,
        target_weight=r.expected_alpha_bps / 100.0,
        expected_cost_bps=r.expected_cost_bps,
        daily_vol=r.sigma_annual / np.sqrt(252),
        adv_usd=r.adv_usd,
        abs_ic=abs(r.ic),
        expected_alpha_bps=r.expected_alpha_bps,
    )
    band_results.append(br)

os.makedirs('output/models/cost_viability', exist_ok=True)
generate_scorecard(results, 'output/models/cost_viability/cost_viability_scorecard.csv')
generate_stress_test_report(stress_results, 'output/models/cost_viability/cost_stress_test.csv')
generate_cost_dominated_report(results, 'output/models/cost_viability/cost_dominated_candidates.csv')
generate_turnover_attribution_report(band_results, 'output/models/cost_viability/turnover_attribution.csv')

# PM-level stats
print()
print('=' * 60)
print('PM-LEVEL COST BOTTLENECK REPORT')
print('=' * 60)
print()

gross_pos = [r for r in results if r.expected_alpha_bps > 0]
print(f'1. Gross-alpha positive: {len(gross_pos)}/{len(results)} ({len(gross_pos)/len(results)*100:.0f}%)')

viable = [r for r in results if r.cost_status == CostStatus.COST_VIABLE]
print(f'2. Cost viable: {len(viable)}/{len(results)} ({len(viable)/len(results)*100:.0f}%)')

dominated = [r for r in results if r.cost_status == CostStatus.COST_DOMINATED]
print(f'3. Cost dominated: {len(dominated)}/{len(results)} ({len(dominated)/len(results)*100:.0f}%)')

reasons = Counter()
for r in results:
    if r.cost_breakdown and r.cost_breakdown.total_bps > 0:
        bd = r.cost_breakdown
        components = [
            ('spread', bd.spread_bps),
            ('impact', bd.temporary_impact_bps + bd.permanent_impact_bps),
            ('commission', bd.commission_bps),
            ('borrow', bd.borrow_bps),
        ]
        max_comp = max(components, key=lambda x: x[1])
        reasons[max_comp[0]] += 1
print(f'4. Dominant cost component: {reasons.most_common()}')

by_horizon = {}
for r in results:
    h = r.horizon
    if h not in by_horizon:
        by_horizon[h] = {'total': 0, 'viable': 0}
    by_horizon[h]['total'] += 1
    if r.cost_status == CostStatus.COST_VIABLE:
        by_horizon[h]['viable'] += 1
print(f'5. Horizon survival:')
for h in sorted(by_horizon.keys()):
    d = by_horizon[h]
    pct = d['viable']/d['total']*100 if d['total'] > 0 else 0
    print(f'   h{h}: {d["viable"]}/{d["total"]} viable ({pct:.0f}%)')

by_family = {}
for r in results:
    f = r.family
    if f not in by_family:
        by_family[f] = {'total': 0, 'viable': 0, 'net_alpha': []}
    by_family[f]['total'] += 1
    if r.cost_status == CostStatus.COST_VIABLE:
        by_family[f]['viable'] += 1
    by_family[f]['net_alpha'].append(r.net_expected_alpha_bps)
print(f'6. Family performance:')
for f in sorted(by_family.keys(), key=lambda x: -by_family[x]['viable']):
    d = by_family[f]
    avg_net = np.mean(d['net_alpha']) if d['net_alpha'] else 0
    print(f'   {f}: {d["viable"]}/{d["total"]} viable, avg net alpha={avg_net:.1f}bps')

top_acr = sorted(results, key=lambda x: -x.alpha_cost_ratio)[:10]
print(f'7. Top 10 alpha/cost ratio:')
for r in top_acr:
    print(f'   {r.candidate_id}: ACR={r.alpha_cost_ratio:.2f}, net={r.net_expected_alpha_bps:.1f}bps, status={r.cost_status.value}')

print(f'8. Min alpha/cost for production: {engine.promotion_gates["min_alpha_cost_ratio"]}')

print(f'9. Capacity of surviving sleeves:')
for r in viable:
    print(f'   {r.candidate_id}: capacity_score={r.capacity_score:.1f}, adv=${r.adv_usd/1e6:.0f}M')

print(f'10. Research direction:')
print(f'    - {len(dominated)} fail on cost dominance')
print(f'    - {len([r for r in results if r.cost_status == CostStatus.TURNOVER_DOMINATED])} fail on turnover')
print(f'    - {len([r for r in results if r.cost_status == CostStatus.ALPHA_TOO_WEAK])} fail on alpha strength')
print(f'    - {len([r for r in results if r.cost_status == CostStatus.LIQUIDITY_INSUFFICIENT])} fail on liquidity')
print(f'    - {len([r for r in results if r.cost_status == CostStatus.MARGINAL])} are marginal')
print(f'    Focus: reduce turnover, target longer horizons, improve feature persistence')
