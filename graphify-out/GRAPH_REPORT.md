# Graph Report - ../Research-paper/Portfolio  (2026-04-14)

## Corpus Check
- Corpus is ~0 words - fits in a single context window. You may not need a graph.

## Summary
- 39 nodes · 50 edges · 6 communities detected
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Top-Down Factor Selection|Top-Down Factor Selection]]
- [[_COMMUNITY_Turnover & Rebalancing Control|Turnover & Rebalancing Control]]
- [[_COMMUNITY_Bottom-Up Construction & Methodology|Bottom-Up Construction & Methodology]]
- [[_COMMUNITY_Factor Efficiency Ratio Framework|Factor Efficiency Ratio Framework]]
- [[_COMMUNITY_Weighting Schemes|Weighting Schemes]]
- [[_COMMUNITY_Sector Neutrality & Risk|Sector Neutrality & Risk]]

## God Nodes (most connected - your core abstractions)
1. `Top-Down Multi-Factor Portfolio Construction` - 9 edges
2. `Exploring Techniques in Multi-Factor Index Construction` - 7 edges
3. `Composite Multi-Factor Score (Average of Single-Factor Z-Scores)` - 6 edges
4. `Bottom-Up Multi-Factor Portfolio Construction` - 5 edges
5. `Sector-Neutral Portfolio Construction` - 5 edges
6. `Factor Efficiency Ratio (FER)` - 5 edges
7. `Active Share / Target Active Share` - 5 edges
8. `Portfolio Turnover` - 4 edges
9. `Factor Imbalance Metric` - 4 edges
10. `Proposed FER — Active Factor Exposure per Unit Tracking Error (Eq. 3)` - 3 edges

## Surprising Connections (you probably didn't know these)
- `Top-Down Multi-Factor Portfolio Construction` --conceptually_related_to--> `Portfolio Turnover`  [INFERRED]
  ../Research-paper/Portfolio/research-exploring-techniques-in-multi-factor-index-construction.pdf → ../Research-paper/Portfolio/research-exploring-techniques-in-multi-factor-index-construction.pdf  _Bridges community 0 → community 1_
- `Bottom-Up Multi-Factor Portfolio Construction` --conceptually_related_to--> `Portfolio Turnover`  [INFERRED]
  ../Research-paper/Portfolio/research-exploring-techniques-in-multi-factor-index-construction.pdf → ../Research-paper/Portfolio/research-exploring-techniques-in-multi-factor-index-construction.pdf  _Bridges community 2 → community 1_
- `Sector-Neutral Portfolio Construction` --conceptually_related_to--> `Factor Imbalance Metric`  [INFERRED]
  ../Research-paper/Portfolio/research-exploring-techniques-in-multi-factor-index-construction.pdf → ../Research-paper/Portfolio/research-exploring-techniques-in-multi-factor-index-construction.pdf  _Bridges community 5 → community 2_
- `Exploring Techniques in Multi-Factor Index Construction` --references--> `Top-Down Multi-Factor Portfolio Construction`  [EXTRACTED]
  ../Research-paper/Portfolio/research-exploring-techniques-in-multi-factor-index-construction.pdf → ../Research-paper/Portfolio/research-exploring-techniques-in-multi-factor-index-construction.pdf  _Bridges community 2 → community 0_
- `Exploring Techniques in Multi-Factor Index Construction` --references--> `Factor Efficiency Ratio (FER)`  [EXTRACTED]
  ../Research-paper/Portfolio/research-exploring-techniques-in-multi-factor-index-construction.pdf → ../Research-paper/Portfolio/research-exploring-techniques-in-multi-factor-index-construction.pdf  _Bridges community 2 → community 3_

## Hyperedges (group relationships)
- **Four-Factor Multi-Factor Model (Quality, Value, Momentum, Low Volatility)** — mfic_quality_factor, mfic_value_factor, mfic_momentum_factor, mfic_low_volatility_factor, mfic_composite_score [EXTRACTED 1.00]
- **Multi-Factor Portfolio Construction Decision Framework** — mfic_top_down_approach, mfic_bottom_up_approach, mfic_sector_neutral, mfic_sector_agnostic, mfic_active_share, mfic_market_cap_weighting, mfic_score_weighting, mfic_hybrid_weighting, mfic_rebalance_frequency [EXTRACTED 1.00]
- **Factor Efficiency Ratio Variants** — mfic_fer_hunstad, mfic_fer_simplified, mfic_fer_proposed, mfic_absolute_fer [EXTRACTED 1.00]
- **Turnover Trade-Off: Active Share, Rebalance Frequency, Factor Decay** — mfic_active_share, mfic_rebalance_frequency, mfic_factor_decay, mfic_portfolio_turnover [EXTRACTED 1.00]

## Communities

### Community 0 - "Top-Down Factor Selection"
Cohesion: 0.27
Nodes (10): Composite Multi-Factor Score (Average of Single-Factor Z-Scores), Low Volatility Factor, Momentum Factor, Quality Factor, Rationale: Top-Down Factor Dilution Compensated by Proportional Tracking Error Reduction, S&P Enhanced Value Index Methodology, S&P Momentum Index Methodology, S&P Quality Index Methodology (+2 more)

### Community 1 - "Turnover & Rebalancing Control"
Cohesion: 0.25
Nodes (8): Active Share / Target Active Share, Factor Decay (Diminishing Factor Exposure Between Rebalances), Portfolio Turnover, Rationale: Linear Relationship Between Active Share and Tracking Error Means No Optimal Concentration Point, Rationale: Semiannual Rebalancing Reduces Turnover Cost at Expense of Factor Decay; Higher Active Share Compensates, Rebalancing Frequency (Quarterly vs Semiannual), Target Stock Count Selection Method, Target Weight Selection Method

### Community 2 - "Bottom-Up Construction & Methodology"
Cohesion: 0.29
Nodes (8): Backtest on S&P 500 Universe (Mar 1995 – May 2020, 116 portfolios), Bottom-Up Multi-Factor Portfolio Construction, Factor Exposure Correlations (Cross-Factor), Factor Imbalance Metric, 'The Merits and Methods of Multi-Factor Investing' — Innes (2018), Exploring Techniques in Multi-Factor Index Construction, Rank-Based Multi-Factor Portfolio Selection, Sector-Agnostic Portfolio Construction

### Community 3 - "Factor Efficiency Ratio Framework"
Cohesion: 0.4
Nodes (5): Absolute FER — Absolute Factor Exposure per Portfolio Concentration (Eq. 4), Factor Efficiency Ratio (FER), FER by Hunstad and Dekhayser (2014) — Risk Decomposition Variant, Simplified FER — Intended vs Unintended Factor Exposure Ratio, Herfindahl-Hirschman Index for Portfolio Concentration

### Community 4 - "Weighting Schemes"
Cohesion: 0.5
Nodes (4): Hybrid Weighting Scheme (FMC x Factor Score), Float-Adjusted Market-Cap Weighting Scheme, Rationale: FMC x Score Hybrid Weighting Maintains Liquidity While Improving Factor Tilt, Factor Score Weighting Scheme

### Community 5 - "Sector Neutrality & Risk"
Cohesion: 0.67
Nodes (4): Proposed FER — Active Factor Exposure per Unit Tracking Error (Eq. 3), Rationale: Sector Neutrality Reduces Unintended Active Risk More Than It Costs in Factor Exposure, Sector-Neutral Portfolio Construction, Tracking Error (Active Risk)

## Knowledge Gaps
- **18 isolated node(s):** `Sector-Agnostic Portfolio Construction`, `FER by Hunstad and Dekhayser (2014) — Risk Decomposition Variant`, `Simplified FER — Intended vs Unintended Factor Exposure Ratio`, `Float-Adjusted Market-Cap Weighting Scheme`, `Factor Decay (Diminishing Factor Exposure Between Rebalances)` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Top-Down Multi-Factor Portfolio Construction` connect `Top-Down Factor Selection` to `Turnover & Rebalancing Control`, `Bottom-Up Construction & Methodology`?**
  _High betweenness centrality (0.373) - this node is a cross-community bridge._
- **Why does `Exploring Techniques in Multi-Factor Index Construction` connect `Bottom-Up Construction & Methodology` to `Top-Down Factor Selection`, `Factor Efficiency Ratio Framework`, `Sector Neutrality & Risk`?**
  _High betweenness centrality (0.347) - this node is a cross-community bridge._
- **Why does `Portfolio Turnover` connect `Turnover & Rebalancing Control` to `Top-Down Factor Selection`, `Bottom-Up Construction & Methodology`?**
  _High betweenness centrality (0.272) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Sector-Neutral Portfolio Construction` (e.g. with `Factor Imbalance Metric` and `Proposed FER — Active Factor Exposure per Unit Tracking Error (Eq. 3)`) actually correct?**
  _`Sector-Neutral Portfolio Construction` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Sector-Agnostic Portfolio Construction`, `FER by Hunstad and Dekhayser (2014) — Risk Decomposition Variant`, `Simplified FER — Intended vs Unintended Factor Exposure Ratio` to the rest of the system?**
  _18 weakly-connected nodes found - possible documentation gaps or missing edges._