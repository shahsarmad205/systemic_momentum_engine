## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"` to keep the graph current

## Runtime Config

### Pipeline diagnostics removed
P30-P34 (SignalDecay, ICDiagnostics, ConditionalAlpha, PITConditions, FeatureDiversity) have been removed from the institutional pipeline — they were exploratory diagnostics, not promotion gates. Per-candidate research diagnostics (`research_diagnostics`, `empirical_baselines`) remain available via config but default OFF.

### Factor mimicking horizon control
Limit precomputed horizons for factor mimicking returns:
```yaml
model_selection:
  search:
    factor_precompute_horizons: [20]  # only precompute for h=20
```
Default (null/absent): precompute for all candidate horizons.
