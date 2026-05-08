# Entrypoint Policy

This repository has one authoritative entrypoint per mandate.

## Authoritative paths

| Mandate | Source of truth |
| --- | --- |
| Research model selection | `run_model_selection.py` |
| Historical portfolio simulation | `run_backtest.py` |
| Daily paper/live orchestration | `run_daily_pipeline.py` |
| QuantConnect deployment | `LeanCloud/BinaryEdge/` |

## Compatibility facades

These files exist only to avoid breaking older local tooling:

- `main.py`
- `qc_main.py`
- `qc_alpha_model.py`

They should not accumulate new business logic. Any new research, live, or QC
behavior belongs in the authoritative path for that mandate.

## Design rule

When adding new workflow code:

1. Decide the mandate first: research, historical simulation, live orchestration, or QC deployment.
2. Add logic to the source-of-truth module for that mandate.
3. Keep compatibility files as thin re-export or delegation layers.
4. If a path needs shared logic, extract a package module and import it from both sides.
