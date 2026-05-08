# Phase A.2 Dead-Code Quarantine

Date: 2026-05-07

This directory holds private research helpers removed from production modules
after static reference checks and runtime tracing showed no active production
call path. They are kept here as non-production reference material only.

Do not import these helpers from production packages. If a future research path
needs one, reintroduce it through a reviewed module with tests and an explicit
contract.

## Quarantined Helpers

- `run_model_selection._chained_oos_metrics`
  - Original location: `run_model_selection.py`
  - Reason: private legacy OOS metric helper, definition-only in static search,
    absent from representative runtime traces, medium risk because it is
    research/report adjacent.
  - Replacement: current evaluation path computes chained metrics inline in the
    active model-evaluation/report code.

- `model_selection.alpha_research._regime_stability`
  - Original location: `model_selection/alpha_research.py`
  - Reason: private legacy regime stability helper, definition-only in static
    search, absent from alpha-research runtime trace, medium risk because it is
    research-adjacent.
  - Replacement: active alpha admission uses `regime_positive_rate` fields
    produced by the IC-decay matrix path.
