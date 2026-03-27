# Test quarantine backlog (cleared)

Historically, several integration tests were excluded in `pyproject.toml` via `pytest --ignore=` so CI stayed green while APIs drifted.

## Formerly ignored modules

| Module | Root cause (summary) | Resolution |
|--------|----------------------|------------|
 `test_overlay_validation.py` | `BacktestConfig` / portfolio API removed | Tests trimmed to current overlay / vol-kill behaviour. |
| `test_position_sizing_multiplicative.py` | Sizing helpers moved | Aligned with `backtesting.position_sizing` / `compose_position_size`. |
| `test_universe_selector.py` | Missing module | Added `strategy/universe_selector.py`; tests import it. |
| `test_vol_kill_switch.py` | Kill-switch treated non-finite vol as “high” | `apply_vol_kill_switch` requires finite vol above threshold. |
| `test_vol_sizing.py` | Missing vol helpers | Implemented in `utils/vol_sizing.py`. |
| `test_backtester_integration_pipeline.py` | Assertion drift (equity / caps) | Expectations updated for current backtester. |
| `test_position_sizing.py` | Kelly / sizing API drift | Matched `kelly_size` and caps. |
| `test_metrics.py` | Metric definitions drift | Updated to current `backtesting.metrics`. |
| `test_signal_generation.py` | Signal pipeline drift | Fixed imports / mocks / lazy deps. |
| `test_weight_learning.py` | Optional `shap`, import graph | Lazy imports; optional SHAP in `weight_model.py`. |

## Policy

- Do not reintroduce permanent `--ignore=` for tests. Use `@pytest.mark.skip` with an issue link only for documented, time-bounded cases.
