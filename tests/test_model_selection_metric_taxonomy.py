import numpy as np

from run_model_selection import _compute_deflated_sharpe, _compute_psr


def test_deflated_sharpe_penalizes_multiple_selection_trials() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(loc=0.001, scale=0.01, size=500)

    psr = _compute_psr(returns)
    dsr_few_trials = _compute_deflated_sharpe(returns, n_trials=1)
    dsr_many_trials = _compute_deflated_sharpe(returns, n_trials=100)

    assert 0.0 <= dsr_many_trials <= dsr_few_trials <= 1.0
    assert dsr_few_trials <= psr + 1e-9
