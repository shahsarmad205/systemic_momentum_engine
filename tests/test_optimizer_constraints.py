import numpy as np

from backtesting.optimizer import PortfolioOptimizer


def test_optimizer_post_constraints_respect_factor_and_liquidity_caps() -> None:
    optimizer = PortfolioOptimizer(
        max_weight=0.20,
        net_exposure_max=0.50,
        long_only=False,
        gross_cap=1.00,
    )

    tickers = ["AAA", "BBB", "CCC"]
    constrained = optimizer.constrain_weights(
        {"AAA": 0.20, "BBB": 0.20, "CCC": -0.05},
        tickers,
        factor_exposures={
            "market_beta": np.array([1.0, 1.0, 1.0]),
            "sector:Technology": np.array([1.0, 1.0, 0.0]),
        },
        factor_bounds={
            "market_beta": 0.15,
            "sector:Technology": 0.12,
        },
        max_weight_overrides={
            "AAA": 0.10,
            "BBB": 0.20,
            "CCC": 0.20,
        },
    )

    w = np.array([constrained[t] for t in tickers], dtype=float)
    assert abs(constrained["AAA"]) <= 0.10 + 1e-12
    assert abs(w.sum()) <= 0.50 + 1e-12
    assert abs(np.dot(np.array([1.0, 1.0, 1.0]), w)) <= 0.15 + 1e-8
    assert abs(np.dot(np.array([1.0, 1.0, 0.0]), w)) <= 0.12 + 1e-8
