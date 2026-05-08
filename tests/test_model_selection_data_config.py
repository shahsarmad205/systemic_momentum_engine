from __future__ import annotations

import run_model_selection as rms
from model_selection.horizon_contract import build_horizon_contract


def test_feature_builder_data_kwargs_uses_wrds_provider_and_namespace(monkeypatch) -> None:
    monkeypatch.setenv("WRDS_USERNAME", "wrds_user")

    cfg = {
        "data": {
            "provider": "wrds",
            "cache_dir": "data/cache",
            "cache_ttl_days": 0,
        },
        "wrds_ticker_to_permno": {"AAPL": 14593},
    }

    kwargs = rms._feature_builder_data_kwargs(cfg)

    assert kwargs["data_provider"] == "wrds"
    assert kwargs["cache_dir"] == "data/cache/wrds"
    assert kwargs["cache_ttl_days"] == 0
    assert kwargs["wrds_username"] == "wrds_user"
    assert kwargs["wrds_ticker_to_permno"] == {"AAPL": 14593}


def test_feature_builder_data_kwargs_preserves_provider_specific_wrds_cache_dir() -> None:
    cfg = {
        "data": {
            "provider": "wrds",
            "cache_dir": "data/cache/wrds",
            "cache_ttl_days": 7,
        }
    }

    kwargs = rms._feature_builder_data_kwargs(cfg)

    assert kwargs["data_provider"] == "wrds"
    assert kwargs["cache_dir"] == "data/cache/wrds"
    assert kwargs["cache_ttl_days"] == 7


def test_cli_horizon_overrides_full_horizon_contract() -> None:
    cfg = {
        "horizon_config": {
            "allow_cross_horizon_evaluation": True,
        },
        "backtest": {
            "rebalance_every_trading_days": 5,
            "holding_period_days": 5,
            "lookahead_horizon_days": 5,
        },
        "model_selection": {
            "lookahead_horizon_days": 5,
            "alpha_research": {"production_horizon": 5},
        },
    }

    contract = build_horizon_contract(cfg, cli_horizon=5)

    assert contract.config.target_horizon_days == 5
    assert contract.config.holding_period_days == 5
    assert contract.config.rebalance_frequency_days == 5
    assert contract.config.ic_evaluation_horizon == 5
    assert set(contract.source_map.values()) >= {"cli.--horizon"}
