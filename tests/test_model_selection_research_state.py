from __future__ import annotations

import pandas as pd

from model_selection.research_state import ResearchStateStore


def test_research_state_store_reuses_frames_and_persists_universe_panel(tmp_path) -> None:
    cfg = {
        "model_selection": {
            "target": {"residualize": True},
            "alpha_research": {"enabled": True},
            "validation": {"primary_path": "long_short_spread"},
        },
        "universe": {"mode": "wrds"},
        "pit_universe_mode": "wrds",
    }
    store = ResearchStateStore(
        root_dir=tmp_path,
        namespace="model_selection",
        payload={
            "cfg": cfg,
            "tickers": ["AAPL", "MSFT"],
        },
    )

    calls = {"count": 0}

    def _builder() -> pd.DataFrame:
        calls["count"] += 1
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2022-01-03"]),
                "ticker": ["AAPL"],
                "feature": [1.0],
            }
        )

    first, from_cache_first = store.get_or_build_frame("feature_panel_program", _builder)
    second, from_cache_second = store.get_or_build_frame("feature_panel_program", _builder)

    assert calls["count"] == 1
    assert from_cache_first is False
    assert from_cache_second is True
    pd.testing.assert_frame_equal(first, second)

    universe_path = store.persist_universe_panel(
        tickers=["AAPL", "MSFT"],
        membership_ranges=None,
        start_date="2022-01-01",
        end_date="2022-12-31",
    )
    universe = pd.read_parquet(universe_path)
    assert set(universe["ticker"]) == {"AAPL", "MSFT"}
    assert {"effective_start", "effective_end"}.issubset(universe.columns)


def test_alpha_research_state_is_independent_of_validation_knobs(tmp_path) -> None:
    base_cfg = {
        "model_selection": {
            "target": {"residualize": True, "net_of_costs": True},
            "alpha_research": {"enabled": True, "production_horizon": 5},
            "validation": {"primary_path": "long_short_spread", "lambda_risk": 2.0},
        },
        "universe": {"mode": "wrds"},
        "pit_universe_mode": "wrds",
    }
    alt_cfg = {
        **base_cfg,
        "model_selection": {
            **base_cfg["model_selection"],
            "validation": {"primary_path": "long_short_spread", "lambda_risk": 9.0},
        },
    }
    feature_contract = {
        "ret_5d": {
            "family": "momentum",
            "expected_sign": 1,
            "horizon_days": 5,
        }
    }

    first = ResearchStateStore.for_alpha_research(
        cfg=base_cfg,
        tickers=["AAPL", "MSFT"],
        start_date="2020-01-01",
        end_date="2022-12-31",
        provider="wrds",
        feature_columns=["ret_5d"],
        feature_panel_signature="panel_sig_1",
        feature_contract=feature_contract,
    )
    second = ResearchStateStore.for_alpha_research(
        cfg=alt_cfg,
        tickers=["AAPL", "MSFT"],
        start_date="2020-01-01",
        end_date="2022-12-31",
        provider="wrds",
        feature_columns=["ret_5d"],
        feature_panel_signature="panel_sig_1",
        feature_contract=feature_contract,
    )
    changed_panel = ResearchStateStore.for_alpha_research(
        cfg=alt_cfg,
        tickers=["AAPL", "MSFT"],
        start_date="2020-01-01",
        end_date="2022-12-31",
        provider="wrds",
        feature_columns=["ret_5d"],
        feature_panel_signature="panel_sig_2",
        feature_contract=feature_contract,
    )
    changed_schema = ResearchStateStore.for_alpha_research(
        cfg=alt_cfg,
        tickers=["AAPL", "MSFT"],
        start_date="2020-01-01",
        end_date="2022-12-31",
        provider="wrds",
        feature_columns=["ret_5d"],
        feature_panel_signature="panel_sig_1",
        feature_contract=feature_contract,
        alpha_research_schema_version="schema_v2",
    )

    assert first.signature == second.signature
    assert first.signature != changed_panel.signature
    assert first.signature != changed_schema.signature
