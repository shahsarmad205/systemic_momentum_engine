from __future__ import annotations

import sys
import types

import pandas as pd

from utils.universe import _load_wrds_universe, load_universe


class _Cfg:
    start_date = "2020-01-01"
    end_date = "2020-12-31"


def test_wrds_universe_defaults_to_start_date_liquid_permnos(monkeypatch) -> None:
    fake_mod = types.ModuleType("utils.wrds_universe")

    class FakeUniverse:
        def __init__(self, db) -> None:
            self.db = db

        def get_sp500_panel(self, start_date, end_date) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "permno": 1,
                        "effective_start": pd.Timestamp("2020-01-01"),
                        "effective_end": pd.Timestamp("2020-06-30"),
                    },
                    {
                        "permno": 2,
                        "effective_start": pd.Timestamp("2020-07-01"),
                        "effective_end": pd.Timestamp("2020-12-31"),
                    },
                ]
            )

        def permno_to_ticker_map(self, permnos, date) -> dict[int, str]:
            return {1: "AAA", 2: "BBB"}

    fake_mod.WRDSUniverse = FakeUniverse
    fake_mod.build_backtest_universe = lambda db, date, min_price, min_dollar_vol: [1]
    fake_mod.connect_wrds = lambda username: object()

    monkeypatch.setenv("WRDS_USERNAME", "tester")
    monkeypatch.setitem(sys.modules, "utils.wrds_universe", fake_mod)

    cfg = _Cfg()
    tickers = _load_wrds_universe(cfg, {"min_price": 10.0, "min_dollar_vol": 1e8})

    assert tickers == ["AAA"]
    assert cfg.wrds_ticker_to_permno == {"AAA": 1}
    assert cfg.pit_universe_mode == "wrds"
    assert cfg.pit_membership_ranges["AAA"][0][0] == pd.Timestamp("2020-01-01")


def test_wrds_universe_can_opt_into_full_membership_panel(monkeypatch) -> None:
    fake_mod = types.ModuleType("utils.wrds_universe")

    class FakeUniverse:
        def __init__(self, db) -> None:
            self.db = db

        def get_sp500_panel(self, start_date, end_date) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "permno": 1,
                        "effective_start": pd.Timestamp("2020-01-01"),
                        "effective_end": pd.Timestamp("2020-06-30"),
                    },
                    {
                        "permno": 2,
                        "effective_start": pd.Timestamp("2020-07-01"),
                        "effective_end": pd.Timestamp("2020-12-31"),
                    },
                ]
            )

        def permno_to_ticker_map(self, permnos, date) -> dict[int, str]:
            return {1: "AAA", 2: "BBB"}

    fake_mod.WRDSUniverse = FakeUniverse
    fake_mod.build_backtest_universe = lambda db, date, min_price, min_dollar_vol: [1]
    fake_mod.connect_wrds = lambda username: object()

    monkeypatch.setenv("WRDS_USERNAME", "tester")
    monkeypatch.setitem(sys.modules, "utils.wrds_universe", fake_mod)

    cfg = _Cfg()
    tickers = _load_wrds_universe(
        cfg,
        {
            "min_price": 10.0,
            "min_dollar_vol": 1e8,
            "include_full_membership_panel": True,
        },
    )

    assert tickers == ["AAA", "BBB"]
    assert cfg.wrds_ticker_to_permno == {"AAA": 1, "BBB": 2}
    assert cfg.pit_membership_ranges["BBB"][0][1] == pd.Timestamp("2020-12-31")


def test_wrds_universe_reads_dates_from_backtest_block(monkeypatch) -> None:
    fake_mod = types.ModuleType("utils.wrds_universe")
    seen: dict[str, object] = {}

    class FakeUniverse:
        def __init__(self, db) -> None:
            self.db = db

        def get_sp500_panel(self, start_date, end_date) -> pd.DataFrame:
            seen["panel_start"] = pd.Timestamp(start_date)
            seen["panel_end"] = pd.Timestamp(end_date)
            return pd.DataFrame(
                [
                    {
                        "permno": 11,
                        "effective_start": pd.Timestamp("2018-01-01"),
                        "effective_end": pd.Timestamp("2018-12-31"),
                    }
                ]
            )

        def permno_to_ticker_map(self, permnos, date) -> dict[int, str]:
            seen["map_date"] = pd.Timestamp(date)
            return {11: "AAA"}

    def fake_build_backtest_universe(db, date, min_price, min_dollar_vol):
        seen["liquidity_date"] = pd.Timestamp(date)
        return [11]

    fake_mod.WRDSUniverse = FakeUniverse
    fake_mod.build_backtest_universe = fake_build_backtest_universe
    fake_mod.connect_wrds = lambda username: object()

    monkeypatch.setenv("WRDS_USERNAME", "tester")
    monkeypatch.setitem(sys.modules, "utils.wrds_universe", fake_mod)

    cfg = {
        "backtest": {"start_date": "2018-01-01", "end_date": "2018-12-31"},
        "universe": {"mode": "wrds", "min_price": 10.0, "min_dollar_vol": 1e8},
    }

    tickers = load_universe(cfg)

    assert tickers == ["AAA"]
    assert seen["liquidity_date"] == pd.Timestamp("2018-01-01")
    assert seen["panel_start"] == pd.Timestamp("2018-01-01")
    assert seen["panel_end"] == pd.Timestamp("2018-12-31")
    assert seen["map_date"] == pd.Timestamp("2018-12-31")
    assert cfg["wrds_ticker_to_permno"] == {"AAA": 11}


def test_wrds_universe_fails_closed_without_explicit_research_fallback(monkeypatch) -> None:
    monkeypatch.delenv("WRDS_USERNAME", raising=False)

    cfg = {
        "backtest": {"start_date": "2018-01-01", "end_date": "2018-12-31"},
        "universe": {"mode": "wrds"},
    }

    try:
        load_universe(cfg)
    except RuntimeError as exc:
        assert "WRDS universe mode requested" in str(exc)
    else:
        raise AssertionError("WRDS universe mode should fail closed when WRDS is unavailable")
