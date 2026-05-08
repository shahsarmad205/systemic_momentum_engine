from __future__ import annotations

import numpy as np
import pandas as pd

from providers.cache_manager import cache_path
from providers.wrds_adapter import WRDSProvider
from utils.wrds_loader import WRDSLoader
from utils.wrds_universe import build_backtest_universe


class FakeWRDSDB:
    """Small deterministic WRDS snapshot for daily integrity tests."""

    def raw_sql(self, sql: str, date_cols=None):  # noqa: ANN001
        lower = sql.lower()
        if "from crsp.dsp500list" in lower:
            return pd.DataFrame({"permno": [10001, 10002, 10003]})
        if "from crsp.dsf" in lower and "abs(prc) as price" in lower:
            return pd.DataFrame(
                {
                    "permno": [10001, 10002, 10003],
                    "price": [20.0, 4.0, 50.0],
                    "dollar_vol": [250_000_000.0, 300_000_000.0, 25_000_000.0],
                }
            )
        if "from crsp.dsedelist" in lower and "distinct permno" in lower:
            return pd.DataFrame({"permno": [10003]})
        if "from crsp.dsf" in lower:
            dates = pd.bdate_range("2020-01-02", periods=22)
            return pd.DataFrame(
                {
                    "permno": [10001] * len(dates),
                    "date": dates,
                    "prc": np.linspace(10.0, 11.0, len(dates)),
                    "ret": [0.01] + [0.001] * (len(dates) - 1),
                    "vol": [1_000_000.0] * len(dates),
                    "shrout": [10_000.0] * len(dates),
                }
            )
        if "from crsp.dsedelist" in lower:
            return pd.DataFrame(
                {
                    "permno": [10001],
                    "dlstdt": pd.to_datetime(["2020-02-04"]),
                    "dlret": [np.nan],
                    "hexcd": [3],
                    "dlstcd": [574],
                }
            )
        raise AssertionError(f"unexpected SQL: {sql[:120]}")


def test_wrds_daily_integrity_universe_filters_delisted_and_illiquid(tmp_path) -> None:
    db = FakeWRDSDB()

    universe = build_backtest_universe(
        db,
        "2020-01-07",
        min_price=10.0,
        min_dollar_vol=100_000_000.0,
        cache_dir=str(tmp_path / "universe"),
        cache_ttl_days=0,
    )

    assert universe == [10001]


def test_wrds_daily_integrity_delisting_return_matches_known_good_snapshot(tmp_path) -> None:
    loader = WRDSLoader(FakeWRDSDB(), cache_dir=str(tmp_path), cache_ttl_days=0)
    panel = loader.load_universe(
        permnos=[10001],
        ticker_map={10001: "AAA"},
        start_date="2020-01-02",
        end_date="2020-02-04",
    )

    aaa = panel["AAA"]
    last = aaa.iloc[-1]
    known_good = {
        "last_date": pd.Timestamp("2020-02-04"),
        "last_ret": -0.55,
        "last_close": float(np.prod([1.01] + [1.001] * 21 + [0.45])),
        "delisting_return_applied": True,
        "dlstcd": 574.0,
    }

    assert aaa.index[-1] == known_good["last_date"]
    assert float(last["ret"]) == known_good["last_ret"]
    assert np.isclose(float(last["Close"]), known_good["last_close"])
    assert bool(last["delisting_return_applied"]) is known_good["delisting_return_applied"]
    assert float(last["dlstcd"]) == known_good["dlstcd"]


def test_wrds_provider_requires_explicit_credentials(monkeypatch) -> None:
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    provider = WRDSProvider(username=None)

    try:
        provider.validate_available()
    except RuntimeError as exc:
        assert "WRDS_USERNAME" in str(exc)
    else:
        raise AssertionError("WRDS provider should fail fast without credentials")


def test_cache_path_provider_namespace_is_explicit() -> None:
    assert cache_path("data/cache", "AAPL", "yahoo").name == "AAPL.parquet"
    assert cache_path("data/cache", "AAPL", "wrds").name == "AAPL_wrds.parquet"
