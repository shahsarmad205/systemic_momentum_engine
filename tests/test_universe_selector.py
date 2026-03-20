import pandas as pd

from UniverseSelector import UniverseSelector


def _make_panel():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")

    def make(ticker, base_vol, nan_idx=None):
        df = pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "Close": 100.0,
                "Volume": base_vol,
            }
        )
        if nan_idx is not None:
            df.loc[nan_idx, "Volume"] = pd.NA
        return df

    # Liquid stock: volume ~ 2M
    aapl = make("AAPL", 2_000_000)
    # Illiquid: volume ~ 100k
    illiq = make("ILLIQ", 100_000)
    # Mixed with NaNs; effective ADV still > 1M after dropping NaNs
    mixed = make("MIXED", 1_500_000, nan_idx=[0, 1])

    panel = pd.concat([aapl, illiq, mixed], ignore_index=True)
    return panel


def test_universe_selector_filters_by_liquidity():
    panel = _make_panel()
    selector = UniverseSelector(min_adv=1_000_000, min_dollar_vol=50_000_000.0, lookback_days=10)
    selected = selector.select(panel)

    # AAPL and MIXED should be included, ILLIQ excluded
    assert "AAPL" in selected
    assert "MIXED" in selected
    assert "ILLIQ" not in selected


def test_universe_selector_handles_missing_volume():
    panel = _make_panel()
    # Force all volume for ILLIQ to NaN; should be dropped entirely
    panel.loc[panel["ticker"] == "ILLIQ", "Volume"] = pd.NA

    selector = UniverseSelector(min_adv=1_000_000, min_dollar_vol=50_000_000.0, lookback_days=10)
    selected = selector.select(panel)

    assert "ILLIQ" not in selected
    # Still keeps the others
    assert "AAPL" in selected
    assert "MIXED" in selected

