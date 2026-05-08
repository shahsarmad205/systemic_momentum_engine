import pandas as pd

from backtesting.analytics import walk_forward_splits


def test_walk_forward_splits_use_calendar_windows_with_embargo():
    splits = walk_forward_splits(
        "2008-01-01",
        "2022-12-31",
        embargo_days=21,
        train_years=7,
        test_years=1,
        step_years=1,
    )

    assert len(splits) == 8

    first = splits[0]
    last = splits[-1]

    assert first["train_start"] == "2008-01-01"
    assert first["train_end"] == "2014-12-11"
    assert first["test_start"] == "2015-01-01"
    assert first["test_end"] == "2016-01-01"
    assert first["embargo_days"] == 21

    assert last["train_start"] == "2015-01-01"
    assert last["test_start"] == "2022-01-01"
    assert last["test_end"] == "2022-12-31"


def test_walk_forward_splits_fallback_to_legacy_block_mode():
    splits = walk_forward_splits(
        "2020-01-01",
        "2020-12-31",
        n_windows=2,
        train_ratio=0.5,
        embargo_days=10,
    )

    assert len(splits) == 2
    assert splits[0]["train_start"] == "2020-01-01"
    assert pd.Timestamp(splits[0]["test_start"]) > pd.Timestamp(splits[0]["train_end"])
