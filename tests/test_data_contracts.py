from __future__ import annotations

import pandas as pd

from utils.data_contracts import cache_covers_session, required_latest_cache_date


def test_cache_covers_session() -> None:
    need = required_latest_cache_date()
    assert isinstance(need, pd.Timestamp)
    assert cache_covers_session(need, need) is True
    assert cache_covers_session(None, need) is False
    assert cache_covers_session(need - pd.Timedelta(days=5), need) is False
