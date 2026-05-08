from __future__ import annotations

import pandas as pd

from features.wrds_fundamental_builder import _compute_features_from_quarterly


def test_compustat_deterioration_features_are_point_in_time_and_nonzero() -> None:
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="B")
    quarterly = pd.DataFrame(
        {
            "datadate": pd.to_datetime(["2019-12-31", "2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"]),
            "avail_date": pd.to_datetime(["2020-02-15", "2020-05-20", "2020-08-25", "2020-12-15", "2021-03-20"]),
            "atq": [100.0, 105.0, 115.0, 120.0, 125.0],
            "niq": [6.0, 4.0, 1.0, -2.0, -4.0],
            "ocf_q": [5.0, 3.0, -1.0, -4.0, -6.0],
            "dlcq": [5.0, 6.0, 8.0, 12.0, 15.0],
            "dlttq": [20.0, 24.0, 32.0, 40.0, 50.0],
            "saleq": [80.0, 78.0, 75.0, 70.0, 65.0],
            "cogsq": [45.0, 48.0, 51.0, 52.0, 53.0],
            "xsgaq": [10.0, 11.0, 13.0, 15.0, 17.0],
            "cshoq": [10.0, 10.2, 10.8, 12.0, 13.5],
            "actq": [50.0, 49.0, 45.0, 42.0, 40.0],
            "lctq": [20.0, 22.0, 25.0, 29.0, 34.0],
        }
    )

    out = _compute_features_from_quarterly(quarterly, dates)
    post_first_rdq = out.loc[pd.Timestamp("2020-02-18")]
    post_deterioration = out.loc[pd.Timestamp("2021-03-22")]

    assert post_first_rdq["gross_margin"] != 0.0
    assert post_deterioration["margin_deterioration"] > 0.0
    assert post_deterioration["dilution_pressure"] > 0.0
    assert post_deterioration["fundamental_deterioration_score"] > 0.5
    assert {"short_interest_ratio", "days_to_cover", "borrow_crowding_risk"}.issubset(out.columns)
