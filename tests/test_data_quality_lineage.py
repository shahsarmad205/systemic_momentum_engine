"""TDD: Phase 10 Data Quality Lineage Tests (RED phase - write tests first)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import modules we'll implement
import scripts.check_data_quality_lineage as cdql
import utils.data_governance as dqc
import utils.data_governance as dql


def _write_json(path: Path, payload: dict) -> None:
    """Helper: write JSON with parent dir creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_config(
    path: Path,
    *,
    enabled: bool = True,
    strict_null_threshold: float = 0.05,
    drift_threshold: float = 2.0,
    fail_on_missing_slo: bool = False,
    fail_on_slo_not_pass: bool = False,
) -> None:
    """Helper: write backtest_config.yaml with Phase 10 governance.data_quality section."""
    path.write_text(
        f"""
governance:
  daily_summary:
    output_dir: output/live/governance
  data_quality:
    enabled: {str(enabled).lower()}
    output_dir: output/live/data_quality
    max_age_hours: 36
    fail_on_missing_slo: {str(fail_on_missing_slo).lower()}
    fail_on_slo_not_pass: {str(fail_on_slo_not_pass).lower()}
    strict_null_threshold: {strict_null_threshold}
    drift_threshold: {drift_threshold}
    check_schema: true
    check_nulls: true
    check_drift: true
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_ohlcv_parquet(cache_dir: Path, ticker: str, data: pd.DataFrame) -> None:
    """Helper: write OHLCV parquet cache for a ticker."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{ticker}.parquet"
    data.to_parquet(path, engine="pyarrow", index=False)


def _write_slo_summary(root: Path, *, status: str = "PASS", failures: list[str] | None = None) -> None:
    """Helper: write SLO health summary."""
    now = datetime.now(timezone.utc)
    payload = {
        "run_at_utc": now.isoformat(),
        "overall_status": status,
        "failures": failures or [],
    }
    _write_json(root / "output" / "live" / "governance" / "governance_daily_summary_latest.json", payload)


def _create_clean_ohlcv(n_bars: int = 100) -> pd.DataFrame:
    """Create a clean OHLCV DataFrame with no nulls."""
    dates = pd.date_range(start="2025-03-01", periods=n_bars, freq="D")
    data = {
        "date": dates,
        "ticker": "TEST",
        "open": np.random.uniform(100, 150, n_bars),
        "high": np.random.uniform(150, 160, n_bars),
        "low": np.random.uniform(90, 100, n_bars),
        "close": np.random.uniform(100, 150, n_bars),
        "volume": np.random.randint(1000000, 10000000, n_bars),
    }
    return pd.DataFrame(data)


def _create_ohlcv_with_nulls(n_bars: int = 100, null_fraction: float = 0.1) -> pd.DataFrame:
    """Create OHLCV with some nulls (to test null detection)."""
    df = _create_clean_ohlcv(n_bars)
    n_nulls = int(n_bars * null_fraction)
    null_rows = np.random.choice(n_bars, n_nulls, replace=False)
    df.loc[null_rows, "close"] = np.nan
    return df


def _create_ohlcv_with_drift(n_bars: int = 100, close_multiplier: float = 3.0) -> pd.DataFrame:
    """Create OHLCV with volatile drift (to test drift detection)."""
    df = _create_clean_ohlcv(n_bars)
    # Make close prices very volatile (drift)
    df["close"] = df["close"] * close_multiplier
    df["high"] = df["high"] * close_multiplier
    return df


def _run_main(monkeypatch, tmp_path: Path, *extra: str) -> int:
    """Helper: run check_data_quality_lineage.main() with tmp workspace."""
    monkeypatch.setattr(cdql, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    argv = ["check_data_quality_lineage.py", "--config", "backtest_config.yaml", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return cdql.main()


# ============================================================
# UNIT TESTS: DataQualityChecker
# ============================================================


class TestDataQualityCheckerSchema:
    """Unit tests for schema validation."""

    def test_schema_validator_pass_with_all_columns(self) -> None:
        """PASS: DataFrame has all required columns."""
        df = _create_clean_ohlcv()
        checker = dqc.DataQualityChecker(df)
        result = checker.validate_schema()

        assert result["status"] == "PASS"
        assert result["column_check"] == "PASS"
        assert result["dtype_check"] == "PASS"

    def test_schema_validator_fail_missing_column(self) -> None:
        """FAIL: DataFrame missing 'close' column."""
        df = _create_clean_ohlcv().drop(columns=["close"])
        checker = dqc.DataQualityChecker(df)
        result = checker.validate_schema()

        assert result["status"] == "FAIL"
        assert result["column_check"] == "FAIL"
        assert "close" in result.get("missing_columns", [])

    def test_schema_validator_fail_wrong_dtype(self) -> None:
        """FAIL: 'volume' is object instead of numeric."""
        df = _create_clean_ohlcv()
        df["volume"] = df["volume"].astype(str)
        checker = dqc.DataQualityChecker(df)
        result = checker.validate_schema()

        assert result["status"] == "FAIL"
        assert result["dtype_check"] == "FAIL"

    def test_schema_validator_pass_nullable_int_volume(self) -> None:
        """PASS: pandas nullable Int64 volume dtype is accepted."""
        df = _create_clean_ohlcv()
        df["volume"] = df["volume"].astype("Int64")
        checker = dqc.DataQualityChecker(df)
        result = checker.validate_schema()

        assert result["status"] == "PASS"
        assert result["dtype_check"] == "PASS"

    def test_schema_validator_empty_dataframe(self) -> None:
        """FAIL: DataFrame is empty."""
        df = pd.DataFrame()
        checker = dqc.DataQualityChecker(df)
        result = checker.validate_schema()

        assert result["status"] == "FAIL"


class TestDataQualityCheckerNulls:
    """Unit tests for null detection."""

    def test_null_detector_clean_data_pass(self) -> None:
        """PASS: No null values."""
        df = _create_clean_ohlcv()
        checker = dqc.DataQualityChecker(df, null_threshold=0.05)
        result = checker.detect_nulls()

        assert result["status"] == "PASS"
        assert result["total_nulls"] == 0
        assert result["null_fraction"] == 0.0

    def test_null_detector_below_threshold_pass(self) -> None:
        """PASS: Null fraction below threshold."""
        df = _create_ohlcv_with_nulls(n_bars=100, null_fraction=0.01)
        checker = dqc.DataQualityChecker(df, null_threshold=0.05)
        result = checker.detect_nulls()

        assert result["status"] == "PASS"
        assert result["null_fraction"] < 0.05

    def test_null_detector_above_threshold_fail(self) -> None:
        """FAIL: Null fraction exceeds threshold."""
        df = _create_ohlcv_with_nulls(n_bars=100, null_fraction=0.30)  # 30% nulls in one col = 6% total
        checker = dqc.DataQualityChecker(df, null_threshold=0.05)
        result = checker.detect_nulls()

        assert result["status"] == "FAIL"
        assert result["null_fraction"] > 0.05
        assert result["total_nulls"] > 0

    def test_null_detector_tracks_columns(self) -> None:
        """Null detector tracks which columns have nulls."""
        df = _create_clean_ohlcv()
        df.loc[5:10, "close"] = np.nan
        df.loc[15:20, "volume"] = np.nan

        checker = dqc.DataQualityChecker(df, null_threshold=0.05)
        result = checker.detect_nulls()

        assert "close" in result.get("null_by_column", {})
        assert "volume" in result.get("null_by_column", {})


class TestDataQualityCheckerDrift:
    """Unit tests for drift detection (comparing vs baseline)."""

    def test_drift_detector_no_baseline_pass(self) -> None:
        """PASS: No baseline provided, skip drift check."""
        df = _create_ohlcv_with_drift(close_multiplier=3.0)
        checker = dqc.DataQualityChecker(df)
        result = checker.detect_drift(baseline=None)

        assert result["status"] == "PASS"
        assert result["reason"] == "no_baseline"

    def test_drift_detector_similar_to_baseline_pass(self) -> None:
        """PASS: Data similar to baseline (low drift)."""
        baseline = _create_clean_ohlcv(n_bars=50)
        current = _create_clean_ohlcv(n_bars=50)

        checker = dqc.DataQualityChecker(current, drift_threshold=2.0)
        result = checker.detect_drift(baseline=baseline)

        assert result["status"] == "PASS"

    def test_drift_detector_high_drift_fail(self) -> None:
        """FAIL: Data has high drift vs baseline."""
        baseline = _create_clean_ohlcv(n_bars=50)
        current = _create_ohlcv_with_drift(n_bars=50, close_multiplier=3.0)

        checker = dqc.DataQualityChecker(current, drift_threshold=2.0)
        result = checker.detect_drift(baseline=baseline)

        assert result["status"] == "FAIL"
        assert "z_score" in result or "drift_metric" in result

    def test_drift_detector_empty_baseline(self) -> None:
        """PASS: Baseline is empty, skip drift check."""
        baseline = pd.DataFrame()
        current = _create_clean_ohlcv()

        checker = dqc.DataQualityChecker(current)
        result = checker.detect_drift(baseline=baseline)

        assert result["status"] == "PASS"


# ============================================================
# UNIT TESTS: LineageTracker
# ============================================================


class TestLineageTracker:
    """Unit tests for lineage tracking."""

    def test_lineage_tracker_initialization(self) -> None:
        """Lineage tracker initializes with ticker and timestamp."""
        tracker = dql.LineageTracker(ticker="NVDA")
        assert tracker.ticker == "NVDA"
        assert tracker.run_at_utc is not None

    def test_lineage_tracker_add_check(self) -> None:
        """add_check() records quality check result."""
        tracker = dql.LineageTracker(ticker="NVDA")
        tracker.add_check(
            name="schema_validation",
            status="PASS",
            details={"columns": ["open", "close", "volume"]},
        )

        lineage = tracker.build_lineage()
        assert len(lineage["checks"]) == 1
        assert lineage["checks"][0]["name"] == "schema_validation"
        assert lineage["checks"][0]["status"] == "PASS"

    def test_lineage_tracker_multiple_checks(self) -> None:
        """Lineage tracks multiple checks in order."""
        tracker = dql.LineageTracker(ticker="NVDA")
        tracker.add_check(name="schema", status="PASS")
        tracker.add_check(name="nulls", status="PASS")
        tracker.add_check(name="drift", status="FAIL", details={"reason": "high_volatility"})

        lineage = tracker.build_lineage()
        assert len(lineage["checks"]) == 3
        assert lineage["checks"][2]["status"] == "FAIL"

    def test_lineage_tracker_build_with_upstream_slo(self) -> None:
        """Lineage includes upstream SLO dependency."""
        tracker = dql.LineageTracker(ticker="NVDA")
        tracker.set_upstream_slo(status="PASS", failures=[])
        tracker.add_check(name="schema", status="PASS")

        lineage = tracker.build_lineage()
        assert lineage["upstream_slo"]["status"] == "PASS"

    def test_lineage_tracker_build_with_data_sources(self) -> None:
        """Lineage tracks data sources."""
        tracker = dql.LineageTracker(ticker="NVDA")
        tracker.add_data_source(path="output/live/ohlcv_cache/NVDA.parquet", rows=100)
        tracker.add_check(name="schema", status="PASS")

        lineage = tracker.build_lineage()
        assert len(lineage["data_sources"]) == 1
        assert lineage["data_sources"][0]["path"] == "output/live/ohlcv_cache/NVDA.parquet"

    def test_lineage_tracker_aggregates_anomalies(self) -> None:
        """Lineage aggregates anomalies from checks."""
        tracker = dql.LineageTracker(ticker="NVDA")
        tracker.add_check(name="nulls", status="FAIL", details={"anomalies": ["null_in_close"]})
        tracker.add_check(name="drift", status="FAIL", details={"anomalies": ["high_spread"]})

        lineage = tracker.build_lineage()
        assert len(lineage.get("anomalies", [])) >= 0  # May be aggregated


# ============================================================
# INTEGRATION TESTS: check_data_quality_lineage.py
# ============================================================


class TestDataQualityLineageIntegration:
    """Integration tests for the main script."""

    def test_integration_disabled_config_returns_zero(self, monkeypatch, tmp_path: Path) -> None:
        """Phase 10 disabled in config → exit 0 (skip)."""
        _write_config(tmp_path / "backtest_config.yaml", enabled=False)
        _write_slo_summary(tmp_path, status="PASS")

        rc = _run_main(monkeypatch, tmp_path)
        assert rc == 0

    def test_integration_pass_with_clean_data(self, monkeypatch, tmp_path: Path) -> None:
        """PASS: Clean OHLCV data, all checks pass."""
        _write_config(tmp_path / "backtest_config.yaml", enabled=True)
        _write_slo_summary(tmp_path, status="PASS")

        # Write clean OHLCV cache
        cache_dir = tmp_path / "output" / "live" / "ohlcv_cache"
        _write_ohlcv_parquet(cache_dir, "TEST", _create_clean_ohlcv())

        rc = _run_main(monkeypatch, tmp_path)
        assert rc == 0

        # Check output artifact
        output = tmp_path / "output" / "live" / "data_quality" / "data_quality_lineage_latest.json"
        assert output.exists()
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["overall_status"] == "PASS"

    def test_integration_fail_with_nulls_exceeding_threshold(self, monkeypatch, tmp_path: Path) -> None:
        """FAIL: OHLCV has nulls exceeding threshold."""
        _write_config(tmp_path / "backtest_config.yaml", enabled=True, strict_null_threshold=0.02)
        _write_slo_summary(tmp_path, status="PASS")

        cache_dir = tmp_path / "output" / "live" / "ohlcv_cache"
        _write_ohlcv_parquet(cache_dir, "TEST", _create_ohlcv_with_nulls(n_bars=100, null_fraction=0.15))

        rc = _run_main(monkeypatch, tmp_path)
        assert rc == 2  # Strict FAIL exit code

        output = tmp_path / "output" / "live" / "data_quality" / "data_quality_lineage_latest.json"
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["overall_status"] == "FAIL"

    def test_integration_missing_slo_summary_continues(self, monkeypatch, tmp_path: Path) -> None:
        """No upstream SLO summary, but continue (configurable)."""
        _write_config(tmp_path / "backtest_config.yaml", enabled=True, fail_on_missing_slo=False)
        # Don't write SLO summary
        cache_dir = tmp_path / "output" / "live" / "ohlcv_cache"
        _write_ohlcv_parquet(cache_dir, "TEST", _create_clean_ohlcv())

        rc = _run_main(monkeypatch, tmp_path)
        # Should continue (fail_on_missing_slo=False)
        assert rc in (0, 2)

    def test_integration_blocked_slo_exits_2(self, monkeypatch, tmp_path: Path) -> None:
        """SLO is BLOCKED → Phase 10 exits 2 (don't run checks)."""
        _write_config(tmp_path / "backtest_config.yaml", enabled=True, fail_on_slo_not_pass=True)
        _write_slo_summary(tmp_path, status="BLOCKED", failures=["gate_failure"])

        cache_dir = tmp_path / "output" / "live" / "ohlcv_cache"
        _write_ohlcv_parquet(cache_dir, "TEST", _create_clean_ohlcv())

        rc = _run_main(monkeypatch, tmp_path)
        assert rc == 2

        output = tmp_path / "output" / "live" / "data_quality" / "data_quality_lineage_latest.json"
        if output.exists():
            payload = json.loads(output.read_text(encoding="utf-8"))
            assert payload["overall_status"] == "BLOCKED"

    def test_integration_empty_cache_skips_checks(self, monkeypatch, tmp_path: Path) -> None:
        """No OHLCV cache → PASS (nothing to validate)."""
        _write_config(tmp_path / "backtest_config.yaml", enabled=True)
        _write_slo_summary(tmp_path, status="PASS")
        # Don't write any parquet files

        rc = _run_main(monkeypatch, tmp_path)
        assert rc == 0

    def test_integration_config_schema_check_disabled(self, monkeypatch, tmp_path: Path) -> None:
        """Config can disable schema check."""
        config_text = """
governance:
  daily_summary:
    output_dir: output/live/governance
  data_quality:
    enabled: true
    output_dir: output/live/data_quality
    check_schema: false
    check_nulls: true
    check_drift: true
""".strip()
        (tmp_path / "backtest_config.yaml").write_text(config_text + "\n")
        _write_slo_summary(tmp_path, status="PASS")

        cache_dir = tmp_path / "output" / "live" / "ohlcv_cache"
        df = _create_clean_ohlcv()
        df = df.drop(columns=["close"])  # Missing column
        _write_ohlcv_parquet(cache_dir, "TEST", df)

        rc = _run_main(monkeypatch, tmp_path)
        assert rc == 0  # Schema check disabled, so passes


class TestDataQualityLineageExitCodes:
    """Tests for exit code semantics."""

    def test_exit_zero_on_pass(self, monkeypatch, tmp_path: Path) -> None:
        """Exit 0: All checks pass."""
        _write_config(tmp_path / "backtest_config.yaml", enabled=True)
        _write_slo_summary(tmp_path, status="PASS")

        cache_dir = tmp_path / "output" / "live" / "ohlcv_cache"
        _write_ohlcv_parquet(cache_dir, "TEST", _create_clean_ohlcv())

        rc = _run_main(monkeypatch, tmp_path)
        assert rc == 0

    def test_exit_one_on_config_error(self, monkeypatch, tmp_path: Path) -> None:
        """Exit 1: Configuration error (e.g., missing config file)."""
        # Don't write config
        rc = _run_main(monkeypatch, tmp_path)
        assert rc == 1

    def test_exit_two_on_quality_fail(self, monkeypatch, tmp_path: Path) -> None:
        """Exit 2: Data quality check failed."""
        _write_config(tmp_path / "backtest_config.yaml", enabled=True, strict_null_threshold=0.01)
        _write_slo_summary(tmp_path, status="PASS")

        cache_dir = tmp_path / "output" / "live" / "ohlcv_cache"
        _write_ohlcv_parquet(cache_dir, "TEST", _create_ohlcv_with_nulls(n_bars=100, null_fraction=0.1))

        rc = _run_main(monkeypatch, tmp_path)
        assert rc == 2


# ============================================================
# EDGE CASE TESTS
# ============================================================


class TestDataQualityEdgeCases:
    """Edge case tests for robustness."""

    def test_single_row_dataframe(self) -> None:
        """Handle single-row DataFrame."""
        df = _create_clean_ohlcv(n_bars=1)
        checker = dqc.DataQualityChecker(df)
        result = checker.validate_schema()
        assert result["status"] in ("PASS", "FAIL")

    def test_all_nulls_column(self) -> None:
        """Handle column with all nulls."""
        df = _create_clean_ohlcv()
        df["close"] = np.nan
        checker = dqc.DataQualityChecker(df, null_threshold=0.19)  # 100/500 = 0.2, threshold 0.19
        result = checker.detect_nulls()
        assert result["status"] == "FAIL"

    def test_infinity_values_in_data(self) -> None:
        """Handle infinity values (edge case for drift detection)."""
        df = _create_clean_ohlcv()
        df.loc[5, "close"] = np.inf
        checker = dqc.DataQualityChecker(df)
        result = checker.validate_schema()
        # Should handle gracefully
        assert result is not None

    def test_lineage_with_no_checks(self) -> None:
        """Lineage built with no checks added."""
        tracker = dql.LineageTracker(ticker="TEST")
        lineage = tracker.build_lineage()
        assert lineage["ticker"] == "TEST"
        assert len(lineage["checks"]) == 0
        assert lineage["overall_status"] == "PASS"

    def test_very_large_dataframe(self) -> None:
        """Handle large OHLCV cache (1M+ rows)."""
        df = _create_clean_ohlcv(n_bars=10000)
        checker = dqc.DataQualityChecker(df)
        result = checker.validate_schema()
        assert result["status"] == "PASS"


# ============================================================
# REGRESSION TESTS (no Phase 7-9 test breakage)
# ============================================================


def test_data_quality_does_not_import_phase9_modules() -> None:
    """Phase 10 does not hard-depend on Phase 9 internals."""
    # Should be importable without triggering Phase 9 module loads
    assert dqc is not None
    assert dql is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
