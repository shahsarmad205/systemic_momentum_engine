from __future__ import annotations

from pathlib import Path

from strategy.regime_multipliers import (
    SUPPORTED_REGIMES,
    get_multiplier,
    load_regime_multipliers,
)


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_missing_config_file_returns_all_defaults(tmp_path: Path, caplog):
    missing = tmp_path / "missing.yaml"
    out = load_regime_multipliers(str(missing))
    assert out == {r: 1.0 for r in SUPPORTED_REGIMES}
    assert "Config file not found" in caplog.text


def test_missing_regime_key_defaults_to_one(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    _write_yaml(
        cfg,
        """
signals:
  signal_confidence_multiplier_sideways: 0.3
""".strip(),
    )
    out = load_regime_multipliers(str(cfg))
    assert out["Sideways"] == 0.3
    assert out["Bull"] == 1.0
    assert out["Bear"] == 1.0
    assert out["Crisis"] == 1.0


def test_invalid_values_fallback_to_one_with_warning(tmp_path: Path, caplog):
    cfg = tmp_path / "cfg.yaml"
    _write_yaml(
        cfg,
        """
signals:
  signal_confidence_multiplier_bull: -1
  signal_confidence_multiplier_bear: 4.2
  signal_confidence_multiplier_sideways: abc
  signal_confidence_multiplier_crisis: 1.5
""".strip(),
    )
    out = load_regime_multipliers(str(cfg))
    assert out["Bull"] == 1.0
    assert out["Bear"] == 1.0
    assert out["Sideways"] == 1.0
    assert out["Crisis"] == 1.5
    assert "outside (0, 3.0]" in caplog.text or "not numeric" in caplog.text


def test_get_multiplier_case_insensitive_lookup():
    mult = {"Bull": 1.1, "Bear": 0.9, "Sideways": 0.3, "Crisis": 0.8}
    assert get_multiplier("sideways", mult) == 0.3
    assert get_multiplier("BULL", mult) == 1.1
    assert get_multiplier("unknown", mult, default=1.0) == 1.0
