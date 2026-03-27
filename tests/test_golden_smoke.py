from __future__ import annotations

from pathlib import Path

import yaml


def test_backtest_config_yaml_loads():
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "backtest_config.yaml"
    assert cfg_path.is_file()
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    assert "tickers" in cfg
    assert len(cfg["tickers"]) > 0
    assert cfg.get("retraining") is None or isinstance(cfg.get("retraining"), dict)
