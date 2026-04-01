from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from run_daily_signals import (
    _apply_split_short_gate,
    _asymmetric_short_threshold_multiplier,
    _dedupe_split_ranked,
    _select_split_books,
    run_daily_signals,
)


def test_asymmetric_short_multiplier_defaults() -> None:
    assert _asymmetric_short_threshold_multiplier("Bull", {}) == 1.5
    assert _asymmetric_short_threshold_multiplier("Sideways", {}) == 1.2
    assert _asymmetric_short_threshold_multiplier("Bear", {}) == 1.0
    assert _asymmetric_short_threshold_multiplier("Crisis", {}) == 2.0


def test_asymmetric_short_multiplier_custom_map() -> None:
    cfg = {"short_threshold_multiplier_by_regime": {"Bull": 2.5, "Crisis": 3.0}}
    assert _asymmetric_short_threshold_multiplier("Bull", cfg) == 2.5
    assert _asymmetric_short_threshold_multiplier("Crisis", cfg) == 3.0
    # fallback to default when not provided
    assert _asymmetric_short_threshold_multiplier("Bear", cfg) == 1.0


def test_apply_split_short_gate_filters_by_regime_multiplier() -> None:
    short_df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "score": [-0.09, -0.15, -0.21],
            "book": ["short", "short", "short"],
        }
    )
    # threshold 0.1 in Bull => gate = -0.15
    out = _apply_split_short_gate(short_df, threshold=0.1, regime="Bull", split_cfg={})
    assert out["ticker"].tolist() == ["B", "C"]


def test_select_split_books_respects_side_caps() -> None:
    long_df = pd.DataFrame(
        {
            "ticker": ["L1", "L2", "L3"],
            "score": [0.30, 0.20, 0.10],
            "book": ["long", "long", "long"],
        }
    )
    short_df = pd.DataFrame(
        {
            "ticker": ["S1", "S2", "S3"],
            "score": [-0.30, -0.20, -0.10],
            "book": ["short", "short", "short"],
        }
    )

    selected = _select_split_books(
        long_df,
        short_df,
        max_longs=2,
        max_shorts=1,
        max_positions=3,
    )
    assert len(selected) == 3
    assert set(selected["ticker"]) == {"L1", "L2", "S1"}


def test_dedupe_split_ranked_keeps_stronger_side_per_ticker() -> None:
    ranked = pd.DataFrame(
        {
            "ticker": ["A", "A", "B"],
            "score": [0.7, -0.6, -0.4],
            "book": ["long", "short", "short"],
            "signal": [1, -1, -1],
        }
    )
    out = _dedupe_split_ranked(ranked)
    assert len(out) == 2
    a = out[out["ticker"] == "A"]
    assert len(a) == 1
    assert float(a.iloc[0]["score"]) == 0.7


def _write_config(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _mock_features(as_of_date: str) -> pd.DataFrame:
    d = pd.Timestamp(as_of_date)
    return pd.DataFrame(
        {
            "date": [d, d, d],
            "ticker": ["AAA", "BBB", "CCC"],
            "f_trend": [0.1, 0.2, -0.1],
            "ret_5d": [0.01, -0.01, 0.02],
            "rolling_vol_20": [0.2, 0.3, 0.4],
            "forward_return": [0.01, -0.01, 0.0],
        }
    )


def test_run_daily_signals_split_disabled_ml_compatibility(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "tickers": ["AAA", "BBB", "CCC"],
        "backtest": {
            "max_positions": 3,
            "max_longs": 2,
            "max_shorts": 1,
            "signal_confidence_multiplier": 0.0,
        },
        "execution": {"enable_shorts": True, "long_only": False},
        "signals": {
            "mode": "ml",
            "ml_model_path": "output/models/best_model.pkl",
            "ml_model_type": "classifier",
            "ml_clip": False,
            "split_models": {"enabled": False},
        },
    }
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path / "backtest_config.yaml", cfg)

    import run_daily_signals as rds

    monkeypatch.setattr("agents.weight_learning_agent.feature_builder.build_feature_matrix", lambda *a, **k: _mock_features("2026-03-01"))
    monkeypatch.setattr(rds, "_detect_regime", lambda *_a, **_k: "Bull")
    monkeypatch.setattr(rds, "load_ensemble_models", lambda *_a, **_k: [object()])
    monkeypatch.setattr(
        rds,
        "compute_ensemble_score",
        lambda *_a, **_k: pd.Series([0.5, -0.4, 0.2], index=["AAA", "BBB", "CCC"]),
    )

    selected = run_daily_signals("2026-03-01")
    assert not selected.empty
    assert selected["ticker"].is_unique
    assert (tmp_path / "output" / "signals" / "2026-03-01_rankings.csv").exists()
    assert (tmp_path / "output" / "portfolio" / "paper_positions.csv").exists()


def test_run_daily_signals_split_enabled_dedupes_conflicts(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "tickers": ["AAA", "BBB", "CCC"],
        "backtest": {
            "max_positions": 3,
            "max_longs": 2,
            "max_shorts": 2,
            "signal_confidence_multiplier": 0.0,
        },
        "execution": {"enable_shorts": True, "long_only": False},
        "signals": {
            "mode": "ml",
            "ml_clip": False,
            "split_models": {
                "enabled": True,
                "long_model_path": "output/models/best_model_long.pkl",
                "short_model_path": "output/models/best_model_short.pkl",
                "long_model_type": "classifier",
                "short_model_type": "short_classifier",
                "short_threshold_multiplier_by_regime": {"Bull": 1.0},
            },
        },
    }
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path / "backtest_config.yaml", cfg)

    import run_daily_signals as rds

    monkeypatch.setattr("agents.weight_learning_agent.feature_builder.build_feature_matrix", lambda *a, **k: _mock_features("2026-03-02"))
    monkeypatch.setattr(rds, "_detect_regime", lambda *_a, **_k: "Bull")

    def _score_stub(_feat_df, *, model_path: str, model_type: str, clip: bool):
        del model_type, clip
        if "long" in model_path:
            return pd.Series([0.70, 0.20, 0.05], index=["AAA", "BBB", "CCC"])
        return pd.Series([-0.60, -0.10, -0.40], index=["AAA", "BBB", "CCC"])

    monkeypatch.setattr(rds, "_score_single_ml_model", _score_stub)

    selected = run_daily_signals("2026-03-02")
    assert not selected.empty
    assert selected["ticker"].is_unique
    aa = selected[selected["ticker"] == "AAA"]
    assert len(aa) == 1
    assert float(aa.iloc[0]["score"]) > 0
    assert (tmp_path / "output" / "signals" / "2026-03-02_long_candidates.csv").exists()
    assert (tmp_path / "output" / "signals" / "2026-03-02_short_candidates.csv").exists()
