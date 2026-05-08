import warnings

import numpy as np
import pandas as pd

from model_selection.validation import (
    DateLevelMarketState,
    EvaluationConfig,
    ExecutionCostConfig,
    PromotionGateConfig,
    ValidationStateCache,
    _neutralize_day_weights,
    _max_sector_exposure,
    _max_sector_exposure_from_aligned_day,
    _next_day_returns,
    _portfolio_beta,
    _portfolio_beta_from_aligned_day,
    _trade_cost_breakdown_from_market_state,
    _vectorized_trade_costs_from_market_state,
    build_target_weights,
    cross_sectional_ic,
    decile_return_diagnostics,
    evaluate_promotion_gates,
    simulate_executable_portfolio,
)


def _sample_scored(days: int = 14, names: int = 8) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=days)
    tickers = [f"T{i}" for i in range(names)]
    rows = []
    for day_idx, dt in enumerate(dates):
        for name_idx, ticker in enumerate(tickers):
            rank_signal = name_idx - (names - 1) / 2.0
            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "score": rank_signal,
                    "forward_return": rank_signal * 0.002,
                    "daily_return": rank_signal * 0.0004 + day_idx * 0.0,
                    "adv_dollar_20": 50_000_000.0,
                    "realised_vol_20d": 0.02,
                    "capm_beta": 1.0 + rank_signal * 0.05,
                    "sector": "Tech" if name_idx % 2 else "Industrials",
                }
            )
    return pd.DataFrame(rows)


def test_cross_sectional_ic_uses_daily_rank_relationship() -> None:
    scored = _sample_scored()
    shock_dates = scored["date"].drop_duplicates().iloc[::3]
    scored.loc[scored["date"].isin(shock_dates) & scored["ticker"].eq("T7"), "forward_return"] = -0.001
    stats = cross_sectional_ic(scored)

    assert stats["cs_ic_n_days"] == scored["date"].nunique()
    assert stats["cs_ic_spearman_mean"] > 0.80
    assert stats["cs_ic_spearman_tstat"] > 10.0
    assert stats["daily_ic_mean"] == stats["cs_ic_spearman_mean"]
    assert stats["daily_ic_std"] > 0.0
    assert stats["daily_ic_annualized_icir"] > stats["cs_ic_spearman_ir"]
    assert stats["daily_ic_hac_tstat"] == stats["cs_ic_spearman_tstat"]


def test_executable_simulator_charges_turnover_and_borrow_costs() -> None:
    scored = _sample_scored()
    cfg = EvaluationConfig(
        max_positions=4,
        min_positions=2,
        path="long_short_spread",
        rebalance_every_days=1,
        factor_neutral=False,
        max_gross=1.0,
        costs=ExecutionCostConfig(commission_bps=2.0, spread_bps=2.0, borrow_bps=100.0),
    )

    returns, pnl = simulate_executable_portfolio(scored, cfg)

    assert len(returns) > 0
    assert pnl["cost_return"].sum() > 0.0
    assert pnl["commission_return"].sum() > 0.0
    assert pnl["spread_return"].sum() > 0.0
    assert pnl["temporary_impact_return"].sum() > 0.0
    assert pnl["permanent_impact_return"].sum() > 0.0
    assert pnl["trade_notional"].sum() > 0.0
    assert pnl["participation_rate_max"].max() > 0.0
    assert pnl["permanent_impact_unamortized_return"].max() > 0.0
    assert pnl["borrow_return"].sum() > 0.0
    assert pnl["daily_return"].sum() < pnl["gross_return"].sum()
    assert {"long_gross_return", "short_gross_return", "long_exposure", "short_exposure"}.issubset(pnl.columns)


def test_array_next_day_returns_match_groupby_shift_contract() -> None:
    scored = _sample_scored(days=6, names=5).sample(frac=1.0, random_state=11)
    expected = (
        scored[["date", "ticker", "daily_return"]]
        .assign(date=lambda x: pd.to_datetime(x["date"], errors="coerce"))
        .sort_values(["ticker", "date"])
    )
    shifted = pd.to_numeric(expected["daily_return"], errors="coerce").groupby(expected["ticker"]).shift(-1)
    baseline = pd.Series(shifted.to_numpy(dtype=float), index=expected.index).reindex(scored.index)

    fast = _next_day_returns(scored, horizon_days=5)

    assert np.allclose(fast.fillna(0.0).to_numpy(), baseline.fillna(0.0).to_numpy())
    assert fast.isna().sum() == baseline.isna().sum()


def test_aligned_exposure_kernels_match_pandas_contract() -> None:
    day = _sample_scored(days=1, names=12)
    weights = pd.Series(
        np.linspace(-0.08, 0.08, len(day)),
        index=day["ticker"].astype(str).to_numpy(dtype=object),
        dtype=float,
    )
    aligned = weights.reindex(day["ticker"].astype(str).to_numpy(dtype=object)).fillna(0.0)

    assert np.isclose(_portfolio_beta_from_aligned_day(day, aligned), _portfolio_beta(day, weights))
    assert np.isclose(
        _max_sector_exposure_from_aligned_day(day, aligned),
        _max_sector_exposure(day, weights),
    )


def test_vectorized_market_state_trade_costs_match_scalar_path() -> None:
    tickers = ("A", "B", "C")
    state = DateLevelMarketState(
        date=pd.Timestamp("2022-01-03"),
        tickers=tickers,
        ticker_to_idx={t: i for i, t in enumerate(tickers)},
        covariance=np.eye(3),
        specific_risk=np.ones(3),
        factor_exposures={},
        adv_dollar=np.array([50_000_000.0, 25_000_000.0, 10_000_000.0]),
        daily_vol=np.array([0.02, 0.03, 0.04]),
        liquidity_caps=np.ones(3),
        participation_scale=np.array([2.0, 4.0, 10.0]),
        max_participation_rate=0.10,
        borrow_penalty_horizon=np.zeros(3),
        crowding_risk=np.zeros(3),
        short_interest_ratio=np.zeros(3),
        squeeze_risk=np.zeros(3),
        short_blocked=np.zeros(3, dtype=bool),
    )
    cfg = EvaluationConfig(
        costs=ExecutionCostConfig(
            capital=100_000_000.0,
            commission_bps=2.0,
            spread_bps=3.0,
            impact_eta=0.10,
            impact_alpha=0.02,
            impact_gamma=0.5,
        )
    )
    trade_weights = np.array([0.01, -0.02, 0.03])
    prev = np.array([0.00, -0.03, -0.01])
    nxt = prev + trade_weights

    vectorized = _vectorized_trade_costs_from_market_state(
        np.asarray(tickers, dtype=object),
        trade_weights,
        prev,
        nxt,
        state,
        cfg,
    )
    scalar = [
        _trade_cost_breakdown_from_market_state(
            float(dw),
            ticker,
            state,
            cfg,
            is_exit=abs(float(nw)) < abs(float(pw)),
        )
        for ticker, dw, pw, nw in zip(tickers, trade_weights, prev, nxt, strict=False)
    ]

    assert vectorized is not None
    assert vectorized["trade_count"] == len(tickers)
    assert np.isclose(vectorized["cost"], sum(x.cost_return for x in scalar))
    assert np.isclose(vectorized["commission"], sum(x.commission_return for x in scalar))
    assert np.isclose(vectorized["spread"], sum(x.spread_return for x in scalar))
    assert np.isclose(vectorized["temporary_impact"], sum(x.temporary_impact_return for x in scalar))
    assert np.isclose(vectorized["permanent_impact"], sum(x.permanent_impact_return for x in scalar))
    assert np.allclose(vectorized["participation_rates"], [x.participation_rate for x in scalar])


def test_factor_neutral_weight_builder_reduces_beta_exposure() -> None:
    scored = _sample_scored(days=1)
    non_neutral = EvaluationConfig(
        max_positions=4,
        min_positions=2,
        path="long_short_spread",
        rebalance_every_days=1,
        factor_neutral=False,
        max_gross=1.0,
        use_optimizer=False,
    )
    neutral = EvaluationConfig(
        max_positions=4,
        min_positions=2,
        path="long_short_spread",
        rebalance_every_days=1,
        factor_neutral=True,
        beta_neutral=True,
        sector_neutral=False,
        max_beta_abs=1e-6,
        max_gross=1.0,
        use_optimizer=False,
    )

    raw_weights = build_target_weights(scored, non_neutral).set_index("ticker")["target_weight"]
    neutral_weights = build_target_weights(scored, neutral).set_index("ticker")["target_weight"]
    beta = scored.set_index("ticker")["capm_beta"]

    raw_beta = float((raw_weights.reindex(beta.index).fillna(0.0) * beta).sum())
    neutral_beta = float((neutral_weights.reindex(beta.index).fillna(0.0) * beta).sum())

    assert abs(neutral_beta) < abs(raw_beta)


def test_factor_neutralization_handles_degenerate_exposures_without_runtime_warning() -> None:
    scored = _sample_scored(days=1)
    scored["sector"] = "Tech"
    scored.loc[scored.index[0], "capm_beta"] = np.inf
    cfg = EvaluationConfig(
        max_positions=4,
        min_positions=2,
        path="long_short_spread",
        rebalance_every_days=1,
        factor_neutral=True,
        beta_neutral=True,
        sector_neutral=True,
        max_gross=1.0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        weights = build_target_weights(scored, cfg)["target_weight"]

    assert np.isfinite(weights.to_numpy(dtype=float)).all()


def test_factor_projection_fails_closed_on_pathological_weight_scale() -> None:
    scored = _sample_scored(days=1)
    weights = pd.Series(1e308, index=scored.index, dtype=float)
    cfg = EvaluationConfig(
        max_positions=4,
        min_positions=2,
        path="long_short_spread",
        rebalance_every_days=1,
        factor_neutral=True,
        beta_neutral=True,
        sector_neutral=True,
        max_gross=1.0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        neutral = _neutralize_day_weights(scored, weights, cfg)

    assert np.isfinite(neutral.to_numpy(dtype=float)).all()
    assert neutral.abs().sum() <= 1.0 + 1e-12


def test_liquidity_cap_prevents_oversized_positions() -> None:
    scored = _sample_scored(days=1)
    scored["adv_dollar_20"] = 1_000_000.0
    cfg = EvaluationConfig(
        max_positions=4,
        min_positions=2,
        path="long_short_spread",
        rebalance_every_days=1,
        factor_neutral=False,
        adv_fraction=0.05,
        max_gross=1.0,
        use_optimizer=False,
        costs=ExecutionCostConfig(capital=10_000_000.0),
    )

    weights = build_target_weights(scored, cfg)["target_weight"].abs()

    assert weights.max() <= 0.005 + 1e-12


def test_optimizer_weight_builder_respects_squeeze_filter_and_adv_caps() -> None:
    scored = _sample_scored(days=80, names=10)
    scored["short_squeeze_risk"] = 0.0
    scored["hard_short_squeeze_filter"] = 0.0
    scored.loc[scored["ticker"].eq("T0"), "short_squeeze_risk"] = 1.0
    scored["adv_dollar_20"] = 1_000_000.0
    cfg = EvaluationConfig(
        path="long_short_spread",
        rebalance_every_days=5,
        factor_neutral=True,
        beta_neutral=True,
        sector_neutral=True,
        max_gross=1.0,
        max_name_weight=0.10,
        adv_fraction=0.05,
        use_optimizer=True,
        min_position_weight=0.0,
        costs=ExecutionCostConfig(capital=10_000_000.0),
    )

    weights = build_target_weights(scored, cfg)
    first = weights.loc[weights["date"].eq(weights["date"].min())].set_index("ticker")["target_weight"]

    assert first.abs().max() <= 0.005 + 1e-12
    assert first.get("T0", 0.0) >= -1e-12


def test_long_only_overlay_derisks_raw_market_beta_to_budget() -> None:
    scored = _sample_scored(days=1, names=10)
    scored["capm_beta"] = 1.0
    cfg = EvaluationConfig(
        path="long_only_overlay",
        rebalance_every_days=1,
        factor_neutral=True,
        beta_neutral=True,
        sector_neutral=False,
        max_beta_abs=0.15,
        max_gross=1.0,
        max_positions=5,
        min_positions=2,
        use_optimizer=False,
        min_position_weight=0.0,
    )

    weights = build_target_weights(scored, cfg)
    first = weights.loc[weights["date"].eq(weights["date"].min())]
    beta = scored.set_index("ticker")["capm_beta"]
    raw_beta = float(
        first.set_index("ticker")["target_weight"].reindex(beta.index).fillna(0.0).mul(beta).sum()
    )

    assert raw_beta <= 0.15 + 1e-12
    assert first["_constructed_derisk_scale"].dropna().iloc[0] < 1.0


def test_validation_state_cache_persists_market_state(tmp_path, monkeypatch) -> None:
    scored = _sample_scored(days=80, names=10)
    cfg = EvaluationConfig(
        path="long_short_spread",
        rebalance_every_days=5,
        factor_neutral=True,
        beta_neutral=True,
        sector_neutral=True,
        max_gross=1.0,
        use_optimizer=True,
    )
    calls = {"count": 0}

    from model_selection import validation as validation_mod

    real_covariance_for_day = validation_mod._covariance_for_day

    def wrapped_covariance_for_day(full_df, day, dt, cfg):
        calls["count"] += 1
        return real_covariance_for_day(full_df, day, dt, cfg)

    monkeypatch.setattr(validation_mod, "_covariance_for_day", wrapped_covariance_for_day)

    first_cache = ValidationStateCache(scored, cfg=cfg, artifact_dir=tmp_path / "state")
    first = first_cache.get(pd.Timestamp(scored["date"].min()))
    assert calls["count"] == 1
    assert len(first.tickers) > 0

    second_cache = ValidationStateCache(scored, cfg=cfg, artifact_dir=tmp_path / "state")
    second = second_cache.get(pd.Timestamp(scored["date"].min()))
    assert calls["count"] == 1
    assert first.tickers == second.tickers


def test_build_target_weights_reuses_precomputed_validation_state(tmp_path, monkeypatch) -> None:
    scored = _sample_scored(days=80, names=10)
    cfg = EvaluationConfig(
        path="long_short_spread",
        rebalance_every_days=5,
        factor_neutral=True,
        beta_neutral=True,
        sector_neutral=True,
        max_gross=1.0,
        use_optimizer=True,
    )
    calls = {"count": 0}

    from model_selection import validation as validation_mod

    real_covariance_for_day = validation_mod._covariance_for_day

    def wrapped_covariance_for_day(full_df, day, dt, cfg):
        calls["count"] += 1
        return real_covariance_for_day(full_df, day, dt, cfg)

    monkeypatch.setattr(validation_mod, "_covariance_for_day", wrapped_covariance_for_day)

    state_cache = ValidationStateCache(scored, cfg=cfg, artifact_dir=tmp_path / "state")
    weights_first = build_target_weights(scored, cfg, state_cache=state_cache)
    first_calls = calls["count"]
    weights_second = build_target_weights(scored, cfg, state_cache=state_cache)

    assert first_calls > 0
    assert calls["count"] == first_calls
    pd.testing.assert_frame_equal(weights_first.reset_index(drop=True), weights_second.reset_index(drop=True))


def test_executable_simulator_reuses_score_fingerprint_cache(tmp_path, monkeypatch) -> None:
    scored = _sample_scored(days=80, names=10)
    cfg = EvaluationConfig(
        path="long_short_spread",
        rebalance_every_days=5,
        factor_neutral=True,
        beta_neutral=True,
        sector_neutral=True,
        max_gross=1.0,
        use_optimizer=True,
    )
    state_cache = ValidationStateCache(scored, cfg=cfg, artifact_dir=tmp_path / "state")
    first_returns, first_pnl = simulate_executable_portfolio(scored, cfg, state_cache=state_cache)

    from model_selection import validation as validation_mod

    def fail_build_target_weights(*args, **kwargs):
        raise AssertionError("simulation cache should bypass target-weight reconstruction")

    monkeypatch.setattr(validation_mod, "build_target_weights", fail_build_target_weights)
    second_returns, second_pnl = simulate_executable_portfolio(scored, cfg, state_cache=state_cache)

    pd.testing.assert_series_equal(first_returns, second_returns)
    pd.testing.assert_frame_equal(first_pnl.reset_index(drop=True), second_pnl.reset_index(drop=True))


def test_decile_diagnostics_detect_ordered_alpha() -> None:
    scored = _sample_scored(days=12, names=30)
    stats = decile_return_diagnostics(scored, target_col="forward_return")

    assert stats["decile_spread"] > 0.0
    assert stats["decile_monotonicity"] >= 0.8
    assert stats["decile_n_days"] == scored["date"].nunique()


def test_promotion_gates_block_unstable_models_and_pass_clean_rows() -> None:
    cfg = PromotionGateConfig(
        min_sharpe=0.5,
        min_cost_aware_sharpe=0.25,
        min_ic_tstat=2.0,
        min_ic_ir=0.75,
        min_beat_rate=0.625,
        max_drawdown=-0.25,
        min_windows=6,
        min_psr=0.60,
        max_beta_abs_mean=0.15,
        max_sector_abs_mean=0.12,
        max_cost_to_gross_pnl=0.50,
    )
    weak = {
        "n_windows": 8,
        "horizon_days": 10,
        "daily_ic_n_days": 504,
        "daily_ic_std": 0.05,
        "oos_sharpe_chained": 0.8,
        "exec_sharpe": 0.7,
        "cs_ic_spearman_tstat": 1.0,
        "cs_ic_spearman_ir": 0.4,
        "oos_beat_rate": 0.75,
        "exec_max_dd": -0.40,
        "oos_psr": 0.9,
        "exec_beta_abs_mean": 0.05,
        "exec_max_sector_abs_mean": 0.04,
        "exec_cost_to_gross_pnl": 0.20,
        "decile_spread": 0.001,
        "decile_monotonicity": 0.75,
        "exec_long_leg_sharpe": 0.50,
        "exec_short_leg_sharpe": 0.40,
        "subsumption_alpha_ann": -0.01,
        "subsumption_alpha_tstat": 0.2,
        "subsumption_r2": 0.95,
        "subsumption_max_abs_loading": 2.0,
    }
    strong = {
        **weak,
        "cs_ic_spearman_tstat": 3.0,
        "cs_ic_spearman_ir": 1.5,
        "exec_max_dd": -0.12,
        "subsumption_alpha_ann": 0.05,
        "subsumption_alpha_tstat": 2.5,
        "subsumption_r2": 0.30,
        "subsumption_max_abs_loading": 0.8,
    }

    weak_result = evaluate_promotion_gates(weak, cfg)
    strong_result = evaluate_promotion_gates(strong, cfg)

    assert weak_result["promotion_pass"] is False
    assert "min_ic_tstat" in weak_result["promotion_failures"]
    assert "max_drawdown" in weak_result["promotion_failures"]
    assert strong_result["promotion_pass"] is True


def test_dynamic_promotion_thresholds_tighten_for_small_samples() -> None:
    cfg = PromotionGateConfig(
        min_ic_tstat=1.30,
        min_ic_ir=0.50,
        min_sharpe=0.35,
        min_cost_aware_sharpe=0.15,
        min_beat_rate=0.55,
        min_windows=1,
    )
    row = {
        "n_windows": 2,
        "horizon_days": 20,
        "daily_ic_n_days": 80,
        "daily_ic_std": 0.10,
        "horizon_adj_ic_tstat": 3.0,
        "horizon_adj_ic_ir": 1.0,
        "oos_sharpe_chained": 0.8,
        "exec_sharpe": 0.7,
        "oos_beat_rate": 0.60,
        "exec_max_dd": -0.05,
        "oos_psr": 0.90,
        "exec_beta_abs_mean": 0.0,
        "exec_max_sector_abs_mean": 0.0,
        "exec_cost_to_gross_pnl": 0.10,
        "decile_spread": 0.01,
        "decile_monotonicity": 1.0,
        "exec_long_leg_sharpe": 0.2,
        "exec_short_leg_sharpe": 0.2,
    }

    result = evaluate_promotion_gates(row, cfg)

    assert result["gate_threshold_effective_min_ic_tstat"] == float("inf")
    assert result["gate_threshold_effective_min_ic_ir"] == float("inf")
    assert "min_ic_tstat" in result["promotion_failures"]
    assert "min_ic_ir" in result["promotion_failures"]


def test_dynamic_promotion_thresholds_report_static_vs_dynamic_and_cost_uplift() -> None:
    cfg = PromotionGateConfig(
        min_ic_tstat=1.30,
        min_ic_ir=0.50,
        min_sharpe=0.35,
        min_cost_aware_sharpe=0.15,
        min_beat_rate=0.55,
        min_windows=1,
    )
    row = {
        "n_windows": 8,
        "horizon_days": 10,
        "daily_ic_n_days": 504,
        "daily_ic_std": 0.05,
        "horizon_adj_ic_tstat": 3.0,
        "horizon_adj_ic_ir": 2.0,
        "oos_sharpe_chained": 1.0,
        "exec_sharpe": 1.0,
        "oos_beat_rate": 0.70,
        "exec_max_dd": -0.05,
        "oos_psr": 0.90,
        "exec_beta_abs_mean": 0.0,
        "exec_max_sector_abs_mean": 0.0,
        "exec_cost_to_gross_pnl": 1.0,
        "exec_turnover_mean": 0.70,
        "decile_spread": 0.01,
        "decile_monotonicity": 1.0,
        "exec_long_leg_sharpe": 0.2,
        "exec_short_leg_sharpe": 0.2,
    }

    result = evaluate_promotion_gates(row, cfg)

    assert result["gate_threshold_static_min_cost_aware_sharpe"] == 0.15
    assert result["gate_threshold_dynamic_min_cost_aware_sharpe"] > 0.15
    assert result["gate_threshold_effective_min_ic_tstat"] >= 1.30
    assert "cost_uplift" in result["gate_threshold_reason_min_cost_aware_sharpe"]
