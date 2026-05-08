"""TargetPanelProvider — precompute forward returns and institutional targets once.

Phase C.2: Eliminates redundant forward-return / target construction across
engines by computing each TargetSpec/HorizonRunContract/data slice exactly
once and sharing the resulting target panel.

Design rules:
1. Wraps existing add_institutional_targets / retarget_panel_for_horizon — does
   not rewrite target formulas.
2. Computes each required TargetSpec once per run/data slice.
3. Exposes target columns to downstream engines without rebuilding.
4. Preserves existing column names.
5. Deterministic: canonical date/ticker ordering, stable fingerprinting.
6. Cache-safe: includes all relevant fingerprints in cache key.
"""
from __future__ import annotations

import hashlib
import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_selection.training import (
    TargetConfig,
    add_institutional_targets,
    retarget_panel_for_horizon,
)
from model_selection.validation import ExecutionCostConfig

logger = logging.getLogger(__name__)

# ── Version bump whenever target construction logic changes ─────────────────
_PROVIDER_VERSION = "1.0.0"

# ── Column groups produced by the provider ──────────────────────────────────
TARGET_COLUMNS = frozenset({
    "target_return_net",
    "target_return",
    "target_rank",
    "target_down_decile",
    "target_up",
    "target_expected_cost",
    "target_expected_participation",
    "target_expected_fixed_cost",
    "target_expected_temporary_impact",
    "target_expected_permanent_impact",
    "target_expected_borrow_cost",
})

COST_COLUMNS = frozenset({
    "target_expected_cost",
    "target_expected_participation",
    "target_expected_fixed_cost",
    "target_expected_temporary_impact",
    "target_expected_permanent_impact",
    "target_expected_borrow_cost",
})


# ── Fingerprint helpers ─────────────────────────────────────────────────────

def _stable_hash(obj: Any) -> str:
    """Deterministic SHA-256 hash for cache-key components."""
    raw = json.dumps(_freeze(obj), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _freeze(obj: Any) -> Any:
    """Recursively convert to JSON-serializable structure."""
    if isinstance(obj, dict):
        return {k: _freeze(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple, set)):
        return [_freeze(v) for v in sorted(obj, key=str)]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return _freeze(vars(obj))
    return obj


def _data_fingerprint(df: pd.DataFrame, max_rows: int = 10_000) -> str:
    """Fingerprint of data shape, columns, and a sample of values."""
    sample = df.head(max_rows)
    shape = sample.shape
    cols = tuple(sorted(sample.columns))
    # Hash first and last rows of numeric columns
    numeric_cols = sorted(sample.select_dtypes(include=[np.number]).columns)
    if numeric_cols:
        first_row = tuple(float(x) for x in sample.iloc[0][numeric_cols].values)
        last_row = tuple(float(x) for x in sample.iloc[-1][numeric_cols].values)
    else:
        first_row = ()
        last_row = ()
    return _stable_hash((shape, cols, first_row, last_row))


def _target_config_fingerprint(cfg: TargetConfig) -> str:
    return _stable_hash({
        "horizon_days": cfg.horizon_days,
        "residualize": cfg.residualize,
        "net_of_costs": cfg.net_of_costs,
        "residual_ridge": cfg.residual_ridge,
        "winsor_q": cfg.winsor_q,
        "max_abs_return": cfg.max_abs_return,
    })


def _cost_config_fingerprint(cfg: ExecutionCostConfig) -> str:
    """Fingerprint of execution cost assumptions."""
    d = {}
    for attr in dir(cfg):
        if attr.startswith("_"):
            continue
        val = getattr(cfg, attr, None)
        if val is not None and not callable(val):
            d[attr] = val
    return _stable_hash(d)


# ── Manifest ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TargetManifest:
    """Metadata about the precomputed target panel."""
    provider_version: str
    target_spec_fingerprints: dict[str, str]  # spec_name -> fingerprint
    data_fingerprint: str
    horizon_fingerprints: dict[int, str]       # horizon_days -> fingerprint
    row_count: int
    date_count: int
    ticker_count: int
    target_columns: tuple[str, ...]
    missingness_report: dict[str, float]       # column -> NaN fraction
    cache_key: str
    elapsed_seconds: float


# ── Telemetry event ─────────────────────────────────────────────────────────

@dataclass
class TargetTelemetryEvent:
    stage: str
    target_specs: list[str]
    horizons: list[int]
    rows: int
    dates: int
    tickers: int
    elapsed_seconds: float
    cache_status: str  # "hit", "miss", "n/a"
    warning: str | None = None


# ── TargetPanelProvider ─────────────────────────────────────────────────────

class TargetPanelProvider:
    """Precompute and serve forward returns + institutional target columns.

    Computes targets once per (data, HorizonRunContract, TargetSpec) and shares
    the resulting panel across all downstream engines.

    Parameters
    ----------
    raw_panel : pd.DataFrame
        Input panel with at least [date, ticker] and either [forward_return]
        or [daily_return].
    target_cfg : TargetConfig
        Base target specification.
    costs : ExecutionCostConfig
        Execution cost assumptions.
    max_name_weight : float
        Maximum single-name weight for cost modeling.
    horizons : list[int] | None
        Horizons to precompute. If None, uses target_cfg.horizon_days only.
    cache : dict | None
        External cache dict for target panels keyed by (horizon, fingerprint).
        If None, internal caching only.
    """

    def __init__(
        self,
        raw_panel: pd.DataFrame,
        *,
        target_cfg: TargetConfig,
        costs: ExecutionCostConfig,
        max_name_weight: float,
        horizons: list[int] | None = None,
        cache: dict | None = None,
    ) -> None:
        self._raw_panel = raw_panel.copy()
        self._target_cfg = target_cfg
        self._costs = costs
        self._max_name_weight = float(max_name_weight)
        self._horizons = sorted(set(horizons or [target_cfg.horizon_days]))
        self._external_cache: dict | None = cache

        # Internal per-horizon cache
        self._panel_cache: dict[int, pd.DataFrame] = {}
        self._telemetry: list[TargetTelemetryEvent] = []

        # Canonicalize ordering once
        self._raw_panel["date"] = pd.to_datetime(self._raw_panel["date"], errors="coerce")
        self._raw_panel = self._raw_panel.sort_values(["date", "ticker"]).reset_index(drop=True)

        # Precompute fingerprints
        self._data_fp = _data_fingerprint(self._raw_panel)
        self._cost_fp = _cost_config_fingerprint(costs)
        self._cfg_fp = _target_config_fingerprint(target_cfg)

    # ── Public API ───────────────────────────────────────────────────────

    def get_target_panel(self, horizon_days: int) -> pd.DataFrame:
        """Return the full target panel for a given horizon.

        Computes once, then serves from cache on subsequent calls.
        """
        h = max(1, int(horizon_days))
        if h not in self._panel_cache:
            t0 = time.monotonic()
            panel = self._build_panel(h)
            elapsed = time.monotonic() - t0
            cache_status = "miss"

            # Check external cache
            if self._external_cache is not None:
                ck = self._cache_key(h)
                if ck in self._external_cache:
                    panel = self._external_cache[ck]
                    cache_status = "hit"
                else:
                    self._external_cache[ck] = panel

            self._panel_cache[h] = panel
            self._telemetry.append(TargetTelemetryEvent(
                stage="build_panel",
                target_specs=[f"horizon_{h}"],
                horizons=[h],
                rows=len(panel),
                dates=int(panel["date"].nunique()),
                tickers=int(panel["ticker"].nunique()),
                elapsed_seconds=elapsed,
                cache_status=cache_status,
            ))
            logger.info(
                "[TargetPanelProvider] Built panel for h=%dd in %.3fs (cache=%s)",
                h, elapsed, cache_status,
            )
        return self._panel_cache[h]

    def get_target_columns(self, horizon_days: int) -> pd.DataFrame:
        """Return only the target columns (not the full panel)."""
        panel = self.get_target_panel(horizon_days)
        cols = ["date", "ticker"] + sorted(TARGET_COLUMNS & set(panel.columns))
        return panel[cols].copy()

    def has_horizon(self, horizon_days: int) -> bool:
        return horizon_days in self._horizons

    def get_manifest(self) -> TargetManifest:
        """Build a manifest describing all precomputed targets."""
        t0 = time.monotonic()
        # Ensure all horizons are built
        for h in self._horizons:
            self.get_target_panel(h)

        # Use last built panel for stats
        last_panel = self._panel_cache[self._horizons[-1]] if self._horizons else self._raw_panel

        missingness = {}
        for col in TARGET_COLUMNS:
            if col in last_panel.columns:
                missingness[col] = float(last_panel[col].isna().mean())

        elapsed = time.monotonic() - t0
        return TargetManifest(
            provider_version=_PROVIDER_VERSION,
            target_spec_fingerprints={
                f"horizon_{h}": self._cache_key(h)
                for h in self._horizons
            },
            data_fingerprint=self._data_fp,
            horizon_fingerprints={h: self._cache_key(h) for h in self._horizons},
            row_count=len(last_panel),
            date_count=int(last_panel["date"].nunique()),
            ticker_count=int(last_panel["ticker"].nunique()),
            target_columns=tuple(sorted(TARGET_COLUMNS & set(last_panel.columns))),
            missingness_report=missingness,
            cache_key=self._cache_key(self._horizons[-1]) if self._horizons else "",
            elapsed_seconds=elapsed,
        )

    def get_telemetry(self) -> list[TargetTelemetryEvent]:
        return list(self._telemetry)

    def get_all_horizon_panels(self) -> dict[int, pd.DataFrame]:
        """Return all precomputed horizon panels, building any missing ones."""
        for h in self._horizons:
            self.get_target_panel(h)
        return dict(self._panel_cache)

    # ── Cache key ────────────────────────────────────────────────────────

    def _cache_key(self, horizon_days: int) -> str:
        """Deterministic cache key for a specific horizon."""
        h = max(1, int(horizon_days))
        components = (
            _PROVIDER_VERSION,
            self._data_fp,
            self._cfg_fp,
            self._cost_fp,
            h,
            self._max_name_weight,
        )
        return _stable_hash(components)

    # ── Internal construction ────────────────────────────────────────────

    def _build_panel(self, horizon_days: int) -> pd.DataFrame:
        """Build the target panel for a single horizon using existing logic."""
        return retarget_panel_for_horizon(
            self._raw_panel,
            horizon_days=horizon_days,
            target_cfg=self._target_cfg,
            costs=self._costs,
            max_name_weight=self._max_name_weight,
        )


# ── Convenience: build provider from run_model_selection context ─────────────

def build_target_provider(
    panel: pd.DataFrame,
    *,
    target_cfg: TargetConfig,
    costs: ExecutionCostConfig,
    max_name_weight: float,
    horizons: list[int] | None = None,
    cache: dict | None = None,
) -> TargetPanelProvider:
    """Factory for TargetPanelProvider with validation."""
    required = {"date", "ticker"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Panel missing required columns: {missing}")
    if "forward_return" not in panel.columns and "daily_return" not in panel.columns:
        raise ValueError("Panel must have either 'forward_return' or 'daily_return'")
    return TargetPanelProvider(
        panel,
        target_cfg=target_cfg,
        costs=costs,
        max_name_weight=max_name_weight,
        horizons=horizons,
        cache=cache,
    )
