from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict, is_dataclass
import errno
import hashlib
import pickle
from pathlib import Path
from collections import OrderedDict
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd

from model_selection.training import (
    FeaturePreprocessor,
    TargetConfig,
    make_training_target,
    retarget_panel_for_horizon,
)
from model_selection.validation import ExecutionCostConfig, EvaluationConfig, ValidationStateCache, DEBUG_DIAGNOSTICS, TRACE_DIAGNOSTICS
from model_selection.target_panel_provider import TargetPanelProvider
TRACE_CACHE_KEYS = False

import logging
logger = logging.getLogger(__name__)


def _freeze_for_cache_key(value):
    """Convert nested config objects into deterministic hashable cache keys."""

    if is_dataclass(value):
        return _freeze_for_cache_key(asdict(value))
    if isinstance(value, dict):
        return tuple(
            (str(k), _freeze_for_cache_key(v))
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_cache_key(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_for_cache_key(v) for v in value))
    if isinstance(value, np.ndarray):
        return tuple(_freeze_for_cache_key(v) for v in value.tolist())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


@dataclass(frozen=True)
class PreparedFold:
    train_df: pd.DataFrame
    eval_df: pd.DataFrame
    preprocessor: FeaturePreprocessor
    x_train: np.ndarray
    x_eval: np.ndarray


class PreparedPanelCache:
    """
    Refactored PreparedPanelCache (Task 3 design).
    Separates reusable horizon-level retargeting from fold-specific slicing.
    """

    def __init__(
        self,
        base_df: pd.DataFrame,
        *,
        target_cfg: TargetConfig,
        costs: ExecutionCostConfig,
        max_name_weight: float,
        winsor_q: float,
        artifact_dir: str | Path | None = None,
        max_cache_size: int = 5,
        min_free_space_mb: float = 2048.0,
        disk_persistence: str = "horizon_only",
        neutralize_factors: bool = False,
        neutralization_ridge: float = 1e-4,
        cache_fingerprint: str = "",
        target_provider: TargetPanelProvider | None = None,
    ) -> None:
        df = base_df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        self.base_df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        self.target_cfg = target_cfg
        self.costs = costs
        self.max_name_weight = float(max_name_weight)
        self.winsor_q = float(winsor_q)
        self.cache_fingerprint = str(cache_fingerprint or "")
        self._target_provider = target_provider

        # P12: Pre-fit factor neutralization
        self.neutralize_factors = bool(neutralize_factors)
        self.neutralization_ridge = float(neutralization_ridge)
        self._factor_neutralized: bool = False
        
        # New Caches
        self._horizon_panel_cache = OrderedDict()  # key: horizon_days -> Full retargeted panel
        self._prepared_fold_cache = OrderedDict()   # key: (train_start, train_end, eval_start, eval_end, horizon, feature_tuple)
        self._training_target_cache = OrderedDict()
        self._validation_state_cache = OrderedDict()
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        if self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.min_free_space_mb = float(min_free_space_mb)
        self.disk_persistence = str(disk_persistence or "horizon_only").strip().lower()
        if self.disk_persistence not in {"none", "horizon_only", "all"}:
            raise ValueError("disk_persistence must be one of: none, horizon_only, all")
        self._artifact_writes_enabled = self.artifact_dir is not None
        self._artifact_write_failures = 0
        self._artifact_disabled_reason = ""
        
        self.max_cache_size = int(max_cache_size)
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._memory_hits_by_name: dict[str, int] = {}
        self._memory_misses_by_name: dict[str, int] = {}
        self._artifact_hits_by_name: dict[str, int] = {}
        self._artifact_misses_by_name: dict[str, int] = {}
        self._artifact_writes_by_name: dict[str, int] = {}
        self._lookup_history: list[tuple] = []
        self._validation_audit_done = False

    def purge(self) -> None:
        self._horizon_panel_cache.clear()
        self._prepared_fold_cache.clear()
        self._training_target_cache.clear()
        self._validation_state_cache.clear()
        self.base_df = None
        self._validation_audit_done = False
        import gc
        gc.collect()

    def _get_from_lru(self, cache: OrderedDict, key: Any, name: str = "unknown") -> Any:
        if key in cache:
            cache.move_to_end(key)
            self._hits += 1
            self._memory_hits_by_name[name] = self._memory_hits_by_name.get(name, 0) + 1
            if TRACE_CACHE_KEYS:
                print(f"[Cache Hit]  {name} | key={key}")
            return cache[key]
        self._misses += 1
        self._memory_misses_by_name[name] = self._memory_misses_by_name.get(name, 0) + 1
        if TRACE_CACHE_KEYS:
            print(f"[Cache Miss] {name} | key={key}")
        return None

    def _add_to_lru(self, cache: OrderedDict, key: Any, value: Any, name: str = "cache") -> None:
        cache[key] = value
        if len(cache) > self.max_cache_size:
            cache.popitem(last=False)
            self._evictions += 1

    def _artifact_path(self, name: str, key: Any) -> Path | None:
        if self.artifact_dir is None:
            return None
        frozen = repr(_freeze_for_cache_key(key))
        digest = self._stable_hash((name, frozen))
        return self.artifact_dir / f"{name}_{digest}.pkl"

    def _should_persist_artifact(self, name: str) -> bool:
        if self.disk_persistence == "none":
            return False
        if self.disk_persistence == "horizon_only":
            return str(name) == "horizon_panel"
        return True

    def _free_space_mb(self) -> float:
        if self.artifact_dir is None:
            return float("inf")
        try:
            usage = shutil.disk_usage(self.artifact_dir)
            return float(usage.free) / (1024.0 * 1024.0)
        except OSError:
            return 0.0

    def disk_cache_writable(self) -> bool:
        """Return whether this cache may safely create persistent artifacts."""
        if self.disk_persistence == "none" or self.artifact_dir is None or not self._artifact_writes_enabled:
            return False
        free_mb = self._free_space_mb()
        if free_mb < float(self.min_free_space_mb):
            self._disable_artifact_writes(
                f"free_space_below_reserve:{free_mb:.1f}MB<{self.min_free_space_mb:.1f}MB"
            )
            return False
        return True

    def _disable_artifact_writes(self, reason: str) -> None:
        if self._artifact_writes_enabled:
            logger.warning("Disabling PreparedPanelCache artifact writes: %s", reason)
        self._artifact_writes_enabled = False
        self._artifact_disabled_reason = str(reason)

    def _read_artifact(self, name: str, key: Any) -> Any:
        if not self._should_persist_artifact(name):
            return None
        path = self._artifact_path(name, key)
        if path is None or not path.exists():
            self._artifact_misses_by_name[name] = self._artifact_misses_by_name.get(name, 0) + 1
            return None
        try:
            with path.open("rb") as fh:
                artifact = pickle.load(fh)
            self._artifact_hits_by_name[name] = self._artifact_hits_by_name.get(name, 0) + 1
            return artifact
        except Exception:
            self._artifact_misses_by_name[name] = self._artifact_misses_by_name.get(name, 0) + 1
            logger.warning("Ignoring unreadable PreparedPanelCache artifact: %s", path)
            return None

    def _write_artifact(self, name: str, key: Any, value: Any) -> None:
        if not self._should_persist_artifact(name):
            return
        path = self._artifact_path(name, key)
        if path is None or not self.disk_cache_writable():
            return
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with tmp.open("wb") as fh:
                pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
            self._artifact_writes_by_name[name] = self._artifact_writes_by_name.get(name, 0) + 1
        except OSError as exc:
            self._artifact_write_failures += 1
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            if exc.errno == errno.ENOSPC:
                self._disable_artifact_writes("no_space_left_on_device")
            else:
                self._disable_artifact_writes(f"artifact_write_error:{exc}")
            return

    def reset_runtime_stats(self) -> None:
        """Reset hit/miss telemetry without clearing cached data.

        Use after deterministic warmup so reported cache metrics describe the
        research phase rather than expected first-build misses.
        """
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._memory_hits_by_name.clear()
        self._memory_misses_by_name.clear()
        self._artifact_hits_by_name.clear()
        self._artifact_misses_by_name.clear()
        self._artifact_writes_by_name.clear()
        self._lookup_history.clear()

    def get_full_retargeted_panel(self, horizon_days: int) -> pd.DataFrame:
        """Cache level 1: Full retargeted panel for a horizon."""
        h = max(1, int(horizon_days))
        key = (h, self.cache_fingerprint)
        panel = self._get_from_lru(self._horizon_panel_cache, key, name="horizon_panel")
        if panel is not None:
            return panel
        artifact = self._read_artifact("horizon_panel", key)
        if artifact is not None:
            self._add_to_lru(self._horizon_panel_cache, key, artifact, name="horizon_panel")
            return artifact

        # C.2: Use TargetPanelProvider when available
        if self._target_provider is not None:
            full_panel = self._target_provider.get_target_panel(h)
        else:
            full_panel = retarget_panel_for_horizon(
                self.base_df,
                horizon_days=h,
                target_cfg=self.target_cfg,
                costs=self.costs,
                max_name_weight=self.max_name_weight,
            )

        # P12: Pre-fit factor neutralization (institutional fix)
        # Neutralize ALL features against factor exposures at the panel level
        # before slicing into folds. This prevents any model from learning
        # beta/factor exposure in the first place.
        if self.neutralize_factors and not self._factor_neutralized:
            full_panel = self._neutralize_feature_panel(full_panel)
            self._factor_neutralized = True

        self._add_to_lru(self._horizon_panel_cache, key, full_panel, name="horizon_panel")
        self._write_artifact("horizon_panel", key, full_panel)
        return full_panel

    def _neutralize_feature_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-fit factor neutralization at the panel level.

        Residualizes every numeric feature column against factor exposures
        (CAPM beta, sector dummies, size, volatility) per date. The target
        column is also residualized so features and target live in the same
        idiosyncratic space.

        This is the Two Sigma / Jane Street approach: neutralization happens
        at the data preparation stage, not at the scoring stage.
        """
        from model_selection.factor_neutralization import neutralize_features_before_training, NeutralizationConfig

        # Identify feature columns (all numeric columns except known non-features)
        _exclude = {
            "date", "ticker", "forward_return", "target_return", "target_up",
            "y_bin", "raw_return", "residual_return", "net_residual_return",
            "daily_return", "regime_label", "regime_score",
            "regime_proba_bull", "regime_proba_bear", "regime_proba_crisis",
            "capm_beta", "sector",
        }
        feature_cols = [
            c for c in panel.columns
            if c not in _exclude and pd.api.types.is_numeric_dtype(panel[c])
        ]

        if not feature_cols:
            return panel

        ncfg = NeutralizationConfig(
            use_market=True,
            use_sector=True,
            use_size=True,
            use_vol=True,
            ridge=self.neutralization_ridge,
            value_cap=5.0,
        )

        target_col = "target_return" if "target_return" in panel.columns else None

        neutralized = neutralize_features_before_training(
            panel,
            feature_cols,
            cfg=ncfg,
            target_col=target_col,
        )

        r2_mean = neutralized.attrs.get("neutralization_r2_mean", float("nan"))
        n_dates = neutralized.attrs.get("neutralization_n_dates", 0)
        if np.isfinite(r2_mean):
            logger.info(
                "[FactorNeutralization] Pre-fit neutralization: R²=%.3f across %d dates | features=%d",
                r2_mean, n_dates, len(feature_cols),
            )

        return neutralized

    def get_prepared_fold(
        self,
        *,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        eval_start: pd.Timestamp,
        eval_end: pd.Timestamp,
        horizon_days: int,
        active_features: list[str],
    ) -> PreparedFold:
        """Construct a PreparedFold by slicing the full panel. Reuse is prioritized."""
        feature_tuple = tuple(sorted(active_features))
        key = (
            pd.Timestamp(train_start).isoformat(),
            pd.Timestamp(train_end).isoformat(),
            pd.Timestamp(eval_start).isoformat(),
            pd.Timestamp(eval_end).isoformat(),
            int(horizon_days),
            feature_tuple,
        )
        self._lookup_history.append(key)
        
        # 1. Check fold-level cache
        prepared = self._get_from_lru(self._prepared_fold_cache, key, name="prepared_fold")
        if prepared is not None:
            return prepared
        artifact = self._read_artifact("prepared_fold", key)
        if artifact is not None:
            self._add_to_lru(self._prepared_fold_cache, key, artifact, name="prepared_fold")
            return artifact
            
        # 2. Slice from horizon-level panel
        full_panel = self.get_full_retargeted_panel(horizon_days)
        if full_panel.empty:
            return PreparedFold(pd.DataFrame(), pd.DataFrame(), FeaturePreprocessor(features=active_features, medians=pd.Series(), lower=pd.Series(), upper=pd.Series(), active_features=[]), np.empty((0,0)), np.empty((0,0)))

        dates = full_panel["date"].values
        train_mask = (dates >= train_start) & (dates < train_end)
        eval_mask = (dates >= eval_start) & (dates < eval_end)
        
        # [Task 3: Eliminate unnecessary copies]
        # Using boolean indexing directly; fit/transform handle views correctly.
        train_df = full_panel[train_mask]
        eval_df = full_panel[eval_mask]
        
        if train_df.empty or eval_df.empty:
             return PreparedFold(train_df, eval_df, FeaturePreprocessor(features=active_features, medians=pd.Series(), lower=pd.Series(), upper=pd.Series(), active_features=[]), np.empty((0,0)), np.empty((0,0)))

        # 3. Fit features on TRAIN, transform both
        preproc = FeaturePreprocessor.fit(train_df, active_features, winsor_q=self.winsor_q)
        x_train = preproc.transform(train_df)
        x_eval = preproc.transform(eval_df)
        
        prepared = PreparedFold(
            train_df=train_df,
            eval_df=eval_df,
            preprocessor=preproc,
            x_train=x_train,
            x_eval=x_eval,
        )
        
        # [One-Time Fold Audit]
        if DEBUG_DIAGNOSTICS and not self._validation_audit_done:
            print("\n[One-Time Fold Audit: First Prepared Fold]")
            print(f"  train_rows : {len(train_df)}")
            print(f"  eval_rows  : {len(eval_df)}")
            print(f"  x_train_shp: {x_train.shape}")
            print(f"  x_eval_shp : {x_eval.shape}")
            print(f"  target_mu  : {train_df['target_return'].mean():.6f}")
            print(f"  target_std : {train_df['target_return'].std():.6f}")
            if len(train_df) > 0:
                print(f"  sample_tkr : {train_df['ticker'].iloc[:5].tolist()}")
                print(f"  sample_dt  : {train_df['date'].iloc[:5].dt.date.tolist()}")
            print("-" * 40)
            self._validation_audit_done = True
            
        self._add_to_lru(self._prepared_fold_cache, key, prepared, name="prepared_fold")
        self._write_artifact("prepared_fold", key, prepared)
        return prepared

    def get_training_target(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        horizon_days: int,
        model_name: str,
        model_kind: str,
        use_risk_adj: bool,
    ) -> np.ndarray:
        """Slice training target from the horizon-level panel."""
        key = (
            pd.Timestamp(start).isoformat(),
            pd.Timestamp(end).isoformat(),
            int(horizon_days),
            str(model_name),
            str(model_kind),
            bool(use_risk_adj),
            self.cache_fingerprint,
        )
        cached = self._get_from_lru(self._training_target_cache, key, name="training_target")
        if cached is not None:
            return cached.copy()
        artifact = self._read_artifact("training_target", key)
        if artifact is not None:
            arr = np.asarray(artifact, dtype=float)
            self._add_to_lru(self._training_target_cache, key, arr, name="training_target")
            return arr.copy()

        full_panel = self.get_full_retargeted_panel(horizon_days)
        dates = full_panel["date"].values
        mask = (dates >= start) & (dates < end)
        sliced = full_panel[mask]
        
        target = make_training_target(
            sliced,
            model_name=str(model_name),
            model_kind=str(model_kind),
            use_risk_adj=bool(use_risk_adj),
        )
        arr = np.nan_to_num(np.asarray(target, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        self._add_to_lru(self._training_target_cache, key, arr, name="training_target")
        self._write_artifact("training_target", key, arr)
        return arr.copy()

    def get_validation_state(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        horizon_days: int,
        evaluation_cfg: EvaluationConfig | None = None,
        *,
        train_start: pd.Timestamp | None = None,
        train_end: pd.Timestamp | None = None,
        active_features: list[str] | None = None,
        model_name: str | None = None,
        model_kind: str | None = None,
        use_risk_adj: bool = True,
    ) -> Any:
        """
        Recreates ValidationStateCache for the given eval window (if evaluation_cfg provided)
        OR returns (x_train, x_eval, y_train, y_eval, metadata) tuple for the validation pipeline.
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        
        # Case A: Return ValidationStateCache (used by research tools and simulate_executable_portfolio)
        if evaluation_cfg is not None:
            key = (
                start_ts.isoformat(),
                end_ts.isoformat(),
                int(horizon_days),
                _freeze_for_cache_key(evaluation_cfg),
            )
            cached = self._get_from_lru(self._validation_state_cache, key, name="validation_state")
            if cached is not None:
                return cached
                
            dates = self.base_df["date"].values
            mask = (dates >= start_ts) & (dates < end_ts)
            eval_df = self.base_df[mask]
            
            state = ValidationStateCache(eval_df, cfg=evaluation_cfg)
            self._add_to_lru(self._validation_state_cache, key, state, name="validation_state")
            return state

        # Case B: Return (x_train, x_eval, y_train, y_eval, metadata) tuple
        # Used for bit-level parity validation and some legacy research paths.
        if train_start is None or train_end is None or active_features is None:
            raise ValueError("get_validation_state requires train_start, train_end, and active_features for tuple return.")

        prepared = self.get_prepared_fold(
            train_start=train_start,
            train_end=train_end,
            eval_start=start_ts,
            eval_end=end_ts,
            horizon_days=horizon_days,
            active_features=active_features,
        )
        
        y_train = self.get_training_target(
            start=train_start,
            end=train_end,
            horizon_days=horizon_days,
            model_name=model_name or "Validation",
            model_kind=model_kind or "regressor",
            use_risk_adj=use_risk_adj,
        )
        
        # y_eval is sliced from the full retargeted panel
        full_panel = self.get_full_retargeted_panel(horizon_days)
        dates_p = full_panel["date"].values
        eval_mask = (dates_p >= start_ts) & (dates_p < end_ts)
        y_eval = full_panel.loc[eval_mask, "target_return"].to_numpy(dtype=float)
        
        metadata = {
            "train_rows": len(prepared.train_df),
            "eval_rows": len(prepared.eval_df),
            "feature_count": len(active_features),
            "horizon": horizon_days,
        }
        
        return prepared.x_train, prepared.x_eval, y_train, y_eval, metadata

    def log_summary(self) -> None:
        total = self._hits + self._misses
        artifact_hits = sum(self._artifact_hits_by_name.values())
        effective_hits = self._hits + artifact_hits
        effective_rate = (effective_hits / total * 100) if total > 0 else 0.0
        memory_rate = (self._hits / total * 100) if total > 0 else 0.0
        logger.info(
            "  [Cache Summary] "
            f"memory_hits={self._hits}, artifact_hits={artifact_hits}, misses={self._misses}, "
            f"evictions={self._evictions}, memory_hit_rate={memory_rate:.1f}%, "
            f"effective_hit_rate={effective_rate:.1f}%"
        )

    def log_uniqueness_report(self) -> None:
        if not self._lookup_history: return
        total = len(self._lookup_history)
        n_unique = len(set(self._lookup_history))
        unique_ratio = float(n_unique / total) if total else 0.0
        print("\n[PreparedPanelCache Uniqueness Report]")
        print(f"  total_lookups  : {total}")
        print(f"  unique_keys    : {n_unique}")
        print(f"  memory_hit_rate: {(self._hits/total*100):.1f}%")
        print(f"  artifact_hits  : {sum(self._artifact_hits_by_name.values())}")
        if total > 0 and unique_ratio > 0.90:
            print("  !! Fold-level prepared_fold cache is structurally non-reusable.")
        print("-" * 40)

    @staticmethod
    def get_rss_mb() -> float:
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    @staticmethod
    def _stable_hash(parts: tuple[str, ...]) -> str:
        """Deterministic hex hash of a tuple of strings. Used for cache-key file names.

        Produces a 16-char hex prefix of SHA-256 of the joined parts.
        Stable across runs: depends only on content, not memory layout.
        """
        joined = "||".join(parts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    def stats(self) -> dict[str, float | int]:
        """Return a snapshot of cache entry counts and hit/miss statistics.

        Keys mirror the old PreparedPanelCache API expected by call sites in
        run_model_selection.py (lines ~6139, ~6552, ~3011):
          raw_panel_cache_entries         — always 0 (no dedicated raw cache post-refactor)
          retargeted_panel_cache_entries  — horizon-level full-panel cache size
          prepared_fold_cache_entries     — fold-slice cache size
          training_target_cache_entries   — always 0 (computed on-demand from horizon panel)
          validation_state_cache_entries  — validation state cache size
          hits / misses / evictions       — LRU accounting across all sub-caches
          total_lookups                   — hits + misses
          prepared_fold_cache_hit_rate    — overall hit-rate expressed as 0-100 (%)
        """
        total = self._hits + self._misses
        hit_rate_pct = (self._hits / total * 100.0) if total > 0 else 0.0
        artifact_hits = sum(self._artifact_hits_by_name.values())
        effective_hits = self._hits + artifact_hits
        effective_hit_rate_pct = (effective_hits / total * 100.0) if total > 0 else 0.0
        fold_lookups = len(self._lookup_history)
        fold_unique = len(set(self._lookup_history))
        fold_unique_ratio = (fold_unique / fold_lookups) if fold_lookups else 0.0
        fold_structurally_unique = bool(fold_lookups > 0 and fold_unique_ratio > 0.90)
        return {
            "raw_panel_cache_entries": 0,
            "retargeted_panel_cache_entries": len(self._horizon_panel_cache),
            "prepared_fold_cache_entries": len(self._prepared_fold_cache),
            "training_target_cache_entries": len(self._training_target_cache),
            "validation_state_cache_entries": len(self._validation_state_cache),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "total_lookups": total,
            "prepared_fold_cache_hit_rate": hit_rate_pct,
            "effective_cache_hit_rate": effective_hit_rate_pct,
            "artifact_hits": artifact_hits,
            "artifact_misses": sum(self._artifact_misses_by_name.values()),
            "artifact_writes": sum(self._artifact_writes_by_name.values()),
            "horizon_memory_hits": int(self._memory_hits_by_name.get("horizon_panel", 0)),
            "horizon_memory_misses": int(self._memory_misses_by_name.get("horizon_panel", 0)),
            "horizon_artifact_hits": int(self._artifact_hits_by_name.get("horizon_panel", 0)),
            "horizon_artifact_misses": int(self._artifact_misses_by_name.get("horizon_panel", 0)),
            "horizon_artifact_writes": int(self._artifact_writes_by_name.get("horizon_panel", 0)),
            "fold_memory_hits": int(self._memory_hits_by_name.get("prepared_fold", 0)),
            "fold_memory_misses": int(self._memory_misses_by_name.get("prepared_fold", 0)),
            "target_memory_hits": int(self._memory_hits_by_name.get("training_target", 0)),
            "target_memory_misses": int(self._memory_misses_by_name.get("training_target", 0)),
            "target_artifact_hits": int(self._artifact_hits_by_name.get("training_target", 0)),
            "target_artifact_misses": int(self._artifact_misses_by_name.get("training_target", 0)),
            "target_artifact_writes": int(self._artifact_writes_by_name.get("training_target", 0)),
            "prepared_fold_lookups": fold_lookups,
            "prepared_fold_unique_keys": fold_unique,
            "prepared_fold_unique_ratio": fold_unique_ratio,
            "prepared_fold_structurally_unique": fold_structurally_unique,
            "prepared_fold_cache_capacity": int(self.max_cache_size),
            "artifact_writes_enabled": bool(self._artifact_writes_enabled),
            "artifact_write_failures": int(self._artifact_write_failures),
            "artifact_disabled_reason": self._artifact_disabled_reason,
            "artifact_free_space_mb": self._free_space_mb(),
            "artifact_min_free_space_mb": float(self.min_free_space_mb),
            "disk_persistence": self.disk_persistence,
        }
