from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from model_selection.horizon_contract import research_cache_fingerprint_from_config as _cache_fingerprint_from_cfg


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def frame_fingerprint(df: pd.DataFrame, *, columns: list[str] | None = None) -> str:
    """Stable content fingerprint for a staged panel artifact."""
    if df is None or df.empty:
        return "empty"
    work = df.loc[:, [c for c in (columns or list(df.columns)) if c in df.columns]].copy()
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    hashed = pd.util.hash_pandas_object(work, index=False, categorize=True)
    digest = hashlib.sha1()
    digest.update(str(tuple(work.columns)).encode("utf-8"))
    digest.update(str(work.shape).encode("utf-8"))
    digest.update(hashed.to_numpy(dtype="uint64", copy=False).tobytes())
    return digest.hexdigest()[:20]


class ResearchStateStore:
    """
    Persistent artifact store for model-selection research state.

    Institutional model selection should not rebuild the full research stack on
    every run. This store stages:
      - point-in-time universe membership metadata
      - price / feature panels
      - alpha-research outputs
      - prepared fold / retargeted panel artifacts (via PreparedPanelCache)

    The store is keyed by a deterministic run signature so repeated runs with
    the same economic contract reuse state while config changes naturally fork
    to a new artifact namespace.
    """

    def __init__(
        self,
        *,
        root_dir: str | Path,
        namespace: str,
        payload: dict[str, Any],
    ) -> None:
        self.root_dir = Path(root_dir)
        self.namespace = str(namespace)
        self.payload = _jsonable(payload)
        self.signature = _signature({"namespace": self.namespace, **self.payload})
        self.path = self.root_dir / f"{self.namespace}_{self.signature}"
        self.path.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.path / "manifest.json"
        if not self._manifest_path.exists():
            self._manifest_path.write_text(
                json.dumps(
                    {
                        "namespace": self.namespace,
                        "signature": self.signature,
                        "payload": self.payload,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

    @classmethod
    def for_model_selection(
        cls,
        *,
        cfg: dict[str, Any],
        tickers: list[str],
        start_date: str,
        end_date: str,
        horizon_days: int,
        feature_subset: list[str] | None,
        provider: str | None,
        cache_fingerprint: str | None = None,
    ) -> "ResearchStateStore":
        payload = {
            "provider": str(provider or ""),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "horizon_days": int(horizon_days),
            "tickers": sorted(str(t).upper() for t in tickers),
            "feature_subset": sorted(str(f) for f in (feature_subset or [])),
            "target_cfg": (cfg.get("model_selection", {}) or {}).get("target", {}) if isinstance(cfg, dict) else {},
            "alpha_research_cfg": (cfg.get("model_selection", {}) or {}).get("alpha_research", {}) if isinstance(cfg, dict) else {},
            "validation_cfg": (cfg.get("model_selection", {}) or {}).get("validation", {}) if isinstance(cfg, dict) else {},
            "universe_mode": str(((cfg.get("universe", {}) or {}).get("mode", "")) if isinstance(cfg, dict) else ""),
            "pit_universe_mode": str(cfg.get("pit_universe_mode", "")) if isinstance(cfg, dict) else "",
            # P38: Research-cache fingerprint
            "cache_fingerprint": str(cache_fingerprint or _cache_fingerprint_from_cfg(cfg)),
        }
        return cls(root_dir="output/research_state", namespace="model_selection", payload=payload)

    @classmethod
    def for_alpha_research(
        cls,
        *,
        cfg: dict[str, Any],
        tickers: list[str],
        start_date: str,
        end_date: str,
        provider: str | None,
        feature_columns: list[str],
        feature_panel_signature: str,
        feature_contract: dict[str, Any],
        alpha_admission_policy: Any | None = None,
        alpha_research_schema_version: str = "",
        cache_fingerprint: str | None = None,
    ) -> "ResearchStateStore":
        ms_cfg = (cfg.get("model_selection", {}) or {}) if isinstance(cfg, dict) else {}
        payload = {
            "provider": str(provider or ""),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "tickers": sorted(str(t).upper() for t in tickers),
            "feature_columns": sorted(str(f) for f in feature_columns),
            "feature_panel_signature": str(feature_panel_signature),
            "feature_contract": feature_contract,
            "target_cfg": ms_cfg.get("target", {}),
            "alpha_research_cfg": ms_cfg.get("alpha_research", {}),
            "alpha_admission_policy": alpha_admission_policy,
            "alpha_research_schema_version": str(alpha_research_schema_version),
            "universe_mode": str(((cfg.get("universe", {}) or {}).get("mode", "")) if isinstance(cfg, dict) else ""),
            "pit_universe_mode": str(cfg.get("pit_universe_mode", "")) if isinstance(cfg, dict) else "",
            "cache_fingerprint": str(cache_fingerprint or _cache_fingerprint_from_cfg(cfg)),
        }
        return cls(root_dir="output/research_state", namespace="alpha_research", payload=payload)

    def subdir(self, name: str) -> Path:
        path = self.path / str(name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def timing_ledger(self, name: str = "timing_ledger.jsonl") -> "TimingLedger":
        return TimingLedger(self.path / str(name))

    def _frame_path(self, name: str) -> Path:
        return self.path / f"{name}.parquet"

    def load_frame(self, name: str) -> pd.DataFrame | None:
        path = self._frame_path(name)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def save_frame(self, name: str, df: pd.DataFrame) -> Path:
        path = self._frame_path(name)
        df.to_parquet(path, index=False)
        return path

    def get_or_build_frame(self, name: str, builder: Callable[[], pd.DataFrame]) -> tuple[pd.DataFrame, bool]:
        cached = self.load_frame(name)
        if cached is not None:
            return cached, True
        built = builder()
        self.save_frame(name, built)
        return built, False

    def persist_universe_panel(
        self,
        *,
        tickers: list[str],
        membership_ranges: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] | None,
        start_date: str,
        end_date: str,
    ) -> Path:
        rows: list[dict[str, object]] = []
        if membership_ranges:
            for ticker, ranges in sorted(membership_ranges.items()):
                for start_ts, end_ts in ranges:
                    rows.append(
                        {
                            "ticker": str(ticker),
                            "effective_start": pd.Timestamp(start_ts).normalize(),
                            "effective_end": pd.Timestamp(end_ts).normalize(),
                        }
                    )
        else:
            start_ts = pd.Timestamp(start_date).normalize()
            end_ts = pd.Timestamp(end_date).normalize()
            for ticker in sorted(str(t).upper() for t in tickers):
                rows.append(
                    {
                        "ticker": str(ticker),
                        "effective_start": start_ts,
                        "effective_end": end_ts,
                    }
                )
        panel = pd.DataFrame(rows)
        return self.save_frame("universe_panel", panel)


class TimingLedger:
    """Append-only JSONL ledger for long-running research phases."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def record(self, event: str, **fields: Any) -> None:
        payload = {
            "event": str(event),
            "ts_epoch": time.time(),
            **{str(k): _jsonable(v) for k, v in fields.items()},
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
