"""Shared config extraction utilities for the model_selection research stack.

Extracted from duplicated _get_*_config helpers across:
  - ic_diagnostics_engine.py:107  (_get_ic_config)
  - conditional_alpha_engine.py:148 (_get_config)
  - feature_diversity_engine.py:158 (_get_config)
  - pit_condition_engine.py:131 (_get_config)
  - signal_decay_engine.py:206 (_get_decay_config)

All follow the same pattern:
    defaults = _DEFAULT_CONFIG[section]
    user = (cfg.get("model_selection", {}) or {}).get(section, {})
    if not user:
        user = (cfg.get(section, {}) or {})
    merged = dict(defaults)
    for k, v in user.items():
        merged[k] = v
    return merged

signal_decay_engine has an extra nested-dict merge for "smoothing" which is
handled by the optional `nested_keys` parameter.
"""
from __future__ import annotations

from typing import Any


def merge_config(
    cfg: dict[str, Any],
    section: str,
    defaults: dict[str, Any],
    *,
    nested_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Extract and merge user config for a given section.

    Precedence (highest to lowest):
      1. cfg["model_selection"][section]
      2. cfg[section]
      3. defaults

    Args:
        cfg: Full configuration dict (e.g. from YAML).
        section: Top-level key for this engine's config (e.g. "ic_diagnostics").
        defaults: Default values for this section.
        nested_keys: Keys whose values are dicts that should be deep-merged
            rather than replaced. Used by signal_decay_engine for "smoothing".

    Returns:
        Merged config dict.
    """
    user = (cfg.get("model_selection", {}) or {}).get(section, {})
    if not user:
        user = (cfg.get(section, {}) or {})
    merged = dict(defaults)
    for k, v in user.items():
        if k in nested_keys and isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged
