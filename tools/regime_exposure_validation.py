from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = Path(__file__).resolve().parents[1]
for p in (str(PKG_ROOT), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from strategy.regime_exposure import (
    CANONICAL_REGIME_EXPOSURE,
    exposure_path_for_regimes,
    resolve_regime_position_scale,
    validate_regime_keys,
)


def main() -> None:
    print("=== Regime Exposure Validation ===")
    print("Canonical mapping:")
    for k, v in CANONICAL_REGIME_EXPOSURE.items():
        print(f"  {k:9s} -> {v:.2f}")

    # 1) Unknown regime handling
    print("\nUnknown regime handling:")
    print("  Safe default:", resolve_regime_position_scale("Unknown", strict=False))
    try:
        resolve_regime_position_scale("Unknown", strict=True)
    except ValueError as e:
        print("  Strict mode:", str(e))

    # 2) Rapid switching stress
    rapid = ["Bull", "Crisis", "Bull", "Bear", "Crisis", "Sideways"] * 10
    rapid_exp = exposure_path_for_regimes(rapid)
    print("\nRapid switching stress:")
    print("  n_steps:", len(rapid_exp))
    print("  min/max exposure:", min(rapid_exp), "/", max(rapid_exp))
    print("  exposure counts:", dict(Counter(rapid_exp)))

    # 3) Long crisis period stress
    crisis = ["Crisis"] * 200
    crisis_exp = exposure_path_for_regimes(crisis)
    print("\nLong crisis period stress:")
    print("  n_steps:", len(crisis_exp))
    print("  unique exposures:", sorted(set(crisis_exp)))

    # 4) Monotonic check
    ordered = ["Bull", "Sideways", "Bear", "Crisis"]
    path = exposure_path_for_regimes(ordered)
    mono = all(path[i] >= path[i + 1] for i in range(len(path) - 1))
    print("\nMonotonic Bull->Crisis:")
    print("  path:", path)
    print("  monotonic:", mono)

    # 5) key coverage check example
    missing, unexpected = validate_regime_keys(
        {"Bull": {"position_scale": 1.0}, "Bear": {"position_scale": 0.6}, "Alien": {"position_scale": 0.1}}
    )
    print("\nKey coverage example:")
    print("  missing:", sorted(missing))
    print("  unexpected:", sorted(unexpected))


if __name__ == "__main__":
    main()

