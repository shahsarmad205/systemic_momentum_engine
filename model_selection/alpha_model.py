from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AlphaModel(Protocol):
    """Pure alpha contract.

    Alpha models may learn cross-sectional scores from features and alpha labels.
    They must not consume portfolio state, implementation costs, constraints,
    target weights, previous holdings, or turnover penalties during fitting.
    """

    def fit(self, X: np.ndarray, y_alpha: np.ndarray) -> "AlphaModel":
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class AlphaModelSpec:
    """Registry metadata for a pure alpha model."""

    name: str
    model: AlphaModel
    uses_proba: bool
    model_kind: str


FORBIDDEN_ALPHA_FIT_KWARGS: frozenset[str] = frozenset(
    {
        "_cost",
        "cost",
        "costs",
        "target_expected_cost",
        "turnover_penalty",
        "cost_penalty",
        "target_weights",
        "previous_weights",
        "portfolio_weights",
        "constraints",
    }
)


def assert_pure_alpha_fit_kwargs(kwargs: dict[str, object]) -> None:
    """Fail closed if portfolio-construction state is passed to model fitting."""

    bad = sorted(k for k in kwargs if k in FORBIDDEN_ALPHA_FIT_KWARGS)
    if bad:
        raise ValueError(
            "AlphaModel.fit received portfolio-construction fields: "
            + ", ".join(bad)
        )
