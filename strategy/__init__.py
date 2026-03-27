"""
Strategy layer — signal filtering, cross-sectional ranking, top-N candidates.
"""

from .cross_sectional import build_cross_sectional_candidates
from .engine import StrategyEngine

__all__ = ["StrategyEngine", "build_cross_sectional_candidates"]
