"""
Strategy layer — signal filtering, cross-sectional ranking, top-N candidates.
"""

from .engine import StrategyEngine
from .cross_sectional import build_cross_sectional_candidates

__all__ = ["StrategyEngine", "build_cross_sectional_candidates"]
