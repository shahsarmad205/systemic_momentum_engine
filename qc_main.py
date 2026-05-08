"""
Compatibility facade for legacy imports.

QuantConnect deployment source of truth lives under ``LeanCloud/BinaryEdge``.
This module intentionally re-exports the authoritative algorithm so older local
parity scripts keep working while the repo converges on a single deployment path.
"""

from LeanCloud.BinaryEdge.main import TrendSignalAlgorithm

__all__ = ["TrendSignalAlgorithm"]
