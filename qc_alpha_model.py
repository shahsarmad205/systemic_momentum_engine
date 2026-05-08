"""
Compatibility facade for legacy QuantConnect alpha-model imports.

The authoritative LEAN deployment implementation lives in
``LeanCloud/BinaryEdge/qc_alpha_model.py``. This wrapper preserves existing
local parity tooling while removing duplicate deployment logic from the repo
root.
"""

from LeanCloud.BinaryEdge.qc_alpha_model import *  # noqa: F403
