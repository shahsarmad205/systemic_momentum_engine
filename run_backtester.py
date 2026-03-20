"""
Convenience wrapper: run the full backtest using the default YAML config.
"""

from __future__ import annotations

import runpy
import sys


def main() -> None:
    # Execute the actual script in-process so argparse/prints behave normally.
    runpy.run_path("run_backtest.py", run_name="__main__")


if __name__ == "__main__":
    sys.exit(main())

