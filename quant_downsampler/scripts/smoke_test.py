"""Run the fast acceptance suite used before a full rebuild."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    main()
