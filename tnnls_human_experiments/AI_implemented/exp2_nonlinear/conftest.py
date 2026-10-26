"""Pytest bootstrap: expose this directory's flat modules to test imports."""

import sys
from pathlib import Path

_DIRECTORY = str(Path(__file__).resolve().parent)
if _DIRECTORY not in sys.path:
    sys.path.insert(0, _DIRECTORY)