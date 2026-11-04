"""Test bootstrap: these experiment modules live in this directory, so the
package must be importable from it regardless of pytest's rootdir."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))