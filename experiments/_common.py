"""Shared boilerplate for experiments. Keeps each exp file readable."""
from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when running experiments as scripts.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
