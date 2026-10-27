"""
Backward-compatibility wrapper for models/evaluator.py.
Redirects to src/agriprice/models/evaluator.py.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

for p in [str(SRC_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from agriprice.models.evaluator import *
