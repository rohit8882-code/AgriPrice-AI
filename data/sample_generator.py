"""
Backward-compatibility wrapper for data/sample_generator.py.
Redirects to src/agriprice/data/sample_generator.py.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

for p in [str(SRC_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from agriprice.data.sample_generator import *

if __name__ == "__main__":
    generate_agricultural_dataset()
