"""
Vercel Serverless Function Entrypoint for AgriPrice-AI.
Exports the FastAPI application for @vercel/python runtime.
"""

import sys
from pathlib import Path

# Add src and project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

for p in [str(SRC_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from agriprice.api.main import app
