"""
Vercel Serverless Function Entrypoint for AgriPrice-AI.
Exports the FastAPI application for @vercel/python runtime.
"""
import sys
from pathlib import Path

# Add project root to sys.path so config, models, and preprocessing are resolvable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.main import app
