"""
Launcher for AgriPrice-AI Streamlit Dashboard.
Redirects to src/agriprice/dashboard/app.py while ensuring proper module resolution.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

for p in [str(SRC_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Run dashboard module directly
with open(SRC_DIR / "agriprice" / "dashboard" / "app.py", "r", encoding="utf-8") as f:
    code = f.read()
    exec(code, globals())
