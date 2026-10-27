"""
Configuration package.
"""

from agriprice.config.settings import (
    settings,
    BASE_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    REFERENCE_DATA_DIR,
    MODEL_DIR,
    LEGACY_MODEL_DIR,
    REPORTS_DIR,
)

__all__ = [
    "settings",
    "BASE_DIR",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "REFERENCE_DATA_DIR",
    "MODEL_DIR",
    "LEGACY_MODEL_DIR",
    "REPORTS_DIR",
]
