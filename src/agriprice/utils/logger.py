"""
Logging utilities for AgriPrice-AI.
Provides centralized structured logging replacing ad-hoc print statements.
"""

import logging
import os
import sys

def get_logger(name: str = "agriprice") -> logging.Logger:
    """
    Returns a configured logger instance with standardized formatting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_str, logging.INFO)
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
