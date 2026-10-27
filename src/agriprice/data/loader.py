"""
Data loading and caching service for prices, weather, and reference tables.
"""

from typing import Tuple, Dict
import pandas as pd

from agriprice.config.settings import DATA_DIR
from agriprice.data.sample_generator import generate_agricultural_dataset
from agriprice.utils.logger import get_logger

logger = get_logger("data_loader")

_DATA_CACHE: Dict[str, pd.DataFrame] = {}


def get_or_load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads cleaned prices and weather data from disk cache, generating fallback if needed.

    Returns:
        Tuple of (df_prices, df_weather) with parsed datetimes.
    """
    prices_path = DATA_DIR / "sample_mandi_prices.csv"
    weather_path = DATA_DIR / "sample_weather.csv"

    if not prices_path.exists() or not weather_path.exists():
        logger.info("Local datasets missing. Triggering automated dataset generator...")
        generate_agricultural_dataset()

    if "prices" not in _DATA_CACHE:
        df_p = pd.read_csv(prices_path)
        df_w = pd.read_csv(weather_path)
        df_p["record_date"] = pd.to_datetime(df_p["record_date"])
        df_w["record_date"] = pd.to_datetime(df_w["record_date"])
        if "data_source" not in df_p.columns:
            df_p["data_source"] = "SIMULATED"
        if "data_source" not in df_w.columns:
            df_w["data_source"] = "SIMULATED"
        _DATA_CACHE["prices"] = df_p
        _DATA_CACHE["weather"] = df_w

    return _DATA_CACHE["prices"], _DATA_CACHE["weather"]


def clear_cache() -> None:
    """Clears in-memory data cache."""
    _DATA_CACHE.clear()
