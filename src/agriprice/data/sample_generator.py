"""
Generates realistic multi-year Indian agricultural mandi datasets for benchmarking and development.
Simulates seasonality, weather shocks, arrival gluts, and data anomalies.
All records produced by this generator are explicitly labeled data_source='SIMULATED'.
"""

import random
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from agriprice.config.settings import settings, DATA_DIR
from agriprice.data.db_manager import (
    init_db, SessionLocal, Commodity, Mandi, MandiPriceDaily, WeatherDaily, HAS_SQLALCHEMY
)
from agriprice.utils.logger import get_logger

logger = get_logger("sample_generator")


def generate_agricultural_dataset(start_date: str = "2023-01-01", end_date: str = "2026-09-25") -> None:
    logger.info("Generating realistic Agmarknet & NASA-POWER agricultural dataset (SIMULATED)...")
    np.random.seed(42)
    random.seed(42)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    commodity_specs: Dict[str, Dict[str, Any]] = {
        "Onion": {
            "category": "Perishable", "base_price": 1800.0, "volatility": 0.35, "msp": None,
            "peak_months": [4, 5, 11, 12], "price_surge_months": [9, 10],
        },
        "Tomato": {
            "category": "Perishable", "base_price": 1500.0, "volatility": 0.45, "msp": None,
            "peak_months": [1, 2, 8, 9], "price_surge_months": [6, 7],
        },
        "Potato": {
            "category": "Perishable", "base_price": 1200.0, "volatility": 0.20, "msp": None,
            "peak_months": [2, 3, 4], "price_surge_months": [10, 11],
        },
        "Wheat": {
            "category": "Non-Perishable", "base_price": 2275.0, "volatility": 0.10, "msp": 2275.0,
            "peak_months": [4, 5], "price_surge_months": [1, 2],
        },
        "Soybean": {
            "category": "Non-Perishable", "base_price": 4600.0, "volatility": 0.15, "msp": 4600.0,
            "peak_months": [10, 11], "price_surge_months": [6, 7],
        },
        "Mustard": {
            "category": "Non-Perishable", "base_price": 5650.0, "volatility": 0.16, "msp": 5650.0,
            "peak_months": [3, 4], "price_surge_months": [9, 10],
        },
        "Rice": {
            "category": "Non-Perishable", "base_price": 2300.0, "volatility": 0.12, "msp": 2300.0,
            "peak_months": [10, 11, 12], "price_surge_months": [6, 7],
        },
        "Cotton": {
            "category": "Commercial", "base_price": 7120.0, "volatility": 0.18, "msp": 7120.0,
            "peak_months": [11, 12, 1], "price_surge_months": [5, 6],
        },
        "Maize": {
            "category": "Non-Perishable", "base_price": 2225.0, "volatility": 0.14, "msp": 2225.0,
            "peak_months": [10, 11, 3], "price_surge_months": [6, 7],
        },
        "Chilli": {
            "category": "Spice", "base_price": 16500.0, "volatility": 0.28, "msp": None,
            "peak_months": [2, 3, 4], "price_surge_months": [8, 9],
        },
        "Turmeric": {
            "category": "Spice", "base_price": 12800.0, "volatility": 0.25, "msp": None,
            "peak_months": [2, 3, 4], "price_surge_months": [9, 10],
        },
        "Gram": {
            "category": "Non-Perishable", "base_price": 5440.0, "volatility": 0.14, "msp": 5440.0,
            "peak_months": [3, 4], "price_surge_months": [8, 9],
        },
        "Groundnut": {
            "category": "Commercial", "base_price": 6780.0, "volatility": 0.16, "msp": 6783.0,
            "peak_months": [10, 11, 12], "price_surge_months": [6, 7],
        },
        "Moong": {
            "category": "Non-Perishable", "base_price": 8680.0, "volatility": 0.15, "msp": 8682.0,
            "peak_months": [9, 10], "price_surge_months": [4, 5],
        },
        "Garlic": {
            "category": "Perishable", "base_price": 9500.0, "volatility": 0.38, "msp": None,
            "peak_months": [2, 3, 4], "price_surge_months": [10, 11],
        },
        "Coriander": {
            "category": "Spice", "base_price": 7600.0, "volatility": 0.22, "msp": None,
            "peak_months": [2, 3, 4], "price_surge_months": [9, 10],
        },
        "Bajra": {
            "category": "Non-Perishable", "base_price": 2500.0, "volatility": 0.13, "msp": 2500.0,
            "peak_months": [10, 11], "price_surge_months": [5, 6],
        },
        "Jaggery": {
            "category": "Commercial", "base_price": 3800.0, "volatility": 0.15, "msp": None,
            "peak_months": [12, 1, 2], "price_surge_months": [7, 8],
        }
    }

    weather_records: List[Dict[str, Any]] = []
    for mandi_name, mandi_info in settings.SUPPORTED_MANDIS.items():
        lat = mandi_info["lat"]
        base_temp = 32.0 - (lat - 10.0) * 0.4

        for current_date in dates:
            doy = current_date.dayofyear
            month = current_date.month

            is_monsoon = 6 <= month <= 9
            rain_prob = 0.55 if is_monsoon else 0.05
            rainfall = np.random.exponential(18.0) if random.random() < rain_prob else 0.0

            temp_seasonal = base_temp + 8.0 * np.sin(2.0 * np.pi * (doy - 100) / 365.25)
            temp_max = temp_seasonal + np.random.normal(0, 2.5)
            temp_min = temp_max - np.random.uniform(8.0, 14.0)

            soil_moisture = 70.0 if is_monsoon else 30.0 + (rainfall * 0.8)
            soil_moisture = min(95.0, max(10.0, soil_moisture + np.random.normal(0, 5)))
            ndvi = 0.65 if is_monsoon else 0.35 + np.random.normal(0, 0.04)

            weather_records.append({
                "record_date": current_date.strftime("%Y-%m-%d"),
                "mandi_name": mandi_name,
                "rainfall_mm": round(rainfall, 1),
                "temp_max_c": round(temp_max, 1),
                "temp_min_c": round(temp_min, 1),
                "soil_moisture_pct": round(soil_moisture, 1),
                "ndvi_index": round(ndvi, 3),
                "data_source": "SIMULATED"
            })

    weather_df = pd.DataFrame(weather_records)

    price_records: List[Dict[str, Any]] = []
    for mandi_name, mandi_info in settings.SUPPORTED_MANDIS.items():
        for crop in mandi_info["primary_crops"]:
            if crop not in commodity_specs:
                continue
            spec = commodity_specs[crop]
            base_p = spec["base_price"]

            for current_date in dates:
                if current_date.weekday() == 6:  # Mandi closed on Sundays
                    continue
                if random.random() < 0.03:  # Local holiday
                    continue

                doy = current_date.dayofyear
                month = current_date.month

                if spec["category"] == "Perishable":
                    base_arrival = 1200.0
                elif spec["category"] == "Spice":
                    base_arrival = 450.0
                elif spec["category"] == "Commercial":
                    base_arrival = 1800.0
                else:
                    base_arrival = 2500.0

                if month in spec["peak_months"]:
                    arrival_factor = np.random.uniform(2.2, 4.0)
                elif month in spec["price_surge_months"]:
                    arrival_factor = np.random.uniform(0.3, 0.6)
                else:
                    arrival_factor = np.random.uniform(0.8, 1.4)

                arrivals = max(50.0, base_arrival * arrival_factor + np.random.normal(0, 100))
                elasticity = -0.35 if spec["category"] in ["Perishable", "Spice"] else -0.20
                arrival_effect = (arrivals / base_arrival) ** elasticity

                years_from_start = (current_date - dates[0]).days / 365.25
                inflation_drift = 1.0 + (0.05 * years_from_start)
                season_wave = 1.0 + (0.25 * np.sin(2.0 * np.pi * (doy - 240) / 365.25))
                noise = np.random.normal(1.0, spec["volatility"] * 0.15)

                shock = 1.0
                if crop == "Tomato" and current_date.year == 2023 and 6 <= month <= 8:
                    shock = 2.8
                elif crop == "Onion" and month in [9, 10] and random.random() < 0.15:
                    shock = 1.6

                modal_price = base_p * arrival_effect * season_wave * inflation_drift * noise * shock
                modal_price = max(200.0, modal_price)

                if spec["msp"] and modal_price < spec["msp"] * 0.92:
                    modal_price = spec["msp"] * np.random.uniform(0.95, 1.05)

                min_price = modal_price * np.random.uniform(0.88, 0.96)
                max_price = modal_price * np.random.uniform(1.04, 1.15)

                if random.random() < 0.004:
                    modal_price = modal_price / 10.0
                elif random.random() < 0.003:
                    min_price = max_price * 1.2

                price_records.append({
                    "record_date": current_date.strftime("%Y-%m-%d"),
                    "mandi_name": mandi_name,
                    "commodity": crop,
                    "arrivals_qtl": round(arrivals, 1),
                    "min_price": round(min_price, 2),
                    "max_price": round(max_price, 2),
                    "modal_price": round(modal_price, 2),
                    "data_source": "SIMULATED"
                })

    df_prices = pd.DataFrame(price_records)

    prices_csv_path = DATA_DIR / "sample_mandi_prices.csv"
    weather_csv_path = DATA_DIR / "sample_weather.csv"
    df_prices.to_csv(prices_csv_path, index=False)
    weather_df.to_csv(weather_csv_path, index=False)

    logger.info("Saved %d price records to %s", len(df_prices), prices_csv_path)
    logger.info("Saved %d weather records to %s", len(weather_df), weather_csv_path)


if __name__ == "__main__":
    generate_agricultural_dataset()
