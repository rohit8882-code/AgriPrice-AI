"""
Configuration settings for AgriPrice-AI platform.
Loads environment variables, defines paths, and manages runtime defaults.
"""

from pathlib import Path
from typing import Dict, List, Any
import os

# Base directory paths
CONFIG_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = CONFIG_DIR.parent
SRC_DIR = PACKAGE_DIR.parent
BASE_DIR = SRC_DIR.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REFERENCE_DATA_DIR = DATA_DIR / "reference"
MODEL_DIR = BASE_DIR / "models_store"
LEGACY_MODEL_DIR = BASE_DIR / "saved_models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist safely
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, REFERENCE_DATA_DIR, MODEL_DIR, REPORTS_DIR]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# Fallback lists if reference data is still building
DEFAULT_COMMODITIES: List[str] = [
    "Onion", "Tomato", "Potato", "Wheat", "Soybean", "Mustard",
    "Rice", "Cotton", "Maize", "Chilli", "Turmeric", "Gram",
    "Groundnut", "Moong", "Garlic", "Coriander", "Bajra", "Jaggery"
]

DEFAULT_MANDIS: Dict[str, Dict[str, Any]] = {
    "Lasalgaon (Nashik, MH)": {
        "state": "Maharashtra", "district": "Nashik", "lat": 20.1472, "lon": 74.2272,
        "primary_crops": ["Onion", "Soybean", "Cotton", "Jaggery"]
    },
    "Pimpalgaon (Nashik, MH)": {
        "state": "Maharashtra", "district": "Nashik", "lat": 20.1700, "lon": 73.9800,
        "primary_crops": ["Onion", "Tomato", "Garlic"]
    },
    "Azadpur (Delhi, DL)": {
        "state": "Delhi", "district": "North Delhi", "lat": 28.7161, "lon": 77.1727,
        "primary_crops": ["Onion", "Tomato", "Potato", "Wheat", "Garlic", "Moong"]
    },
    "Kolar (Kolar, KA)": {
        "state": "Karnataka", "district": "Kolar", "lat": 13.1378, "lon": 78.1292,
        "primary_crops": ["Tomato", "Maize", "Chilli", "Groundnut"]
    },
    "Indore (Indore, MP)": {
        "state": "Madhya Pradesh", "district": "Indore", "lat": 22.7196, "lon": 75.8577,
        "primary_crops": ["Soybean", "Wheat", "Potato", "Gram", "Garlic", "Coriander", "Moong"]
    },
    "Khanna (Ludhiana, PB)": {
        "state": "Punjab", "district": "Ludhiana", "lat": 30.7068, "lon": 76.2166,
        "primary_crops": ["Wheat", "Rice", "Jaggery", "Bajra"]
    },
    "Alwar (Alwar, RJ)": {
        "state": "Rajasthan", "district": "Alwar", "lat": 27.5530, "lon": 76.6346,
        "primary_crops": ["Mustard", "Onion", "Gram", "Bajra", "Coriander", "Moong"]
    },
    "Rajkot (Rajkot, GJ)": {
        "state": "Gujarat", "district": "Rajkot", "lat": 22.3039, "lon": 70.8022,
        "primary_crops": ["Cotton", "Gram", "Wheat", "Groundnut", "Garlic"]
    },
    "Guntur (Guntur, AP)": {
        "state": "Andhra Pradesh", "district": "Guntur", "lat": 16.3067, "lon": 80.4365,
        "primary_crops": ["Chilli", "Cotton", "Rice", "Coriander"]
    },
    "Agra (Agra, UP)": {
        "state": "Uttar Pradesh", "district": "Agra", "lat": 27.1767, "lon": 78.0081,
        "primary_crops": ["Potato", "Mustard", "Wheat", "Bajra", "Jaggery"]
    },
    "Burdwan (Purba Bardhaman, WB)": {
        "state": "West Bengal", "district": "Purba Bardhaman", "lat": 23.2324, "lon": 87.8615,
        "primary_crops": ["Rice", "Potato", "Groundnut"]
    },
    "Nizamabad (Nizamabad, TS)": {
        "state": "Telangana", "district": "Nizamabad", "lat": 18.6725, "lon": 78.0941,
        "primary_crops": ["Turmeric", "Maize", "Soybean", "Chilli"]
    },
    "Karnal (Karnal, HR)": {
        "state": "Haryana", "district": "Karnal", "lat": 29.6857, "lon": 76.9905,
        "primary_crops": ["Rice", "Wheat", "Mustard", "Bajra", "Moong"]
    },
    "Gulabbagh (Purnia, BR)": {
        "state": "Bihar", "district": "Purnia", "lat": 25.7771, "lon": 87.4753,
        "primary_crops": ["Maize", "Rice", "Wheat"]
    },
    "Dindigul (Dindigul, TN)": {
        "state": "Tamil Nadu", "district": "Dindigul", "lat": 10.3673, "lon": 77.9803,
        "primary_crops": ["Tomato", "Onion", "Groundnut"]
    },
    "Guwahati (Kamrup, AS)": {
        "state": "Assam", "district": "Kamrup", "lat": 26.1445, "lon": 91.7362,
        "primary_crops": ["Rice", "Mustard", "Potato", "Turmeric"]
    },
    "Cuttack (Cuttack, OD)": {
        "state": "Odisha", "district": "Cuttack", "lat": 20.4625, "lon": 85.8830,
        "primary_crops": ["Rice", "Groundnut", "Moong", "Chilli"]
    },
    "Kochi (Ernakulam, KL)": {
        "state": "Kerala", "district": "Ernakulam", "lat": 9.9312, "lon": 76.2673,
        "primary_crops": ["Rice", "Turmeric", "Chilli"]
    },
    "Raipur (Raipur, CG)": {
        "state": "Chhattisgarh", "district": "Raipur", "lat": 21.2514, "lon": 81.6296,
        "primary_crops": ["Rice", "Maize", "Soybean", "Gram"]
    },
    "Ranchi (Ranchi, JH)": {
        "state": "Jharkhand", "district": "Ranchi", "lat": 23.3441, "lon": 85.3096,
        "primary_crops": ["Potato", "Tomato", "Rice", "Maize"]
    },
    "Shimla (Shimla, HP)": {
        "state": "Himachal Pradesh", "district": "Shimla", "lat": 31.1048, "lon": 77.1734,
        "primary_crops": ["Potato", "Tomato", "Garlic", "Maize"]
    },
    "Haldwani (Nainital, UK)": {
        "state": "Uttarakhand", "district": "Nainital", "lat": 29.2183, "lon": 79.5130,
        "primary_crops": ["Wheat", "Rice", "Tomato", "Potato", "Mustard"]
    },
    "Margao (South Goa, GA)": {
        "state": "Goa", "district": "South Goa", "lat": 15.2832, "lon": 73.9862,
        "primary_crops": ["Rice", "Groundnut", "Chilli"]
    },
    "Agartala (West Tripura, TR)": {
        "state": "Tripura", "district": "West Tripura", "lat": 23.8315, "lon": 91.2868,
        "primary_crops": ["Rice", "Potato", "Chilli", "Turmeric"]
    },
    "Shillong (East Khasi Hills, ML)": {
        "state": "Meghalaya", "district": "East Khasi Hills", "lat": 25.5788, "lon": 91.8933,
        "primary_crops": ["Potato", "Tomato", "Maize", "Turmeric"]
    },
    "Imphal (Imphal West, MN)": {
        "state": "Manipur", "district": "Imphal West", "lat": 24.8170, "lon": 93.9368,
        "primary_crops": ["Rice", "Potato", "Chilli", "Maize"]
    },
    "Dimapur (Dimapur, NL)": {
        "state": "Nagaland", "district": "Dimapur", "lat": 25.9090, "lon": 93.7266,
        "primary_crops": ["Maize", "Rice", "Chilli", "Turmeric"]
    },
    "Naharlagun (Papum Pare, AR)": {
        "state": "Arunachal Pradesh", "district": "Papum Pare", "lat": 27.1064, "lon": 93.6934,
        "primary_crops": ["Rice", "Maize", "Mustard", "Potato"]
    },
    "Aizawl (Aizawl, MZ)": {
        "state": "Mizoram", "district": "Aizawl", "lat": 23.7271, "lon": 92.7176,
        "primary_crops": ["Maize", "Rice", "Chilli", "Turmeric"]
    },
    "Gangtok (East Sikkim, SK)": {
        "state": "Sikkim", "district": "East Sikkim", "lat": 27.3389, "lon": 88.6065,
        "primary_crops": ["Maize", "Potato", "Coriander", "Turmeric"]
    },
    "Srinagar (Srinagar, JK)": {
        "state": "Jammu and Kashmir", "district": "Srinagar", "lat": 34.0837, "lon": 74.7973,
        "primary_crops": ["Potato", "Garlic", "Mustard", "Tomato"]
    }
}

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class SettingsConfig(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=str(BASE_DIR / ".env"),
            env_file_encoding="utf-8",
            extra="ignore"
        )

        PROJECT_NAME: str = "AgriPrice-AI"
        VERSION: str = "1.0.0"
        API_V1_PREFIX: str = "/v1"
        DATABASE_URL: str = f"sqlite:///{BASE_DIR}/agriprice.db"
        POSTGRES_TIMESCALE_URL: str = "postgresql://postgres:postgres@localhost:5432/agriprice"
        DATA_GOV_API_KEY: str = "DEMO_KEY"
        DATA_GOV_IN_API_KEY: str = "DEMO_KEY"
        TELEGRAM_BOT_TOKEN: str = ""
        LOG_LEVEL: str = "INFO"
        NASA_POWER_BASE_URL: str = "https://power.larc.nasa.gov/api/temporal/daily/point"
        AGMARKNET_BASE_URL: str = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        FORECAST_HORIZONS: List[int] = [7, 15, 30]
        SUPPORTED_COMMODITIES: List[str] = DEFAULT_COMMODITIES
        SUPPORTED_MANDIS: Dict[str, Dict[str, Any]] = DEFAULT_MANDIS

    settings = SettingsConfig()

except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class Settings:
        PROJECT_NAME: str = "AgriPrice-AI"
        VERSION: str = "1.0.0"
        API_V1_PREFIX: str = "/v1"
        DATABASE_URL: str = f"sqlite:///{BASE_DIR}/agriprice.db"
        POSTGRES_TIMESCALE_URL: str = "postgresql://postgres:postgres@localhost:5432/agriprice"
        DATA_GOV_API_KEY: str = os.getenv("DATA_GOV_API_KEY", "DEMO_KEY")
        DATA_GOV_IN_API_KEY: str = os.getenv("DATA_GOV_IN_API_KEY", "DEMO_KEY")
        TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
        LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        NASA_POWER_BASE_URL: str = "https://power.larc.nasa.gov/api/temporal/daily/point"
        AGMARKNET_BASE_URL: str = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        FORECAST_HORIZONS: List[int] = field(default_factory=lambda: [7, 15, 30])
        SUPPORTED_COMMODITIES: List[str] = field(default_factory=lambda: list(DEFAULT_COMMODITIES))
        SUPPORTED_MANDIS: Dict[str, Dict[str, Any]] = field(default_factory=lambda: dict(DEFAULT_MANDIS))

    settings = Settings()
