"""
Training script: trains Multi-Horizon Quantile XGBoost models and stores them in models_store/.
"""

import sys
from pathlib import Path

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agriprice.config.settings import settings, MODEL_DIR
from agriprice.data.loader import get_or_load_data
from agriprice.preprocessing.cleaning import detect_and_clean_outliers, impute_missing_trading_days
from agriprice.features.feature_engineering import create_agricultural_features
from agriprice.models.ml_regressor import AgriculturalXGBoostForecaster
from agriprice.utils.logger import get_logger

logger = get_logger("train_script")


def train_all_models():
    logger.info("Starting model training pipeline...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df_prices, df_weather = get_or_load_data()

    # Get unique commodity-mandi combinations available in the dataset
    pairs = df_prices[["commodity", "mandi_name"]].drop_duplicates().values
    logger.info("Found %d commodity-mandi series to train.", len(pairs))

    trained_count = 0
    for commodity, mandi in pairs:
        subset = df_prices[
            (df_prices["commodity"] == commodity) & (df_prices["mandi_name"] == mandi)
        ].copy()

        if len(subset) < 45:
            continue

        cleaned, _ = detect_and_clean_outliers(subset)
        daily = impute_missing_trading_days(cleaned)
        features = create_agricultural_features(daily, df_weather, horizons=settings.FORECAST_HORIZONS)

        forecaster = AgriculturalXGBoostForecaster(horizons=settings.FORECAST_HORIZONS)
        forecaster.train(features, test_days=30)
        saved_path = forecaster.save(commodity, mandi)
        trained_count += 1
        logger.info("Trained & saved: %s", saved_path.name)

    logger.info("Completed training %d models. Saved in %s", trained_count, MODEL_DIR)


if __name__ == "__main__":
    train_all_models()
