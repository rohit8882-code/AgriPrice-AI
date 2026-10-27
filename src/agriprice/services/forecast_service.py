"""
Forecast service module: coordinates multi-horizon inference, caching, and model persistence.
"""

from datetime import timedelta, timezone, datetime
from typing import Dict, Any, Tuple
import pandas as pd

from agriprice.config.settings import settings, MODEL_DIR, LEGACY_MODEL_DIR
from agriprice.data.loader import get_or_load_data
from agriprice.preprocessing.cleaning import detect_and_clean_outliers, impute_missing_trading_days
from agriprice.features.feature_engineering import create_agricultural_features
from agriprice.models.ml_regressor import AgriculturalXGBoostForecaster
from agriprice.utils.logger import get_logger

logger = get_logger("forecast_service")

# In-memory model cache
_MODEL_CACHE: Dict[str, AgriculturalXGBoostForecaster] = {}


def get_forecast_for_commodity_mandi(
    commodity: str,
    mandi_name: str
) -> Dict[str, Any]:
    """
    Computes multi-horizon price forecast with confidence intervals and advisory.

    Args:
        commodity: Target commodity name
        mandi_name: Target APMC mandi name

    Returns:
        Structured forecast dictionary ready for API serialization.
    """
    df_prices, df_weather = get_or_load_data()

    mandi_crop_df = df_prices[
        (df_prices["commodity"].str.lower() == commodity.lower()) &
        (df_prices["mandi_name"] == mandi_name)
    ].copy()

    if len(mandi_crop_df) < 30:
        raise ValueError(f"Insufficient historical data ({len(mandi_crop_df)} records) for {commodity} in {mandi_name}.")

    data_source = str(mandi_crop_df["data_source"].iloc[0]) if "data_source" in mandi_crop_df.columns else "SIMULATED"

    # 1. Clean outliers and impute closed days
    cleaned_df, _ = detect_and_clean_outliers(mandi_crop_df)
    daily_df = impute_missing_trading_days(cleaned_df)

    # 2. Extract features
    feat_df = create_agricultural_features(daily_df, df_weather, horizons=settings.FORECAST_HORIZONS)

    # 3. Model retrieval / training
    cache_key = f"{commodity}_{mandi_name}"
    safe_name = f"{commodity}_{mandi_name}".replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    primary_model_path = MODEL_DIR / f"xgb_{safe_name}.pkl"
    legacy_model_path = LEGACY_MODEL_DIR / f"xgb_{safe_name}.pkl"

    if cache_key in _MODEL_CACHE:
        forecaster = _MODEL_CACHE[cache_key]
    elif primary_model_path.exists():
        forecaster = AgriculturalXGBoostForecaster(horizons=settings.FORECAST_HORIZONS)
        forecaster.load(primary_model_path)
        _MODEL_CACHE[cache_key] = forecaster
    elif legacy_model_path.exists():
        forecaster = AgriculturalXGBoostForecaster(horizons=settings.FORECAST_HORIZONS)
        forecaster.load(legacy_model_path)
        _MODEL_CACHE[cache_key] = forecaster
    else:
        forecaster = AgriculturalXGBoostForecaster(horizons=settings.FORECAST_HORIZONS)
        forecaster.train(feat_df, test_days=30)
        forecaster.save(commodity, mandi_name)
        _MODEL_CACHE[cache_key] = forecaster

    latest_row = feat_df.iloc[[-1]]
    latest_date = latest_row["record_date"].iloc[0].date()
    current_price = float(latest_row["modal_price"].iloc[0])

    preds = forecaster.predict_latest(latest_row)
    importances = forecaster.get_feature_importances(horizon=7, top_n=5)

    horizon_results: Dict[str, Dict[str, Any]] = {}
    for h_key, pred_data in preds.items():
        h_days = pred_data["horizon_days"]
        target_date = latest_date + timedelta(days=h_days)
        p50 = pred_data["p50"]

        direction = "NEUTRAL"
        if p50 > current_price * 1.03:
            direction = "BULLISH"
        elif p50 < current_price * 0.97:
            direction = "BEARISH"

        horizon_results[h_key] = {
            "horizon_days": h_days,
            "target_date": target_date,
            "predicted_p10": pred_data["p10"],
            "predicted_p50": p50,
            "predicted_p90": pred_data["p90"],
            "trend_direction": direction
        }

    p50_15d = horizon_results.get("15d", horizon_results.get("7d"))["predicted_p50"]
    pct_change_15d = ((p50_15d - current_price) / current_price) * 100.0

    if pct_change_15d > 5.0:
        advisory = (
            f"HOLD GUIDANCE: Prices projected to surge by {pct_change_15d:.1f}% over the next 15 days "
            f"due to tightening arrivals. Recommended to utilize warehouse storage."
        )
    elif pct_change_15d < -5.0:
        advisory = (
            f"SELL GUIDANCE: Prices expected to drop by {abs(pct_change_15d):.1f}% as peak harvest "
            f"arrivals arrive. Recommended to liquidate stock."
        )
    else:
        advisory = "STABLE MARKET: Minimal price shift (<3%) expected. Standard spot selling advised."

    state_name = settings.SUPPORTED_MANDIS.get(mandi_name, {}).get("state", "India")

    return {
        "commodity": commodity,
        "mandi_name": mandi_name,
        "state": state_name,
        "last_reported_date": latest_date,
        "current_modal_price": round(current_price, 2),
        "unit": "INR / Quintal",
        "forecasts": horizon_results,
        "top_driving_factors": [
            {"feature_name": f[0].replace("_", " ").title(), "relative_importance_pct": f[1]}
            for f in importances
        ],
        "advisory_recommendation": advisory,
        "data_source": data_source
    }
