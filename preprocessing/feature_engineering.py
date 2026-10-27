"""
Feature Engineering Pipeline for Agricultural Commodities.
Extracts temporal, rolling, cyclical, arrival-elasticity, and meteorological features.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple

def create_agricultural_features(
    df_prices: pd.DataFrame,
    df_weather: pd.DataFrame = None,
    horizons: List[int] = [7, 15, 30]
) -> pd.DataFrame:
    """
    Constructs a rich tabular feature set for a single mandi-commodity time series.
    Expects df_prices sorted by record_date with continuous daily observations.
    """
    df = df_prices.copy()
    df["record_date"] = pd.to_datetime(df["record_date"])
    df = df.sort_values(by="record_date").reset_index(drop=True)

    # 1. Autoregressive Lags (Prices)
    lags = [1, 2, 3, 7, 14, 21, 30, 60]
    for lag in lags:
        df[f"modal_price_lag_{lag}"] = df["modal_price"].shift(lag)
        df[f"arrivals_lag_{lag}"] = df["arrivals_qtl"].shift(lag)

    # 2. Rolling Momentum & Volatility
    for window in [7, 14, 30]:
        df[f"price_roll_mean_{window}"] = df["modal_price"].shift(1).rolling(window=window).mean()
        df[f"price_roll_std_{window}"] = df["modal_price"].shift(1).rolling(window=window).std()
        df[f"price_roll_min_{window}"] = df["modal_price"].shift(1).rolling(window=window).min()
        df[f"price_roll_max_{window}"] = df["modal_price"].shift(1).rolling(window=window).max()
        df[f"arrivals_roll_mean_{window}"] = df["arrivals_qtl"].shift(1).rolling(window=window).mean()

    # 3. Supply-Price Interaction & Elasticity Ratios
    # Ratio of current 7-day arrivals to 30-day baseline arrivals (Glut vs Scarcity signal)
    df["arrival_glut_ratio"] = (
        df["arrivals_roll_mean_7"] / (df["arrivals_roll_mean_30"] + 1e-4)
    )
    # Price spread (Intraday volatility ratio)
    if "min_price" in df.columns and "max_price" in df.columns:
        df["price_spread"] = (df["max_price"] - df["min_price"]) / (df["modal_price"] + 1e-4)

    # 4. Cyclical Calendar Features (Fourier Encodings)
    doy = df["record_date"].dt.dayofyear
    month = df["record_date"].dt.month
    dow = df["record_date"].dt.dayofweek

    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    df["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * month / 12.0)
    df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)

    # 5. Agricultural Seasons in India
    # Kharif harvest: Oct - Dec (Months 10, 11, 12)
    df["is_kharif_harvest"] = month.isin([10, 11, 12]).astype(int)
    # Rabi harvest: Mar - May (Months 3, 4, 5)
    df["is_rabi_harvest"] = month.isin([3, 4, 5]).astype(int)
    # Lean summer period: June - July
    df["is_lean_period"] = month.isin([6, 7]).astype(int)

    # 6. Weather Integration (if provided)
    if df_weather is not None and not df_weather.empty:
        w_df = df_weather.copy()
        w_df["record_date"] = pd.to_datetime(w_df["record_date"])
        # Merge on record_date
        merge_cols = ["record_date"]
        if "mandi_name" in w_df.columns and "mandi_name" in df.columns:
            merge_cols.append("mandi_name")
        df = pd.merge(df, w_df, on=merge_cols, how="left")
        
        # Cumulative precipitation & temperature shocks
        if "rainfall_mm" in df.columns:
            df["rainfall_mm"] = df["rainfall_mm"].fillna(0.0)
            df["rain_cum_3d"] = df["rainfall_mm"].shift(1).rolling(3).sum()
            df["rain_cum_7d"] = df["rainfall_mm"].shift(1).rolling(7).sum()
            # Unseasonal rain indicator (> 25mm in non-monsoon months)
            df["unseasonal_rain_shock"] = (
                (df["rainfall_mm"] > 25.0) & (~month.isin([6, 7, 8, 9]))
            ).astype(int)

        if "temp_max_c" in df.columns:
            df["temp_max_c"] = df["temp_max_c"].ffill().bfill()
            df["temp_shock_heatwave"] = (df["temp_max_c"] > 40.0).astype(int)

    # 7. Multi-Horizon Targets for Supervised Learning
    for h in horizons:
        df[f"target_price_{h}d"] = df["modal_price"].shift(-h)

    return df
