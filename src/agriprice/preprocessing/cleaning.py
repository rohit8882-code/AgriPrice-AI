"""
Agricultural Data Cleaning and Quality Assurance Pipeline.
Handles Mandi closed days, missing values, and typographical entry errors.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def detect_and_clean_outliers(
    df: pd.DataFrame, window: int = 21, z_thresh: float = 3.5
) -> Tuple[pd.DataFrame, int]:
    """
    Cleans pricing data using domain-specific rules and Rolling Median Absolute Deviation (MAD).

    Args:
        df: Input DataFrame with modal_price, min_price, max_price
        window: Rolling window size in days
        z_thresh: Modified Z-Score cutoff threshold

    Returns:
        Tuple of (cleaned_df, number_of_anomalies_detected)
    """
    df = df.copy()

    # 1. Domain Sanity Filter: Modal price must lie between min_price and max_price
    # Swap if accidentally inverted in manual APMC log entry
    if "min_price" in df.columns and "max_price" in df.columns:
        mask_inverted = df["min_price"] > df["max_price"]
        df.loc[mask_inverted, ["min_price", "max_price"]] = df.loc[
            mask_inverted, ["max_price", "min_price"]
        ].values

        # Re-align modal price if it falls outside [min_price, max_price] due to single-digit typo
        mask_out_of_bounds = (df["modal_price"] < df["min_price"]) | (
            df["modal_price"] > df["max_price"]
        )
        df.loc[mask_out_of_bounds, "modal_price"] = (
            df.loc[mask_out_of_bounds, "min_price"] + df.loc[mask_out_of_bounds, "max_price"]
        ) / 2.0

    # 2. Rolling Modified Z-score via MAD (Median Absolute Deviation)
    rolling_median = df["modal_price"].rolling(window=window, center=True, min_periods=3).median()
    rolling_mad = (
        (df["modal_price"] - rolling_median)
        .abs()
        .rolling(window=window, center=True, min_periods=3)
        .median()
    )

    # Avoid zero division
    rolling_mad = rolling_mad.replace(0, np.nan).bfill().ffill()
    fallback_std = float(df["modal_price"].std() or 1.0)
    rolling_mad = rolling_mad.fillna(fallback_std)

    modified_z = (0.6745 * (df["modal_price"] - rolling_median).abs()) / rolling_mad

    outlier_mask = modified_z > z_thresh
    anomalies_detected = int(outlier_mask.sum())

    # Impute outliers with the rolling median rather than dropping to preserve continuity
    df.loc[outlier_mask, "modal_price"] = rolling_median[outlier_mask]

    return df, anomalies_detected


def impute_missing_trading_days(
    df: pd.DataFrame, date_col: str = "record_date"
) -> pd.DataFrame:
    """
    Reindexes a mandi commodity series to a daily calendar.
    Fills closed days (Sundays/holidays) via Forward-Fill for price and 0 for arrivals.

    Args:
        df: Input DataFrame with timestamp/date column
        date_col: Name of date column

    Returns:
        Daily contiguous DataFrame
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)

    if df.empty:
        return df

    # Create complete date range
    full_idx = pd.date_range(start=df[date_col].min(), end=df[date_col].max(), freq="D")
    df = df.set_index(date_col).reindex(full_idx)
    df.index.name = date_col

    # Forward-fill prices for up to 4 consecutive closed days
    price_cols = ["modal_price", "min_price", "max_price"]
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].ffill(limit=4).bfill()

    # Closed days have 0 physical market arrivals
    if "arrivals_qtl" in df.columns:
        df["arrivals_qtl"] = df["arrivals_qtl"].fillna(0.0)

    # Metadata columns forward filled
    meta_cols = ["mandi_name", "commodity", "state", "district", "data_source"]
    for col in meta_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    return df.reset_index()
