"""
Unit and Integration Tests for AgriPrice-AI Data and Feature Pipelines.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from preprocessing.cleaning import detect_and_clean_outliers, impute_missing_trading_days
from preprocessing.feature_engineering import create_agricultural_features
from models.evaluator import calculate_wmape, calculate_mae, calculate_rmse, calculate_mda

def test_outlier_detection_and_typo_fix():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "record_date": dates,
        "mandi_name": ["Lasalgaon"] * 10,
        "commodity": ["Onion"] * 10,
        "arrivals_qtl": [1000.0] * 10,
        "min_price": [1800.0] * 10,
        "max_price": [2200.0] * 10,
        "modal_price": [2000.0] * 10
    })
    
    # Inject typo: modal price typo to ₹200 (missed zero)
    df.loc[5, "modal_price"] = 200.0
    # Inject inverted min/max
    df.loc[3, "min_price"] = 2500.0
    df.loc[3, "max_price"] = 1800.0
    
    cleaned_df, outliers = detect_and_clean_outliers(df)
    
    # Inverted min/max should be swapped back
    assert cleaned_df.loc[3, "min_price"] <= cleaned_df.loc[3, "max_price"]
    # Typo price should be fixed
    assert cleaned_df.loc[5, "modal_price"] > 1000.0

def test_missing_trading_days_imputation():
    # Only Monday, Wednesday, Friday
    dates = [datetime(2024, 1, 1), datetime(2024, 1, 3), datetime(2024, 1, 5)]
    df = pd.DataFrame({
        "record_date": dates,
        "mandi_name": ["Lasalgaon"] * 3,
        "commodity": ["Onion"] * 3,
        "arrivals_qtl": [500.0, 600.0, 700.0],
        "min_price": [1900.0, 1950.0, 2000.0],
        "max_price": [2100.0, 2150.0, 2200.0],
        "modal_price": [2000.0, 2050.0, 2100.0]
    })
    
    imputed = impute_missing_trading_days(df)
    # Should now span all 5 continuous days (Jan 1 to Jan 5)
    assert len(imputed) == 5
    # Closed day (Jan 2) should have 0 arrivals and forward filled price
    jan2 = imputed[imputed["record_date"] == "2024-01-02"].iloc[0]
    assert jan2["arrivals_qtl"] == 0.0
    assert jan2["modal_price"] == 2000.0

def test_feature_engineering_integrity():
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    df = pd.DataFrame({
        "record_date": dates,
        "mandi_name": ["Lasalgaon"] * 90,
        "commodity": ["Onion"] * 90,
        "arrivals_qtl": np.random.uniform(500, 1500, size=90),
        "min_price": [1800.0] * 90,
        "max_price": [2200.0] * 90,
        "modal_price": np.random.uniform(1850, 2150, size=90)
    })
    
    feat_df = create_agricultural_features(df, horizons=[7, 15, 30])
    
    # Check that lags, rolling features, and Fourier transforms exist
    assert "modal_price_lag_7" in feat_df.columns
    assert "price_roll_mean_14" in feat_df.columns
    assert "sin_doy" in feat_df.columns
    assert "cos_doy" in feat_df.columns
    assert "target_price_7d" in feat_df.columns

def test_evaluation_metrics():
    y_true = np.array([2000.0, 2200.0, 2100.0, 1900.0])
    y_pred = np.array([2050.0, 2150.0, 2120.0, 1850.0])
    y_prior = np.array([1950.0, 2000.0, 2200.0, 2000.0])
    
    wmape = calculate_wmape(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mda = calculate_mda(y_true, y_pred, y_prior)
    
    assert 0 < wmape < 5.0
    assert mae > 0
    assert rmse >= mae
    assert 0 <= mda <= 100.0
