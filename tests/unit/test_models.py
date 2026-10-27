"""
Unit and Integration Tests for AgriPrice-AI Machine Learning Models.
Tests SARIMAX Baseline, Multi-Horizon Quantile XGBoost, and PyTorch Bi-LSTM with Temporal Attention.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from agriprice.features.feature_engineering import create_agricultural_features
from agriprice.models.baseline_sarimax import SarimaxBaseline
from agriprice.models.ml_regressor import AgriculturalXGBoostForecaster
from agriprice.models.deep_learning import DeepLearningPipeline, TORCH_AVAILABLE


@pytest.fixture
def synthetic_series():
    """Generates continuous 90-day time series for rapid model testing."""
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    df = pd.DataFrame({
        "record_date": dates,
        "mandi_name": ["Lasalgaon"] * 90,
        "commodity": ["Onion"] * 90,
        "arrivals_qtl": np.random.uniform(800, 1500, size=90),
        "min_price": np.random.uniform(1800, 2000, size=90),
        "max_price": np.random.uniform(2200, 2500, size=90),
        "modal_price": np.random.uniform(1950, 2250, size=90)
    })
    feat_df = create_agricultural_features(df, horizons=[7, 15, 30])
    return feat_df


def test_sarimax_baseline(synthetic_series):
    model = SarimaxBaseline(order=(1, 1, 0), seasonal_order=(0, 0, 0, 0))
    train_df = synthetic_series.iloc[:70]
    test_df = synthetic_series.iloc[70:]

    model.fit(train_df, exog_features=["sin_doy", "cos_doy"])
    preds = model.predict(steps=len(test_df), df_future_exog=test_df)

    assert "p10" in preds and "p50" in preds and "p90" in preds
    assert len(preds["p50"]) == len(test_df)
    assert np.all(preds["p50"] > 0)


def test_xgboost_multi_quantile(synthetic_series, tmp_path):
    forecaster = AgriculturalXGBoostForecaster(horizons=[7, 15])
    eval_results = forecaster.train(synthetic_series, test_days=15)

    assert "7d" in eval_results
    assert eval_results["7d"]["wmape_pct"] >= 0.0
    assert eval_results["7d"]["mae_inr_qtl"] >= 0.0

    latest_row = synthetic_series.iloc[[-1]]
    preds = forecaster.predict_latest(latest_row)

    assert "7d" in preds and "15d" in preds
    p10 = preds["7d"]["p10"]
    p50 = preds["7d"]["p50"]
    p90 = preds["7d"]["p90"]

    # Quantile monotonicity
    assert p10 <= p50 <= p90

    # Feature importances
    importances = forecaster.get_feature_importances(horizon=7, top_n=3)
    assert len(importances) > 0
    assert importances[0][1] >= 0.0

    # Save & reload
    save_path = tmp_path / "test_xgb.pkl"
    joblib.dump({"models": forecaster.models, "features": forecaster.feature_cols}, save_path)

    reloaded = AgriculturalXGBoostForecaster(horizons=[7, 15])
    reloaded.load(save_path)
    preds_reloaded = reloaded.predict_latest(latest_row)
    assert preds_reloaded["7d"]["p50"] == p50


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_pytorch_bilstm_attention(synthetic_series):
    dl_pipeline = DeepLearningPipeline(seq_len=20, hidden_dim=32, num_layers=1)
    metrics = dl_pipeline.train(
        synthetic_series,
        target_col="target_price_7d",
        epochs=3,
        batch_size=16,
        test_days=15
    )

    assert "wmape_pct" in metrics
    assert metrics["wmape_pct"] > 0

    preds = dl_pipeline.predict_latest(synthetic_series.tail(25))
    assert "p10" in preds and "p50" in preds and "p90" in preds
    assert preds["p10"] <= preds["p50"] <= preds["p90"]
    assert "attention_weights" in preds
