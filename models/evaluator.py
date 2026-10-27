"""
Domain-Specific Evaluation Metrics for Agricultural Commodity Price Forecasting.
Calculates WMAPE, MAE, RMSE, and Mean Directional Accuracy (MDA).
"""
import numpy as np
from typing import Dict, Union

def calculate_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Mean Absolute Percentage Error (WMAPE):
    Weights percentage error by volume/price magnitude, preventing division by zero
    or explosion when prices collapse to distress levels (e.g. ₹2/kg).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total_actual = np.sum(np.abs(y_true))
    if total_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual * 100.0)

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in INR / Quintal."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (penalizes catastrophic misses)."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def calculate_mda(y_true: np.ndarray, y_pred: np.ndarray, y_prior: np.ndarray) -> float:
    """
    Mean Directional Accuracy (MDA):
    Evaluates whether the model correctly forecasted if prices will RISE or FALL.
    Paramount for trader arbitrage and farmer hold-vs-sell guidance.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prior = np.asarray(y_prior)
    
    actual_dir = np.sign(y_true - y_prior)
    pred_dir = np.sign(y_pred - y_prior)
    
    correct_dir = (actual_dir == pred_dir)
    return float(np.mean(correct_dir) * 100.0)

def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prior: np.ndarray = None
) -> Dict[str, float]:
    """Computes full evaluation score dictionary."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    metrics = {
        "wmape_pct": round(calculate_wmape(y_true, y_pred), 2),
        "mae_inr_qtl": round(calculate_mae(y_true, y_pred), 2),
        "rmse_inr_qtl": round(calculate_rmse(y_true, y_pred), 2)
    }
    
    if y_prior is not None:
        metrics["mda_directional_acc_pct"] = round(calculate_mda(y_true, y_pred, y_prior), 2)
        
    return metrics
