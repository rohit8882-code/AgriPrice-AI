"""
Domain-Specific Evaluation Metrics for Agricultural Commodity Price Forecasting.
Calculates WMAPE, MAE, RMSE, and Mean Directional Accuracy (MDA).
"""

from typing import Dict, Optional, Union
import numpy as np


def calculate_wmape(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    Weighted Mean Absolute Percentage Error (WMAPE):
    Weights percentage error by volume/price magnitude, preventing division by zero
    or explosion when prices collapse to distress levels (e.g. ₹2/kg).
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    total_actual = float(np.sum(np.abs(y_t)))
    if total_actual == 0.0:
        return 0.0
    return float(np.sum(np.abs(y_t - y_p)) / total_actual * 100.0)


def calculate_mae(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Mean Absolute Error in INR / Quintal."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def calculate_rmse(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Root Mean Squared Error (penalizes catastrophic misses)."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def calculate_mda(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prior: Union[np.ndarray, list]
) -> float:
    """
    Mean Directional Accuracy (MDA):
    Evaluates whether the model correctly forecasted if prices will RISE or FALL.
    Paramount for trader arbitrage and farmer hold-vs-sell guidance.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    y_pr = np.asarray(y_prior)

    actual_dir = np.sign(y_t - y_pr)
    pred_dir = np.sign(y_p - y_pr)

    correct_dir = actual_dir == pred_dir
    return float(np.mean(correct_dir) * 100.0)


def evaluate_forecast(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prior: Optional[Union[np.ndarray, list]] = None,
) -> Dict[str, float]:
    """Computes full evaluation score dictionary."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    metrics = {
        "wmape_pct": round(calculate_wmape(y_t, y_p), 2),
        "mae_inr_qtl": round(calculate_mae(y_t, y_p), 2),
        "rmse_inr_qtl": round(calculate_rmse(y_t, y_p), 2),
    }

    if y_prior is not None:
        metrics["mda_directional_acc_pct"] = round(calculate_mda(y_t, y_p, y_prior), 2)

    return metrics
