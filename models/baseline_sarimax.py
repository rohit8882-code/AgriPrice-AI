"""
SARIMAX Statistical Baseline Model for Time Series Benchmarking.
Incorporates seasonal autoregression and exogenous weather/calendar features.
"""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Dict, Any, Tuple
from models.evaluator import evaluate_forecast

warnings.filterwarnings("ignore")

class SarimaxBaseline:
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fitted = None
        self.exog_cols = []

    def fit(self, df_train: pd.DataFrame, exog_features: list = None) -> "SarimaxBaseline":
        """
        Fits SARIMAX on train split with optional exogenous regressors.
        """
        y_train = df_train["modal_price"].values
        exog_train = None
        if exog_features:
            self.exog_cols = [c for c in exog_features if c in df_train.columns]
            if self.exog_cols:
                exog_train = df_train[self.exog_cols].fillna(0).values

        model = SARIMAX(
            endog=y_train,
            exog=exog_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model_fitted = model.fit(disp=False, maxiter=50)
        return self

    def predict(self, steps: int = 30, df_future_exog: pd.DataFrame = None) -> Dict[str, np.ndarray]:
        """
        Forecasts future steps with 80% confidence interval (approximating P10 and P90).
        """
        if self.model_fitted is None:
            raise ValueError("Model must be fitted before predict.")

        exog_future = None
        if self.exog_cols and df_future_exog is not None:
            exog_future = df_future_exog[self.exog_cols].fillna(0).iloc[:steps].values

        forecast_res = self.model_fitted.get_forecast(steps=steps, exog=exog_future)
        pred_mean = np.maximum(100.0, forecast_res.predicted_mean)
        conf_int = forecast_res.conf_int(alpha=0.20) # 80% CI -> P10, P90

        return {
            "p10": np.maximum(50.0, conf_int[:, 0]),
            "p50": pred_mean,
            "p90": conf_int[:, 1]
        }
