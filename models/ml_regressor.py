"""
XGBoost & Gradient Boosting Multi-Horizon Regressor for Agricultural Commodities.
Trains models across 7, 15, and 30-day forecast horizons with quantile intervals (P10, P50, P90).
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from xgboost import XGBRegressor
from config.settings import MODEL_DIR
from models.evaluator import evaluate_forecast

class AgriculturalXGBoostForecaster:
    def __init__(self, horizons: List[int] = [7, 15, 30]):
        self.horizons = horizons
        self.models: Dict[int, Dict[str, XGBRegressor]] = {} # horizon -> {p10, p50, p90}
        self.feature_cols: List[str] = []

    def get_default_features(self, df: pd.DataFrame) -> List[str]:
        """Filters available features while excluding metadata and target columns."""
        ignore_cols = [
            "record_date", "mandi_name", "commodity", "state", "district", "data_source"
        ] + [c for c in df.columns if c.startswith("target_")]
        
        features = [
            c for c in df.columns 
            if c not in ignore_cols and np.issubdtype(df[c].dtype, np.number)
        ]
        return features

    def train(self, df_features: pd.DataFrame, test_days: int = 45) -> Dict[str, Dict]:
        """
        Trains multi-quantile XGBoost regressors across defined horizons.
        Uses out-of-time evaluation split on the most recent `test_days`.
        """
        min_lag = "modal_price_lag_14" if "modal_price_lag_14" in df_features.columns else "modal_price_lag_7"
        df = df_features.dropna(subset=[min_lag]).copy()
        self.feature_cols = self.get_default_features(df)
        
        actual_test_days = min(test_days, max(5, int(len(df) * 0.15)))
        split_idx = len(df) - actual_test_days
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        evaluation_results = {}

        for h in self.horizons:
            target_col = f"target_price_{h}d"
            if target_col not in df.columns:
                continue

            # Drop rows where target is NaN (at the end of series)
            tr = train_df.dropna(subset=[target_col])
            te = test_df.dropna(subset=[target_col])

            X_tr, y_tr = tr[self.feature_cols], tr[target_col]
            X_te, y_te = te[self.feature_cols], te[target_col]

            self.models[h] = {}

            # Train Quantile Regressors: P50 (Median point estimate)
            model_p50 = XGBRegressor(
                n_estimators=160,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42
            )
            model_p50.fit(X_tr, y_tr)
            self.models[h]["p50"] = model_p50

            # Quantile Pinball Regressors for P10 (Lower band) and P90 (Upper band)
            model_p10 = XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=0.10,
                n_estimators=120,
                max_depth=4,
                learning_rate=0.06,
                random_state=42
            )
            model_p10.fit(X_tr, y_tr)
            self.models[h]["p10"] = model_p10

            model_p90 = XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=0.90,
                n_estimators=120,
                max_depth=4,
                learning_rate=0.06,
                random_state=42
            )
            model_p90.fit(X_tr, y_tr)
            self.models[h]["p90"] = model_p90

            # Evaluate on Test Set
            preds_p50 = model_p50.predict(X_te)
            preds_p10 = model_p10.predict(X_te)
            preds_p90 = model_p90.predict(X_te)
            
            # Ensure logical monotonicity: p10 <= p50 <= p90
            preds_p10 = np.minimum(preds_p10, preds_p50)
            preds_p90 = np.maximum(preds_p90, preds_p50)

            y_prior = te["modal_price"].values
            metrics = evaluate_forecast(y_te.values, preds_p50, y_prior)
            evaluation_results[f"{h}d"] = metrics

        return evaluation_results

    def predict_latest(self, latest_feature_row: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Generates multi-horizon forecast predictions given the most recent feature vector.
        """
        X = latest_feature_row[self.feature_cols].fillna(0)
        forecasts = {}

        for h, q_models in self.models.items():
            p10 = float(q_models["p10"].predict(X)[0])
            p50 = float(q_models["p50"].predict(X)[0])
            p90 = float(q_models["p90"].predict(X)[0])

            # Monotonicity boundary check
            p10 = min(p10, p50)
            p90 = max(p90, p50)

            forecasts[f"{h}d"] = {
                "horizon_days": h,
                "p10": round(p10, 2),
                "p50": round(p50, 2),
                "p90": round(p90, 2)
            }
        return forecasts

    def get_feature_importances(self, horizon: int = 7, top_n: int = 10) -> List[Tuple[str, float]]:
        """Returns top driving features for interpretability."""
        if horizon not in self.models or "p50" not in self.models[horizon]:
            return []
        importances = self.models[horizon]["p50"].feature_importances_
        sorted_pairs = sorted(zip(self.feature_cols, importances), key=lambda x: x[1], reverse=True)
        return [(name, round(float(imp) * 100, 2)) for name, imp in sorted_pairs[:top_n]]

    def save(self, commodity: str, mandi_name: str) -> Path:
        safe_name = f"{commodity}_{mandi_name}".replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        path = MODEL_DIR / f"xgb_{safe_name}.pkl"
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump({"models": self.models, "features": self.feature_cols}, path)
        except (OSError, PermissionError, IOError):
            pass
        return path

    def load(self, path: Path) -> "AgriculturalXGBoostForecaster":
        data = joblib.load(path)
        self.models = data["models"]
        self.feature_cols = data["features"]
        return self
