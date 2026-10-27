"""
Pydantic v2 schemas for AgriPrice-AI API requests and responses.
All responses include a data_source field (REAL or SIMULATED) per Rule 3.
"""

from typing import List, Dict, Optional, Any
from datetime import date, datetime
from pydantic import BaseModel, Field


class ForecastHorizonPoint(BaseModel):
    horizon_days: int
    target_date: date
    predicted_p10: float = Field(..., description="10th percentile - lower price boundary (INR/Qtl)")
    predicted_p50: float = Field(..., description="50th percentile - median expected price (INR/Qtl)")
    predicted_p90: float = Field(..., description="90th percentile - upper price boundary (INR/Qtl)")
    trend_direction: str = Field(..., description="BULLISH, BEARISH, or NEUTRAL")


class FeatureDriver(BaseModel):
    feature_name: str
    relative_importance_pct: float


class CommodityForecastResponse(BaseModel):
    commodity: str
    mandi_name: str
    state: str
    last_reported_date: date
    current_modal_price: float
    unit: str = "INR / Quintal"
    forecasts: Dict[str, ForecastHorizonPoint]
    top_driving_factors: List[FeatureDriver]
    advisory_recommendation: str
    data_source: str = Field(default="SIMULATED", description="Data provenance: REAL or SIMULATED")


class ArbitrageOpportunity(BaseModel):
    commodity: str
    origin_mandi: str
    destination_mandi: str
    distance_km: float
    origin_price: float
    destination_price: float
    gross_spread_inr: float
    estimated_freight_inr: float
    net_profit_potential_inr_qtl: float
    recommendation: str
    data_source: str = Field(default="SIMULATED", description="Data provenance: REAL or SIMULATED")


class MarketAlert(BaseModel):
    alert_id: str
    commodity: str
    mandi_name: str
    alert_type: str  # 'PRICE_CRASH_RISK', 'PRICE_SURGE_ALERT', 'UNSEASONAL_WEATHER_SHOCK'
    severity: str    # 'HIGH', 'MEDIUM', 'INFO'
    headline: str
    details: str
    triggered_at: datetime
    data_source: str = Field(default="SIMULATED", description="Data provenance: REAL or SIMULATED")


class HistoricalPricePoint(BaseModel):
    record_date: date
    modal_price: float
    min_price: float
    max_price: float
    arrivals_qtl: float


class HistoricalPriceResponse(BaseModel):
    commodity: str
    mandi_name: str
    total_records: int
    history: List[HistoricalPricePoint]
    data_source: str = Field(default="SIMULATED", description="Data provenance: REAL or SIMULATED")


class ModelBenchmarkMetric(BaseModel):
    model_name: str
    wmape_pct: float
    mae_inr_qtl: float
    rmse_inr_qtl: float
    mda_directional_acc_pct: Optional[float] = None


class BenchmarkComparisonResponse(BaseModel):
    commodity: str
    mandi_name: str
    test_horizon_days: int
    benchmarks: List[ModelBenchmarkMetric]
    data_source: str = Field(default="SIMULATED", description="Data provenance: REAL or SIMULATED")


class ScenarioSimulationRequest(BaseModel):
    commodity: str = "Onion"
    mandi_name: str = "Lasalgaon (Nashik, MH)"
    sim_rain_shock_mm: float = 0.0
    sim_arrival_glut_multiplier: float = 1.0
    sim_export_ban: bool = False


class ScenarioSimulationResponse(BaseModel):
    commodity: str
    mandi_name: str
    baseline_forecast_15d: float
    simulated_forecast_15d: float
    impact_pct: float
    scenario_summary: str
    recommended_action: str
    data_source: str = Field(default="SIMULATED", description="Data provenance: REAL or SIMULATED")


class MandiWeatherReport(BaseModel):
    mandi_name: str
    state: str
    record_date: date
    rainfall_mm: float
    temp_max_c: float
    temp_min_c: float
    soil_moisture_pct: float
    ndvi_index: float
    weather_status: str
    data_source: str = Field(default="SIMULATED", description="Data provenance: REAL or SIMULATED")
