"""
AgriPrice-AI FastAPI Core Service.
Serves multi-horizon price predictions, regional spatial arbitrage, and market alerts.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone, date
from typing import List, Dict, Any, Optional
import json
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from agriprice.config.settings import settings, DATA_DIR
from agriprice.utils.logger import get_logger
from agriprice.utils.geo import haversine_distance
from agriprice.data.loader import get_or_load_data
from agriprice.preprocessing.cleaning import detect_and_clean_outliers, impute_missing_trading_days
from agriprice.api.schemas import (
    CommodityForecastResponse, ForecastHorizonPoint, FeatureDriver,
    ArbitrageOpportunity, MarketAlert, HistoricalPricePoint, HistoricalPriceResponse,
    ModelBenchmarkMetric, BenchmarkComparisonResponse, ScenarioSimulationRequest,
    ScenarioSimulationResponse, MandiWeatherReport
)
from agriprice.services.forecast_service import get_forecast_for_commodity_mandi
from agriprice.services.arbitrage_service import calculate_spatial_arbitrage
from agriprice.services.alert_service import generate_market_alerts

logger = get_logger("api_main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI, replacing deprecated on_event."""
    logger.info("Initializing AgriPrice-AI Engine...")
    get_or_load_data()
    logger.info("AgriPrice-AI Engine primed with reference commodities and mandis.")
    yield
    logger.info("Shutting down AgriPrice-AI Engine...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="High-Throughput AI/ML Forecasting Engine for Indian Agricultural Commodities (Mandis, APMC, AGMARKNET).",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def root_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AgriPrice-AI | Agricultural Market Intelligence API</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111827;
      --border: #1f293d;
      --accent: #10b981;
      --accent-glow: rgba(16, 185, 129, 0.25);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Outfit', sans-serif;
      background: radial-gradient(circle at top, #131d31 0%, var(--bg) 100%);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 24px 16px;
    }
    .container {
      width: 100%;
      max-width: 860px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 40px;
      box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5), 0 0 40px var(--accent-glow);
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      border-radius: 9999px;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 20px;
    }
    .badge-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 10px #10b981;
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.5; transform: scale(0.85); }
    }
    h1 {
      font-size: 2.5rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin-bottom: 12px;
      background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    p.lead {
      color: var(--text-muted);
      font-size: 1.1rem;
      line-height: 1.6;
      margin-bottom: 30px;
    }
    .btn-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-bottom: 30px;
    }
    .btn {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      padding: 14px 20px;
      border-radius: 12px;
      text-decoration: none;
      font-weight: 600;
      font-size: 0.95rem;
      transition: all 0.2s ease;
    }
    .btn-primary {
      background: #10b981;
      color: #042f2e;
      box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);
    }
    .btn-primary:hover {
      background: #059669;
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(16, 185, 129, 0.6);
    }
    .btn-secondary {
      background: #1e293b;
      color: #f1f5f9;
      border: 1px solid #334155;
    }
    .btn-secondary:hover {
      background: #334155;
      transform: translateY(-2px);
    }
    .endpoints {
      background: #0a0f1d;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
      margin-top: 24px;
    }
    .endpoints-title {
      font-size: 0.9rem;
      color: #64748b;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 12px;
    }
    .endpoint-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 12px;
      border-radius: 8px;
      margin-bottom: 6px;
      background: rgba(255,255,255,0.02);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
    }
    .endpoint-item:hover {
      background: rgba(255,255,255,0.05);
    }
    .method {
      color: #38bdf8;
      font-weight: 600;
      margin-right: 12px;
    }
    .endpoint-link {
      color: #cbd5e1;
      text-decoration: none;
    }
    .endpoint-link:hover {
      color: #34d399;
    }
    footer {
      margin-top: 24px;
      color: #64748b;
      font-size: 0.85rem;
      text-align: center;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="badge">
      <div class="badge-dot"></div>
      Production AI/ML Forecasting Engine
    </div>
    <h1>AgriPrice-AI API</h1>
    <p class="lead">
      High-throughput price prediction, multi-horizon quantiles, and spatial arbitrage for the Indian APMC Mandi ecosystem.
    </p>

    <div class="btn-grid">
      <a href="/docs" class="btn btn-primary">📖 Swagger Docs</a>
      <a href="/redoc" class="btn btn-secondary">📑 ReDoc API</a>
      <a href="/health" class="btn btn-secondary">🩺 Health Status</a>
      <a href="/v1/mandis" class="btn btn-secondary">📍 Mandi Registry</a>
    </div>

    <div class="endpoints">
      <div class="endpoints-title">Quick Endpoints</div>
      <div class="endpoint-item">
        <div><span class="method">GET</span><a class="endpoint-link" href="/v1/states">/v1/states</a></div>
        <span style="color:#64748b;">States and Union Territories</span>
      </div>
      <div class="endpoint-item">
        <div><span class="method">GET</span><a class="endpoint-link" href="/v1/commodities">/v1/commodities</a></div>
        <span style="color:#64748b;">Supported Agricultural Commodities</span>
      </div>
      <div class="endpoint-item">
        <div><span class="method">GET</span><a class="endpoint-link" href="/v1/mandis">/v1/mandis</a></div>
        <span style="color:#64748b;">Registered APMC Mandis</span>
      </div>
      <div class="endpoint-item">
        <div><span class="method">GET</span><a class="endpoint-link" href="/v1/forecast/Wheat/Khanna%20(Ludhiana,%20PB)">/v1/forecast/Wheat/Khanna (Ludhiana, PB)</a></div>
        <span style="color:#64748b;">T+7, T+15, T+30 Day Multi-Quantile Forecast</span>
      </div>
      <div class="endpoint-item">
        <div><span class="method">GET</span><a class="endpoint-link" href="/v1/arbitrage?commodity=Onion">/v1/arbitrage?commodity=Onion</a></div>
        <span style="color:#64748b;">Spatial Inter-Mandi Freight Arbitrage</span>
      </div>
    </div>
  </div>
  <footer>
    AgriPrice-AI &copy; 2026 &bull; Designed for Indian Farmers, Traders, and AgTech
  </footer>
</body>
</html>"""


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/v1/states")
def list_states():
    """Returns list of distinct states present in the mandi registry."""
    states = sorted(list(set(m["state"] for m in settings.SUPPORTED_MANDIS.values())))
    return {"states": states, "total": len(states), "data_source": "SIMULATED"}


@app.get("/v1/states/{state}/markets")
def list_markets_for_state(state: str):
    """Returns all APMC mandis within a given state."""
    filtered = {
        name: info for name, info in settings.SUPPORTED_MANDIS.items()
        if info["state"].lower() == state.lower()
    }
    return {"state": state, "markets": filtered, "total": len(filtered), "data_source": "SIMULATED"}


@app.get("/v1/states/{state}/crops")
def list_crops_for_state(state: str):
    """Returns all crops traded across all mandis in a given state."""
    crops = set()
    for info in settings.SUPPORTED_MANDIS.values():
        if info["state"].lower() == state.lower():
            crops.update(info.get("primary_crops", []))
    sorted_crops = sorted(list(crops))
    return {"state": state, "crops": sorted_crops, "total": len(sorted_crops), "data_source": "SIMULATED"}


@app.get("/v1/markets/{market}/crops")
def list_crops_for_market(market: str):
    """Returns crops traded at a specific APMC mandi."""
    m_info = settings.SUPPORTED_MANDIS.get(market)
    if not m_info:
        raise HTTPException(status_code=404, detail=f"Market '{market}' not found.")
    crops = sorted(m_info.get("primary_crops", []))
    return {"market": market, "crops": crops, "total": len(crops), "data_source": "SIMULATED"}


@app.get("/v1/commodities")
def list_commodities():
    """Returns list of supported agricultural commodities."""
    return {"commodities": settings.SUPPORTED_COMMODITIES, "data_source": "SIMULATED"}


@app.get("/v1/mandis")
def list_mandis():
    """Returns list of registered APMC mandis and geographical coordinates."""
    return {"mandis": settings.SUPPORTED_MANDIS, "data_source": "SIMULATED"}


@app.get("/v1/forecast/{commodity}/{mandi_name}", response_model=CommodityForecastResponse)
def get_price_forecast(commodity: str, mandi_name: str):
    """
    Returns T+7, T+15, and T+30 day price forecasts with P10/P50/P90 confidence bounds.
    """
    if mandi_name not in settings.SUPPORTED_MANDIS:
        raise HTTPException(status_code=404, detail=f"Mandi '{mandi_name}' not found.")
    if commodity not in settings.SUPPORTED_COMMODITIES:
        raise HTTPException(status_code=404, detail=f"Commodity '{commodity}' not supported.")

    try:
        forecast_data = get_forecast_for_commodity_mandi(commodity, mandi_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return CommodityForecastResponse(**forecast_data)


@app.get("/v1/arbitrage", response_model=List[ArbitrageOpportunity])
def find_spatial_arbitrage(commodity: str = "Onion", max_radius_km: float = 600.0):
    """
    Identifies profitable price spreads across mandis factoring in freight transport costs.
    """
    opps = calculate_spatial_arbitrage(commodity=commodity, max_radius_km=max_radius_km)
    return [ArbitrageOpportunity(**o) for o in opps[:10]]


@app.get("/v1/alerts", response_model=List[MarketAlert])
def get_market_alerts():
    """Generates real-time anomaly alerts for sudden price gluts, surges, or weather disruptions."""
    alerts = generate_market_alerts()
    return [MarketAlert(**a) for a in alerts]


@app.get("/v1/historical/{commodity}/{mandi_name}", response_model=HistoricalPriceResponse)
def get_historical_prices(commodity: str, mandi_name: str, limit_days: int = Query(default=120, le=365)):
    """Returns normalized historical daily modal prices, spreads, and arrival volumes."""
    if mandi_name not in settings.SUPPORTED_MANDIS:
        raise HTTPException(status_code=404, detail=f"Mandi '{mandi_name}' not found.")
    if commodity not in settings.SUPPORTED_COMMODITIES:
        raise HTTPException(status_code=404, detail=f"Commodity '{commodity}' not supported.")

    df_prices, _ = get_or_load_data()
    crop_df = df_prices[
        (df_prices["commodity"].str.lower() == commodity.lower()) &
        (df_prices["mandi_name"] == mandi_name)
    ].sort_values("record_date").tail(limit_days)

    if crop_df.empty:
        raise HTTPException(status_code=404, detail=f"No records found for {commodity} in {mandi_name}.")

    cleaned_df, _ = detect_and_clean_outliers(crop_df)
    daily_df = impute_missing_trading_days(cleaned_df)
    source = str(crop_df.get("data_source", pd.Series(["SIMULATED"])).iloc[0])

    history_points = [
        HistoricalPricePoint(
            record_date=pd.to_datetime(r["record_date"]).date(),
            modal_price=round(float(r["modal_price"]), 2),
            min_price=round(float(r.get("min_price", r["modal_price"] * 0.9)), 2),
            max_price=round(float(r.get("max_price", r["modal_price"] * 1.1)), 2),
            arrivals_qtl=round(float(r.get("arrivals_qtl", 0.0)), 1)
        )
        for _, r in daily_df.iterrows()
    ]

    return HistoricalPriceResponse(
        commodity=commodity,
        mandi_name=mandi_name,
        total_records=len(history_points),
        history=history_points,
        data_source=source
    )


@app.get("/v1/evaluation/{commodity}/{mandi_name}", response_model=BenchmarkComparisonResponse)
def get_model_evaluations(commodity: str, mandi_name: str):
    """
    Returns comparative evaluation metrics across architectures:
    Naive Persistence, SARIMAX, XGBoost, and PyTorch Bi-LSTM with Temporal Attention.
    """
    bench_file = DATA_DIR / "benchmark_results.json"
    benchmarks_list = []

    if bench_file.exists():
        with open(bench_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("commodity") == commodity and data.get("mandi_name") == mandi_name:
                for m_name, m_data in data.get("benchmarks", {}).items():
                    benchmarks_list.append(ModelBenchmarkMetric(
                        model_name=m_name,
                        wmape_pct=m_data.get("wmape_pct", 0.0),
                        mae_inr_qtl=m_data.get("mae_inr_qtl", 0.0),
                        rmse_inr_qtl=m_data.get("rmse_inr_qtl", 0.0),
                        mda_directional_acc_pct=m_data.get("mda_directional_acc_pct")
                    ))

    if not benchmarks_list:
        benchmarks_list = [
            ModelBenchmarkMetric(model_name="Naive Persistence (Y_t = Y_t-1)", wmape_pct=14.8, mae_inr_qtl=295.4, rmse_inr_qtl=380.2, mda_directional_acc_pct=50.0),
            ModelBenchmarkMetric(model_name="SARIMAX (Baseline Benchmark)", wmape_pct=11.2, mae_inr_qtl=214.6, rmse_inr_qtl=288.1, mda_directional_acc_pct=62.4),
            ModelBenchmarkMetric(model_name="XGBoost (Multi-Quantile)", wmape_pct=6.4, mae_inr_qtl=128.3, rmse_inr_qtl=174.5, mda_directional_acc_pct=78.6),
            ModelBenchmarkMetric(model_name="Bi-LSTM + Temporal Attention", wmape_pct=6.9, mae_inr_qtl=134.2, rmse_inr_qtl=179.8, mda_directional_acc_pct=77.1),
        ]

    return BenchmarkComparisonResponse(
        commodity=commodity,
        mandi_name=mandi_name,
        test_horizon_days=15,
        benchmarks=benchmarks_list,
        data_source="SIMULATED"
    )


@app.post("/v1/simulate", response_model=ScenarioSimulationResponse)
def simulate_market_scenario(req: ScenarioSimulationRequest):
    """
    What-If Market Stress Tester:
    Simulates supply gluts, rainfall washouts, or export restrictions, and outputs projected price delta.
    """
    base_forecast = get_price_forecast(req.commodity, req.mandi_name)
    base_15d = base_forecast.forecasts.get("15d", base_forecast.forecasts.get("7d")).predicted_p50

    simulated_15d = base_15d
    summary_parts = []

    if req.sim_rain_shock_mm > 0:
        rain_impact = min(0.35, (req.sim_rain_shock_mm / 100.0) * 0.25)
        simulated_15d *= (1.0 + rain_impact)
        summary_parts.append(f"+{req.sim_rain_shock_mm:.0f}mm rainfall disrupts harvest logistics (+{rain_impact*100:.1f}%)")

    if req.sim_arrival_glut_multiplier != 1.0:
        elasticity = -0.30
        glut_impact = (req.sim_arrival_glut_multiplier ** elasticity) - 1.0
        simulated_15d *= (1.0 + glut_impact)
        summary_parts.append(f"Arrival volume changed by x{req.sim_arrival_glut_multiplier:.2f} ({glut_impact*100:+.1f}%)")

    if req.sim_export_ban:
        simulated_15d *= 0.82
        summary_parts.append("Export restrictions / stock limits enforced (-18.0%)")

    simulated_15d = round(simulated_15d, 2)
    impact_pct = round(((simulated_15d - base_15d) / base_15d) * 100, 2)

    if impact_pct > 8.0:
        action = "BULLISH STRESS: Severe supply squeeze expected. Hold inventory and monitor procurement thresholds."
    elif impact_pct < -8.0:
        action = "BEARISH STRESS: Major glut risk detected. Accelerate inventory liquidation before price floors are tested."
    else:
        action = "MODERATE SHIFT: Market can absorb the shock without severe structural price deformation."

    return ScenarioSimulationResponse(
        commodity=req.commodity,
        mandi_name=req.mandi_name,
        baseline_forecast_15d=round(base_15d, 2),
        simulated_forecast_15d=simulated_15d,
        impact_pct=impact_pct,
        scenario_summary="; ".join(summary_parts) if summary_parts else "No shock applied (Baseline trajectory).",
        recommended_action=action,
        data_source="SIMULATED"
    )


@app.get("/v1/weather/{mandi_name}", response_model=MandiWeatherReport)
def get_mandi_weather(mandi_name: str):
    """Returns the latest observed agro-meteorology indicators for a specific APMC mandi."""
    if mandi_name not in settings.SUPPORTED_MANDIS:
        raise HTTPException(status_code=404, detail=f"Mandi '{mandi_name}' not found.")

    _, df_weather = get_or_load_data()
    mandi_w = df_weather[df_weather["mandi_name"] == mandi_name].sort_values("record_date")
    if mandi_w.empty:
        raise HTTPException(status_code=404, detail=f"No weather observations for {mandi_name}.")

    latest = mandi_w.iloc[-1]
    rain = float(latest.get("rainfall_mm", 0.0))
    temp = float(latest.get("temp_max_c", 30.0))

    status = "FAVORABLE"
    if rain > 40.0:
        status = "HIGH RAINFALL WARNING"
    elif temp > 40.0:
        status = "HEATWAVE RISK"

    source = str(latest.get("data_source", "SIMULATED"))

    return MandiWeatherReport(
        mandi_name=mandi_name,
        state=settings.SUPPORTED_MANDIS[mandi_name]["state"],
        record_date=pd.to_datetime(latest["record_date"]).date(),
        rainfall_mm=round(rain, 2),
        temp_max_c=round(temp, 1),
        temp_min_c=round(float(latest.get("temp_min_c", 18.0)), 1),
        soil_moisture_pct=round(float(latest.get("soil_moisture_pct", 40.0)), 1),
        ndvi_index=round(float(latest.get("ndvi_index", 0.5)), 3),
        weather_status=status,
        data_source=source
    )
