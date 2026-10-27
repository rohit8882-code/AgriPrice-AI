"""
Integration Tests for AgriPrice-AI FastAPI Endpoints.
Uses TestClient to validate responses, schemas, and error boundaries.
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_list_commodities():
    response = client.get("/v1/commodities")
    assert response.status_code == 200
    data = response.json()
    assert "commodities" in data
    assert "Onion" in data["commodities"]


def test_list_mandis():
    response = client.get("/v1/mandis")
    assert response.status_code == 200
    data = response.json()
    assert "mandis" in data
    assert "Lasalgaon (Nashik, MH)" in data["mandis"]


def test_get_price_forecast():
    response = client.get("/v1/forecast/Onion/Lasalgaon (Nashik, MH)")
    assert response.status_code == 200
    data = response.json()
    assert data["commodity"] == "Onion"
    assert data["mandi_name"] == "Lasalgaon (Nashik, MH)"
    assert "forecasts" in data
    assert "7d" in data["forecasts"]
    assert "15d" in data["forecasts"]
    assert "30d" in data["forecasts"]
    
    # Check quantile order
    f_7d = data["forecasts"]["7d"]
    assert f_7d["predicted_p10"] <= f_7d["predicted_p50"] <= f_7d["predicted_p90"]
    assert "advisory_recommendation" in data


def test_get_historical_prices():
    response = client.get("/v1/historical/Onion/Lasalgaon (Nashik, MH)?limit_days=30")
    assert response.status_code == 200
    data = response.json()
    assert data["commodity"] == "Onion"
    assert len(data["history"]) > 0
    first = data["history"][0]
    assert "record_date" in first
    assert "modal_price" in first
    assert "arrivals_qtl" in first


def test_get_model_evaluations():
    response = client.get("/v1/evaluation/Onion/Lasalgaon (Nashik, MH)")
    assert response.status_code == 200
    data = response.json()
    assert "benchmarks" in data
    assert len(data["benchmarks"]) >= 3


def test_find_arbitrage():
    response = client.get("/v1/arbitrage?commodity=Onion&max_radius_km=1500")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_get_market_alerts():
    response = client.get("/v1/alerts")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_get_mandi_weather():
    response = client.get("/v1/weather/Lasalgaon (Nashik, MH)")
    assert response.status_code == 200
    data = response.json()
    assert data["mandi_name"] == "Lasalgaon (Nashik, MH)"
    assert "temp_max_c" in data
    assert "weather_status" in data


def test_scenario_simulation():
    payload = {
        "commodity": "Onion",
        "mandi_name": "Lasalgaon (Nashik, MH)",
        "sim_rain_shock_mm": 50.0,
        "sim_arrival_glut_multiplier": 1.0,
        "sim_export_ban": False
    }
    response = client.post("/v1/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "baseline_forecast_15d" in data
    assert "simulated_forecast_15d" in data
    assert data["impact_pct"] > 0 # Rain shock increases price
    assert "recommended_action" in data


def test_invalid_mandi_and_crop():
    response = client.get("/v1/forecast/FakeCrop/Lasalgaon (Nashik, MH)")
    assert response.status_code == 404

    response = client.get("/v1/forecast/Onion/NonExistentMandi")
    assert response.status_code == 404
