"""
Unit Tests for AgriPrice-AI Spatial Arbitrage and Haversine Distance Engine.
"""

from agriprice.config.settings import settings
from agriprice.utils.geo import haversine_distance
from agriprice.services.arbitrage_service import calculate_spatial_arbitrage


def test_haversine_distance():
    # Lasalgaon to Azadpur Delhi is approx 1,000-1,200 km
    lasal = settings.SUPPORTED_MANDIS["Lasalgaon (Nashik, MH)"]
    delhi = settings.SUPPORTED_MANDIS["Azadpur (Delhi, DL)"]

    dist = haversine_distance(lasal["lat"], lasal["lon"], delhi["lat"], delhi["lon"])
    assert 950.0 < dist < 1250.0

    # Same location should be 0.0 km
    zero_dist = haversine_distance(lasal["lat"], lasal["lon"], lasal["lat"], lasal["lon"])
    assert zero_dist == 0.0


def test_spatial_arbitrage_ranking():
    opps = calculate_spatial_arbitrage(commodity="Onion", max_radius_km=2500.0)
    assert isinstance(opps, list)

    if len(opps) > 1:
        for i in range(len(opps) - 1):
            assert opps[i]["net_profit_potential_inr_qtl"] >= opps[i + 1]["net_profit_potential_inr_qtl"]
            assert opps[i]["origin_price"] < opps[i]["destination_price"]
            assert opps[i]["estimated_freight_inr"] > 0
            assert "data_source" in opps[i]
