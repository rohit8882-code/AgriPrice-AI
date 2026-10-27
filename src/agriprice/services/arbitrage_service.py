"""
Arbitrage Calculation Service: spatial price spreads adjusted for road freight.
"""

from typing import List, Dict, Any
from agriprice.config.settings import settings
from agriprice.utils.geo import haversine_distance
from agriprice.data.loader import get_or_load_data


def calculate_spatial_arbitrage(
    commodity: str = "Onion",
    max_radius_km: float = 600.0,
    freight_per_km_qtl: float = 0.35,
    loading_handling_inr: float = 40.0,
    min_profit_threshold: float = 120.0
) -> List[Dict[str, Any]]:
    """
    Identifies profitable price spreads across mandis factoring in freight transport costs.

    Args:
        commodity: Target crop
        max_radius_km: Maximum transport radius in km
        freight_per_km_qtl: Truck freight cost in INR per km per quintal
        loading_handling_inr: Loading and mandi handling charges in INR per quintal
        min_profit_threshold: Minimum net profit required in INR/qtl

    Returns:
        Sorted list of arbitrage opportunities descending by net profit.
    """
    df_prices, _ = get_or_load_data()
    latest_records = (
        df_prices[df_prices["commodity"].str.lower() == commodity.lower()]
        .sort_values("record_date")
        .groupby("mandi_name")
        .last()
        .reset_index()
    )

    opportunities: List[Dict[str, Any]] = []
    mandis_list = latest_records.to_dict("records")

    for i in range(len(mandis_list)):
        for j in range(len(mandis_list)):
            if i == j:
                continue
            m1 = mandis_list[i]
            m2 = mandis_list[j]

            m1_info = settings.SUPPORTED_MANDIS.get(m1["mandi_name"])
            m2_info = settings.SUPPORTED_MANDIS.get(m2["mandi_name"])
            if not m1_info or not m2_info:
                continue

            dist = haversine_distance(m1_info["lat"], m1_info["lon"], m2_info["lat"], m2_info["lon"])
            if dist > max_radius_km or dist == 0:
                continue

            p_origin = float(m1["modal_price"])
            p_dest = float(m2["modal_price"])
            gross_spread = p_dest - p_origin

            if gross_spread <= 0:
                continue

            freight_cost = round((dist * freight_per_km_qtl) + loading_handling_inr, 2)
            net_profit = round(gross_spread - freight_cost, 2)

            if net_profit > min_profit_threshold:
                opportunities.append({
                    "commodity": commodity,
                    "origin_mandi": m1["mandi_name"],
                    "destination_mandi": m2["mandi_name"],
                    "distance_km": dist,
                    "origin_price": round(p_origin, 2),
                    "destination_price": round(p_dest, 2),
                    "gross_spread_inr": round(gross_spread, 2),
                    "estimated_freight_inr": freight_cost,
                    "net_profit_potential_inr_qtl": net_profit,
                    "recommendation": (
                        f"High Arbitrage Potential: Transport from {m1['mandi_name']} to {m2['mandi_name']} "
                        f"yields ~₹{net_profit:.0f}/Qtl net gain."
                    ),
                    "data_source": str(m1.get("data_source", "SIMULATED"))
                })

    opportunities.sort(key=lambda x: x["net_profit_potential_inr_qtl"], reverse=True)
    return opportunities
