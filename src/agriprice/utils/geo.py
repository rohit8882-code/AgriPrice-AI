"""
Geospatial calculations for Indian APMC mandis and logistics routing.
"""

import math

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates great-circle distance between two Mandis in kilometers using the Haversine formula.

    Args:
        lat1: Latitude of first mandi
        lon1: Longitude of first mandi
        lat2: Latitude of second mandi
        lon2: Longitude of second mandi

    Returns:
        Distance in kilometers rounded to 1 decimal place.
    """
    radius_earth_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(radius_earth_km * c, 1)
