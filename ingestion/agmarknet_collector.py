"""
Agmarknet & OGD India Data Ingestion Collector.
Fetches daily variety-wise modal prices and arrivals across Indian APMC mandis.
"""
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from config.settings import settings

class AgmarknetCollector:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.DATA_GOV_IN_API_KEY
        self.base_url = settings.AGMARKNET_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AgriPriceAI-Collector/1.0",
            "Accept": "application/json"
        })

    def fetch_mandi_data(
        self,
        commodity: str,
        state: Optional[str] = None,
        limit: int = 1000,
        retries: int = 3,
        backoff_factor: float = 1.5
    ) -> pd.DataFrame:
        """
        Queries Agmarknet API on data.gov.in with retry and exponential backoff.
        """
        params = {
            "api-key": self.api_key,
            "format": "json",
            "offset": 0,
            "limit": limit,
            "filters[commodity]": commodity
        }
        if state:
            params["filters[state]"] = state

        for attempt in range(1, retries + 1):
            try:
                response = self.session.get(self.base_url, params=params, timeout=12)
                if response.status_code == 200:
                    data = response.json()
                    records = data.get("records", [])
                    if records:
                        return self._normalize_records(records)
                elif response.status_code in [429, 502, 503]:
                    time.sleep(backoff_factor ** attempt)
                    continue
                else:
                    break
            except (requests.RequestException, Exception) as exc:
                time.sleep(backoff_factor ** attempt)
                if attempt == retries:
                    print(f"[!] Agmarknet API unreachable or key expired ({exc}). Falling back to local data store.")

        return pd.DataFrame()

    def _normalize_records(self, raw_records: List[Dict]) -> pd.DataFrame:
        """Standardizes disparate Agmarknet column naming conventions."""
        cleaned = []
        for r in raw_records:
            try:
                arrival_date_str = r.get("arrival_date", "")
                parsed_date = datetime.strptime(arrival_date_str, "%d/%m/%Y").date() if arrival_date_str else datetime.today().date()
                cleaned.append({
                    "record_date": parsed_date,
                    "state": r.get("state"),
                    "district": r.get("district"),
                    "mandi_name": r.get("market"),
                    "commodity": r.get("commodity"),
                    "variety": r.get("variety", "Normal"),
                    "arrivals_qtl": float(r.get("arrivals_in_qtl", 0) or 0),
                    "min_price": float(r.get("min_price", 0) or 0),
                    "max_price": float(r.get("max_price", 0) or 0),
                    "modal_price": float(r.get("modal_price", 0) or 0)
                })
            except (ValueError, TypeError):
                continue
        return pd.DataFrame(cleaned)

if __name__ == "__main__":
    collector = AgmarknetCollector()
    df = collector.fetch_mandi_data(commodity="Onion", limit=10)
    print(f"Sample records fetched: {len(df)}")
