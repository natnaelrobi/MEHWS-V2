import requests
from typing import Dict, Any, Optional
from config import settings

def fetch_seasonal_climate(lat: float, lon: float, days: int = 180) -> Optional[Dict[str, Any]]:
    """
    Fetches long-range daily forecasts for precipitation and maximum temperature.
    """
    url = (
        f"{settings.OPEN_METEO_SEASONAL_URL}?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=precipitation_sum,temperature_2m_max"
        f"&forecast_days={days}&timezone=auto"
    )
    try:
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException:
        fallback_url = (
            f"{settings.OPEN_METEO_BASE_URL}?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=precipitation_sum,temperature_2m_max"
            f"&forecast_days={min(days, 16)}&timezone=auto"
        )
        try:
            res_fb = requests.get(fallback_url, timeout=10)
            res_fb.raise_for_status()
            return res_fb.json()
        except Exception:
            return None