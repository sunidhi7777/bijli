from __future__ import annotations
from datetime import date, timedelta
import requests
import pandas as pd

# Optional weather covariates. To avoid forecasting with future observed weather,
# this archive fetch is for historical EDA only. Live use needs weather forecasts.
CITIES = {"Andhra Pradesh": (16.5, 80.64), "Delhi": (28.61, 77.21),
          "Gujarat": (22.3, 70.8), "Karnataka": (12.97, 77.59),
          "Maharashtra": (19.08, 72.88), "Tamil Nadu": (13.08, 80.27),
          "Uttar Pradesh": (26.85, 80.95), "West Bengal": (22.57, 88.36)}

def fetch_weather(state: str, start: date, end: date) -> pd.DataFrame:
    if state not in CITIES:
        raise ValueError(f"No representative city configured for {state}")
    lat, lon = CITIES[state]
    params = {"latitude": lat, "longitude": lon, "start_date": start.isoformat(),
              "end_date": end.isoformat(), "daily": "temperature_2m_mean,precipitation_sum",
              "timezone": "Asia/Kolkata"}
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60)
    r.raise_for_status()
    data = r.json()["daily"]
    return pd.DataFrame({"date": pd.to_datetime(data["time"]), "temp_mean_c": data["temperature_2m_mean"],
                         "precipitation_mm": data["precipitation_sum"]})
