import pandas as pd
import requests
from datetime import datetime

def fetch_and_align_dekadal_data(lat, lon, forecast_days=180):
    """
    Fetches daily weather data from Open-Meteo and aggregates it into 
    exact CHIRPS-style dekadal intervals (1st, 11th, 21st of each month) 
    to match the feature distribution and cadence used during model training.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=precipitation_sum,temperature_2m_max"
        f"&timezone=Africa%2FNairobi"
        f"&forecast_days={forecast_days}"
    )
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            df_daily = pd.DataFrame(data['daily'])
            df_daily['time'] = pd.to_datetime(df_daily['time'])
            
            # Map each daily record to its corresponding dekad start date (1st, 11th, or 21st)
            def get_dekad_start(dt):
                day = dt.day
                if day <= 10:
                    return dt.replace(day=1)
                elif day <= 20:
                    return dt.replace(day=11)
                else:
                    return dt.replace(day=21)
            
            df_daily['dekad_date'] = df_daily['time'].apply(get_dekad_start)
            
            # Aggregate: Total rainfall sum and mean temperature per dekad
            df_dekad = df_daily.groupby('dekad_date').agg({
                'precipitation_sum': 'sum',
                'temperature_2m_max': 'mean'
            }).reset_index()
            
            df_dekad.rename(columns={'dekad_date': 'time'}, inplace=True)
            df_dekad.set_index('time', inplace=True)
            
            return df_dekad
            
    except Exception as e:
        print(f"API fetch error: {e}")
        
    return pd.DataFrame()
