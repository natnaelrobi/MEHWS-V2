"""
real_feature_engineering.py

Replaces the feature construction currently done inline in dashboard.py
(fetch_live_weather, convert_daily_to_dekadal, generate_dekadal_hazard_predictions,
compute_national_hazard_map, safe_model_predict's default-fill branch).

Goal: every one of the 27 features the models were trained on is either:
  (a) computed from REAL live/historical data, or
  (b) explicitly left as NaN so the model's own SimpleImputer(strategy='median')
      -- which already has the correct training-set medians baked in -- fills it,
      instead of a guessed hardcoded constant.

Drop-in usage from dashboard.py:
    from services.real_feature_engineering import (
        TRAINED_ZONE_PCODES, fetch_real_weather, build_model_features
    )
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
CLIMATOLOGY_PATH = BASE_DIR / "zone_dekad_climatology.csv"

# ---------------------------------------------------------------------------
# 1. Zones the models actually saw during training.
#    34 of 63 admin2 zones were silently dropped during training (missing
#    slope/soil/ndvi/river-distance data broke the lag feature dropna step).
#    Predicting for anything outside this list is out-of-distribution --
#    the model has zero examples from that region. Filter the zone selector
#    in dashboard.py to these pcodes, or clearly flag predictions outside it.
# ---------------------------------------------------------------------------
TRAINED_ZONE_PCODES = {
    # pcode: adm2_name  (fill in / verify against your eth_admin3_gzt.csv join)
    # Populate this from the training CSV: df2['pcode'].unique() after the
    # notebook's dropna step. Kept as a placeholder set here -- see
    # get_trained_zone_pcodes() below to regenerate it directly from your CSV.
}


def get_trained_zone_pcodes(training_csv_path: str) -> set:
    """Run this once against your original training CSV to get the exact
    pcode list the models were actually fit on, then hardcode the result
    into TRAINED_ZONE_PCODES above (or load it at startup)."""
    df = pd.read_csv(training_csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["pcode", "date"])
    for col in ["rfh", "r1h", "r3h", "soil_moisture_mean", "ndvi_mean"]:
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = df.groupby("pcode")[col].shift(lag)
    df = df.dropna(subset=[f"{c}_lag3" for c in
                            ["rfh", "r1h", "r3h", "soil_moisture_mean", "ndvi_mean"]])
    return set(df["pcode"].unique())


# ---------------------------------------------------------------------------
# 2. Climatology lookup: real per-zone, per-dekad-of-year rainfall normals,
#    computed directly from your training CSV (see zone_dekad_climatology.csv).
#    This is what `rfh_avg` actually means -- it CANNOT come from a weather
#    forecast API, only from your own historical record.
# ---------------------------------------------------------------------------
_climatology_cache = None

def load_climatology() -> pd.DataFrame:
    global _climatology_cache
    if _climatology_cache is None:
        _climatology_cache = pd.read_csv(CLIMATOLOGY_PATH)
    return _climatology_cache


def dekad_key_for_date(dt: pd.Timestamp) -> str:
    dekad = 1 if dt.day <= 10 else (2 if dt.day <= 20 else 3)
    return f"{dt.month:02d}-{dekad}"


def lookup_climatology(pcode: str, dt: pd.Timestamp) -> dict:
    """Real rfh_avg / r1h_avg / r3h_avg for this zone at this point in the year."""
    clim = load_climatology()
    key = dekad_key_for_date(dt)
    row = clim[(clim["pcode"] == pcode) & (clim["dekad_key"] == key)]
    if row.empty:
        return {"rfh_avg": np.nan, "r1h_avg_climatology": np.nan, "r3h_avg_climatology": np.nan}
    r = row.iloc[0]
    return {
        "rfh_avg": r["rfh_avg_climatology"],
        "r1h_avg_climatology": r["r1h_avg_climatology"],
        "r3h_avg_climatology": r["r3h_avg_climatology"],
    }


# ---------------------------------------------------------------------------
# 3. Real weather fetch.
#    Key fix: use `past_days` + `forecast_days` together in ONE call to
#    Open-Meteo's forecast endpoint. This gives REAL observed/reanalysis
#    rainfall for the past ~90 days (needed to compute r1h/r3h correctly)
#    PLUS a genuinely skillful forecast for the next up-to-16 days.
#    Do NOT request forecast_days=180 -- Open-Meteo's forecast endpoint caps
#    at 16 days; that parameter silently fails to deliver 6 months of data.
# ---------------------------------------------------------------------------
def fetch_real_weather(lat: float, lon: float, past_days: int = 90, forecast_days: int = 16) -> pd.DataFrame:
    forecast_days = min(forecast_days, 16)   # hard API ceiling -- do not exceed
    past_days = min(past_days, 92)           # Open-Meteo's practical ceiling for this param

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=precipitation_sum,soil_moisture_0_to_7cm_mean"
        f"&past_days={past_days}&forecast_days={forecast_days}"
        "&timezone=Africa%2FNairobi"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()["daily"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    return df


def aggregate_to_dekads(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Real dekadal aggregation: sum of rain per dekad -> this IS 'rfh'."""
    df = df_daily.copy()
    df["dekad_date"] = df["time"].apply(
        lambda d: d.replace(day=1) if d.day <= 10 else
                  (d.replace(day=11) if d.day <= 20 else d.replace(day=21))
    )
    agg = {"precipitation_sum": "sum"}
    if "soil_moisture_0_to_7cm_mean" in df.columns:
        agg["soil_moisture_0_to_7cm_mean"] = "mean"
    df_dekad = df.groupby("dekad_date").agg(agg).reset_index()
    df_dekad = df_dekad.rename(columns={"dekad_date": "time", "precipitation_sum": "rfh"})
    df_dekad = df_dekad.sort_values("time").reset_index(drop=True)

    # r1h = rolling 3-dekad (~1 month) sum, r3h = rolling 9-dekad (~3 month) sum
    df_dekad["r1h"] = df_dekad["rfh"].rolling(window=3, min_periods=1).sum()
    df_dekad["r3h"] = df_dekad["rfh"].rolling(window=9, min_periods=1).sum()

    # real lags (previous actual dekads, not guessed constants)
    for lag in [1, 2, 3]:
        df_dekad[f"rfh_lag{lag}"] = df_dekad["rfh"].shift(lag)
        df_dekad[f"r1h_lag{lag}"] = df_dekad["r1h"].shift(lag)
        df_dekad[f"r3h_lag{lag}"] = df_dekad["r3h"].shift(lag)

    return df_dekad


# ---------------------------------------------------------------------------
# 4. Full feature builder matching the model's 27 expected columns.
#    Anything genuinely unavailable (SFED, RP, SFED_BASELINE, NDVI lags
#    beyond the latest known static value) is left as NaN on purpose --
#    let the pipeline's own imputer (fit on real training medians) fill it,
#    rather than overwriting with a guessed constant.
# ---------------------------------------------------------------------------
def build_model_features(lat: float, lon: float, pcode: str, spatial_row: dict) -> pd.DataFrame:
    df_daily = fetch_real_weather(lat, lon)
    df_dekad = aggregate_to_dekads(df_daily)

    # join real climatology per dekad-of-year for this zone
    clim_rows = df_dekad["time"].apply(lambda d: lookup_climatology(pcode, d))
    clim_df = pd.DataFrame(list(clim_rows))
    df_dekad["rfh_avg"] = clim_df["rfh_avg"]

    # static terrain -- these ARE real, already correctly available per zone
    df_dekad["slope_mean"] = spatial_row.get("slope_mean", np.nan)
    df_dekad["dist_to_river_m"] = spatial_row.get("dist_to_river_m", np.nan)
    df_dekad["area_sqkm"] = spatial_row.get("area_sqkm", np.nan)

    # live soil moisture from Open-Meteo, if present, else NaN (imputer handles it)
    if "soil_moisture_0_to_7cm_mean" in df_dekad.columns:
        sm = df_dekad["soil_moisture_0_to_7cm_mean"]
        df_dekad["soil_moisture_mean"] = sm
        for lag in [1, 2, 3]:
            df_dekad[f"soil_moisture_mean_lag{lag}"] = sm.shift(lag)
    else:
        for col in ["soil_moisture_mean", "soil_moisture_mean_lag1",
                    "soil_moisture_mean_lag2", "soil_moisture_mean_lag3"]:
            df_dekad[col] = np.nan

    # NDVI: no live feed wired up. Use the latest known static per-zone value
    # for 'ndvi_mean' (reasonable -- NDVI moves slowly), leave lags as NaN
    # rather than repeating the same guessed number 3 times.
    df_dekad["ndvi_mean"] = spatial_row.get("ndvi_mean", np.nan)
    df_dekad["ndvi_mean_lag1"] = np.nan
    df_dekad["ndvi_mean_lag2"] = np.nan
    df_dekad["ndvi_mean_lag3"] = np.nan

    # FloodScan-derived features: genuinely not obtainable from a weather API.
    # Leave NaN so the imputer's real training median is used, OR wire in a
    # live FloodScan/GloFAS feed if you have API access to one.
    df_dekad["SFED"] = np.nan
    df_dekad["RP"] = np.nan
    df_dekad["SFED_BASELINE"] = np.nan

    return df_dekad
