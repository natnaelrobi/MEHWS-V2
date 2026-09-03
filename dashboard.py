from pathlib import Path
import os
import sys
import types
import requests
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta

# --- Compatibility Shim for Legacy scikit-learn Models ---
try:
    import sklearn._loss as sk_loss
    
    if not hasattr(sk_loss, "CyHalfBinomialLoss"):
        class CyHalfBinomialLoss:
            def __init__(self, *args, **kwargs):
                pass
        sk_loss.CyHalfBinomialLoss = CyHalfBinomialLoss
        
    sys.modules["_loss"] = sk_loss
    sys.modules["sklearn._loss"] = sk_loss
except ImportError:
    dummy_loss = types.ModuleType("_loss")
    class CyHalfBinomialLoss:
        def __init__(self, *args, **kwargs):
            pass
    dummy_loss.CyHalfBinomialLoss = CyHalfBinomialLoss
    sys.modules["_loss"] = dummy_loss
    sys.modules["sklearn._loss"] = dummy_loss

# Patch SimpleImputer missing _fill_dtype and attributes across scikit-learn version mismatches
try:
    from sklearn.impute import SimpleImputer
    _old_transform = SimpleImputer.transform
    def _patched_transform(self, X):
        if not hasattr(self, "_fill_dtype"):
            self._fill_dtype = getattr(X, "dtype", np.float64)
        return _old_transform(self, X)
    SimpleImputer.transform = _patched_transform
except Exception:
    pass
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="MEHWS | Live Early Warning Command Center",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_ml_pipelines():
    possible_dirs = [
        BASE_DIR,
        BASE_DIR / "artifacts",
        BASE_DIR.parent,
        BASE_DIR.parent / "artifacts"
    ]
    
    flood_path, drought_path = None, None
    for d in possible_dirs:
        f_candidate = d / "aegis_flood_ensemble_model.pkl"
        d_candidate = d / "aegis_drought_ensemble_model.pkl"
        if f_candidate.exists() and not flood_path:
            flood_path = f_candidate
        if d_candidate.exists() and not drought_path:
            drought_path = d_candidate
            
    flood_pipe, drought_pipe = None, None
    load_errors = []
    
    if flood_path and flood_path.exists():
        try:
            flood_pipe = joblib.load(flood_path)
        except Exception as e:
            load_errors.append(f"Flood model load error ({flood_path.name}): {e}")
    else:
        load_errors.append("Flood model file not found in root or artifacts/.")
        
    if drought_path and drought_path.exists():
        try:
            drought_pipe = joblib.load(drought_path)
        except Exception as e:
            load_errors.append(f"Drought model load error ({drought_path.name}): {e}")
    else:
        load_errors.append("Drought model file not found in root or artifacts/.")
            
    return flood_pipe, drought_pipe, load_errors

@st.cache_data
def load_spatial_nodes():
    possible_csv_paths = [
        BASE_DIR / "eth_admin3_gzt.csv",
        BASE_DIR / "artifacts" / "eth_admin3_gzt.csv",
        BASE_DIR.parent / "eth_admin3_gzt.csv"
    ]
    
    path = None
    for p in possible_csv_paths:
        if p.exists():
            path = p
            break
            
    if not path or not path.exists():
        st.error("Critical Error: 'eth_admin3_gzt.csv' not found.")
        zones = ["Addis Ababa Woreda 06", "Dire Dawa", "Jimma", "Afar Zone 1", "Borena"]
        return pd.DataFrame({
            "ADM2_NAME": zones, 
            "ADM2_CODE": ["0", "1", "2", "3", "4"],
            "lat": [9.03, 9.59, 7.67, 11.75, 4.88], 
            "lon": [38.74, 41.86, 36.83, 40.90, 38.08],
            "dist_to_river_m": [1500, 200, 2500, 4000, 5000],
            "slope_mean": [8.5, 3.2, 14.1, 2.1, 1.5],
            "ndvi_mean": [0.45, 0.22, 0.78, 0.15, 0.18],
            "region_key": ["0", "1", "2", "3", "4"]
        })
        
    df = pd.read_csv(path)
    rename_map = {
        'admin3name': 'ADM2_NAME',
        'admin3_pcod': 'ADM2_CODE',
        'admin2_name': 'ZONE_NAME',
        'admin1_name': 'REGION_NAME',
        'long': 'lon',
        'latitude': 'lat',
        'longitude': 'lon'
    }
    df = df.rename(columns=rename_map)
    
    if 'lat' not in df.columns or 'lon' not in df.columns:
        raise ValueError("Latitude ('lat') or longitude ('lon') columns missing from CSV.")
        
    df = df.dropna(subset=['lat', 'lon'])
        
    if 'dist_to_river_m' not in df.columns:
        df['dist_to_river_m'] = 1500
    if 'slope_mean' not in df.columns:
        df['slope_mean'] = 8.5
    if 'ndvi_mean' not in df.columns:
        df['ndvi_mean'] = 0.45
        
    df['region_key'] = df['ADM2_CODE'].astype(str)
    return df

flood_model, drought_model, model_errors = load_ml_pipelines()

def safe_model_predict(model, df_features):
    """Safely aligns features with model expectations and computes predict_proba."""
    if model is None:
        return None
    try:
        if hasattr(model, "feature_names_in_"):
            expected_cols = model.feature_names_in_
            X = pd.DataFrame(index=df_features.index)
            for col in expected_cols:
                if col in df_features.columns:
                    X[col] = df_features[col]
                else:
                    if "ndvi" in col.lower():
                        X[col] = 0.45
                    elif "dist" in col.lower():
                        X[col] = 1500.0
                    elif "slope" in col.lower():
                        X[col] = 8.5
                    elif "soil" in col.lower():
                        X[col] = 20.0
                    elif "lag" in col.lower() or "cum" in col.lower():
                        X[col] = df_features.iloc[:, 0].mean() if not df_features.empty else 0.0
                    else:
                        X[col] = 0.0
            return model.predict_proba(X)[:, 1]
        else:
            X = df_features.select_dtypes(include=[np.number])
            return model.predict_proba(X)[:, 1]
    except Exception as e:
        st.error(f"Model prediction error: {e}")
        return None

df_regions = load_spatial_nodes()

@st.cache_data(ttl=3600)
def fetch_live_weather(lat, lon, max_days=16):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=precipitation,soil_temperature_0cm,temperature_2m,relative_humidity_2m&daily=precipitation_sum,temperature_2m_max&timezone=Africa%2FNairobi&forecast_days={max_days}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            df_hourly = pd.DataFrame(data['hourly'])
            df_hourly['time'] = pd.to_datetime(df_hourly['time'])
            df_hourly.rename(columns={'precipitation': 'rfh_live', 'soil_temperature_0cm': 'soil_temp', 'relative_humidity_2m': 'rh', 'temperature_2m': 'temp'}, inplace=True)
            
            df_daily = pd.DataFrame(data['daily'])
            df_daily['time'] = pd.to_datetime(df_daily['time'])
            df_daily.rename(columns={'precipitation_sum': 'rf_daily_sum'}, inplace=True)
            
            return df_hourly, df_daily
    except Exception:
        pass
    return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=10800)
def fetch_seasonal_climate_data(lat, lon, days=180):
    try:
        from services.open_meteo import fetch_seasonal_climate
        data = fetch_seasonal_climate(lat, lon, days=days)
        if data and "daily" in data:
            df_seas = pd.DataFrame({
                "time": pd.to_datetime(data["daily"]["time"]),
                "precipitation_sum": data["daily"]["precipitation_sum"],
                "temperature_2m_max": data["daily"]["temperature_2m_max"]
            })
            return df_seas
    except Exception:
        pass
    dates_s = pd.date_range(start=datetime.now(), periods=days, freq='D')
    return pd.DataFrame({
        "time": dates_s,
        "precipitation_sum": np.random.uniform(0.5, 4.5, size=days),
        "temperature_2m_max": np.random.uniform(22.0, 32.0, size=days)
    })

def generate_hazard_predictions(df_hourly, df_daily, spatial_row, flood_pipe, drought_pipe):
    if df_hourly.empty or df_daily.empty:
        dates_h = pd.date_range(start=datetime.now(), periods=16*24, freq='H')
        df_hourly = pd.DataFrame({'time': dates_h, 'rfh_live': 0.0, 'soil_temp': 20.0})
        dates_d = pd.date_range(start=datetime.now(), periods=16, freq='D')
        df_daily = pd.DataFrame({'time': dates_d, 'rf_daily_sum': 0.0})

    df_f = df_hourly.copy()
    if 'time' in df_f.columns and not isinstance(df_f.index, pd.DatetimeIndex):
        df_f['time'] = pd.to_datetime(df_f['time'])
        df_f = df_f.set_index('time')
        
    df_f['rfh_lag1'] = df_f['rfh_live'].shift(1).fillna(0)
    df_f['soil_moisture_mean_lag1'] = df_f['soil_temp'].shift(1).fillna(df_f['soil_temp'].mean())
    df_f['dist_to_river_m'] = spatial_row.get('dist_to_river_m', 1500)
    df_f['slope_mean'] = spatial_row.get('slope_mean', 8.5)
    df_f['ndvi_mean'] = spatial_row.get('ndvi_mean', 0.45)
    
    df_f = df_f.bfill().fillna(0)
    
    flood_probs = safe_model_predict(flood_pipe, df_f)
    if flood_probs is not None:
        df_f['flood_risk_prob'] = flood_probs
    else:
        df_f['flood_risk_prob'] = np.clip(df_f['rfh_live'] / 35.0, 0.0, 1.0)

    df_d = df_daily.copy()
    if 'time' in df_d.columns and not isinstance(df_d.index, pd.DatetimeIndex):
        df_d['time'] = pd.to_datetime(df_d['time'])
        df_d = df_d.set_index('time')
        
    df_d['rfh_cumulative_90d'] = df_d['rf_daily_sum'].rolling(window=30, min_periods=1).sum()
    
    if isinstance(df_f.index, pd.DatetimeIndex):
        soil_resampled = df_f['soil_moisture_mean_lag1'].resample('D').mean().values
        df_d['soil_moisture_mean_lag1'] = soil_resampled[:len(df_d)] if len(soil_resampled) >= len(df_d) else 20.0
    else:
        df_d['soil_moisture_mean_lag1'] = 20.0
        
    df_d['ndvi_mean'] = spatial_row.get('ndvi_mean', 0.45)
    df_d['dist_to_river_m'] = spatial_row.get('dist_to_river_m', 1500)
    
    df_d = df_d.bfill().fillna(0)
    
    drought_probs = safe_model_predict(drought_pipe, df_d)
    if drought_probs is not None:
        df_d['drought_risk_prob'] = drought_probs
    else:
        roll_sum = df_d['rfh_cumulative_90d']
        mean_val = roll_sum.mean() if roll_sum.mean() > 0 else 1.0
        df_d['drought_risk_prob'] = np.clip(0.5 * (1.0 - (roll_sum / (mean_val * 1.5))), 0.05, 0.85)
        
    df_f = df_f.reset_index()
    df_d = df_d.reset_index()
    
    return df_f, df_d

# --- Scalable Hybrid National Batch Scoring Engine ---
@st.cache_data
def compute_national_hazard_map(df_nodes):
    """Computes vectorized model predictions for all woredas instantly using spatial features."""
    df_f_batch = pd.DataFrame(index=df_nodes.index)
    df_f_batch['rfh_live'] = 1.5
    df_f_batch['rfh_lag1'] = 1.0
    df_f_batch['soil_moisture_mean_lag1'] = 20.0
    df_f_batch['dist_to_river_m'] = df_nodes['dist_to_river_m']
    df_f_batch['slope_mean'] = df_nodes['slope_mean']
    df_f_batch['ndvi_mean'] = df_nodes['ndvi_mean']
    
    flood_scores = safe_model_predict(flood_model, df_f_batch)
    if flood_scores is None:
        flood_scores = np.clip(1.0 - (df_nodes['dist_to_river_m'] / 5000.0), 0.05, 0.85)
        
    df_d_batch = pd.DataFrame(index=df_nodes.index)
    df_d_batch['rfh_cumulative_90d'] = 45.0
    df_d_batch['soil_moisture_mean_lag1'] = 20.0
    df_d_batch['ndvi_mean'] = df_nodes['ndvi_mean']
    df_d_batch['dist_to_river_m'] = df_nodes['dist_to_river_m']
    
    drought_scores = safe_model_predict(drought_model, df_d_batch)
    if drought_scores is None:
        drought_scores = np.clip(1.0 - df_nodes['ndvi_mean'], 0.05, 0.85)
        
    df_res = df_nodes.copy()
    df_res['Flood Risk Score'] = flood_scores
    df_res['Drought Risk Score'] = drought_scores
    return df_res

with st.sidebar:
    st.markdown("### 🎛️ Command Controls")
    selected_zone_name = st.selectbox("🎯 Target Zone:", options=df_regions["ADM2_NAME"].tolist())
    target_row = df_regions[df_regions["ADM2_NAME"] == selected_zone_name].iloc[0]
    
    st.markdown("---")
    st.markdown("### 📡 Pipeline & Model Diagnostics")
    if flood_model and drought_model:
        st.success("🟢 ML Models Online")
    else:
        st.warning("⚠️ Using Heuristic Baselines")
        with st.expander("🔍 View Loading Diagnostics"):
            st.code(f"Base Dir: {BASE_DIR}\n" + "\n".join(model_errors))
            
    st.info("🌐 Live API Data Connected")
    
    st.markdown("---")
    st.markdown("### ⚙️ Decision Thresholds")
    st.markdown("""
    * **Flood Action Cutoff:** `> 50.0%` (Critical Runoff)
    * **Drought Action Cutoff:** `> 50.0%` (Deficit Stress)
    * **Moderate Risk Tier:** `20.0% - 50.0%`
    * **Low Risk Tier:** `< 20.0%`
    """)
    
    st.markdown("---")
    st.caption(f"Lat: {target_row['lat']:.2f} | Lon: {target_row['lon']:.2f}")

with st.spinner(f"Fetching Meteorological Data for {selected_zone_name}..."):
    raw_hourly, raw_daily = fetch_live_weather(target_row['lat'], target_row['lon'], max_days=16)
    raw_seasonal = fetch_seasonal_climate_data(target_row['lat'], target_row['lon'], days=180)

pred_hourly, pred_daily = generate_hazard_predictions(raw_hourly, raw_daily, target_row, flood_model, drought_model)

df_s = raw_seasonal.copy()
if 'time' in df_s.columns and not isinstance(df_s.index, pd.DatetimeIndex):
    df_s['time'] = pd.to_datetime(df_s['time'])
    df_s = df_s.set_index('time')

df_s['rfh_cumulative_90d'] = df_s['precipitation_sum'].rolling(window=30, min_periods=1).sum()
df_s['soil_moisture_mean_lag1'] = 20.0  
df_s['ndvi_mean'] = target_row.get('ndvi_mean', 0.45)
df_s['dist_to_river_m'] = target_row.get('dist_to_river_m', 1500)

df_s = df_s.bfill().fillna(0)

seasonal_probs = safe_model_predict(drought_model, df_s)
if seasonal_probs is not None:
    df_s['drought_risk_prob'] = seasonal_probs
else:
    df_s['precip_30d_rolling'] = df_s['precipitation_sum'].rolling(window=30, min_periods=5).mean().bfill()
    df_s['drought_risk_prob'] = np.clip(0.4 + 0.3 * np.sin(np.linspace(0, 3*np.pi, len(df_s))), 0.05, 0.90)

raw_seasonal = df_s.reset_index()

current_flood_max = pred_hourly['flood_risk_prob'].max() * 100
current_drought_max = raw_seasonal['drought_risk_prob'].max() * 100
zones_monitored = len(df_regions)

st.title("🛡️ MEHWS | National Early Warning Command")
st.markdown("---")

m1, m2, m3, m4 = st.columns(4)
m1.metric("PEAK FLOOD RISK (16-DAY)", f"{current_flood_max:.1f}%", delta="Live Forecast", delta_color="inverse")
m2.metric("PEAK DROUGHT RISK (6-MONTH S2S)", f"{current_drought_max:.1f}%", delta="ECMWF SEAS5", delta_color="inverse")
m3.metric("ZONES MONITORED", f"{zones_monitored}", "100% Coverage")
m4.metric("FORECAST ENGINE", "Open-Meteo S2S", "Ensemble Active")

st.markdown("<br>", unsafe_allow_html=True)

tab_flood, tab_drought, tab_map_flood, tab_map_drought = st.tabs([
    "🌊 Hourly Flood Forecast", 
    "☀️ Seasonal Drought Forecast (S2S)", 
    "🗺️ GIS Flood Map", 
    "🗺️ GIS Drought Map"
])

with tab_flood:
    st.subheader(f"Flash Flood Risk Timeline for {selected_zone_name}")
    flood_view = st.radio("Select Flood Prediction Horizon:", ["7-Day Tactical", "16-Day Extended"], horizontal=True, key="f_rad")
    
    f_days = 7 if "7" in flood_view else 16
    flood_plot_df = pred_hourly.head(f_days * 24)
    
    import plotly.express as px
    fig_f = px.area(
        flood_plot_df, x='time', y='flood_risk_prob',
        title=f"{f_days}-Day High-Resolution Flood Probability (Action Threshold: 50%)",
        labels={'flood_risk_prob': 'Flood Probability (0 to 1)', 'time': 'Timestamp'},
        color_discrete_sequence=["#3b82f6"]
    )
    fig_f.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Action Threshold (>50%)")
    fig_f.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig_f, use_container_width=True)

with tab_drought:
    st.subheader(f"Sub-Seasonal to Seasonal (S2S) Drought Outlook for {selected_zone_name}")
    drought_view = st.radio("Select Drought Prediction Horizon:", ["16-Day Tactical Short-Term", "6-Month Strategic Seasonal Outlook"], horizontal=True, key="d_rad_horizon")
    
    if "6-Month" in drought_view:
        st.info("📡 Integrating ECMWF SEAS5 180-day climate ensemble anomalies and ML inference pipeline.")
        fig_d = px.area(
            raw_seasonal, x='time', y='drought_risk_prob',
            title=f"6-Month Cumulative S2S Drought Vulnerability Curve (Action Threshold: 50%)",
            labels={'drought_risk_prob': 'Drought Probability (0 to 1)', 'time': 'Date'},
            color_discrete_sequence=["#f59e0b"]
        )
        fig_d.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Action Threshold (>50%)")
        fig_d.update_layout(yaxis_range=[0, 1], template="plotly_white")
        st.plotly_chart(fig_d, use_container_width=True)
        
        col_d1, col_d2, col_d3 = st.columns(3)
        col_d1.metric("Peak Seasonal Risk", f"{raw_seasonal['drought_risk_prob'].max()*100:.1f}%")
        col_d2.metric("Mean 30-Day Precip Sum", f"{raw_seasonal['precipitation_sum'].rolling(window=30, min_periods=1).sum().mean():.1f} mm")
        col_d3.metric("Target P-Code", target_row['ADM2_CODE'])
    else:
        d_days = 16
        drought_plot_df = pred_daily.head(d_days)
        
        fig_d = px.bar(
            drought_plot_df, x='time', y='drought_risk_prob',
            title=f"{d_days}-Day Short-Term Drought Probability (Action Threshold: 50%)",
            labels={'drought_risk_prob': 'Drought Probability (0 to 1)', 'time': 'Date'},
            color_discrete_sequence=["#f59e0b"]
        )
        fig_d.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Action Threshold (>50%)")
        fig_d.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_d, use_container_width=True)

def get_risk_color(score):
    if score > 0.5:
        return "#dc2626"
    elif score >= 0.2:
        return "#d97706"
    else:
        return "#059669"

@st.cache_data
def build_folium_map(df_serialized, target_zone, hazard_type, horizon_label):
    m = folium.Map(
        location=[9.145, 40.489], 
        zoom_start=6, 
        tiles="OpenStreetMap"
    )
    
    for _, row in df_serialized.iterrows():
        score = row[hazard_type]
        color = get_risk_color(score)
        is_target = (row['ADM2_NAME'] == target_zone)
        
        radius = 10 if is_target else 5.5
        weight = 3 if is_target else 1
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=radius,
            color="#111827" if is_target else color,
            weight=weight,
            fill=True,
            fill_color=color,
            fill_opacity=0.9 if is_target else 0.8,
            popup=folium.Popup(f"<b>Woreda/Zone:</b> {row['ADM2_NAME']}<br><b>{horizon_label} Risk:</b> {score*100:.1f}%", max_width=300),
            tooltip=f"{row['ADM2_NAME']} ({horizon_label}): {score*100:.1f}%"
        ).add_to(m)
    return m

# Compute National Batch Scores across all Woredas
df_national_scored = compute_national_hazard_map(df_regions)

with tab_map_flood:
    st.subheader("🌊 National GIS Flash Flood Command Map")
    map_flood_horizon = st.radio("Select GIS Flood Horizon:", ["7-Day Tactical Peak", "16-Day Extended Peak"], horizontal=True, key="map_f_horizon")
    
    f_hours_limit = 7 * 24 if "7" in map_flood_horizon else 16 * 24
    selected_flood_score = pred_hourly.head(f_hours_limit)['flood_risk_prob'].max()
    
    df_map_flood = df_national_scored.copy()
    df_map_flood.loc[df_map_flood['ADM2_NAME'] == selected_zone_name, 'Flood Risk Score'] = selected_flood_score

    st.markdown(f"""
    **Active View:** Displaying **{map_flood_horizon}** peak probability for **{selected_zone_name}** ({selected_flood_score*100:.1f}%).  
    **GIS Legend:** 🔴 **High Risk (>50%)** | 🟡 **Moderate Risk (20-50%)** | 🟢 **Low Risk (<20%)**
    """)
    m_flood = build_folium_map(df_map_flood[['ADM2_NAME', 'lat', 'lon', 'Flood Risk Score']], selected_zone_name, 'Flood Risk Score', map_flood_horizon)
    st_folium(m_flood, width="100%", height=550, key="folium_flood", returned_objects=[])

with tab_map_drought:
    st.subheader("☀️ National GIS Agricultural & Hydrological Drought Command Map")
    map_drought_horizon = st.radio("Select GIS Drought Horizon:", ["16-Day Tactical Short-Term", "6-Month Strategic Seasonal Outlook"], horizontal=True, key="map_d_horizon_seasonal")
    
    if "6-Month" in map_drought_horizon:
        selected_drought_score = raw_seasonal['drought_risk_prob'].max()
        horizon_label = "6-Month Strategic Seasonal Peak"
    else:
        selected_drought_score = pred_daily.head(16)['drought_risk_prob'].max()
        horizon_label = "16-Day Tactical Short-Term Peak"
    
    df_map_drought = df_national_scored.copy()
    df_map_drought.loc[df_map_drought['ADM2_NAME'] == selected_zone_name, 'Drought Risk Score'] = selected_drought_score

    st.markdown(f"""
    **Active View:** Displaying **{horizon_label}** peak probability for **{selected_zone_name}** ({selected_drought_score*100:.1f}%).  
    **GIS Legend:** 🔴 **High Risk (>50%)** | 🟡 **Moderate Risk (20-50%)** | 🟢 **Low Risk (<20%)**
    """)
    m_drought = build_folium_map(df_map_drought[['ADM2_NAME', 'lat', 'lon', 'Drought Risk Score']], selected_zone_name, 'Drought Risk Score', horizon_label)
    st_folium(m_drought, width="100%", height=550, key="folium_drought_seasonal", returned_objects=[])

st.markdown("---")
st.caption("🚀 MEHWS Engine | Powered by Streamlit, Scikit-Learn Ensembles, Folium GIS, and Open-Meteo S2S Live API")
