import streamlit as st
import pandas as pd
import requests
from datetime import datetime

st.set_page_config(
    page_title="Live Multi-Hazard Early Warning System (MEHWS)", 
    page_icon="🌍", 
    layout="wide"
)

st.title("🌍 Live Multi-Hazard Early Warning System (MEHWS)")
st.markdown("Real-time geospatial risk monitoring powered by live meteorological data feeds.")

# --- Sidebar Controls ---
st.sidebar.header("Monitoring Parameters")
flood_thresh = st.sidebar.slider("Flood Risk Action Threshold", 0.0, 1.0, 0.5, 0.05)
drought_thresh = st.sidebar.slider("Drought Risk Action Threshold", 0.0, 1.0, 0.5, 0.05)

# --- Define Monitored Target Stations/Zones ---
@st.cache_data
def get_station_metadata():
    return [
        {"ADM2_NAME": "Dire Dawa Administration", "lat": 9.5936, "lon": 41.8661},
        {"ADM2_NAME": "Addis Ababa - Woreda 06", "lat": 9.0192, "lon": 38.7525},
        {"ADM2_NAME": "Afar - Zone 1", "lat": 11.7500, "lon": 41.0000},
        {"ADM2_NAME": "Somali - Gode", "lat": 5.9500, "lon": 43.5667},
        {"ADM2_NAME": "Oromia - Borana", "lat": 4.9000, "lon": 38.0833},
        {"ADM2_NAME": "Amhara - Bahir Dar", "lat": 11.5937, "lon": 37.3908}
    ]

# --- Fetch Real-Time Meteorological Data from Open-Meteo API ---
@st.cache_data(ttl=3600)
def fetch_live_weather_data(stations):
    live_records = []
    for station in stations:
        lat = station["lat"]
        lon = station["lon"]
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=precipitation,rain&daily=precipitation_sum&timezone=auto"
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                current_precip = data.get("current", {}).get("precipitation", 0.0) or 0.0
                daily_list = data.get("daily", {}).get("precipitation_sum", [0.0])
                recent_precip_sum = sum([p for p in daily_list if p is not None]) if daily_list else 0.0
                
                # Dynamic risk scoring heuristic based on real precipitation data
                # Normalizing values for demonstration: high precipitation -> higher flood risk; prolonged zero/low precipitation -> drought risk
                flood_score = min(float(current_precip) / 15.0 + float(recent_precip_sum) / 100.0, 1.0)
                drought_score = max(1.0 - (float(recent_precip_sum) / 20.0), 0.0) if recent_precip_sum < 5.0 else 0.1
            else:
                current_precip, recent_precip_sum, flood_score, drought_score = 0.0, 0.0, 0.0, 0.0
        except Exception:
            current_precip, recent_precip_sum, flood_score, drought_score = 0.0, 0.0, 0.0, 0.0

        live_records.append({
            "ADM2_NAME": station["ADM2_NAME"],
            "lat": lat,
            "lon": lon,
            "Current Precip (mm)": current_precip,
            "Recent Precip Sum (mm)": recent_precip_sum,
            "Flood Risk Score": round(flood_score, 2),
            "Drought Risk Score": round(drought_score, 2)
        })
    return pd.DataFrame(live_records)

stations = get_station_metadata()
with st.spinner("Fetching live meteorological data from global weather feeds..."):
    df_national = fetch_live_weather_data(stations)

# --- Dashboard Layout & Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Active Monitored Nodes", len(df_national))
col2.metric("Flood Action Triggers", len(df_national[df_national['Flood Risk Score'] >= flood_thresh]))
col3.metric("Drought Action Triggers", len(df_national[df_national['Drought Risk Score'] >= drought_thresh]))

st.markdown("### 🗺️ Live Spatial Risk Map")
st.map(df_national[['lat', 'lon']])

st.markdown("### 📊 Real-Time Meteorological & Risk Matrix")
display_df = df_national.copy()
display_df['Flood Risk (%)'] = (display_df['Flood Risk Score'] * 100).round(1)
display_df['Drought Risk (%)'] = (display_df['Drought Risk Score'] * 100).round(1)
st.dataframe(
    display_df[['ADM2_NAME', 'Current Precip (mm)', 'Recent Precip Sum (mm)', 'Flood Risk (%)', 'Drought Risk (%)']], 
    use_container_width=True
)

# --- Action Section ---
st.markdown("---")
col_action1, col_action2 = st.columns(2)

with col_action1:
    st.subheader("🔍 Live Threshold Analysis")
    if st.button("Evaluate Critical Hazard Zones", type="primary"):
        critical_floods = df_national[df_national['Flood Risk Score'] >= flood_thresh]
        critical_droughts = df_national[df_national['Drought Risk Score'] >= drought_thresh]
        
        if not critical_floods.empty or not critical_droughts.empty:
            st.warning("Action required in the following monitored zones based on live data:")
            for _, row in critical_floods.iterrows():
                st.write(- **{row['ADM2_NAME']}**: Flash Flood Risk at **{row['Flood Risk Score']*100:.1f}%** (Precip: {row['Current Precip (mm)']} mm))
            for _, row in critical_droughts.iterrows():
                st.write(- **{row['ADM2_NAME']}**: Agricultural Drought Risk at **{row['Drought Risk Score']*100:.1f}%**)
        else:
            st.success("All live stations are currently operating under safe thresholds.")

with col_action2:
    st.subheader("📥 Live Situation Report Export")
    report_df = df_national.copy()
    report_df['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_bytes = report_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Live SitRep Data (CSV)",
        data=csv_bytes,
        file_name=f"MEHWS_Live_SitRep_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
