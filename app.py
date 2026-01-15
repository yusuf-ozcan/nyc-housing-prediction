import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import zipfile

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="NYC AI Market Value Estimator", 
    layout="wide", 
    page_icon="🏙️"
)

# --- 2. LOAD PRE-TRAINED ZIPPED MODEL ---
@st.cache_resource
def load_trained_model():
    # Dosya yollarını tanımlıyoruz
    zip_path = 'models/nyc_house_model.pkl.zip'
    model_path = 'models/nyc_house_model.pkl'
    
    # models klasörü yoksa oluştur
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Eğer .pkl dosyası henüz çıkarılmamışsa zip'ten çıkar
    if not os.path.exists(model_path):
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('models')
        else:
            # Eğer zip dosyası da yoksa hata ver
            st.error("⚠️ Critical Error: 'models/nyc_house_model.pkl.zip' not found on GitHub!")
            st.stop()
    
    return joblib.load(model_path)

# Modeli hafızaya yükle
model = load_trained_model()

# --- 3. UI DESIGN (Aynı kalıyor) ---
BOROUGH_MAP = {
    "Manhattan": {"lat": 40.7580, "lon": -73.9855, "code": "MN"},
    "Brooklyn": {"lat": 40.6782, "lon": -73.9442, "code": "BK"},
    "Queens": {"lat": 40.7282, "lon": -73.7949, "code": "QN"},
    "Bronx": {"lat": 40.8448, "lon": -73.8648, "code": "BX"},
    "Staten Island": {"lat": 40.5795, "lon": -74.1502, "code": "SI"}
}

st.title("🏙️ NYC AI Market Value Estimator")
st.markdown("Professional Real estate valuation based on NYC Open Data and High-Performance Random Forest.")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 Location & Property Type")
    selected_b_name = st.selectbox("Select Borough", list(BOROUGH_MAP.keys()))
    
    default_lat = BOROUGH_MAP[selected_b_name]["lat"]
    default_lon = BOROUGH_MAP[selected_b_name]["lon"]
    b_code = BOROUGH_MAP[selected_b_name]["code"]

    land_use = st.selectbox("Land Use Category", options=[1, 2, 3, 4], 
                            format_func=lambda x: {1:"Residential-1 Family", 2:"Residential-Multi", 3:"Mixed Use", 4:"Commercial"}.get(x))
    
    with st.expander("Fine-tune Coordinates"):
        u_lat = st.number_input("Latitude", value=default_lat, format="%.4f")
        u_lon = st.number_input("Longitude", value=default_lon, format="%.4f")

with col2:
    st.subheader("🏗️ Structural Specifications")
    bldg_area = st.number_input("Building Area (sqft)", value=2200, min_value=10)
    lot_area = st.number_input("Lot Area (sqft)", value=2500, min_value=10)
    
    min_u = 1 if land_use in [1, 2] else 0
    units = st.number_input("Residential Units", min_value=min_u, value=max(1, min_u))
    
    floors = st.slider("Number of Floors", 1, 120, 2)
    age = st.slider("Building Age (Years)", 0, 250, 45)

# Inference Logic
area_per_unit = bldg_area / (units + 1)
area_per_unit_clipped = min(area_per_unit, 5000.0)

st.divider()

if st.button("🚀 Calculate Estimated Value", type="primary"):
    # Input matching the high-quality model features
    input_data = pd.DataFrame([[
        b_code, lot_area, bldg_area, floors, units, age, str(float(land_use)), u_lat, u_lon, area_per_unit_clipped
    ]], columns=['borough_y', 'lotarea', 'bldgarea', 'numfloors', 'unitsres', 'building_age', 'landuse', 'latitude', 'longitude', 'area_per_unit'])
    
    log_pred = model.predict(input_data)[0]
    final_price = np.expm1(log_pred)
    
    st.success(f"### Estimated Market Value: ${final_price:,.2f}")
    
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Price per SqFt", f"${(final_price/bldg_area):,.2f}")
    res_col2.metric("Efficiency Ratio", f"{area_per_unit_clipped:.0f} sqft/unit")

st.sidebar.markdown("### ℹ️ Model Insights")
st.sidebar.info("""
- **Algorithm:** High-Precision Random Forest
- **Source:** Pre-trained Offline Model
- **Optimization:** Log-Scaled Target
- **Portfolio:** [yusufozcan.space](https://yusufozcan.space)
""")