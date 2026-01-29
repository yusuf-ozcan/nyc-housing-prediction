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
    zip_path = 'models/nyc_house_model.pkl.zip'
    model_path = 'models/nyc_house_model.pkl'
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    if not os.path.exists(model_path):
        if os.path.exists(zip_path):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall('models')
            except Exception as e:
                st.error(f"Zip extraction failed: {e}")
                st.stop()
        else:
            st.error(f"⚠️ Critical Error: '{zip_path}' not found!")
            st.stop()
    
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

model = load_trained_model()

# --- 3. UI DESIGN ---
BOROUGH_MAP = {
    "Manhattan": {"lat": 40.7580, "lon": -73.9855, "code": "MN"},
    "Brooklyn": {"lat": 40.6782, "lon": -73.9442, "code": "BK"},
    "Queens": {"lat": 40.7282, "lon": -73.7949, "code": "QN"},
    "Bronx": {"lat": 40.8448, "lon": -73.8648, "code": "BX"},
    "Staten Island": {"lat": 40.5795, "lon": -74.1502, "code": "SI"}
}

st.title("🏙️ NYC AI Market Value Estimator")
st.markdown("Professional real estate valuation based on NYC Open Data.")
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
    
    u_lat = st.number_input("Latitude", value=default_lat, format="%.4f")
    u_lon = st.number_input("Longitude", value=default_lon, format="%.4f")

with col2:
    st.subheader("🏗️ Structural Specifications")
    bldg_area = st.number_input("Building Area (sqft)", value=2200, min_value=1)
    lot_area = st.number_input("Lot Area (sqft)", value=2500, min_value=1)
    
    units = st.number_input("Residential Units", min_value=0, value=1)
    floors = st.slider("Number of Floors", 1, 120, 2)
    age = st.slider("Building Age (Years)", 0, 250, 45)

# --- 4. CALCULATION LOGIC ---
# Bölme hatasını ve aşırı uç değerleri önlemek için clipping
safe_units = float(units) if units > 0 else 1.0
area_per_unit = float(bldg_area) / safe_units
area_per_unit_clipped = min(area_per_unit, 5000.0)

st.divider()

if st.button("🚀 Calculate Estimated Value", type="primary"):
    try:
        # Girdileri DataFrame'e çevir
        input_data = pd.DataFrame([[
            str(b_code), float(lot_area), float(bldg_area), float(floors), 
            float(units), float(age), str(float(land_use)), 
            float(u_lat), float(u_lon), float(area_per_unit_clipped)
        ]], columns=['borough_y', 'lotarea', 'bldgarea', 'numfloors', 'unitsres', 'building_age', 'landuse', 'latitude', 'longitude', 'area_per_unit'])
        
        # Ham log tahmini al
        log_pred = model.predict(input_data)[0]
        
        # GÜVENLİK KONTROLÜ: Eğer log_pred 25'ten büyükse expm1 sonucu sonsuza gider
        if log_pred > 25:
             st.warning("⚠️ The result is too high to be realistic. Please check if 'Building Area' is correct.")
        else:
            final_price = np.expm1(log_pred)
            
            if np.isinf(final_price) or np.isnan(final_price):
                st.error("⚠️ Invalid prediction value ($inf). Check structural inputs.")
            else:
                st.success(f"### Estimated Market Value: ${final_price:,.2f}")
                
                res_col1, res_col2 = st.columns(2)
                # Price per SqFt güvenli bölme
                sqft_price = final_price / bldg_area if bldg_area > 0 else 0
                res_col1.metric("Price per SqFt", f"${sqft_price:,.2f}")
                res_col2.metric("Efficiency Ratio", f"{area_per_unit_clipped:.0f} sqft/unit")
                
    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.sidebar.info("Model: Random Forest Regressor | Developer: Yusuf Özcan")