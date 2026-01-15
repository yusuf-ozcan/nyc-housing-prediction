import streamlit as st
import pandas as pd
import numpy as np
import joblib

# PROFESSIONAL ERROR HANDLING: Ensure model exists
try:
    model = joblib.load('models/nyc_house_model.pkl')
except:
    st.set_page_config(page_title="Model Error", icon="⚠️")
    st.error("### ⚠️ Machine Learning Model Not Found")
    st.info("The prediction engine is missing. Please run `python train_model.py` first.")
    st.stop()

st.set_page_config(page_title="NYC AI Market Value Estimator", layout="wide")

# NYC Borough Coordination Presets (Used for UI-Model Integrity)
BOROUGH_MAP = {
    "Manhattan": {"lat": 40.7580, "lon": -73.9855, "code": "MN"},
    "Brooklyn": {"lat": 40.6782, "lon": -73.9442, "code": "BK"},
    "Queens": {"lat": 40.7282, "lon": -73.7949, "code": "QN"},
    "Bronx": {"lat": 40.8448, "lon": -73.8648, "code": "BX"},
    "Staten Island": {"lat": 40.5795, "lon": -74.1502, "code": "SI"}
}

st.title("🏙️ NYC AI Market Value Estimator")
st.markdown("Professional Real estate valuation based on NYC Open Data and Random Forest Machine Learning.")
st.divider()

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 Location & Type")
    selected_b_name = st.selectbox("Select Borough", list(BOROUGH_MAP.keys()))
    
    # Auto-sync coordinates with borough choice for data integrity
    default_lat = BOROUGH_MAP[selected_b_name]["lat"]
    default_lon = BOROUGH_MAP[selected_b_name]["lon"]
    b_code = BOROUGH_MAP[selected_b_name]["code"]

    land_use = st.selectbox("Land Use Category", options=[1, 2, 3, 4], 
                            format_func=lambda x: {1:"Residential-1 Family", 2:"Residential-Multi", 3:"Mixed Use", 4:"Commercial"}.get(x))
    
    # Coordinate override (Optional for user)
    with st.expander("Fine-tune Coordinates"):
        u_lat = st.number_input("Latitude", value=default_lat, format="%.4f")
        u_lon = st.number_input("Longitude", value=default_lon, format="%.4f")

with col2:
    st.subheader("🏗️ Building Physical Specs")
    bldg_area = st.number_input("Building Area (sqft)", value=2200, min_value=10)
    lot_area = st.number_input("Lot Area (sqft)", value=2500, min_value=10)
    
    # Business Logic: Residential land use cannot have 0 units
    min_u = 1 if land_use in [1, 2] else 0
    units = st.number_input("Residential Units", min_value=min_u, value=max(1, min_u))
    
    floors = st.slider("Number of Floors", 1, 120, 2)
    age = st.slider("Building Age (Years)", 0, 250, 45)

# INFERENCE LOGIC: Feature Engineering (Sync with training)
area_per_unit = bldg_area / (units + 1)
area_per_unit_clipped = min(area_per_unit, 5000.0)

st.divider()

if st.button("🚀 Calculate Market Value", type="primary"):
    # Create input DataFrame matching the model's expected feature order
    input_data = pd.DataFrame([[
        b_code, lot_area, bldg_area, floors, units, age, str(float(land_use)), u_lat, u_lon, area_per_unit_clipped
    ]], columns=['borough_y', 'lotarea', 'bldgarea', 'numfloors', 'unitsres', 'building_age', 'landuse', 'latitude', 'longitude', 'area_per_unit'])
    
    # 1. Predict (Result is in Log-scale from training)
    log_pred = model.predict(input_data)[0]
    
    # 2. Reverse Log-Transformation (np.expm1 is the inverse of np.log1p)
    final_price = np.expm1(log_pred)
    
    # Display Results
    st.success(f"### Estimated Market Value: ${final_price:,.2f}")
    
    # Indicators
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Price per SqFt", f"${(final_price/bldg_area):,.2f}")
    res_col2.metric("Efficiency Ratio", f"{area_per_unit_clipped:.0f} sqft/unit")

    if area_per_unit > 5000:
        st.warning("Note: Physical specs were clipped for model safety (extreme size detected).")

st.sidebar.info("""
**Model Details:**
- Random Forest Regressor
- Log-Scale target optimization
- Outlier resilience (Clipped Features)
- Spatial Awareness via Lat/Lon
""")