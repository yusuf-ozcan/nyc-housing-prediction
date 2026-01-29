import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import zipfile

# ===============================
# 1. PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="NYC AI Market Value Estimator",
    layout="wide",
    page_icon="🏙️"
)

# ===============================
# 2. LOAD MODEL
# ===============================
@st.cache_resource
def load_trained_model():
    zip_path = "models/nyc_house_model.pkl.zip"
    model_path = "models/nyc_house_model.pkl"

    os.makedirs("models", exist_ok=True)

    if not os.path.exists(model_path):
        if not os.path.exists(zip_path):
            st.error("❌ Model zip file not found.")
            st.stop()

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("models")

    return joblib.load(model_path)

model = load_trained_model()

# ===============================
# 3. UI
# ===============================
BOROUGH_MAP = {
    "Manhattan": {"lat": 40.7580, "lon": -73.9855, "code": "MN"},
    "Brooklyn": {"lat": 40.6782, "lon": -73.9442, "code": "BK"},
    "Queens": {"lat": 40.7282, "lon": -73.7949, "code": "QN"},
    "Bronx": {"lat": 40.8448, "lon": -73.8648, "code": "BX"},
    "Staten Island": {"lat": 40.5795, "lon": -74.1502, "code": "SI"}
}

st.title("🏙️ NYC AI Market Value Estimator")
st.caption("AI-powered real estate valuation based on NYC Open Data")
st.divider()

col1, col2 = st.columns(2)

with col1:
    borough = st.selectbox("Borough", BOROUGH_MAP.keys())
    b_code = BOROUGH_MAP[borough]["code"]

    lat = st.number_input("Latitude", value=BOROUGH_MAP[borough]["lat"], format="%.5f")
    lon = st.number_input("Longitude", value=BOROUGH_MAP[borough]["lon"], format="%.5f")

    land_use = st.selectbox(
        "Land Use",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Residential (1 Family)",
            2: "Residential (Multi)",
            3: "Mixed Use",
            4: "Commercial"
        }[x]
    )

with col2:
    bldg_area = st.number_input("Building Area (sqft)", 500, 200000, 2200)
    lot_area = st.number_input("Lot Area (sqft)", 200, 200000, 2500)
    units = st.number_input("Residential Units", 0, 500, 1)
    floors = st.slider("Floors", 1, 120, 2)
    age = st.slider("Building Age", 0, 300, 45)

# ===============================
# 4. FEATURE ENGINEERING
# ===============================
safe_units = max(units, 1)
area_per_unit = min(bldg_area / safe_units, 6000)

# ===============================
# 5. PREDICTION
# ===============================
st.divider()

if st.button("🚀 Calculate Estimated Value", type="primary"):
    try:
        X = pd.DataFrame([{
            "borough_y": b_code,
            "lotarea": float(lot_area),
            "bldgarea": float(bldg_area),
            "numfloors": float(floors),
            "unitsres": float(units),
            "building_age": float(age),
            "landuse": int(land_use),
            "latitude": float(lat),
            "longitude": float(lon),
            "area_per_unit": float(area_per_unit)
        }])

        log_pred = model.predict(X)[0]

        # ---- SAFE RANGE ----
        if log_pred < 5 or log_pred > 33:
            st.error("⚠️ Prediction out of realistic bounds. Check input values.")
            st.stop()

        price = np.expm1(log_pred)

        # Hard cap safety (NYC edge cases)
        price = min(price, 150_000_000)

        st.success(f"### 💰 Estimated Market Value: ${price:,.0f}")

        c1, c2 = st.columns(2)
        c1.metric("Price / SqFt", f"${price / bldg_area:,.0f}")
        c2.metric("Area per Unit", f"{area_per_unit:,.0f} sqft")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ===============================
# 6. SIDEBAR
# ===============================
st.sidebar.info(
    """
    **Model:** Random Forest Regressor  
    **Target:** log(SalePrice)  
    **Developer:** Yusuf Özcan  
    """
)
