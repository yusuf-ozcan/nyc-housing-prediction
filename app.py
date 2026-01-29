import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================
# PAGE
# =========================
st.set_page_config("NYC AI Market Value Estimator", "🏙️", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("models/nyc_house_model.pkl")

model = load_model()

# =========================
# BOROUGHS
# =========================
BOROUGHS = {
    "Manhattan": ("MN", 40.7580, -73.9855),
    "Brooklyn": ("BK", 40.6782, -73.9442),
    "Queens": ("QN", 40.7282, -73.7949),
    "Bronx": ("BX", 40.8448, -73.8648),
    "Staten Island": ("SI", 40.5795, -74.1502),
}

st.title("🏙️ NYC AI Market Value Estimator")
st.divider()

c1, c2 = st.columns(2)

with c1:
    borough = st.selectbox("Borough", BOROUGHS.keys())
    b_code, lat, lon = BOROUGHS[borough]

    landuse_map = {
        "Single Family": "1",
        "Multi Family": "2",
        "Commercial / Mixed": "4"
    }
    landuse = st.selectbox("Property Type", landuse_map.keys())

with c2:
    bldgarea = st.number_input("Building Area (sqft)", 300, 150000, 2200)
    lotarea = st.number_input("Lot Area (sqft)", 200, 150000, 2500)
    units = st.number_input("Units", 1, 200, 1)
    floors = st.slider("Floors", 1, 80, 2)
    age = st.slider("Building Age", 0, 150, 40)

# =========================
# PREDICTION
# =========================
st.divider()

if st.button("🚀 Calculate Value", type="primary"):
    units = max(units, 1)
    area_per_unit = min(bldgarea / units, 5000)

    X = pd.DataFrame([{
        "borough_y": b_code,
        "landuse": landuse_map[landuse],
        "lotarea": lotarea,
        "bldgarea": bldgarea,
        "numfloors": floors,
        "unitsres": units,
        "building_age": age,
        "area_per_unit": area_per_unit,
        "lat_bin": round(lat, 3),
        "lon_bin": round(lon, 3)
    }])

    log_price = float(model.predict(X)[0])

    # 🔒 HARD DOMAIN CLAMP (NYC REALITY)
    log_price = np.clip(log_price, 11.5, 17.7)

    price = np.expm1(log_price)

    # Final sanity
    if price / bldgarea > 4000:
        price = bldgarea * 2500

    st.success(f"### 💰 Estimated Market Value: ${price:,.0f}")

    m1, m2 = st.columns(2)
    m1.metric("Price per SqFt", f"${price/bldgarea:,.0f}")
    m2.metric("Confidence (log)", f"{log_price:.2f}")

    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=12)
