import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. PAGE CONFIGURATION (Must be at the top)
st.set_page_config(
    page_title="NYC AI Market Value Estimator", 
    layout="wide", 
    page_icon="🏙️"
)

# --- 2. AUTOMATIC MODEL TRAINING ENGINE ---
# Since .pkl files over 100MB are blocked by GitHub, 
# this function trains the model on the cloud if it's missing.
@st.cache_resource
def load_or_train_model():
    model_path = 'models/nyc_house_model.pkl'
    data_path = 'nyc_housing_base.csv'
    
    if not os.path.exists(model_path):
        if not os.path.exists(data_path):
            st.error(f"⚠️ Critical Error: '{data_path}' not found! Please upload the dataset to GitHub.")
            st.stop()
            
        with st.status("🔄 Model not found. Training on cloud, please wait...", expanded=True) as status:
            df = pd.read_csv(data_path)
            
            # Data Cleaning
            df = df[df['sale_price'] > 10000] # Filter symbolic sales
            df['area_per_unit'] = (df['bldgarea'] / (df['unitsres'] + 1)).clip(upper=5000)
            df['landuse'] = df['landuse'].astype(str)
            df['borough_y'] = df['borough_y'].astype(str)
            
            target = 'sale_price'
            df = df.dropna(subset=[target, 'latitude', 'longitude'])
            y = np.log1p(df[target]) # Log transformation for skewed pricing
            
            features = ['borough_y', 'lotarea', 'bldgarea', 'numfloors', 
                        'unitsres', 'building_age', 'landuse', 'latitude', 'longitude', 'area_per_unit']
            X = df[features]
            
            # Machine Learning Pipeline
            numeric_features = ['lotarea', 'bldgarea', 'numfloors', 'unitsres', 'building_age', 'latitude', 'longitude', 'area_per_unit']
            categorical_features = ['borough_y', 'landuse']
            
            preprocessor = ColumnTransformer(transformers=[
                ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))]), numeric_features),
                ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
            ])
            
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
            ])
            
            model.fit(X, y)
            
            if not os.path.exists('models'): 
                os.makedirs('models')
                
            joblib.dump(model, model_path)
            status.update(label="✅ Model trained and ready!", state="complete")
    
    return joblib.load(model_path)

# Load the model
model = load_or_train_model()

# --- 3. UI DESIGN ---
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

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 Location & Property Type")
    selected_b_name = st.selectbox("Select Borough", list(BOROUGH_MAP.keys()))
    
    # Auto-sync coordinates
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
    
    # Logic: Residential use must have at least 1 unit
    min_u = 1 if land_use in [1, 2] else 0
    units = st.number_input("Residential Units", min_value=min_u, value=max(1, min_u))
    
    floors = st.slider("Number of Floors", 1, 120, 2)
    age = st.slider("Building Age (Years)", 0, 250, 45)

# Feature Engineering for Inference
area_per_unit = bldg_area / (units + 1)
area_per_unit_clipped = min(area_per_unit, 5000.0)

st.divider()

if st.button("🚀 Calculate Estimated Value", type="primary"):
    # Prepare input for model
    input_data = pd.DataFrame([[
        b_code, lot_area, bldg_area, floors, units, age, str(float(land_use)), u_lat, u_lon, area_per_unit_clipped
    ]], columns=['borough_y', 'lotarea', 'bldgarea', 'numfloors', 'unitsres', 'building_age', 'landuse', 'latitude', 'longitude', 'area_per_unit'])
    
    # Predict and reverse Log transformation
    log_pred = model.predict(input_data)[0]
    final_price = np.expm1(log_pred)
    
    st.success(f"### Estimated Market Value: ${final_price:,.2f}")
    
    # KPIs
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Price per SqFt", f"${(final_price/bldg_area):,.2f}")
    res_col2.metric("Efficiency Ratio", f"{area_per_unit_clipped:.0f} sqft/unit")

    if area_per_unit > 5000:
        st.warning("Note: Structural specs were clipped for model safety (extreme density detected).")

st.sidebar.markdown("### ℹ️ Model Insights")
st.sidebar.info("""
- **Algorithm:** Random Forest Regressor
- **Optimization:** Log-Scaled Target
- **Features:** Geospatial & Physical
- **Deployment:** Cloud Auto-Train
""")
# yusufozcan.space | GitHub.com/yusuf-ozcan 