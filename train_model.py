import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("nyc_housing_base.csv")

# =========================
# 2. CLEAN REAL MARKET SALES
# =========================
df = df[
    (df["sale_price"] > 200_000) &
    (df["sale_price"] < df["sale_price"].quantile(0.98))
]

df = df.dropna(subset=[
    "sale_price", "bldgarea", "lotarea",
    "latitude", "longitude"
])

# =========================
# 3. FEATURE ENGINEERING
# =========================
df["unitsres"] = df["unitsres"].clip(lower=1)
df["area_per_unit"] = (df["bldgarea"] / df["unitsres"]).clip(upper=5000)

# 🔑 CRITICAL FIX: spatial binning
df["lat_bin"] = df["latitude"].round(3)
df["lon_bin"] = df["longitude"].round(3)

df["borough_y"] = df["borough_y"].astype(str)
df["landuse"] = df["landuse"].astype(str)

# =========================
# 4. TARGET
# =========================
y = np.log1p(df["sale_price"])

features = [
    "borough_y", "landuse",
    "lotarea", "bldgarea", "numfloors",
    "unitsres", "building_age",
    "area_per_unit", "lat_bin", "lon_bin"
]

X = df[features]

# =========================
# 5. PIPELINE
# =========================
numeric_features = [
    "lotarea", "bldgarea", "numfloors",
    "unitsres", "building_age",
    "area_per_unit", "lat_bin", "lon_bin"
]

categorical_features = ["borough_y", "landuse"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("rf", model)
])

# =========================
# 6. TRAIN
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

print("R2 (log-space):", pipeline.score(X_test, y_test))

# =========================
# 7. SAVE
# =========================
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/nyc_house_model.pkl")

print("✅ Model saved: models/nyc_house_model.pkl")
