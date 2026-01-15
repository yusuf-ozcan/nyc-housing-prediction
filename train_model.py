import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

# 1. Load data
print("Loading data...")
df = pd.read_csv('nyc_housing_base.csv')

# 2. Data Cleaning & Professional Filtering
# Remove symbolic sales (family transfers etc.) to prevent model bias
df = df[df['sale_price'] > 10000]
# Drop rows with critical missing spatial data
df = df.dropna(subset=['sale_price', 'latitude', 'longitude'])

# 3. Advanced Feature Engineering
# Create physical logic feature: Area per unit (helps identify luxury vs dense housing)
# Clip at 5000 to prevent extreme outliers from dominating the Random Forest
df['area_per_unit'] = (df['bldgarea'] / (df['unitsres'] + 1)).clip(upper=5000)

# Convert categorical codes to string to ensure they are treated as Nominal, not Ordinal
df['landuse'] = df['landuse'].astype(str)
df['borough_y'] = df['borough_y'].astype(str)

# 4. Target Transformation (The "Pro" Move)
# Log transformation handles the extreme price skewness in NYC real estate
y = np.log1p(df['sale_price']) 

# 5. Feature Selection
features = ['borough_y', 'lotarea', 'bldgarea', 'numfloors', 
            'unitsres', 'building_age', 'landuse', 'latitude', 'longitude', 'area_per_unit']
X = df[features]

# 6. Pipeline Construction
numeric_features = ['lotarea', 'bldgarea', 'numfloors', 'unitsres', 'building_age', 'latitude', 'longitude', 'area_per_unit']
categorical_features = ['borough_y', 'landuse']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 7. Model Training (Optimized for Performance)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Advanced NYC Model (Log-Scale)...")
model_pipeline.fit(X_train, y_train)

# Note: Score is calculated on the log-transformed target
print(f"Model Training Complete. Log-Scale R2 Score: {model_pipeline.score(X_test, y_test):.4f}")

# 8. Save Model
if not os.path.exists('models'): 
    os.makedirs('models')

joblib.dump(model_pipeline, 'models/nyc_house_model.pkl')
print("Final model saved successfully in models/nyc_house_model.pkl")