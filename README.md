# 🏙️ NYC AI Market Value Estimator

An end-to-end Machine Learning application that predicts property market values across New York City's five boroughs. This tool utilizes historical NYC Open Data (PLUTO) and a Random Forest Regressor to provide instant, data-driven valuations.

## 🚀 Live Demo
**Check out the live app here:** [nyc-price-ai.streamlit.app](https://nyc-price-ai.streamlit.app/)

## 🧠 Project Architecture & ML Logic
The model is designed to navigate the complex real estate landscape of NYC by focusing on geospatial and structural features:

- **Algorithm:** Random Forest Regressor.
- **Data Optimization:** Applied `np.log1p` transformation to property prices to handle extreme market skews (normalizing high-value outliers).
- **Geospatial Intelligence:** Integrated Latitude and Longitude coordinates to capture hyper-local neighborhood price trends.
- **Feature Engineering:** - **Area per Unit:** A custom efficiency ratio to distinguish between luxury low-density and high-density residential buildings.
  - **Building Age:** Calculated from year built to account for depreciation and historical value.
- **Cloud-Native Pipeline:** Implements an "Auto-Train" mechanism on Streamlit Cloud to manage large model files and bypass GitHub's 100MB limit.



## 🛠️ Tech Stack
| Category | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **ML Libraries** | Scikit-Learn, Pandas, NumPy, Joblib |
| **Interface** | Streamlit |
| **Deployment** | Streamlit Cloud |
| **Version Control** | GitHub |

## 📂 Project Structure
```text
├── app.py                # Core Streamlit dashboard & Auto-train logic
├── train_model.py        # Local training script for development
├── requirements.txt      # Python dependencies
├── .gitignore            # Excludes large binaries and temp files
└── nyc_housing_base.csv  # Processed NYC property dataset
          
