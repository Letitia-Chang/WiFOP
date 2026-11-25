"""
features.py

Single source of truth for the feature set and data path shared across
every training/evaluation script in this rework. Previously this list was
copy-pasted near-identically into 9+ files (train_xgboost_weighted.py,
train_xgboost_smote.py, train_random_forest_wildfire.py,
train_catboost_wildfire.py, svm_downsample_smote.py, evaluate_model.py,
ablation_land_cover.py, bootstrap_ci.py, tune_models.py) — any future
feature/leakage fix had to be hand-propagated to all of them, which is
exactly the kind of thing that let the original leakage bugs go unnoticed
(see docs/results.md's "Still open" section). Import from here instead of
redefining.
"""

DATA_PATH = "../data/processed/california_5km_training_panel_2020-2023_clean.csv"
TARGET = "fire_tomorrow"

# Weather/NDVI/fire-history features (day-to-day dynamics).
WEATHER_NDVI_FIRE_FEATURES = [
    "PRECTOTCORR", "PS", "RH2M", "T2M", "WS10M",
    "PRECTOT_7d_sum", "T2M_3d_mean", "RH2M_3d_min", "WS10M_3d_max",
    "ndvi", "ndvi_7d_mean",
    "fires_last_7d", "fires_last_14d",
]

# Land-cover composition (static per year per grid cell).
LAND_COVER_FEATURES = [
    "pct_11", "pct_12", "pct_21", "pct_22", "pct_23", "pct_24", "pct_31",
    "pct_41", "pct_42", "pct_43", "pct_52", "pct_71", "pct_81", "pct_82", "pct_90", "pct_95",
]

# The feature set XGBoost/CatBoost/RF/SVM are trained on by default.
BASE_FEATURES = WEATHER_NDVI_FIRE_FEATURES + LAND_COVER_FEATURES

# Nonlinear interaction terms engineered for CatBoost/RF/SVM (not used by
# the two XGBoost baselines — see each training script for why).
NONLINEAR_FEATURES = ["vpd_proxy", "hot_windy", "dry_windy", "fuel_moisture_proxy"]


def add_nonlinear_features(df):
    """Adds NONLINEAR_FEATURES columns to df in place and returns it."""
    df["vpd_proxy"] = (1 - df["RH2M"] / 100) * df["T2M"]
    df["hot_windy"] = df["T2M"] * df["WS10M"]
    df["dry_windy"] = (100 - df["RH2M"]) * df["WS10M"]
    df["fuel_moisture_proxy"] = df["ndvi"] * df["RH2M"]
    return df
