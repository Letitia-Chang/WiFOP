"""
train_random_forest_wildfire.py

Random forest baseline for fire_tomorrow, on the real 2020-2023 clean panel
(src/clean_panel.py output) — rewritten from the original version, which
assumed a different, never-recovered full-scale dataset
(ca_5km_daily_panel_2020_2023_6_9.csv, with full-name NLCD columns and
lat/lon per row) that isn't part of this rework. Kept the original's
feature engineering (vpd_proxy, hot_windy, dry_windy, fuel_moisture_proxy)
and class_weight="balanced" approach; uses the same year-based test split +
spatial-block validation split as the XGBoost scripts
(docs/leakage_audit.md #4), instead of the original's random split.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, classification_report, confusion_matrix
)

DATA_PATH = "../data/processed/california_5km_training_panel_2020-2023_clean.csv"
MODEL_PATH = "../models/rf_model.pkl"
THRESHOLD_PATH = "../reports/metrics/rf_threshold_2020-2023.json"
META_PATH = "../reports/metrics/rf_metadata_2020-2023.json"

BASE_FEATURES = [
    "PRECTOTCORR", "PS", "RH2M", "T2M", "WS10M",
    "PRECTOT_7d_sum", "T2M_3d_mean", "RH2M_3d_min", "WS10M_3d_max",
    "ndvi", "ndvi_7d_mean",
    "fires_last_7d", "fires_last_14d",
    "pct_11","pct_12","pct_21","pct_22","pct_23","pct_24","pct_31",
    "pct_41","pct_42","pct_43","pct_52","pct_71","pct_81","pct_82","pct_90","pct_95"
]
NONLINEAR_FEATURES = ["vpd_proxy", "hot_windy", "dry_windy", "fuel_moisture_proxy"]
FEATURE_COLS = BASE_FEATURES + NONLINEAR_FEATURES
TARGET = "fire_tomorrow"

print("Loading cleaned panel...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

print("Engineering nonlinear interaction features...")
df["vpd_proxy"] = (1 - df["RH2M"] / 100) * df["T2M"]
df["hot_windy"] = df["T2M"] * df["WS10M"]
df["dry_windy"] = (100 - df["RH2M"]) * df["WS10M"]
df["fuel_moisture_proxy"] = df["ndvi"] * df["RH2M"]

before = len(df)
df = df.dropna(subset=FEATURE_COLS + [TARGET, "block_id"])
print(f"Rows before: {before}, after dropping NaN: {len(df)}")

# Year-based test split + spatial-block validation split — see
# train_xgboost_weighted.py for the full rationale.
test_mask = df["date"].dt.year == 2023
train_pool = df.loc[~test_mask]
X_test, y_test = df.loc[test_mask, FEATURE_COLS], df.loc[test_mask, TARGET]

rng = np.random.RandomState(42)
blocks = train_pool["block_id"].unique()
rng.shuffle(blocks)
n_val_blocks = max(1, int(len(blocks) * 0.2))
val_blocks = set(blocks[:n_val_blocks])
val_mask = train_pool["block_id"].isin(val_blocks)

X_train, y_train = train_pool.loc[~val_mask, FEATURE_COLS], train_pool.loc[~val_mask, TARGET]
X_val, y_val = train_pool.loc[val_mask, FEATURE_COLS], train_pool.loc[val_mask, TARGET]

print(f"Train rows: {len(X_train)} | Spatial-val rows: {len(X_val)} | 2023 test rows: {len(X_test)}")
print(f"Fire cases in train: {y_train.sum()} | spatial-val: {y_val.sum()} | 2023 test: {y_test.sum()}")

print("Training Random Forest model...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)

y_val_proba = rf.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_proba)
auc_pr = average_precision_score(y_val, y_val_proba)
print(f"\nSpatial-val ROC-AUC: {roc_auc:.4f} | AUC-PR: {auc_pr:.4f}")

precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx])
print(f"Optimal threshold (max F1): {best_threshold:.4f}")

y_pred_best = (y_val_proba >= best_threshold).astype(int)
print("\nSpatial-val Classification Report:")
print(classification_report(y_val, y_pred_best, digits=3))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_best))

y_test_proba = rf.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_test_proba)
test_auc_pr = average_precision_score(y_test, y_test_proba)
y_test_pred = (y_test_proba >= best_threshold).astype(int)

print(f"\n2023 held-out test ROC-AUC: {test_auc:.4f} | AUC-PR: {test_auc_pr:.4f}")
print("2023 held-out test Classification Report:")
print(classification_report(y_test, y_test_pred, digits=3))
print("2023 held-out test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

Path("../models").mkdir(parents=True, exist_ok=True)
joblib.dump(rf, MODEL_PATH)

with open(THRESHOLD_PATH, "w") as f:
    json.dump({"threshold": best_threshold}, f)

metadata = {
    "spatial_val_roc_auc": float(roc_auc),
    "spatial_val_auc_pr": float(auc_pr),
    "test_2023_roc_auc": float(test_auc),
    "test_2023_auc_pr": float(test_auc_pr),
    "best_threshold": best_threshold,
    "num_train_rows": int(len(X_train)),
    "num_val_rows": int(len(X_val)),
    "num_test_rows": int(len(X_test)),
    "train_fire_cases": int(y_train.sum()),
    "val_fire_cases": int(y_val.sum()),
    "test_fire_cases": int(y_test.sum()),
    "features": FEATURE_COLS,
}
with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("\nRandom Forest model, threshold, and metadata saved successfully.")
