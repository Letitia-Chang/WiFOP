import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import json
import joblib

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_PATH = "../data/processed/california_5km_training_panel_2023-07_clean.csv"
MODEL_PATH = "../models/xgb_smote_model.pkl"
THRESHOLD_PATH = "../models/xgb_smote_threshold.json"
META_PATH = "../models/xgb_smote_metadata.json"

# ----------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------
print("Loading cleaned panel...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# Feature set
feature_cols = [
    "PRECTOTCORR", "PS", "RH2M", "T2M", "WS10M",
    "PRECTOT_7d_sum", "T2M_3d_mean", "RH2M_3d_min", "WS10M_3d_max",
    "ndvi", "ndvi_7d_mean",
    "fires_last_7d", "fires_last_14d",
    "pct_11","pct_12","pct_21","pct_22","pct_23","pct_24","pct_31",
    "pct_41","pct_42","pct_43","pct_52","pct_71","pct_81","pct_82","pct_90","pct_95"
]

# Predict next-day ignition (see docs/leakage_audit.md #1) — fire_today
# (same-day) is no longer the target.
target = "fire_tomorrow"

# Remove missing rows
before = len(df)
df = df.dropna(subset=feature_cols + [target])
print(f"Rows before: {before}, after dropping NaN: {len(df)}")

# ----------------------------------------------------------------------
# Train / Validation Split — time-based, not random (docs/leakage_audit.md
# #4). A random row split lets adjacent-day rows from the same grid_id
# leak across train/val via autocorrelated weather/NDVI and the (now-fixed
# but still autocorrelated) fire-history features. This panel only covers
# one month (2023-07) rather than the project's full 2020-2022 train /
# 2023 test range, so the closest available fix is a chronological split:
# the last 20% of dates become validation, everything earlier is training.
# ----------------------------------------------------------------------
df = df.sort_values("date")
split_date = df["date"].quantile(0.8, interpolation="lower")
train_mask = df["date"] <= split_date
X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, target]
X_val, y_val = df.loc[~train_mask, feature_cols], df.loc[~train_mask, target]
print(f"Train dates: {df.loc[train_mask,'date'].min().date()}..{df.loc[train_mask,'date'].max().date()} | "
      f"Val dates: {df.loc[~train_mask,'date'].min().date()}..{df.loc[~train_mask,'date'].max().date()}")

print(f"Training rows: {len(X_train)} | Validation rows: {len(X_val)}")
print(f"Fire cases in train: {y_train.sum()} | Fire cases in val: {y_val.sum()}")

# ----------------------------------------------------------------------
# SMOTE Oversampling
# ----------------------------------------------------------------------
print("Applying SMOTE oversampling...")
sm = SMOTE(random_state=42, sampling_strategy=1.0)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"After SMOTE → Training rows: {len(X_train_res)}")
print(f"Fire cases after SMOTE: {y_train_res.sum()}")

# ----------------------------------------------------------------------
# Train XGBoost
# ----------------------------------------------------------------------
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    learning_rate=0.05,
    max_depth=6,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

print("Training XGBoost model (SMOTE)...")
model.fit(X_train_res, y_train_res)

# ----------------------------------------------------------------------
# Evaluate ROC-AUC
# ----------------------------------------------------------------------
y_val_proba = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_val_proba)
print(f"ROC-AUC: {auc:.4f}")

# ----------------------------------------------------------------------
# Threshold Optimization (Precision-Recall)
# ----------------------------------------------------------------------
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

best_idx = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx])

print("\nOptimal threshold (max F1): {:.4f}".format(best_threshold))

# ----------------------------------------------------------------------
# Evaluate at Optimal Threshold
# ----------------------------------------------------------------------
y_pred_best = (y_val_proba >= best_threshold).astype(int)

print("\nClassification Report (optimal threshold):")
print(classification_report(y_val, y_pred_best, digits=3))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_best))

# ----------------------------------------------------------------------
# Save Model + Threshold + Metadata
# ----------------------------------------------------------------------
joblib.dump(model, MODEL_PATH)

with open(THRESHOLD_PATH, "w") as f:
    json.dump({"threshold": best_threshold}, f)

metadata = {
    "roc_auc": float(auc),
    "best_threshold": best_threshold,
    "num_train_rows": int(len(X_train_res)),
    "num_val_rows": int(len(X_val)),
    "train_fire_cases_after_smote": int(y_train_res.sum()),
    "val_fire_cases": int(y_val.sum()),
    "features": feature_cols
}

with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("Model, threshold, and metadata saved successfully.")
