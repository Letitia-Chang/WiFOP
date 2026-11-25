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

from features import BASE_FEATURES as feature_cols, DATA_PATH, TARGET as target

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MODEL_PATH = "../models/xgb_smote_model.pkl"
THRESHOLD_PATH = "../models/xgb_smote_threshold.json"
META_PATH = "../models/xgb_smote_metadata.json"

# ----------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------
print("Loading cleaned panel...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# Feature set, target: see features.py (predicting next-day ignition,
# docs/leakage_audit.md #1 — fire_today same-day is no longer the target).

# Remove missing rows
before = len(df)
df = df.dropna(subset=feature_cols + [target, "block_id"])
print(f"Rows before: {before}, after dropping NaN: {len(df)}")

# ----------------------------------------------------------------------
# Train / Test Split — year-based (docs/leakage_audit.md #4): test is the
# full held-out 2023 fire season, train is 2020-2022, matching the
# project brief. A random row split would let adjacent-day rows from the
# same grid_id leak across train/test via autocorrelated weather/NDVI.
#
# Within the 2020-2022 training pool, the validation set used for
# threshold tuning is a SPATIAL BLOCK split, not a random or date split:
# fires cluster geographically, so a random/date split still lets two
# adjacent grid cells end up on opposite sides. block_id (from
# clean_panel.py) groups grid cells into ~55 km bins; 20% of blocks
# (by distinct block_id, not by row count) become validation.
# ----------------------------------------------------------------------
test_mask = df["date"].dt.year == 2023
train_pool = df.loc[~test_mask]
X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, target]

rng = np.random.RandomState(42)
blocks = train_pool["block_id"].unique()
rng.shuffle(blocks)
n_val_blocks = max(1, int(len(blocks) * 0.2))
val_blocks = set(blocks[:n_val_blocks])
val_mask = train_pool["block_id"].isin(val_blocks)

X_train, y_train = train_pool.loc[~val_mask, feature_cols], train_pool.loc[~val_mask, target]
X_val, y_val = train_pool.loc[val_mask, feature_cols], train_pool.loc[val_mask, target]

print(f"Train rows: {len(X_train)} | Spatial-val rows: {len(X_val)} | 2023 test rows: {len(X_test)}")
print(f"Train blocks: {len(blocks) - n_val_blocks} | Val blocks: {n_val_blocks}")
print(f"Fire cases in train: {y_train.sum()} | Fire cases in spatial-val: {y_val.sum()} | "
      f"Fire cases in 2023 test: {y_test.sum()}")

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
# Final Evaluation — held-out 2023 test year (threshold/model selection
# above never touched this data)
# ----------------------------------------------------------------------
y_test_proba = model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_test_proba)
y_test_pred = (y_test_proba >= best_threshold).astype(int)

print(f"\n2023 held-out test ROC-AUC: {test_auc:.4f}")
print("2023 held-out test Classification Report:")
print(classification_report(y_test, y_test_pred, digits=3))
print("2023 held-out test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# ----------------------------------------------------------------------
# Save Model + Threshold + Metadata
# ----------------------------------------------------------------------
joblib.dump(model, MODEL_PATH)

with open(THRESHOLD_PATH, "w") as f:
    json.dump({"threshold": best_threshold}, f)

metadata = {
    "spatial_val_roc_auc": float(auc),
    "test_2023_roc_auc": float(test_auc),
    "best_threshold": best_threshold,
    "num_train_rows": int(len(X_train_res)),
    "num_val_rows": int(len(X_val)),
    "num_test_rows": int(len(X_test)),
    "train_fire_cases_after_smote": int(y_train_res.sum()),
    "val_fire_cases": int(y_val.sum()),
    "test_fire_cases": int(y_test.sum()),
    "features": feature_cols
}

with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("Model, threshold, and metadata saved successfully.")
