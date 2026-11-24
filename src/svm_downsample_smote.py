"""
svm_downsample_smote.py

Linear SVM baseline for fire_tomorrow (downsample + SMOTE + SelectKBest),
on the real 2020-2023 clean panel (src/clean_panel.py output) — rewritten
from the original version, which assumed a different, never-recovered
full-scale dataset (ca_5km_daily_panel_2020_2023_6_9.csv, with its own
engineered neighbor/lat-lon columns) that isn't part of this rework.

Kept the original's core approach (downsample negatives 20:1 before SMOTE,
since full SMOTE + LinearSVC over millions of rows doesn't scale; tune the
decision threshold by F1 on validation). Uses the same year-based test
split + spatial-block validation split as the other baselines
(docs/leakage_audit.md #4) instead of the original's random stratified
split.
"""

import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=UserWarning, message="Precision is ill-defined")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_STATE = 42
DATA_PATH = "../data/processed/california_5km_training_panel_2020-2023_clean.csv"
TARGET = "fire_tomorrow"
DOWNSAMPLE_RATIO = 20  # negatives kept per positive, before SMOTE
C_VAL = 1.0
K_VAL = 25

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


def tune_threshold(clf_pipeline, X_val, y_val):
    y_scores = clf_pipeline.decision_function(X_val)
    thresholds = np.arange(-2.0, 2.1, 0.1)
    best_f1, best_t = -1, 0
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_val, y_pred, labels=[1], average=None, zero_division=0)
        if f1[0] > best_f1:
            best_f1, best_t = f1[0], t
    return best_f1, best_t


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

test_mask = df["date"].dt.year == 2023
train_pool = df.loc[~test_mask]
X_test = df.loc[test_mask, FEATURE_COLS].to_numpy(dtype=np.float32)
y_test = df.loc[test_mask, TARGET].to_numpy(dtype=int)

rng = np.random.RandomState(RANDOM_STATE)
blocks = train_pool["block_id"].unique()
rng.shuffle(blocks)
n_val_blocks = max(1, int(len(blocks) * 0.2))
val_blocks = set(blocks[:n_val_blocks])
val_mask = train_pool["block_id"].isin(val_blocks)

X_train_full = train_pool.loc[~val_mask, FEATURE_COLS].to_numpy(dtype=np.float32)
y_train_full = train_pool.loc[~val_mask, TARGET].to_numpy(dtype=int)
X_val = train_pool.loc[val_mask, FEATURE_COLS].to_numpy(dtype=np.float32)
y_val = train_pool.loc[val_mask, TARGET].to_numpy(dtype=int)

print(f"Train rows: {len(X_train_full)} | Spatial-val rows: {len(X_val)} | 2023 test rows: {len(X_test)}")
print(f"Fire cases in train: {y_train_full.sum()} | spatial-val: {y_val.sum()} | 2023 test: {y_test.sum()}")

print(f"\n--- Downsampling negatives to {DOWNSAMPLE_RATIO}:1 before SMOTE ---")
pos_idx = np.where(y_train_full == 1)[0]
neg_idx = np.where(y_train_full == 0)[0]
num_neg_to_keep = min(len(neg_idx), len(pos_idx) * DOWNSAMPLE_RATIO)
neg_sample_idx = rng.choice(neg_idx, size=num_neg_to_keep, replace=False)
keep_idx = np.concatenate([pos_idx, neg_sample_idx])
rng.shuffle(keep_idx)
X_train_sampled = X_train_full[keep_idx]
y_train_sampled = y_train_full[keep_idx]
print(f"Downsampled training set size: {len(y_train_sampled)} ({y_train_sampled.sum()} fire cases)")

print(f"\n--- Training SVM pipeline (K={K_VAL}) ---")
clf_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
    ("feature_selection", SelectKBest(score_func=mutual_info_classif, k=K_VAL)),
    ("clf", LinearSVC(C=C_VAL, class_weight="balanced", max_iter=10000, random_state=RANDOM_STATE, dual="auto")),
])
clf_pipeline.fit(X_train_sampled, y_train_sampled)

best_f1, best_t = tune_threshold(clf_pipeline, X_val, y_val)
print(f"\nSpatial-val best threshold (max F1 for fire class): {best_t:.2f} (F1={best_f1:.4f})")

val_scores = clf_pipeline.decision_function(X_val)
val_auc = roc_auc_score(y_val, val_scores)
y_val_pred = (val_scores >= best_t).astype(int)
print(f"Spatial-val ROC-AUC: {val_auc:.4f}")
print("Spatial-val Classification Report:")
print(classification_report(y_val, y_val_pred, digits=3))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

test_scores = clf_pipeline.decision_function(X_test)
test_auc = roc_auc_score(y_test, test_scores)
y_test_pred = (test_scores >= best_t).astype(int)
print(f"\n2023 held-out test ROC-AUC: {test_auc:.4f}")
print("2023 held-out test Classification Report:")
print(classification_report(y_test, y_test_pred, digits=3))
print("2023 held-out test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\n================= Feature Importance Ranking (SVM Coefficients) =================\n")
svm_model = clf_pipeline.named_steps["clf"]
coefficients = svm_model.coef_[0]
fs = clf_pipeline.named_steps["feature_selection"]
selected_mask = fs.get_support()
feature_names = np.array(FEATURE_COLS)[selected_mask]

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute_Importance": np.abs(coefficients),
}).sort_values(by="Absolute_Importance", ascending=False).reset_index(drop=True)
feature_df.index.name = "Rank"
feature_df.index = feature_df.index + 1
print(feature_df.to_string())

import json
metadata = {
    "spatial_val_roc_auc": float(val_auc),
    "test_2023_roc_auc": float(test_auc),
    "best_threshold": float(best_t),
    "num_train_rows_downsampled": int(len(y_train_sampled)),
    "num_val_rows": int(len(y_val)),
    "num_test_rows": int(len(y_test)),
    "train_fire_cases_downsampled": int(y_train_sampled.sum()),
    "val_fire_cases": int(y_val.sum()),
    "test_fire_cases": int(y_test.sum()),
    "selected_features": feature_names.tolist(),
}
with open("../reports/metrics/svm_metadata_2020-2023.json", "w") as f:
    json.dump(metadata, f, indent=4)
print("\nSaved ../reports/metrics/svm_metadata_2020-2023.json")
