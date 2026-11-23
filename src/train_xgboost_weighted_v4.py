# Added spatial features, tune parameter to optimize f1 score
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
import joblib
import json
import os

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_PATH = "../data/COMBINE_WIFOP_DATASET/data/processed/ca_5km_daily_panel_2020_2023_6_9.csv"

BEST_MODEL_PATH = "../models/2020_2023/xgb_optuna_spatial_model.pkl"
BEST_THRESHOLD_PATH = "../models/2020_2023/xgb_optuna_spatial_threshold.json"
BEST_META_PATH = "../models/2020_2023/xgb_optuna_spatial_metadata.json"

OPTUNA_STUDY_PATH = "../models/2020_2023/xgb_optuna_spatial_study.pkl"
PLOTS_DIR = "../models/2020_2023/plots_optuna_spatial/"
TRIAL_LOG = "../models/2020_2023/optuna_spatial_trial_log.csv"

os.makedirs(PLOTS_DIR, exist_ok=True)

if not os.path.exists(TRIAL_LOG):
    pd.DataFrame(columns=[
        "trial", "f1", "pr_auc", "roc_auc",
        "learning_rate", "max_depth", "min_child_weight",
        "subsample", "colsample_bytree", "gamma",
        "reg_lambda", "reg_alpha", "n_estimators"
    ]).to_csv(TRIAL_LOG, index=False)

# ----------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------
print("Loading combined panel...")
df = pd.read_csv(DATA_PATH, low_memory=False)

df["date"] = pd.to_datetime(df["date"])
df["grid_id"] = df["grid_id"].astype("category")

target = "fire_today"

# ----------------------------------------------------------------------
# Spatial Feature Engineering
# ----------------------------------------------------------------------
print("Building BallTree for spatial neighbors...")

grid_meta = df[["grid_id", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
coords_rad = np.radians(grid_meta[["lat", "lon"]].values)
tree = BallTree(coords_rad, metric="haversine")

# ---------------------- 8-Nearest Neighbor Fire Propagation ----------------------
print("Computing 8-nearest neighbors...")

K = 8 + 1
dist_nn, idx_nn = tree.query(coords_rad, k=K)

neighbor_map = {}
for i, g in enumerate(grid_meta["grid_id"]):
    neigh_ids = grid_meta.iloc[idx_nn[i]]["grid_id"].tolist()
    neigh_ids = [n for n in neigh_ids if n != g]  # remove self
    neighbor_map[g] = neigh_ids

def add_neighbor_lag(df, lag_col):
    new_col = "neighbor_" + lag_col
    df[new_col] = 0

    for date, group in df.groupby("date"):
        lag_map = dict(zip(group["grid_id"], group[lag_col]))

        vals = []
        for gid in group["grid_id"]:
            total = sum(lag_map.get(n, 0) for n in neighbor_map[gid])
            vals.append(total)

        df.loc[group.index, new_col] = vals

lag_features = ["fires_last_1d", "fires_last_3d", "fires_last_7d", "fires_last_14d"]

for lf in lag_features:
    add_neighbor_lag(df, lf)

# ---------------------- Radius-Based Smoothing ----------------------
print("Computing radius smoothing (10 km)...")

radius_km = 10
radius_rad = radius_km / 6371.0

radius_neighbors = tree.query_radius(coords_rad, r=radius_rad)

def add_radius_mean(df, col, new_col):
    df[new_col] = 0
    for date, group in df.groupby("date"):
        val_map = dict(zip(group["grid_id"], group[col]))

        out_vals = []
        for gid in group["grid_id"]:
            idx_match = grid_meta.index[grid_meta["grid_id"] == gid][0]
            neigh_idx = radius_neighbors[idx_match]
            neigh_ids = grid_meta.iloc[neigh_idx]["grid_id"]

            vals = [val_map.get(n, np.nan) for n in neigh_ids]
            out_vals.append(np.nanmean(vals))

        df.loc[group.index, new_col] = out_vals

# Add VPD before smoothing
df["vpd_proxy"] = (1 - df["RH2M"]/100) * df["T2M"]

radius_cols = {
    "T2M": "T2M_spatial_mean_10km",
    "RH2M": "RH2M_spatial_mean_10km",
    "WS10M": "WS10M_spatial_mean_10km",
    "ndvi": "ndvi_spatial_mean_10km",
    "vpd_proxy": "vpd_spatial_mean_10km"
}

for col, new_col in radius_cols.items():
    add_radius_mean(df, col, new_col)

# ---------------------- Additional Nonlinear Interactions ----------------------
df["hot_windy"] = df["T2M"] * df["WS10M"]
df["dry_windy"] = (100 - df["RH2M"]) * df["WS10M"]
df["fuel_moisture_proxy"] = df["ndvi"] * df["RH2M"]

# Dynamic NLCD extraction
nlcd_cols = [c for c in df.columns if c.startswith("pct_")]

# ----------------------------------------------------------------------
# Final Feature List
# ----------------------------------------------------------------------
feature_cols = [
    # Weather
    "T2M", "RH2M", "WS10M", "PS", "PRECTOTCORR",
    "T2M_3d_mean", "RH2M_3d_min", "WS10M_3d_max", "PRECTOT_7d_sum",

    # Vegetation
    "ndvi", "ndvi_7d_mean",

    # Fire lags
    "fires_last_7d", "fires_last_14d",
    "fires_last_1d", "fires_last_3d",

    # Nonlinear
    "vpd_proxy", "hot_windy", "dry_windy", "fuel_moisture_proxy",

    # Neighbor features
    "neighbor_fires_last_1d", "neighbor_fires_last_3d",
    "neighbor_fires_last_7d", "neighbor_fires_last_14d",

    # Radius-smoothing
    "T2M_spatial_mean_10km", "RH2M_spatial_mean_10km",
    "WS10M_spatial_mean_10km", "ndvi_spatial_mean_10km",
    "vpd_spatial_mean_10km"
] + nlcd_cols

print(f"Using {len(feature_cols)} features.")

# ----------------------------------------------------------------------
# Time-based split (train 2020–2022, test 2023)
# ----------------------------------------------------------------------
train = df[df["date"] < "2023-01-01"]
val = df[df["date"] >= "2023-01-01"]

X_train = train[feature_cols]
y_train = train[target]

X_val = val[feature_cols]
y_val = val[target]

# Compute class weight
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print("scale_pos_weight =", scale_pos_weight)

# ----------------------------------------------------------------------
# Weighted Accuracy (Option B)
# ----------------------------------------------------------------------
def weighted_accuracy(y_true, y_pred, w_pos, w_neg=1.0):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    return w_pos * tpr + w_neg * tnr

# ----------------------------------------------------------------------
# Optuna objective (OPTIMIZE F1)
# ----------------------------------------------------------------------
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,

        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 4, 14),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
    f1 = max(2 * (precisions * recalls) / (precisions + recalls + 1e-9))

    # log trial
    row = [[
        trial.number, f1,
        average_precision_score(y_val, y_proba),
        roc_auc_score(y_val, y_proba),
        params["learning_rate"], params["max_depth"],
        params["min_child_weight"], params["subsample"],
        params["colsample_bytree"], params["gamma"],
        params["reg_lambda"], params["reg_alpha"], params["n_estimators"]
    ]]

    pd.DataFrame(row).to_csv(TRIAL_LOG, mode="a", header=False, index=False)

    return f1

# ----------------------------------------------------------------------
# Run Optuna
# ----------------------------------------------------------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

joblib.dump(study, OPTUNA_STUDY_PATH)

print("\nBest params:", study.best_params)

# ----------------------------------------------------------------------
# Retrain best model
# ----------------------------------------------------------------------
best_params = study.best_params
best_params.update({
    "objective": "binary:logistic",
    "tree_method": "hist",
    "scale_pos_weight": scale_pos_weight,
})

model = XGBClassifier(**best_params)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# Final Evaluation
# ----------------------------------------------------------------------
y_proba = model.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

best_idx = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx])

y_pred_best = (y_proba >= best_threshold).astype(int)

wa = weighted_accuracy(y_val, y_pred_best, w_pos=scale_pos_weight)

print("\nWeighted Accuracy:", wa)
print("Best Threshold:", best_threshold)

# ----------------------------------------------------------------------
# Save model + metadata
# ----------------------------------------------------------------------
joblib.dump(model, BEST_MODEL_PATH)

with open(BEST_THRESHOLD_PATH, "w") as f:
    json.dump({"threshold": best_threshold}, f)

metadata = {
    "best_params": best_params,
    "best_threshold": best_threshold,
    "weighted_accuracy": wa,
    "roc_auc": float(roc_auc_score(y_val, y_proba)),
    "auc_pr": float(average_precision_score(y_val, y_proba)),
    "train_rows": int(len(X_train)),
    "val_rows": int(len(X_val)),
    "train_fire_cases": int(y_train.sum()),
    "val_fire_cases": int(y_val.sum()),
    "features": feature_cols
}

with open(BEST_META_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("\nAll done — spatial XGBoost model saved!")
