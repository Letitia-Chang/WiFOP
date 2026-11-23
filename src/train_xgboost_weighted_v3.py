import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, classification_report,
    confusion_matrix, auc
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib
import json

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_PATH = "../data/COMBINE_WIFOP_DATASET/data/processed/ca_5km_daily_panel_2020_2023_6_9.csv"

BEST_MODEL_PATH = "../models/2020_2023/xgb_optuna_best_model.pkl"
BEST_THRESHOLD_PATH = "../models/2020_2023/xgb_optuna_best_threshold.json"
BEST_META_PATH = "../models/2020_2023/xgb_optuna_metadata.json"

OPTUNA_STUDY_PATH = "../models/2020_2023/xgb_optuna_study.pkl"
PLOTS_DIR = "../models/2020_2023/plots_optuna/"
TRIAL_LOG = "../models/2020_2023/optuna_trial_log.csv"

import os
os.makedirs(PLOTS_DIR, exist_ok=True)

# Initialize CSV log if not exists
if not os.path.exists(TRIAL_LOG):
    pd.DataFrame(columns=[
        "trial", "pr_auc", "roc_auc", "precision", "recall", "f1",
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
# Feature Engineering
# ----------------------------------------------------------------------
print("Engineering nonlinear features...")

df["vpd_proxy"] = (1 - df["RH2M"] / 100) * df["T2M"]
df["hot_windy"] = df["T2M"] * df["WS10M"]
df["dry_windy"] = (100 - df["RH2M"]) * df["WS10M"]
df["fuel_moisture_proxy"] = df["ndvi"] * df["RH2M"]

optional_lags = ["fires_last_1d", "fires_last_3d"]
available_lags = [c for c in optional_lags if c in df.columns]

feature_cols = [
    "T2M", "RH2M", "WS10M", "PS", "PRECTOTCORR",
    "T2M_3d_mean", "RH2M_3d_min", "WS10M_3d_max", "PRECTOT_7d_sum",
    "ndvi", "ndvi_7d_mean",
    "fires_last_7d", "fires_last_14d"
] + available_lags + [
    "pct_Open Water", "pct_Perennial Ice/Snow",
    "pct_Developed, Open Space", "pct_Developed, Low Intensity",
    "pct_Developed, Medium Intensity", "pct_Developed, High Intensity",
    "pct_Barren Land (Rock/Sand/Clay)", "pct_Deciduous Forest",
    "pct_Evergreen Forest", "pct_Mixed Forest", "pct_Shrub/Scrub",
    "pct_Grassland/Herbaceous", "pct_Pasture/Hay",
    "pct_Cultivated Crops", "pct_Woody Wetlands",
    "pct_Emergent Herbaceous Wetlands",
    "vpd_proxy", "hot_windy", "dry_windy", "fuel_moisture_proxy"
]

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

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# Compute class weight
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / max(pos, 1)
print("scale_pos_weight =", scale_pos_weight)

# ----------------------------------------------------------------------
# Optuna objective function
# ----------------------------------------------------------------------
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,

        # Search space
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

    model.set_params(early_stopping_rounds=50)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        # early_stopping_rounds=50,
        verbose=False
    )

    # Evaluate PR-AUC (goal)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    pr_auc = average_precision_score(y_val, y_val_proba)
    roc_auc = roc_auc_score(y_val, y_val_proba)

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

    best_idx = np.argmax(f1_scores)
    precision = float(precisions[best_idx])
    recall = float(recalls[best_idx])
    f1 = float(f1_scores[best_idx])

    # Save additional metrics for visualization
    trial.set_user_attr("roc_auc", roc_auc_score(y_val, y_val_proba))

    # Log to CSV
    row = pd.DataFrame([[trial.number, pr_auc, roc_auc, precision, recall, f1,
        params["learning_rate"], params["max_depth"], params["min_child_weight"],
        params["subsample"], params["colsample_bytree"], params["gamma"],
        params["reg_lambda"], params["reg_alpha"], params["n_estimators"]]],
        columns=pd.read_csv(TRIAL_LOG).columns)

    row.to_csv(TRIAL_LOG, mode="a", index=False, header=False)

    return pr_auc

# ----------------------------------------------------------------------
# Run Optuna
# ----------------------------------------------------------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, show_progress_bar=True)

# Save the study
joblib.dump(study, OPTUNA_STUDY_PATH)

print("\nBest params:", study.best_params)
print("Best PR-AUC:", study.best_value)

# ----------------------------------------------------------------------
# Retrain the BEST model
# ----------------------------------------------------------------------
best_params = study.best_params
best_params.update({
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "scale_pos_weight": scale_pos_weight,
})

model = XGBClassifier(**best_params)

print("\nRetraining best model on full 2020–2022 training...")
model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# Evaluate best model
# ----------------------------------------------------------------------
y_val_proba = model.predict_proba(X_val)[:, 1]

auc_roc = roc_auc_score(y_val, y_val_proba)
auc_pr = average_precision_score(y_val, y_val_proba)

# Optimal threshold (F1)
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx])

print("\nBest threshold:", best_threshold)

# ----------------------------------------------------------------------
# Save visualization curves
# ----------------------------------------------------------------------
# PR Curve
plt.figure()
plt.plot(recalls, precisions)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig(PLOTS_DIR + "precision_recall_curve.png")

# ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_val, y_val_proba)
plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.savefig(PLOTS_DIR + "roc_curve.png")

# Threshold Curve
plt.figure()
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.legend()
plt.title("Threshold vs Precision/Recall")
plt.savefig(PLOTS_DIR + "threshold_curve.png")

# Optuna optimization history
optuna.visualization.plot_optimization_history(study).write_image(
    PLOTS_DIR + "optuna_optimization_history.png"
)

# Hyperparameter importance
optuna.visualization.plot_param_importances(study).write_image(
    PLOTS_DIR + "optuna_param_importance.png"
)

print("\nSaved all plots to:", PLOTS_DIR)

# ----------------------------------------------------------------------
# Save model + threshold + metadata
# ----------------------------------------------------------------------
joblib.dump(model, BEST_MODEL_PATH)
with open(BEST_THRESHOLD_PATH, "w") as f:
    json.dump({"threshold": best_threshold}, f)

metadata = {
    "best_params": best_params,
    "best_threshold": best_threshold,
    "auc_roc": float(auc_roc),
    "auc_pr": float(auc_pr),
    "train_rows": int(len(X_train)),
    "val_rows": int(len(X_val)),
    "train_fire_cases": int(y_train.sum()),
    "val_fire_cases": int(y_val.sum()),
}

with open(BEST_META_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("\nAll done — best model saved!")
