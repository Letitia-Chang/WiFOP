"""
ablation_land_cover.py

Feature-group ablation for fire_tomorrow: how much of the weighted
XGBoost model's top-k recall/lift comes from static per-year land-cover
composition (pct_*) vs. day-to-day weather/NDVI/fire-history signal?

Land cover doesn't change within a year, so a model that leans heavily on
it (see SHAP rankings in docs/results.md) may mostly be re-deriving "which
grid cells are historically fire-prone" rather than exploiting next-day
weather dynamics. Three models, same XGBoost hyperparameters, same
year-based + spatial-block split as train_xgboost_weighted.py, same test
set and evaluation (src/evaluate_model.py's top_k_table):

  full          — all features (control, should roughly match
                  xgb_weighted_model.pkl's numbers)
  no_land_cover — weather + NDVI + fire-history only, pct_* dropped
  land_cover_only — pct_* only, weather/NDVI/fire-history dropped

If no_land_cover's lift collapses toward the historical-rate baseline
(src/baseline_models.py) while land_cover_only's lift stays close to
full's, that confirms the model's skill is mostly spatial memorization,
not weather-driven next-day prediction.

Usage:
    python src/ablation_land_cover.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from evaluate_model import DATA_PATH, FIG_DIR, TARGET, plot_lift_recall, top_k_table

WEATHER_NDVI_FIRE_FEATURES = [
    "PRECTOTCORR", "PS", "RH2M", "T2M", "WS10M",
    "PRECTOT_7d_sum", "T2M_3d_mean", "RH2M_3d_min", "WS10M_3d_max",
    "ndvi", "ndvi_7d_mean",
    "fires_last_7d", "fires_last_14d",
]
LAND_COVER_FEATURES = [
    "pct_11", "pct_12", "pct_21", "pct_22", "pct_23", "pct_24", "pct_31",
    "pct_41", "pct_42", "pct_43", "pct_52", "pct_71", "pct_81", "pct_82", "pct_90", "pct_95",
]

FEATURE_SETS = {
    "full": WEATHER_NDVI_FIRE_FEATURES + LAND_COVER_FEATURES,
    "no_land_cover": WEATHER_NDVI_FIRE_FEATURES,
    "land_cover_only": LAND_COVER_FEATURES,
}


def load_split(feature_cols):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=feature_cols + [TARGET, "block_id"])

    test_mask = df["date"].dt.year == 2023
    train_pool = df.loc[~test_mask]
    X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, TARGET]

    rng = np.random.RandomState(42)
    blocks = train_pool["block_id"].unique()
    rng.shuffle(blocks)
    n_val_blocks = max(1, int(len(blocks) * 0.2))
    val_blocks = set(blocks[:n_val_blocks])
    val_mask = train_pool["block_id"].isin(val_blocks)

    X_train, y_train = train_pool.loc[~val_mask, feature_cols], train_pool.loc[~val_mask, TARGET]
    return X_train, y_train, X_test, y_test


def run_ablation(name, feature_cols):
    print(f"\n=== Ablation: {name} ({len(feature_cols)} features) ===")
    X_train, y_train, X_test, y_test = load_split(feature_cols)
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale = neg / max(pos, 1)

    model = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=scale, learning_rate=0.05, max_depth=6,
        n_estimators=500, subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"Test ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")

    table = top_k_table(y_test, y_proba)
    print(table.to_string(index=False))
    plot_lift_recall(table, f"ablation_{name}")

    return {
        "ablation": name,
        "n_features": len(feature_cols),
        "test_roc_auc": float(auc),
        "test_pr_auc": float(pr_auc),
        "top_k_table": table.to_dict(orient="records"),
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_ablation(name, cols) for name, cols in FEATURE_SETS.items()]
    out_path = Path("../reports/metrics/ablation_land_cover.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
