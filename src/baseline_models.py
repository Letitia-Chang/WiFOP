"""
baseline_models.py

Sanity-check baselines for fire_tomorrow, evaluated the same way as the
real models (src/evaluate_model.py's top-k recall/lift table) on the same
2023 held-out test set. Needed before trusting any model's "7.5x lift at
top 10%" claim: without an anchor, a trivial score could achieve similar
numbers for free (see docs/critique_2026-11.md item #4).

Two baselines, both require no training beyond a per-grid-cell lookup:

1. historical_rate — score each test row by its grid cell's fire rate in
   the 2020-2022 training period. Constant across time, varies only by
   location. If this baseline nearly matches the real models' lift, the
   models are mostly re-deriving "which cells are historically fire-prone"
   rather than exploiting day-to-day weather/NDVI signal.

2. persistence — score each test row by its own fires_last_14d feature
   (already lagged to exclude today, see docs/leakage_audit.md #2). This
   is the simplest "if it's been burning lately, it's still risky" rule,
   using no land-cover or weather information at all.

Usage:
    python src/baseline_models.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate_model import DATA_PATH, FIG_DIR, TARGET, plot_lift_recall, top_k_table

BLOCK_COL = "block_id"


def load_panel():
    print("Loading cleaned panel...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=[TARGET, BLOCK_COL, "fires_last_14d"])
    return df


def run_baseline(name, y_true, score):
    auc = roc_auc_score(y_true, score)
    pr_auc = average_precision_score(y_true, score)
    print(f"\n--- {name} ---")
    print(f"ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")
    table = top_k_table(y_true, score)
    print(table.to_string(index=False))
    plot_lift_recall(table, name)
    return {
        "model": name,
        "roc_auc": float(auc),
        "pr_auc": float(pr_auc),
        "top_k_table": table.to_dict(orient="records"),
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_panel()

    train_mask = df["date"].dt.year < 2023
    test_mask = df["date"].dt.year == 2023
    train, test = df.loc[train_mask], df.loc[test_mask]
    print(f"Train rows: {len(train):,} | Test rows: {len(test):,} | Test fires: {int(test[TARGET].sum())}")

    # --- Historical fire-rate baseline ---
    cell_rate = train.groupby("grid_id")[TARGET].mean()
    overall_rate = train[TARGET].mean()
    hist_score = test["grid_id"].map(cell_rate).fillna(overall_rate).to_numpy()

    # --- Persistence baseline ---
    persistence_score = test["fires_last_14d"].to_numpy()

    results = [
        run_baseline("baseline_historical_rate", test[TARGET], hist_score),
        run_baseline("baseline_persistence", test[TARGET], persistence_score),
    ]

    out_path = Path("../reports/metrics/baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
