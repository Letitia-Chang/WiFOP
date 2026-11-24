"""
evaluate_model.py

Operational evaluation of a trained fire_tomorrow model on the held-out
2023 test year: top-k recall, lift, calibration, and SHAP feature
importance. This is the evaluation the README calls out as more honest
than a single classification report for this problem — fires are such a
tiny fraction of grid-cell-days that ROC-AUC looks much better than
precision/recall at any one threshold, but the real operational question
is "if we can only issue warnings for the riskiest N% of grid-cell-days,
how many fires do we actually catch?"

Reproduces the same test split as the training scripts (2023 held out,
same feature set) so results are directly comparable to
docs/results.md's ROC-AUC numbers.

Usage:
    python src/evaluate_model.py --model weighted
    python src/evaluate_model.py --model smote
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

DATA_PATH = "../data/processed/california_5km_training_panel_2020-2023_clean.csv"
FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"

FEATURE_COLS = [
    "PRECTOTCORR", "PS", "RH2M", "T2M", "WS10M",
    "PRECTOT_7d_sum", "T2M_3d_mean", "RH2M_3d_min", "WS10M_3d_max",
    "ndvi", "ndvi_7d_mean",
    "fires_last_7d", "fires_last_14d",
    "pct_11","pct_12","pct_21","pct_22","pct_23","pct_24","pct_31",
    "pct_41","pct_42","pct_43","pct_52","pct_71","pct_81","pct_82","pct_90","pct_95"
]
TARGET = "fire_tomorrow"

MODELS = {
    "weighted": {
        "path": "../models/xgb_weighted_model.pkl",
        "metadata": "../reports/metrics/xgb_weighted_metadata_2020-2023.json",
    },
    "smote": {
        "path": "../models/xgb_smote_model.pkl",
        "metadata": "../reports/metrics/xgb_smote_metadata_2020-2023.json",
    },
}


def load_test_set():
    print("Loading cleaned panel...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=FEATURE_COLS + [TARGET, "block_id"])
    test_mask = df["date"].dt.year == 2023
    test = df.loc[test_mask]
    return test[FEATURE_COLS], test[TARGET]


def top_k_table(y_true, y_proba, ks=(0.001, 0.005, 0.01, 0.02, 0.05, 0.10)):
    n = len(y_true)
    base_rate = y_true.mean()
    order = np.argsort(-y_proba)
    y_sorted = np.asarray(y_true)[order]
    rows = []
    for k in ks:
        n_k = max(1, int(round(n * k)))
        caught = y_sorted[:n_k].sum()
        rate_k = caught / n_k
        rows.append({
            "top_k_pct": k * 100,
            "n_flagged": n_k,
            "fires_caught": int(caught),
            "fires_total": int(y_true.sum()),
            "recall": caught / y_true.sum(),
            "precision": rate_k,
            "lift": rate_k / base_rate if base_rate > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def plot_lift_recall(table, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(table["top_k_pct"], table["recall"], marker="o")
    ax1.set_xlabel("Top-k% of grid-cell-days flagged (by predicted risk)")
    ax1.set_ylabel("Recall (fraction of fires caught)")
    ax1.set_title(f"Top-k recall — {model_name}")
    ax1.grid(alpha=0.3)

    ax2.plot(table["top_k_pct"], table["lift"], marker="o", color="darkorange")
    ax2.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax2.set_xlabel("Top-k% of grid-cell-days flagged (by predicted risk)")
    ax2.set_ylabel("Lift (fire rate in top-k ÷ overall fire rate)")
    ax2.set_title(f"Lift — {model_name}")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path = FIG_DIR / f"topk_lift_{model_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_calibration(y_true, y_proba, model_name):
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", label=f"model (Brier={brier:.5f})")
    ax.set_xlabel("Mean predicted probability (per quantile bin)")
    ax.set_ylabel("Observed fire rate")
    ax.set_title(f"Calibration — {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = FIG_DIR / f"calibration_{model_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return brier


def plot_shap(model, X_sample, model_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    fig = plt.figure(figsize=(8, 7))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    fig.tight_layout()
    out_path = FIG_DIR / f"shap_summary_{model_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = pd.Series(mean_abs, index=X_sample.columns).sort_values(ascending=False)
    return ranked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--shap-sample", type=int, default=20000,
                     help="rows to sample for SHAP (full 2.5M-row test set is too slow)")
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = MODELS[args.model]

    print(f"Loading model: {cfg['path']}")
    model = joblib.load(cfg["path"])

    X_test, y_test = load_test_set()
    print(f"Test rows: {len(X_test):,} | fires: {int(y_test.sum())}")

    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Top-k recall / lift ===")
    table = top_k_table(y_test, y_proba)
    print(table.to_string(index=False))
    plot_lift_recall(table, args.model)

    print("\n=== Calibration ===")
    brier = plot_calibration(y_test, y_proba, args.model)
    print(f"Brier score: {brier:.5f}")

    print("\n=== SHAP ===")
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_test), size=min(args.shap_sample, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx]
    shap_ranked = plot_shap(model, X_sample, args.model)
    print("Top 10 features by mean |SHAP value|:")
    print(shap_ranked.head(10).to_string())

    out = {
        "model": args.model,
        "test_rows": int(len(X_test)),
        "test_fires": int(y_test.sum()),
        "brier_score": float(brier),
        "top_k_table": table.to_dict(orient="records"),
        "shap_top10": shap_ranked.head(10).to_dict(),
    }
    out_path = Path(f"../reports/metrics/operational_eval_{args.model}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
