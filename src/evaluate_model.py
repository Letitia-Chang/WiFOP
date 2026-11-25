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
    python src/evaluate_model.py --model catboost
    python src/evaluate_model.py --model rf
    python src/evaluate_model.py --model svm
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
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

from features import BASE_FEATURES, DATA_PATH, NONLINEAR_FEATURES, TARGET, add_nonlinear_features

FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"

MODELS = {
    "weighted": {
        "path": "../models/xgb_weighted_model.pkl",
        "metadata": "../reports/metrics/xgb_weighted_metadata_2020-2023.json",
        "features": BASE_FEATURES,
        "kind": "sklearn_tree",
    },
    "smote": {
        "path": "../models/xgb_smote_model.pkl",
        "metadata": "../reports/metrics/xgb_smote_metadata_2020-2023.json",
        "features": BASE_FEATURES,
        "kind": "sklearn_tree",
    },
    "catboost": {
        "path": "../models/catboost_model.cbm",
        "metadata": "../reports/metrics/catboost_metadata_2020-2023.json",
        "features": BASE_FEATURES + NONLINEAR_FEATURES,
        "kind": "catboost",
    },
    "rf": {
        "path": "../models/rf_model.pkl",
        "metadata": "../reports/metrics/rf_metadata_2020-2023.json",
        "features": BASE_FEATURES + NONLINEAR_FEATURES,
        "kind": "sklearn_tree",
    },
    "svm": {
        "path": "../models/svm_model.pkl",
        "metadata": "../reports/metrics/svm_metadata_2020-2023.json",
        "features": BASE_FEATURES + NONLINEAR_FEATURES,
        "kind": "linear_svm_pipeline",
    },
}


def load_test_set(feature_cols):
    print("Loading cleaned panel...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    if set(NONLINEAR_FEATURES) & set(feature_cols):
        add_nonlinear_features(df)

    df = df.dropna(subset=feature_cols + [TARGET, "block_id"])
    test_mask = df["date"].dt.year == 2023
    test = df.loc[test_mask]
    return test[feature_cols], test[TARGET]


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


def _shap_chunk(model, X_chunk):
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X_chunk)


def _tree_shap_values(model, X_sample, n_jobs):
    """TreeExplainer cost scales with rows x tree depth/leaves, not with
    the model object, so splitting rows across processes parallelizes
    almost linearly (no shared state needed — each worker rebuilds its
    own explainer from the model, which is picklable).
    """
    n_jobs = max(1, min(n_jobs, len(X_sample)))
    if n_jobs == 1:
        shap_values = shap.TreeExplainer(model).shap_values(X_sample)
    else:
        chunks = np.array_split(X_sample, n_jobs)
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_shap_chunk, [model] * len(chunks), chunks))
        if isinstance(results[0], list):
            shap_values = [np.concatenate([r[i] for r in results], axis=0) for i in range(len(results[0]))]
        else:
            shap_values = np.concatenate(results, axis=0)

    # Normalize to a single 2-D (rows, features) array for the positive class.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[..., 1]
    return shap_values


def plot_shap(model, X_sample, model_name, kind, n_jobs=1):
    if kind == "linear_svm_pipeline":
        # LinearSVC has no native tree/kernel SHAP support cheap enough for
        # this row count; use the pipeline's own learned coefficients
        # (post feature-selection) as the importance ranking instead, same
        # as svm_downsample_smote.py reports at training time.
        clf = model.named_steps["clf"]
        fs = model.named_steps["feature_selection"]
        selected_mask = fs.get_support()
        feature_names = np.array(X_sample.columns)[selected_mask]
        importance = np.abs(clf.coef_[0])
        ranked = pd.Series(importance, index=feature_names).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(7, 5))
        top = ranked.head(15).iloc[::-1]
        ax.barh(top.index, top.values, color="steelblue")
        ax.set_xlabel("|LinearSVC coefficient| (post feature-selection, standardized inputs)")
        ax.set_title(f"Feature importance — {model_name}")
        fig.tight_layout()
        out_path = FIG_DIR / f"shap_summary_{model_name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path} (coefficient-based, not SHAP — LinearSVC has no cheap SHAP explainer)")
        return ranked

    shap_values = _tree_shap_values(model, X_sample, n_jobs)

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
    ap.add_argument("--shap-jobs", type=int, default=1,
                     help="processes to split the SHAP sample across (tree models only)")
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = MODELS[args.model]
    kind = cfg["kind"]

    print(f"Loading model: {cfg['path']} (kind={kind})")
    if kind == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(cfg["path"])
    else:
        model = joblib.load(cfg["path"])

    X_test, y_test = load_test_set(cfg["features"])
    print(f"Test rows: {len(X_test):,} | fires: {int(y_test.sum())}")

    if kind == "linear_svm_pipeline":
        # LinearSVC has no predict_proba; use a sigmoid of the decision
        # function as a probability-like score for ranking/calibration/
        # Brier. Not a calibrated probability, but preserves ranking
        # exactly (which is all top-k/lift need) and is the standard
        # stand-in used to compare an SVM's calibration to the tree models'.
        scores = model.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-scores))
    else:
        y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Top-k recall / lift ===")
    table = top_k_table(y_test, y_proba)
    print(table.to_string(index=False))
    plot_lift_recall(table, args.model)

    print("\n=== Calibration ===")
    brier = plot_calibration(y_test, y_proba, args.model)
    print(f"Brier score: {brier:.5f}")

    out_path = Path(f"../reports/metrics/operational_eval_{args.model}.json")
    out = {
        "model": args.model,
        "test_rows": int(len(X_test)),
        "test_fires": int(y_test.sum()),
        "brier_score": float(brier),
        "top_k_table": table.to_dict(orient="records"),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path} (top-k/calibration only — SHAP not run yet)")

    print("\n=== SHAP ===" if kind != "linear_svm_pipeline" else "\n=== Feature importance ===")
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_test), size=min(args.shap_sample, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx]

    if kind != "linear_svm_pipeline" and len(X_sample) > 50:
        import time
        probe = X_sample.iloc[:50]
        t0 = time.time()
        shap.TreeExplainer(model).shap_values(probe)
        rate = (time.time() - t0) / len(probe)
        est_serial = rate * len(X_sample)
        print(f"SHAP timing probe: {rate:.4f} s/row serial -> est. {est_serial/60:.1f} min serial "
              f"for {len(X_sample)} rows, ~{est_serial/max(1, args.shap_jobs)/60:.1f} min with "
              f"--shap-jobs {args.shap_jobs}")

    shap_ranked = plot_shap(model, X_sample, args.model, kind, n_jobs=args.shap_jobs)
    print("Top 10 features by importance:")
    print(shap_ranked.head(10).to_string())

    out["shap_top10"] = shap_ranked.head(10).to_dict()
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
