"""
bootstrap_ci.py

Block-bootstrap confidence intervals for a model's 2023 test-set PR-AUC,
ROC-AUC, and top-10%-recall. Every number in docs/results.md so far is a
single point estimate on one held-out year — there's no sense of how much
those numbers would move on a different sample of the same underlying
risk. Resampling BLOCKS (not individual rows) respects the spatial
autocorrelation the rest of this project already treats as real (see
docs/leakage_audit.md #4) — resampling rows directly would understate the
true variance, since nearby grid-cell-days aren't independent.

Usage:
    python src/bootstrap_ci.py --model weighted --n-boot 200
    python src/bootstrap_ci.py --model tuned_weighted --n-boot 200
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features import BASE_FEATURES, DATA_PATH, NONLINEAR_FEATURES, TARGET, add_nonlinear_features

MODELS = {
    "weighted": {"path": "../models/xgb_weighted_model.pkl", "features": BASE_FEATURES, "kind": "sklearn"},
    "smote": {"path": "../models/xgb_smote_model.pkl", "features": BASE_FEATURES, "kind": "sklearn"},
    "catboost": {"path": "../models/catboost_model.cbm", "features": BASE_FEATURES + NONLINEAR_FEATURES, "kind": "catboost"},
    "rf": {"path": "../models/rf_model.pkl", "features": BASE_FEATURES + NONLINEAR_FEATURES, "kind": "sklearn"},
    "tuned_weighted": {"path": "../models/tuned_weighted_model.pkl", "features": BASE_FEATURES, "kind": "sklearn"},
    "tuned_catboost": {"path": "../models/tuned_catboost_model.cbm", "features": BASE_FEATURES + NONLINEAR_FEATURES, "kind": "catboost"},
    "tuned_rf": {"path": "../models/tuned_rf_model.pkl", "features": BASE_FEATURES + NONLINEAR_FEATURES, "kind": "sklearn"},
}


def load_test_set(feature_cols):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    if set(NONLINEAR_FEATURES) & set(feature_cols):
        add_nonlinear_features(df)
    df = df.dropna(subset=feature_cols + [TARGET, "block_id"])
    test = df.loc[df["date"].dt.year == 2023]
    return test[feature_cols], test[TARGET], test["block_id"].to_numpy()


def top10_recall(y_true, score):
    n = len(y_true)
    n_k = max(1, int(round(n * 0.10)))
    order = np.argsort(-score)
    caught = np.asarray(y_true)[order][:n_k].sum()
    total = np.asarray(y_true).sum()
    return caught / total if total > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = MODELS[args.model]
    print(f"Loading model: {cfg['path']}")
    if cfg["kind"] == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(cfg["path"])
    else:
        model = joblib.load(cfg["path"])

    X_test, y_test, block_ids = load_test_set(cfg["features"])
    y_test = y_test.to_numpy()
    score = model.predict_proba(X_test)[:, 1]
    print(f"Test rows: {len(y_test):,} | fires: {int(y_test.sum())} | distinct blocks: {len(np.unique(block_ids))}")

    point_pr_auc = average_precision_score(y_test, score)
    point_roc_auc = roc_auc_score(y_test, score)
    point_top10 = top10_recall(y_test, score)
    print(f"Point estimate — PR-AUC {point_pr_auc:.4f} | ROC-AUC {point_roc_auc:.4f} | top-10% recall {point_top10:.4f}")

    # Pre-group row indices by block for fast resampling.
    block_to_idx = {}
    for i, b in enumerate(block_ids):
        block_to_idx.setdefault(b, []).append(i)
    unique_blocks = np.array(list(block_to_idx.keys()))

    rng = np.random.RandomState(args.seed)
    pr_aucs, roc_aucs, top10s = [], [], []
    for b in range(args.n_boot):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([block_to_idx[blk] for blk in sampled_blocks])
        y_b, s_b = y_test[idx], score[idx]
        if y_b.sum() == 0:
            continue
        pr_aucs.append(average_precision_score(y_b, s_b))
        roc_aucs.append(roc_auc_score(y_b, s_b))
        top10s.append(top10_recall(y_b, s_b))
        if (b + 1) % 50 == 0:
            print(f"  bootstrap {b + 1}/{args.n_boot}")

    def ci(values):
        lo, hi = np.percentile(values, [2.5, 97.5])
        return float(lo), float(hi)

    result = {
        "model": args.model,
        "n_boot": len(pr_aucs),
        "point_estimate": {"pr_auc": float(point_pr_auc), "roc_auc": float(point_roc_auc), "top10_recall": float(point_top10)},
        "ci_95": {
            "pr_auc": ci(pr_aucs),
            "roc_auc": ci(roc_aucs),
            "top10_recall": ci(top10s),
        },
    }
    print(f"\n95% CI (block bootstrap, n={len(pr_aucs)}):")
    print(f"  PR-AUC:        {point_pr_auc:.4f}  [{result['ci_95']['pr_auc'][0]:.4f}, {result['ci_95']['pr_auc'][1]:.4f}]")
    print(f"  ROC-AUC:       {point_roc_auc:.4f}  [{result['ci_95']['roc_auc'][0]:.4f}, {result['ci_95']['roc_auc'][1]:.4f}]")
    print(f"  Top-10% recall:{point_top10:.4f}  [{result['ci_95']['top10_recall'][0]:.4f}, {result['ci_95']['top10_recall'][1]:.4f}]")

    out_path = Path(f"../reports/metrics/bootstrap_ci_{args.model}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
