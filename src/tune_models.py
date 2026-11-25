"""
tune_models.py

Hyperparameter re-tuning for the weighted-XGBoost / CatBoost / RF models
against the FIXED pipeline (leakage fixes + year-based + spatial-block
split). The original team's Optuna tuning (train_xgboost_weighted_v3/v4.py)
ran on the old, leaky pipeline, and every model in this rework (see
docs/results.md) used hand-picked default hyperparameters — nobody had
re-tuned against the real, fixed data, so a 3/1000ths PR-AUC gap between
e.g. CatBoost and weighted XGBoost can't be trusted as a real difference;
it could just be tuning noise. This re-tunes each model with a modest
Optuna budget on the spatial-block validation split and re-evaluates the
best config on the 2023 test set.

Deliberately a SMALL trial budget (default 25) — this is meant to remove
the "nobody tuned this at all" confound, not to squeeze out the last 0.001
of PR-AUC. Saves tuned_{model}_model.{pkl,cbm} and
tuned_{model}_metadata.json; does not overwrite the original
*_2020-2023.json files.

Usage:
    python src/tune_models.py --model weighted --trials 25
    python src/tune_models.py --model catboost --trials 25
    python src/tune_models.py --model rf --trials 25
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from features import BASE_FEATURES, DATA_PATH, NONLINEAR_FEATURES, TARGET, add_nonlinear_features


def load_split(feature_cols, add_nonlinear):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    if add_nonlinear:
        add_nonlinear_features(df)

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
    X_val, y_val = train_pool.loc[val_mask, feature_cols], train_pool.loc[val_mask, TARGET]
    return X_train, y_train, X_val, y_val, X_test, y_test


def tune_weighted_xgb(trials):
    from xgboost import XGBClassifier
    X_train, y_train, X_val, y_val, X_test, y_test = load_split(BASE_FEATURES, add_nonlinear=False)
    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    def objective(trial):
        params = dict(
            objective="binary:logistic", eval_metric="logloss", scale_pos_weight=scale,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            random_state=42, n_jobs=-1,
        )
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    print(f"Best val PR-AUC: {study.best_value:.5f} | params: {study.best_params}")

    best = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", scale_pos_weight=scale,
        random_state=42, n_jobs=-1, **study.best_params,
    )
    best.fit(X_train, y_train)
    return best, study, X_val, y_val, X_test, y_test, "../models/tuned_weighted_model.pkl"


def tune_catboost(trials):
    from catboost import CatBoostClassifier
    X_train, y_train, X_val, y_val, X_test, y_test = load_split(
        BASE_FEATURES + NONLINEAR_FEATURES, add_nonlinear=True)
    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    def objective(trial):
        params = dict(
            loss_function="Logloss", eval_metric="AUC", class_weights=[1, scale],
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            iterations=trial.suggest_int("iterations", 300, 1000, step=100),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            thread_count=-1, verbose=False, random_state=42,
        )
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    print(f"Best val PR-AUC: {study.best_value:.5f} | params: {study.best_params}")

    best = CatBoostClassifier(
        loss_function="Logloss", eval_metric="AUC", class_weights=[1, scale],
        thread_count=-1, verbose=200, random_state=42, **study.best_params,
    )
    best.fit(X_train, y_train)
    return best, study, X_val, y_val, X_test, y_test, "../models/tuned_catboost_model.cbm"


def tune_rf(trials):
    X_train, y_train, X_val, y_val, X_test, y_test = load_split(
        BASE_FEATURES + NONLINEAR_FEATURES, add_nonlinear=True)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 8, 25),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    print(f"Best val PR-AUC: {study.best_value:.5f} | params: {study.best_params}")

    best = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42, **study.best_params)
    best.fit(X_train, y_train)
    return best, study, X_val, y_val, X_test, y_test, "../models/tuned_rf_model.pkl"


TUNERS = {"weighted": tune_weighted_xgb, "catboost": tune_catboost, "rf": tune_rf}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=list(TUNERS), required=True)
    ap.add_argument("--trials", type=int, default=25)
    args = ap.parse_args()

    model, study, X_val, y_val, X_test, y_test, model_path = TUNERS[args.model](args.trials)

    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    val_pr_auc = average_precision_score(y_val, y_val_proba)
    val_roc_auc = roc_auc_score(y_val, y_val_proba)
    test_pr_auc = average_precision_score(y_test, y_test_proba)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    print(f"\nFinal — spatial-val ROC-AUC {val_roc_auc:.4f} PR-AUC {val_pr_auc:.4f} | "
          f"test ROC-AUC {test_roc_auc:.4f} PR-AUC {test_pr_auc:.4f}")

    if model_path.endswith(".cbm"):
        model.save_model(model_path)
    else:
        joblib.dump(model, model_path)
    print(f"Saved {model_path}")

    meta = {
        "model": args.model,
        "trials": args.trials,
        "best_params": study.best_params,
        "spatial_val_roc_auc": float(val_roc_auc),
        "spatial_val_pr_auc": float(val_pr_auc),
        "test_2023_roc_auc": float(test_roc_auc),
        "test_2023_pr_auc": float(test_pr_auc),
    }
    meta_path = f"../reports/metrics/tuned_{args.model}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
