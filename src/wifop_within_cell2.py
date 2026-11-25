"""Outside the repo. Within-cell timing check on the 2023 test rows; run from WiFOP/src after wifop_extra.py.
For each fire row: share of NON-fire rows in the same group with a lower score (ties count half), i.e. a per-group AUROC.
Groups: grid cell (season still counts as 'timing') and grid cell x month (season removed)."""
import json, os, sys
import joblib, numpy as np, pandas as pd
sys.path.insert(0, ".")
from evaluate_model import load_test_set, MODELS, DATA_PATH, TARGET, BASE_FEATURES, NONLINEAR_FEATURES
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
O = "../reports/metrics/"


def grouped_auc(y, s, g, nboot=200, blocks=None):
    d = pd.DataFrame(dict(y=np.asarray(y, int), s=np.asarray(s, float), g=g))
    has = d.groupby("g").y.transform("sum") > 0
    d = d[has]
    d["r"] = d.groupby("g").s.rank(method="average")
    a = d.groupby("g").agg(n=("y", "size"), p=("y", "sum"))
    pr = d[d.y == 1].groupby("g").r.sum()
    a["auc"] = (pr - a.p * (a.p + 1) / 2) / (a.p * (a.n - a.p))
    a = a[(a.n - a.p) > 0]
    w = a.p / a.p.sum()
    out = {"groups": len(a), "fires": int(a.p.sum()), "mean_group_auc": float(a.auc.mean()), "fire_weighted_auc": float((a.auc * w).sum())}
    rng = np.random.RandomState(0)
    bs = [a.auc.to_numpy()[rng.randint(0, len(a), len(a))].mean() for _ in range(nboot)]
    out["mean_group_auc_ci95_groups"] = [float(x) for x in np.percentile(bs, [2.5, 97.5])]
    return out


def main():
    cols = BASE_FEATURES + NONLINEAR_FEATURES
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["vpd_proxy"] = (1 - df["RH2M"] / 100) * df["T2M"]; df["hot_windy"] = df["T2M"] * df["WS10M"]
    df["dry_windy"] = (100 - df["RH2M"]) * df["WS10M"]; df["fuel_moisture_proxy"] = df["ndvi"] * df["RH2M"]
    df = df.dropna(subset=cols + [TARGET, "block_id"]); te = df[df.date.dt.year == 2023]
    keep = (te["fires_last_14d"] == 0).to_numpy()  # rows with no recent fire in the cell: removes post-fire days and follow-on fire rows
    te_all, te = te, te[keep]
    y = te[TARGET].to_numpy(); cell = te.grid_id.to_numpy(); cm = te.grid_id.astype(str) + "_" + te.date.dt.month.astype(str)
    print(len(te), int(y.sum()))
    scores = {}
    for n in ("weighted", "catboost", "rf"):
        c = MODELS[n]
        if c["kind"] == "catboost":
            from catboost import CatBoostClassifier
            m = CatBoostClassifier(); m.load_model(c["path"])
        else:
            m = joblib.load(c["path"])
        scores[n] = m.predict_proba(te[c["features"]])[:, 1]
        scores[n + "_fit_check"] = None
        del scores[n + "_fit_check"]
    k = pd.read_pickle(O + "ablation_test_keys.pkl") if os.path.exists(O + "ablation_test_keys.pkl") else None
    if k is not None and len(k) == len(te_all) and (k.grid_id.to_numpy() == te_all.grid_id.to_numpy()).all() and (k.date.to_numpy() == te_all.date.to_numpy()).all():
        for n in ("full", "no_land_cover", "land_cover_only", "weather_plus_fires", "weather_only", "fires_only"):
            p = O + f"scores_ablation_{n}.npy"
            if os.path.exists(p): scores["ablation_" + n] = np.load(p)[keep]
    else: print("ablation keys missing or misaligned; skipped")
    for q in ("hist_model_rows_all_blocks", "xgb_full_plus_hist__test_rate_all_blocks"):
        z = np.load(O + f"scores_hist_{q}.npy")
        assert len(z) == len(te_all); scores[q] = z[keep]
    res = {}
    for n, s in scores.items():
        res[n] = {"overall_auc": float(roc_auc_score(y, s)), "within_cell": grouped_auc(y, s, cell), "within_cell_month": grouped_auc(y, s, cm.to_numpy())}
        print(n, json.dumps(res[n]))
        json.dump(res, open(O + "within_cell_auc_newfires.json", "w"), indent=2)


if __name__ == "__main__":
    # self-check: perfect within-group ordering -> 1.0, reversed -> 0.0
    y = np.array([0, 0, 1, 0, 1, 0]); g = np.array([1, 1, 1, 2, 2, 2])
    assert grouped_auc(y, [1, 2, 3, 1, 5, 2], g, 5)["mean_group_auc"] == 1.0
    assert grouped_auc(y, [3, 2, 1, 5, 1, 4], g, 5)["mean_group_auc"] == 0.0
    print("selfcheck ok")
    main()
