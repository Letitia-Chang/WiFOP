"""Outside the repo; run from WiFOP/src after all training. On the NEW 2023 test rows: old models vs new models, historical-rate
baseline (fires of 2020-2022 in the new panel), tie-aware top-k, and paired block bootstrap of differences."""
import json, os, sys
import joblib, numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from features import BASE_FEATURES, NONLINEAR_FEATURES, DATA_PATH, TARGET, add_nonlinear_features
from wifop_extra import tie_aware
from wifop_hist_feature import tie_aware_recall
O = "../reports/metrics/"
OLD = O + "old_run/models/"; NEW = "../models/"
X29, X33 = BASE_FEATURES, BASE_FEATURES + NONLINEAR_FEATURES
SPEC = {"weighted": ("xgb_weighted_model.pkl", X29), "smote": ("xgb_smote_model.pkl", X29), "catboost": ("catboost_model.cbm", X33), "rf": ("rf_model.pkl", X33), "svm": ("svm_model.pkl", X33)}


def score(p, f, te):
    if p.endswith(".cbm"):
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(); m.load_model(p)
    else:
        m = joblib.load(p)
    if not hasattr(m, "predict_proba"): return np.asarray(m.decision_function(te[f]))
    return m.predict_proba(te[f])[:, 1]


df = pd.read_csv(DATA_PATH, parse_dates=["date"]); df = df.dropna(subset=[TARGET, "block_id"]); df = add_nonlinear_features(df)
ok = df[X33].notna().all(axis=1)
hist_src = df[df.date.dt.year < 2023]
te = df[ok & (df.date.dt.year == 2023)]; y = te[TARGET].to_numpy(); print("test", len(te), int(y.sum()))
S = {"hist_rate": te.grid_id.map(hist_src.groupby("grid_id")[TARGET].mean()).fillna(hist_src[TARGET].mean()).to_numpy()}
for n, (fn, f) in SPEC.items():
    for lab, d in (("old", OLD), ("new", NEW)):
        if os.path.exists(d + fn) and (lab == "new" or n in ("weighted", "catboost", "rf")):
            S[f"{n}_{lab}"] = score(d + fn, f, te); print("scored", n, lab, flush=True)
res = {}
for k, s in S.items():
    t = tie_aware(y, s)
    res[k] = {"roc_auc": float(auc(y, s)), "pr_auc": float(ap(y, s)), "tie_aware": t}
    print(k, round(res[k]["roc_auc"], 4), round(res[k]["pr_auc"], 4), [round(float(x["expected_recall"]), 4) for x in t], "lift", [round(float(x["expected_lift"]), 2) for x in t], flush=True)
b = te.block_id.to_numpy(); ub, inv = np.unique(b, return_inverse=True); order = np.argsort(inv, kind="stable"); st = np.searchsorted(inv[order], np.arange(len(ub) + 1)); G = [order[st[i]:st[i + 1]] for i in range(len(ub))]
P = [(f"{n}_new", f"{n}_old") for n in ("weighted", "catboost", "rf") if f"{n}_old" in S and f"{n}_new" in S] + [("hist_rate", f"{n}_new") for n in ("weighted", "catboost", "rf") if f"{n}_new" in S]
need = {k for p in P for k in p}; D = {f"{a} - {q}": [] for a, q in P}; rg = np.random.RandomState(0)
for _ in range(200):
    ix = np.concatenate([G[i] for i in rg.randint(0, len(ub), len(ub))]); yy = y[ix]
    if yy.sum() == 0: continue
    v = {k: (ap(yy, S[k][ix]), auc(yy, S[k][ix]), tie_aware_recall(yy, S[k][ix], int(round(len(yy) * .1)))) for k in need}
    for a, q in P: D[f"{a} - {q}"].append([v[a][i] - v[q][i] for i in range(3)])
for k, v in D.items():
    c = np.percentile(np.array(v), [2.5, 97.5], axis=0); res["boot " + k] = {m: c[:, i].tolist() for i, m in enumerate(("pr_auc", "roc_auc", "top10_recall"))}; print(k, {m: [round(float(x), 4) for x in c[:, i]] for i, m in enumerate(("pr", "roc", "top10"))})
json.dump(res, open(O + "rerun_compare.json", "w"), indent=2, default=float)
