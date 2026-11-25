"""Outside the repo; run from WiFOP/src. (1) historical-rate baseline with matched TRAINING rows too;
(2) weighted XGBoost with a historical-rate feature. For training rows the feature is leave-one-YEAR-out, because a
rate that contains the row's own label would leak the target; test rows use all of 2020-2022. (3) paired block bootstrap."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from xgboost import XGBClassifier, DMatrix
from evaluate_model import DATA_PATH, TARGET, BASE_FEATURES
from wifop_extra import tie_aware
O = "../reports/metrics/"


def loyo_rate(tr):
    """Per cell: fire rate over the OTHER training years; NaN if the cell has no row in the other years."""
    yr = tr.date.dt.year
    g = tr.groupby([tr.grid_id, yr])[TARGET].agg(["sum", "count"])
    tot = g.groupby(level=0).sum()
    idx = pd.MultiIndex.from_arrays([tr.grid_id, yr])
    own = g.reindex(idx).to_numpy(); al = tot.reindex(tr.grid_id).to_numpy()
    n = al[:, 1] - own[:, 1]
    return np.where(n > 0, (al[:, 0] - own[:, 0]) / np.maximum(n, 1), np.nan)


def top10(y, s):
    k = int(round(len(y) * .1))
    return tie_aware_recall(y, s, k)


def tie_aware_recall(y, s, nk):
    v, inv = np.unique(-s, return_inverse=True)
    c, f = np.bincount(inv), np.bincount(inv, weights=y); cc, cf = np.cumsum(c), np.cumsum(f)
    g = int(np.searchsorted(cc, nk)); an, af = (cc[g - 1], cf[g - 1]) if g else (0, 0)
    return (af + f[g] * (nk - an) / c[g]) / y.sum()


def main():
    df = pd.read_csv(DATA_PATH, usecols=["date", "grid_id", "block_id", TARGET] + BASE_FEATURES, parse_dates=["date"])
    df = df.dropna(subset=[TARGET, "block_id"])
    ok = df[BASE_FEATURES].notna().all(axis=1)
    yr = df.date.dt.year
    hist_all = df[yr < 2023]                       # what baseline_models.py uses (all rows)
    m = df[ok]; tr_all = m[m.date.dt.year < 2023]; te = m[m.date.dt.year == 2023]
    rng = np.random.RandomState(42); blocks = tr_all["block_id"].unique(); rng.shuffle(blocks)
    val = set(blocks[: max(1, int(len(blocks) * 0.2))])
    tr = tr_all[~tr_all.block_id.isin(val)]
    y = te[TARGET].to_numpy(); seen = ~te.block_id.isin(val).to_numpy()
    print(f"train {len(tr):,} test {len(te):,} fires {int(y.sum())} | test rows in training blocks {int(seen.sum()):,}, fires {int(y[seen].sum())}")

    def rate(src):
        return te.grid_id.map(src.groupby("grid_id")[TARGET].mean()).to_numpy(float)
    r_all, r_nonan, r_match = rate(hist_all), rate(tr_all), rate(tr)
    scores = {"hist_all_rows": np.where(np.isnan(r_all), hist_all[TARGET].mean(), r_all),
              "hist_model_rows_all_blocks": np.where(np.isnan(r_nonan), tr_all[TARGET].mean(), r_nonan),
              "hist_model_rows_train_blocks": np.where(np.isnan(r_match), tr[TARGET].mean(), r_match)}
    print("test rows whose cell has no rate:", int(np.isnan(r_all).sum()), int(np.isnan(r_nonan).sum()), int(np.isnan(r_match).sum()))

    scale = (tr[TARGET] == 0).sum() / max((tr[TARGET] == 1).sum(), 1)
    kw = dict(objective="binary:logistic", eval_metric="logloss", scale_pos_weight=scale, learning_rate=0.05, max_depth=6, n_estimators=500,
              subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    X, Xt = tr[BASE_FEATURES].copy(), te[BASE_FEATURES].copy()
    base = XGBClassifier(**kw).fit(X, tr[TARGET]); scores["xgb_full"] = base.predict_proba(Xt)[:, 1]
    X["hist_rate"] = loyo_rate(tr)
    mdl = XGBClassifier(**kw).fit(X, tr[TARGET])
    for nm, r in (("train_blocks", r_match), ("all_blocks", r_nonan)):  # test feature: rates from the model's own training blocks / all blocks
        Xt["hist_rate"] = r
        scores[f"xgb_full_plus_hist__test_rate_{nm}"] = mdl.predict_proba(Xt)[:, 1]
    res = {}
    for n, s in scores.items():
        for sub, mk in (("all_test", np.ones(len(y), bool)), ("seen_blocks", seen)):
            res[f"{n}__{sub}"] = {"n": int(mk.sum()), "fires": int(y[mk].sum()), "roc_auc": float(auc(y[mk], s[mk])), "pr_auc": float(ap(y[mk], s[mk])),
                                  "tie_aware": tie_aware(y[mk], s[mk])}
            t = res[f"{n}__{sub}"]
            print(n, sub, round(t["roc_auc"], 4), round(t["pr_auc"], 4), [round(x["expected_recall"], 4) for x in t["tie_aware"]], flush=True)
        json.dump(res, open(O + "hist_feature.json", "w"), indent=2, default=float)
    c = mdl.get_booster().predict(DMatrix(Xt.iloc[np.random.RandomState(42).choice(len(Xt), 20000, replace=False)]), pred_contribs=True)[:, :-1]
    sh = pd.Series(np.abs(c).mean(0), index=Xt.columns).sort_values(ascending=False)
    print("SHAP (log-odds scale, all_blocks rate):\n", sh.head(12).to_string()); print("hist_rate rank:", int(list(sh.index).index("hist_rate")) + 1)
    res["shap_top12"] = sh.head(12).to_dict()

    b = te.block_id.to_numpy(); ub, inv = np.unique(b, return_inverse=True); order = np.argsort(inv, kind="stable")
    st = np.searchsorted(inv[order], np.arange(len(ub) + 1)); groups = [order[st[i]:st[i + 1]] for i in range(len(ub))]
    pairs = [("xgb_full_plus_hist__test_rate_train_blocks", "hist_model_rows_train_blocks"), ("xgb_full_plus_hist__test_rate_all_blocks", "hist_model_rows_all_blocks"),
             ("xgb_full_plus_hist__test_rate_all_blocks", "xgb_full"), ("hist_model_rows_all_blocks", "xgb_full")]
    rg = np.random.RandomState(0); D = {f"{a} - {q}": [] for a, q in pairs}
    for _ in range(200):
        ix = np.concatenate([groups[i] for i in rg.randint(0, len(ub), len(ub))]); yy = y[ix]
        if yy.sum() == 0: continue
        need = {k for p in pairs for k in p}; a_ = {k: (ap(yy, scores[k][ix]), tie_aware_recall(yy, scores[k][ix], int(round(len(yy) * .1)))) for k in need}
        for a, q in pairs: D[f"{a} - {q}"].append((a_[a][0] - a_[q][0], a_[a][1] - a_[q][1]))
    for k, v in D.items():
        v = np.array(v); res["boot " + k] = {"pr_auc_diff_ci95": np.percentile(v[:, 0], [2.5, 97.5]).tolist(), "top10_recall_diff_ci95": np.percentile(v[:, 1], [2.5, 97.5]).tolist()}
        print("boot", k, res["boot " + k])
    json.dump(res, open(O + "hist_feature.json", "w"), indent=2, default=float)
    for k, s in scores.items(): np.save(O + f"scores_hist_{k}.npy", s)


if __name__ == "__main__":
    d = pd.DataFrame({"grid_id": [1, 1, 1, 2], "date": pd.to_datetime(["2020-07-01", "2021-07-01", "2021-07-02", "2020-07-01"]), TARGET: [1, 0, 1, 1]})
    r = loyo_rate(d); assert r[0] == .5 and r[1] == 1.0 and r[2] == 1.0 and np.isnan(r[3])
    print("selfcheck ok")
    main()
