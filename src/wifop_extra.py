"""Outside the repo. (1) tie-aware top-k for the two baselines; (2) stricter ablations. Run from WiFOP/src."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from evaluate_model import DATA_PATH, TARGET, top_k_table
KS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10)
OUT = "../reports/metrics/"


def tie_aware(y, s):
    """Expected recall when ties at the cutoff are broken at random: everything strictly above, plus the tied
    group's fires times the share of the tied group that fits. Also best/worst case tie-breaking."""
    y, s = np.asarray(y), np.asarray(s, float)
    n, tot = len(y), y.sum()
    vals, inv = np.unique(-s, return_inverse=True)  # ascending -s, i.e. highest score first
    cnt, fires = np.bincount(inv), np.bincount(inv, weights=y)
    cc, cf = np.cumsum(cnt), np.cumsum(fires)
    rows = []
    for k in KS:
        nk = max(1, int(round(n * k)))
        g = int(np.searchsorted(cc, nk))  # group containing the cutoff
        above_n, above_f = (cc[g - 1], cf[g - 1]) if g else (0, 0)
        room, gn, gf = nk - above_n, cnt[g], fires[g]
        exp = above_f + gf * room / gn
        lo, hi = above_f + max(0, room - (gn - gf)), above_f + min(room, gf)
        rows.append(dict(top_k_pct=k * 100, n_flagged=nk, tied_at_cutoff=int(gn), distinct_scores=len(vals), expected_recall=exp / tot,
                         worst_recall=lo / tot, best_recall=hi / tot, expected_lift=(exp / nk) / (tot / n)))
    return rows


def baselines():
    from evaluate_model import BASE_FEATURES
    df = pd.read_csv(DATA_PATH, usecols=["date", "grid_id", "block_id", TARGET] + BASE_FEATURES, parse_dates=["date"])
    df = df.dropna(subset=[TARGET, "block_id", "fires_last_14d"])
    df["model_row"] = df[BASE_FEATURES].notna().all(axis=1)  # rows the real models are scored on
    tr, te = df[df.date.dt.year < 2023], df[df.date.dt.year == 2023]
    rate = te.grid_id.map(tr.groupby("grid_id")[TARGET].mean()).fillna(tr[TARGET].mean()).to_numpy()
    res = {}
    for rows, mask in (("all_rows", np.ones(len(te), bool)), ("model_rows", te.model_row.to_numpy())):
        y = te[TARGET].to_numpy()[mask]
        print(rows, len(y), int(y.sum()))
        for name, s in (("historical_rate", rate[mask]), ("persistence", te.fires_last_14d.to_numpy()[mask])):
            res[f"{name}__{rows}"] = {"n": len(y), "fires": int(y.sum()), "roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
                                      "tie_aware": tie_aware(y, s), "script_top_k": top_k_table(pd.Series(y), s).to_dict(orient="records")}
            print(name, rows, json.dumps(res[f"{name}__{rows}"], indent=1, default=float))
    json.dump(res, open(OUT + "baseline_tie_aware.json", "w"), indent=2, default=float)


WEATHER = ["PRECTOTCORR", "RH2M", "T2M", "WS10M", "PRECTOT_7d_sum", "T2M_3d_mean", "RH2M_3d_min", "WS10M_3d_max"]
FIRES = ["fires_last_7d", "fires_last_14d"]
STRICT = {"weather_plus_fires": WEATHER + FIRES, "weather_only": WEATHER, "fires_only": FIRES}


def ablation():
    from ablation_land_cover import FEATURE_SETS
    sets = {**FEATURE_SETS, **STRICT}
    cols = sorted({c for v in sets.values() for c in v})
    df = pd.read_csv(DATA_PATH, usecols=cols + ["date", "grid_id", "block_id", TARGET], parse_dates=["date"])
    df = df.dropna(subset=cols + [TARGET, "block_id"])  # one common row set, so every variant sees the same test rows
    te = df.date.dt.year == 2023
    pool = df[~te]
    rng = np.random.RandomState(42)
    blocks = pool["block_id"].unique()
    rng.shuffle(blocks)
    val = set(blocks[: max(1, int(len(blocks) * 0.2))])
    tr, test = pool[~pool.block_id.isin(val)], df[te]
    print(f"train {len(tr):,} test {len(test):,} fires {int(test[TARGET].sum())}")
    scale = (tr[TARGET] == 0).sum() / max((tr[TARGET] == 1).sum(), 1)
    test[["grid_id", "date", "block_id", TARGET]].to_pickle(OUT + "ablation_test_keys.pkl")
    res = []
    for name, f in sets.items():
        m = XGBClassifier(objective="binary:logistic", eval_metric="logloss", scale_pos_weight=scale, learning_rate=0.05, max_depth=6,
                          n_estimators=500, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        m.fit(tr[f], tr[TARGET])
        p = m.predict_proba(test[f])[:, 1]
        np.save(OUT + f"scores_ablation_{name}.npy", p)
        r = {"ablation": name, "features": f, "test_roc_auc": float(roc_auc_score(test[TARGET], p)), "test_pr_auc": float(average_precision_score(test[TARGET], p)),
             "top_k_table": top_k_table(test[TARGET], p).to_dict(orient="records")}
        print(name, len(f), r["test_roc_auc"], r["test_pr_auc"])
        print(pd.DataFrame(r["top_k_table"]).to_string(index=False))
        res.append(r)
        json.dump(res, open(OUT + "ablation_strict.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    y, s = np.array([1, 0, 0, 1]), np.array([1., 1., 0., 0.])  # self-check: k=50% takes the tied top pair exactly
    globals()["KS"] = (0.5,); r = tie_aware(y, s)[0]
    assert abs(r["expected_recall"] - 0.5) < 1e-12 and r["best_recall"] == r["worst_recall"] == 0.5
    KS = (0.25,); r = tie_aware(y, s)[0]
    assert abs(r["expected_recall"] - 0.25) < 1e-12 and r["best_recall"] == 0.5 and r["worst_recall"] == 0.0
    KS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10)
    baselines(); ablation()
