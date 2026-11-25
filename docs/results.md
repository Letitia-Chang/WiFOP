# Results — full 2020-2023 pipeline on real data

This is the first run of the rewritten pipeline (leakage fixes + per-year
NLCD + year-based/spatial-block split) against real data for the full
project range, 2020-2023 fire season (Jun-Sep) in California. Earlier runs
in this repo's history used either synthetic data (to verify the code ran)
or a single-month local test panel inherited from the team repo.

## Data acquisition (see docs/inventory.md for details)

All sources were re-downloaded directly from their public APIs for the
full 2020-2023 range, since the original team's raw exports/data weren't
available:

- **NASA POWER weather**: `scripts/download_nasa_power_regional.py`, the
  regional API, fire season Jun-Sep each year. 165,920 rows.
- **IRWIN fire labels**: `scripts/download_irwin_wfigs.py`, NIFC's public
  WFIGS_Incident_Locations FeatureServer (IRWIN-backed, no account/token
  needed) in place of the original (unscripted, one-month-only) IRWIN
  export. 49,539 CA incidents.
- **NDVI**: `scripts/download_ndvi.py`, NOAA VIIRS NDVI CDR, fetched via
  HTTP byte-range requests so only the CA-sized slice of each global daily
  file is transferred. 13.5M rows across all 4 years.
- **NLCD**: `scripts/download_annual_nlcd.py`, Annual NLCD via WCS, native
  30m resolution, one composition table per year.
- **Grid**: `src/grid_generation.py`, 53,345 cells at 5km resolution.

## Bugs found and fixed while running this for real

1. **NASA weather join was backwards.** `combine_data.py`'s
   `assign_grid_id` snapped each of NASA POWER's ~320 coarse (0.5°) station
   points onto its single nearest fine (5km) grid cell — so only ~320 of
   53,345 grid cells ever got a direct weather value (99.36% missing in the
   panel). Fixed by assigning each grid cell to its nearest station instead
   (the join direction that actually makes sense for a coarse source onto a
   fine grid), via a separate BallTree lookup. Confirmed with a synthetic
   test before re-running on the real 26M-row panel; weather coverage went
   from 0.64% to 100%.
2. **NDVI download pattern only matched one satellite.** NOAA's VIIRS NDVI
   CDR changed source satellite/product name mid-series (`NPP13C1_S-NPP`
   through 2022, `JP113C1_NOAA-20` from 2023); the original regex only
   matched the 2023 naming, so the first download run silently got only one
   of the four years. Fixed by matching both naming patterns.
3. **WMS vs WCS for NLCD** (found earlier, see docs/inventory.md) — WMS
   `GetMap` returns rendered color-palette indices, not real land-cover
   codes; switched to WCS 1.0.0 `GetCoverage` in the native CRS.

## Final pipeline run

- `combine_data.py`: 26,032,360 rows (53,345 grid cells × ~488 fire-season
  dates, 2020-2023). NASA weather 0% missing; NDVI/NLCD ~51-54% missing
  (expected — the rectangular grid bbox includes ocean and out-of-state
  corners that have no land-cover/vegetation data). fire_today rate ≈
  0.092%.
- `clean_panel.py`: 26,195 grid cells retained (NASA + NLCD coverage
  required); final clean training panel 12,756,965 rows, 34 columns
  (includes `fire_tomorrow` label, `block_id` for spatial-block CV, and all
  leakage-fixed rolling/NDVI features).

## Model results

Both models predict `fire_tomorrow` (next-day ignition), trained on
2020-2022, with a spatial-block validation split (20% of ~55km blocks) used
for threshold tuning, and 2023 held out entirely as the test year.

| Model | Train rows | Spatial-val ROC-AUC | **2023 test ROC-AUC** | 2023 test AUC-PR | 2023 test fire recall @ threshold |
|---|---|---|---|---|---|
| XGBoost, class-weighted (`train_xgboost_weighted.py`) | 6,000,709 | 0.909 | **0.928** | 0.081 | 0.233 (969/4,154 fires caught) |
| XGBoost + SMOTE (`train_xgboost_smote.py`) | 11,981,770 (after SMOTE) | 0.885 | **0.904** | 0.057 | 0.158 (657/4,154 fires caught) |
| CatBoost (`train_catboost_wildfire.py`) | 6,000,709 | 0.911 | **0.927** | 0.084 | 0.246 (1,022/4,154 fires caught) |
| Random Forest (`train_random_forest_wildfire.py`) | 6,000,709 | 0.902 | **0.924** | 0.063 | 0.261 (1,085/4,154 fires caught) |
| Linear SVM + downsample + SMOTE (`svm_downsample_smote.py`) | 206,304 (downsampled 20:1 + SMOTE) | 0.837 | **0.877** | 0.012 | 0.322 (1,339/4,154 fires caught) |

Fire base rate in the 2023 test year: 4,154 fires out of 2,522,200
grid-cell-days (≈0.16%). Full metadata in `reports/metrics/*_2020-2023.json`
per model (named `_2020-2023` to keep the original July-2023-only run's
metadata files, which document the leakage-inflated results that motivated
this rework, intact alongside them).

**All 5 models land in the same 0.88-0.93 ROC-AUC range** — the gap between
boosted trees (XGBoost/CatBoost, ~0.92-0.93) and Random Forest (0.924) is
small, and the linear SVM is the clear laggard (0.877), consistent with
this being a problem with real nonlinear structure (land-cover thresholds,
interaction effects) that a linear decision boundary can't fully capture.
CatBoost edges out XGBoost-weighted slightly on both ROC-AUC and AUC-PR
without any of the original script's hand-engineered spatial neighbor
features (which weren't reproduced here — see `train_catboost_wildfire.py`'s
docstring). The SVM's much higher recall-at-threshold (32%) is a side effect
of its heavy downsampling + SMOTE + F1-tuned threshold, not evidence it's
actually better operationally — its AUC-PR/precision at that recall level
would need checking (not computed here) before reading too much into it.

The **original train_random_forest_wildfire.py / train_catboost_wildfire.py
/ svm_downsample_smote.py all assumed a different, never-recovered
full-scale dataset** (`ca_5km_daily_panel_2020_2023_6_9.csv`, with lat/lon
per row and full-name NLCD columns) — see each script's docstring for what
was changed to run them against this rework's actual clean panel instead.

## PR-AUC comparison against the original project's numbers

The original project's best reported PR-AUC was **0.0851**
(`reports/metrics/optuna_trial_log_v3.csv`, trial 16 — a hyperparameter
search for `train_xgboost_weighted`, same-day `fire_today` target, on the
July-2023-only local test panel, with the `fires_last_7d`/`14d` label-leak
bug from `docs/leakage_audit.md` #2 still present). Its sibling
`optuna_trial_log_v2.csv` tops out at 0.070. These are the numbers the
leakage audit's "original results were inflated" claim refers to.

| Model (this rework, real 2020-2023 data, `fire_tomorrow`) | 2023 test PR-AUC |
|---|---|
| Original project's best (leakage-inflated, `fire_today`, July 2023 only) | 0.0851 |
| CatBoost | 0.084 |
| XGBoost, class-weighted | 0.081 |
| Random Forest | 0.063 |
| XGBoost + SMOTE | 0.057 |
| Linear SVM + downsample + SMOTE | 0.012 |

**Reading this comparison:** CatBoost and the weighted XGBoost land within
about 0.001-0.004 of the original's 0.0851 — on a *harder*, more honest
problem. The comparison isn't apples-to-apples in the original's favor:
this run predicts tomorrow's fire from today's information (`fire_tomorrow`)
instead of today's fire from today's information (`fire_today`), evaluates
on a full year the model never saw rather than a 20% random split of the
same month it trained on, and the leaking `fires_last_7d`/`14d` feature
that was very likely propping up the original number is gone. Matching
the original's PR-AUC on a meaningfully harder task is a good sign for this
rework, not a wash — the original 0.085 wasn't "the right answer with some
noise removed," it was inflated by a bug, and the fact that fixing the bug
and making the task strictly harder (next-day, held-out year, no spatial
leakage) still lands in the same PR-AUC neighborhood suggests the
underlying signal (weather + land cover + vegetation) is real and roughly
this strong, not that the original number was an unreachable ceiling.

**Reading these numbers:** a 0.93 ROC-AUC with only ~23% recall at the
F1-optimal threshold is not a contradiction — this is an extremely
imbalanced problem (fires are a tiny fraction of grid-cell-days), so ROC-AUC
(which rewards ranking rare positives above the vast negative class) looks
much better than precision/recall at any single operating threshold. The
operational evaluation below (top-k recall, lift curves, calibration) is a
more honest way to judge this than the classification report alone.

**ROC-AUC here isn't directly comparable to the original project's
reported 0.085-0.087 figure, which was PR-AUC, not ROC-AUC** — see the
PR-AUC comparison section below for the like-for-like version of this
comparison.

## Operational evaluation

`src/evaluate_model.py` (2023 held-out test set, 2,522,200 rows, 4,154
fires ≈ 0.16% base rate). This is the more honest way to judge this problem
than a single classification report — ROC-AUC rewards ranking rare
positives above the vast negative class, but the operational question is
"if we can only act on the riskiest N% of grid-cell-days, how many fires
does that actually catch?"

All five models were run through the same evaluation (`src/evaluate_model.py
--model {weighted,smote,catboost,rf,svm}`, default 20,000-row SHAP sample,
same 2023 held-out test set). The RF run used `--shap-jobs 16` on PACE-ICE
(GT's HPC cluster) — `shap.TreeExplainer` on a 300-tree, depth-20 forest was
too slow single-threaded locally (a 20,000-row serial run was projected at
~7.6h; splitting rows across 16 processes finished the real run in 31
minutes). Everything else ran locally.

### Top-k recall / lift

| Top-k% flagged | Weighted | SMOTE | CatBoost | RF | SVM |
|---|---|---|---|---|---|
| 0.1% | 12.9% (129x) | 11.0% (110x) | 13.7% (137x) | 11.6% (116x) | 0.5% (5x) |
| 0.5% | 27.1% (54x) | 18.2% (36x) | 27.6% (55x) | 23.3% (47x) | 4.8% (10x) |
| 1.0% | 34.9% (35x) | 23.7% (24x) | 35.2% (35x) | 30.7% (31x) | 10.1% (10x) |
| 2.0% | 44.2% (22x) | 33.0% (17x) | 44.7% (22x) | 41.2% (21x) | 25.4% (13x) |
| 5.0% | 59.9% (12x) | 49.8% (10x) | 60.9% (12x) | 59.5% (12x) | 45.6% (9x) |
| 10.0% | 75.1% (7.5x) | 66.9% (6.7x) | 75.1% (7.5x) | 74.9% (7.5x) | 60.3% (6.0x) |

Figures are the recall/lift values; parenthesized lift as before.

Flagging just the riskiest 10% of grid-cell-days catches **~75%** of all
fires with any of the three best models (weighted XGBoost, CatBoost, RF) —
a far more useful number for an early-warning use case than the 20-30%
recall seen in the classification reports at the F1-optimal threshold.
**CatBoost and weighted XGBoost are essentially tied and both lead RF and
SMOTE at every top-k level**; the linear SVM is a clear laggard throughout,
consistent with it also having the weakest ROC-AUC/PR-AUC of the five.
Figures: `reports/figures/topk_lift_{weighted,smote,catboost,rf,svm}.png`.

### Calibration

| Model | Brier score |
|---|---|
| SMOTE | 0.0264 |
| RF | 0.0227 |
| Weighted | 0.0696 |
| CatBoost | 0.0855 |
| SVM | 0.1664 |

Brier score alone is misleading here and shouldn't be read as a ranking of
model quality — it rewards predicting probabilities close to the (tiny,
~0.16%) true base rate everywhere, which a model that's *worse* at ranking
can still do well if its probabilities are more conservative on average.
RF's low Brier score in particular is deceptive: its probabilities are
badly overconfident given the base rate (a constant base-rate prediction
would itself score ~0.00164 Brier, so 0.0227 reflects systematic
overestimation, not good calibration), likely an artifact of
`class_weight="balanced"` reweighting the loss without recalibrating the
output. SVM's score is similarly not a true probability — it's a sigmoid of
`decision_function`, used only to make calibration/Brier computable at all,
since `LinearSVC` has no native `predict_proba`. Given CatBoost/weighted
XGBoost/RF all win on top-k recall/lift above, those are the better choices
for this use case despite not having the lowest Brier scores; calibration
would matter more if the predicted probabilities were used directly (e.g.
cost-weighted decisions) rather than for ranking/thresholding.
Figures: `reports/figures/calibration_{weighted,smote,catboost,rf,svm}.png`.

### SHAP feature importance

The three tree models (weighted XGBoost, CatBoost, RF) agree closely on the
top drivers, dominated by land-cover composition, not weather:

| Rank | Weighted | SMOTE | CatBoost | RF |
|---|---|---|---|---|
| 1 | `pct_21` (developed, open space) | `pct_21` | `pct_21` | `pct_21` |
| 2 | `PS` (surface pressure) | `PS` | `PS` | `pct_22` (developed, low intensity) |
| 3 | `ndvi_7d_mean` | `pct_22` | `pct_22` | `pct_23` (developed, medium intensity) |
| 4 | `pct_22` | `PRECTOTCORR` | `ndvi_7d_mean` | `PS` |
| 5 | `pct_52` (shrub/scrub) | `pct_23` | `pct_23` | `ndvi_7d_mean` |

`pct_21`/`pct_22`/`pct_23` (developed land at increasing intensity) ranking
above raw weather variables, across all three tree models, is consistent
with wildfire ignition being strongly tied to human activity/wildland-urban
interface, not just fuel/fire weather — matches the original project's
stated WUI hypothesis. Worth flagging as a direction to dig into further
(e.g. ignition-cause breakdown if available) rather than treating as fully
explained here.

The linear SVM's top features are different in kind, not just order —
`vpd_proxy`, `T2M`, `RH2M` (weather/fire-weather features) dominate instead
of land cover. This isn't SHAP for the SVM (LinearSVC has no cheap SHAP
explainer); it's `|coefficient|` on the standardized, feature-selected
inputs, so it's not directly comparable in magnitude to the tree models'
SHAP values, only in rank — but the fact that a linear model leans on
weather while every tree model leans on land cover suggests the land-cover
signal is non-linear or threshold-like (e.g. "near this much developed
land" rather than "more developed land is linearly riskier"), which a
linear decision boundary can't capture as well.
Figures: `reports/figures/shap_summary_{weighted,smote,catboost,rf,svm}.png`.
Full numbers: `reports/metrics/operational_eval_{weighted,smote,catboost,rf,svm}.json`.

## Still open

- The grid-generation duplication and NDVI-pipeline duplication questions
  from `docs/inventory.md`'s open questions are unrelated to this run and
  still unresolved.
