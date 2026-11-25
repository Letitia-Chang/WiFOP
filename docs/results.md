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

## Critique follow-up: baselines, ablation, tuning, and uncertainty

Everything above treats the five models' numbers at face value. A
self-critique afterward (same instinct as the leakage audit: a result this
clean deserves suspicion, not relief) asked four questions before trusting
any of it further. Short version: **none of the trained models beats a
simple "how often has this grid cell burned before" baseline on PR-AUC or
top-k recall**, hyperparameter tuning doesn't close that gap, and the three
tree models aren't statistically distinguishable from each other. There's a
small, real, above-chance "which day" signal once you control for "which
cell" and exclude days following an active fire — but it's modest, and it's
not clearly coming from weather. Scripts: `src/baseline_models.py`,
`src/ablation_land_cover.py`, `src/tune_models.py`, `src/bootstrap_ci.py`,
plus four follow-up scripts written during this check (`src/wifop_extra.py`,
`src/wifop_hist_feature.py`, `src/wifop_within_cell2.py`, and
`src/wifop_within_cell.py` — the latter is a first, flawed attempt, kept
labeled for the record, not as a result to cite).

### 1. Does the model beat a trivial baseline?

Two baselines, same 2023 test set, same evaluation, with scores heavily
tied (both are effectively low-cardinality: persistence is an integer
count, historical rate has one value per grid cell) — reported
**tie-aware** (expected recall under random tie-breaking at the cutoff, not
just the raw sort order, which is otherwise arbitrary among ties):

| | ROC-AUC | PR-AUC | Top-10% recall (expected) |
|---|---|---|---|
| **historical fire rate per cell** (2020-2022) | 0.902 | **0.102** | **0.789** (worst/best 0.767–0.880) |
| persistence (`fires_last_14d` as score) | 0.684 | 0.065 | 0.434 (mostly tied/random past 2%) |
| XGBoost (weighted, full features) | 0.928 | 0.081 | 0.752 |
| CatBoost | 0.930 | 0.084 | — |
| Random Forest | 0.924 | 0.063 | 0.749 |

**The historical-rate baseline — nothing more than "what fraction of past
days did this exact cell burn on" — beats every trained model on PR-AUC and
top-10% recall.** The models only win on ROC-AUC. The honest statement is
*"the current models do not beat per-cell fire history at ranking,"* not
*"there is no weather signal"* — see sections 2–3 below for what survives
that distinction. The baseline does see slightly more data than the models
(it's computed from all 2020-2022 rows including the validation blocks and
rows the models drop for missing features), a small asymmetry in the
baseline's favor, noted rather than corrected for.
Full numbers: `reports/metrics/baseline_tie_aware.json`.
Figures: `reports/figures/topk_lift_baseline_{historical_rate,persistence}.png`.

### 2. How much of the model's skill is land cover vs. weather?

Four XGBoost refits (same hyperparameters, same split), all scored on one
common row set (2,522,200 test rows, 4,154 fires) so the comparison is
apples-to-apples:

| Feature set | ROC-AUC | PR-AUC | Top-10% recall | Top-0.1% recall (lift) |
|---|---|---|---|---|
| **full** (29 features, control) | 0.928 | 0.081 | 0.752 | 0.133 (133x) |
| land cover only (16 features) | 0.923 | 0.076 | 0.731 | 0.124 (124x) |
| no land cover (13 features: weather/NDVI/fire-history) | 0.876 | 0.068 | 0.574 | 0.129 (129x) |
| weather + fire-history only (strict, drops NDVI/`PS` too) | 0.796 | 0.062 | 0.492 | 0.125 |
| weather only (no fire-history at all) | 0.746 | 0.007 | 0.331 | 0.020 (20.5x) |
| fire-history only (`fires_last_7d`/`14d`) | 0.684 | 0.057 | 0.451 | 0.128 |

**Land cover alone gets most of the way to the full model** (10% recall
0.731 vs. 0.752) and **beats the full model's own first baseline comparison
at the same cutoff** (0.731 vs. historical rate's 0.789 is still a loss, but
land-cover-only is much closer to the real model than weather-only is).
**Weather alone is weak** — it ranks rows well above chance (PR-AUC 0.0071
is ~4.3x the base rate, 0.1% lift of 20.5x) but far below fire history or
land cover, and nowhere near the full model. **The top 0.1% is driven almost
entirely by recent fire history**, not land cover (0.133 full vs. 0.129
no-land-cover at that single cutoff) — land cover matters much more in the
middle of the ranking (10%) than at the very top.
Full numbers: `reports/metrics/ablation_land_cover.json`, `ablation_strict.json`.

### 3. Does giving the model the historical rate as a feature help?

Added the per-cell historical rate as an extra feature to weighted XGBoost
(leave-one-year-out for training rows, to avoid leaking each row's own
label into its own feature) and compared against the baseline alone, with a
**paired block bootstrap** (200 resamples) on the *differences*, which is
the only fair way to ask "does X beat Y" when both are evaluated on the
same spatially-clustered test set:

- XGBoost + historical rate **vs. historical rate alone**: PR-AUC
  difference 95% CI **[-0.017, -0.0001]** — no better, possibly slightly
  worse. Top-10% recall difference CI straddles zero.
- XGBoost + historical rate **vs. XGBoost without it**: recall difference CI
  **[+0.020, +0.048]** (a real improvement), PR-AUC CI straddles zero.
- Historical rate alone **vs. full XGBoost**: PR-AUC difference CI
  **[+0.005, +0.036]** — the baseline really is better, not just noise.
- SHAP (log-odds scale): once included, `hist_rate` is the #1 ranked
  feature, ahead of `pct_21`, `ndvi_7d_mean`, and `PS`.

**Reading:** giving the model the location information helps it a bit over
not having it, but even with that information it still doesn't clearly beat
the raw location signal on the metrics that matter (PR-AUC, top-k recall) —
only on ROC-AUC. Weather and land cover add *some* marginal value on top of
location, not enough to make the combined model a clear win over location
alone with these settings.
Full numbers: `reports/metrics/hist_feature.json`.

### 4. Hyperparameter re-tuning and uncertainty

25-trial Optuna search (PR-AUC objective, spatial-block validation,
unseeded — a caveat, not a correctness issue) per tree model, plus 200-
resample block-bootstrap 95% CIs (resampling spatial blocks, not individual
rows, to respect the spatial autocorrelation the rest of this project
already treats as real) for both the default and tuned versions:

| Model | Test PR-AUC (default) | Test PR-AUC (tuned) | 95% CI (default) |
|---|---|---|---|
| Weighted XGBoost | 0.081 | 0.085 | [0.026, 0.123] |
| CatBoost | 0.084 | 0.083 | [0.024, 0.134] |
| Random Forest | 0.063 | 0.074 | [0.021, 0.110] |

**Tuning makes no detectable difference** — every tuned CI overlaps its
default CI almost entirely (these are separate, not paired, intervals, so
"no detectable difference" is the accurate claim, not "tuning made no
difference"). More importantly, **the default confidence intervals are wide
enough that the three tree models are not separable from each other** — the
~0.003 PR-AUC gap between CatBoost and weighted XGBoost reported earlier in
this doc is well within bootstrap noise, not a real ranking. Best tuned
hyperparameters consistently favor *shallower* trees (e.g. weighted XGBoost:
depth 3, not the default 6) — consistent with the top-k signal being mostly
spatial rather than needing deep interaction terms to model fine-grained
weather dynamics.
Full numbers: `reports/metrics/tuned_{weighted,catboost,rf}_metadata.json`,
`bootstrap_ci_{weighted,catboost,rf,tuned_weighted,tuned_catboost,tuned_rf}.json`.

### 5. Is there real day-to-day "which day" skill, or only "which cell"?

The strongest test: restrict to test rows with **no fire in the same cell
in the past 14 days** (removes "an active fire is still going" as a
shortcut), group rows by grid cell (and separately by cell × month, to
remove season as a form of "timing"), and compute a per-group AUROC (is a
fire day ranked above that cell's own non-fire days?) — chance is 0.5,
and this isolates genuine day-to-day timing skill from "knowing which cells
burn":

| Model | Overall ROC-AUC | Within-cell AUROC | Within-cell-month AUROC |
|---|---|---|---|
| Weighted XGBoost (full) | 0.904 | 0.554 | 0.540 |
| CatBoost | 0.899 | 0.548 | 0.537 |
| Random Forest | 0.898 | 0.540 | 0.528 |
| no land cover | 0.826 | 0.530 | 0.529 |
| land cover only | 0.903 | **0.500** (constant within a cell) | 0.500 |
| weather + fire-history (strict) | 0.707 | 0.515 | 0.514 |
| weather only | 0.706 | 0.513 | **0.503** |
| historical rate alone | 0.860 | 0.500 (constant within a cell) | 0.500 |

**Most of the overall 0.90 ROC-AUC really is "which cell," not "which
day."** A small but real above-chance timing signal survives for the
full-feature models (~0.54-0.55 within-cell, falling to ~0.53-0.54 once
season is also removed) — not nothing, but far from the headline numbers.
**Weather alone is close to pure chance once season is controlled for**
(0.503) — so the modest timing skill that does exist is not clearly a
weather effect. It shows up more with land cover and NDVI present than
without (full 0.546 vs. no-land-cover 0.529, within-cell-month), which is
surprising since land cover is constant within a cell-year and shouldn't
explain *day-to-day* timing on its own — **this is flagged as an unexplained
result**, not attributed to any specific mechanism (candidate explanations
not yet tested: `PS`/NDVI interactions with weather, or an edge-of-season
effect). The within-cell bootstrap also ignores that neighboring cells burn
together (spatial clustering), so these confidence intervals are likely too
narrow.

An earlier version of this check (`src/wifop_within_cell.py`,
`within_cell_auc.json` — removed) didn't exclude post-fire days, which
pushed fire-history-using models to *below* 0.5: it was measuring "an
active fire is ongoing" rather than "predicting a new one." Caught and
redone (`wifop_within_cell2.py`, `within_cell_auc_newfires.json`) before any
conclusion was drawn from it — worth keeping as a record of how easy it is
to build a plausible-looking but wrong evaluation, the same lesson as the
original leakage audit.

### Bottom line

The honest summary, after all four checks: **this project's models are
mostly a sophisticated way of re-deriving which grid cells have historically
been fire-prone**, with a real but modest layer of genuine day-to-day
predictive skill on top, whose source isn't yet pinned down (not clearly
weather, surprisingly tied to land cover/NDVI in a way that isn't
mechanistically explained). This doesn't make the project worthless — a
model that reliably ranks "where" is itself operationally useful, and the
original WUI/land-cover hypothesis is, if anything, reinforced — but it's a
meaningfully more modest claim than "75% recall at the top 10%" suggested on
its own, and it wouldn't have been visible without deliberately trying to
falsify the earlier results.

## Still open

- The grid-generation duplication and NDVI-pipeline duplication questions
  from `docs/inventory.md`'s open questions are unrelated to this run and
  still unresolved.
- ~~**Code duplication risk.**~~ **Done.** `src/features.py` is now the
  single source of truth for `BASE_FEATURES`/`NONLINEAR_FEATURES`/
  `DATA_PATH`/`TARGET`, imported by all nine scripts that previously
  redefined them (`train_xgboost_weighted.py`, `train_xgboost_smote.py`,
  `train_random_forest_wildfire.py`, `train_catboost_wildfire.py`,
  `svm_downsample_smote.py`, `evaluate_model.py`, `ablation_land_cover.py`,
  `bootstrap_ci.py`, `tune_models.py`).
- ~~**Fire-label date semantics were never audited.**~~ **Audited; see
  `docs/leakage_audit.md` section 5.** Two findings: (a) WFIGS's `date`
  field (`FireDiscoveryDateTime`) is a discovery/confirmation timestamp,
  not ignition — but WFIGS has no ignition-date field at all, so this is a
  structural data limitation shared by every public wildfire dataset, not
  a fixable bug; the within-system reporting lag is negligible (~11 min
  median) at the daily granularity this project uses. (b) Found a real,
  separate bug while checking: the label download never filtered by
  incident type, so **2.45% of the 49,539 "fire" labels this whole project
  has used are prescribed burns (RX), not wildfires**. Fixed in
  `scripts/download_irwin_wfigs.py` (now filters `IncidentTypeCategory='WF'`)
  — **not yet re-run through the full pipeline/retrain**, so this doesn't
  change any number reported elsewhere in this document yet.
- ~~**No reproducibility harness.**~~ **Done.** `environment.yml` and
  `requirements.txt` pin the exact package versions this project was run
  with. The leakage-critical logic in `clean_panel.py` was pulled out into
  testable functions (`src/panel_transforms.py`) and
  `tests/test_panel_transforms.py` adds 11 regression tests that would
  fail if any of the 4 leakage fixes were silently reintroduced —
  including the exact example this item named ("assert the rolling
  fire-history feature excludes today"). `.github/workflows/tests.yml`
  runs them on every push/PR (synthetic data only, <1 second — the real
  26M-row pipeline run is deliberately NOT part of CI). Verified the tests
  actually catch a regression, not just trivially pass, by temporarily
  reintroducing the fire-history inclusivity bug and confirming the test
  failed, then restoring the fix. The refactor itself (moving logic from
  script to functions) was not re-verified against the full real dataset
  end-to-end — only against the new unit tests and a manual read-through —
  since running the full pipeline is expensive; flagged here rather than
  assumed.
- **The within-cell bootstrap and the land-cover/NDVI timing-signal result
  (section 5 above) are unexplained** — worth a follow-up investigation
  (e.g. test `PS`/NDVI interaction terms, or an edge-of-season effect)
  before citing that result as more than a flagged oddity.
