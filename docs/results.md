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

| Model | Train rows | Spatial-val ROC-AUC | **2023 test ROC-AUC** | 2023 test fire recall @ threshold |
|---|---|---|---|---|
| XGBoost, class-weighted (`train_xgboost_weighted.py`) | 6,000,709 | 0.909 | **0.928** | 0.233 (969/4,154 fires caught) |
| XGBoost + SMOTE (`train_xgboost_smote.py`) | 11,981,770 (after SMOTE) | 0.885 | **0.904** | 0.158 (657/4,154 fires caught) |

Fire base rate in the 2023 test year: 4,154 fires out of 2,522,200
grid-cell-days (≈0.16%). Full metadata:
`reports/metrics/xgb_weighted_metadata_2020-2023.json`,
`reports/metrics/xgb_smote_metadata_2020-2023.json` (named `_2020-2023` to
keep the original July-2023-only run's metadata files, which document the
leakage-inflated results that motivated this rework, intact alongside
them).

**Reading these numbers:** a 0.93 ROC-AUC with only ~23% recall at the
F1-optimal threshold is not a contradiction — this is an extremely
imbalanced problem (fires are a tiny fraction of grid-cell-days), so ROC-AUC
(which rewards ranking rare positives above the vast negative class) looks
much better than precision/recall at any single operating threshold. The
project's planned operational evaluation (top-k recall, lift curves,
calibration — see README) is a more honest way to judge this than the
classification report alone, and hasn't been done yet.

**These numbers are not directly comparable to the original project's
reported 0.085-0.087 PR-AUC** — different metric (PR-AUC vs ROC-AUC here),
different target (`fire_today` same-day vs `fire_tomorrow` next-day), and
the leakage fixes removed a feature (`fires_last_7d`/`14d` without the
label leak) that was very likely propping up the original numbers. A
like-for-like PR-AUC comparison is still open.

## Still open

- Operational evaluation (top-k recall, lift, calibration, SHAP) — not run.
- CatBoost/random forest/SVM baselines — not re-run on this data yet, only
  the two XGBoost scripts.
- PR-AUC alongside ROC-AUC, for comparison with the original project's
  reported metric.
- The grid-generation duplication and NDVI-pipeline duplication questions
  from `docs/inventory.md`'s open questions are unrelated to this run and
  still unresolved.
