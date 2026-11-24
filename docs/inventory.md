# Source inventory

This repo is a solo rework of **WiFOP (Wildfire Outbreak Prediction)**, the final
group project for ISYE 6740 (Fall 2025), Team 9: Tingya (Letitia) Chang, Diyang Pang,
Yini Li, Xiaoyi Huang, Nilavrah Sensarma. Everything below was copied from two
local source repos (both left untouched):

- **Team repo**: `~/Desktop/Projects/03 WiFOP/` (remote: `github.gatech.edu/nsensarma3/WiFOP`)
- **My solo repo**: `~/Desktop/Projects/Wildfire-Outbreak-Prediction/` (remote: `github.gatech.edu/tchang328/...`)

Only code, configs, and small result artifacts (metrics JSON, feature-importance /
trial-log CSVs, small notebooks) were copied. No raw/interim/processed data and no
model binaries (`.pkl`) were copied — see root `.gitignore`.

Author column is from `git log -1 --format=%an -- <path>` in the team repo (blank
= file was never committed there — it existed locally/untracked only).

## Data & feature pipeline

| New path | Source | Purpose | Inputs | Outputs | Author | Needed going forward? |
|---|---|---|---|---|---|---|
| `scripts/make_grid.py` | solo repo `src/make_grid.py` | Build the 5 km CA grid | CA boundary | `california_grid_5km.csv` | Letitia | Yes — canonical grid builder |
| `src/grid_generation.py` | team `src/grid_generation.py` | Alternate/later grid-generation script | — | grid csv | Letitia | Check for duplication with `make_grid.py`; keep one |
| `scripts/download_NASA_POWER.py` | solo repo | Download NASA POWER weather by tile/date | NASA POWER API | raw weather CSVs | Letitia | Yes |
| `scripts/clean_data_NASA_POWER.py` | solo repo | Clean/reshape NASA POWER tiles | raw POWER CSVs | cleaned weather CSV | Letitia | Yes |
| `scripts/download_api_era5.py` | solo repo | ERA5 weather download (abandoned alt. to NASA POWER) | ERA5 API | raw ERA5 | Letitia | No — superseded by NASA POWER; keep for the "what didn't work" note only |
| `scripts/download_data_firms.py` | solo repo | Download NASA FIRMS fire detections (abandoned alt. to IRWIN) | FIRMS API | raw FIRMS CSV | Letitia | No — team used IRWIN instead; keep for context only |
| `src/nasa_power/nasa_power_data_fetch.bash` | team repo | Shell driver for monthly POWER pulls | NASA POWER API | tile CSVs | Nilavrah Sensarma | No — superseded by `scripts/download_nasa_power_regional.py` (same API/vars, full 2020-2023 range) |
| `src/nasa_power/nasa_power_yearly_data_fetch.bash` | team repo | Shell driver for yearly POWER pulls | NASA POWER API | tile CSVs | Nilavrah Sensarma | No — superseded, same reason. Kept as reference; it's what `scripts/download_nasa_power_regional.py` was modeled on |
| `src/fire_labels/IRWIN.py` | team `src/fire labels/IRWIN.py` | Pull/parse IRWIN fire incident records (the label source) | IRWIN API/export | `irwin.csv` | Xiaoyi Huang | No — superseded by `scripts/download_irwin_wfigs.py`. This script expected a pre-exported `irwin_2023-07.csv` of unknown origin and only covered one month; the public NIFC WFIGS FeatureServer (IRWIN-backed, no account needed) covers the full 2020-2023 range directly |
| `src/fire_labels/FIRMS_data.py` | team `src/fire labels/FIRMS data.py` | Pull FIRMS fire detections (secondary/cross-check) | FIRMS API | FIRMS csv | Xiaoyi Huang | Maybe — cross-validation of labels only |
| `src/fire_labels/fire_data.py` | team `src/fire labels/fire data.py` | Combine/clean fire label sources | IRWIN/FIRMS raw | fire label table | Xiaoyi Huang | Yes, pending review |
| `src/ndvi_pipeline.py` | team repo | Main NDVI pipeline (fetch + daily interpolation, rolling means, drop flags) | NDVI source rasters/API | `ndvi.csv` | Diyang Pang | No — superseded by `scripts/download_ndvi.py` for acquisition (same NOAA VIIRS CDR source, but fetched via HTTP byte-range so only the CA slice of each ~64MB global file is pulled, and matches both the pre-2023 and 2023+ satellite naming). The look-ahead interpolation this script's `clean_panel.py`-style gap-filling might have is moot for the new download path (no gap-filling performed), but still worth reviewing if this script is ever revived |
| `src/ndvi_pipeline_processing/clean_ndvi.py` | team repo | NDVI cleaning helper | raw NDVI | cleaned NDVI | Diyang Pang | Yes, pending review |
| `src/ndvi_pipeline_processing/ndvi_pipeline2.py` | team repo | Second NDVI pipeline variant | raw NDVI | `ndvi.csv` variant | Diyang Pang | Check which NDVI pipeline is canonical |
| `src/ndvi_pipeline_processing/setting.json`, `src/setting.json` | team repo | Config (paths/params) for NDVI pipeline | — | — | Diyang Pang | Yes, as config |
| `src/SRTM_NLCD/nlcd_composition.py` (was `2023_NLCD.py`) | team repo, rewritten | NLCD land-cover fraction extraction per grid cell, now per year (2020-2023) from Annual NLCD | per-year NLCD raster tiles | `nlcd_1km_CA_composition_<year>.csv` | Yini Li (rewrite: Letitia) | Yes — done |
| `src/SRTM_NLCD/slope_aspect.py` | team repo | SRTM-derived slope/aspect features | SRTM DEM | slope/aspect csv | Yini Li | Unused in final feature list (not in `feature_cols` of training scripts) — confirm before dropping |
| `src/combine_data.py` | team repo, rewritten | Joins NASA/NDVI/IRWIN/NLCD onto the grid×date panel; builds `fire_today` | real `nasa.csv`, `ndvi.csv`, `irwin.csv`, per-year `nlcd_1km_CA_composition_<year>.csv`, grid (all 2020-2023) | `california_5km_daily_panel_2020-2023.csv` (26M rows) | Letitia | Yes — done. Also fixed a real bug: the NASA weather join snapped NASA's ~320 coarse station points onto the fine grid instead of the other way around, leaving 99.36% of rows with no weather; now every grid cell is assigned its nearest station. NLCD joins by `(grid_id, year)`. See `docs/results.md` |
| `src/clean_panel.py` | team repo, rewritten | Filters valid cells, interpolates NDVI, adds rolling weather + fire-history + `fire_tomorrow` label + spatial `block_id` | full 2020-2023 daily panel | `california_5km_training_panel_2020-2023_clean.csv` (12.76M rows) | Letitia | Yes — done, leakage issues fixed (see `docs/leakage_audit.md`), run on real data. See `docs/results.md` |

## Modeling scripts

| New path | Source | Purpose | Author | Needed going forward? |
|---|---|---|---|---|
| `src/train_xgboost_weighted.py` / `_v2.py` / `_v3.py` / `_v4.py` | team repo | Weighted XGBoost (class-weight / `scale_pos_weight`), iterative versions | Letitia (v1/v2 committed; v3/v4 untracked — later local iterations, same author) | Keep latest (`_v4`) as the baseline weighted model; drop earlier versions once rewritten pipeline is confirmed |
| `src/train_xgboost_smote.py` / `_v2.py` | team repo | XGBoost + SMOTE oversampling | Letitia | Keep `_v2`; SMOTE is applied after the train/val split, which is correct |
| `src/train_random_forest_wildfire.py` | team repo | Random forest baseline | Letitia | Keep for baseline comparison |
| `src/train_catboost_wildfire.py` | team repo | CatBoost with spatial features | Letitia | Keep — my spatial-features model, relevant to the spatial-CV rework |
| `src/train_lightgbm_wildfire.py` | team repo | LightGBM variant | untracked (no git author — never committed to team repo) | Review before reuse — not part of the reported results table |
| `src/svm_revised_data.py`, `src/svm_downsample_smote.py` | team repo | Linear SVM w/ SMOTE, downsampling | untracked / Diyang Pang | Keep `svm_downsample_smote.py` as the SVM baseline; `svm_revised_data.py` looks like a data-prep variant, review |
| `notebooks/dl_model.ipynb`, `notebooks/xgboost_mlp.ipynb`, `notebooks/xgboost_cnn.ipynb` | team `models/*.ipynb` | MLP / CNN and hybrid models (the "didn't work" results: PR-AUC 0.03–0.05) | mixed team | Keep as reference only, for the README's "what didn't work" section |
| `notebooks/catboost.ipynb` | team `models/catboost.ipynb` | CatBoost notebook (exploratory version of `train_catboost_wildfire.py`) | mixed | Reference only |

## Result artifacts kept

| New path | Source | Notes |
|---|---|---|
| `reports/metrics/xgb_weighted_metadata.json`, `xgb_weighted_threshold.json` | team `models/` | Metadata + F1-optimal threshold for the weighted XGBoost run |
| `reports/metrics/xgb_smote_metadata.json`, `xgb_smote_threshold.json` | team `models/` | Same, for the SMOTE run |
| `reports/metrics/optuna_trial_log_v2.csv`, `_v3.csv` | team `models/` | Optuna hyperparameter search logs for the weighted XGBoost tuning |

## Configs

| New path | Source | Notes |
|---|---|---|
| `configs/wifop.yml` | team repo | Conda env for the original pipeline |
| `configs/requirements.txt` | solo repo | Was empty (0 bytes) in the source repo — needs to be rebuilt from the active conda env |

## Not copied (left in source repos)

- All `data/` (raw, interim, processed, `COMBINE_WIFOP_DATASET`, `final_datasets`) — 8.6 GB, per the no-data rule.
- `models/*.pkl` (trained model binaries), `src/catboost_info/` (CatBoost training logs/tfevents).
- `app/test.app`, `app/test2_YL`, `app/test_YL`, `notebooks/test.ipynb` — all 0 bytes, placeholder/junk files.
- `src/.ipynb_checkpoints/`, `.DS_Store` files.
- Team's original `README.md`, `local_test.ipynb` — kept only as reference material for writing the new README/credits; not copied verbatim.
- Solo repo's `scripts/test.py` — scratch file, not needed.

## Dataset refresh (2026-11-23)

Checked each external source for newer/more comprehensive versions:

- **NLCD → swapping to Annual NLCD.** The original project used a single static
  NLCD land-cover year applied to all of 2020–2023. USGS now publishes **Annual
  NLCD** (Collection 1.2, released June 2026), a yearly land-cover product
  covering 1985–2025. This lets `pct_11`, `pct_21`, … vary by year instead of
  being frozen, which is a real fix, not just a version bump. Decision: rewrite
  `src/SRTM_NLCD/2023_NLCD.py` to pull per-year Annual NLCD (2020, 2021, 2022,
  2023) instead of one static year.
  - Access options considered: (a) MRLC Viewer's Data Download tool — free,
    AOI-based async order (confirmation number + download link, no account
    needed), https://www.mrlc.gov/viewer/ — tried first, but the order
    submission couldn't be confirmed working through the UI (no confirmation
    dialog or order number ever appeared); (b) AWS S3
    `s3://usgs-landcover/annual-nlcd/c1/v0/cu/mosaic/` — Requester Pays, and the
    AWS credentials present on this machine are invalid/expired; (c) **used** —
    the same GeoServer WMS the Viewer itself calls
    (`https://www.mrlc.gov/geoserver/wms`), queried directly with
    `GetMap&FORMAT=image/geotiff&TIME=<year>-01-01` — free, no account/email,
    fully scriptable. `scripts/download_annual_nlcd.py` tiles the CA bounding
    box (EPSG:3857) into 4096x4096px chunks at native 30 m resolution and
    fetches each as its own GeoTIFF.
  - **Correctness note:** the first download pass used WMS `GetMap`, which
    returns a *rendered* image — pixel values are color-palette indices
    (0-16), not real NLCD class codes. Caught this by inspecting a tile's
    `colorinterp` (`palette`) and comparing unique values against the real
    NLCD code list (11, 21, 22, ... 95). Fixed by switching to WCS 1.0.0
    `GetCoverage` in the layer's *native* CRS (EPSG:5070) with a plain
    `time=` param — WCS 2.0.1's `subset=time(...)` syntax hits a server bug
    on this layer, and requesting any CRS other than the native one silently
    returns all-nodata, so both the version and the CRS matter.
  - **Done:** downloaded all 4 years (2020–2023) for California at native 30 m
    resolution via `scripts/download_annual_nlcd.py` — 266 tiles/year, 0
    failures, real class codes verified. Converted to per-year 1 km land-cover
    composition CSVs via `src/SRTM_NLCD/nlcd_composition.py` (renamed from
    `2023_NLCD.py`) — 647,990 rows/year, 16 class columns, in
    `data/processed/nlcd_1km_CA_composition_<year>.csv` (gitignored).
  - **Next:** update `src/combine_data.py` to join NLCD by year as well as by
    grid cell (currently joins a single static NLCD table onto all dates).
- **NASA POWER:** data moved from v10 to v10.2.1 (Sep 2025), API now v2.0 — same
  source, no reason to switch. Decision: keep NASA POWER, just re-pull with the
  current API/version when the pipeline is rebuilt (not a dataset swap).
- **NOAA VIIRS NDVI CDR:** still Version 1; record just extended further in
  time. Decision: no change.
- **IRWIN (fire labels):** backend moved to v11 (March 2026) but it's the same
  restricted-access system (NWCG-affiliated / NIFC open data only); not a
  swappable "newer dataset." Decision: no change.

## Open questions for later steps

1. Which grid-generation script is canonical, `scripts/make_grid.py` (solo) or `src/grid_generation.py` (team)? They need to be diffed before the rewrite.
2. Two NDVI pipelines exist (`ndvi_pipeline.py` vs. `ndvi_pipeline_processing/ndvi_pipeline2.py`) — need to confirm which one produced the `ndvi.csv` actually used in the final combined dataset.
3. `slope_aspect.py` output isn't referenced in any training script's `feature_cols` — looks unused in the final model; confirm with teammates' notes before dropping.
4. ~~The `combine_data.py` / `clean_panel.py` pair in this repo writes/reads filenames ending in `_2023-07`...~~ **Resolved (2026-11-23):** both scripts are rewritten and have now been run end-to-end on real data for the full 2020–2023 range (filenames now end in `_2020-2023`) — see `docs/results.md`. The original team's full-scale script that produced the slide results was never found; this rewrite is a new full-scale pipeline, not a recovery of the original one, so the two aren't guaranteed to match exactly (different NLCD vintage handling, different data-acquisition paths, and the leakage fixes).
