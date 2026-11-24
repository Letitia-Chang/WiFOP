# WiFOP — Wildfire Outbreak Prediction (rework in progress)

Predicting next-day wildfire ignition on a 5 km grid over California using daily
weather, vegetation, and land-cover data, 2020–2023.

This repo is a solo rework of the final project for ISYE 6740 (Computational Data
Analysis), Fall 2025, Georgia Tech. **Original project team (Group 9):**
Tingya (Letitia) Chang, Diyang Pang, Yini Li, Xiaoyi Huang, Nilavrah Sensarma.

**What this rework is:** a cleanup and correction of the original group project,
done solo, after finding a label-leakage bug in the original pipeline (see
[`docs/leakage_audit.md`](docs/leakage_audit.md)). The original results reported
in the class presentation were inflated by that bug; this repo fixes it, adds
proper spatial/temporal validation, and adds operational evaluation (top-k
recall, lift, calibration, SHAP) on top of the original modeling work.

**Status:** leakage fixes + full 2020–2023 pipeline rewrite done and run
end-to-end on real data; see [`docs/results.md`](docs/results.md) for
metrics. Operational evaluation (top-k recall, lift, calibration, SHAP) and
a like-for-like PR-AUC comparison against the original numbers are still
open. See [`docs/inventory.md`](docs/inventory.md) for what was carried
over from the two source repos and what's been rewritten.

## Credit

- Grid generation, data fusion, and the leakage audit/rewrite: Letitia Chang (this repo).
- NASA POWER weather pipeline: Nilavrah Sensarma.
- NDVI pipeline: Diyang Pang.
- NLCD/SRTM land-cover and terrain features: Yini Li.
- IRWIN/FIRMS fire-label pipeline: Xiaoyi Huang.
- Modeling (XGBoost, random forest, CatBoost, SVM, MLP/CNN): joint work, see
  `docs/inventory.md` for per-script authorship.

See [`docs/results.md`](docs/results.md) for the current results table.
Reproduction steps and figures to come; operational evaluation is still open.
