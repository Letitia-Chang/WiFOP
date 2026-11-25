"""
clean_panel.py

Prepares the training-ready wildfire panel:
- Loads the 2020-2023 fire-season raw panel
- Tags each grid_id with a spatial block id (for spatial-block CV downstream)
- Filters grid cells with NASA + NLCD coverage
- Fills NDVI (per grid cell, per season) via forward-fill ONLY — no
  interpolation, no bfill (docs/leakage_audit.md #3 and section 6)
- Adds rolling window features (3d / 7d) and fire-history features
  (fires_last_7d/14d, ndvi_7d_mean), all shifted so the current day is
  excluded from its own "last N days" window (docs/leakage_audit.md #2),
  and all grouped by (grid_id, year) so a window never reaches back into
  the previous fire season (docs/leakage_audit.md section 6)
- Builds the fire_tomorrow label by shifting fire_today back one day per
  grid cell PER SEASON, so every feature (keyed to day t) predicts day
  t+1 within the same fire season, matching the "next-day ignition"
  project brief (docs/leakage_audit.md #1 and section 6)
- Outputs a clean ML-ready dataset

The transforms themselves live in panel_transforms.py, which
tests/test_panel_transforms.py regression-tests against synthetic data —
see docs/results.md's "Still open" item #7 for why.
"""

import pandas as pd

from panel_transforms import (
    add_fire_history_features,
    add_ndvi_rolling_mean,
    add_rolling_feature,
    assign_block_id,
    build_fire_tomorrow_label,
    filter_valid_cells,
    interpolate_ndvi,
)

INPUT_FILE  = "../data/processed/california_5km_daily_panel_2020-2023.csv"
OUTPUT_FILE = "../data/processed/california_5km_training_panel_2020-2023_clean.csv"
GRID_FILE   = "../data/california_grid_5km.csv"
SPATIAL_BLOCK_DEG = 0.5  # ~55 km blocks, for spatial-block CV (see docs/leakage_audit.md #4)

print("Loading panel...")
df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

grid = pd.read_csv(GRID_FILE)
df = assign_block_id(df, grid, SPATIAL_BLOCK_DEG)

print("Filtering grid cells with NASA + NLCD coverage...")
df = filter_valid_cells(df)
print(f"Remaining grid cells: {df['grid_id'].nunique()}")
print(f"Remaining rows: {df.shape[0]}")

print("Interpolating NDVI per grid_id...")
df = interpolate_ndvi(df)

print("Adding rolling weather features...")
df = add_rolling_feature(df, "T2M",         3, "mean", "T2M_3d_mean")
df = add_rolling_feature(df, "RH2M",        3, "min",  "RH2M_3d_min")
df = add_rolling_feature(df, "WS10M",       3, "max",  "WS10M_3d_max")
df = add_rolling_feature(df, "PRECTOTCORR", 7, "sum",  "PRECTOT_7d_sum")
df = add_ndvi_rolling_mean(df)

print("Adding fire history features...")
df = add_fire_history_features(df)

print("Building fire_tomorrow label...")
df = build_fire_tomorrow_label(df)

print("Final cleaning...")
# Remove rows where NASA is missing (essential)
df = df[df["PRECTOTCORR"].notna()]

print("Saving clean panel...")
df.to_csv(OUTPUT_FILE, index=False)

print(f"\n🎉 DONE — Clean training panel saved to:\n{OUTPUT_FILE}")
print(f"Final shape: {df.shape}")
