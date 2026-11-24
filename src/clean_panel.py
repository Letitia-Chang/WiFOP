"""
clean_panel.py

Prepares the training-ready wildfire panel:
- Loads the 2020-2023 fire-season raw panel
- Tags each grid_id with a spatial block id (for spatial-block CV downstream)
- Filters grid cells with NASA + NLCD coverage
- Interpolates NDVI (per grid), forward-only — no bfill (see docs/leakage_audit.md #3)
- Adds rolling window features (3d / 7d) and fire-history features
  (fires_last_7d/14d, ndvi_7d_mean), all shifted so the current day is
  excluded from its own "last N days" window (docs/leakage_audit.md #2)
- Builds the fire_tomorrow label by shifting fire_today back one day per
  grid cell, so every feature (keyed to day t) predicts day t+1, matching
  the "next-day ignition" project brief (docs/leakage_audit.md #1)
- Outputs a clean ML-ready dataset
"""

import pandas as pd
import numpy as np

INPUT_FILE  = "../data/processed/california_5km_daily_panel_2020-2023.csv"
OUTPUT_FILE = "../data/processed/california_5km_training_panel_2020-2023_clean.csv"
GRID_FILE   = "../data/california_grid_5km.csv"
SPATIAL_BLOCK_DEG = 0.5  # ~55 km blocks, for spatial-block CV (see docs/leakage_audit.md #4)

print("Loading panel...")
df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

# Spatial block id: every grid_id in the same coarse lat/lon bin gets the
# same block, so a later spatial-block split keeps a whole neighborhood on
# one side of train/val instead of leaking adjacent cells across both
# (point #4 in docs/leakage_audit.md — fires cluster geographically).
grid = pd.read_csv(GRID_FILE)
grid["block_id"] = (
    (grid["centroid_lat"] // SPATIAL_BLOCK_DEG).astype(int).astype(str)
    + "_"
    + (grid["centroid_lon"] // SPATIAL_BLOCK_DEG).astype(int).astype(str)
)
df = df.merge(grid[["grid_id", "block_id"]], on="grid_id", how="left")

# ----------------------------------------------------------
# 1. FILTER VALID GRID CELLS (Option C: NASA + NLCD required)
# ----------------------------------------------------------

print("Filtering grid cells with NASA + NLCD coverage...")

# NASA coverage check
has_nasa = df.groupby("grid_id")["PRECTOTCORR"].apply(lambda s: s.notna().any())

# NLCD check
has_nlcd = df.groupby("grid_id")["pct_11"].apply(lambda s: s.notna().any())

valid_ids = has_nasa[has_nasa].index.intersection(has_nlcd[has_nlcd].index)
df = df[df["grid_id"].isin(valid_ids)]

print(f"Remaining grid cells: {len(valid_ids)}")
print(f"Remaining rows: {df.shape[0]}")

# ----------------------------------------------------------
# 2. NDVI INTERPOLATION (PER GRID)
# ----------------------------------------------------------

print("Interpolating NDVI per grid_id...")

df = df.sort_values(["grid_id", "date"])
# Forward-only: interpolate/ffill using past and same-day values only. No
# bfill — that would leak a future satellite reading into an earlier day's
# "known as of day t" feature. A cell whose first reading is missing stays
# NaN here and is dropped in the final-cleaning step below.
df["ndvi"] = df.groupby("grid_id")["ndvi"].transform(
    lambda x: x.interpolate("linear", limit_direction="forward").ffill()
)

# ----------------------------------------------------------
# 3. ROLLING WEATHER FEATURES
# ----------------------------------------------------------

def add_rolling_feature(df, col, window, agg, new_col):
    df[new_col] = (
        df.groupby("grid_id")[col]
          .transform(lambda s: getattr(s.rolling(window), agg)())
    )

print("Adding rolling weather features...")

add_rolling_feature(df, "T2M",          3, "mean", "T2M_3d_mean")
add_rolling_feature(df, "RH2M",         3, "min",  "RH2M_3d_min")
add_rolling_feature(df, "WS10M",        3, "max",  "WS10M_3d_max")
add_rolling_feature(df, "PRECTOTCORR",  7, "sum",  "PRECTOT_7d_sum")

# NDVI rolling — shift(1) first so the current day's own NDVI isn't counted
# as part of its own "last N days" mean (same inclusivity issue as point 2
# in the fire-history features below).
df["ndvi_7d_mean"] = df.groupby("grid_id")["ndvi"].transform(
    lambda s: s.shift(1).rolling(7).mean()
)

# ----------------------------------------------------------
# 4. FIRE HISTORY FEATURES
# ----------------------------------------------------------

print("Adding fire history features...")

# shift(1) before rolling: pandas' rolling() is inclusive of the current
# row, so without the shift, fires_last_7d on day t would include
# fire_today on day t itself — directly encoding the label into a
# "historical" feature. Shifting first means "last 7 days" excludes today.
df["fires_last_7d"]  = df.groupby("grid_id")["fire_today"].transform(lambda s: s.shift(1).rolling(7).sum())
df["fires_last_14d"] = df.groupby("grid_id")["fire_today"].transform(lambda s: s.shift(1).rolling(14).sum())

df["fires_last_7d"] = df["fires_last_7d"].fillna(0)
df["fires_last_14d"] = df["fires_last_14d"].fillna(0)

# ----------------------------------------------------------
# 5. LABEL: NEXT-DAY FIRE (fire_tomorrow)
# ----------------------------------------------------------

print("Building fire_tomorrow label...")

# The project predicts next-day ignition, but fire_today/weather/NDVI/
# fire-history features above are all keyed to the same date. Shifting
# fire_today back one day per grid cell makes the label "did a fire start
# the day AFTER this row's date" while every feature stays keyed to this
# row's date — i.e. only information known by the end of day t predicts
# day t+1. The last date per grid cell has no "tomorrow" to look up and is
# dropped.
df = df.sort_values(["grid_id", "date"])
df["fire_tomorrow"] = df.groupby("grid_id")["fire_today"].shift(-1)
df = df[df["fire_tomorrow"].notna()]
df["fire_tomorrow"] = df["fire_tomorrow"].astype(int)

# ----------------------------------------------------------
# 6. FINAL CLEANING
# ----------------------------------------------------------

print("Final cleaning...")

# Remove rows where NASA is missing (essential)
df = df[df["PRECTOTCORR"].notna()]

print("Saving clean panel...")
df.to_csv(OUTPUT_FILE, index=False)

print(f"\n🎉 DONE — Clean training panel saved to:\n{OUTPUT_FILE}")
print(f"Final shape: {df.shape}")
