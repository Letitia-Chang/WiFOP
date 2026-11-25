"""
panel_transforms.py

The core per-grid-cell transforms used by clean_panel.py, pulled out into
testable functions. This exists so the fixes documented in
docs/leakage_audit.md have a regression test guarding them
(tests/test_panel_transforms.py) — before this, the logic lived as
top-level script code in clean_panel.py, which meant a future edit could
silently reintroduce one of the fixed leakage bugs with nothing to catch
it (see docs/results.md's "Still open" item #7). clean_panel.py imports
and calls these in the same order as before.

2026-11-25 correction (docs/leakage_audit.md section 6): two more bugs
found while preparing a rerun, both present since this file was first
written and NOT caught by the original leakage audit or its tests:

1. `interpolate_ndvi`'s `.interpolate("linear", limit_direction="forward")`
   does NOT make interior gaps forward-only — `limit_direction` only
   controls which end of a run of NaNs gets extrapolated; a gap *between*
   two known readings is always linearly interpolated using BOTH
   endpoints regardless of `limit_direction`. So an interior NDVI gap
   still leaked a future satellite reading into the earlier days of the
   gap. Fixed: plain `.ffill()` (no `.interpolate()` call at all) — the
   only way to guarantee a value is built from past-or-present
   readings only.

2. Every grouped rolling/shift operation here grouped by `grid_id` alone.
   The data is Jun-Sep only (fire season), so within one grid cell's rows,
   day-ordered-by-date, September of year N is immediately followed by
   June of year N+1 — there's no gap in the ROW sequence even though
   there's an 8-month gap in actual TIME. `shift`/`rolling` operate on row
   position, not elapsed time, so without also grouping by year: NDVI/fire
   history windows at the start of a season quietly pull in late-prior-
   season data, and (more seriously) `fire_tomorrow` on each year's last
   day (Sep 30, except the final year) was being set to the FOLLOWING
   YEAR'S June 1 fire status instead of being dropped as undefined. Fixed:
   every grouping below is (grid_id, year), not grid_id alone.
"""

import numpy as np
import pandas as pd


def _season_groups(df):
    """(grid_id, year) — the grouping key for every per-cell-per-season
    operation below. Plain grid_id alone is wrong: see the file docstring,
    correction item 2."""
    return [df["grid_id"], df["date"].dt.year]


def assign_block_id(df, grid, spatial_block_deg):
    """Tags every grid_id with a coarse lat/lon spatial block, for the
    spatial-block CV split (docs/leakage_audit.md #4) — fires cluster
    geographically, so a later split keeps a whole neighborhood on one
    side of train/val instead of leaking adjacent cells across both."""
    grid = grid.copy()
    grid["block_id"] = (
        (grid["centroid_lat"] // spatial_block_deg).astype(int).astype(str)
        + "_"
        + (grid["centroid_lon"] // spatial_block_deg).astype(int).astype(str)
    )
    return df.merge(grid[["grid_id", "block_id"]], on="grid_id", how="left")


def filter_valid_cells(df, nasa_col="PRECTOTCORR", nlcd_col="pct_11"):
    """Keeps only grid cells with at least one non-null NASA weather
    reading and at least one non-null NLCD land-cover reading."""
    has_nasa = df.groupby("grid_id")[nasa_col].apply(lambda s: s.notna().any())
    has_nlcd = df.groupby("grid_id")[nlcd_col].apply(lambda s: s.notna().any())
    valid_ids = has_nasa[has_nasa].index.intersection(has_nlcd[has_nlcd].index)
    return df[df["grid_id"].isin(valid_ids)]


def interpolate_ndvi(df, col="ndvi"):
    """Forward-fill only, per (grid cell, year) — never interpolate using
    a future reading. REGRESSION GUARD (docs/leakage_audit.md #3, and the
    2026-11-25 correction in this file's docstring): plain `.ffill()`
    carries the last KNOWN-AS-OF-TODAY value forward; it cannot use a
    later reading, unlike `.interpolate("linear")`, which always uses the
    endpoints on BOTH sides of an interior gap regardless of
    `limit_direction`. A cell/season whose very first reading is missing
    stays NaN (and is dropped downstream) rather than being filled from
    the future or from a different season."""
    df = df.sort_values(["grid_id", "date"])
    df[col] = df.groupby(_season_groups(df))[col].transform(lambda x: x.ffill())
    return df


def add_rolling_feature(df, col, window, agg, new_col, shift=0):
    """Per-(grid-cell, year) rolling aggregate — grouping by year too
    stops a window from reaching back into the previous fire season
    (docs/leakage_audit.md section 6, correction item 2). `shift=1`
    excludes the current row from its own window (needed for
    fire-history/NDVI features below, not for same-day-known weather
    smoothing like T2M_3d_mean)."""
    df[new_col] = df.groupby(_season_groups(df))[col].transform(
        lambda s: getattr(s.shift(shift).rolling(window), agg)()
    )
    return df


def add_fire_history_features(df, target_col="fire_today"):
    """fires_last_7d/14d. REGRESSION GUARD (docs/leakage_audit.md #2):
    pandas' `.rolling(window)` is inclusive of the current row by default,
    so without `shift(1)` first, `fires_last_7d` on day t would include
    `fire_today` on day t itself — directly encoding the label into a
    "historical" feature. Shifting first means "last N days" excludes
    today. Grouped by (grid_id, year): see section 6 correction — without
    the year, early-June rolling windows would reach back into the
    previous September."""
    for window in (7, 14):
        col = f"fires_last_{window}d"
        df[col] = df.groupby(_season_groups(df))[target_col].transform(
            lambda s, w=window: s.shift(1).rolling(w).sum()
        )
        df[col] = df[col].fillna(0)
    return df


def add_ndvi_rolling_mean(df, col="ndvi", window=7, new_col="ndvi_7d_mean"):
    """shift(1) before rolling, same inclusivity reasoning as the fire
    history features above, applied to NDVI. Grouped by (grid_id, year)
    for the same cross-season reason."""
    df[new_col] = df.groupby(_season_groups(df))[col].transform(
        lambda s: s.shift(1).rolling(window).mean()
    )
    return df


def build_fire_tomorrow_label(df, target_col="fire_today", label_col="fire_tomorrow"):
    """REGRESSION GUARD (docs/leakage_audit.md #1): the project predicts
    NEXT-day ignition, but fire_today/weather/NDVI/fire-history features
    are all keyed to the same date. Shifting fire_today back one day per
    grid cell makes the label "did a fire start the day AFTER this row's
    date" while every feature stays keyed to this row's date — i.e. only
    information known by the end of day t predicts day t+1.

    Grouped by (grid_id, YEAR), not grid_id alone (docs/leakage_audit.md
    section 6, correction item 2): each cell's LAST DAY OF EACH SEASON has
    no real "tomorrow" (Sep 30 is not followed by Oct 1 in this Jun-Sep-
    only dataset) and must be dropped — grouping by grid_id alone was
    instead shifting in the FOLLOWING YEAR'S June 1 fire status as if it
    were "tomorrow," a real label error on about 79k rows."""
    df = df.sort_values(["grid_id", "date"])
    df[label_col] = df.groupby(_season_groups(df))[target_col].shift(-1)
    df = df[df[label_col].notna()]
    df[label_col] = df[label_col].astype(int)
    return df
