"""
panel_transforms.py

The core per-grid-cell transforms used by clean_panel.py, pulled out into
testable functions. This exists so the fixes documented in
docs/leakage_audit.md have a regression test guarding them
(tests/test_panel_transforms.py) — before this, the logic lived as
top-level script code in clean_panel.py, which meant a future edit could
silently reintroduce one of the fixed leakage bugs with nothing to catch
it (see docs/results.md's "Still open" item #7). clean_panel.py imports
and calls these in the same order as before; this is a refactor, not a
behavior change.
"""

import numpy as np
import pandas as pd


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
    """Forward-only interpolation/fill per grid cell — never backfill.
    REGRESSION GUARD (docs/leakage_audit.md #3): `.bfill()` would leak a
    future satellite reading into an earlier day's "known as of day t"
    feature. A cell whose very first reading is missing stays NaN (and is
    dropped downstream) rather than being filled from the future."""
    df = df.sort_values(["grid_id", "date"])
    df[col] = df.groupby("grid_id")[col].transform(
        lambda x: x.interpolate("linear", limit_direction="forward").ffill()
    )
    return df


def add_rolling_feature(df, col, window, agg, new_col, shift=0):
    """Per-grid-cell rolling aggregate. `shift=1` excludes the current row
    from its own window (needed for fire-history/NDVI features below, not
    for same-day-known weather smoothing like T2M_3d_mean)."""
    df[new_col] = df.groupby("grid_id")[col].transform(
        lambda s: getattr(s.shift(shift).rolling(window), agg)()
    )
    return df


def add_fire_history_features(df, target_col="fire_today"):
    """fires_last_7d/14d. REGRESSION GUARD (docs/leakage_audit.md #2):
    pandas' `.rolling(window)` is inclusive of the current row by default,
    so without `shift(1)` first, `fires_last_7d` on day t would include
    `fire_today` on day t itself — directly encoding the label into a
    "historical" feature. Shifting first means "last N days" excludes
    today."""
    for window in (7, 14):
        col = f"fires_last_{window}d"
        df[col] = df.groupby("grid_id")[target_col].transform(
            lambda s, w=window: s.shift(1).rolling(w).sum()
        )
        df[col] = df[col].fillna(0)
    return df


def add_ndvi_rolling_mean(df, col="ndvi", window=7, new_col="ndvi_7d_mean"):
    """shift(1) before rolling, same inclusivity reasoning as the fire
    history features above, applied to NDVI."""
    df[new_col] = df.groupby("grid_id")[col].transform(
        lambda s: s.shift(1).rolling(window).mean()
    )
    return df


def build_fire_tomorrow_label(df, target_col="fire_today", label_col="fire_tomorrow"):
    """REGRESSION GUARD (docs/leakage_audit.md #1): the project predicts
    NEXT-day ignition, but fire_today/weather/NDVI/fire-history features
    are all keyed to the same date. Shifting fire_today back one day per
    grid cell makes the label "did a fire start the day AFTER this row's
    date" while every feature stays keyed to this row's date — i.e. only
    information known by the end of day t predicts day t+1. The last date
    per grid cell has no "tomorrow" to look up and is dropped."""
    df = df.sort_values(["grid_id", "date"])
    df[label_col] = df.groupby("grid_id")[target_col].shift(-1)
    df = df[df[label_col].notna()]
    df[label_col] = df[label_col].astype(int)
    return df
