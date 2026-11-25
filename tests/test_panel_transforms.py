"""
test_panel_transforms.py

Regression tests guarding the four leakage fixes documented in
docs/leakage_audit.md. Each fix was found by reading the code carefully
once; nothing stops a future edit from silently reintroducing the exact
same bug, which is the problem this file exists to solve (see
docs/results.md's "Still open" item #7). Runs on tiny synthetic
DataFrames, no real data needed — fast enough for CI on every push.

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from panel_transforms import (  # noqa: E402
    add_fire_history_features,
    add_ndvi_rolling_mean,
    add_rolling_feature,
    assign_block_id,
    build_fire_tomorrow_label,
    filter_valid_cells,
    interpolate_ndvi,
)


def make_cell(grid_id, start_date, fire_days, n_days=20, ndvi_start=None):
    """One grid cell's worth of n_days consecutive daily rows."""
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    fire_today = np.zeros(n_days, dtype=int)
    for d in fire_days:
        fire_today[d] = 1
    ndvi = np.linspace(0.2, 0.8, n_days) if ndvi_start is None else ndvi_start
    return pd.DataFrame({
        "grid_id": grid_id, "date": dates, "fire_today": fire_today,
        "ndvi": ndvi, "PRECTOTCORR": 1.0,
    })


# ---------------------------------------------------------------------
# Leakage fix #1: fire_tomorrow = fire_today shifted back one day per cell
# ---------------------------------------------------------------------

def test_fire_tomorrow_is_next_days_fire_today():
    df = make_cell("A", "2023-01-01", fire_days=[5])
    out = build_fire_tomorrow_label(df.copy())
    # Day 4 (the day BEFORE the fire) should have fire_tomorrow == 1.
    row = out[out["date"] == pd.Timestamp("2023-01-01") + pd.Timedelta(days=4)]
    assert row["fire_tomorrow"].iloc[0] == 1
    # The fire day itself should have fire_tomorrow == 0 (no fire the day after).
    row = out[out["date"] == pd.Timestamp("2023-01-01") + pd.Timedelta(days=5)]
    assert row["fire_tomorrow"].iloc[0] == 0


def test_fire_tomorrow_drops_last_day_per_cell():
    df = make_cell("A", "2023-01-01", fire_days=[])
    out = build_fire_tomorrow_label(df.copy())
    # The last date per cell has no "tomorrow" and must be dropped, not
    # silently filled with 0/NaN-as-0 (that would fabricate a label).
    assert out["date"].max() < df["date"].max()
    assert len(out) == len(df) - 1


def test_fire_tomorrow_does_not_leak_same_day_label():
    """REGRESSION GUARD for docs/leakage_audit.md #1: the label must not
    equal the row's own fire_today (that would be predicting 'today', not
    'tomorrow')."""
    df = make_cell("A", "2023-01-01", fire_days=[3, 4, 10])
    out = build_fire_tomorrow_label(df.copy())
    same_day_leak = (out["fire_tomorrow"] == out["fire_today"]).all()
    assert not same_day_leak, "fire_tomorrow must not just echo fire_today"


# ---------------------------------------------------------------------
# Leakage fix #2: fires_last_7d/14d must exclude the current day
# ---------------------------------------------------------------------

def test_fires_last_7d_excludes_today():
    """REGRESSION GUARD for docs/leakage_audit.md #2: a fire TODAY must not
    make fires_last_7d >= 1 on that same day — that would encode the label
    into its own 'historical' feature."""
    df = make_cell("A", "2023-01-01", fire_days=[10])
    out = add_fire_history_features(df.copy())
    fire_day_row = out[out["date"] == pd.Timestamp("2023-01-01") + pd.Timedelta(days=10)]
    assert fire_day_row["fires_last_7d"].iloc[0] == 0
    assert fire_day_row["fires_last_14d"].iloc[0] == 0
    # But the day AFTER the fire should count it.
    next_day_row = out[out["date"] == pd.Timestamp("2023-01-01") + pd.Timedelta(days=11)]
    assert next_day_row["fires_last_7d"].iloc[0] == 1


def test_fires_last_7d_window_is_exactly_seven_prior_days():
    df = make_cell("A", "2023-01-01", fire_days=[0])  # fire on the first day
    out = add_fire_history_features(df.copy())
    # 7 days later (day index 7): still within the 7-day lookback (days 0-6
    # relative to day 7 means day 0 is 7 days prior - at the edge).
    row_day7 = out[out["date"] == pd.Timestamp("2023-01-01") + pd.Timedelta(days=7)]
    row_day8 = out[out["date"] == pd.Timestamp("2023-01-01") + pd.Timedelta(days=8)]
    assert row_day7["fires_last_7d"].iloc[0] == 1
    assert row_day8["fires_last_7d"].iloc[0] == 0  # fire has rolled out of the window


def test_ndvi_7d_mean_excludes_today():
    """Same inclusivity issue as fire history, applied to NDVI."""
    df = make_cell("A", "2023-01-01", fire_days=[], ndvi_start=np.array(
        [0.1] * 10 + [0.9] + [0.1] * 9
    ))
    out = add_ndvi_rolling_mean(df.copy())
    spike_day = out[out["date"] == pd.Timestamp("2023-01-01") + pd.Timedelta(days=10)]
    # The spike day's own 0.9 must not pull its own ndvi_7d_mean toward 0.9.
    assert spike_day["ndvi_7d_mean"].iloc[0] < 0.2


# ---------------------------------------------------------------------
# Leakage fix #3: NDVI interpolation must never use future data (no bfill)
# ---------------------------------------------------------------------

def test_ndvi_interpolation_never_backfills_leading_gap():
    """REGRESSION GUARD for docs/leakage_audit.md #3: if a cell's first N
    readings are missing, they must stay NaN, not get filled from a later
    (future) reading."""
    ndvi = np.array([np.nan, np.nan, np.nan, 0.5, 0.6, 0.7] + [0.7] * 14)
    df = make_cell("A", "2023-01-01", fire_days=[], ndvi_start=ndvi)
    out = interpolate_ndvi(df.copy())
    assert out["ndvi"].iloc[:3].isna().all(), "leading gap must stay NaN, not be backfilled"
    assert out["ndvi"].iloc[3] == 0.5


def test_ndvi_interpolation_fills_interior_gaps_correctly():
    """An interior gap (surrounded by known values on both sides) is fine
    to interpolate — that's not a leak, both endpoints are already known
    by the time the gap would be filled in a real deployment."""
    ndvi = np.array([0.2, 0.4, np.nan, np.nan, 1.0] + [1.0] * 15)
    df = make_cell("A", "2023-01-01", fire_days=[], ndvi_start=ndvi)
    out = interpolate_ndvi(df.copy())
    assert np.isclose(out["ndvi"].iloc[2], 0.6, atol=1e-6)
    assert np.isclose(out["ndvi"].iloc[3], 0.8, atol=1e-6)


def test_ndvi_interpolation_is_per_grid_cell():
    """Interpolation must not leak across grid cells — cell B's NDVI
    values must never fill cell A's gaps."""
    a = make_cell("A", "2023-01-01", fire_days=[], ndvi_start=np.array([np.nan] * 5 + [0.5] * 15))
    b = make_cell("B", "2023-01-01", fire_days=[], ndvi_start=np.full(20, 0.1))
    out = interpolate_ndvi(pd.concat([a, b], ignore_index=True))
    cell_a = out[out["grid_id"] == "A"]
    assert cell_a["ndvi"].iloc[:5].isna().all()


# ---------------------------------------------------------------------
# Leakage fix #4: spatial-block id groups geographically adjacent cells
# ---------------------------------------------------------------------

def test_spatial_block_id_groups_nearby_cells_together():
    grid = pd.DataFrame({
        "grid_id": ["A", "B", "C"],
        "centroid_lat": [34.01, 34.02, 36.50],
        "centroid_lon": [-118.01, -118.02, -120.50],
    })
    df = pd.DataFrame({"grid_id": ["A", "B", "C"]})
    out = assign_block_id(df, grid, spatial_block_deg=0.5)
    blocks = dict(zip(out["grid_id"], out["block_id"]))
    assert blocks["A"] == blocks["B"], "cells 0.01 deg apart must share a block"
    assert blocks["A"] != blocks["C"], "cells 2.5 deg apart must NOT share a block"


# ---------------------------------------------------------------------
# Sanity: filter_valid_cells keeps only cells with real coverage
# ---------------------------------------------------------------------

def test_filter_valid_cells_drops_cells_with_no_coverage():
    good = make_cell("A", "2023-01-01", fire_days=[])
    good["pct_11"] = 10.0
    bad = make_cell("B", "2023-01-01", fire_days=[])
    bad["PRECTOTCORR"] = np.nan  # no NASA coverage at all
    bad["pct_11"] = 10.0
    df = pd.concat([good, bad], ignore_index=True)
    out = filter_valid_cells(df)
    assert set(out["grid_id"].unique()) == {"A"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
