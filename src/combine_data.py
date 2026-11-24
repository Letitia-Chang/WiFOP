import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

# ============================================================
# 0. Load 5 km grid
# ============================================================

print("\n[1/10] Loading 5km grid...")
grid = pd.read_csv("../data/california_grid_5km.csv")

grid = grid.rename(columns={"centroid_lat": "lat", "centroid_lon": "lon"})
assert "lat" in grid and "lon" in grid, "Grid rename failed!"

# Build BallTree
print("[2/10] Building BallTree...")
tree = BallTree(np.radians(grid[["lat","lon"]].to_numpy()), metric='haversine')

def assign_grid_id(df, lat_col, lon_col):
    assert lat_col in df.columns, f"Missing {lat_col} in dataframe!"
    assert lon_col in df.columns, f"Missing {lon_col} in dataframe!"

    coords = np.radians(df[[lat_col, lon_col]].to_numpy())
    dist, idx = tree.query(coords, k=1)
    df["grid_id"] = grid.iloc[idx.flatten()].grid_id.values
    return df

# ============================================================
# 1. NASA
# ============================================================

print("\n[3/10] Loading NASA...")
nasa = pd.read_csv("../data/interim/nasa.csv")

# FORCE rename
nasa.rename(columns={"LAT": "latitude", "LON": "longitude", "DATE": "date"}, inplace=True)
assert "latitude" in nasa and "longitude" in nasa, "NASA rename failed!"

nasa["date"] = pd.to_datetime(nasa["date"])

# NASA POWER's native grid is ~0.5 degrees (~55 km) — far coarser than the
# 5 km project grid, so assign_grid_id (which snaps each NASA row onto its
# single nearest of the 53k fine grid cells) would only ever populate the
# handful of grid cells that happen to be nearest to one of the ~320 NASA
# points, leaving >99% of grid cells with no weather at all. Instead, do
# the join in the other direction: find each of the 53k grid cells'
# nearest NASA station, then merge NASA's daily values onto the grid via
# that station assignment.
print("Assigning each grid cell to its nearest NASA station...")
stations = nasa[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
stations["station_id"] = stations.index
station_tree = BallTree(np.radians(stations[["latitude", "longitude"]].to_numpy()), metric="haversine")
_, station_idx = station_tree.query(np.radians(grid[["lat", "lon"]].to_numpy()), k=1)
grid["station_id"] = station_idx.flatten()

nasa = nasa.merge(stations, on=["latitude", "longitude"], how="left")
nasa = nasa.drop(columns=["latitude","longitude"])
nasa_daily = nasa.groupby(["station_id","date"], as_index=False).mean()

# ============================================================
# 2. NDVI
# ============================================================

print("\n[4/10] Loading NDVI...")
ndvi = pd.read_csv("../data/interim/ndvi.csv")
assert "latitude" in ndvi and "longitude" in ndvi, "NDVI rename failed!"

ndvi["date"] = pd.to_datetime(ndvi["date"])

print("Assigning NDVI grid IDs...")
ndvi = assign_grid_id(ndvi, "latitude", "longitude")
ndvi = ndvi.drop(columns=["latitude","longitude"])
ndvi_daily = ndvi.groupby(["grid_id","date"], as_index=False).mean()

# ============================================================
# 3. IRWIN
# ============================================================

print("\n[5/10] Loading IRWIN...")
irwin = pd.read_csv("../data/interim/irwin.csv")
assert "lat" in irwin and "lon" in irwin, "IRWIN columns missing!"

irwin["date"] = pd.to_datetime(irwin["date"])

print("Assigning IRWIN grid IDs...")
irwin = assign_grid_id(irwin, "lat", "lon")
assert "grid_id" in irwin, "IRWIN grid_id assignment FAILED!"

irwin = irwin.drop(columns=["lat","lon","incident_id"])

fire_labels = (
    irwin.groupby(["grid_id","date"]).size().reset_index(name="fire_count")
)
fire_labels["fire_today"] = (fire_labels["fire_count"] > 0).astype(int)

# ============================================================
# 4. NLCD (Annual NLCD — one land-cover composition per year, not a single
#    static table, so pct_* features vary by year instead of being frozen)
# ============================================================

print("\n[6/10] Loading NLCD (per year)...")
NLCD_YEARS = [2020, 2021, 2022, 2023]
nlcd_parts = []
for year in NLCD_YEARS:
    part = pd.read_csv(f"../data/processed/nlcd_1km_CA_composition_{year}.csv")
    assert "latitude" in part and "longitude" in part, f"NLCD {year} columns missing!"
    assert "year" in part, f"NLCD {year} missing year column!"
    nlcd_parts.append(part)
nlcd = pd.concat(nlcd_parts, ignore_index=True)

print("Assigning NLCD grid IDs...")
nlcd = assign_grid_id(nlcd, "latitude", "longitude")
nlcd = nlcd.drop(columns=["latitude","longitude"])

nlcd_grid = nlcd.groupby(["grid_id","year"], as_index=False).mean()

# ============================================================
# 5. Build daily panel
# ============================================================

all_dates = nasa_daily["date"].sort_values().unique()
panel = pd.MultiIndex.from_product(
    [grid["grid_id"], all_dates],
    names=["grid_id","date"]
).to_frame(index=False)

# ============================================================
# 6. Merge
# ============================================================

print("\nMerging datasets...")
panel["year"] = panel["date"].dt.year
panel = panel.merge(grid[["grid_id","station_id"]], on="grid_id", how="left")
panel = panel.merge(nasa_daily, on=["station_id","date"], how="left")
panel = panel.drop(columns=["station_id"])
panel = panel.merge(ndvi_daily, on=["grid_id","date"], how="left")
panel = panel.merge(fire_labels[["grid_id","date","fire_today"]], on=["grid_id","date"], how="left")
panel = panel.merge(nlcd_grid, on=["grid_id","year"], how="left")
panel = panel.drop(columns=["year"])

panel["fire_today"] = panel["fire_today"].fillna(0).astype(int)

# ============================================================
# 7. Save
# ============================================================

print("\nSaving...")
panel.to_csv("../data/processed/california_5km_daily_panel_2020-2023.csv", index=False)

print("\n🎉 DONE — merged panel built successfully!")
print(panel.head())
