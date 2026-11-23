# src/make_grid.py
import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np

def make_grid(lat_min=32, lat_max=42, lon_min=-125, lon_max=-114, cell_size=0.1):
    """Make rectangular grid over California (~11 km cells for 0.1°)."""
    lon_grid = np.arange(lon_min, lon_max, cell_size)
    lat_grid = np.arange(lat_min, lat_max, cell_size)
    polygons, ids = [], []
    for i, lon in enumerate(lon_grid):
        for j, lat in enumerate(lat_grid):
            poly = Polygon([
                (lon, lat),
                (lon + cell_size, lat),
                (lon + cell_size, lat + cell_size),
                (lon, lat + cell_size)
            ])
            polygons.append(poly)
            ids.append(f"{i}_{j}")
    return gpd.GeoDataFrame({"cell_id": ids, "geometry": polygons}, crs="EPSG:4326")

if __name__ == "__main__":
    grid = make_grid()
    grid.to_file("data/processed/grid_california.geojson", driver="GeoJSON")
