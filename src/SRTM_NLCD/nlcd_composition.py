# requirements: pip install rasterio pyproj pandas numpy
"""
nlcd_composition.py

Converts the per-year Annual NLCD tiles fetched by
scripts/download_annual_nlcd.py (data/raw/nlcd_annual/<year>/tile_*.tif) into
one land-cover-composition CSV per year: a 1 km grid over California, with a
pct_<class> column per NLCD class giving the percent of that class within
each 1 km cell. Downstream, src/combine_data.py nearest-neighbor-joins these
onto the 5 km project grid per year, so land cover now varies by year instead
of being frozen to a single static vintage.

Usage:
    python src/SRTM_NLCD/nlcd_composition.py --years 2020 2021 2022 2023
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_origin, xy
from pyproj import Transformer

TILE_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "nlcd_annual"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

# California bounding box in lon/lat (WGS84): (min_lon, min_lat, max_lon, max_lat)
CA_BBOX_LONLAT = (-124.48, 32.53, -114.13, 42.01)

# Equal-area CRS for CONUS; set a local CRS if doing AK/HI/PR
DST_CRS = "EPSG:5070"
DST_RES_METERS = 1000  # 1 km

RENORMALIZE_ROW_SUM_TO_100 = False


def mosaic_year(year):
    """Merge all downloaded tiles for one year into a single array + transform."""
    tile_dir = TILE_ROOT / str(year)
    tile_paths = sorted(tile_dir.glob("tile_*.tif"))
    if not tile_paths:
        raise FileNotFoundError(
            f"No tiles found in {tile_dir} — run scripts/download_annual_nlcd.py first."
        )
    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic_arr, mosaic_transform = merge(srcs, nodata=250)
        src_crs = srcs[0].crs
        src_nodata = srcs[0].nodata
    finally:
        for s in srcs:
            s.close()
    return mosaic_arr[0], mosaic_transform, src_crs, src_nodata


def composition_for_year(year):
    src_arr, src_transform, src_crs, src_nodata = mosaic_year(year)

    unique_vals = np.unique(src_arr)
    if src_nodata is not None:
        unique_vals = unique_vals[unique_vals != src_nodata]
    codes_present = unique_vals.astype(int).tolist()

    # Build the 1 km target grid from the CA bbox projected to DST_CRS
    bbox_proj = transform_bounds("EPSG:4326", DST_CRS, *CA_BBOX_LONLAT, densify_pts=21)
    xmin, ymin, xmax, ymax = bbox_proj
    dst_width = int(math.ceil((xmax - xmin) / DST_RES_METERS))
    dst_height = int(math.ceil((ymax - ymin) / DST_RES_METERS))
    dst_transform = from_origin(xmin, ymax, DST_RES_METERS, DST_RES_METERS)

    # For each class, resample a 0/1 mask with average -> fraction per 1 km cell
    class_to_fraction = {}
    for code in codes_present:
        bin_src = (src_arr == code).astype("float32")
        dst_frac = np.zeros((dst_height, dst_width), dtype="float32")
        reproject(
            source=bin_src,
            destination=dst_frac,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=DST_CRS,
            resampling=Resampling.average,
            dst_nodata=0.0,
        )
        class_to_fraction[code] = dst_frac

    codes = sorted(class_to_fraction.keys())
    stack = np.stack([class_to_fraction[c] for c in codes], axis=0)  # (K, H, W)
    sum_frac = stack.sum(axis=0)

    if RENORMALIZE_ROW_SUM_TO_100:
        safe_sum = np.where(sum_frac > 0, sum_frac, 1.0)
        stack = stack / safe_sum

    stack_pct = 100.0 * stack
    # sum_frac is near-but-not-exactly 0 (float resampling noise) for cells
    # that are entirely outside the mosaic (ocean, out-of-bbox) — use a small
    # epsilon rather than > 0 so those aren't kept as bogus near-zero rows.
    valid_mask = sum_frac > 1e-6

    rows, cols = np.indices((dst_height, dst_width))
    xs, ys = xy(dst_transform, rows, cols, offset="center")
    xs = np.array(xs).flatten()
    ys = np.array(ys).flatten()

    to_lonlat = Transformer.from_crs(DST_CRS, "EPSG:4326", always_xy=True)
    lon, lat = to_lonlat.transform(xs, ys)

    df = pd.DataFrame({"longitude": lon, "latitude": lat, "year": year})
    for i, code in enumerate(codes):
        df[f"pct_{int(code)}"] = stack_pct[i].flatten()

    df = df[valid_mask.flatten()].reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        df = composition_for_year(year)
        out_path = OUT_DIR / f"nlcd_1km_CA_composition_{year}.csv"
        df.to_csv(out_path, index=False)
        class_cols = [c for c in df.columns if c.startswith("pct_")]
        print(f"[{year}] wrote {len(df):,} rows, {len(class_cols)} class columns -> {out_path}")


if __name__ == "__main__":
    main()
