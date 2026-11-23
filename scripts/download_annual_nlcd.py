"""
download_annual_nlcd.py

Fetches USGS Annual NLCD Land Cover (Collection 1.2) for California, at native
30 m resolution, for a list of years, directly from the public GeoServer WCS
behind the MRLC Viewer (https://www.mrlc.gov/geoserver/wcs). No account, API
key, or download order is required.

Uses WCS 1.0.0 GetCoverage in the layer's native CRS (EPSG:5070), which
returns the actual NLCD class codes (11, 21, 22, ...). Two alternatives were
tried and rejected:
  - WMS GetMap (any format/style) returns a rendered image whose pixel values
    are palette/color-table indices, not the real class codes.
  - WCS 2.0.1 GetCoverage with `subset=time(...)` hits a server-side bug on
    this ImageMosaic layer (NoApplicableCode / "startTime is null").
WCS 1.0.0 with a plain `time=` parameter and `crs=EPSG:5070` (the coverage's
native CRS — requesting a different response CRS silently returns all
nodata) avoids both issues.

The requested extent is larger than a single WCS request can return, so the
California bounding box (projected to EPSG:5070) is tiled into fixed-size
chunks and each tile is fetched as its own GeoTIFF. Output goes to
data/raw/nlcd_annual/<year>/ (gitignored) as one GeoTIFF per tile;
mosaicking happens downstream in src/SRTM_NLCD/nlcd_composition.py.

Usage:
    python scripts/download_annual_nlcd.py --years 2020 2021 2022 2023
    python scripts/download_annual_nlcd.py --years 2020 --tile-size 4096 --workers 4
"""

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

WCS_URL = "https://www.mrlc.gov/geoserver/wcs"
COVERAGE = "mrlc_Land-Cover_conus_year_data:Land-Cover_conus_year_data"
NATIVE_RES_M = 30.0  # Annual NLCD native pixel size, meters
NATIVE_CRS = "EPSG:5070"  # must match the coverage's native CRS, or WCS 1.0.0 returns all-nodata

# California bounding box, EPSG:5070 (projected from lon -124.48..-114.13 / lat 32.53..42.01)
CA_BBOX_5070 = (-2_620_977.0, 1_446_559.0, -1_482_244.0, 2_253_850.0)

OUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw" / "nlcd_annual"


def tile_grid(bbox, tile_size_px, res_m):
    """Yield (col, row, tile_bbox) covering bbox with square tiles of tile_size_px pixels at res_m."""
    x0, y0, x1, y1 = bbox
    tile_span_m = tile_size_px * res_m
    n_cols = math.ceil((x1 - x0) / tile_span_m)
    n_rows = math.ceil((y1 - y0) / tile_span_m)
    for row in range(n_rows):
        ty0 = y0 + row * tile_span_m
        ty1 = min(ty0 + tile_span_m, y1)
        for col in range(n_cols):
            tx0 = x0 + col * tile_span_m
            tx1 = min(tx0 + tile_span_m, x1)
            yield col, row, (tx0, ty0, tx1, ty1)


def fetch_tile(year, col, row, tile_bbox, tile_size_px, res_m, out_dir, retries=3):
    tx0, ty0, tx1, ty1 = tile_bbox
    width_px = max(1, round((tx1 - tx0) / res_m))
    height_px = max(1, round((ty1 - ty0) / res_m))
    out_path = out_dir / f"tile_r{row:03d}_c{col:03d}.tif"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path, "skipped (exists)"

    params = {
        "service": "WCS",
        "version": "1.0.0",
        "request": "GetCoverage",
        "coverage": COVERAGE,
        "crs": NATIVE_CRS,
        "bbox": f"{tx0},{ty0},{tx1},{ty1}",
        "width": width_px,
        "height": height_px,
        "format": "GeoTIFF",
        "time": f"{year}-01-01T00:00:00.000Z",
    }

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(WCS_URL, params=params, timeout=120)
            resp.raise_for_status()
            if resp.content[:4] not in (b"II*\x00", b"MM\x00*"):
                raise RuntimeError(f"non-TIFF response ({resp.content[:300]!r})")
            out_path.write_bytes(resp.content)
            return out_path, f"ok ({len(resp.content)} bytes)"
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * attempt)
    return out_path, f"FAILED after {retries} attempts: {last_err}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--tile-size", type=int, default=4096, help="tile edge length in pixels (native 30m res)")
    ap.add_argument("--workers", type=int, default=4, help="concurrent requests")
    args = ap.parse_args()

    for year in args.years:
        out_dir = OUT_ROOT / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        tiles = list(tile_grid(CA_BBOX_5070, args.tile_size, NATIVE_RES_M))
        print(f"[{year}] {len(tiles)} tiles -> {out_dir}")

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(fetch_tile, year, col, row, tbbox, args.tile_size, NATIVE_RES_M, out_dir): (col, row)
                for col, row, tbbox in tiles
            }
            done = 0
            for fut in as_completed(futures):
                col, row = futures[fut]
                path, status = fut.result()
                done += 1
                print(f"[{year}] ({done}/{len(tiles)}) r{row:03d}_c{col:03d}: {status}")

    print("Done.")


if __name__ == "__main__":
    main()
