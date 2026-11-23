"""
download_annual_nlcd.py

Fetches USGS Annual NLCD Land Cover (Collection 1.2) for California, at native
30 m resolution, for a list of years, directly from the public GeoServer WMS
behind the MRLC Viewer (https://www.mrlc.gov/geoserver/wms). No account, API
key, or download order is required.

The requested extent is larger than a single WMS request can return, so the
California bounding box is tiled into fixed-size chunks (in EPSG:3857 meters)
and each tile is fetched as its own GeoTIFF with FORMAT=image/geotiff and a
TIME= parameter selecting the year. Output goes to data/raw/nlcd_annual/<year>/
(gitignored) as one GeoTIFF per tile; mosaicking/reprojecting onto the 5 km
grid happens downstream in the SRTM_NLCD feature-extraction step.

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

WMS_URL = "https://www.mrlc.gov/geoserver/wms"
LAYER = "mrlc_Land-Cover_conus_year_data:Land-Cover_conus_year_data"
NATIVE_RES_M = 30.0  # Annual NLCD native pixel size, meters

# California bounding box, EPSG:3857 (Web Mercator), covers lon -124.48..-114.13 / lat 32.53..42.01
CA_BBOX_3857 = (-13_857_050.0, 3_858_773.0, -12_704_893.0, 5_159_316.0)

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
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": LAYER,
        "styles": "",
        "format": "image/geotiff",
        "crs": "EPSG:3857",
        "bbox": f"{tx0},{ty0},{tx1},{ty1}",
        "width": width_px,
        "height": height_px,
        "time": f"{year}-01-01",
    }

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(WMS_URL, params=params, timeout=120)
            resp.raise_for_status()
            if resp.content[:4] not in (b"II*\x00", b"MM\x00*"):
                raise RuntimeError(f"non-TIFF response ({resp.content[:200]!r})")
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
        tiles = list(tile_grid(CA_BBOX_3857, args.tile_size, NATIVE_RES_M))
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
