"""
download_ndvi.py

Downloads NOAA VIIRS NDVI CDR (v001, JP113C1, NOAA-20) for California fire
season (Jun-Sep) 2020-2023, clipped to the CA bounding box, and writes one
tidy CSV to data/interim/ndvi.csv (date, longitude, latitude, ndvi) — the
input src/combine_data.py expects.

Rewritten from src/ndvi_pipeline.py's full-file-download approach: each
daily NetCDF granule is ~64 MB globally, and 4 years of fire season would be
~30 GB to fetch and store just to keep a small CA-sized slice of it. Instead
this opens each file remotely over HTTP (the NCEI server supports byte-range
requests) via fsspec + the h5netcdf engine, and only pulls the bytes needed
for the already-sliced CA subset — nothing is written to disk except the
final tidy CSV.

Usage:
    python scripts/download_ndvi.py --years 2020 2021 2022 2023
    python scripts/download_ndvi.py --years 2023 --workers 6
"""

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import requests
import xarray as xr

INDEX_URL_TMPL = (
    "https://www.ncei.noaa.gov/data/"
    "land-normalized-difference-vegetation-index/access/{year}/"
)
VIIRS_PATTERN = re.compile(r"VIIRS-Land_v001_JP113C1_NOAA-20_(\d{8})_[^\"']*\.nc")

CA_BBOX_LONLAT = (-124.48, 32.53, -114.13, 42.01)  # lon_min, lat_min, lon_max, lat_max
SEASON_START_MD, SEASON_END_MD = (6, 1), (9, 30)  # Jun-Sep, matches NASA POWER season scope

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "interim" / "ndvi.csv"


def list_year_urls(year):
    idx_url = INDEX_URL_TMPL.format(year=year)
    html = requests.get(idx_url, timeout=60).text
    season_start = date(year, *SEASON_START_MD)
    season_end = date(year, *SEASON_END_MD)
    urls = {}
    for m in VIIRS_PATTERN.finditer(html):
        fname = m.group(0)
        file_date = datetime.strptime(m.group(1), "%Y%m%d").date()
        if season_start <= file_date <= season_end:
            urls[file_date] = idx_url + fname
    return [urls[d] for d in sorted(urls)]


def fetch_day(url, retries=3):
    lon_min, lat_min, lon_max, lat_max = CA_BBOX_LONLAT
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with fsspec.open(url, mode="rb") as f:
                ds = xr.open_dataset(f, engine="h5netcdf", decode_cf=False)
                da = ds["NDVI"]
                # latitude is stored descending (90 -> -90); slice accordingly
                sub = da.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
                scale = sub.attrs.get("scale_factor", 1.0)
                offset = sub.attrs.get("add_offset", 0.0)
                fill = sub.attrs.get("_FillValue")
                arr = sub.values.astype("float64").squeeze(axis=0)
                if fill is not None:
                    arr = np.where(arr == fill, np.nan, arr)
                arr = arr * scale + offset

                lats = sub.coords["latitude"].values
                lons = sub.coords["longitude"].values
                date_str = ds.attrs.get("time_coverage_start", "")[:10]

                lon_grid, lat_grid = np.meshgrid(lons, lats)
                df = pd.DataFrame(
                    {
                        "date": date_str,
                        "longitude": lon_grid.ravel(),
                        "latitude": lat_grid.ravel(),
                        "ndvi": arr.ravel(),
                    }
                )
                df = df.dropna(subset=["ndvi"])
                return df
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(3 * attempt)
    raise RuntimeError(f"failed {url}: {last_err}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    all_urls = []
    for year in args.years:
        urls = list_year_urls(year)
        print(f"[{year}] {len(urls)} VIIRS files in season window")
        all_urls.extend(urls)

    parts = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_day, u): u for u in all_urls}
        done = 0
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                df = fut.result()
                parts.append(df)
                status = f"ok ({len(df)} rows)"
            except Exception as e:  # noqa: BLE001
                status = f"FAILED: {e}"
            done += 1
            print(f"({done}/{len(all_urls)}) {url.rsplit('/', 1)[-1]}: {status}")

    full = pd.concat(parts, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(full):,} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
