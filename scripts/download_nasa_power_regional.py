"""
download_nasa_power_regional.py

Downloads NASA POWER daily weather (T2M, RH2M, WS10M, PRECTOTCORR, PS) for
California fire season (Jun-Sep) 2020-2023, using the POWER "regional" API
(https://power.larc.nasa.gov/docs/services/api/temporal/regional/), which
returns a 0.5-degree grid of points per request rather than one point at a
time. This is the same endpoint/variable set as the team's
src/nasa_power/nasa_power_yearly_data_fetch.bash, rewritten in Python with
the output reshaped directly into the wide per-(lat,lon,date) table
combine_data.py expects (LAT, LON, DATE, T2M, RH2M, WS10M, PRECTOTCORR, PS).

Note the filename: this repo's filesystem is case-insensitive, and the solo
repo's existing scripts/download_NASA_POWER.py (point-based downloader) is a
different file that this must not collide with.

The API accepts only one parameter per request, so each (variable, tile,
year) combination is a separate request; a tile is capped at 10 degrees of
longitude, so CA is split into two.

Usage:
    python scripts/download_nasa_power_regional.py
    python scripts/download_nasa_power_regional.py --years 2020 2021 2022 2023
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/regional"
VARIABLES = ["T2M", "RH2M", "WS10M", "PRECTOTCORR", "PS"]
TILES = [(-124.5, -119.5), (-119.5, -114.1)]  # (lon_min, lon_max)
LAT_MIN, LAT_MAX = 32.5, 42.0
SEASON_START_MD, SEASON_END_MD = "0601", "0930"  # Jun-Sep fire season

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "interim" / "nasa.csv"


def fetch_one(var, lon_min, lon_max, year, retries=3):
    params = {
        "parameters": var,
        "community": "AG",
        "longitude-min": lon_min,
        "longitude-max": lon_max,
        "latitude-min": LAT_MIN,
        "latitude-max": LAT_MAX,
        "start": f"{year}{SEASON_START_MD}",
        "end": f"{year}{SEASON_END_MD}",
        "format": "json",
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(POWER_URL, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            if "features" not in data:
                raise RuntimeError(f"no features in response: {data.get('messages')}")
            rows = []
            for feat in data["features"]:
                lon, lat = feat["geometry"]["coordinates"][:2]
                for date_str, value in feat["properties"]["parameter"][var].items():
                    rows.append((lat, lon, date_str, value))
            return pd.DataFrame(rows, columns=["LAT", "LON", "DATE", var])
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(3 * attempt)
    raise RuntimeError(f"failed {var} tile({lon_min},{lon_max}) {year}: {last_err}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023])
    args = ap.parse_args()

    all_parts = []
    for year in args.years:
        for lon_min, lon_max in TILES:
            tile_frames = []
            for var in VARIABLES:
                print(f"[{year}] tile({lon_min},{lon_max}) {var} ...")
                tile_frames.append(fetch_one(var, lon_min, lon_max, year))
                time.sleep(0.5)
            tile_df = tile_frames[0]
            for extra in tile_frames[1:]:
                tile_df = tile_df.merge(extra, on=["LAT", "LON", "DATE"], how="outer")
            all_parts.append(tile_df)

    df = pd.concat(all_parts, ignore_index=True)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
