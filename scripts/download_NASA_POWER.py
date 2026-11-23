# API: NASA POWER (Weather)
# Spatial resolution: ≈ 0.5° (~50 km)
# Temporal resolution: Daily

import os
import requests
import pandas as pd
from time import sleep

def download_power_grid(lat_min=32, lat_max=42, lon_min=-125, lon_max=-114, 
                        step=1.0, start="2025-07-01", end="2025-07-31",
                        out_dir="data/weather"):
    """
    Download NASA POWER daily weather data (T2M, RH2M, WS2M, PRECTOTCORR)
    for a grid of latitude/longitude points.
    """

    os.makedirs(out_dir, exist_ok=True)
    lats = [round(x, 2) for x in list(frange(lat_min, lat_max, step))]
    lons = [round(x, 2) for x in list(frange(lon_min, lon_max, step))]

    print(f"🌍 Fetching POWER data for {len(lats)*len(lons)} grid points...")

    for lat in lats:
        for lon in lons:
            filename = f"POWER_{lat}_{lon}_{start}_{end}.csv"
            out_path = os.path.join(out_dir, filename)
            if os.path.exists(out_path):
                print(f"⏭️ Skipping {filename} (already exists)")
                continue

            # Build NASA POWER API URL
            url = (
                "https://power.larc.nasa.gov/api/temporal/daily/point?"
                f"start={start.replace('-', '')}&end={end.replace('-', '')}"
                f"&latitude={lat}&longitude={lon}"
                "&parameters=T2M,RH2M,WS2M,PRECTOTCORR"
                "&community=AG&format=CSV"
            )

            try:
                print(f"⬇️ Downloading {filename} ...")
                r = requests.get(url)
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    f.write(r.content)
                sleep(1)  # rate limit
            except Exception as e:
                print(f"⚠️ Failed for ({lat}, {lon}): {e}")

    print("✅ All downloads complete!")

def frange(start, stop, step):
    """Range function for floats."""
    x = start
    while x <= stop:
        yield x
        x += step

# Example usage
if __name__ == "__main__":
    download_power_grid(
        lat_min=32, lat_max=42,
        lon_min=-125, lon_max=-114,
        step=1.0,                # 1° grid spacing (≈ 100 km)
        start="2024-07-01",
        end="2024-07-31"
    )
