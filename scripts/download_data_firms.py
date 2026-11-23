# Step 1 — FIRMS Fire Data
import requests
import pandas as pd

MAP_KEY = "aa031feb165377ba318bd9cdb132bfce"
SOURCE = "VIIRS_SNPP_NRT"
LAT = [32, 42]      # min, max
LONG = [-125, -114] # min, max
# INTERVAL = 2
# DATE = "2025-07-01"

def download_firms(interval, date):
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{LONG[0]},{LAT[0]},{LONG[1]},{LAT[1]}/{interval}/{date}"
    df = pd.read_csv(url)
    print(df)
    out_path = f"data/raw/firms_{interval}_{date}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {out_path} with {len(df)} records")

if __name__ == "__main__":
    download_firms("2", "2025-07-01")
