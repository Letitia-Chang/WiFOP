"""
download_irwin_wfigs.py

Downloads California fire incident records for 2020-2023 (the label source
for fire_today/fire_tomorrow) from NIFC's public WFIGS_Incident_Locations
ArcGIS FeatureServer (https://data-nifc.opendata.arcgis.com/), which is
IRWIN-backed and requires no account, token, or NWCG affiliation — unlike
IRWIN's own submission API. Replaces the original src/fire_labels/IRWIN.py
flow (which expected a pre-exported irwin_2023-07.csv of unknown origin and
only covered one month) for the full 4-year range.

Paginates the FeatureServer's query endpoint (2000-record page limit) over
FireDiscoveryDateTime 2020-01-01..2023-12-31, POOState = 'US-CA', and writes
columns (date, lat, lon, incident_id) directly to data/interim/irwin.csv,
matching what src/combine_data.py expects.

Usage:
    python scripts/download_irwin_wfigs.py
"""

from pathlib import Path

import pandas as pd
import requests

QUERY_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Incident_Locations/FeatureServer/0/query"
)
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "interim" / "irwin.csv"
PAGE_SIZE = 2000
WHERE = (
    "POOState='US-CA' AND FireDiscoveryDateTime >= timestamp '2020-01-01 00:00:00' "
    "AND FireDiscoveryDateTime <= timestamp '2023-12-31 23:59:59'"
)
OUT_FIELDS = "IrwinID,FireDiscoveryDateTime,InitialLatitude,InitialLongitude"


def fetch_page(offset, retries=3):
    params = {
        "where": WHERE,
        "outFields": OUT_FIELDS,
        "f": "json",
        "resultOffset": offset,
        "resultRecordCount": PAGE_SIZE,
        "orderByFields": "OBJECTID",
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(QUERY_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(data["error"])
            return data["features"]
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"failed page at offset {offset}: {last_err}")


def main():
    rows = []
    offset = 0
    while True:
        features = fetch_page(offset)
        if not features:
            break
        for feat in features:
            p = feat["attributes"]
            rows.append(
                {
                    "incident_id": p.get("IrwinID"),
                    "fire_discovery_ms": p.get("FireDiscoveryDateTime"),
                    "lat": p.get("InitialLatitude"),
                    "lon": p.get("InitialLongitude"),
                }
            )
        print(f"offset {offset}: +{len(features)} rows (total {len(rows)})")
        offset += len(features)
        if len(features) < PAGE_SIZE:
            break

    CA_BBOX = (-124.48, 32.53, -114.13, 42.01)  # lon_min, lat_min, lon_max, lat_max

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["fire_discovery_ms", "lat", "lon"])
    df["date"] = pd.to_datetime(df["fire_discovery_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates(subset=["incident_id"])

    # A few records have (0,0) or otherwise bad coordinates despite the
    # POOState='US-CA' filter (bad source data) — drop anything outside the
    # CA bounding box rather than trusting POOState alone.
    lon_min, lat_min, lon_max, lat_max = CA_BBOX
    before = len(df)
    df = df[df["lat"].between(lat_min, lat_max) & df["lon"].between(lon_min, lon_max)]
    print(f"Dropped {before - len(df)} rows outside the CA bounding box")

    out = df[["date", "lat", "lon", "incident_id"]].copy()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out):,} CA fire incidents (2020-2023) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
