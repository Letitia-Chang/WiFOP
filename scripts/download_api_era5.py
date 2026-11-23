# Step 2. ERA5_LAND
import cdsapi

def download_era5(start_date, end_date):
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-land",
        {
            "variable": [
                "2m_temperature", "2m_dewpoint_temperature", "total_precipitation",
                "u_component_of_wind_10m", "v_component_of_wind_10m",
                "volumetric_soil_water_layer_1",
            ],
            "area": [42, -125, 32, -114],  # N,W,S,E (California)
            "format": "netcdf",
            "year": start_date[:4],
            "month": start_date[5:7],
            "day": start_date[8:10],
            "time": ["00:00", "06:00", "12:00", "18:00"],
        },
        f"data/raw/era5_{start_date}_{end_date}.nc"
    )
    print("✅ ERA5 data downloaded")

# Read output file
# import xarray as xr
# ds = xr.open_dataset("data/raw/era5_2024-07-01_2024-07-07.nc")
# print(ds)

if __name__ == "__main__":
    download_era5("2025-07-01", "2025-07-07")
