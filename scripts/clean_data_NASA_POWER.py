# import pandas as pd

# df = pd.read_csv("POWER_Point_Daily_20240701_20240731_037d50N_0120d50W_LST.csv", skiprows=10)
# df = df.rename(columns=str.strip)  # remove spaces from headers
# df['DATE'] = pd.to_datetime(dict(year=df['YEAR'], month=df['MO'], day=df['DY']))

# print(df[['DATE', 'T2M', 'RH2M', 'WS2M', 'PRECTOTCORR']].head())

import os
import pandas as pd
from glob import glob

def combine_power_data(input_dir="data/weather", output_path="data/processed/weather_all.csv"):
    """
    Combine all NASA POWER .csv files in a directory into a single dataframe.
    Each file must correspond to one (lat, lon) point.
    """
    all_files = glob(os.path.join(input_dir, "POWER_*.csv"))
    if not all_files:
        print("⚠️ No files found in", input_dir)
        return

    records = []
    for file in all_files:
        # Extract lat/lon from filename
        fname = os.path.basename(file)
        try:
            parts = fname.split("_")
            lat = float(parts[1])
            lon = float(parts[2])
        except Exception:
            print(f"⚠️ Skipping malformed filename: {fname}")
            continue

        # Read the data (skip metadata rows)
        try:
            df = pd.read_csv(file, skiprows=12)
            df = df.rename(columns=str.strip)
            df["LAT"] = lat
            df["LON"] = lon
            # Construct datetime column
            df["DATE"] = pd.to_datetime(dict(year=df["YEAR"], month=df["MO"], day=df["DY"]))
            records.append(df[["DATE", "LAT", "LON", "T2M", "RH2M", "WS2M", "PRECTOTCORR"]])
        except Exception as e:
            print(f"⚠️ Error reading {fname}: {e}")

    # Combine everything into one DataFrame
    # all_data = pd.concat(records, ignore_index=True)
    # all_data.sort_values(["DATE", "LAT", "LON"], inplace=True)

    # Save merged dataset
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # all_data.to_csv(output_path, index=False)
    # print(f"✅ Saved merged file: {output_path} ({len(all_data)} rows)")
    # return all_data


# Example usage
if __name__ == "__main__":
    combine_power_data()
    # df_all = combine_power_data()
    # print(df_all.head())
