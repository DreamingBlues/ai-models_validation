#!/usr/bin/env python3
# Abtin Olaee 2026

import json
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "input_json": "./sensorData/situ_data.json",
    "output_nc": "./sensorData/station_data.nc",
    
    # in UTC time
    "start_time": "2025-01-01 00:00:00",
    "end_time":   "2025-01-11 00:00:00",
    "freq":       "6h",
}


# ==============================================================================
# HELPERS
# ==============================================================================
def clean_var_name(var_name):
    """
    Removes common API suffixes to standardize names.
    Ex: 'air_temp_set_1' -> 'air_temp'
    """
    return var_name.replace("_set_1", "").replace("_set_2d", "")



# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # Step 0: Setup
    # Load json synoptic weather staion file
    print(f"Loading {CONFIG['input_json']}...")
    with open(CONFIG['input_json'], "r") as f:
        data = json.load(f)

    # Step 1: Create the time step templete following UTC
    master_time = pd.date_range(
        start=CONFIG["start_time"], 
        end=CONFIG["end_time"], 
        freq=CONFIG["freq"],
        tz="UTC" 
    )


    # Step 2: Iterate through every station, extract and align data
    station_ids, lats, lons, elevs, aligned_data = [], [], [], [], []
    all_observed_vars = set() # weather variables need to be unique

    # Get list of stations
    stations_list = data.get("STATION", [])
    if not stations_list:
        raise ValueError("JSON does not contain 'STATION' list")

    print("Aligning station data to 6h timesteps...")
    
    for stat in tqdm(stations_list, unit="station", desc="Extracting Station Data"):
        # ensure proper station ID
        sid = stat.get("STID")
        if not sid: continue 
        
        # ensure proper coordinate tracking
        try:
            lat = float(stat.get("LATITUDE", np.nan))
            lon = float(stat.get("LONGITUDE", np.nan))
            elev = float(stat.get("ELEVATION", np.nan))
        except (ValueError, TypeError):
            continue 

        # extract observation into pandas dataset
        obs = stat.get("OBSERVATIONS", {})
        if "date_time" not in obs:
            continue 
        df = pd.DataFrame(obs)
        
        # extract and standardize time
        df["date_time"] = pd.to_datetime(df["date_time"])
        df = df.set_index("date_time")

        # ensure every timestep is unique, prevent overlapping of two observations
        df = df[~df.index.duplicated(keep='first')]
        df_aligned = df.reindex(master_time, method='nearest', tolerance=pd.Timedelta("30min"))

        # catalog station's metadata
        station_ids.append(sid)
        lats.append(lat)
        lons.append(lon)
        elevs.append(elev)
        aligned_data.append(df_aligned)
        
        for col in df_aligned.columns:
            all_observed_vars.add(clean_var_name(col))


    # Step 3: Build the 2D Arrays (Variables)
    n_stations = len(station_ids)
    n_times = len(master_time)
    
    print(f"Constructing Xarray Dataset for {n_stations} stations by {n_times} timesteps...")
    
    # build Dataset:
    # Dimensions: 'station' (rows) and 'time' (columns).
    # Coordinates: Maps station index to physical metadata (Lat/Lon/Elev).
    # Data Variables: Each variable (e.g., air_temp) is a 2D matrix [station x time].
    ds = xr.Dataset(
        coords={
            "station": (["station"], station_ids),
            "time": (["time"], master_time),
            "latitude": (["station"], lats),
            "longitude": (["station"], lons),
            "elevation": (["station"], elevs),
        }
    )

    # list of variables to strictly exclude due to non-numeric values 
    BLACKLIST = {"QC_SUMMARY", "qc_summary", "weather_cond_code", "cloud_layer_1_code", "cloud_layer_2_code", "cloud_layer_3_code"}

    # iterates for each variable
    for var_name in tqdm(sorted(list(all_observed_vars)), desc="Building Variables"):
        if var_name in BLACKLIST:
            continue
            
        # construct matrix Dimensions: 'station' (rows) and 'time' (columns).
        matrix = np.full((n_stations, n_times), np.nan, dtype=np.float32)
        
        # iterate through each station
        for i, df in enumerate(aligned_data):
            # standardize variable name
            matching_cols = [c for c in df.columns if clean_var_name(c) == var_name]
            
            if matching_cols:
                col_data = df[matching_cols[0]].values
                try:
                    # extract data into numeric, then insert into appropriate location in matrix
                    numeric_data = pd.to_numeric(col_data, errors='coerce')
                    matrix[i, :] = numeric_data.astype(np.float32)
                except Exception:
                    # if fail, then leave as NaNs
                    continue
        
        #  ensures variable data is not all NaNs
        if np.any(~np.isnan(matrix)):
            ds[var_name] = (["station", "time"], matrix)

    # Step 4: Save to NetCDF
    print("Saving to NetCDF...")
    
    encoding = {v: {"zlib": True, "complevel": 1, "_FillValue": np.nan} for v in ds.data_vars}
    
    ds.to_netcdf(CONFIG["output_nc"], encoding=encoding, engine="h5netcdf")
    print(f"Success! Saved to {CONFIG['output_nc']}")

if __name__ == "__main__":
    main()