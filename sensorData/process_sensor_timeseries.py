#!/usr/bin/env python3
# Abtin Olaee 2025

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
    "input_json": "/shome/u014930890/PGE_Projects/sensorData/timeseries",
    "output_nc": "/shome/u014930890/PGE_Projects/sensorData/station_data.nc",
    
    # Define the "Master Time Grid"
    # NOTE: These are assumed to be UTC because we add tz="UTC" below
    "start_time": "2025-01-01 00:00:00",
    "end_time":   "2025-01-11 00:00:00",
    "freq":       "6h",  # Lowercase 'h' fixes the FutureWarning
}

def clean_var_name(var_name):
    """
    Removes common API suffixes to standardize names.
    Ex: 'air_temp_set_1' -> 'air_temp'
    """
    return var_name.replace("_set_1", "").replace("_set_2d", "")

def main():
    print(f"Loading {CONFIG['input_json']}...")
    with open(CONFIG['input_json'], "r") as f:
        data = json.load(f)

    # 1. Create the Master Time Index (UTC Aware)
    # FIX: Added tz="UTC" to match the input JSON data
    master_time = pd.date_range(
        start=CONFIG["start_time"], 
        end=CONFIG["end_time"], 
        freq=CONFIG["freq"],
        tz="UTC" 
    )
    print(f"Master Time Grid: {len(master_time)} steps ({master_time[0]} to {master_time[-1]})")

    # 2. Extract Data & Align to Master Time
    station_ids = []
    lats = []
    lons = []
    elevs = []
    
    aligned_data = []

    # Get list of stations
    stations_list = data.get("STATION", [])
    if not stations_list:
        raise ValueError("JSON does not contain 'STATION' list")

    print("Aligning station data to master timeline...")
    
    all_observed_vars = set()

    for stat in tqdm(stations_list, unit="station"):
        # -- Metadata Extraction --
        sid = stat.get("STID")
        if not sid: continue 
        
        try:
            lat = float(stat.get("LATITUDE", np.nan))
            lon = float(stat.get("LONGITUDE", np.nan))
            elev = float(stat.get("ELEVATION", np.nan))
        except (ValueError, TypeError):
            continue 

        # -- Observation Extraction --
        obs = stat.get("OBSERVATIONS", {})
        if "date_time" not in obs:
            continue 

        df = pd.DataFrame(obs)
        
        # This automatically picks up the "Z" (UTC) from your JSON
        df["date_time"] = pd.to_datetime(df["date_time"])
        df = df.set_index("date_time")

        # -- ALIGNMENT MAGIC --
        df = df[~df.index.duplicated(keep='first')]
        
        # Now both df.index and master_time are UTC, so this works!
        df_aligned = df.reindex(master_time, method='nearest', tolerance=pd.Timedelta("30min"))

        station_ids.append(sid)
        lats.append(lat)
        lons.append(lon)
        elevs.append(elev)
        aligned_data.append(df_aligned)
        
        for col in df_aligned.columns:
            all_observed_vars.add(clean_var_name(col))

    # 3. Build the 2D Arrays (Variables)
    n_stations = len(station_ids)
    n_times = len(master_time)
    
    print(f"\nConstructing Xarray Dataset for {n_stations} stations x {n_times} times...")
    
    ds = xr.Dataset(
        coords={
            "station": (["station"], station_ids),
            "time": (["time"], master_time),
            "latitude": (["station"], lats),
            "longitude": (["station"], lons),
            "elevation": (["station"], elevs),
        }
    )

    # List of variables to strictly exclude (add any others you know are dicts)
    BLACKLIST = {"QC_SUMMARY", "qc_summary", "weather_cond_code", "cloud_layer_1_code", "cloud_layer_2_code", "cloud_layer_3_code"}

    for var_name in tqdm(sorted(list(all_observed_vars)), desc="Building Variables"):
        # Skip known non-numeric fields or blacklisted ones
        if var_name in BLACKLIST:
            continue
            
        matrix = np.full((n_stations, n_times), np.nan, dtype=np.float32)
        
        for i, df in enumerate(aligned_data):
            matching_cols = [c for c in df.columns if clean_var_name(c) == var_name]
            
            if matching_cols:
                col_data = df[matching_cols[0]].values
                
                try:
                    # distinct check: if the column is object type, it might contain dicts
                    # We try to coerce to numeric, turning errors (like dicts) into NaN
                    numeric_data = pd.to_numeric(col_data, errors='coerce')
                    matrix[i, :] = numeric_data.astype(np.float32)
                except Exception:
                    # If pandas fails hard (rare), or generic error, just leave as NaNs
                    continue
        
        # Only add the variable to the dataset if it has at least SOME valid data
        # (This prevents adding empty variables that were all dicts)
        if not np.all(np.isnan(matrix)):
            ds[var_name] = (["station", "time"], matrix)

    # 4. Save to NetCDF
    print("Saving to NetCDF...")
    
    encoding = {v: {"zlib": True, "complevel": 1, "_FillValue": np.nan} for v in ds.data_vars}
    
    ds.to_netcdf(CONFIG["output_nc"], encoding=encoding, engine="h5netcdf")
    print(f"Success! Saved to {CONFIG['output_nc']}")

if __name__ == "__main__":
    main()