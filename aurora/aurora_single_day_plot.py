#!/usr/bin/env python3
# Abtin Olaee 2026
# Plots Model Weighted Average over region for one Day against station data. 

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from scipy.stats import pearsonr
from matplotlib.dates import DateFormatter, DayLocator

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "aurora_nc": "/shome/u014930890/PGE_Projects/aurora/processed_data/aurora_processed_CA_Day{day}.nc",
    "situ_path": "/shome/u014930890/PGE_Projects/sensorData/station_data.nc",

    "output_png": "/shome/u014930890/PGE_Projects/aurora/plots/aurora_validation_{region}_Day{day}.png",

    "title": "Aurora 2.5 Validation",
    "model_var": "wind_speed",
    "situ_var": "wind_speed",
    "units": "m/s",
}

REGION_BOXES = {
    "CA": (-124.50, -114.00, 32.30, 42.00),
    "BA": (-122.80, -121.50, 36.85, 38.25),
    "LA": (-118.80, -117.30, 33.50, 34.40),
    "SD": (-117.40, -116.85, 32.50, 33.316),
}

# =============================================================================
# HELPERS
# =============================================================================

def load_situ_series(nc_path, region, var_name):
    """
    Loads station data and returns:
    1. Aggregated time series (pd.Series)
    2. List of (lat, lon) tuples for all stations in the region
    """
    if not os.path.exists(nc_path):
        print(f"[Error] Station NetCDF not found: {nc_path}")
        return None, []

    ds = xr.open_dataset(nc_path, engine="h5netcdf")
    
    # 1. Filter by Region using Lat/Lon (ex. CA)
    minlon, maxlon, minlat, maxlat = REGION_BOXES[region]
    
    mask = (
        (ds.latitude.values >= minlat) & (ds.latitude.values <= maxlat) &
        (ds.longitude.values >= minlon) & (ds.longitude.values <= maxlon)
    )
    
    subset = ds.isel(station=mask) # select stations within region mask
    
    # Error handling for missing stations
    if subset.sizes['station'] == 0:
        print(f"[Warning] No stations found in region {region}")
        return None, []
    
    # Error handling for incorrect variable
    if var_name not in subset:
        print(f"[Warning] Variable {var_name} not in station dataset")
        return None, []

    # 2. Extract Coordinates for Model Matching
    # Create a list of tuples: [(lat, lon), (lat, lon), ...]
    station_lats = subset.latitude.values
    station_lons = subset.longitude.values
    coords = list(zip(station_lats, station_lons))

    # 3. Aggregate (Spatial Mean) to get Time Series
    print(f"Aggregating {subset.sizes['station']} stations for {region}...")
    series = subset[var_name].mean(dim="station", skipna=True).to_series()

    # 4. Normalize Timezone
    series.index = pd.to_datetime(series.index)
    if getattr(series.index, "tz", None) is not None:
        series.index = series.index.tz_localize(None)
    
    return series.sort_index(), coords



def load_model_series(nc_path, var_name, station_coords):
    """
    Loads model data at the grid cells where station is located.
    Returns weighting model average using nearest neighbor.
    """
    # Error Handling
    if not os.path.exists(nc_path):
        print(f"Warning: File not found {nc_path}")
        return None
    
    if not station_coords:
        print("[Warning] No station coordinates provided to model loader.")
        return pd.Series(dtype=float)

    ds = xr.open_dataset(nc_path, engine="h5netcdf")

    if var_name not in ds:
        raise KeyError(f"'{var_name}' not in dataset. Vars: {list(ds.data_vars.keys())}")

    # 1. Unpack station coordinates (Assume -180 to +180 coordinate convention)
    target_lats = np.array([c[0] for c in station_coords])
    target_lons = np.array([c[1] for c in station_coords])

    # 2. Point-wise Selection
    # We create DataArrays for the targets so xarray knows we want point-wise selection
    tgt_lat_da = xr.DataArray(target_lats, dims="station_id")
    tgt_lon_da = xr.DataArray(target_lons, dims="station_id")

    # Select nearest grid cell for each station
    # allows for multiple selections of specific grid for weighted average
    selected_points = ds[var_name].sel(
        latitude=tgt_lat_da, 
        longitude=tgt_lon_da, 
        method="nearest"
    )

    # 3. Average across the 'station_id' dimension
    mean_series = selected_points.mean(dim="station_id", skipna=True).to_series()
    
    # 4. Normalize Time
    mean_series.index = pd.to_datetime(mean_series.index)
    if getattr(mean_series.index, "tz", None) is not None:
        mean_series.index = mean_series.index.tz_localize(None)

    return mean_series.sort_index().dropna()



def compute_metrics(model_data, station_data):
    """
    Compares model data against station data and calculates error metrics.
    RMSE, MAE, MAPE, R
    Prints out performance to console. 
    """
    
    # 1. Align timesteps in model and situ data
    common_times = model_data.index.intersection(station_data.index)
    if len(common_times) <= 10:
        return None

    model_subset = model_data.loc[common_times].astype(float)
    station_subset = station_data.loc[common_times].astype(float)

    # 2. Calculate Performance/Error
    err = model_subset - station_subset
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    mask = station_subset > 0.1 # filter out significantly small observations
    if np.any(mask):
        mape = float(np.mean(np.abs((model_subset[mask] - station_subset[mask]) / station_subset[mask])) * 100.0)
    else:
        mape = float("nan") # if all station data is zero dont compute

    r, p_value = pearsonr(model_subset.values, station_subset.values)
    
    # 3. Return values
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r": float(r),
        "n": int(len(common_times)),
        "common_idx": common_times,
    }



# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate Aurora vs Synoptic (Point Weighted)")
    parser.add_argument("--day", type=str, required=True)
    parser.add_argument("--region", type=str, default="CA", choices=list(REGION_BOXES.keys()))
    args = parser.parse_args()

    nc_file = CONFIG["aurora_nc"].format(day=args.day)

    print(f"Model File:   {nc_file}")
    print(f"Station File: {CONFIG['situ_path']}")

    # 1. Load Station Data FIRST to get coordinates
    truth, coords = load_situ_series(CONFIG["situ_path"], args.region, CONFIG["situ_var"])
    
    if coords:
        print(f"Identified {len(coords)} station locations. Extraction model data...")
        # 2. Load Model Data specific to those coordinates
        model = load_model_series(nc_file, CONFIG["model_var"], coords)
    else:
        print("No station coordinates found. Skipping model load.")
        model = pd.Series(dtype=float)

    # Metrics
    metrics = {}
    if truth is not None and not model.empty:
        metrics = compute_metrics(model, truth)

    if metrics:
        print("\n" + "=" * 50)
        print(f"METRICS ({args.region} Day {args.day})  N={metrics['n']}")
        print("-" * 50)
        print(f"RMSE: {metrics['rmse']:.4f} {CONFIG['units']}")
        print(f"MAE:  {metrics['mae']:.4f} {CONFIG['units']}")
        print(f"MAPE: {metrics['mape']:.2f} %")
        print(f"R:    {metrics['r']:.4f}")
        print("=" * 50 + "\n")
    else:
        print("[Warning] Cannot compute metrics (empty intersection or missing data).")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    if not model.empty:
        ax.plot(model.index, model.values, label="Aurora", linewidth=2.5, alpha=0.9)
    else:
        ax.text(0.5, 0.5, "Aurora Data Missing", ha="center", va="center", transform=ax.transAxes)

    if truth is not None:
        if not model.empty:
            w = pd.Timedelta(hours=12)
            truth_plot = truth.loc[model.index.min() - w : model.index.max() + w]
        else:
            truth_plot = truth
        
        if not truth_plot.empty:
            ax.plot(truth_plot.index, truth_plot.values, label="Synoptic Obs", color='black', linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_title(f"{CONFIG['title']} - {args.region} (Day {args.day})", fontsize=14)
    ax.set_ylabel(f"{CONFIG['model_var']} ({CONFIG['units']})")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right")

    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
    ax.set_xlabel("Date")
    fig.autofmt_xdate()

    out_name = CONFIG["output_png"].format(region=args.region, day=args.day)
    plt.savefig(out_name, bbox_inches="tight")
    print(f"Saved Plot: {out_name}")


if __name__ == "__main__":
    main()