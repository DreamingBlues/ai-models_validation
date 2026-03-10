# Plots Model Weighted Average over region for one Day against station data. 
# Abtin Olaee 2026

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
    "model_path": "./{model_abr}/processed_data/{model_abr}_processed_CA_Day{day}.nc",
    "situ_path": "./sensorData/station_data.nc",

    "output_png": "./{model_abr}/plots/{model_abr}_validation_{region}_Day{day}.png",

    "title": "{model_name} Validation",
    "model_var": "wind_speed",
    "situ_var": "wind_speed",
    "units": "m/s",
}

MODELS = {
    "ifs":       {"model_abr": "ifs",       "model_name": "IFS"},
    "fcn2":      {"model_abr": "fcn2",      "model_name": "FCNetv2-small"},
    "fcn3":      {"model_abr": "fcn3",      "model_name": "FCNetv3 Deterministic"},
    "aurora":    {"model_abr": "aurora",    "model_name": "Aurora"},
    "graphcast": {"model_abr": "graphcast", "model_name": "GraphCast"},
    "nbm":       {"model_abr": "nbm",       "model_name": "NBM"}
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
    Filters station data by region boudning box and computes a spatially averaged time series.

    Args:
        nc_path (str): The file path to the NetCDF dataset containing in-situ station data.
        region (str): The name of the region to filter by (must exist in REGION_BOXES).
        var_name (str): The specific data variable to extract (e.g. air_temp).

    Returns:
        pd.Series: A time-indexed pandas Series representing the spatial mean of 
            the variable across all stations within the specified region.
        coords (list): A list of (latitude, longitude) pairs for all stations included in 
            the regional subset.
    """
    if not os.path.exists(nc_path):
        print(f"[Error] Station NetCDF not found: {nc_path}")
        return None, []

    ds = xr.open_dataset(nc_path, engine="h5netcdf")
    
    # Step 1: filter by begion using Lat/Lon
    minlon, maxlon, minlat, maxlat = REGION_BOXES[region]
    
    mask = (
        (ds.latitude.values >= minlat) & (ds.latitude.values <= maxlat) &
        (ds.longitude.values >= minlon) & (ds.longitude.values <= maxlon)
    )
    
    subset = ds.isel(station=mask) # select stations within region mask
    
    # error handling for missing stations and incorrect variable
    if subset.sizes['station'] == 0:
        print(f"[Warning] No stations found in region {region}")
        return None, []
    if var_name not in subset:
        print(f"[Warning] Variable {var_name} not in station dataset")
        return None, []

    # Step 2: extract coordinates for Model matching
    # create a list of tuples, ex. [(lat, lon), (lat, lon), ...]
    station_lats = subset.latitude.values
    station_lons = subset.longitude.values
    coords = list(zip(station_lats, station_lons))

    # Step 3: aggregate (Spatial Mean) to get Time Series
    print(f"Aggregating {subset.sizes['station']} stations for {region}...")
    series = subset[var_name].mean(dim="station", skipna=True).to_series()

    # Step 4: normalize timezone
    series.index = pd.to_datetime(series.index)
    if getattr(series.index, "tz", None) is not None:
        series.index = series.index.tz_localize(None)
    
    return series.sort_index(), coords


def load_model_series(nc_path, var_name, station_coords):
    """
    Loads model data at the grid cells where station is located.
    Returns weighting model average using nearest neighbor.

    Args:
        nc_path (str): The file path to the NetCDF/HDF5 dataset containing model data.
        var_name (str): The specific data variable to extract (e.g., 'air_temp', 'wind_speed').
        station_coords (list of tuples): A list of (lat, lon) pairs representing the sensor locations 
            to sample from the model grid.

    Returns:
        pd.Series: A time-indexed pandas Series representing the spatial mean of the 
            specified variable across all provided station coordinates, with NaNs removed.
    """
    # error handling
    if not os.path.exists(nc_path):
        print(f"Warning: File not found {nc_path}")
        return None
    if not station_coords:
        print("[Warning] No station coordinates provided to model loader.")
        return pd.Series(dtype=float)

    ds = xr.open_dataset(nc_path, engine="h5netcdf")

    if var_name not in ds:
        raise KeyError(f"'{var_name}' not in dataset. Vars: {list(ds.data_vars.keys())}")

    # Step 1: Unpack station coordinates (Assume -180 to +180 coordinate convention)
    target_lats = np.array([c[0] for c in station_coords])
    target_lons = np.array([c[1] for c in station_coords])

    # Step 2: Point-wise Selection
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

    # step 3: Average across the 'station_id' dimension
    mean_series = selected_points.mean(dim="station_id", skipna=True).to_series()

    # normalize time
    mean_series.index = pd.to_datetime(mean_series.index)
    if getattr(mean_series.index, "tz", None) is not None:
        mean_series.index = mean_series.index.tz_localize(None)

    return mean_series.sort_index().dropna()


def compute_metrics(model_data, station_data):
    """
    Compares model data against station data and calculates error metrics.
    RMSE, MAE, MAPE, R

    Args:
        model_data (pd.Series): Time-indexed series containing model predictions.
        station_data (pd.Series): Time-indexed series containing ground-truth observations.

    Returns:
        dict: dictionary containing keys for error metrics 
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
    parser = argparse.ArgumentParser(description="Validate Model vs Synoptic (Point Weighted)")
    parser.add_argument("--day", type=str, required=True, help="In format DD")
    parser.add_argument("--region", type=str, default="CA", choices=list(REGION_BOXES.keys()))
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    args = parser.parse_args()

    # Look up the model info
    m_info = MODELS[args.model]
    model_abr = m_info["model_abr"]
    model_name = m_info["model_name"]

    # Format the paths dynamically
    nc_file = CONFIG["model_path"].format(model_abr=model_abr, day=args.day)
    out_name = CONFIG["output_png"].format(model_abr=model_abr, region=args.region, day=args.day)

    print(f"Model File:   {nc_file}")
    print(f"Station File: {CONFIG['situ_path']}")

    # Step 1: Load Station Data FIRST to get coordinates
    truth, coords = load_situ_series(CONFIG["situ_path"], args.region, CONFIG["situ_var"])
    
    if coords:
        print(f"Identified {len(coords)} station locations. Extracting model data...")
        model = load_model_series(nc_file, CONFIG["model_var"], coords)
    else:
        print("No station coordinates found. Skipping model load.")
        model = pd.Series(dtype=float)

    # Step 2: Calculate metrics and output to console, Will not be saved, just for validation purposes
    metrics = {}
    if truth is not None and not model.empty:
        metrics = compute_metrics(model, truth)

    if metrics:
        print("\n" + "=" * 50)
        print(f"METRICS ({model_name} | {args.region} Day {args.day})  N={metrics['n']}")
        print("-" * 50)
        print(f"RMSE: {metrics['rmse']:.4f} {CONFIG['units']}")
        print(f"MAE:  {metrics['mae']:.4f} {CONFIG['units']}")
        print(f"MAPE: {metrics['mape']:.2f} %")
        print(f"R:    {metrics['r']:.4f}")
        print("=" * 50 + "\n")
    else:
        print("[Warning] Cannot compute metrics (empty intersection or missing data).")

    # Step 3: Create plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    if not model.empty:
        ax.plot(model.index, model.values, label=model_name, linewidth=2.5, alpha=0.9)
    else:
        ax.text(0.5, 0.5, f"{model_abr} Data Missing", ha="center", va="center", transform=ax.transAxes)

    if truth is not None:
        if not model.empty:
            w = pd.Timedelta(hours=12)
            truth_plot = truth.loc[model.index.min() - w : model.index.max() + w]
        else:
            truth_plot = truth
        
        if not truth_plot.empty:
            ax.plot(truth_plot.index, truth_plot.values, label="Synoptic Obs", color='black', linestyle="--", linewidth=1.5, alpha=0.8)

    # format the title using the model_name
    title_str = CONFIG["title"].format(model_name=model_name)
    ax.set_title(f"{title_str} - {args.region} (Day {args.day})", fontsize=14)
    ax.set_ylabel(f"{CONFIG['model_var']} ({CONFIG['units']})")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right")

    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
    ax.set_xlabel("Date")
    fig.autofmt_xdate()

    # Ensure output directory exists before saving
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    plt.savefig(out_name, bbox_inches="tight")
    print(f"Saved Plot: {out_name}")


if __name__ == "__main__":
    main()