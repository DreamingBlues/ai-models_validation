# Plots the leadtimes of all given models for a selected time frame, and calculate error metrics
# Abtin Olaee 2026

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
import pickle
from scipy.stats import pearsonr
from matplotlib.dates import DateFormatter, DayLocator

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "model_path": "./{model_abr}/processed_data/{model_abr}_processed_CA_Day{day}.nc",
    "situ_path": "./sensorData/station_data.nc",

    "days_to_process": ["01", "03", "05", "06", "07"],
    "ref_day": "07",

    "output_png": "./{model_abr}/plots/{model_abr}_leadtime_comparison_{region}.png",
    "output_csv": "./{model_abr}/metrics/{model_abr}_leadtime_metrics_{region}.csv",
    "title": "{model_name} Lead Time Comparison",

    "model_var": "wind_speed",
    "situ_var": "wind_speed",
    "units": "m/s",

    "plot_start": "2025-01-07 00:00:00",
    "plot_end": "2025-01-11 00:00:00",
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

COLORS = {
    "01": "darkblue",
    "03": "darkmagenta",
    "05": "crimson",
    "06": "darkorange",
    "07": "gold",
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
    parser = argparse.ArgumentParser(description="Plot Lead Time Comparison")
    parser.add_argument("--region", type=str, default="CA", choices=list(REGION_BOXES.keys()))
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    args = parser.parse_args()
    region = args.region

    # Model configuration based on argument
    m_info = MODELS[args.model]
    model_abr = m_info["model_abr"]
    model_name = m_info["model_name"]

    print(f"Processing Lead Times for {region}...")

    # Step 1: Load Situ data and hold coordinates
    # We need the coordinates to tell the model loader which grid cells to pick
    situ_series, station_coords = load_situ_series(CONFIG["situ_path"], region, CONFIG["situ_var"])

    if situ_series is None or not station_coords:
        print("No station data found. Exiting.")
        sys.exit(1)


    # Step 2: Load all model runs using station coordinates
    series_dict = {}
    for day in CONFIG["days_to_process"]:
        fpath = CONFIG["model_path"].format(model_abr=model_abr, day=day)        
        print(f"Loading Day {day}: {fpath}")
        
        ts = load_model_series(fpath, CONFIG["model_var"], station_coords)
        
        if ts is not None and not ts.empty:
            series_dict[day] = ts
        else:
            print(f"  > Skipped Day {day} (missing/empty)")

    if not series_dict:
        sys.exit(1)

    # Step 3: Setup plot
    plt.figure(figsize=(11, 6), dpi=150)

    # Header
    print("\n" + "=" * 105)
    print(f"{'Run / Lead Time':<32} | {'N':<6} | {'RMSE':<10} | {'MAE':<10} | {'MAPE (%)':<10} | {'Corr':<6}")
    print("-" * 105)

    ref_day_int = int(CONFIG["ref_day"])
    metrics_list = []

    for day in CONFIG["days_to_process"]:
        if day not in series_dict:
            continue

        s_curr = series_dict[day]
        day_int = int(day)
        diff_days = ref_day_int - day_int

        if day == CONFIG["ref_day"]:
            label = f"Day {day} (Current)"
            lead_hours = 0
        else:
            lead_hours = diff_days * 24
            label = f"Day {day} ({lead_hours}h Lead)"

        rmse_val = mae_val = mape_val = r_val = np.nan
        rmse_str = mae_str = mape_str = r_str = "N/A"
        n_str = "0"

        if situ_series is not None:
            out = compute_metrics(s_curr, situ_series)
            if out is not None:
                rmse_val = out["rmse"]
                mae_val  = out["mae"]
                mape_val = out["mape"]
                r_val    = out["r"]
                n_common = out["n"]

                rmse_str = f"{rmse_val:.2f}"
                mae_str  = f"{mae_val:.2f}"
                mape_str = f"{mape_val:.2f}" if np.isfinite(mape_val) else "NaN"
                r_str    = f"{r_val:.2f}"
                n_str    = str(n_common)

        print(f"{label:<32} | {n_str:<6} | {rmse_str:<10} | {mae_str:<10} | {mape_str:<10} | {r_str:<6}")

        metrics_list.append({
            "Model": model_name,
            "Run_Day": day,
            "Lead_Hours": lead_hours,
            "N_Common": int(n_str) if n_str.isdigit() else 0,
            "RMSE": rmse_val,
            "MAE": mae_val,
            "MAPE": mape_val,
            "Correlation": r_val,
        })

        plt.plot(
            s_curr.index, s_curr.values,
            color=COLORS.get(day, "gray"),
            linewidth=2,
            label=label,
            alpha=0.8
        )

    print("=" * 105 + "\n")

    # Step 5: Save metrics for table performance comparison
    if metrics_list:
        csv_file = CONFIG["output_csv"].format(region=region, model_abr=model_abr)
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        pd.DataFrame(metrics_list).to_csv(csv_file, index=False, float_format="%.4f")
        print(f"Metrics exported to: {csv_file}")

    # Step 6: Plot situ averages
    if situ_series is not None:
        ref_series = series_dict.get(CONFIG["ref_day"])
        if ref_series is not None:
            w = pd.Timedelta(hours=6)
            start_plot = ref_series.index.min() - w
            end_plot = ref_series.index.max() + w
            situ_plot = situ_series.loc[start_plot:end_plot]
        else:
            all_idx = pd.DatetimeIndex([])
            for s in series_dict.values():
                all_idx = all_idx.union(s.index)
            situ_plot = situ_series.loc[all_idx.min():all_idx.max()]

        plt.plot(
            situ_plot.index, situ_plot.values,
            color="black", linestyle="--", linewidth=2.0,
            label="Synoptic (Truth)", zorder=10
        )

    # Formatting
    formatted_title = CONFIG['title'].format(model_name=model_name)
    plt.title(f"{formatted_title} - {region} ({CONFIG['model_var']})", fontsize=14, pad=10)
    plt.ylabel(f"Wind Speed ({CONFIG['units']})", fontsize=12)
    plt.xlabel("Date (UTC)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc="upper right")

    plt.xlim(left=pd.Timestamp(CONFIG["plot_start"]), right=pd.Timestamp(CONFIG["plot_end"]))

    ax = plt.gca()
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d\n%H:%M"))
    plt.gcf().autofmt_xdate()

    out_file = CONFIG["output_png"].format(region=region, model_abr=model_abr)
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Comparison plot saved to: {out_file}")


if __name__ == "__main__":
    main()