#!/usr/bin/env python3
# Abtin Olaee 2025

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
    "model": "Aurora 2.5",
    "aurora_file_pattern": "/shome/u014930890/PGE_Projects/aurora/processed_data/aurora_processed_CA_Day{day}.nc",
    "situ_path": "/shome/u014930890/PGE_Projects/sensorData/station_data.nc",

    "days_to_process": ["01", "03", "05", "06", "07"],
    "ref_day": "07",

    "output_png": "/shome/u014930890/PGE_Projects/aurora/plots/aurora_leadtime_comparison_{region}.png",
    "output_csv": "/shome/u014930890/PGE_Projects/aurora/metrics/aurora_leadtime_metrics_{region}.csv",
    "title": "Aurora 2.5 Lead Time Comparison",

    "model_var": "wind_speed",
    "situ_var": "wind_speed",
    "units": "m/s",

    "plot_start": "2025-01-07 00:00:00",
    "plot_end": "2025-01-11 00:00:00",
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
    parser = argparse.ArgumentParser(description="Plot Aurora Lead Time Comparison (Weighted)")
    parser.add_argument("--region", type=str, default="CA", choices=list(REGION_BOXES.keys()))
    args = parser.parse_args()
    region = args.region

    print(f"--- Processing Lead Times for Region: {region} ---")

    # 1) Load Truth (Situ) FIRST to get coordinates
    # We need the coordinates to tell the model loader which grid cells to pick
    situ_series, station_coords = load_situ_series(CONFIG["situ_path"], region, CONFIG["situ_var"])

    if situ_series is None or not station_coords:
        print("No station data found. Exiting.")
        sys.exit(1)

    # 2) Load all model runs using station coordinates
    series_dict = {}
    for day in CONFIG["days_to_process"]:
        fpath = CONFIG["aurora_file_pattern"].format(day=day)
        print(f"Loading Day {day}: {fpath}")
        
        # Call the strict function (removed 'region' arg)
        ts = load_model_series(fpath, CONFIG["model_var"], station_coords)
        
        # CHANGE: Check 'ts.empty' because Function #1 returns an empty Series on some errors
        if ts is not None and not ts.empty:
            series_dict[day] = ts
        else:
            print(f"  > Skipped Day {day} (missing/empty)")

    if not series_dict:
        print("No valid Aurora NetCDF files loaded. Exiting.")
        sys.exit(1)

    # 3) Setup plot
    plt.figure(figsize=(11, 6), dpi=150)

    # Header
    print("\n" + "=" * 105)
    print(f"{'Run / Lead Time':<32} | {'N':<6} | {'RMSE':<10} | {'MAE':<10} | {'MAPE (%)':<10} | {'Corr':<6}")
    print("-" * 105)

    ref_day_int = int(CONFIG["ref_day"])
    metrics_list = []

    # 4) Loop & Plot
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
            "Model": CONFIG["model"],
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

    # 5) Save metrics
    if metrics_list:
        csv_file = CONFIG["output_csv"].format(region=region)
        pd.DataFrame(metrics_list).to_csv(csv_file, index=False, float_format="%.4f")
        print(f"Metrics exported to: {csv_file}")

    # 6) Plot Truth
    if situ_series is not None:
        # Determine plot range based on available data
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

    # 7) Formatting
    plt.title(f"{CONFIG['title']} - {region} ({CONFIG['model_var']})", fontsize=14, pad=10)
    plt.ylabel(f"Wind Speed ({CONFIG['units']})", fontsize=12)
    plt.xlabel("Date (UTC)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc="upper right")

    plt.xlim(left=pd.Timestamp(CONFIG["plot_start"]), right=pd.Timestamp(CONFIG["plot_end"]))

    ax = plt.gca()
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d\n%H:%M"))
    plt.gcf().autofmt_xdate()

    out_file = CONFIG["output_png"].format(region=region)
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Comparison plot saved to: {out_file}")


if __name__ == "__main__":
    main()