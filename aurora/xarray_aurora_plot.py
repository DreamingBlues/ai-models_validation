#!/usr/bin/env python3
# Abtin Olaee 2025

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
    # Updated path to match your previous processing script (aurora_new)
    "aurora_nc": "/shome/u014930890/PGE_Projects/aurora/processed_data/aurora_processed_CA_Day{day}.nc",
    
    # Updated to point to your new Station NetCDF
    "situ_path": "/shome/u014930890/PGE_Projects/station_data_2d.nc",
    
    "output_png": "aurora_validation_{region}_Day{day}.png",
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
# LOADERS
# =============================================================================

def load_model_series(nc_path: str, region: str, var_name: str) -> pd.Series:
    if not os.path.exists(nc_path):
        raise FileNotFoundError(nc_path)

    ds = xr.open_dataset(nc_path, engine="h5netcdf")

    if var_name not in ds:
        raise KeyError(f"'{var_name}' not in dataset. Vars: {list(ds.data_vars.keys())}")

    minlon, maxlon, minlat, maxlat = REGION_BOXES[region]

    # Handle latitude direction
    lat_increasing = bool(ds.latitude[1] > ds.latitude[0])
    lat_slice = slice(minlat, maxlat) if lat_increasing else slice(maxlat, minlat)

    # Handle 0..360 longitude convention
    ds_lon_min = float(ds.longitude.min().item())
    if ds_lon_min >= 0 and minlon < 0:
        minlon = minlon % 360
        maxlon = maxlon % 360

    subset = ds.sel(latitude=lat_slice, longitude=slice(minlon, maxlon))
    if subset.sizes.get("latitude", 0) == 0 or subset.sizes.get("longitude", 0) == 0:
        return pd.Series(dtype=float)

    mean_series = subset[var_name].mean(dim=("latitude", "longitude"), skipna=True).to_series()
    mean_series.index = pd.to_datetime(mean_series.index)
    
    # Normalize Timezone
    if getattr(mean_series.index, "tz", None) is not None:
        mean_series.index = mean_series.index.tz_localize(None)

    return mean_series.sort_index().dropna()


def load_situ_series(nc_path, region, var_name):
    """
    Loads station data from the new 2D NetCDF format.
    """
    if not os.path.exists(nc_path):
        print(f"[Error] Station NetCDF not found: {nc_path}")
        return None

    # Open the station dataset
    ds = xr.open_dataset(nc_path, engine="h5netcdf")
    
    # 1. Filter by Region (Lat/Lon)
    minlon, maxlon, minlat, maxlat = REGION_BOXES[region]
    
    # Create a boolean mask for stations inside the box
    # We use .values to ensure we get a numpy boolean array for indexing
    mask = (
        (ds.latitude.values >= minlat) & (ds.latitude.values <= maxlat) &
        (ds.longitude.values >= minlon) & (ds.longitude.values <= maxlon)
    )
    
    # 2. Select Stations
    subset = ds.isel(station=mask)
    
    if subset.sizes['station'] == 0:
        print(f"[Warning] No stations found in region {region}")
        return None
        
    if var_name not in subset:
        print(f"[Warning] Variable {var_name} not in station dataset")
        return None

    # 3. Aggregate (Spatial Mean) to get Time Series
    print(f"Aggregating {subset.sizes['station']} stations for {region}...")
    series = subset[var_name].mean(dim="station", skipna=True).to_series()

    # 4. Normalize Timezone (Critical for alignment with model data)
    series.index = pd.to_datetime(series.index)
    if getattr(series.index, "tz", None) is not None:
        series.index = series.index.tz_localize(None)
    
    return series.sort_index()

# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(model: pd.Series, truth: pd.Series) -> dict:
    common = model.index.intersection(truth.index)
    if common.empty:
        return {}

    m = model.loc[common].astype(float)
    t = truth.loc[common].astype(float)

    err = m - t
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    # MAPE: exclude very small truth values to avoid blow-ups
    mask = t > 0.1
    if np.any(mask):
        mape = float(np.mean(np.abs((m[mask] - t[mask]) / t[mask])) * 100.0)
    else:
        mape = float("nan")

    r, _ = pearsonr(m.values, t.values)

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r": float(r),
        "n": int(len(common)),
        "common_idx": common,
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate Aurora vs Synoptic stations")
    parser.add_argument("--day", type=str, required=True)
    parser.add_argument("--region", type=str, default="CA", choices=list(REGION_BOXES.keys()))
    args = parser.parse_args()

    nc_file = CONFIG["aurora_nc"].format(day=args.day)

    print(f"Model File:   {nc_file}")
    print(f"Station File: {CONFIG['situ_path']}")

    # Load
    model = load_model_series(nc_file, args.region, CONFIG["model_var"])
    truth = load_situ_series(CONFIG["situ_path"], args.region, CONFIG["situ_var"])

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
        if truth is None:
            print("[Warning] Synoptic (station) dataset not available or no stations in region.")
        elif model.empty:
            print("[Warning] Model series is empty (all NaN or empty slice).")
        else:
            print("[Warning] No overlapping timestamps between model and synoptic series.")
            print(f"Model range: {model.index.min()} to {model.index.max()}")
            print(f"Situ range:  {truth.index.min()} to {truth.index.max()}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    if not model.empty:
        ax.plot(model.index, model.values, label="Aurora Model", linewidth=2.5, alpha=0.9)
    else:
        ax.text(0.5, 0.5, "Aurora Data Missing (Empty/NaN)", ha="center", va="center", transform=ax.transAxes)

    if truth is not None:
        # For readability, clip stations to model window if model exists
        if not model.empty:
            w = pd.Timedelta(hours=12)
            truth_plot = truth.loc[model.index.min() - w : model.index.max() + w]
        else:
            truth_plot = truth
        
        if not truth_plot.empty:
            ax.plot(truth_plot.index, truth_plot.values, label="Synoptic", color='black', linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_title(f"{CONFIG['title']} - {args.region} (Day {args.day})", fontsize=14)
    ax.set_ylabel(f"{CONFIG['model_var']} ({CONFIG['units']})")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right")

    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
    ax.set_xlabel("Time")
    fig.autofmt_xdate()

    out_name = CONFIG["output_png"].format(region=args.region, day=args.day)
    plt.savefig(out_name, bbox_inches="tight")
    print(f"Saved Plot: {out_name}")


if __name__ == "__main__":
    main()