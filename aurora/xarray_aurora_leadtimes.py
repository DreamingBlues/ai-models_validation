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
    # Pattern for processed Aurora NetCDF files
    "aurora_file_pattern": "/shome/u014930890/PGE_Projects/aurora_new/processed_data/aurora_processed_CA_Day{day}.nc",

    "situ_path": "/shome/u014930890/PGE_Projects/station_data.pkl",

    "days_to_process": ["01", "03", "05", "06", "07"],
    "ref_day": "07",

    "output_png": "aurora_leadtime_comparison_{region}.png",
    "output_csv": "aurora_leadtime_metrics_{region}.csv",
    "title": "Aurora 2.5 Lead Time Comparison",

    # NetCDF variable name
    "model_var": "wind_speed",
    "situ_var": "wind_speed",
    "units": "m/s",

    # Force plot window
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

def load_aurora_series_nc(nc_path, region_name, var_name):
    """
    Loads Aurora NetCDF, subsets to region box, and returns spatial mean time series.
    Assumes coords: time, latitude, longitude.
    """
    if not os.path.exists(nc_path):
        print(f"Warning: File not found {nc_path}")
        return None

    minlon, maxlon, minlat, maxlat = REGION_BOXES[region_name]

    try:
        ds = xr.open_dataset(nc_path, engine="h5netcdf")
    except Exception as e:
        print(f"Warning: Could not open NetCDF {nc_path}: {e}")
        return None

    if var_name not in ds:
        print(f"Warning: Variable '{var_name}' not in {nc_path}. Vars: {list(ds.data_vars.keys())}")
        return None

    # Handle latitude direction (ascending vs descending)
    lat_increasing = bool(ds.latitude[1] > ds.latitude[0])
    lat_slice = slice(minlat, maxlat) if lat_increasing else slice(maxlat, minlat)

    # Handle lon convention: file is likely -180..180 already, but check anyway
    ds_lon_min = float(ds.longitude.min().item())
    if ds_lon_min >= 0 and minlon < 0:
        minlon = minlon % 360
        maxlon = maxlon % 360

    subset = ds.sel(latitude=lat_slice, longitude=slice(minlon, maxlon))

    if subset.sizes.get("latitude", 0) == 0 or subset.sizes.get("longitude", 0) == 0:
        print(f"Warning: Empty slice for {region_name} in {os.path.basename(nc_path)}")
        return None

    ts = subset[var_name].mean(dim=("latitude", "longitude"), skipna=True).to_series()
    ts.index = pd.to_datetime(ts.index)
    if getattr(ts.index, "tz", None) is not None:
        ts.index = ts.index.tz_localize(None)

    ts = ts.sort_index().dropna()
    if ts.empty:
        return None
    return ts


def load_situ_series(situ_path, region_name, var_key):
    if not os.path.exists(situ_path):
        print(f"Warning: Situ file not found {situ_path}")
        return None

    print(f"Loading Situ Data: {situ_path}")
    with open(situ_path, "rb") as f:
        data = pickle.load(f)

    minlon, maxlon, minlat, maxlat = REGION_BOXES[region_name]
    situ_cols = []

    for (name, (lon, lat)), entry in data.items():
        if not (minlon <= lon <= maxlon and minlat <= lat <= maxlat):
            continue

        df = entry.get("situ")
        if not isinstance(df, pd.DataFrame):
            continue
        if var_key not in df.columns:
            continue

        s = df[var_key].copy()
        s.name = name
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)

        situ_cols.append(s)

    if not situ_cols:
        return None

    print(f"  > Found {len(situ_cols)} stations in {region_name}.")
    df_all = pd.concat(situ_cols, axis=1).sort_index()
    return df_all.mean(axis=1)


def compute_metrics(model_s, truth_s):
    common = model_s.index.intersection(truth_s.index)
    if len(common) <= 10:
        return None

    m = model_s.loc[common].astype(float)
    t = truth_s.loc[common].astype(float)

    rmse = float(np.sqrt(((m - t) ** 2).mean()))
    mae = float(np.mean(np.abs(m - t)))
    r = float(pearsonr(m.values, t.values)[0])

    mask = t > 0.1
    if np.any(mask):
        mape = float(np.mean(np.abs((m[mask] - t[mask]) / t[mask])) * 100.0)
    else:
        mape = float("nan")

    return rmse, mae, mape, r, len(common)

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot Aurora Lead Time Comparison (NetCDF)")
    parser.add_argument("--region", type=str, default="CA", choices=list(REGION_BOXES.keys()))
    args = parser.parse_args()
    region = args.region

    print(f"--- Processing Lead Times for Region: {region} ---")

    # 1) Load all model runs
    series_dict = {}
    for day in CONFIG["days_to_process"]:
        fpath = CONFIG["aurora_file_pattern"].format(day=day)
        print(f"Loading Day {day}: {fpath}")
        ts = load_aurora_series_nc(fpath, region, CONFIG["model_var"])
        if ts is not None:
            series_dict[day] = ts
        else:
            print(f"  > Skipped Day {day} (missing/empty)")

    if not series_dict:
        print("No valid Aurora NetCDF files loaded. Exiting.")
        sys.exit(1)

    # 2) Load truth (situ)
    situ_series = load_situ_series(CONFIG["situ_path"], region, CONFIG["situ_var"])

    # 3) Setup plot
    plt.figure(figsize=(11, 6), dpi=150)

    # Header
    print("\n" + "=" * 105)
    print(f"{'Run / Lead Time':<32} | {'N':<6} | {'RMSE':<10} | {'MAE':<10} | {'MAPE (%)':<10} | {'Corr':<6}")
    print("-" * 105)

    ref_day_int = int(CONFIG["ref_day"])
    metrics_list = []

    # 4) Loop
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
                rmse_val, mae_val, mape_val, r_val, n_common = out
                rmse_str = f"{rmse_val:.2f}"
                mae_str = f"{mae_val:.2f}"
                mape_str = f"{mape_val:.2f}" if np.isfinite(mape_val) else "NaN"
                r_str = f"{r_val:.2f}"
                n_str = str(n_common)

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

    # 5) Save metrics to CSV
    if metrics_list:
        csv_file = CONFIG["output_csv"].format(region=region)
        pd.DataFrame(metrics_list).to_csv(csv_file, index=False, float_format="%.4f")
        print(f"Metrics exported to: {csv_file}")

    # 6) Plot situ (truth)
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
