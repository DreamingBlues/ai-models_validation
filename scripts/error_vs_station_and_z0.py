# Multi-Model Time-Averaged Grid-Cell Error vs Station Count and z0_era5
# Verification period: Jan 07 00 UTC through Jan 10 18 UTC
# Each point = one occupied model grid cell
# Error = time-averaged grid-cell MAE, RMSE, or MAPE over the full period
# Abtin Olaee 2026

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from viz_config import (
    DEFAULT_INIT_DAY,
    DEFAULT_REGION,
    LEAD_DAYS,
    MAPE_MIN_OBS,
    MODELS,
    PATHS,
    PLOT_WINDOW,
    REGIONS,
    VARIABLES,
)
from viz_utils import (
    clean_time_index,
    ensure_parent_dir,
    model_path,
    open_dataset_safe,
    output_path as configured_output_path,
    project_path,
    region_mask,
    trim_to_period,
)

# HELPERS
def load_station_subset(path, region, var_name):
    path = project_path(path)

    if not path.exists():
        print(f"[Error] Station NetCDF not found: {path}")
        return None

    try:
        ds = open_dataset_safe(path)
    except Exception as e:
        print(f"[Error] Could not open station NetCDF: {e}")
        return None

    if var_name not in ds:
        print(f"[Error] Variable '{var_name}' not found in station file.")
        return None

    if VARIABLES["z0"] not in ds:
        print(f"[Error] Variable '{VARIABLES['z0']}' not found in station file.")
        return None

    mask = region_mask(
        ds["latitude"].values,
        ds["longitude"].values,
        region,
    )

    subset = ds.isel(station=mask)

    if subset.sizes["station"] == 0:
        print(f"[Warning] No stations found within {region} bounds.")
        return None

    print(f"Loaded {subset.sizes['station']} stations for {region}")
    return subset


def add_trendline(ax, x, y, color):
    """
    Adds a linear trendline and returns the slope.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    if np.sum(valid) < 2:
        return np.nan

    x_valid = x[valid]
    y_valid = y[valid]

    if len(np.unique(x_valid)) < 2:
        return np.nan

    slope, intercept = np.polyfit(x_valid, y_valid, 1)

    x_line = np.linspace(0, np.nanmax(x_valid), 100)
    y_line = slope * x_line + intercept

    ax.plot(
        x_line,
        y_line,
        color=color,
        linestyle="--",
        linewidth=1.8,
        alpha=0.9,
    )

    return slope


def compute_model_gridcell_timeavg_errors(
    ds_model,
    ds_stations,
    model_var,
    station_var,
    metric,
    start_time,
    end_time
):
    """
    Computes one time-averaged error value per occupied model grid cell.

    For each occupied grid cell:
        1. Average all stations inside that grid cell at each timestep.
        2. Extract model value at that same grid cell.
        3. Match common timestamps within Jan 7-10.
        4. Compute MAE_i, RMSE_i, or MAPE_i over time.

    Output:
        station_counts, z0_era5s, signed_biases, errors
    """

    grid_lat = ds_model.latitude.values
    grid_lon = ds_model.longitude.values

    if grid_lat.ndim == 1 and grid_lon.ndim == 1:
        grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)
    else:
        grid_lat_2d, grid_lon_2d = grid_lat, grid_lon

    flat_lats = grid_lat_2d.ravel()
    flat_lons = grid_lon_2d.ravel()

    valid_grid = np.isfinite(flat_lats) & np.isfinite(flat_lons)

    if not np.any(valid_grid):
        return None

    grid_points = np.column_stack((flat_lats[valid_grid], flat_lons[valid_grid]))
    tree = cKDTree(grid_points)

    st_lats = ds_stations.latitude.values
    st_lons = ds_stations.longitude.values
    st_points = np.column_stack((st_lats, st_lons))

    _, grid_indices_valid = tree.query(st_points, k=1)

    valid_flat_indices = np.where(valid_grid)[0]
    grid_indices = valid_flat_indices[grid_indices_valid]

    unique_grid_indices = np.unique(grid_indices)

    station_counts = []
    z0_era5s = []
    signed_biases = []
    errors = []

    for idx in unique_grid_indices:
        member_mask = grid_indices == idx
        num_stations = int(np.sum(member_mask))

        obs_s = (
            ds_stations
            .isel(station=member_mask)[station_var]
            .mean(dim="station", skipna=True)
            .to_series()
        )

        obs_s = clean_time_index(obs_s)
        obs_s = trim_to_period(obs_s, start_time, end_time)

        try:
            mod_s = (
                ds_model[model_var]
                .sel(
                    latitude=flat_lats[idx],
                    longitude=flat_lons[idx],
                    method="nearest"
                )
                .to_series()
            )
        except Exception as e:
            print(f"  > Could not extract model grid cell {idx}: {e}")
            continue

        mod_s = clean_time_index(mod_s)
        mod_s = trim_to_period(mod_s, start_time, end_time)

        common_times = obs_s.index.intersection(mod_s.index)

        if len(common_times) == 0:
            continue

        obs_valid = obs_s.loc[common_times]
        mod_valid = mod_s.loc[common_times]

        err = mod_valid - obs_valid

        valid = (
            np.isfinite(obs_valid.values) &
            np.isfinite(mod_valid.values) &
            np.isfinite(err.values)
        )

        if metric == "mape":
            valid = valid & (np.abs(obs_valid.values) > MAPE_MIN_OBS)

        if np.sum(valid) == 0:
            continue

        obs_valid_arr = obs_valid.values[valid]
        err_arr = err.values[valid]

        if metric == "rmse":
            grid_error = np.sqrt(np.nanmean(err_arr ** 2))

        elif metric == "mae":
            grid_error = np.nanmean(np.abs(err_arr))

        elif metric == "mape":
            grid_error = np.nanmean(np.abs(err_arr / obs_valid_arr)) * 100.0

        else:
            raise ValueError(f"Unsupported metric: {metric}")

        signed_bias = np.nanmean(err_arr)

        elev_cell = (
            ds_stations[VARIABLES["z0"]]
            .isel(station=member_mask)
            .mean(skipna=True)
            .item()
        )

        if (
            pd.notna(grid_error) and
            pd.notna(signed_bias) and
            pd.notna(elev_cell) and
            num_stations > 0
        ):
            station_counts.append(num_stations)
            z0_era5s.append(elev_cell)
            signed_biases.append(signed_bias)
            errors.append(grid_error)

    if not errors:
        return None

    return {
        "station_counts": np.array(station_counts, dtype=float),
        "z0_era5s": np.array(z0_era5s, dtype=float),
        "signed_biases": np.array(signed_biases, dtype=float),
        "errors": np.array(errors, dtype=float),
    }


# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Plot multi-model time-averaged grid-cell error vs station count and z0_era5."
    )

    parser.add_argument(
        "--region",
        type=str,
        default=DEFAULT_REGION,
        choices=list(REGIONS.keys()),
        help="Region to process. Default: LA."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Models to process, in the order given."
    )

    parser.add_argument(
        "--init-day",
        dest="init_day",
        type=str,
        default=DEFAULT_INIT_DAY,
        choices=LEAD_DAYS,
        help="Forecast initialization day. Default: 05."
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="mape",
        choices=["mae", "rmse", "mape"],
        help="Time-averaged grid-cell error metric."
    )

    args = parser.parse_args()

    region = args.region
    selected_models = args.models
    init_day = args.init_day
    metric = args.metric.lower()

    if metric == "mae":
        metric_label = "MAE"
        metric_units = VARIABLES["units"]
    elif metric == "rmse":
        metric_label = "RMSE"
        metric_units = VARIABLES["units"]
    elif metric == "mape":
        metric_label = "MAPE"
        metric_units = "%"
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    start_time = pd.Timestamp(PLOT_WINDOW["start"])
    end_time = pd.Timestamp(PLOT_WINDOW["end"])

    init_time = pd.Timestamp(f"2025-01-{init_day} 00:00:00")

    print(f"Region: {region}")
    print(f"Models: {selected_models}")
    print(f"Initialization: {init_time}")
    print(f"Verification window: {start_time} to {end_time} exclusive")
    print(f"Metric: {metric_label}")

    if metric == "mape":
        print(
            f"MAPE threshold: observed wind speed > "
            f"{MAPE_MIN_OBS} {VARIABLES['units']}"
        )

    # Load stations once
    ds_stations = load_station_subset(
        PATHS["station"],
        region,
        VARIABLES["station_wind"],
    )

    if ds_stations is None:
        sys.exit(1)


    # Compute all model results
    results = {}

    for model_key in selected_models:
        model_name = MODELS[model_key]
        model_file = model_path(model_key, init_day)

        print(f"\nProcessing {model_name}: {model_file}")

        if not model_file.exists():
            print("  > Missing file. Skipping.")
            continue

        try:
            ds_model = open_dataset_safe(model_file)
        except Exception as e:
            print(f"  > Could not open model file: {e}")
            continue

        if VARIABLES["model_wind"] not in ds_model:
            print(f"  > Variable '{VARIABLES['model_wind']}' not found. Skipping.")
            ds_model.close()
            continue

        try:
            model_results = compute_model_gridcell_timeavg_errors(
                ds_model=ds_model,
                ds_stations=ds_stations,
                model_var=VARIABLES["model_wind"],
                station_var=VARIABLES["station_wind"],
                metric=metric,
                start_time=start_time,
                end_time=end_time
            )
        except Exception as e:
            print(f"  > Error computing grid-cell errors: {e}")
            ds_model.close()
            continue

        ds_model.close()

        if model_results is None:
            print("  > No valid grid-cell errors. Skipping.")
            continue

        results[model_key] = model_results

        print(
            f"  > Occupied cells: {len(model_results['errors'])}, "
            f"Mean {metric_label}: {np.nanmean(model_results['errors']):.3f} {metric_units}, "
            f"Median {metric_label}: {np.nanmedian(model_results['errors']):.3f} {metric_units}, "
            f"Mean bias: {np.nanmean(model_results['signed_biases']):.3f} {VARIABLES['units']}"
        )

    if not results:
        print("[Error] No valid model results found.")
        sys.exit(1)

    # Plot
    valid_models = [m for m in selected_models if m in results]
    n_models = len(valid_models)

    fig, axes = plt.subplots(
        nrows=n_models,
        ncols=2,
        figsize=(12, 2.65 * n_models),
        dpi=150,
        sharey=True
    )

    if n_models == 1:
        axes = np.array([axes])

    all_errors = np.concatenate([
        results[m]["errors"] for m in valid_models
    ])

    y_max = np.nanmax(all_errors) * 1.10

    for row_idx, model_key in enumerate(valid_models):
        model_name = MODELS[model_key]
        data = results[model_key]

        ax_density = axes[row_idx, 0]
        ax_elev = axes[row_idx, 1]

        # Left column: station density
        ax_density.scatter(
            data["station_counts"],
            data["errors"],
            marker=".",
            color="blue",
            s=25
        )

        slope_density = add_trendline(
            ax=ax_density,
            x=data["station_counts"],
            y=data["errors"],
            color="blue"
        )

        ax_density.set_xlim(left=0)
        ax_density.set_ylim(bottom=0, top=y_max)
        ax_density.margins(x=0, y=0)
        ax_density.grid(True, alpha=0.3)

        # Right column: station z0_era5
        ax_elev.scatter(
            data["z0_era5s"],
            data["errors"],
            marker=".",
            color="red",
            s=25
        )

        slope_elev = add_trendline(
            ax=ax_elev,
            x=data["z0_era5s"],
            y=data["errors"],
            color="red"
        )

        ax_elev.set_xlim(left=0)
        ax_elev.set_ylim(bottom=0, top=y_max)
        ax_elev.margins(x=0, y=0)
        ax_elev.grid(True, alpha=0.3)

        # Model label on far left
        ax_density.text(
            -0.33,
            0.5,
            model_name,
            transform=ax_density.transAxes,
            fontsize=12,
            fontweight="bold",
            va="center",
            ha="right",
            rotation=0
        )

        # Trendline slope annotations
        if pd.notna(slope_density):
            if metric == "mape":
                slope_text = f"slope={slope_density:.3f}"
            else:
                slope_text = f"slope={slope_density:.3f}"

            ax_density.text(
                0.03,
                0.90,
                slope_text,
                transform=ax_density.transAxes,
                fontsize=9,
                ha="left",
                va="center"
            )

        if pd.notna(slope_elev):
            slope_per_km = slope_elev * 1000.0

            if metric == "mape":
                slope_text = f"slope={slope_per_km:.3f}"
            else:
                slope_text = f"slope={slope_per_km:.3f}"

            ax_elev.text(
                0.03,
                0.90,
                slope_text,
                transform=ax_elev.transAxes,
                fontsize=9,
                ha="left",
                va="center"
            )

        # Only show x labels on bottom row
        if row_idx == n_models - 1:
            ax_density.set_xlabel("Number of Stations per Grid Cell", fontsize=12)
            ax_elev.set_xlabel("Mean Station z0_era5 AMSL (m)", fontsize=12)
        else:
            ax_density.set_xlabel("")
            ax_elev.set_xlabel("")
            ax_density.tick_params(labelbottom=False)
            ax_elev.tick_params(labelbottom=False)

        ax_density.tick_params(axis="both", labelsize=9)
        ax_elev.tick_params(axis="both", labelsize=9)

    # Column titles
    axes[0, 0].set_title("Station Density", fontsize=13, fontweight="bold", pad=8)
    axes[0, 1].set_title("Station z0_era5", fontsize=13, fontweight="bold", pad=8)

    # Shared labels and title
    fig.suptitle(
        f"72-hr Grid-Cell Wind Speed Error by Model\n"
        f"Initialized Jan {init_day}, Verification Jan 08 UTC | {region}",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )

    fig.supylabel(
        f"Grid-Cell {metric_label} ({metric_units})",
        fontsize=12,
        x=0.45
    )

    plt.tight_layout(rect=[0.22, 0.03, 1, 0.96])
    fig.subplots_adjust(hspace=0.18, wspace=0.12)

    # Save
    out_path = configured_output_path(
        "error_vs_station_z0",
        metric=metric,
        region=region,
    )
    ensure_parent_dir(out_path)

    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    ds_stations.close()

    print(f"\nSaved plot to: {out_path}")


if __name__ == "__main__":
    main()
