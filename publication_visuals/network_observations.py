# Grid-cell-weighted station observation comparison by network
# Plots observed 10-m wind speed only, split by station network
# Abtin Olaee 2026

import argparse
import sys

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter, DayLocator
from matplotlib.lines import Line2D

from viz_config import (
    DEFAULT_REGIONS,
    PATHS,
    PLOT_WINDOW,
    REGIONS,
    VARIABLES,
)
from viz_utils import (
    clean_time_index,
    detect_network_var,
    ensure_parent_dir,
    model_path,
    network_colors_by_name,
    network_label,
    open_dataset_safe,
    output_path as configured_output_path,
    project_path,
    region_mask as configured_region_mask,
    trim_to_plot_window,
)


REFERENCE_GRID_MODEL = "ifs"
REFERENCE_GRID_DAY = "05"
NETWORK_VAR = None
NETWORK_COLORS_BY_NAME = network_colors_by_name()

# =============================================================================
# HELPERS
# =============================================================================


def load_reference_grid():
    """
    Uses a model file only to get the latitude/longitude grid.
    If the file is missing, falls back to 0.25-degree rounded cells.
    """

    path = model_path(REFERENCE_GRID_MODEL, REFERENCE_GRID_DAY)

    if not path.exists():
        print(f"[Warning] Reference grid file not found: {path}")
        print("[Warning] Falling back to rounded 0.25-degree grid cells.")
        return None, None

    ds = open_dataset_safe(path)

    if "latitude" not in ds or "longitude" not in ds:
        print("[Warning] Reference grid missing latitude/longitude.")
        print("[Warning] Falling back to rounded 0.25-degree grid cells.")
        return None, None

    grid_lats = np.asarray(ds["latitude"].values)
    grid_lons = np.asarray(ds["longitude"].values)

    return grid_lats, grid_lons


def assign_grid_cells(station_lats, station_lons, grid_lats=None, grid_lons=None):
    """
    Assign each station to a model grid cell.

    If reference grid is available:
        cell_id = nearest model latitude index + nearest model longitude index

    If reference grid is unavailable:
        cell_id = rounded 0.25-degree lat/lon
    """

    station_lats = np.asarray(station_lats)
    station_lons = np.asarray(station_lons)

    if grid_lats is not None and grid_lons is not None:
        lat_idx = np.abs(grid_lats[:, None] - station_lats[None, :]).argmin(axis=0)
        lon_idx = np.abs(grid_lons[:, None] - station_lons[None, :]).argmin(axis=0)

        return np.array([f"{i}_{j}" for i, j in zip(lat_idx, lon_idx)])

    rounded_lats = np.round(station_lats / 0.25) * 0.25
    rounded_lons = np.round(station_lons / 0.25) * 0.25

    return np.array([f"{lat:.3f}_{lon:.3f}" for lat, lon in zip(rounded_lats, rounded_lons)])


def grid_cell_weighted_mean(subset, grid_lats=None, grid_lons=None):
    """
    Grid-cell-weighted mean observation.

    Step 1: Assign each station to a model grid cell.
    Step 2: Average all stations inside each grid cell.
    Step 3: Average occupied grid cells equally.

    This prevents dense station clusters from dominating the regional mean.
    """

    var_name = VARIABLES["station_wind"]

    station_lats = subset["latitude"].values
    station_lons = subset["longitude"].values

    cell_ids = assign_grid_cells(
        station_lats=station_lats,
        station_lons=station_lons,
        grid_lats=grid_lats,
        grid_lons=grid_lons,
    )

    unique_cells = np.unique(cell_ids)

    cell_series = []

    for cell in unique_cells:
        station_mask = cell_ids == cell

        if not np.any(station_mask):
            continue

        cell_mean = subset[var_name].isel(station=station_mask).mean(
            dim="station",
            skipna=True,
        )

        cell_series.append(cell_mean)

    if not cell_series:
        return None, 0, 0

    gridcell_da = xr.concat(cell_series, dim="grid_cell")

    regional_mean = gridcell_da.mean(dim="grid_cell", skipna=True)

    series = regional_mean.to_series()
    series = clean_time_index(series)
    series = trim_to_plot_window(series)

    n_stations = subset.sizes["station"]
    n_grid_cells = len(unique_cells)

    return series, n_stations, n_grid_cells


# =============================================================================
# DATA LOADING
# =============================================================================

def build_observation_data(regions):
    nc_path = project_path(PATHS["station"])

    if not nc_path.exists():
        print(f"[Error] Station file not found: {nc_path}")
        sys.exit(1)

    ds = open_dataset_safe(nc_path)

    var_name = VARIABLES["station_wind"]

    if var_name not in ds:
        raise KeyError(f"{var_name} not found in station dataset.")

    network_var = detect_network_var(ds, configured_name=NETWORK_VAR)

    grid_lats, grid_lons = load_reference_grid()

    data = {}
    ymax = 0.0

    for region in regions:
        region_mask = configured_region_mask(
            ds.latitude.values,
            ds.longitude.values,
            region,
        )

        region_ds = ds.isel(station=region_mask)

        if region_ds.sizes["station"] == 0:
            print(f"[Warning] No stations found in {region}")
            continue

        network_values = region_ds[network_var].values
        unique_networks = pd.unique(network_values)

        data[region] = {}

        print(f"\n{region}: {region_ds.sizes['station']} total stations")

        for net_value in unique_networks:
            if pd.isna(net_value):
                continue

            net_mask = network_values == net_value
            net_ds = region_ds.isel(station=net_mask)

            label = network_label(net_value)

            series, n_stations, n_grid_cells = grid_cell_weighted_mean(
                subset=net_ds,
                grid_lats=grid_lats,
                grid_lons=grid_lons,
            )

            if series is None or series.empty:
                continue

            data[region][label] = {
                "series": series,
                "n_stations": n_stations,
                "n_grid_cells": n_grid_cells,
            }

            local_max = np.nanmax(series.values)
            if np.isfinite(local_max):
                ymax = max(ymax, local_max)

            print(
                f"  {label}: "
                f"{n_stations} stations, {n_grid_cells} occupied grid cells"
            )

    return data, ymax


# =============================================================================
# PLOTTING
# =============================================================================

def plot_observations_by_network(data, ymax, regions):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(regions),
        figsize=(12, 4.8),
        dpi=350,
        sharey=False,
    )

    if len(regions) == 1:
        axes = [axes]

    legend_handles = {}

    for ax, region in zip(axes, regions):
        if region not in data:
            ax.axis("off")
            continue

        for network_label, info in data[region].items():
            series = info["series"]

            color = NETWORK_COLORS_BY_NAME.get(network_label, None)

            line, = ax.plot(
                series.index,
                series.values,
                linewidth=2.0,
                color=color,
                label=network_label,
            )

            legend_handles[network_label] = line

        ax.set_title(region, fontsize=14, fontweight="bold")
        ax.set_xlim(
            pd.Timestamp(PLOT_WINDOW["start"]),
            pd.Timestamp(PLOT_WINDOW["end"]),
        )

        region_ymax = 0.0

        for network_label, info in data[region].items():
            series = info["series"]
            local_max = np.nanmax(series.values)

            if np.isfinite(local_max):
                region_ymax = max(region_ymax, local_max)

        if region_ymax > 0:
            ax.set_ylim(0, region_ymax * 1.15)

        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_locator(DayLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))

        ax.set_xlabel("Date (UTC)", fontsize=12)

    axes[0].set_ylabel(
        f"Grid-Cell-Weighted Mean Wind Speed ({VARIABLES['units']})",
        fontsize=12,
    )

    fig.suptitle(
        "10-m Station Observations by Network\nGrid-Cell-Weighted Regional Mean",
        fontsize=16,
        fontweight="bold",
        y=1.03,
    )

    fig.legend(
        handles=list(legend_handles.values()),
        labels=list(legend_handles.keys()),
        loc="lower center",
        ncol=max(1, len(legend_handles)),
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.autofmt_xdate(rotation=35, ha="right")

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.82,
        bottom=0.20,
        wspace=0.15,
    )

    region_tag = "_".join(regions)
    out_path = configured_output_path("network_observations", region=region_tag)
    ensure_parent_dir(out_path)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot grid-cell-weighted station observations by network."
    )

    parser.add_argument(
        "--regions",
        nargs="+",
        type=str,
        default=DEFAULT_REGIONS,
        choices=list(REGIONS.keys()),
        help="Regions to process. Default: CA LA."
    )

    args = parser.parse_args()

    regions = args.regions

    data, ymax = build_observation_data(regions)

    if not data:
        print("[Error] No observation data loaded. Exiting.")
        sys.exit(1)

    plot_observations_by_network(
        data=data,
        ymax=ymax,
        regions=regions,
    )


if __name__ == "__main__":
    main()
