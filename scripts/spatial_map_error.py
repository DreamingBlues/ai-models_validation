# Synoptic Observation + Signed Wind Speed Difference Map Generator
# Top panel: station-derived synoptic wind speed grid
# Bottom panels: model forecast - station-averaged observation
# Valid time: 2025-01-08 00:00 UTC
# Forecast init: Jan 05 -> framed as 72-hour lead time
# Abtin Olaee 2026

import argparse
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.path import Path as MplPath
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm
from mpl_toolkits.basemap import Basemap
from scipy.spatial import cKDTree

from viz_config import (
    DEFAULT_INIT_DAY,
    DEFAULT_REGION,
    DEFAULT_VALID_TIME,
    LEAD_DAYS,
    MODELS,
    PATHS,
    REGIONS,
    VARIABLES,
)
from viz_utils import (
    ensure_parent_dir,
    model_path,
    open_dataset_safe,
    output_path as configured_output_path,
    project_path,
    region_mask as configured_region_mask,
)


OBS_MIN = 0.0
OBS_MAX = 16.0
OBS_N_LEVELS = 8
OBS_CMAP = "turbo"

# =============================================================================
# GEOJSON HELPERS
# =============================================================================

def _iter_polygons_from_geojson(geojson_obj):
    def polygon_from_coords(coords):
        exterior = [(float(x), float(y)) for x, y in coords[0]]
        holes = [[(float(x), float(y)) for x, y in ring] for ring in coords[1:]]
        return exterior, holes

    for feat in geojson_obj.get("features", []):
        geom = feat.get("geometry", feat)

        if geom["type"] == "Polygon":
            yield polygon_from_coords(geom["coordinates"])

        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                yield polygon_from_coords(poly)


def geojson_point_mask(lons, lats, geojson_path):
    geojson_path = project_path(geojson_path)

    if not geojson_path.exists():
        print(f"[Warning] GeoJSON not found: {geojson_path}")
        print("[Warning] Continuing without GeoJSON mask.")
        return np.ones(len(lons), dtype=bool)

    with open(geojson_path) as f:
        gj = json.load(f)

    points = np.column_stack((lons, lats))
    mask = np.zeros(len(points), dtype=bool)

    for exterior, holes in _iter_polygons_from_geojson(gj):
        path = MplPath(exterior)
        inside = path.contains_points(points)

        for hole in holes:
            inside &= ~MplPath(hole).contains_points(points)

        mask |= inside

    return mask


# =============================================================================
# BASIC HELPERS
# =============================================================================

def get_time_dim(da):
    for dim in ["time", "valid_time", "datetime", "date_time"]:
        if dim in da.dims:
            return dim

    for coord in ["time", "valid_time", "datetime", "date_time"]:
        if coord in da.coords:
            return coord

    raise ValueError(f"Could not find time dimension in DataArray dims: {da.dims}")


def normalize_datetime_index(values):
    times = pd.to_datetime(values)

    if getattr(times, "tz", None) is not None:
        times = times.tz_localize(None)

    return times


def select_time_exact(da, target_time):
    time_dim = get_time_dim(da)

    times = normalize_datetime_index(da[time_dim].values)
    target = pd.Timestamp(target_time)

    if target.tz is not None:
        target = target.tz_localize(None)

    matches = np.where(times == target)[0]

    if len(matches) == 0:
        available = [str(t) for t in times[:8]]
        raise ValueError(
            f"Target time {target} not found in {time_dim}. "
            f"First available times: {available}"
        )

    return da.isel({time_dim: int(matches[0])})


def scalar_value(value):
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return np.nan
        return float(arr.squeeze())
    except Exception:
        return np.nan


def get_axis_ticks(region, minlon, maxlon, minlat, maxlat):
    if region == "CA":
        lat_step = 2.0
        lon_step = 2.0
    else:
        lat_step = 0.25
        lon_step = 0.25

    lat_start = np.ceil(minlat / lat_step) * lat_step
    lat_end = np.floor(maxlat / lat_step) * lat_step

    lon_start = np.ceil(minlon / lon_step) * lon_step
    lon_end = np.floor(maxlon / lon_step) * lon_step

    lat_ticks = np.arange(lat_start, lat_end + lat_step, lat_step)
    lon_ticks = np.arange(lon_start, lon_end + lon_step, lon_step)

    return lat_ticks, lon_ticks


# =============================================================================
# DATA LOADING
# =============================================================================

def load_station_subset(region):
    station_nc = project_path(PATHS["station"])
    station_var = VARIABLES["station_wind"]

    if not station_nc.exists():
        print(f"[Error] Station NetCDF not found: {station_nc}")
        return None

    ds = open_dataset_safe(station_nc)

    if station_var not in ds:
        print(f"[Error] Station variable not found: {station_var}")
        print(f"Available variables: {list(ds.data_vars.keys())}")
        return None

    lats = ds["latitude"].values
    lons = ds["longitude"].values

    region_mask = configured_region_mask(lats, lons, region)

    geo_mask = geojson_point_mask(
        lons=lons,
        lats=lats,
        geojson_path=PATHS["geojson"],
    )

    final_mask = region_mask & geo_mask
    subset = ds.isel(station=final_mask)

    if subset.sizes["station"] == 0:
        print(f"[Error] No stations found in region {region}.")
        return None

    print(f"Loaded {subset.sizes['station']} stations for {region}")

    return subset


def load_model_dataset(model_key, day):
    fpath = model_path(model_key, day)

    if not fpath.exists():
        print(f"[Warning] Missing model file for {model_key}: {fpath}")
        return None

    ds = open_dataset_safe(fpath)

    if VARIABLES["model_wind"] not in ds:
        raise KeyError(
            f"{VARIABLES['model_wind']} not found in {fpath}. "
            f"Available variables: {list(ds.data_vars.keys())}"
        )

    return ds


def load_reference_model(day, selected_models):
    for model_key in selected_models:
        ds = load_model_dataset(model_key, day)
        if ds is not None:
            print(f"Using reference grid from {MODELS[model_key]}")
            return ds

    return None


# =============================================================================
# GRID COMPUTATION
# =============================================================================

def build_synoptic_observation_grid(ds_ref_model, ds_stations, valid_time):
    """
    Builds the station-derived synoptic observation grid once.

    Each station is assigned to the nearest reference model grid cell.
    All stations inside each occupied grid cell are averaged.
    Empty grid cells remain NaN.
    """

    station_var = VARIABLES["station_wind"]

    obs_slice = select_time_exact(ds_stations[station_var], valid_time)

    grid_lat = ds_ref_model["latitude"].values
    grid_lon = ds_ref_model["longitude"].values

    if grid_lat.ndim == 1 and grid_lon.ndim == 1:
        grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)
    else:
        grid_lat_2d, grid_lon_2d = grid_lat, grid_lon

    flat_lats = grid_lat_2d.ravel()
    flat_lons = grid_lon_2d.ravel()

    grid_points = np.column_stack((flat_lats, flat_lons))
    tree = cKDTree(grid_points)

    st_lats = ds_stations["latitude"].values
    st_lons = ds_stations["longitude"].values
    st_points = np.column_stack((st_lats, st_lons))

    _, grid_indices = tree.query(st_points, k=1)

    obs_flat = np.full(flat_lats.shape, np.nan)

    unique_grid_indices = np.unique(grid_indices)

    for idx in unique_grid_indices:
        member_mask = grid_indices == idx

        obs_vals = obs_slice.isel(station=member_mask).values
        obs_val = np.nanmean(obs_vals)

        if np.isfinite(obs_val):
            obs_flat[idx] = obs_val

    obs_grid = obs_flat.reshape(grid_lat_2d.shape)

    valid_grid_indices = np.where(np.isfinite(obs_flat))[0]
    valid_obs = obs_flat[valid_grid_indices]

    print(f"Synoptic occupied grid cells: {len(valid_grid_indices)}")

    return {
        "obs_grid": obs_grid,
        "obs_flat": obs_flat,
        "grid_lat_2d": grid_lat_2d,
        "grid_lon_2d": grid_lon_2d,
        "flat_lats": flat_lats,
        "flat_lons": flat_lons,
        "valid_grid_indices": valid_grid_indices,
        "valid_obs": valid_obs,
        "station_lats": st_lats,
        "station_lons": st_lons,
    }


def compute_model_difference_on_reference_grid(ds_model, obs_cache, valid_time):
    """
    Computes signed difference on the reference observation grid.

        difference = model forecast - station-derived observation

    Only grid cells with valid observations are evaluated.
    """

    model_var = VARIABLES["model_wind"]

    model_slice = select_time_exact(ds_model[model_var], valid_time)

    diff_flat = np.full(obs_cache["obs_flat"].shape, np.nan)

    for idx in obs_cache["valid_grid_indices"]:
        obs_val = obs_cache["obs_flat"][idx]

        if not np.isfinite(obs_val):
            continue

        lat_val = obs_cache["flat_lats"][idx]
        lon_val = obs_cache["flat_lons"][idx]

        model_point = model_slice.sel(
            latitude=lat_val,
            longitude=lon_val,
            method="nearest"
        )

        model_val = scalar_value(model_point.values)

        if not np.isfinite(model_val):
            continue

        diff_flat[idx] = model_val - obs_val

    diff_grid = diff_flat.reshape(obs_cache["grid_lat_2d"].shape)
    valid_diffs = diff_flat[np.isfinite(diff_flat)]

    return {
        "diff_grid": diff_grid,
        "valid_diffs": valid_diffs,
        "n_valid_cells": len(valid_diffs),
    }


def compute_all_grids(region, day, valid_time, selected_models):
    ds_stations = load_station_subset(region)

    if ds_stations is None:
        sys.exit(1)

    ds_ref_model = load_reference_model(day, selected_models)

    if ds_ref_model is None:
        print("[Error] No model files found for reference grid.")
        sys.exit(1)

    obs_cache = build_synoptic_observation_grid(
        ds_ref_model=ds_ref_model,
        ds_stations=ds_stations,
        valid_time=valid_time
    )

    results = {}

    for model_key in selected_models:
        model_name = MODELS[model_key]
        print(f"\nProcessing {model_name}")

        ds_model = load_model_dataset(model_key, day)

        if ds_model is None:
            continue

        try:
            result = compute_model_difference_on_reference_grid(
                ds_model=ds_model,
                obs_cache=obs_cache,
                valid_time=valid_time
            )

            results[model_key] = result

            if result["n_valid_cells"] > 0:
                mean_diff = np.nanmean(result["valid_diffs"])
                max_abs = np.nanmax(np.abs(result["valid_diffs"]))
                print(f"  Valid grid cells: {result['n_valid_cells']}")
                print(f"  Mean difference: {mean_diff:.3f} {VARIABLES['units']}")
                print(f"  Max abs difference: {max_abs:.3f} {VARIABLES['units']}")
            else:
                print("  No valid grid cells found.")

        except Exception as e:
            print(f"  Skipped due to error: {e}")

        ds_model.close()

    ds_ref_model.close()
    ds_stations.close()

    if not results:
        print("[Error] No model grids were created.")
        sys.exit(1)

    return obs_cache, results


# =============================================================================
# PLOTTING
# =============================================================================

def setup_basemap(ax, region):
    minlon, maxlon, minlat, maxlat = REGIONS[region]

    m = Basemap(
        projection="merc",
        epsg=4326,
        llcrnrlon=minlon,
        llcrnrlat=minlat,
        urcrnrlon=maxlon,
        urcrnrlat=maxlat,
        resolution="i",
        ax=ax
    )

    try:
        m.arcgisimage(
            server="http://server.arcgisonline.com/arcgis",
            service="World_Shaded_Relief",
            xpixels=1200,
            verbose=False,
        )
    except Exception:
        m.shadedrelief(scale=0.35)

    return m


def draw_boundaries_and_ticks(m, region, row=None, col=None, top_panel=False):
    minlon, maxlon, minlat, maxlat = REGIONS[region]

    lat_ticks, lon_ticks = get_axis_ticks(
        region=region,
        minlon=minlon,
        maxlon=maxlon,
        minlat=minlat,
        maxlat=maxlat
    )

    m.drawcoastlines(color="0.20", linewidth=0.6, zorder=15)
    m.drawstates(color="0.35", linewidth=0.5, zorder=15)
    m.drawcountries(color="0.35", linewidth=0.5, zorder=15)

    if top_panel:
        lat_labels = [1, 0, 0, 0]
        lon_labels = [0, 0, 0, 1]
    else:
        lat_labels = [1 if col == 0 else 0, 0, 0, 0]
        lon_labels = [0, 0, 0, 1 if row == 2 else 0]

    m.drawparallels(
        lat_ticks,
        labels=lat_labels,
        fontsize=9,
        color=(0, 0, 0, 0),
        textcolor="black",
        linewidth=0.001,
        dashes=[1, 1],
    )

    m.drawmeridians(
        lon_ticks,
        labels=lon_labels,
        fontsize=9,
        color=(0, 0, 0, 0),
        textcolor="black",
        linewidth=0.001,
        dashes=[1, 1],
    )


def make_observation_colormap():
    cmap = plt.get_cmap(OBS_CMAP).copy()
    cmap.set_bad((1, 1, 1, 0))
    cmap.set_under(cmap(0.0))
    cmap.set_over(cmap(1.0))

    levels = np.linspace(OBS_MIN, OBS_MAX, OBS_N_LEVELS + 1)

    norm = BoundaryNorm(
        levels,
        ncolors=cmap.N,
        clip=False
    )

    return cmap, norm, levels


def make_difference_colormap(diff_limit):
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad((1, 1, 1, 0))

    norm = TwoSlopeNorm(
        vmin=-diff_limit,
        vcenter=0.0,
        vmax=diff_limit
    )

    return cmap, norm

def get_relative_diff_limit(results):
    all_diffs = []

    for result in results.values():
        vals = result.get("valid_diffs", np.array([]))
        vals = vals[np.isfinite(vals)]

        if len(vals) > 0:
            all_diffs.append(vals)

    if not all_diffs:
        return 1.0

    all_diffs = np.concatenate(all_diffs)

    max_abs = np.nanmax(np.abs(all_diffs))

    if not np.isfinite(max_abs) or max_abs == 0:
        return 1.0

    return max_abs


def map_lead_label(day, valid_time):
    init_time = pd.Timestamp(f"2025-01-{int(day):02d} 00:00:00")
    valid_timestamp = pd.Timestamp(valid_time)
    lead_hours = int((valid_timestamp - init_time).total_seconds() // 3600)
    return f"{lead_hours}-hour Lead Time"


def plot_dashboard(
    obs_cache,
    results,
    region,
    day,
    valid_time,
    selected_models,
    show_stations,
):
    obs_cmap, obs_norm, obs_levels = make_observation_colormap()
    diff_limit = get_relative_diff_limit(results)
    diff_cmap, diff_norm = make_difference_colormap(diff_limit)

    fig = plt.figure(figsize=(11, 15), dpi=150)

    gs = fig.add_gridspec(
        nrows=4,
        ncols=2,
        height_ratios=[1.05, 1.0, 1.0, 1.0],
        hspace=0.28,
        wspace=0.08
    )

    # -------------------------------------------------------------------------
    # Top synoptic observation panel
    # -------------------------------------------------------------------------
    ax_obs = fig.add_subplot(gs[0, :])
    m_obs = setup_basemap(ax_obs, region)

    x_obs_grid, y_obs_grid = m_obs(
        obs_cache["grid_lon_2d"],
        obs_cache["grid_lat_2d"]
    )

    obs_hm = m_obs.pcolormesh(
        x_obs_grid,
        y_obs_grid,
        obs_cache["obs_grid"],
        cmap=obs_cmap,
        norm=obs_norm,
        shading="auto",
        alpha=0.82,
        zorder=5
    )

    if show_stations:
        x_st, y_st = m_obs(
            obs_cache["station_lons"],
            obs_cache["station_lats"]
        )

        m_obs.scatter(
            x_st,
            y_st,
            c="0.15",
            s=5,
            marker=".",
            alpha=0.45,
            linewidths=0,
            zorder=12
        )

    draw_boundaries_and_ticks(m_obs, region, top_panel=True)

    ax_obs.set_title(
        "Station-Derived Observed Wind Speed",
        fontsize=14,
        fontweight="bold"
    )

    cb_obs = fig.colorbar(
        obs_hm,
        ax=ax_obs,
        orientation="vertical",
        fraction=0.025,
        pad=0.015,
        extend="max",
        boundaries=obs_levels,
        ticks=obs_levels
    )

    cb_obs.set_label(
        f"Observed Wind Speed ({VARIABLES['units']})",
        fontsize=11
    )

    cb_obs.set_ticks(obs_levels)

    # -------------------------------------------------------------------------
    # Bottom model difference panels
    # -------------------------------------------------------------------------
    model_axes = []
    last_diff_hm = None

    for panel_idx, model_key in enumerate(selected_models):
        row = panel_idx // 2
        col = panel_idx % 2

        ax = fig.add_subplot(gs[row + 1, col])
        model_axes.append(ax)

        model_name = MODELS[model_key]

        m = setup_basemap(ax, region)

        if model_key in results:
            result = results[model_key]

            x_grid, y_grid = m(
                obs_cache["grid_lon_2d"],
                obs_cache["grid_lat_2d"]
            )

            last_diff_hm = m.pcolormesh(
                x_grid,
                y_grid,
                result["diff_grid"],
                cmap=diff_cmap,
                norm=diff_norm,
                shading="auto",
                alpha=0.78,
                zorder=5
            )

            if show_stations:
                x_st, y_st = m(
                    obs_cache["station_lons"],
                    obs_cache["station_lats"]
                )

                m.scatter(
                    x_st,
                    y_st,
                    c="0.15",
                    s=4,
                    marker=".",
                    alpha=0.35,
                    linewidths=0,
                    zorder=12
                )

        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12
            )

        draw_boundaries_and_ticks(
            m=m,
            region=region,
            row=row,
            col=col,
            top_panel=False
        )

        ax.set_title(
            model_name,
            fontsize=13,
            fontweight="bold"
        )

    if last_diff_hm is not None:
        cb_diff = fig.colorbar(
            last_diff_hm,
            ax=model_axes,
            orientation="vertical",
            fraction=0.035,
            pad=0.025,
            extend="both"
        )

        cb_diff.set_label(
            f"Forecast - Observation Wind Speed Difference ({VARIABLES['units']})",
            fontsize=12
        )

    valid_str = pd.Timestamp(valid_time).strftime("%Y-%m-%d %H:%M UTC")
    lead_label = map_lead_label(day, valid_time)

    fig.suptitle(
        f"Station-Derived Wind Speed and {lead_label} Forecast Difference\n"
        f"Valid: {valid_str} | Forecast initialized: Jan {int(day):02d}, 2025",
        fontsize=16,
        fontweight="bold",
        y=0.985
    )

    time_tag = pd.Timestamp(valid_time).strftime("%Y%m%d_%H%M")
    out_path = configured_output_path(
        "spatial_map",
        lead_label=lead_label.replace(" ", "_"),
        region=region,
        day=day,
        time_tag=time_tag,
    )
    ensure_parent_dir(out_path)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create synoptic observation and signed wind speed difference maps."
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
        "--plot-time",
        type=str,
        default=DEFAULT_VALID_TIME,
        help="Valid time to plot. Default: 2025-01-08 00:00:00"
    )


    args = parser.parse_args()

    print(f"Region: {args.region}")
    print(f"Forecast day file: Day{args.init_day}")
    print(f"Valid time: {args.plot_time}")
    print("Difference: forecast - observation")

    obs_cache, results = compute_all_grids(
        region=args.region,
        day=args.init_day,
        valid_time=args.plot_time,
        selected_models=args.models,
    )

    plot_dashboard(
        obs_cache=obs_cache,
        results=results,
        region=args.region,
        day=args.init_day,
        valid_time=args.plot_time,
        selected_models=args.models,
        show_stations=True
    )


if __name__ == "__main__":
    main()
