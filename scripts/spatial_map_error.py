# Figure 4. Time-averaged station-derived 10-m wind speed and forecast bias over the January 7–10, 2025 Los Angeles
# The top row show the mean observed wind-speed field, the mean grid cell elevation, and the mean grid cell TRI
# The lower panels show the mean signed bias for each model

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, Normalize, TwoSlopeNorm
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox
from mpl_toolkits.basemap import Basemap
from rasterio.windows import Window
from rasterio.warp import transform as warp_transform
from rasterio.warp import transform_bounds
from scipy.spatial import cKDTree

from viz_config import (
    DEFAULT_INIT_DAY,
    DEFAULT_REGION,
    LEAD_DAYS,
    MODELS,
    PATHS,
    PLOT_WINDOW,
    REGIONS,
    VARIABLES,
)
from viz_utils import (
    ensure_parent_dir,
    model_path,
    open_dataset_safe,
    project_path,
)


# CONFIGURATION

OBS_MIN = 0.0
OBS_MAX = 12.0
OBS_N_LEVELS = 8
OBS_CMAP = "turbo"

ELEVATION_CMAP = "terrain"
TRI_CMAP = "YlOrBr"

DEM_DIR_DEFAULT = "./data/dem_tiles/"
DEM_BLOCK_SIZE = 1024
TERRAIN_CACHE_DIR = "./data/terrain_cache/"

TIME_COORDINATES = ("time", "valid_time", "datetime", "date_time")
FIGURE_COLUMNS = 3

COLORBAR_WIDTH = 0.015
COLORBAR_PAD = 0.015
BIAS_COLORBAR_HEIGHT_FRACTION = 0.72
TOP_COLORBAR_HEIGHT = 0.012
TOP_COLORBAR_PAD = 0.012


# LONGITUDE AND COORDINATE HELPERS

def to_signed_longitude(values):
    """Convert longitude values to the -180..180 convention."""
    values = np.asarray(values, dtype=float)
    return ((values + 180.0) % 360.0) - 180.0


def to_360_longitude(values):
    """Convert longitude values to the 0..360 convention."""
    return np.mod(np.asarray(values, dtype=float), 360.0)


def align_longitudes(values, reference_values):
    """Convert longitudes to the convention used by a reference array."""
    reference = np.asarray(reference_values, dtype=float)
    finite_reference = reference[np.isfinite(reference)]

    if finite_reference.size == 0:
        raise ValueError("Cannot determine longitude convention from empty values")

    if np.nanmedian(finite_reference) > 180.0:
        return to_360_longitude(values)

    return to_signed_longitude(values)


def format_coordinate(value: float) -> str:
    return f"{float(value):g}"


def format_latitude_tick(value: float) -> str:
    return format_coordinate(abs(value))


def format_longitude_tick(value: float) -> str:
    signed_longitude = float(to_signed_longitude(value))
    return format_coordinate(abs(signed_longitude))


def longitude_convention(values) -> str:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "unknown"
    return "0-360" if np.nanmedian(finite) > 180.0 else "-180 to 180"


def region_coordinate_mask(lats, lons, region):
    """Mask points using region bounds regardless of longitude convention."""
    minlon, maxlon, minlat, maxlat = REGIONS[region]
    aligned_lons = align_longitudes(lons, [minlon, maxlon])

    lat_mask = (lats >= minlat) & (lats <= maxlat)

    if minlon <= maxlon:
        lon_mask = (aligned_lons >= minlon) & (aligned_lons <= maxlon)
    else:
        lon_mask = (aligned_lons >= minlon) | (aligned_lons <= maxlon)

    return lat_mask & lon_mask


def get_axis_ticks(region):
    minlon, maxlon, minlat, maxlat = REGIONS[region]
    step = 2.0 if region == "CA" else 0.25

    lat_start = np.ceil(minlat / step) * step
    lat_end = np.floor(maxlat / step) * step
    lon_start = np.ceil(minlon / step) * step
    lon_end = np.floor(maxlon / step) * step

    lat_ticks = np.arange(lat_start, lat_end + step, step)
    lon_ticks = np.arange(lon_start, lon_end + step, step)
    return lat_ticks, lon_ticks


# GEOJSON HELPERS

def iter_geojson_polygons(geojson_obj):
    def parse_polygon(coords):
        exterior = [(float(x), float(y)) for x, y in coords[0]]
        holes = [
            [(float(x), float(y)) for x, y in ring]
            for ring in coords[1:]
        ]
        return exterior, holes

    for feature in geojson_obj.get("features", []):
        geometry = feature.get("geometry", feature)
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates", [])

        if geometry_type == "Polygon":
            yield parse_polygon(coordinates)
        elif geometry_type == "MultiPolygon":
            for polygon in coordinates:
                yield parse_polygon(polygon)


def geojson_point_mask(lons, lats, geojson_path):
    """Return points inside the GeoJSON polygon using signed longitudes."""
    path = project_path(geojson_path)

    if not path.exists():
        print(f"[Warning] GeoJSON not found: {path}")
        print("[Warning] Continuing without a GeoJSON mask.")
        return np.ones(len(lons), dtype=bool)

    with path.open(encoding="utf-8") as file:
        geojson_obj = json.load(file)

    points = np.column_stack((to_signed_longitude(lons), lats))
    mask = np.zeros(len(points), dtype=bool)

    for exterior, holes in iter_geojson_polygons(geojson_obj):
        inside = MplPath(exterior).contains_points(points)
        for hole in holes:
            inside &= ~MplPath(hole).contains_points(points)
        mask |= inside

    return mask


def geojson_grid_mask(lons_2d, lats_2d, geojson_path):
    """Return a mask for model-cell centers inside the California polygon."""
    path = project_path(geojson_path)

    if not path.exists():
        return np.ones(np.asarray(lons_2d).shape, dtype=bool)

    with path.open(encoding="utf-8") as file:
        geojson_obj = json.load(file)

    points = np.column_stack(
        (
            to_signed_longitude(lons_2d).ravel(),
            np.asarray(lats_2d, dtype=float).ravel(),
        )
    )

    mask = np.zeros(len(points), dtype=bool)

    for exterior, holes in iter_geojson_polygons(geojson_obj):
        inside = MplPath(exterior).contains_points(points)
        for hole in holes:
            inside &= ~MplPath(hole).contains_points(points)
        mask |= inside

    return mask.reshape(np.asarray(lons_2d).shape)


# TIME HELPERS

def get_time_dimension(data_array):
    for name in TIME_COORDINATES:
        if name in data_array.dims or name in data_array.coords:
            return name

    raise ValueError(
        f"Could not identify a time coordinate in {data_array.dims}"
    )


def normalize_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize(None) if timestamp.tz is not None else timestamp


def normalize_datetime_index(values) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex(pd.to_datetime(values))
    return times.tz_localize(None) if times.tz is not None else times


def select_time_period(data_array, start_time, end_time):
    """Select timestamps in the inclusive-start, exclusive-end interval."""
    time_dimension = get_time_dimension(data_array)
    times = normalize_datetime_index(data_array[time_dimension].values)
    start = normalize_timestamp(start_time)
    end = normalize_timestamp(end_time)

    if end <= start:
        raise ValueError("end_time must be later than start_time")

    indices = np.flatnonzero((times >= start) & (times < end))
    if indices.size == 0:
        sample = ", ".join(str(time) for time in times[:8])
        raise ValueError(
            f"No timestamps found from {start} to {end} (end exclusive). "
            f"First available timestamps: {sample}"
        )

    return data_array.isel({time_dimension: indices})


def dataarray_to_series(data_array) -> pd.Series:
    """Convert a one-value-per-time DataArray into a sorted Series."""
    time_dimension = get_time_dimension(data_array)

    invalid_dimensions = [
        dimension
        for dimension in data_array.dims
        if dimension != time_dimension and data_array.sizes[dimension] != 1
    ]
    if invalid_dimensions:
        raise ValueError(
            "Expected one value per timestamp; non-singleton dimensions remain: "
            f"{invalid_dimensions}"
        )

    for dimension in list(data_array.dims):
        if dimension != time_dimension and data_array.sizes[dimension] == 1:
            data_array = data_array.isel({dimension: 0}, drop=True)

    times = normalize_datetime_index(data_array[time_dimension].values)
    values = np.asarray(data_array.values, dtype=float).reshape(-1)

    if len(times) != len(values):
        raise ValueError("Time coordinate and data values have different lengths")

    return pd.Series(values, index=times).groupby(level=0).mean().sort_index()


# DATA LOADING

def load_station_subset(region):
    path = project_path(PATHS["station"])
    variable = VARIABLES["station_wind"]

    if not path.exists():
        raise FileNotFoundError(f"Station NetCDF not found: {path}")

    dataset = open_dataset_safe(path)

    if variable not in dataset:
        available_variables = list(dataset.data_vars)
        dataset.close()
        raise KeyError(
            f"Station variable '{variable}' not found. "
            f"Available variables: {available_variables}"
        )

    lats = np.asarray(dataset["latitude"].values, dtype=float)
    lons = np.asarray(dataset["longitude"].values, dtype=float)

    mask = region_coordinate_mask(lats, lons, region)
    mask &= geojson_point_mask(lons, lats, PATHS["geojson"])

    subset = dataset.isel(station=mask)
    if subset.sizes.get("station", 0) == 0:
        dataset.close()
        raise ValueError(f"No stations found in region {region}")

    print(
        f"Loaded {subset.sizes['station']} stations for {region} "
        f"(longitude convention: {longitude_convention(lons)})"
    )
    return subset


def load_model_dataset(model_key, day):
    path = model_path(model_key, day)

    if not path.exists():
        print(f"[Warning] Missing model file for {model_key}: {path}")
        return None

    dataset = open_dataset_safe(path)
    variable = VARIABLES["model_wind"]

    if variable not in dataset:
        available_variables = list(dataset.data_vars)
        dataset.close()
        raise KeyError(
            f"'{variable}' not found in {path}. "
            f"Available variables: {available_variables}"
        )

    return dataset


def load_reference_model(day, selected_models):
    for model_key in selected_models:
        dataset = load_model_dataset(model_key, day)
        if dataset is not None:
            print(f"Using reference grid from {MODELS[model_key]}")
            return dataset

    raise FileNotFoundError("No model files were available for a reference grid")


# OBSERVATION AND MODEL-BIAS GRID COMPUTATION

def build_observation_grid(
    reference_dataset,
    station_dataset,
    start_time,
    end_time,
):
    """Build a time-averaged station observation field on the model grid."""
    observation_period = select_time_period(
        station_dataset[VARIABLES["station_wind"]],
        start_time,
        end_time,
    )

    time_dimension = get_time_dimension(observation_period)
    period_times = normalize_datetime_index(
        observation_period[time_dimension].values
    )

    grid_lats = np.asarray(reference_dataset["latitude"].values, dtype=float)
    grid_lons = np.asarray(reference_dataset["longitude"].values, dtype=float)

    if grid_lats.ndim == 1 and grid_lons.ndim == 1:
        grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lons, grid_lats)
    else:
        grid_lat_2d, grid_lon_2d = grid_lats, grid_lons

    flat_lats = grid_lat_2d.ravel()
    flat_lons = grid_lon_2d.ravel()

    station_lats = np.asarray(station_dataset["latitude"].values, dtype=float)
    station_lons_raw = np.asarray(
        station_dataset["longitude"].values,
        dtype=float,
    )

    station_lons_grid = align_longitudes(station_lons_raw, flat_lons)
    grid_points = np.column_stack((flat_lats, flat_lons))
    station_points = np.column_stack((station_lats, station_lons_grid))
    _, grid_indices = cKDTree(grid_points).query(station_points, k=1)

    observation_flat = np.full(flat_lats.shape, np.nan)
    observation_series_by_grid = {}

    for grid_index in np.unique(grid_indices):
        station_mask = grid_indices == grid_index
        cell_values = observation_period.isel(station=station_mask).mean(
            dim="station",
            skipna=True,
        )
        cell_series = dataarray_to_series(cell_values).dropna()

        if cell_series.empty:
            continue

        grid_index = int(grid_index)
        observation_series_by_grid[grid_index] = cell_series
        observation_flat[grid_index] = float(cell_series.mean())

    valid_grid_indices = np.array(
        sorted(observation_series_by_grid),
        dtype=int,
    )

    print(
        f"Reference-grid longitude convention: "
        f"{longitude_convention(flat_lons)}"
    )
    print(f"Time-averaged occupied grid cells: {len(valid_grid_indices)}")

    return {
        "obs_grid": observation_flat.reshape(grid_lat_2d.shape),
        "obs_flat": observation_flat,
        "obs_series_by_grid": observation_series_by_grid,
        "grid_lat_2d": grid_lat_2d,
        "grid_lon_2d": grid_lon_2d,
        "flat_lats": flat_lats,
        "flat_lons": flat_lons,
        "valid_grid_indices": valid_grid_indices,
        "station_lats": station_lats,
        "station_lons_map": station_lons_grid,
        "period_first_time": period_times.min(),
        "period_last_time": period_times.max(),
    }


def compute_model_bias_grid(
    model_dataset,
    observation_cache,
    start_time,
    end_time,
):
    """Compute mean signed bias at each occupied reference grid cell."""
    model_period = select_time_period(
        model_dataset[VARIABLES["model_wind"]],
        start_time,
        end_time,
    )

    bias_flat = np.full(observation_cache["obs_flat"].shape, np.nan)
    paired_timestamp_counts = []

    for grid_index in observation_cache["valid_grid_indices"]:
        observation_series = observation_cache["obs_series_by_grid"][
            int(grid_index)
        ]
        latitude = observation_cache["flat_lats"][grid_index]
        reference_longitude = observation_cache["flat_lons"][grid_index]
        model_longitude = float(
            align_longitudes(
                [reference_longitude],
                model_dataset["longitude"].values,
            )[0]
        )

        model_point = model_period.sel(
            latitude=latitude,
            longitude=model_longitude,
            method="nearest",
        )
        model_series = dataarray_to_series(model_point)
        common_times = observation_series.index.intersection(model_series.index)

        if common_times.empty:
            continue

        observations = observation_series.loc[common_times].to_numpy(float)
        forecasts = model_series.loc[common_times].to_numpy(float)
        valid = np.isfinite(observations) & np.isfinite(forecasts)

        if not np.any(valid):
            continue

        bias = np.mean(forecasts[valid] - observations[valid])
        if np.isfinite(bias):
            bias_flat[grid_index] = float(bias)
            paired_timestamp_counts.append(int(valid.sum()))

    valid_biases = bias_flat[np.isfinite(bias_flat)]
    return {
        "diff_grid": bias_flat.reshape(observation_cache["grid_lat_2d"].shape),
        "valid_diffs": valid_biases,
        "n_valid_cells": len(valid_biases),
        "n_time_pairs": np.asarray(paired_timestamp_counts, dtype=int),
    }


def compute_all_grids(region, day, start_time, end_time, selected_models):
    station_dataset = load_station_subset(region)

    try:
        reference_dataset = load_reference_model(day, selected_models)
        try:
            observation_cache = build_observation_grid(
                reference_dataset,
                station_dataset,
                start_time,
                end_time,
            )
        finally:
            reference_dataset.close()

        results = {}

        for model_key in selected_models:
            print(f"\nProcessing {MODELS[model_key]}")
            model_dataset = load_model_dataset(model_key, day)
            if model_dataset is None:
                continue

            result = compute_model_bias_grid(
                model_dataset,
                observation_cache,
                start_time,
                end_time,
            )
            model_dataset.close()

            results[model_key] = result

            if result["n_valid_cells"] == 0:
                print("  No valid grid cells found.")
                continue

            print(f"  Valid grid cells: {result['n_valid_cells']}")
            print(
                f"  Mean time-averaged bias: "
                f"{np.nanmean(result['valid_diffs']):.3f} {VARIABLES['units']}"
            )

        if not results:
            raise RuntimeError("No model grids were created")

        return observation_cache, results
    finally:
        station_dataset.close()


# DEM / TERRAIN COMPUTATION

def centers_to_edges(centers):
    centers = np.asarray(centers, dtype=float)

    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("At least two 1-D grid centers are required")

    if not np.all(np.diff(centers) > 0):
        raise ValueError("Grid centers must be strictly increasing")

    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def build_reference_terrain_grid(observation_cache, region):
    """Create an accumulator for terrain on the observation/reference grid."""
    grid_lat_2d = np.asarray(observation_cache["grid_lat_2d"], dtype=float)
    grid_lon_2d = np.asarray(observation_cache["grid_lon_2d"], dtype=float)

    if grid_lat_2d.ndim != 2 or grid_lon_2d.ndim != 2:
        raise ValueError("Reference latitude/longitude grids must be 2-D")

    # The processed model grids are rectilinear; recover their 1-D axes.
    lat_axis = grid_lat_2d[:, 0]
    lon_axis = grid_lon_2d[0, :]

    minlon, maxlon, minlat, maxlat = REGIONS[region]
    region_lons_aligned = align_longitudes([minlon, maxlon], lon_axis)

    lat_mask = (lat_axis >= minlat) & (lat_axis <= maxlat)
    if region_lons_aligned[0] <= region_lons_aligned[1]:
        lon_mask = (
            (lon_axis >= region_lons_aligned[0])
            & (lon_axis <= region_lons_aligned[1])
        )
    else:
        lon_mask = (
            (lon_axis >= region_lons_aligned[0])
            | (lon_axis <= region_lons_aligned[1])
        )

    lat_indices_original = np.flatnonzero(lat_mask)
    lon_indices_original = np.flatnonzero(lon_mask)

    if len(lat_indices_original) < 2 or len(lon_indices_original) < 2:
        raise ValueError(f"Reference grid does not contain enough cells in {region}")

    region_lats = lat_axis[lat_indices_original]
    region_lons = lon_axis[lon_indices_original]

    lat_order = np.argsort(region_lats)
    lon_order = np.argsort(region_lons)

    lat_sorted = region_lats[lat_order]
    lon_sorted = region_lons[lon_order]

    nlat = len(lat_sorted)
    nlon = len(lon_sorted)
    ncell = nlat * nlon

    plot_lon_2d, plot_lat_2d = np.meshgrid(
        to_signed_longitude(region_lons),
        region_lats,
    )

    polygon_mask = geojson_grid_mask(
        plot_lon_2d,
        plot_lat_2d,
        PATHS["geojson"],
    )

    return {
        "lat_indices_original": lat_indices_original,
        "lon_indices_original": lon_indices_original,
        "region_lats": region_lats,
        "region_lons": region_lons,
        "lat_order": lat_order,
        "lon_order": lon_order,
        "lat_sorted": lat_sorted,
        "lon_sorted": lon_sorted,
        "lat_edges": centers_to_edges(lat_sorted),
        "lon_edges": centers_to_edges(lon_sorted),
        "nlat": nlat,
        "nlon": nlon,
        "ncell": ncell,
        "plot_lat_2d": plot_lat_2d,
        "plot_lon_2d": plot_lon_2d,
        "polygon_mask": polygon_mask,
        "elevation_sum": np.zeros(ncell, dtype=np.float64),
        "elevation_count": np.zeros(ncell, dtype=np.int64),
        "tri_sum": np.zeros(ncell, dtype=np.float64),
        "tri_count": np.zeros(ncell, dtype=np.int64),
    }


def discover_dem_tiles(dem_dir):
    dem_dir = project_path(dem_dir)

    if not dem_dir.exists():
        raise FileNotFoundError(f"DEM directory not found: {dem_dir}")

    paths = sorted(
        list(dem_dir.rglob("*.tif"))
        + list(dem_dir.rglob("*.tiff"))
        + list(dem_dir.rglob("*.TIF"))
        + list(dem_dir.rglob("*.TIFF"))
    )
    paths = list(dict.fromkeys(paths))

    if not paths:
        raise FileNotFoundError(f"No GeoTIFF DEM tiles found under {dem_dir}")

    print(f"Found {len(paths)} DEM tiles under {dem_dir}")
    return paths


def tile_intersects_region(dataset, region):
    minlon, maxlon, minlat, maxlat = REGIONS[region]

    if dataset.crs is None:
        bounds = dataset.bounds
        west, south, east, north = (
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
        )
    else:
        west, south, east, north = transform_bounds(
            dataset.crs,
            "EPSG:4326",
            *dataset.bounds,
            densify_pts=21,
        )

    return not (
        east < minlon
        or west > maxlon
        or north < minlat
        or south > maxlat
    )


def compute_riley_tri(halo_array):
    """Compute 8-neighbor Riley TRI for the core of a 1-pixel-halo array."""
    center = halo_array[1:-1, 1:-1]
    height, width = center.shape

    tri_squared = np.zeros((height, width), dtype=np.float64)
    valid = np.isfinite(center)

    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            if row_offset == 0 and col_offset == 0:
                continue

            neighbor = halo_array[
                1 + row_offset : 1 + row_offset + height,
                1 + col_offset : 1 + col_offset + width,
            ]

            neighbor_valid = np.isfinite(neighbor)
            valid &= neighbor_valid

            difference = np.zeros_like(center, dtype=np.float64)
            pair_valid = np.isfinite(center) & neighbor_valid
            difference[pair_valid] = neighbor[pair_valid] - center[pair_valid]
            tri_squared += difference * difference

    tri = np.sqrt(tri_squared)
    tri[~valid] = np.nan
    return tri


def block_coordinates(dataset, row_start, col_start, height, width):
    """Return DEM pixel-center coordinates in WGS84."""
    transform = dataset.transform

    if not np.isclose(transform.b, 0.0) or not np.isclose(transform.d, 0.0):
        raise ValueError("Rotated DEM rasters are not supported")

    columns = np.arange(col_start, col_start + width, dtype=float)
    rows = np.arange(row_start, row_start + height, dtype=float)

    x_values = transform.c + (columns + 0.5) * transform.a
    y_values = transform.f + (rows + 0.5) * transform.e

    if dataset.crs is None or dataset.crs.is_geographic:
        return y_values, x_values

    x_2d, y_2d = np.meshgrid(x_values, y_values)
    lon_flat, lat_flat = warp_transform(
        dataset.crs,
        "EPSG:4326",
        x_2d.ravel(),
        y_2d.ravel(),
    )

    lat_2d = np.asarray(lat_flat, dtype=float).reshape(height, width)
    lon_2d = np.asarray(lon_flat, dtype=float).reshape(height, width)
    return lat_2d, lon_2d


def accumulate_terrain_values(grid, latitudes, longitudes, elevation, tri):
    """Accumulate DEM elevation and TRI into the reference grid cells."""
    if np.ndim(latitudes) == 1 and np.ndim(longitudes) == 1:
        aligned_lons = align_longitudes(longitudes, grid["lon_sorted"])

        lat_indices = np.searchsorted(
            grid["lat_edges"], latitudes, side="right"
        ) - 1
        lon_indices = np.searchsorted(
            grid["lon_edges"], aligned_lons, side="right"
        ) - 1

        valid_rows = (lat_indices >= 0) & (lat_indices < grid["nlat"])
        valid_cols = (lon_indices >= 0) & (lon_indices < grid["nlon"])

        cell_ids = lat_indices[:, None] * grid["nlon"] + lon_indices[None, :]
        location_valid = valid_rows[:, None] & valid_cols[None, :]

    else:
        latitudes = np.asarray(latitudes, dtype=float)
        longitudes = align_longitudes(longitudes, grid["lon_sorted"])

        lat_indices = np.searchsorted(
            grid["lat_edges"], latitudes, side="right"
        ) - 1
        lon_indices = np.searchsorted(
            grid["lon_edges"], longitudes, side="right"
        ) - 1

        location_valid = (
            (lat_indices >= 0)
            & (lat_indices < grid["nlat"])
            & (lon_indices >= 0)
            & (lon_indices < grid["nlon"])
        )
        cell_ids = lat_indices * grid["nlon"] + lon_indices

    elevation_valid = location_valid & np.isfinite(elevation)
    if np.any(elevation_valid):
        ids = cell_ids[elevation_valid].astype(np.int64)
        vals = elevation[elevation_valid].astype(np.float64)
        grid["elevation_sum"] += np.bincount(
            ids, weights=vals, minlength=grid["ncell"]
        )
        grid["elevation_count"] += np.bincount(
            ids, minlength=grid["ncell"]
        )

    tri_valid = location_valid & np.isfinite(tri)
    if np.any(tri_valid):
        ids = cell_ids[tri_valid].astype(np.int64)
        vals = tri[tri_valid].astype(np.float64)
        grid["tri_sum"] += np.bincount(
            ids, weights=vals, minlength=grid["ncell"]
        )
        grid["tri_count"] += np.bincount(
            ids, minlength=grid["ncell"]
        )


def process_dem_tiles(grid, dem_paths, region, block_size):
    used_tiles = 0

    for tile_number, path in enumerate(dem_paths, start=1):
        with rasterio.open(path) as dataset:
            if not tile_intersects_region(dataset, region):
                continue

            used_tiles += 1
            print(
                f"DEM tile {used_tiles}: {path.name} "
                f"({tile_number}/{len(dem_paths)})"
            )

            for row_start in range(0, dataset.height, block_size):
                height = min(block_size, dataset.height - row_start)

                for col_start in range(0, dataset.width, block_size):
                    width = min(block_size, dataset.width - col_start)

                    halo_window = Window(
                        col_start - 1,
                        row_start - 1,
                        width + 2,
                        height + 2,
                    )

                    halo = dataset.read(
                        1,
                        window=halo_window,
                        boundless=True,
                        masked=True,
                    ).astype(np.float64)
                    halo = halo.filled(np.nan)

                    elevation = halo[1:-1, 1:-1]
                    tri = compute_riley_tri(halo)

                    latitudes, longitudes = block_coordinates(
                        dataset,
                        row_start,
                        col_start,
                        height,
                        width,
                    )

                    accumulate_terrain_values(
                        grid,
                        latitudes,
                        longitudes,
                        elevation,
                        tri,
                    )

    if used_tiles == 0:
        raise RuntimeError(
            f"No DEM tiles intersected the requested region {region}"
        )

    print(f"Processed {used_tiles} DEM tiles intersecting {region}")


def finalize_terrain_metric(grid, metric):
    if metric == "elevation":
        sums = grid["elevation_sum"]
        counts = grid["elevation_count"]
    elif metric == "tri":
        sums = grid["tri_sum"]
        counts = grid["tri_count"]
    else:
        raise ValueError(f"Unsupported terrain metric: {metric}")

    values_sorted = np.full(grid["ncell"], np.nan, dtype=np.float64)
    valid = counts > 0
    values_sorted[valid] = sums[valid] / counts[valid]
    values_sorted = values_sorted.reshape(grid["nlat"], grid["nlon"])

    values_original = np.full(
        (len(grid["region_lats"]), len(grid["region_lons"])),
        np.nan,
        dtype=np.float64,
    )
    values_original[np.ix_(grid["lat_order"], grid["lon_order"])] = values_sorted
    return values_original


def compute_terrain_grids(
    observation_cache,
    region,
    dem_dir,
    block_size,
    force_recompute=False,
):
    cache_dir = project_path(TERRAIN_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"terrain_{region}.npz"

    if cache_path.exists() and not force_recompute:
        print(f"Loading cached terrain data: {cache_path}")

        with np.load(cache_path) as cached:
            terrain = {
                "elevation": cached["elevation"],
                "tri": cached["tri"],
                "plot_lat_2d": cached["plot_lat_2d"],
                "plot_lon_2d": cached["plot_lon_2d"],
            }

        print(
            f"Mean-elevation cells: "
            f"{np.sum(np.isfinite(terrain['elevation']))}"
        )
        print(
            f"Mean-TRI cells: "
            f"{np.sum(np.isfinite(terrain['tri']))}"
        )

        return terrain

    if force_recompute:
        print("Forcing terrain recalculation.")
    else:
        print(f"No terrain cache found: {cache_path}")

    print("Calculating elevation and TRI from DEM tiles...")

    terrain_grid = build_reference_terrain_grid(
        observation_cache,
        region,
    )

    dem_paths = discover_dem_tiles(dem_dir)

    process_dem_tiles(
        grid=terrain_grid,
        dem_paths=dem_paths,
        region=region,
        block_size=block_size,
    )

    elevation = finalize_terrain_metric(
        terrain_grid,
        "elevation",
    )

    tri = finalize_terrain_metric(
        terrain_grid,
        "tri",
    )

    terrain = {
        "elevation": elevation,
        "tri": tri,
        "plot_lat_2d": terrain_grid["plot_lat_2d"],
        "plot_lon_2d": terrain_grid["plot_lon_2d"],
    }

    np.savez_compressed(
        cache_path,
        elevation=terrain["elevation"],
        tri=terrain["tri"],
        plot_lat_2d=terrain["plot_lat_2d"],
        plot_lon_2d=terrain["plot_lon_2d"],
    )

    print(f"Saved terrain cache: {cache_path}")

    print(
        f"Mean-elevation cells: {np.sum(np.isfinite(elevation))}; "
        f"range {np.nanmin(elevation):.1f} to "
        f"{np.nanmax(elevation):.1f} m"
    )

    print(
        f"Mean-TRI cells: {np.sum(np.isfinite(tri))}; "
        f"range {np.nanmin(tri):.2f} to "
        f"{np.nanmax(tri):.2f} m"
    )

    return terrain

# PLOTTING HELPERS

def setup_basemap(axis, region):
    minlon, maxlon, minlat, maxlat = REGIONS[region]
    basemap = Basemap(
        projection="merc",
        epsg=4326,
        llcrnrlon=minlon,
        llcrnrlat=minlat,
        urcrnrlon=maxlon,
        urcrnrlat=maxlat,
        resolution="i",
        ax=axis,
    )

    try:
        basemap.arcgisimage(
            server="http://server.arcgisonline.com/arcgis",
            service="World_Shaded_Relief",
            xpixels=1200,
            verbose=False,
        )
    except Exception:
        basemap.shadedrelief(scale=0.35)

    return basemap


def draw_map_decorations(
    basemap,
    region,
    *,
    show_latitude_ticks,
    show_longitude_ticks,
):
    lat_ticks, lon_ticks = get_axis_ticks(region)

    basemap.drawcoastlines(color="0.20", linewidth=0.6, zorder=15)
    basemap.drawstates(color="0.35", linewidth=0.5, zorder=15)
    basemap.drawcountries(color="0.35", linewidth=0.5, zorder=15)

    basemap.drawparallels(
        lat_ticks,
        labels=[int(show_latitude_ticks), 0, 0, 0],
        fontsize=8,
        fmt=format_latitude_tick,
        color=(0, 0, 0, 0),
        textcolor="black",
        linewidth=0.001,
        dashes=[1, 1],
    )
    basemap.drawmeridians(
        lon_ticks,
        labels=[0, 0, 0, int(show_longitude_ticks)],
        fontsize=8,
        fmt=format_longitude_tick,
        color=(0, 0, 0, 0),
        textcolor="black",
        linewidth=0.001,
        dashes=[1, 1],
    )


def draw_station_points(basemap, observation_cache, *, size, alpha):
    x_values, y_values = basemap(
        observation_cache["station_lons_map"],
        observation_cache["station_lats"],
    )
    basemap.scatter(
        x_values,
        y_values,
        c="0.15",
        s=size,
        marker=".",
        alpha=alpha,
        linewidths=0,
        zorder=12,
    )


def observation_colormap():
    colormap = plt.get_cmap(OBS_CMAP).copy()
    colormap.set_bad((1, 1, 1, 0))
    colormap.set_under(colormap(0.0))
    colormap.set_over(colormap(1.0))

    levels = np.linspace(OBS_MIN, OBS_MAX, OBS_N_LEVELS + 1)
    normalization = BoundaryNorm(levels, ncolors=colormap.N, clip=False)
    return colormap, normalization, levels


def bias_colormap(limit):
    colormap = plt.get_cmap("RdBu_r").copy()
    colormap.set_bad((1, 1, 1, 0))
    normalization = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    return colormap, normalization


def maximum_absolute_bias(results):
    values = [
        result["valid_diffs"][np.isfinite(result["valid_diffs"])]
        for result in results.values()
        if np.any(np.isfinite(result["valid_diffs"]))
    ]

    if not values:
        return 1.0

    limit = float(np.nanmax(np.abs(np.concatenate(values))))
    return limit if np.isfinite(limit) and limit > 0 else 1.0


def terrain_norm(values, start_at_zero=True):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]

    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)

    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))

    if start_at_zero:
        vmin = min(0.0, vmin)

    if vmax <= vmin:
        vmax = vmin + 1.0

    return Normalize(vmin=vmin, vmax=vmax)


def initialization_offset_label(day, verification_start):
    initialization_time = pd.Timestamp(f"2025-01-{int(day):02d} 00:00:00")
    verification_start = normalize_timestamp(verification_start)
    hours = int(
        (verification_start - initialization_time).total_seconds() // 3600
    )
    return f"{hours}-h initialization offset"


def add_shared_coordinate_labels(figure, map_axes):
    figure.canvas.draw()
    map_bounds = Bbox.union([axis.get_position() for axis in map_axes])

    x_center = (map_bounds.x0 + map_bounds.x1) / 2
    y_center = (map_bounds.y0 + map_bounds.y1) / 2

    figure.text(
        x_center,
        max(0.012, map_bounds.y0 - 0.042),
        "Longitude (°W)",
        ha="center",
        va="center",
        fontsize=12,
    )
    figure.text(
        max(0.012, map_bounds.x0 - 0.055),
        y_center,
        "Latitude (°N)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=12,
    )


def add_horizontal_top_colorbar(figure, axis, mesh, label, *, extend="neither"):
    """Add a compact colorbar directly below one top-row map."""
    figure.canvas.draw()
    bounds = axis.get_position()

    cax = figure.add_axes(
        [
            bounds.x0,
            bounds.y0 - TOP_COLORBAR_PAD - TOP_COLORBAR_HEIGHT,
            bounds.width,
            TOP_COLORBAR_HEIGHT,
        ]
    )

    colorbar = figure.colorbar(
        mesh,
        cax=cax,
        orientation="horizontal",
        extend=extend,
    )
    colorbar.set_label(label, fontsize=9, labelpad=3)
    colorbar.ax.tick_params(labelsize=8, pad=1)


def add_bias_colorbar(figure, model_axes, bias_mesh):
    if bias_mesh is None:
        return

    figure.canvas.draw()
    model_bounds = Bbox.union([axis.get_position() for axis in model_axes])

    height = model_bounds.height * BIAS_COLORBAR_HEIGHT_FRACTION
    bottom = model_bounds.y0 + (model_bounds.height - height) / 2

    cax = figure.add_axes(
        [
            model_bounds.x1 + COLORBAR_PAD,
            bottom,
            COLORBAR_WIDTH,
            height,
        ]
    )

    colorbar = figure.colorbar(
        bias_mesh,
        cax=cax,
        orientation="vertical",
        extend="both",
    )
    colorbar.set_label(
        f"Wind Speed Bias ({VARIABLES['units']})",
        fontsize=11,
        labelpad=8,
    )
    colorbar.ax.tick_params(labelsize=8)


# COMBINED FIGURE

def plot_dashboard(
    observation_cache,
    terrain,
    results,
    region,
    day,
    start_time,
    selected_models,
    show_stations,
):
    obs_cmap, obs_norm, _ = observation_colormap()
    diff_cmap, diff_norm = bias_colormap(maximum_absolute_bias(results))

    elevation_cmap = plt.get_cmap(ELEVATION_CMAP).copy()
    elevation_cmap.set_bad((1, 1, 1, 0))
    elevation_norm = terrain_norm(terrain["elevation"], start_at_zero=True)

    tri_cmap = plt.get_cmap(TRI_CMAP).copy()
    tri_cmap.set_bad((1, 1, 1, 0))
    tri_norm = terrain_norm(terrain["tri"], start_at_zero=True)

    valid_models = [model for model in selected_models if model in results]
    if not valid_models:
        raise RuntimeError("No model-bias results available for plotting")

    model_rows = int(np.ceil(len(valid_models) / FIGURE_COLUMNS))

    figure = plt.figure(
        figsize=(14.2, 4.0 + 3.55 * model_rows),
        dpi=150,
    )

    grid = figure.add_gridspec(
        nrows=model_rows + 2,
        ncols=FIGURE_COLUMNS,
        height_ratios=[1.0, 0.15] + [1.0] * model_rows,
        hspace=0.20,
        wspace=0.10,
    )

    map_axes = []

    obs_axis = figure.add_subplot(grid[0, 0])
    map_axes.append(obs_axis)
    obs_map = setup_basemap(obs_axis, region)
    x_grid, y_grid = obs_map(
        observation_cache["grid_lon_2d"],
        observation_cache["grid_lat_2d"],
    )
    obs_mesh = obs_map.pcolormesh(
        x_grid,
        y_grid,
        observation_cache["obs_grid"],
        cmap=obs_cmap,
        norm=obs_norm,
        shading="auto",
        alpha=0.82,
        zorder=5,
    )

    if show_stations:
        draw_station_points(
            obs_map,
            observation_cache,
            size=4,
            alpha=0.42,
        )

    draw_map_decorations(
        obs_map,
        region,
        show_latitude_ticks=True,
        show_longitude_ticks=False,
    )
    obs_axis.set_title(
        "Average Observed Wind Speed",
        fontsize=12,
        fontweight="bold",
    )

    elevation_axis = figure.add_subplot(grid[0, 1])
    map_axes.append(elevation_axis)
    elevation_map = setup_basemap(elevation_axis, region)
    terrain_x, terrain_y = elevation_map(
        terrain["plot_lon_2d"],
        terrain["plot_lat_2d"],
    )
    elevation_mesh = elevation_map.pcolormesh(
        terrain_x,
        terrain_y,
        terrain["elevation"],
        cmap=elevation_cmap,
        norm=elevation_norm,
        shading="auto",
        alpha=0.88,
        zorder=5,
    )
    draw_map_decorations(
        elevation_map,
        region,
        show_latitude_ticks=False,
        show_longitude_ticks=False,
    )
    elevation_axis.set_title(
        "Mean Terrain Elevation",
        fontsize=12,
        fontweight="bold",
    )

    tri_axis = figure.add_subplot(grid[0, 2])
    map_axes.append(tri_axis)
    tri_map = setup_basemap(tri_axis, region)
    tri_x, tri_y = tri_map(
        terrain["plot_lon_2d"],
        terrain["plot_lat_2d"],
    )
    tri_mesh = tri_map.pcolormesh(
        tri_x,
        tri_y,
        terrain["tri"],
        cmap=tri_cmap,
        norm=tri_norm,
        shading="auto",
        alpha=0.88,
        zorder=5,
    )
    draw_map_decorations(
        tri_map,
        region,
        show_latitude_ticks=False,
        show_longitude_ticks=False,
    )
    tri_axis.set_title(
        "Mean Terrain Ruggedness Index (TRI)",
        fontsize=12,
        fontweight="bold",
    )

    model_axes = []
    bias_mesh = None

    for panel_index, model_key in enumerate(valid_models):
        row = panel_index // FIGURE_COLUMNS
        column = panel_index % FIGURE_COLUMNS

        axis = figure.add_subplot(grid[row + 2, column])
        model_axes.append(axis)
        map_axes.append(axis)
        basemap = setup_basemap(axis, region)

        result = results[model_key]
        x_grid, y_grid = basemap(
            observation_cache["grid_lon_2d"],
            observation_cache["grid_lat_2d"],
        )
        bias_mesh = basemap.pcolormesh(
            x_grid,
            y_grid,
            result["diff_grid"],
            cmap=diff_cmap,
            norm=diff_norm,
            shading="auto",
            alpha=0.78,
            zorder=5,
        )

        if show_stations:
            draw_station_points(
                basemap,
                observation_cache,
                size=3,
                alpha=0.30,
            )

        draw_map_decorations(
            basemap,
            region,
            show_latitude_ticks=(column == 0),
            show_longitude_ticks=(row == model_rows - 1),
        )
        axis.set_title(
            MODELS[model_key],
            fontsize=12,
            fontweight="bold",
        )

    total_slots = model_rows * FIGURE_COLUMNS
    for empty_index in range(len(valid_models), total_slots):
        row = empty_index // FIGURE_COLUMNS
        column = empty_index % FIGURE_COLUMNS
        axis = figure.add_subplot(grid[row + 2, column])
        axis.axis("off")

    first_time = pd.Timestamp(observation_cache["period_first_time"])
    last_time = pd.Timestamp(observation_cache["period_last_time"])
    offset_label = initialization_offset_label(day, start_time)

    figure.suptitle(
        "Time-Averaged Wind Speed, Terrain, and Forecast Bias",
        fontsize=16,
        fontweight="bold",
        y=0.988,
    )
    figure.text(
        0.5,
        0.963,
        (
            f"Verification: {first_time:%Y-%m-%d %H:%M} to "
            f"{last_time:%Y-%m-%d %H:%M} UTC | {offset_label} | "
            f"Initialized Jan {int(day):02d}, 2025"
        ),
        ha="center",
        va="top",
        fontsize=10.5,
    )

    figure.subplots_adjust(
        left=0.075,
        right=0.91,
        top=0.90,
        bottom=0.09,
    )

    add_horizontal_top_colorbar(
        figure,
        obs_axis,
        obs_mesh,
        f"Wind Speed ({VARIABLES['units']})",
        extend="max",
    )
    add_horizontal_top_colorbar(
        figure,
        elevation_axis,
        elevation_mesh,
        "Mean Elevation (m)",
    )
    add_horizontal_top_colorbar(
        figure,
        tri_axis,
        tri_mesh,
        "Mean TRI (m)",
    )

    add_bias_colorbar(
        figure,
        model_axes,
        bias_mesh,
    )

    figure.canvas.draw()
    add_shared_coordinate_labels(figure, map_axes)

    period_tag = f"{first_time:%Y%m%d_%H%M}_to_{last_time:%Y%m%d_%H%M}"
    output_path = project_path(
        "./figs/maps/spatial_map_bias_and_terrain.png"
    )
    ensure_parent_dir(output_path)

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"\nSaved: {output_path}")


# COMMAND LINE

def parse_arguments():
    parser = argparse.ArgumentParser(description="Create terrain and forecast-bias maps.")
    parser.add_argument("--region", default=DEFAULT_REGION, choices=list(REGIONS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--init-day", dest="init_day", default=DEFAULT_INIT_DAY, choices=LEAD_DAYS)
    parser.add_argument("--start-time", default=PLOT_WINDOW["start"])
    parser.add_argument("--end-time", default=PLOT_WINDOW["end"])
    parser.add_argument("--dem-dir", default=DEM_DIR_DEFAULT)
    parser.add_argument("--block-size", type=int, default=DEM_BLOCK_SIZE)
    parser.add_argument("--hide-stations", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    start_time = normalize_timestamp(args.start_time)
    end_time = normalize_timestamp(args.end_time)

    print(f"Region: {args.region}")
    print(f"Forecast day file: Day{args.init_day}")
    print(f"Verification start: {start_time} (inclusive)")
    print(f"Verification end: {end_time} (exclusive)")
    print(f"DEM directory: {project_path(args.dem_dir)}")
    print("Terrain aggregation: mean 30-m DEM elevation and mean Riley TRI")
    print("Bias: time mean of forecast - observation")

    observation_cache, results = compute_all_grids(
        region=args.region,
        day=args.init_day,
        start_time=start_time,
        end_time=end_time,
        selected_models=args.models,
    )

    terrain = compute_terrain_grids(
        observation_cache=observation_cache,
        region=args.region,
        dem_dir=args.dem_dir,
        block_size=args.block_size,
    )

    plot_dashboard(
        observation_cache=observation_cache,
        terrain=terrain,
        results=results,
        region=args.region,
        day=args.init_day,
        start_time=start_time,
        selected_models=args.models,
        show_stations=not args.hide_stations,
    )


if __name__ == "__main__":
    main()
