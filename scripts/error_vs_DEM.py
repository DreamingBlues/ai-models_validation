# Figure 5. Time-averaged grid-cell MAE and DEM-derived terrain characteristics DEFAULT California domain. 
# MAE against mean DEM elevation,  MAE against grid cell mean TRI
# Each point represents one occupied model grid cell, while the dashed lines show the fitted linear relationships


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import rasterio
from rasterio.warp import transform_geom
from rasterio.features import bounds as geometry_bounds, geometry_mask
from rasterio.merge import merge as raster_merge


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
    project_path,
    region_mask,
    trim_to_period,
)


# DEFAULTS

DEFAULT_DEM_PATH = "./data/dem_tiles"


# PATH HELPERS

def resolve_input_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    return project_path(path)


# STATION DATA

def load_station_subset(path, region, var_name):
    path = project_path(path)

    if not path.exists():
        print(f"[Error] Station NetCDF not found: {path}")
        return None

    ds = open_dataset_safe(path)

    if var_name not in ds:
        print(f"[Error] Variable '{var_name}' not found in station file.")
        ds.close()
        return None

    mask_region = region_mask(
        ds["latitude"].values,
        ds["longitude"].values,
        region,
    )

    subset = ds.isel(station=mask_region)

    if subset.sizes["station"] == 0:
        print(f"[Warning] No stations found within {region} bounds.")
        ds.close()
        return None

    print(f"Loaded {subset.sizes['station']} stations for {region}")

    return subset


# DEM HELPERS

def open_dem_tiles(path):
    path = resolve_input_path(path)

    if not path.exists():
        print(f"[Error] DEM path not found: {path}")
        return None

    if path.is_file():
        tif_paths = [path]
    else:
        tif_paths = sorted(path.rglob("*.tif"))
        tif_paths += sorted(path.rglob("*.tiff"))

    # Remove duplicates in case a filesystem/path pattern returns the same file.
    tif_paths = list(dict.fromkeys(tif_paths))

    if not tif_paths:
        print(f"[Error] No .tif/.tiff DEM tiles found in: {path}")
        return None

    datasets = []

    try:
        for tif_path in tif_paths:
            datasets.append(rasterio.open(tif_path))
    except Exception as e:
        for ds in datasets:
            ds.close()
        print(f"[Error] Could not open DEM tiles: {e}")
        return None

    crs_values = {str(ds.crs) for ds in datasets}

    if None in [ds.crs for ds in datasets]:
        for ds in datasets:
            ds.close()
        print("[Error] One or more DEM tiles do not contain a CRS.")
        return None

    if len(crs_values) != 1:
        for ds in datasets:
            ds.close()
        print(
            "[Error] DEM tiles use multiple coordinate systems. "
            "All tiles must use the same CRS."
        )
        return None

    first = datasets[0]

    print("\nDEM tile information:")
    print(f"  Directory/file: {path}")
    print(f"  Number of tiles: {len(datasets)}")
    print(f"  CRS: {first.crs}")
    print(
        f"  Pixel size: {abs(first.transform.a):.3f} x "
        f"{abs(first.transform.e):.3f}"
    )
    print(f"  NoData: {first.nodata}")

    return {
        "datasets": datasets,
        "crs": first.crs,
        "path": path,
    }


def close_dem_tiles(dem_tiles):
    if dem_tiles is None:
        return

    for ds in dem_tiles["datasets"]:
        ds.close()


def coordinate_edges(values):
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("coordinate_edges requires a 1-D coordinate array.")

    if len(values) < 2:
        raise ValueError("At least two coordinate values are required.")

    midpoint = (values[:-1] + values[1:]) / 2.0

    edges = np.empty(len(values) + 1, dtype=float)

    edges[1:-1] = midpoint

    edges[0] = values[0] - (midpoint[0] - values[0])
    edges[-1] = values[-1] + (values[-1] - midpoint[-1])

    return edges


def build_grid_geometry(ds_model):
    grid_lat = np.asarray(ds_model.latitude.values, dtype=float)
    grid_lon = np.asarray(ds_model.longitude.values, dtype=float)

    if grid_lat.ndim != 1 or grid_lon.ndim != 1:
        raise ValueError(
            "DEM terrain statistics currently require the model to use "
            "1-D latitude and longitude coordinates."
        )

    lat_edges = coordinate_edges(grid_lat)
    lon_edges = coordinate_edges(grid_lon)

    return grid_lat, grid_lon, lat_edges, lon_edges


def grid_cell_polygon_wgs84(
    lat_index,
    lon_index,
    lat_edges,
    lon_edges,
):
    lat1 = lat_edges[lat_index]
    lat2 = lat_edges[lat_index + 1]

    lon1 = lon_edges[lon_index]
    lon2 = lon_edges[lon_index + 1]

    south = min(lat1, lat2)
    north = max(lat1, lat2)

    west = min(lon1, lon2)
    east = max(lon1, lon2)

    polygon = {
        "type": "Polygon",
        "coordinates": [[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south],
        ]]
    }

    return polygon, (west, south, east, north)


def calculate_riley_tri(elevation, valid_mask):
    elevation = np.asarray(elevation, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    tri = np.full(elevation.shape, np.nan, dtype=float)

    if elevation.shape[0] < 3 or elevation.shape[1] < 3:
        return tri

    center = elevation[1:-1, 1:-1]
    center_valid = valid_mask[1:-1, 1:-1]

    neighbor_slices = [
        (slice(0, -2), slice(0, -2)),
        (slice(0, -2), slice(1, -1)),
        (slice(0, -2), slice(2, None)),
        (slice(1, -1), slice(0, -2)),
        (slice(1, -1), slice(2, None)),
        (slice(2, None), slice(0, -2)),
        (slice(2, None), slice(1, -1)),
        (slice(2, None), slice(2, None)),
    ]

    tri_sum = np.zeros(center.shape, dtype=float)
    tri_valid = center_valid.copy()

    for row_slice, col_slice in neighbor_slices:
        neighbor = elevation[row_slice, col_slice]
        neighbor_valid = valid_mask[row_slice, col_slice]

        tri_valid &= neighbor_valid
        tri_sum += (neighbor - center) ** 2

    tri_inner = np.sqrt(tri_sum)
    tri_inner[~tri_valid] = np.nan

    tri[1:-1, 1:-1] = tri_inner

    return tri


def compute_dem_stats_for_grid_cell(
    dem_tiles,
    polygon_wgs84,
):
    try:
        polygon_dem = transform_geom(
            src_crs="EPSG:4326",
            dst_crs=dem_tiles["crs"],
            geom=polygon_wgs84,
            precision=-1,
        )
    except Exception as e:
        print(f"    > DEM polygon reprojection failed: {e}")
        return np.nan, np.nan, 0, 0

    try:
        geom_left, geom_bottom, geom_right, geom_top = geometry_bounds(
            polygon_dem
        )
    except Exception as e:
        print(f"    > Could not determine DEM polygon bounds: {e}")
        return np.nan, np.nan, 0, 0

    # TRI needs one neighboring DEM pixel outside the target cell on all sides.
    first_dem = dem_tiles["datasets"][0]
    pixel_x = abs(float(first_dem.transform.a))
    pixel_y = abs(float(first_dem.transform.e))

    read_left = geom_left - pixel_x
    read_right = geom_right + pixel_x
    read_bottom = geom_bottom - pixel_y
    read_top = geom_top + pixel_y

    overlapping_tiles = []

    for dem in dem_tiles["datasets"]:
        tile_bounds = dem.bounds

        overlaps = not (
            tile_bounds.right <= read_left
            or tile_bounds.left >= read_right
            or tile_bounds.top <= read_bottom
            or tile_bounds.bottom >= read_top
        )

        if overlaps:
            overlapping_tiles.append(dem)

    if not overlapping_tiles:
        return np.nan, np.nan, 0, 0

    try:
        mosaic, mosaic_transform = raster_merge(
            overlapping_tiles,
            bounds=(
                read_left,
                read_bottom,
                read_right,
                read_top,
            ),
            indexes=[1],
            masked=True,
        )
    except Exception as e:
        print(f"    > DEM tile mosaic failed: {e}")
        return np.nan, np.nan, 0, 0

    if mosaic.size == 0:
        return np.nan, np.nan, 0, 0

    band = np.ma.asarray(mosaic[0])
    elevation = np.asarray(band.data, dtype=float)

    valid_dem = (
        ~np.ma.getmaskarray(band)
        & np.isfinite(elevation)
    )

    try:
        inside_polygon = geometry_mask(
            [polygon_dem],
            out_shape=band.shape,
            transform=mosaic_transform,
            invert=True,
        )
    except Exception as e:
        print(f"    > DEM polygon mask failed: {e}")
        return np.nan, np.nan, 0, 0

    # Mean elevation from valid DEM pixels whose centers fall inside the model grid cell.
    elevation_mask = valid_dem & inside_polygon
    elevation_values = elevation[elevation_mask]

    if elevation_values.size == 0:
        return np.nan, np.nan, 0, 0

    mean_elevation = float(np.mean(elevation_values))

    # Riley TRI from each valid 3 x 3 DEM neighborhood. The one-pixel DEM buffer
    # above prevents model-cell boundaries from artificially eliminating TRI
    # values at the edge of the cell.

    tri = calculate_riley_tri(
        elevation=elevation,
        valid_mask=valid_dem,
    )

    tri_mask = inside_polygon & np.isfinite(tri)
    tri_values = tri[tri_mask]

    if tri_values.size == 0:
        return mean_elevation, np.nan, int(elevation_values.size), 0

    mean_tri = float(np.mean(tri_values))

    return (
        mean_elevation,
        mean_tri,
        int(elevation_values.size),
        int(tri_values.size),
    )


# STATISTICAL HELPERS

def add_trendline(ax, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    if np.sum(valid) < 2:
        return np.nan

    x_valid = x[valid]
    y_valid = y[valid]

    if len(np.unique(x_valid)) < 2:
        return np.nan

    slope, intercept = np.polyfit(
        x_valid,
        y_valid,
        1,
    )

    x_line = np.linspace(
        np.nanmin(x_valid),
        np.nanmax(x_valid),
        100,
    )

    y_line = slope * x_line + intercept

    ax.plot(
        x_line,
        y_line,
        color="black",
        linestyle="--",
        linewidth=1.8,
        alpha=0.9,
    )

    return slope


def calculate_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    if np.sum(valid) < 3:
        return np.nan, np.nan

    x_valid = x[valid]
    y_valid = y[valid]

    if (
        len(np.unique(x_valid)) < 2 or
        len(np.unique(y_valid)) < 2
    ):
        return np.nan, np.nan

    rho, p_value = spearmanr(
        x_valid,
        y_valid,
    )

    return float(rho), float(p_value)


# GRID-CELL ERROR + TERRAIN ANALYSIS

def compute_model_gridcell_timeavg_errors(
    ds_model,
    ds_stations,
    dem_tiles,
    dem_cache,
    model_var,
    station_var,
    metric,
    start_time,
    end_time,
):
    """
    Compute one time-averaged error value and terrain statistics for every
    occupied model grid cell.

    For each occupied model grid cell:

        1. Assign stations to the nearest model grid-cell center.
        2. Average observations from stations inside that grid cell.
        3. Extract the model forecast at that grid cell.
        4. Match common timestamps within the verification period.
        5. Compute time-averaged MAE, RMSE, or MAPE.
        6. Construct the geographic boundaries of the model grid cell.
        7. Reproject the grid-cell polygon into the DEM CRS.
        8. Extract every valid DEM pixel within the grid cell.
        9. Calculate:
              - mean terrain elevation
              - mean Riley Terrain Ruggedness Index (TRI)

    The DEM cache prevents the same terrain cell from being read repeatedly
    for each forecast model.
    """

    (
        grid_lat,
        grid_lon,
        lat_edges,
        lon_edges,
    ) = build_grid_geometry(ds_model)

    grid_lon_2d, grid_lat_2d = np.meshgrid(
        grid_lon,
        grid_lat,
    )

    flat_lats = grid_lat_2d.ravel()
    flat_lons = grid_lon_2d.ravel()

    grid_points = np.column_stack(
        (
            flat_lats,
            flat_lons,
        )
    )

    tree = cKDTree(grid_points)

    st_lats = np.asarray(
        ds_stations.latitude.values,
        dtype=float,
    )

    st_lons = np.asarray(
        ds_stations.longitude.values,
        dtype=float,
    )

    station_points = np.column_stack(
        (
            st_lats,
            st_lons,
        )
    )

    _, grid_indices = tree.query(
        station_points,
        k=1,
    )

    unique_grid_indices = np.unique(grid_indices)

    mean_elevations = []
    mean_tris = []

    signed_biases = []
    errors = []

    station_counts = []
    grid_center_lats = []
    grid_center_lons = []
    dem_pixel_counts = []

    n_lon = len(grid_lon)

    for flat_index in unique_grid_indices:

        member_mask = grid_indices == flat_index
        num_stations = int(np.sum(member_mask))

        # Convert flattened model-grid index back into latitude/longitude index

        lat_index = int(flat_index // n_lon)
        lon_index = int(flat_index % n_lon)

        cell_lat = float(grid_lat[lat_index])
        cell_lon = float(grid_lon[lon_index])

        # Station-derived observed grid-cell time series

        obs_s = (
            ds_stations
            .isel(station=member_mask)[station_var]
            .mean(
                dim="station",
                skipna=True,
            )
            .to_series()
        )

        obs_s = clean_time_index(obs_s)
        obs_s = trim_to_period(
            obs_s,
            start_time,
            end_time,
        )

        # Model grid-cell time series
        try:
            mod_s = (
                ds_model[model_var]
                .sel(
                    latitude=cell_lat,
                    longitude=cell_lon,
                    method="nearest",
                )
                .to_series()
            )

        except Exception as e:
            print(
                f"  > Could not extract model grid cell "
                f"({cell_lat:.3f}, {cell_lon:.3f}): {e}"
            )
            continue

        mod_s = clean_time_index(mod_s)
        mod_s = trim_to_period(
            mod_s,
            start_time,
            end_time,
        )

        # Match timestamps
        common_times = obs_s.index.intersection(
            mod_s.index
        )

        if len(common_times) == 0:
            continue

        obs_valid = obs_s.loc[common_times]
        mod_valid = mod_s.loc[common_times]

        error_series = mod_valid - obs_valid

        valid = (
            np.isfinite(obs_valid.values)
            & np.isfinite(mod_valid.values)
            & np.isfinite(error_series.values)
        )

        if metric == "mape":
            valid &= (
                np.abs(obs_valid.values)
                > MAPE_MIN_OBS
            )

        if np.sum(valid) == 0:
            continue

        obs_arr = obs_valid.values[valid]
        error_arr = error_series.values[valid]

    
        # Gri-cell error
        if metric == "mae":

            grid_error = np.mean(
                np.abs(error_arr)
            )

        elif metric == "rmse":

            grid_error = np.sqrt(
                np.mean(error_arr ** 2)
            )

        elif metric == "mape":

            grid_error = (
                np.mean(
                    np.abs(
                        error_arr / obs_arr
                    )
                )
                * 100.0
            )

        else:
            raise ValueError(
                f"Unsupported metric: {metric}"
            )

        signed_bias = np.mean(error_arr)

        # DEM terrain statistics
        polygon_wgs84, bounds = grid_cell_polygon_wgs84(
            lat_index=lat_index,
            lon_index=lon_index,
            lat_edges=lat_edges,
            lon_edges=lon_edges,
        )

        # Cache is based on geographic cell boundaries.
        cache_key = tuple(
            round(value, 7)
            for value in bounds
        )

        if cache_key in dem_cache:

            (
                mean_elevation,
                mean_tri,
                n_dem_pixels,
                n_tri_pixels,
            ) = dem_cache[cache_key]

        else:

            (
                mean_elevation,
                mean_tri,
                n_dem_pixels,
                n_tri_pixels,
            ) = compute_dem_stats_for_grid_cell(
                dem_tiles=dem_tiles,
                polygon_wgs84=polygon_wgs84,
            )

            dem_cache[cache_key] = (
                mean_elevation,
                mean_tri,
                n_dem_pixels,
                n_tri_pixels,
            )

        if not (
            np.isfinite(grid_error)
            and np.isfinite(signed_bias)
            and np.isfinite(mean_elevation)
            and np.isfinite(mean_tri)
            and n_dem_pixels > 0
            and n_tri_pixels > 0
        ):
            continue

        
        # Save valid cell
        mean_elevations.append(mean_elevation)
        mean_tris.append(mean_tri)

        signed_biases.append(signed_bias)
        errors.append(grid_error)

        station_counts.append(num_stations)

        grid_center_lats.append(cell_lat)
        grid_center_lons.append(cell_lon)

        dem_pixel_counts.append(n_dem_pixels)

    if len(errors) == 0:
        return None

    return {
        "mean_elevations": np.asarray(
            mean_elevations,
            dtype=float,
        ),
        "mean_tris": np.asarray(
            mean_tris,
            dtype=float,
        ),
        "errors": np.asarray(
            errors,
            dtype=float,
        ),
        "signed_biases": np.asarray(
            signed_biases,
            dtype=float,
        ),
        "station_counts": np.asarray(
            station_counts,
            dtype=int,
        ),
        "grid_center_lats": np.asarray(
            grid_center_lats,
            dtype=float,
        ),
        "grid_center_lons": np.asarray(
            grid_center_lons,
            dtype=float,
        ),
        "dem_pixel_counts": np.asarray(
            dem_pixel_counts,
            dtype=int,
        ),
    }


# PLOTTING

def metric_information(metric):

    if metric == "mae":
        return (
            "MAE",
            VARIABLES["units"],
        )

    if metric == "rmse":
        return (
            "RMSE",
            VARIABLES["units"],
        )

    if metric == "mape":
        return (
            "MAPE",
            "%",
        )

    raise ValueError(
        f"Unsupported metric: {metric}"
    )


def slope_annotation(
    slope,
    rho,
    p_value,
    metric,
):
    if not np.isfinite(slope):
        return ""

    slope_100m = slope * 100.0

    if metric == "mape":
        slope_line = (
            f"slope={slope_100m:.2f} %-pts per 100 m"
        )
    else:
        slope_line = (
            f"slope={slope_100m:.3f} m/s per 100 m"
        )

    if np.isfinite(rho):

        if np.isfinite(p_value):
            correlation_line = (
                f"Spearman ρ={rho:.2f}, "
                f"p={p_value:.3f}"
            )
        else:
            correlation_line = (
                f"Spearman ρ={rho:.2f}"
            )

        return (
            slope_line
            + "\n"
            + correlation_line
        )

    return slope_line


def plot_results(
    results,
    selected_models,
    metric,
    region,
    init_day,
):
    metric_label, metric_units = metric_information(
        metric
    )

    valid_models = [
        model
        for model in selected_models
        if model in results
    ]

    n_models = len(valid_models)

    fig, axes = plt.subplots(
        nrows=n_models,
        ncols=2,
        figsize=(12, 2.65 * n_models),
        dpi=150,
        sharey=True,
    )

    if n_models == 1:
        axes = np.array([axes])

    all_errors = np.concatenate(
        [
            results[model]["errors"]
            for model in valid_models
        ]
    )

    y_max = np.nanmax(all_errors) * 1.10

    for row_idx, model_key in enumerate(
        valid_models
    ):

        model_name = MODELS[model_key]
        data = results[model_key]

        ax_mean = axes[row_idx, 0]
        ax_tri = axes[row_idx, 1]

        # Mean DEM elevation

        x_mean = data["mean_elevations"]
        y = data["errors"]

        ax_mean.scatter(
            x_mean,
            y,
            marker=".",
            color="blue",
            s=25,
        )

        slope_mean = add_trendline(
            ax=ax_mean,
            x=x_mean,
            y=y,
        )

        annotation_mean = f"Slope = {slope_mean:+.1e}"
        
        ax_mean.text(
            0.03,
            0.92,
            annotation_mean,
            transform=ax_mean.transAxes,
            fontsize=8.5,
            ha="left",
            va="top",
        )

        ax_mean.set_ylim(
            bottom=0,
            top=y_max,
        )

        ax_mean.grid(
            True,
            alpha=0.3,
        )

        ax_mean.set_xlim(left=0)

        ax_mean.margins(
            x=0,
            y=0,
        )

        # Terrain Ruggedness Index

        x_tri = data["mean_tris"]

        ax_tri.scatter(
            x_tri,
            y,
            marker=".",
            color="red",
            s=25,
        )

        slope_tri = add_trendline(
            ax=ax_tri,
            x=x_tri,
            y=y,
        )

        annotation_tri = f"Slope = {slope_tri:+.1e}"

        ax_tri.text(
            0.03,
            0.92,
            annotation_tri,
            transform=ax_tri.transAxes,
            fontsize=8.5,
            ha="left",
            va="top",
        )

        ax_tri.set_xlim(left=0)

        ax_tri.set_ylim(
            bottom=0,
            top=y_max,
        )

        ax_tri.grid(
            True,
            alpha=0.3,
        )

        ax_tri.margins(
            x=0,
            y=0,
        )

        # Model label

        ax_mean.text(
            -0.28,
            0.5,
            model_name,
            transform=ax_mean.transAxes,
            fontsize=12,
            fontweight="bold",
            va="center",
            ha="right",
        )

        # X labels

        if row_idx == n_models - 1:

            ax_mean.set_xlabel(
                "Mean DEM Elevation (m)",
                fontsize=11,
            )

            ax_tri.set_xlabel(
                "Mean Terrain Ruggedness Index (m)",
                fontsize=11,
            )

        else:

            ax_mean.tick_params(
                labelbottom=False
            )

            ax_tri.tick_params(
                labelbottom=False
            )

        ax_mean.tick_params(
            axis="both",
            labelsize=9,
        )

        ax_tri.tick_params(
            axis="both",
            labelsize=9,
        )

    # Column titles

    axes[0, 0].set_title(
        "Mean Terrain Elevation",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )

    axes[0, 1].set_title(
        "Terrain Ruggedness",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )

    # Figure title

    init_time = pd.Timestamp(
        f"2025-01-{int(init_day):02d}"
    )

    # Portable on Windows/Linux/macOS; avoids platform-specific %-d/%#d.
    init_label = (
        f"{init_time.strftime('%b')} "
        f"{init_time.day}, "
        f"{init_time.year}"
    )

    fig.suptitle(
        f"Grid-Cell Wind Speed Error vs. DEM Topography\n"
        f"Initialized {init_label} | {region}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    fig.supylabel(
        f"Grid-Cell {metric_label} ({metric_units})",
        fontsize=12,
        x=0.37,
    )

    plt.tight_layout(
        rect=[
            0.15,
            0.03,
            1.0,
            0.96,
        ]
    )

    fig.subplots_adjust(
        hspace=0.18,
        wspace=0.12,
    )

    # OUTPUT

    output_path = project_path(
        PATHS["figs"]
    ) / "scatter" / (
        f"{metric}_dem_elevation_tri_{region}.png"
    )

    ensure_parent_dir(
        output_path
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
        dpi=300,
    )

    plt.close(fig)

    print(
        f"\nSaved plot to: {output_path}"
    )


# MAIN

def main():
    parser = argparse.ArgumentParser(description="Plot grid-cell error against DEM terrain.")
    parser.add_argument("--region", default=DEFAULT_REGION, choices=list(REGIONS.keys()))
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()), choices=list(MODELS.keys()))
    parser.add_argument("--init-day", dest="init_day", default=DEFAULT_INIT_DAY, choices=LEAD_DAYS)
    parser.add_argument("--metric", default="mae", choices=["mae", "rmse", "mape"])
    parser.add_argument("--dem", default=DEFAULT_DEM_PATH)

    args = parser.parse_args()

    region = args.region
    selected_models = args.models
    init_day = args.init_day
    metric = args.metric.lower()

    metric_label, metric_units = metric_information(metric)

    start_time = pd.Timestamp(PLOT_WINDOW["start"])
    end_time = pd.Timestamp(PLOT_WINDOW["end"])
    init_time = pd.Timestamp(f"2025-01-{int(init_day):02d} 00:00:00")

    print(f"Region: {region}")
    print(f"Models: {selected_models}")
    print(f"Initialization: {init_time}")
    print(
        f"Verification window: "
        f"{start_time} to {end_time}"
    )
    print(f"Metric: {metric_label}")
    print(f"DEM: {args.dem}")

    if metric == "mape":
        print(
            f"MAPE threshold: observed wind speed > "
            f"{MAPE_MIN_OBS} {VARIABLES['units']}"
        )

    # LOAD STATIONS

    ds_stations = load_station_subset(
        PATHS["station"],
        region,
        VARIABLES["station_wind"],
    )

    if ds_stations is None:
        return

    # OPEN DEM TILES

    dem_tiles = open_dem_tiles(args.dem)

    if dem_tiles is None:
        ds_stations.close()
        return

    dem_cache = {}

    # COMPUTE MODEL RESULTS

    results = {}

    for model_key in selected_models:

        model_name = MODELS[model_key]
        model_file = model_path(model_key, init_day)

        print(f"\nProcessing {model_name}: {model_file}")

        if not model_file.exists():
            print("  > Missing file. Skipping.")
            continue

        ds_model = open_dataset_safe(model_file)

        if VARIABLES["model_wind"] not in ds_model:
            print(f"  > Variable '{VARIABLES['model_wind']}' not found. Skipping.")
            ds_model.close()
            continue

        model_results = compute_model_gridcell_timeavg_errors(
            ds_model=ds_model,
            ds_stations=ds_stations,
            dem_tiles=dem_tiles,
            dem_cache=dem_cache,
            model_var=VARIABLES["model_wind"],
            station_var=VARIABLES["station_wind"],
            metric=metric,
            start_time=start_time,
            end_time=end_time,
        )

        ds_model.close()

        if model_results is None:
            print(
                "  > No valid grid-cell results. "
                "Skipping."
            )
            continue

        results[model_key] = model_results

        mean_rho, mean_p = calculate_spearman(
            model_results["mean_elevations"],
            model_results["errors"],
        )

        tri_rho, tri_p = calculate_spearman(
            model_results["mean_tris"],
            model_results["errors"],
        )

        print(
            f"  > Occupied cells: "
            f"{len(model_results['errors'])}"
        )

        print(
            f"  > Mean {metric_label}: "
            f"{np.nanmean(model_results['errors']):.3f} "
            f"{metric_units}"
        )

        print(
            f"  > Median {metric_label}: "
            f"{np.nanmedian(model_results['errors']):.3f} "
            f"{metric_units}"
        )

        print(
            f"  > Mean DEM elevation: "
            f"{np.nanmean(model_results['mean_elevations']):.1f} m"
        )

        print(
            f"  > Mean terrain ruggedness (TRI): "
            f"{np.nanmean(model_results['mean_tris']):.1f} m"
        )

        print(
            f"  > Error vs mean elevation: "
            f"Spearman rho={mean_rho:.3f}, "
            f"p={mean_p:.4f}"
        )

        print(
            f"  > Error vs terrain ruggedness (TRI): "
            f"Spearman rho={tri_rho:.3f}, "
            f"p={tri_p:.4f}"
        )

    # CLOSE INPUT DATA

    close_dem_tiles(dem_tiles)
    ds_stations.close()

    if not results:
        print("[Error] No valid model results found.")
        return

    # CREATE FIGURE

    plot_results(
        results=results,
        selected_models=selected_models,
        metric=metric,
        region=region,
        init_day=init_day,
    )


if __name__ == "__main__":
    main()
