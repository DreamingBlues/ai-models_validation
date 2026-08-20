#!/usr/bin/env python3
# This script requires high-res data to have already been processed.
# RUN nbm_process_raw_data.py BEFORE!
# Abtin Olaee 2026

import argparse
import pathlib

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


# CONFIGURATION
CONFIG = {
    "highres_nbm_template": (
        "/shome/u014930890/pge_projects/model-comparison/"
        "nbm/processed_data/highres/nbm_processed_CA_Day{day}.nc"
    ),

    "ifs_template_path": (
        "/shome/u014930890/pge_projects/model-comparison/"
        "ifs/processed_data/ifs_processed_CA_Day01.nc"
    ),

    "output_nc_template": (
        "/shome/u014930890/pge_projects/model-comparison/"
        "nbm/processed_data/nbm_processed_CA_Day{day}.nc"
    ),

    "model_name": "NBM",
    "description": "NBM surface variables regridded to exact IFS grid",
    "geojson": "Con_Cali_Border_WGS84.geojson",

    # Conservative cutoff in lon/lat degrees.
    # Prevents distant nearest-neighbor filling outside true NBM coverage.
    "max_distance_deg": 0.20,
}

VARS = ["air_temp", "wind_speed", "wind_direction"]



def normalize_lon(lon):
    return ((lon + 180.0) % 360.0) - 180.0


def get_ifs_layout(ds_ifs):
    """
    Get exact IFS spatial dimension names and target lat/lon grid.
    Expected final format:
        variable(time, latitude, longitude)
        latitude(latitude)
        longitude(longitude)
    """
    template_var = None
    for var in VARS:
        if var in ds_ifs:
            template_var = var
            break

    if template_var is None:
        raise KeyError(f"IFS template missing expected variables: {VARS}")

    dims = ds_ifs[template_var].dims

    if "time" in dims:
        spatial_dims = tuple(d for d in dims if d != "time")
    else:
        spatial_dims = dims

    if len(spatial_dims) != 2:
        raise ValueError(f"IFS variable must have 2 spatial dims, got {spatial_dims}")

    y_dim, x_dim = spatial_dims

    if "latitude" not in ds_ifs.coords or "longitude" not in ds_ifs.coords:
        raise KeyError("IFS template must contain latitude and longitude coordinates.")

    lat = ds_ifs["latitude"]
    lon = ds_ifs["longitude"]

    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError(
            "This final-format writer expects IFS latitude and longitude to be 1D."
        )

    if lat.dims[0] != y_dim or lon.dims[0] != x_dim:
        raise ValueError(
            f"IFS coordinate dims do not match variable dims: "
            f"lat {lat.dims}, lon {lon.dims}, variable spatial dims {spatial_dims}"
        )

    lon2d, lat2d = np.meshgrid(
        normalize_lon(lon.values.astype(np.float64)),
        lat.values.astype(np.float64),
    )

    return template_var, y_dim, x_dim, lat2d, lon2d


def get_ifs_mask(ds_ifs, var, y_dim, x_dim):
    """
    Use the IFS valid-data footprint so final NBM has the same spatial mask.
    """
    if var not in ds_ifs:
        return np.ones((ds_ifs.sizes[y_dim], ds_ifs.sizes[x_dim]), dtype=bool)

    da = ds_ifs[var]

    if "time" in da.dims:
        arr = da.transpose("time", y_dim, x_dim).values
        return np.any(np.isfinite(arr), axis=0)

    arr = da.transpose(y_dim, x_dim).values
    return np.isfinite(arr)


def validate_highres_nbm(ds_nbm):
    required_coords = ["latitude", "longitude"]
    required_vars = VARS + ["ca_mask"]

    for name in required_coords:
        if name not in ds_nbm.coords:
            raise KeyError(f"High-res NBM missing coordinate: {name}")

    for name in required_vars:
        if name not in ds_nbm:
            raise KeyError(f"High-res NBM missing variable: {name}")

    lat = ds_nbm["latitude"]
    lon = ds_nbm["longitude"]

    if lat.ndim != 2 or lon.ndim != 2:
        raise ValueError(
            "High-res NBM must use 2D latitude(y,x) and longitude(y,x)."
        )

    if lat.dims != lon.dims:
        raise ValueError("High-res NBM latitude and longitude dims do not match.")

    y_dim, x_dim = lat.dims

    for var in VARS:
        expected_dims = ("time", y_dim, x_dim)
        if ds_nbm[var].dims != expected_dims:
            raise ValueError(
                f"{var} dims are {ds_nbm[var].dims}, expected {expected_dims}"
            )

    if ds_nbm["ca_mask"].dims != (y_dim, x_dim):
        raise ValueError(
            f"ca_mask dims are {ds_nbm['ca_mask'].dims}, expected {(y_dim, x_dim)}"
        )

    return y_dim, x_dim


def build_nbm_to_ifs_mapping(
    ds_nbm,
    target_lat2d,
    target_lon2d,
    max_distance_deg,
):
    """
    Build one nearest-neighbor mapping from native NBM cells to IFS target cells.

    Important:
    - Uses NBM ca_mask as the valid source footprint.
    - Does not use variable finite values to choose neighbors.
    - Applies distance cutoff to prevent filling distant missing areas.
    """
    src_lat = ds_nbm["latitude"].values.astype(np.float64)
    src_lon = normalize_lon(ds_nbm["longitude"].values.astype(np.float64))
    ca_mask = ds_nbm["ca_mask"].values.astype(bool)

    coord_valid = np.isfinite(src_lat) & np.isfinite(src_lon)
    source_valid = ca_mask & coord_valid

    if not np.any(source_valid):
        raise ValueError("No valid NBM source cells found inside ca_mask.")

    src_points = np.column_stack([
        src_lon[source_valid],
        src_lat[source_valid],
    ])

    tgt_points = np.column_stack([
        target_lon2d.ravel(),
        target_lat2d.ravel(),
    ])

    tree = cKDTree(src_points)
    dist, idx = tree.query(tgt_points, k=1)

    within_cutoff = dist <= max_distance_deg

    source_flat_indices = np.flatnonzero(source_valid.ravel())
    nearest_source_flat = source_flat_indices[idx]

    return nearest_source_flat, within_cutoff.reshape(target_lat2d.shape), dist.reshape(target_lat2d.shape)


def remap_var_to_ifs(
    ds_nbm,
    var,
    src_y_dim,
    src_x_dim,
    nearest_source_flat,
    target_shape,
    allowed_target_mask,
):
    """
    Map NBM values to IFS grid.

    This does not interpolate.
    It takes the exact nearest native NBM grid-cell value.
    """
    da = ds_nbm[var].transpose("time", src_y_dim, src_x_dim)
    src = da.values.astype(np.float32)

    nt = src.shape[0]
    ny, nx = target_shape

    out = np.full((nt, ny, nx), np.nan, dtype=np.float32)

    src_flat = src.reshape(nt, -1)
    mapped_flat = src_flat[:, nearest_source_flat]

    allowed_flat = allowed_target_mask.ravel()

    out_flat = out.reshape(nt, -1)
    out_flat[:, allowed_flat] = mapped_flat[:, allowed_flat]

    return out


def copy_ifs_final_coords(ds_ifs):
    """
    Copy IFS 1D latitude/longitude coordinates exactly.
    Time is replaced with NBM time.
    """
    coords = {}

    for coord_name in ["latitude", "longitude"]:
        coord = ds_ifs[coord_name]
        coords[coord_name] = (
            coord.dims,
            coord.values,
            dict(coord.attrs),
        )

    return coords


def validate_output(ds_out, ds_ifs, y_dim, x_dim):
    for var in VARS:
        expected_dims = ("time", y_dim, x_dim)
        if ds_out[var].dims != expected_dims:
            raise ValueError(f"{var} dims are {ds_out[var].dims}, expected {expected_dims}")

        if ds_out[var].sizes[y_dim] != ds_ifs.sizes[y_dim]:
            raise ValueError(f"{var} {y_dim} size mismatch.")

        if ds_out[var].sizes[x_dim] != ds_ifs.sizes[x_dim]:
            raise ValueError(f"{var} {x_dim} size mismatch.")

    np.testing.assert_array_equal(
        ds_out["latitude"].values,
        ds_ifs["latitude"].values,
        err_msg="Output latitude does not exactly match IFS latitude.",
    )

    np.testing.assert_array_equal(
        ds_out["longitude"].values,
        ds_ifs["longitude"].values,
        err_msg="Output longitude does not exactly match IFS longitude.",
    )



def main():
    parser = argparse.ArgumentParser(
        description="Regrid high-res NBM buffer to exact low-res IFS grid"
    )
    parser.add_argument("--day", type=str, required=True, help="Day string, e.g. 01 or 05")

    args = parser.parse_args()

    highres_path = pathlib.Path(CONFIG["highres_nbm_template"].format(day=args.day))
    ifs_path = pathlib.Path(CONFIG["ifs_template_path"])
    output_nc = pathlib.Path(CONFIG["output_nc_template"].format(day=args.day))

    if not highres_path.exists():
        raise FileNotFoundError(f"Missing high-res NBM file: {highres_path}")

    if not ifs_path.exists():
        raise FileNotFoundError(f"Missing IFS template file: {ifs_path}")

    print("Step 1/5: Load high-res NBM buffer...")
    ds_nbm = xr.open_dataset(highres_path)

    src_y_dim, src_x_dim = validate_highres_nbm(ds_nbm)

    print(f"High-res NBM source dims: {src_y_dim}, {src_x_dim}")
    print(f"High-res NBM source shape: {ds_nbm.sizes[src_y_dim]} x {ds_nbm.sizes[src_x_dim]}")

    print("Step 2/5: Load IFS template grid...")
    ds_ifs = xr.open_dataset(ifs_path)

    template_var, y_dim, x_dim, target_lat2d, target_lon2d = get_ifs_layout(ds_ifs)

    print(f"IFS template variable: {template_var}")
    print(f"IFS final dims: time, {y_dim}, {x_dim}")
    print(f"IFS final shape: {ds_ifs.sizes[y_dim]} x {ds_ifs.sizes[x_dim]}")

    print("Step 3/5: Build NBM-to-IFS nearest-neighbor mapping...")
    nearest_source_flat, within_cutoff, distance = build_nbm_to_ifs_mapping(
        ds_nbm=ds_nbm,
        target_lat2d=target_lat2d,
        target_lon2d=target_lon2d,
        max_distance_deg=CONFIG["max_distance_deg"],
    )

    print(f"Distance cutoff: {args.max_distance_deg} degrees")
    print(f"IFS cells within cutoff: {int(within_cutoff.sum())} / {within_cutoff.size}")
    print(f"Nearest distance range: {float(np.nanmin(distance)):.4f} to {float(np.nanmax(distance)):.4f}")

    print("Step 4/5: Remap NBM variables to IFS grid...")
    mapped = {}

    for var in VARS:
        print(f"  Mapping {var}...")

        ifs_mask = get_ifs_mask(ds_ifs, var, y_dim, x_dim)
        allowed_target_mask = ifs_mask & within_cutoff

        mapped[var] = remap_var_to_ifs(
            ds_nbm=ds_nbm,
            var=var,
            src_y_dim=src_y_dim,
            src_x_dim=src_x_dim,
            nearest_source_flat=nearest_source_flat,
            target_shape=(ds_ifs.sizes[y_dim], ds_ifs.sizes[x_dim]),
            allowed_target_mask=allowed_target_mask,
        )

        finite_count = int(np.isfinite(mapped[var][0]).sum()) if mapped[var].shape[0] > 0 else 0
        print(f"    finite cells at first time: {finite_count} / {mapped[var].shape[1] * mapped[var].shape[2]}")

    print("Step 5/5: Build final low-res NBM dataset...")

    coords = {
        "time": (
            ["time"],
            pd.to_datetime(ds_nbm["time"].values),
            dict(ds_nbm["time"].attrs) if "time" in ds_nbm.coords else {},
        )
    }

    coords.update(copy_ifs_final_coords(ds_ifs))

    ds_out = xr.Dataset(
        data_vars={
            "air_temp": (
                ["time", y_dim, x_dim],
                mapped["air_temp"],
                {
                    "units": "degC",
                    "standard_name": "air_temperature",
                    "long_name": "2-meter air temperature",
                },
            ),
            "wind_speed": (
                ["time", y_dim, x_dim],
                mapped["wind_speed"],
                {
                    "units": "m s-1",
                    "standard_name": "wind_speed",
                    "long_name": "10-meter wind speed",
                },
            ),
            "wind_direction": (
                ["time", y_dim, x_dim],
                mapped["wind_direction"],
                {
                    "units": "degree",
                    "standard_name": "wind_from_direction",
                    "long_name": "10-meter wind direction",
                },
            ),
        },
        coords=coords,
        attrs={
            "model": CONFIG["model_name"],
            "description": CONFIG["description"],
            "init_time": ds_nbm.attrs.get("init_time", ""),
            "geojson": CONFIG["geojson"],
            "template_grid": str(ifs_path),
            "source_file": str(highres_path),
            "source_grid": "NBM high-res curvilinear buffer",
            "regrid_method": "nearest_neighbor_no_interpolation",
            "max_distance_deg": float(args.max_distance_deg),
            "resolution_deg": float(abs(ds_ifs["latitude"].values[1] - ds_ifs["latitude"].values[0]))
            if ds_ifs["latitude"].size > 1 else np.nan,
            "original_grid_shape": list(ds_nbm.attrs.get("original_grid_shape", [])),
            "conventions": "CF-1.8",
        },
    )

    print("Validating final NBM coordinates against IFS...")
    validate_output(ds_out, ds_ifs, y_dim, x_dim)

    chunk_lat = min(ds_out.sizes[y_dim], 64)
    chunk_lon = min(ds_out.sizes[x_dim], 64)

    enc = {
        v: {
            "zlib": True,
            "shuffle": True,
            "complevel": 5,
            "_FillValue": np.float32(np.nan),
            "chunksizes": (1, chunk_lat, chunk_lon),
        }
        for v in ds_out.data_vars
    }

    output_nc.parent.mkdir(parents=True, exist_ok=True)

    if output_nc.exists():
        output_nc.unlink()

    ds_out.to_netcdf(output_nc, engine="h5netcdf", encoding=enc)

    print(f"Saved: {output_nc}")
    print("Confirmed: final NBM uses exact IFS latitude/longitude coordinates and layout.")


if __name__ == "__main__":
    main()