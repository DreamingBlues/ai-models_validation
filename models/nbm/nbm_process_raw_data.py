#!/usr/bin/env python3
# Abtin Olaee 2026

import argparse
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import pygrib
import xarray as xr
from matplotlib.path import Path as MplPath
from tqdm import tqdm


# CONFIGURATION
CONFIG = {
    # For days 1-4
    "input_dir_template": "/fs/ember-fs2/adata/afarguell/ai_models/nbm_data/202501{day}_00/core/co/",

    # For days 5-7
    #"input_dir_template": "../NBM_10day/data/202501{day}_00/core/co/",

    "geojson_path": "./Con_Cali_Border_WGS84.geojson",
    "var_ref_path": "./nbm/synoptic_varlist_nbm.csv",
    "output_nc_template": "./nbm/processed_data/highres/nbm_processed_CA_Day{day}.nc",

    "model_name": "NBM",
    "description": "High-resolution NBM buffer cropped to CA bbox with 2D source geolocation and CA mask",

    # Use 0.0 for strict polygon masking.
    # The previous 0.25 expanded the CA mask too much.
    "mask_radius_deg": 0.0,
}

OUTPUT_VARS = ["air_temp", "wind_speed", "wind_direction"]



def normalize_lon(lon):
    return ((lon + 180.0) % 360.0) - 180.0


def get_forecast_hour(filename):
    match = re.search(r"\.f(\d{3})\.", str(filename))
    if match:
        return int(match.group(1))
    return None


def load_var_ref(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required_cols = {"standard_name", "raw_short_name", "standard_units", "raw_units"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in varlist: {sorted(missing)}")

    var_meta = {}
    for _, row in df.iterrows():
        std = str(row["standard_name"]).strip()
        var_meta[std] = {
            "raw_short_name": str(row["raw_short_name"]).strip(),
            "raw_units": str(row["raw_units"]).strip(),
            "standard_units": str(row["standard_units"]).strip(),
        }

    return var_meta


def canonical_var_name(name):
    name = str(name).strip()

    if name in ("t2m", "air_temp", "air_temperature"):
        return "air_temp"

    if name in ("wind_speed", "ws", "10si"):
        return "wind_speed"

    if name in ("wind_direction", "wind_from_direction", "wdir", "10wdir"):
        return "wind_direction"

    return name


def normalize_units(units):
    return str(units).strip().lower().replace(" ", "")


def convert_to_standard_units(var_name, data, raw_units):
    var_name = canonical_var_name(var_name)
    u = normalize_units(raw_units)

    data = np.asarray(data, dtype=np.float32)

    if var_name == "air_temp":
        if u in ("k", "kelvin"):
            return data - np.float32(273.15)
        if u in ("degc", "c", "celsius", "°c"):
            return data
        if u in ("degf", "f", "fahrenheit", "°f"):
            return (data - np.float32(32.0)) * np.float32(5.0 / 9.0)
        raise ValueError(f"Unsupported raw units for air_temp: {raw_units}")

    if var_name == "wind_speed":
        if u in ("ms-1", "m/s", "ms**-1", "metersecond-1", "meterssecond-1"):
            return data
        if u == "mph":
            return data / np.float32(2.2369362921)
        if u in ("kt", "kts", "knot", "knots"):
            return data / np.float32(1.94384)
        raise ValueError(f"Unsupported raw units for wind_speed: {raw_units}")

    if var_name == "wind_direction":
        if (
            u in ("degree", "degrees", "deg", "°")
            or "degree" in u
            or "deg" in u
        ):
            return data
        raise ValueError(f"Unsupported raw units for wind_direction: {raw_units}")

    raise ValueError(f"Unknown variable: {var_name}")



def _iter_polygons_from_geojson(geojson_obj):
    def polygon_from_coords(coords):
        exterior = [(float(x), float(y)) for x, y in coords[0]]
        holes = [[(float(x), float(y)) for x, y in ring] for ring in coords[1:]]
        return exterior, holes

    features = geojson_obj.get("features", [])

    for feat in features:
        geom = feat.get("geometry", feat)

        if geom["type"] == "Polygon":
            yield polygon_from_coords(geom["coordinates"])

        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                yield polygon_from_coords(poly)


def get_spatial_subset(lats, lons, geojson_path, radius=0.0):
    """
    Returns:
        slice_y, slice_x, mask_crop

    mask_crop is True inside the CA GeoJSON polygon.
    The output bbox is the smallest y/x rectangle containing the mask.
    """
    with open(geojson_path) as f:
        gj = json.load(f)

    lons_norm = normalize_lon(lons)

    flat_lats = lats.ravel()
    flat_lons = lons_norm.ravel()
    points = np.column_stack((flat_lons, flat_lats))

    mask = np.zeros(flat_lats.size, dtype=bool)
    polys = list(_iter_polygons_from_geojson(gj))

    for exterior, holes in polys:
        path = MplPath(exterior)
        ext = path.get_extents()

        bbox_idx = (
            (flat_lons >= ext.xmin) & (flat_lons <= ext.xmax) &
            (flat_lats >= ext.ymin) & (flat_lats <= ext.ymax)
        )

        if np.any(bbox_idx):
            subset_points = points[bbox_idx]
            is_inside = path.contains_points(subset_points, radius=radius)

            for hole in holes:
                is_inside &= ~MplPath(hole).contains_points(subset_points, radius=radius)

            mask[bbox_idx] |= is_inside

    mask_2d = mask.reshape(lats.shape)

    if not np.any(mask_2d):
        return None, None, None

    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)

    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]

    slice_y = slice(y0, y1 + 1)
    slice_x = slice(x0, x1 + 1)

    mask_crop = mask_2d[slice_y, slice_x]

    return slice_y, slice_x, mask_crop




def select_first(grbs, queries):
    "Prune GRIB search queries to find successful matches, Works with find_message()"
    for query in queries:
        try:
            matches = grbs.select(**query)
            if matches:
                return matches[0]
        except (ValueError, RuntimeError, KeyError):
            pass

    return None


def find_message(grbs, canonical_name, meta):
    raw_short_name = meta.get("raw_short_name", "").strip()

    if canonical_name == "air_temp":
        queries = [
            {"shortName": raw_short_name},
            {"shortName": "2t"},
            {"shortName": "t2m"},
        ]

    elif canonical_name == "wind_speed":
        queries = [
            {"shortName": "10si"},
            {
                "shortName": "10si",
                "typeOfLevel": "heightAboveGround",
                "level": 10,
            },
            {
                "shortName": "10si",
                "typeOfLevel": "heightAboveGround",
                "level": 10,
                "productDefinitionTemplateNumber": 0,
            },
            {"shortName": raw_short_name},
        ]

    elif canonical_name == "wind_direction":
        queries = [
            {
                "shortName": "10wdir",
                "typeOfLevel": "heightAboveGround",
                "level": 10,
            },
            {"shortName": "10wdir"},
            {"shortName": "wdir", "level": 10},
            {"shortName": raw_short_name},
        ]

    else:
        return None

    return select_first(grbs, queries)


def masked_to_nan(values):
    if np.ma.isMaskedArray(values):
        return np.ma.filled(values, np.nan).astype(np.float32)

    return np.asarray(values, dtype=np.float32)




def main():
    parser = argparse.ArgumentParser(description="Process raw NBM GRIB -> high-res curvilinear NetCDF buffer")
    parser.add_argument("--day", type=str, required=True, help="Day string, e.g. '01', '05'")
    args = parser.parse_args()

    input_dir = pathlib.Path(CONFIG["input_dir_template"].format(day=args.day))
    output_nc = pathlib.Path(CONFIG["output_nc_template"].format(day=args.day))

    if not input_dir.exists():
        print(f"[Error] Input directory does not exist: {input_dir}")
        sys.exit(1)

    print("Step 1/4: Loading varlist and finding GRIB files...")
    var_meta = load_var_ref(CONFIG["var_ref_path"])

    all_files = sorted(input_dir.glob("*.grib2"))
    files_to_process = []

    for path in all_files:
        fhr = get_forecast_hour(path)
        if fhr is None:
            continue

        if fhr == 1 or (fhr > 0 and fhr % 6 == 0):
            files_to_process.append((fhr, path))

    files_to_process.sort(key=lambda x: x[0])

    if not files_to_process:
        print("[Error] No files found matching the time criteria.")
        sys.exit(1)

    print(f"Found {len(files_to_process)} forecast files.")

    print("Step 2/4: Opening first GRIB to build cropped CA bbox and 2D coordinates...")
    first_fhr, first_file = files_to_process[0]

    with pygrib.open(str(first_file)) as grbs:
        first_msg = next(grbs)
        lats, lons = first_msg.latlons()
        lats = lats.astype(np.float32)
        lons = lons.astype(np.float32)
        analysis_date = first_msg.analDate

    slice_y, slice_x, ca_mask = get_spatial_subset(
        lats=lats,
        lons=lons,
        geojson_path=CONFIG["geojson_path"],
        radius=CONFIG["mask_radius_deg"],
    )

    if slice_y is None:
        print("[Error] No NBM grid cells inside CA mask.")
        sys.exit(1)

    lat_2d = lats[slice_y, slice_x].astype(np.float32)
    lon_2d = normalize_lon(lons[slice_y, slice_x]).astype(np.float32)

    nt = len(files_to_process)
    ny, nx = ca_mask.shape

    print(f"Original NBM grid shape: {lats.shape}")
    print(f"Cropped high-res shape: {ny} x {nx}")
    print(f"CA mask cells: {int(ca_mask.sum())} / {ca_mask.size}")
    print(f"Latitude range: {float(np.nanmin(lat_2d)):.4f} to {float(np.nanmax(lat_2d)):.4f}")
    print(f"Longitude range: {float(np.nanmin(lon_2d)):.4f} to {float(np.nanmax(lon_2d)):.4f}")

    data = {
        "air_temp": np.full((nt, ny, nx), np.nan, dtype=np.float32),
        "wind_speed": np.full((nt, ny, nx), np.nan, dtype=np.float32),
        "wind_direction": np.full((nt, ny, nx), np.nan, dtype=np.float32),
    }

    times = []

    print("Step 3/4: Extracting NBM variables...")
    with tqdm(total=nt, desc="Processing NBM", unit="file") as pbar:
        for i, (fhr, filepath) in enumerate(files_to_process):
            times.append(analysis_date + pd.Timedelta(hours=fhr))

            try:
                with pygrib.open(str(filepath)) as grbs:
                    for std_name, meta in var_meta.items():
                        canonical = canonical_var_name(std_name)

                        if canonical not in data:
                            continue

                        msg = find_message(grbs, canonical, meta)

                        if msg is None:
                            continue

                        values = masked_to_nan(msg.values[slice_y, slice_x])

                        raw_units = getattr(msg, "units", None)
                        if raw_units is None or str(raw_units).strip() == "":
                            raw_units = meta["raw_units"]

                        arr = convert_to_standard_units(canonical, values, raw_units)
                        data[canonical][i, :, :] = arr.astype(np.float32)

            except Exception as e:
                print(f"\n[Warning] Error processing file f{fhr:03d}: {filepath}")
                print(f"          {e}")

            pbar.update(1)

    print("Step 4/4: Writing high-res curvilinear NetCDF buffer...")

    var_attrs = {
        "air_temp": {
            "units": "degC",
            "standard_name": "air_temperature",
            "long_name": "2-meter air temperature",
            "coordinates": "latitude longitude",
        },
        "wind_speed": {
            "units": "m s-1",
            "standard_name": "wind_speed",
            "long_name": "10-meter wind speed",
            "coordinates": "latitude longitude",
        },
        "wind_direction": {
            "units": "degree",
            "standard_name": "wind_from_direction",
            "long_name": "10-meter wind direction",
            "coordinates": "latitude longitude",
        },
    }

    ds = xr.Dataset(
        data_vars={
            "air_temp": (["time", "y", "x"], data["air_temp"], var_attrs["air_temp"]),
            "wind_speed": (["time", "y", "x"], data["wind_speed"], var_attrs["wind_speed"]),
            "wind_direction": (["time", "y", "x"], data["wind_direction"], var_attrs["wind_direction"]),

            # Keep the mask separate instead of destroying source values with NaN.
            # Downstream low-res processing can apply this mask or the IFS mask explicitly.
            "ca_mask": (
                ["y", "x"],
                ca_mask.astype(np.uint8),
                {
                    "long_name": "California GeoJSON mask on cropped NBM grid",
                    "flag_values": np.array([0, 1], dtype=np.uint8),
                    "flag_meanings": "outside_california inside_california",
                },
            ),
        },
        coords={
            "time": (
                ["time"],
                pd.to_datetime(times),
                {
                    "standard_name": "time",
                    "long_name": "forecast valid time",
                },
            ),
            "y": (
                ["y"],
                np.arange(ny, dtype=np.int32),
                {
                    "long_name": "cropped NBM y index",
                    "axis": "Y",
                },
            ),
            "x": (
                ["x"],
                np.arange(nx, dtype=np.int32),
                {
                    "long_name": "cropped NBM x index",
                    "axis": "X",
                },
            ),
            "latitude": (
                ["y", "x"],
                lat_2d,
                {
                    "units": "degrees_north",
                    "standard_name": "latitude",
                    "long_name": "latitude",
                },
            ),
            "longitude": (
                ["y", "x"],
                lon_2d,
                {
                    "units": "degrees_east",
                    "standard_name": "longitude",
                    "long_name": "longitude",
                },
            ),
        },
        attrs={
            "model": CONFIG["model_name"],
            "description": CONFIG["description"],
            "init_time": analysis_date.isoformat(),
            "geojson": pathlib.Path(CONFIG["geojson_path"]).name,
            "source_grid": "NBM native curvilinear grid",
            "spatial_dims": "y x",
            "coordinates": "latitude(y,x) longitude(y,x)",
            "mask_radius_deg": float(CONFIG["mask_radius_deg"]),
            "original_grid_shape": list(lats.shape),
            "cropped_grid_shape": [int(ny), int(nx)],
            "crop_y_start_stop": [int(slice_y.start), int(slice_y.stop)],
            "crop_x_start_stop": [int(slice_x.start), int(slice_x.stop)],
            "conventions": "CF-1.8",
        },
    )

    chunk_y = min(ny, 128)
    chunk_x = min(nx, 128)

    enc = {
        "air_temp": {
            "zlib": True,
            "shuffle": True,
            "complevel": 5,
            "_FillValue": np.float32(np.nan),
            "chunksizes": (1, chunk_y, chunk_x),
        },
        "wind_speed": {
            "zlib": True,
            "shuffle": True,
            "complevel": 5,
            "_FillValue": np.float32(np.nan),
            "chunksizes": (1, chunk_y, chunk_x),
        },
        "wind_direction": {
            "zlib": True,
            "shuffle": True,
            "complevel": 5,
            "_FillValue": np.float32(np.nan),
            "chunksizes": (1, chunk_y, chunk_x),
        },
        "ca_mask": {
            "zlib": True,
            "shuffle": True,
            "complevel": 5,
            "dtype": "uint8",
            "chunksizes": (chunk_y, chunk_x),
        },
        "latitude": {
            "zlib": True,
            "shuffle": True,
            "complevel": 5,
            "_FillValue": np.float32(np.nan),
            "chunksizes": (chunk_y, chunk_x),
        },
        "longitude": {
            "zlib": True,
            "shuffle": True,
            "complevel": 5,
            "_FillValue": np.float32(np.nan),
            "chunksizes": (chunk_y, chunk_x),
        },
    }

    output_nc.parent.mkdir(parents=True, exist_ok=True)

    if output_nc.exists():
        output_nc.unlink()

    ds.to_netcdf(output_nc, engine="h5netcdf", encoding=enc)

    print(f"Saved: {output_nc}")
    print("High-res NBM buffer written with:")
    print("  variables: time, y, x")
    print("  coordinates: latitude(y,x), longitude(y,x)")
    print("  mask: ca_mask(y,x)")


if __name__ == "__main__":
    main()