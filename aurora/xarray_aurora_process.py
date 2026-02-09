#!/usr/bin/env python3
# Abtin Olaee 2025

import pygrib
import numpy as np
import pathlib
import sys
import json
import argparse
import pandas as pd
import xarray as xr
from datetime import datetime
from matplotlib.path import Path as MplPath

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "grib_path_template": "/shome/u014930890/PGE_Projects/aurora_raw_data/data/aurora-2.5-pretrained_{day}.grib",
    "geojson_path": "/shome/u014930890/PGE_Projects/Con_Cali_Border_WGS84.geojson",
    "var_ref_path": "/shome/u014930890/PGE_Projects/aurora_new/synoptic_varlist_aurora.csv",
    "output_nc_template": "/shome/u014930890/PGE_Projects/aurora_new/processed_data/aurora_processed_CA_Day{day}.nc",
    "model_name": "National Blend of Models (NBM)",
    "description": "NBM surface variables masked to CA GeoJSON",
}

# ==============================================================================
# SIMPLE PROGRESS BAR (no external deps)
# ==============================================================================

def progress(prefix, i, total, width=40):
    if total <= 0:
        return
    frac = i / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    pct = int(frac * 100)
    sys.stdout.write(f"\r{prefix} [{bar}] {pct:3d}% ({i}/{total})")
    sys.stdout.flush()
    if i >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()

# ==============================================================================
# VARLIST + UNIT NORMALIZATION
# ==============================================================================

def load_var_ref(csv_path):
    """
    CSV schema:
      standard_name,raw_short_name,standard_units,raw_units
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required_cols = {"standard_name", "raw_short_name", "standard_units", "raw_units"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in varlist: {sorted(missing)}")

    var_meta = {}
    for _, row in df.iterrows():
        std = str(row["standard_name"]).strip()
        raw = str(row["raw_short_name"]).strip()
        var_meta[std] = {
            "raw_short_name": raw,
            "raw_units": str(row["raw_units"]).strip(),
            "standard_units": str(row["standard_units"]).strip(),
        }
    return var_meta


def convert_to_standard_units(var_name, data, raw_units):
    """
    STANDARD UNITS ONLY:
      - t2m -> degC
      - u10/v10 -> m s-1
    """
    if var_name == "t2m":
        if raw_units == "K":
            return data - 273.15
        if raw_units == "degC":
            return data
        if raw_units == "degF":
            return (data - 32.0) * (5.0 / 9.0)
        raise ValueError(f"Unsupported raw units for t2m: {raw_units}")

    if var_name in ("u10", "v10"):
        if raw_units == "m s-1":
            return data
        if raw_units == "mph":
            return data / 2.2369362921
        if raw_units == "ft s-1":
            return data / 3.280839895
        raise ValueError(f"Unsupported raw units for {var_name}: {raw_units}")

    raise ValueError(f"Unknown variable: {var_name}")


def calculate_wind(u, v):
    ws = np.sqrt(u**2 + v**2)
    wd = (270 - np.degrees(np.arctan2(v, u))) % 360
    return ws, wd

# ==============================================================================
# GEOJSON MASKING
# ==============================================================================

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


def get_spatial_subset(lats, lons, geojson_path):
    print(f"Loading GeoJSON: {geojson_path}")
    with open(geojson_path) as f:
        gj = json.load(f)

    lons_norm = ((lons + 180) % 360) - 180
    flat_lats, flat_lons = lats.ravel(), lons_norm.ravel()

    mask = np.zeros(flat_lats.size, dtype=bool)
    points = np.column_stack((flat_lons, flat_lats))

    polys = list(_iter_polygons_from_geojson(gj))
    print(f"Building mask from {len(polys)} polygons...")

    total = len(polys)
    for idx, (exterior, holes) in enumerate(polys, start=1):
        path = MplPath(exterior)
        ext = path.get_extents()

        bbox_idx = (
            (flat_lons >= ext.xmin) & (flat_lons <= ext.xmax) &
            (flat_lats >= ext.ymin) & (flat_lats <= ext.ymax)
        )
        if np.any(bbox_idx):
            subset_points = points[bbox_idx]
            is_inside = path.contains_points(subset_points)
            for hole in holes:
                is_inside &= ~MplPath(hole).contains_points(subset_points)
            mask[bbox_idx] |= is_inside

        progress("Mask polygons", idx, total)

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
    print(f"Cropped Grid Shape: {mask_crop.shape}")
    return slice_y, slice_x, mask_crop

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Process GRIB -> NetCDF (CA masked), using synoptic varlist.")
    parser.add_argument("--day", type=str, required=True)
    args = parser.parse_args()

    grib_file = CONFIG["grib_path_template"].format(day=args.day)
    output_nc = CONFIG["output_nc_template"].format(day=args.day)

    print(f"Input GRIB: {grib_file}")
    print(f"Output NC:  {output_nc}")

    # Step 1: Read varlist
    print("Step 1/5: Loading varlist...")
    var_meta = load_var_ref(CONFIG["var_ref_path"])
    raw_shortnames_needed = {m["raw_short_name"] for m in var_meta.values()}
    print(f"Varlist standard_names: {list(var_meta.keys())}")
    print(f"Looking for GRIB shortNames: {sorted(raw_shortnames_needed)}")

    # Step 2: Init grid + mask
    print("Step 2/5: Opening GRIB and building CA mask...")
    grbs = pygrib.open(grib_file)
    first_msg = grbs[1]
    lats, lons = first_msg.latlons()

    slice_y, slice_x, mask = get_spatial_subset(lats, lons, CONFIG["geojson_path"])
    if slice_y is None:
        print("No data inside mask.")
        return

    lat_1d = lats[slice_y, slice_x][:, 0]
    lon_1d = ((lons[slice_y, slice_x] + 180) % 360 - 180)[0, :]

    # Step 3: Scan GRIB messages (progress by message count estimate)
    print("Step 3/5: Scanning GRIB messages...")
    # We can’t know count cheaply without iterating; count first for accurate progress.
    grbs.rewind()
    total_msgs = 0
    for _ in grbs:
        total_msgs += 1

    grbs.rewind()
    data_by_time = {}
    for i, grb in enumerate(grbs, start=1):
        sn = grb.shortName
        if sn in raw_shortnames_needed:
            ts = grb.validDate.isoformat()
            data_by_time.setdefault(ts, {})[sn] = grb.values[slice_y, slice_x]
        progress("Scan GRIB", i, total_msgs)

    grbs.close()

    times = sorted(data_by_time.keys())
    if not times:
        print("[Error] No matching GRIB messages found for requested shortNames.")
        return

    # Sanity check
    first_ts = times[0]
    print(f"First timestep: {first_ts}")
    print(f"Raw fields present: {sorted(data_by_time[first_ts].keys())}")

    # Step 4: Convert + compute derived outputs
    print("Step 4/5: Converting to standard units and computing wind products...")
    nt, ny, nx = len(times), mask.shape[0], mask.shape[1]

    t2m_c = np.full((nt, ny, nx), np.nan, dtype=np.float32)
    u10_ms = np.full((nt, ny, nx), np.nan, dtype=np.float32)
    v10_ms = np.full((nt, ny, nx), np.nan, dtype=np.float32)

    for i, ts in enumerate(times, start=1):
        raw_fields = data_by_time[ts]

        for std_name, meta in var_meta.items():
            raw_sn = meta["raw_short_name"]
            if raw_sn not in raw_fields:
                continue

            arr = raw_fields[raw_sn]
            arr = convert_to_standard_units(std_name, arr, meta["raw_units"])
            arr[~mask] = np.nan

            if std_name == "t2m":
                t2m_c[i - 1]  = arr
            elif std_name == "u10":
                u10_ms[i - 1] = arr
            elif std_name == "v10":
                v10_ms[i - 1] = arr

        progress("Process times", i, nt)

    ws, wd = calculate_wind(u10_ms, v10_ms)

    # Step 5: Build dataset + write NetCDF (simple progress: build then write)
    print("Step 5/5: Writing NetCDF...")
    ds = xr.Dataset(
        data_vars={
            "air_temp": (["time", "latitude", "longitude"], t2m_c,
                         {"units": "degC", "standard_name": "air_temperature", "long_name": "2-meter air temperature"}),
            "wind_speed": (["time", "latitude", "longitude"], ws.astype(np.float32),
                           {"units": "m s-1", "standard_name": "wind_speed", "long_name": "10-meter wind speed"}),
            "wind_direction": (["time", "latitude", "longitude"], wd.astype(np.float32),
                               {"units": "degree", "standard_name": "wind_from_direction", "long_name": "10-meter wind direction"}),
        },
        coords={
            "time": pd.to_datetime([datetime.fromisoformat(t) for t in times]),
            "latitude": (["latitude"], lat_1d, {"units": "degrees_north", "standard_name": "latitude", "axis": "Y"}),
            "longitude": (["longitude"], lon_1d, {"units": "degrees_east", "standard_name": "longitude", "axis": "X"}),
        },
        attrs={
            "title": "CA NBM Surface Forecast",
            "model": CONFIG["model_name"],
            "description": CONFIG["description"],
            "init_time": times[0],
            "geojson": pathlib.Path(CONFIG["geojson_path"]).name,
            "resolution_deg": float(abs(lat_1d[1] - lat_1d[0])) if len(lat_1d) > 1 else np.nan,
            "original_grid_shape": list(lats.shape),
            "conventions": "CF-1.8",
        },
    )

    # Compression + chunking
    chunk_lat = min(ny, 64)
    chunk_lon = min(nx, 64)
    enc = {
        v: {"zlib": True, "shuffle": True, "complevel": 5,
            "_FillValue": np.nan, "chunksizes": (1, ny, nx)}
        for v in ds.data_vars
    }

    # No fine-grained progress available inside to_netcdf
    ds.to_netcdf(output_nc, engine="h5netcdf", encoding=enc)
    print(f"Saved: {output_nc}")


if __name__ == "__main__":
    main()
