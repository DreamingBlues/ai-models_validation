#!/usr/bin/env python3
# Abtin Olaee 2026

import numpy as np
import pathlib
import sys
import json
import argparse
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from matplotlib.path import Path as MplPath

# CONFIGURATION
CONFIG = {
    "raw_path_template": "/fs/ember-fs2/adata/afarguell/ai_models/earth2studio/fcn3_det_2025-01-{day}T00:00.nc",
    "geojson_path": "./Con_Cali_Border_WGS84.geojson",
    "var_ref_path": "./fcn3/synoptic_varlist_fcn3.csv",
    "output_nc_template": "./fcn3/processed_data/fcn3_processed_CA_Day{day}.nc",
    "model_name": "FourcastnetV3 Deterministic",
    "description": "FourcastnetV3 surface variables masked to CA GeoJSON",
}

# HELPERS
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
            # allow for boundary cells to be included
            is_inside = path.contains_points(subset_points, radius=0.25)
            for hole in holes:
                is_inside &= ~MplPath(hole).contains_points(subset_points, radius=0.25)
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



def main():
    parser = argparse.ArgumentParser(description="Process MODEL -> NetCDF, using synoptic varlist.")
    parser.add_argument(        
        "--day",
        type=str,
        required=True,
        help="Day string for filename (e.g. '01', '03', '05' for 2025-01-DD)."
    )
    args = parser.parse_args()

    # FIX 1: Ensure variable names match
    model_file = CONFIG["raw_path_template"].format(day=args.day)
    output_nc = CONFIG["output_nc_template"].format(day=args.day)

    print(f"Input NetCDF: {model_file}")
    print(f"Output NC:  {output_nc}")

    # Step 1: Read varlist
    print("Step 1/5: Loading varlist...")
    var_meta = load_var_ref(CONFIG["var_ref_path"])
    # We don't strictly need raw_shortnames_needed logic anymore, but keeping it doesn't hurt.
    print(f"Varlist standard_names: {list(var_meta.keys())}")

    # Step 2: Init grid + mask
    print("Step 2/5: Opening NetCDF and building CA mask...")
    try:
        # FIX 1: Use model_file here
        ds_in = xr.open_dataset(model_file)
    except Exception as e:
        print(f"Error opening NetCDF: {e}")
        return

    # FCN3 has 1D lat/lon arrays; we need 2D meshgrid for the masker
    lat_1d_in = ds_in['lat'].values
    lon_1d_in = ds_in['lon'].values
    lons_2d, lats_2d = np.meshgrid(lon_1d_in, lat_1d_in)

    # Use lats_2d/lons_2d for the spatial subset function
    lats, lons = lats_2d, lons_2d

    slice_y, slice_x, mask = get_spatial_subset(lats, lons, CONFIG["geojson_path"])
    if slice_y is None:
        print("No data inside mask.")
        return

    # These are the cropped 1D coordinates we will use for output
    lat_1d = lats[slice_y, slice_x][:, 0]
    lon_1d = ((lons[slice_y, slice_x] + 180) % 360 - 180)[0, :]

    # Step 3: Parse Time Dimension (Lead Time handling)
    print("Step 3/5: parsing time dimension...")
    
    # Get base time (e.g., 2025-01-01 00:00)
    base_time_val = ds_in['time'].values[0]
    base_time = pd.to_datetime(base_time_val)
    
    # Get lead times (e.g., [0, 6, 12, ...])
    lead_times = ds_in['lead_time'].values
    
    # Calculate valid timestamps for output
    valid_times = [base_time + timedelta(hours=int(lt)) for lt in lead_times]
    nt = len(valid_times)
    
    print(f"Found {nt} time steps (Lead times: {lead_times[0]} to {lead_times[-1]} hrs)")

    # FIX 2: Deleted "data_by_time" check (leftover from GRIB logic)

    # Step 4: Extract Data & Convert
    print("Step 4/5: Extracting variables and converting units...")

    # Initialize output arrays (Time, Lat, Lon)
    ny, nx = mask.shape
    t2m_c = np.full((nt, ny, nx), np.nan, dtype=np.float32)
    u10_ms = np.full((nt, ny, nx), np.nan, dtype=np.float32)
    v10_ms = np.full((nt, ny, nx), np.nan, dtype=np.float32)

    for i, lt in enumerate(lead_times):
        # We access data via .isel(time=0, lead_time=i)
        
        # --- Temperature (t2m) ---
        if 't2m' in ds_in:
            # Extract and mask immediately
            raw = ds_in['t2m'].isel(time=0, lead_time=i)[slice_y, slice_x].values
            # Convert K -> C (assumes input is K)
            t2m_c[i] = raw - 273.15
            t2m_c[i][~mask] = np.nan

        # --- U Component (u10m) ---
        if 'u10m' in ds_in:
            raw = ds_in['u10m'].isel(time=0, lead_time=i)[slice_y, slice_x].values
            u10_ms[i] = raw
            u10_ms[i][~mask] = np.nan

        # --- V Component (v10m) ---
        if 'v10m' in ds_in:
            raw = ds_in['v10m'].isel(time=0, lead_time=i)[slice_y, slice_x].values
            v10_ms[i] = raw
            v10_ms[i][~mask] = np.nan
        
        progress("Processing lead times", i + 1, nt)

    ds_in.close()
    
    # Calculate Wind Speed / Direction
    ws, wd = calculate_wind(u10_ms, v10_ms)

    # Step 5: Write NetCDF (Using xarray)
    print("Step 5/5: Writing Output NetCDF...")
    ds_out = xr.Dataset(
        data_vars={
            "air_temp": (["time", "latitude", "longitude"], t2m_c,
                         {"units": "degC", "standard_name": "air_temperature", "long_name": "2-meter air temperature"}),
            "wind_speed": (["time", "latitude", "longitude"], ws.astype(np.float32),
                           {"units": "m s-1", "standard_name": "wind_speed", "long_name": "10-meter wind speed"}),
            "wind_direction": (["time", "latitude", "longitude"], wd.astype(np.float32),
                               {"units": "degree", "standard_name": "wind_from_direction", "long_name": "10-meter wind direction"}),
        },
        coords={
            # Ensure valid_times are converted to pandas datetimes to match original format
            "time": pd.to_datetime(valid_times),
            # FIX 3: Use lat_1d / lon_1d (not lat_1d_out)
            "latitude": (["latitude"], lat_1d, {"units": "degrees_north", "standard_name": "latitude", "axis": "Y"}),
            "longitude": (["longitude"], lon_1d, {"units": "degrees_east", "standard_name": "longitude", "axis": "X"}),
        },
        attrs={
            "model": CONFIG["model_name"],
            "description": CONFIG["description"],
            "init_time": valid_times[0].isoformat(),
            "geojson": pathlib.Path(CONFIG["geojson_path"]).name,
            # FIX 3: Use lat_1d
            "resolution_deg": float(abs(lat_1d[1] - lat_1d[0])) if len(lat_1d) > 1 else np.nan,
            "original_grid_shape": list(lats_2d.shape), 
            "conventions": "CF-1.8",
        },
    )

    # Compression & Saving
    chunk_lat = min(ny, 64)
    chunk_lon = min(nx, 64)
    enc = {
        v: {"zlib": True, "shuffle": True, "complevel": 5, "_FillValue": np.nan, "chunksizes": (1, ny, nx)}
        for v in ds_out.data_vars
    }

    ds_out.to_netcdf(output_nc, engine="h5netcdf", encoding=enc)
    print(f"Saved: {output_nc}")

if __name__ == "__main__":
    main()
