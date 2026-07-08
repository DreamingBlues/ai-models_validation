#!/usr/bin/env python3
# Abtin Olaee 2026

import json
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

CONFIG = {
    "input_json": "./sensorData/situ_data.json",
    "pge_input": "./sensorData/pge_agl.csv",
    "era5_z0": "./sensorData/era5_z0.nc",

    "output_station_nc": "./sensorData/station_data.nc",

    "start_time": "2025-01-01 00:00:00",
    "end_time": "2025-01-11 00:00:00",
    "freq": "6h",

    "network_sensor_agl": {
        1: 10.0,    # ASOS/AWOS
        2: 6.1,     # RAWS
        229: 6.1,   # PG&E
        231: 7.62,  # SCE
    },

    "pge_network_id": 229,
}


def clean_var_name(var_name):
    return var_name.replace("_set_1", "").replace("_set_2d", "")

# STEP 1: Extrapolate Json into Xarray
def build_station_dataset(data):
    master_time = pd.date_range(
        start=CONFIG["start_time"],
        end=CONFIG["end_time"],
        freq=CONFIG["freq"],
    )

    stations_list = data.get("STATION", [])
    if not stations_list:
        raise ValueError("JSON does not contain 'STATION' list")

    station_ids = []
    network_ids = []
    lats = []
    lons = []
    elevs = []
    aligned_data = []
    all_observed_vars = set()

    print("Aligning station observations to 6-hour timesteps...")

    for stat in tqdm(stations_list, unit="station", desc="Extracting stations"):
        sid = stat.get("STID")
        if not sid:
            continue

        obs = stat.get("OBSERVATIONS", {})
        if "date_time" not in obs:
            continue

        try:
            lat = float(stat.get("LATITUDE", np.nan))
            lon = float(stat.get("LONGITUDE", np.nan))
            elev = float(stat.get("ELEVATION", np.nan))
        except (ValueError, TypeError):
            continue

        date_time = obs.get("date_time", [])
        n_time = len(date_time)

        safe_obs = {"date_time": date_time}

        for key, values in obs.items():
            if key != "date_time" and isinstance(values, list) and len(values) == n_time:
                safe_obs[key] = values

        df = pd.DataFrame(safe_obs)
        df["date_time"] = pd.to_datetime(df["date_time"])
        df = df.set_index("date_time")
        df = df[~df.index.duplicated(keep="first")]

        df_aligned = df.reindex(
            master_time,
            method="nearest",
            tolerance=pd.Timedelta("30min")
        )

        station_ids.append(sid)
        network_ids.append(stat.get("MNET_ID", np.nan))
        lats.append(lat)
        lons.append(lon)
        elevs.append(elev)
        aligned_data.append(df_aligned)

        for col in df_aligned.columns:
            all_observed_vars.add(clean_var_name(col))

    n_stations = len(station_ids)
    n_times = len(master_time)

    print(f"Building dataset: {n_stations} stations x {n_times} timesteps")

    ds = xr.Dataset(
        coords={
            "station": (["station"], station_ids),
            "time": (["time"], master_time),
            "latitude": (["station"], lats),
            "longitude": (["station"], lons),
            "elevation": (["station"], elevs),
            "network": (["station"], network_ids),
        }
    )

    blacklist = {
        "QC_SUMMARY",
        "qc_summary",
        "weather_cond_code",
        "cloud_layer_1_code",
        "cloud_layer_2_code",
        "cloud_layer_3_code",
    }

    for var_name in tqdm(sorted(all_observed_vars), desc="Building variables"):
        if var_name in blacklist:
            continue

        matrix = np.full((n_stations, n_times), np.nan, dtype=np.float32)

        for i, df in enumerate(aligned_data):
            matching_cols = [c for c in df.columns if clean_var_name(c) == var_name]

            if not matching_cols:
                continue

            try:
                numeric_data = pd.to_numeric(df[matching_cols[0]].values, errors="coerce")
                matrix[i, :] = numeric_data.astype(np.float32)
            except Exception:
                continue

        if np.any(np.isfinite(matrix)):
            ds[var_name] = (["station", "time"], matrix)

    return ds



# STEP 2: extract sensor heights
def get_wind_sensor_agl(stat):
    sensor_vars = stat.get("SENSOR_VARIABLES", {})

    # backup in case older merged file has typo
    if not sensor_vars:
        sensor_vars = stat.get("SENSOR_VARIABELS", {})

    wind_meta = sensor_vars.get("wind_speed", {})

    for _, sensor_info in wind_meta.items():
        if isinstance(sensor_info, dict) and "position" in sensor_info:
            try:
                return float(sensor_info["position"])
            except (TypeError, ValueError):
                pass

    return np.nan


def build_json_agl_lookup(data):
    agl_lookup = {}

    for stat in data.get("STATION", []):
        stid = stat.get("STID")
        if not stid:
            continue

        agl_lookup[str(stid).strip()] = get_wind_sensor_agl(stat)

    return agl_lookup


def build_pge_csv_agl_lookup(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"WARNING: {path} not found. PG&E stations will use network fallback.")
        return {}

    df["stid"] = df["stid"].astype(str).str.strip()
    df["Wind Monitor Height"] = pd.to_numeric(
        df["Wind Monitor Height"],
        errors="coerce"
    )

    # feet to meters
    df["Wind Monitor Height"] = df["Wind Monitor Height"] * 0.3048

    lookup = {}

    for _, row in df.iterrows():
        stid = row["stid"]
        h = row["Wind Monitor Height"]

        if stid not in lookup or np.isnan(lookup[stid]):
            lookup[stid] = np.float32(h) if pd.notna(h) else np.nan

    return lookup


def attach_height_agl(ds, json_lookup, pge_lookup):
    station_ids = ds["station"].values
    networks = ds["network"].values

    height_agl = np.full(len(station_ids), np.nan, dtype=np.float32)
    source = np.full(len(station_ids), "missing", dtype=object)

    for i, stid_raw in enumerate(station_ids):
        stid = str(stid_raw).strip()

        try:
            network_id = int(networks[i])
        except (TypeError, ValueError):
            network_id = None

        # 1. Try Synoptic JSON metadata
        h = json_lookup.get(stid, np.nan)
        if np.isfinite(h):
            height_agl[i] = np.float32(h)
            source[i] = "json"
            continue

        # 2. Try PG&E CSV
        if network_id == CONFIG["pge_network_id"]:
            h = pge_lookup.get(stid, np.nan)
            if np.isfinite(h):
                height_agl[i] = np.float32(h)
                source[i] = "pge_csv"
                continue

        # 3. Fall back to network assumptions
        if network_id in CONFIG["network_sensor_agl"]:
            height_agl[i] = np.float32(CONFIG["network_sensor_agl"][network_id])
            source[i] = "network_fallback"

    ds["height_agl"] = (["station"], height_agl)
    ds["height_agl"].attrs["long_name"] = "wind sensor height above ground level"
    ds["height_agl"].attrs["units"] = "m"

    ds["height_agl_source"] = (["station"], source.astype(str))
    ds["height_agl_source"].attrs["long_name"] = "source used for height_agl"
    ds["height_agl_source"].attrs["values"] = "json, pge_csv, network_fallback, missing"

    return ds


# STEP 3: SAMPLE ERA5 ROUGHNESS AND CORRECT WIND SPEED TO 10 M
def sample_station_z0(ds, era5_path):
    era = xr.open_dataset(era5_path)

    z0_field = era["z0"]

    station_lat = xr.DataArray(ds["latitude"].values, dims="station")
    station_lon = xr.DataArray(ds["longitude"].values, dims="station")

    z0_station = z0_field.sel(
        latitude=station_lat,
        longitude=station_lon,
        method="nearest"
    )

    return z0_station.astype(np.float32)


def correct_to_10m(ws, z, z0):
    valid = (
        np.isfinite(ws) &
        np.isfinite(z) &
        np.isfinite(z0) &
        (z > 0.0) &
        (z0 > 0.0) &
        (z > z0) &
        (10.0 > z0)
    )

    factor = xr.where(valid, np.log(10.0 / z0) / np.log(z / z0), np.nan)
    return xr.where(valid, ws * factor, np.nan).astype(np.float32)


def add_corrected_wind(ds):
    if "wind_speed" not in ds:
        raise KeyError("Dataset does not contain 'wind_speed'")

    print("Sampling ERA5 z0 at station locations...")
    ds["z0_era5"] = sample_station_z0(ds, CONFIG["era5_z0"])
    ds["z0_era5"].attrs["long_name"] = "ERA5 surface roughness sampled at station location"
    ds["z0_era5"].attrs["units"] = "m"

    print("Computing corrected 10 m wind speed...")
    ds["ws_10m_corr"] = correct_to_10m(
        ds["wind_speed"],
        xr.DataArray(ds["height_agl"].values, dims="station"),
        ds["z0_era5"]
    )

    ds["ws_10m_corr"].attrs["long_name"] = "wind speed corrected to 10 m using ERA5 roughness"
    ds["ws_10m_corr"].attrs["units"] = ds["wind_speed"].attrs.get("units", "unknown")
    ds["ws_10m_corr"].attrs["formula"] = "ws_10m_corr = wind_speed * ln(10/z0) / ln(z/z0)"

    return ds


# SAVE
def save_netcdf(ds, path):
    encoding = {}

    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.floating):
            encoding[v] = {"zlib": True, "complevel": 1, "_FillValue": np.nan}
        else:
            encoding[v] = {}

    ds.to_netcdf(path, encoding=encoding, engine="h5netcdf")
    print(f"Saved: {path}")


# MAIN
def main():
    print(f"Loading {CONFIG['input_json']}...")
    with open(CONFIG["input_json"], "r", encoding="utf-8") as f:
        data = json.load(f)

    ds = build_station_dataset(data)

    #save_netcdf(ds, CONFIG["output_station_nc"])

    json_lookup = build_json_agl_lookup(data)
    pge_lookup = build_pge_csv_agl_lookup(CONFIG["pge_input"])

    print("Attaching height_agl from JSON, PG&E CSV, then network fallback...")
    ds = attach_height_agl(ds, json_lookup, pge_lookup)

    ds = add_corrected_wind(ds)

    save_netcdf(ds, CONFIG["output_station_nc"])

    print("\nSummary:")
    print(f"  stations total: {ds.sizes['station']}")
    print(f"  height_agl from JSON: {int(np.sum(ds['height_agl_source'].values == 'json'))}")
    print(f"  height_agl from PG&E CSV: {int(np.sum(ds['height_agl_source'].values == 'pge_csv'))}")
    print(f"  height_agl from network fallback: {int(np.sum(ds['height_agl_source'].values == 'network_fallback'))}")
    print(f"  height_agl still missing: {int(np.sum(ds['height_agl_source'].values == 'missing'))}")
    print(f"  finite ws_10m_corr values: {int(np.sum(np.isfinite(ds['ws_10m_corr'].values)))}")


if __name__ == "__main__":
    main()