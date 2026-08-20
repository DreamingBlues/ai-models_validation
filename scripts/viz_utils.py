# Shared helper functions for publication visual scripts to reduce redundancy

from pathlib import Path

from viz_config import (
    DEFAULT_REGIONS,
    MODELS,
    NETWORK_COLORS,
    NETWORK_NAMES,
    OUTPUTS,
    PATHS,
    PLOT_WINDOW,
    PROJECT_ROOT,
    REF_DAY,
    REGIONS,
    VARIABLES,
)


NETWORK_VAR_CANDIDATES = [
    "network",
    "network_id",
    "mnet_id",
    "network_number",
    "station_network",
    "source_network",
]


def project_path(path):
    """Return an absolute path under the publication_visuals directory."""
    return PROJECT_ROOT / Path(path)


def model_path(model_key, day):
    """Return the processed model NetCDF path for a model key and init day."""
    return project_path(PATHS["model"].format(model_key=model_key, day=day))


def output_path(output_key, **kwargs):
    """Return a configured output path by key."""
    return project_path(OUTPUTS[output_key].format(**kwargs))


def ensure_parent_dir(path):
    """Create a path's parent directory and return the path as a Path object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def open_dataset_safe(path):
    """Open a NetCDF dataset, preferring h5netcdf with an xarray fallback."""
    import xarray as xr

    try:
        return xr.open_dataset(path, engine="h5netcdf")
    except Exception:
        return xr.open_dataset(path)


def lead_hours(day, ref_day=REF_DAY):
    """Return forecast lead hours for an initialization day."""
    return (int(ref_day) - int(day)) * 24


def lead_label(day, ref_day=REF_DAY):
    """Return a display label and hour value for an initialization day."""
    hours = lead_hours(day, ref_day=ref_day)
    return f"{hours}h Lead", hours


def clean_time_index(series):
    """Normalize a pandas Series index to naive sorted datetimes."""
    import pandas as pd

    series.index = pd.to_datetime(series.index)

    if getattr(series.index, "tz", None) is not None:
        series.index = series.index.tz_localize(None)

    return series.sort_index()


def trim_to_period(series, start, end):
    """Trim a Series to [start, end)."""
    import pandas as pd

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    return series.loc[(series.index >= start) & (series.index <= end)]


def trim_to_plot_window(series):
    """Trim a Series to the configured publication plot window."""
    return trim_to_period(series, PLOT_WINDOW["start"], PLOT_WINDOW["end"])


def load_station_series(region):
    """Load the regional mean station series and its station coordinates."""
    nc_path = project_path(PATHS["station"])
    var_name = VARIABLES["station_wind"]

    if not nc_path.exists():
        print(f"[Error] Station file not found: {nc_path}")
        return None, []

    ds = open_dataset_safe(nc_path)

    if var_name not in ds:
        print(f"[Error] Variable {var_name} not found in station dataset.")
        return None, []

    mask = region_mask(
        ds.latitude.values,
        ds.longitude.values,
        region,
    )

    subset = ds.isel(station=mask)

    if subset.sizes["station"] == 0:
        print(f"[Warning] No stations found for {region}")
        return None, []

    station_coords = list(zip(subset.latitude.values, subset.longitude.values))

    series = subset[var_name].mean(dim="station", skipna=True).to_series()
    series = clean_time_index(series)
    series = trim_to_plot_window(series)

    print(f"{region}: loaded {subset.sizes['station']} stations")

    return series, station_coords


def load_model_series(model_key, day, station_coords):
    """Load a model series sampled at the supplied station coordinates."""
    import numpy as np
    import xarray as xr

    nc_path = model_path(model_key, day)

    if not nc_path.exists():
        print(f"[Warning] Missing model file: {nc_path}")
        return None

    ds = open_dataset_safe(nc_path)

    var_name = VARIABLES["model_wind"]

    if var_name not in ds:
        raise KeyError(f"{var_name} not found in {nc_path}")

    target_lats = np.array([c[0] for c in station_coords])
    target_lons = np.array([c[1] for c in station_coords])

    selected = ds[var_name].sel(
        latitude=xr.DataArray(target_lats, dims="station_id"),
        longitude=xr.DataArray(target_lons, dims="station_id"),
        method="nearest",
    )

    series = selected.mean(dim="station_id", skipna=True).to_series()
    series = clean_time_index(series)
    series = trim_to_plot_window(series)

    return series.dropna()


def region_bounds(region):
    """Return minlon, maxlon, minlat, maxlat for a configured region."""
    return REGIONS[region]


def region_mask(latitudes, longitudes, region):
    """Return a boolean mask for points inside a configured region."""
    import numpy as np

    minlon, maxlon, minlat, maxlat = region_bounds(region)
    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)

    return (
        (latitudes >= minlat)
        & (latitudes <= maxlat)
        & (longitudes >= minlon)
        & (longitudes <= maxlon)
    )


def resolve_model_order(selected_models=None):
    """Return selected model keys, preserving the configured default order."""
    if selected_models is None or len(selected_models) == 0:
        return list(MODELS.keys())

    return list(selected_models)


def resolve_regions(selected_regions=None):
    """Return selected regions, preserving the configured default order."""
    if selected_regions is None or len(selected_regions) == 0:
        return list(DEFAULT_REGIONS)

    return list(selected_regions)


def detect_network_var(ds, configured_name=None):
    """Find the station network variable in a station dataset."""
    if configured_name is not None:
        if configured_name not in ds:
            raise KeyError(f"{configured_name} not found in station dataset.")
        return configured_name

    for name in NETWORK_VAR_CANDIDATES:
        if name in ds:
            return name

    raise KeyError("Could not auto-detect station network variable.")


def network_label(value):
    """Return the display label for a station network id or raw value."""
    try:
        value_int = int(value)
    except Exception:
        return str(value)

    return NETWORK_NAMES.get(value_int, f"Network {value_int}")


def network_color(value, default=None):
    """Return the configured color for a station network id or label."""
    try:
        return NETWORK_COLORS.get(int(value), default)
    except Exception:
        return network_colors_by_name().get(str(value), default)


def network_colors_by_name():
    """Return network colors keyed by display label."""
    return {
        NETWORK_NAMES[network_id]: color
        for network_id, color in NETWORK_COLORS.items()
        if network_id in NETWORK_NAMES
    }


def upsert_csv_columns(csv_path, records, key_columns):
    """
    Create or update a CSV by matching rows on key_columns.

    Non-key columns in records are added or overwritten for matching rows.
    """
    import pandas as pd

    csv_path = Path(csv_path)
    new_df = pd.DataFrame(records)

    if new_df.empty:
        return 0

    for column in key_columns:
        if column not in new_df:
            new_df[column] = pd.NA

    for column in key_columns:
        new_df[column] = new_df[column].astype(str)

    new_df = new_df.drop_duplicates(subset=key_columns, keep="last")

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path, dtype={column: str for column in key_columns})
    else:
        existing_df = pd.DataFrame(columns=key_columns)

    for column in key_columns:
        if column not in existing_df:
            existing_df[column] = pd.NA
        existing_df[column] = existing_df[column].astype(str)

    existing_df = existing_df.drop_duplicates(subset=key_columns, keep="last")

    value_columns = [column for column in new_df.columns if column not in key_columns]

    combined = existing_df.merge(
        new_df,
        on=key_columns,
        how="outer",
        suffixes=("", "__new"),
    )

    for column in value_columns:
        new_column = f"{column}__new"

        if new_column in combined:
            if column in combined:
                combined[column] = combined[new_column].combine_first(combined[column])
                combined = combined.drop(columns=[new_column])
            else:
                combined = combined.rename(columns={new_column: column})

    ordered_columns = list(key_columns)
    ordered_columns.extend(column for column in combined.columns if column not in ordered_columns)
    combined = combined[ordered_columns]

    ensure_parent_dir(csv_path)
    combined.to_csv(csv_path, index=False)

    return len(new_df)
