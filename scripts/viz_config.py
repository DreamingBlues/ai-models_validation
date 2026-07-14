"""Shared project constants for publication visual scripts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


PATHS = {
    "station": "./data/station_data.nc",
    "geojson": "./data/Con_Cali_Border_WGS84.geojson",
    "model": "./models/{model_key}/processed_data/{model_key}_processed_CA_Day{day}.nc",
    "figs": "./figs",
}


OUTPUTS = {
    "temporal_leadtimes": "./figs/plots/leadtime_side_by_side_CA_LA.png",
    "temporal_bias": "./figs/plots/leadtime_bias_side_by_side_CA_LA.png",
    "temporal_metrics": "./figs/data/temporal_error_metrics_{region}.csv",
    "temporal_error_table": "./figs/tables/leadtime_metrics_CA_LA.png",
    "spatial_error": "./figs/plots/spatial_error_{region}.png",
    "spatial_metrics": "./figs/data/spatial_error_metrics_{region}.csv",
    "spatial_map": (
        "./figs/maps/synoptic_and_signed_wind_difference_"
        "{lead_label}_{region}_Day{day}_{time_tag}.png"
    ),
    "error_vs_station_z0": "./figs/scatter/{metric}_density_z0_era5_{region}.png",
    "network_observations": "./figs/plots/station_observations_by_network_{region}.png",
    "network_station_map": "./figs/maps/station_network_map_{region}.png",
}


MODELS = {
    "ifs": "IFS",
    "nbm": "NBM",
    "fcn3": "FourCastNet v3",
    "fcn2": "FourCastNet v2",
    "aurora": "Aurora",
    "graphcast": "GraphCast",
}


REGIONS = {
    "CA": (-124.50, -114.00, 32.30, 42.00),
    "LA": (-118.875, -117.375, 33.625, 34.625),
}


VARIABLES = {
    "model_wind": "wind_speed",
    "station_wind": "ws_10m_corr",
    "z0": "z0_era5",
    "units": "m/s",
}


LEAD_DAYS = ["01", "03", "05", "06", "07"]
LEAD_HOURS_ORDER = [144, 96, 48, 24, 0]
REF_DAY = "07"
DEFAULT_INIT_DAY = "05"


PLOT_WINDOW = {
    "start": "2025-01-07 00:00:00",
    "end": "2025-01-11 00:00:00",
}


DEFAULT_REGION = "LA"
DEFAULT_REGIONS = ["CA", "LA"]
DEFAULT_VALID_TIME = "2025-01-08 00:00:00"
DEFAULT_LEAD_LABEL = "72-hour Lead Time"


TEMPORAL_KEY_COLUMNS = [
    "region",
    "model_key",
    "model_name",
    "run_day",
    "leadtime_hr",
    "lead_label",
    "date_time",
]


NETWORK_NAMES = {
    1: "ASOS/AWOS",
    2: "RAWS",
    229: "PG&E",
    231: "SCE",
}


NETWORK_COLORS = {
    1: "#2b6cb0",
    2: "#e66101",
    229: "#00a6c8",
    231: "#238b45",
}


NETWORK_PLOT_ORDER = [229, 231, 2, 1]


MAPE_MIN_OBS = 0.1
SPATIAL_DIFF_LIMIT = 8.0
