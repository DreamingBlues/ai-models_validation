# Appendix Figures B
# Spatial error metric dashboard
# Layout: metrics as rows, models as columns
# Metrics: RMSE, MAE, MAPE, spatial Pearson r


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from scipy.spatial import cKDTree
from matplotlib.dates import DateFormatter, DayLocator
from matplotlib.lines import Line2D

from viz_config import (
    DEFAULT_REGION,
    LEAD_DAYS,
    LEAD_HOURS_ORDER,
    MODELS,
    PATHS,
    PLOT_WINDOW,
    REGIONS,
    VARIABLES,
)
from viz_utils import (
    clean_time_index,
    ensure_parent_dir,
    lead_label,
    model_path,
    open_dataset_safe,
    output_path,
    project_path,
    region_mask,
    resolve_model_order,
    trim_to_plot_window,
)


WRITE_CSV = False

METRIC_COLORS = {
    "rmse": "#1F77B4",   # deep blue
    "mae": "#17BECF",    # cyan blue
    "mape": "#FF7F0E",   # orange
    "r": "#9467BD",      # purple
}


METRIC_INFO = {
    "rmse": {
        "title": "RMSE",
        "ylabel": f"RMSE ({VARIABLES['units']})",
    },
    "mae": {
        "title": "MAE",
        "ylabel": f"MAE ({VARIABLES['units']})",
    },
    "mape": {
        "title": "MAPE",
        "ylabel": "MAPE (%)",
    },
    "r": {
        "title": "Pearson r",
        "ylabel": "Pearson r",
    },
}


# Manual y-axis limits. None uses the data range.
Y_LIMITS = {
    "rmse": None,
    "mae": None,
    "mape": None,
    "r": (-1.05, 1.05),
}


def scalar_value(value):
    if isinstance(value, pd.Series):
        value = value.dropna()
        if value.empty:
            return np.nan
        value = value.iloc[0]

    return float(value)


def filter_df_to_plot_window(df):
    start = pd.Timestamp(PLOT_WINDOW["start"])
    end = pd.Timestamp(PLOT_WINDOW["end"])

    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])

    return df.loc[
        (df["date_time"] >= start) &
        (df["date_time"] <= end)
    ]


# DATA LOADING

def load_station_subset(nc_path, region, var_name):
    nc_path = project_path(nc_path)

    if not nc_path.exists():
        print(f"[Error] Station NetCDF not found: {nc_path}")
        return None

    ds = open_dataset_safe(nc_path)

    mask = region_mask(
        ds.latitude.values,
        ds.longitude.values,
        region,
    )

    subset = ds.isel(station=mask)

    if subset.sizes["station"] == 0:
        print(f"[Warning] No stations found in region {region}")
        return None

    if var_name not in subset:
        print(f"[Warning] Variable {var_name} not in station dataset")
        return None

    print(f"Loaded {subset.sizes['station']} stations for {region}")

    return subset


def load_model_dataset(nc_path, var_name):
    if not nc_path.exists():
        print(f"[Warning] File not found: {nc_path}")
        return None

    ds = open_dataset_safe(nc_path)

    if var_name not in ds:
        raise KeyError(f"'{var_name}' not in dataset. Vars: {list(ds.data_vars.keys())}")

    return ds


# METRIC COMPUTATION

def compute_gridcell_metrics_timeseries(ds_model, ds_stations, model_var, station_var):
    grid_lat = ds_model.latitude.values
    grid_lon = ds_model.longitude.values

    if grid_lat.ndim == 1 and grid_lon.ndim == 1:
        grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)
    else:
        grid_lat_2d, grid_lon_2d = grid_lat, grid_lon

    flat_lats = grid_lat_2d.ravel()
    flat_lons = grid_lon_2d.ravel()

    grid_points = np.column_stack((flat_lats, flat_lons))
    tree = cKDTree(grid_points)

    st_lats = ds_stations.latitude.values
    st_lons = ds_stations.longitude.values
    st_points = np.column_stack((st_lats, st_lons))

    _, grid_indices = tree.query(st_points, k=1)

    timestep_values = {}

    unique_grid_indices = np.unique(grid_indices)

    for idx in unique_grid_indices:
        member_mask = grid_indices == idx

        cell_stations = ds_stations.isel(station=member_mask)

        obs_series = cell_stations[station_var].mean(
            dim="station",
            skipna=True
        ).to_series()

        lat_val = flat_lats[idx]
        lon_val = flat_lons[idx]

        model_point = ds_model[model_var].sel(
            latitude=lat_val,
            longitude=lon_val,
            method="nearest"
        )

        model_series = model_point.to_series()

        obs_series = clean_time_index(obs_series)
        model_series = clean_time_index(model_series)

        obs_series = trim_to_plot_window(obs_series)
        model_series = trim_to_plot_window(model_series)

        common_times = obs_series.index.intersection(model_series.index)

        for t in common_times:
            m_val = scalar_value(model_series.loc[t])
            o_val = scalar_value(obs_series.loc[t])

            if pd.notna(m_val) and pd.notna(o_val):
                if t not in timestep_values:
                    timestep_values[t] = {
                        "model_vals": [],
                        "obs_vals": [],
                    }

                timestep_values[t]["model_vals"].append(m_val)
                timestep_values[t]["obs_vals"].append(o_val)

    results = []

    for t in sorted(timestep_values.keys()):
        m_arr = np.array(timestep_values[t]["model_vals"], dtype=float)
        o_arr = np.array(timestep_values[t]["obs_vals"], dtype=float)

        valid = np.isfinite(m_arr) & np.isfinite(o_arr)

        m_arr = m_arr[valid]
        o_arr = o_arr[valid]

        if len(m_arr) == 0:
            continue

        errors = m_arr - o_arr

        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))

        valid_mape = np.abs(o_arr) > 1e-6
        if valid_mape.sum() > 0:
            mape = np.mean(np.abs(errors[valid_mape] / o_arr[valid_mape])) * 100.0
        else:
            mape = np.nan

        if (
            len(m_arr) >= 2 and
            np.nanstd(m_arr) > 0 and
            np.nanstd(o_arr) > 0
        ):
            r = np.corrcoef(m_arr, o_arr)[0, 1]
        else:
            r = np.nan

        results.append({
            "time": t,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r": r,
            "n_grid_cells": len(m_arr),
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).set_index("time")

    return df.sort_index()


def compute_all_metrics(region, selected_models):
    ds_stations = load_station_subset(
        PATHS["station"],
        region,
        VARIABLES["station_wind"],
    )

    if ds_stations is None:
        return pd.DataFrame()

    records = []

    for model_key in selected_models:
        model_name = MODELS[model_key]

        print(f"\nProcessing model: {model_name}")

        for day in LEAD_DAYS:
            label, lead_hours = lead_label(day)
            fpath = model_path(model_key, day)

            print(f"Loading Day {day}: {fpath}")

            ds_model = load_model_dataset(
                fpath,
                VARIABLES["model_wind"],
            )

            if ds_model is None:
                print(f"Skipped Day {day}: model file is missing.")
                continue

            metrics_df = compute_gridcell_metrics_timeseries(
                ds_model=ds_model,
                ds_stations=ds_stations,
                model_var=VARIABLES["model_wind"],
                station_var=VARIABLES["station_wind"],
            )
            ds_model.close()

            if metrics_df.empty:
                print(f"Skipped Day {day}: metrics dataframe is empty.")
                continue

            for t, row in metrics_df.iterrows():
                records.append({
                    "model_key": model_key,
                    "model_name": model_name,
                    "run_day": day,
                    "leadtime_hr": lead_hours,
                    "lead_label": label,
                    "date_time": pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S"),
                    "rmse": row["rmse"],
                    "mae": row["mae"],
                    "mape": row["mape"],
                    "r": row["r"],
                    "n_grid_cells": row["n_grid_cells"],
                    "region": region,
                    "variable": VARIABLES["model_wind"],
                    "units": VARIABLES["units"],
                })

    ds_stations.close()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    df["date_time"] = pd.to_datetime(df["date_time"])

    df["_model_order"] = pd.Categorical(
        df["model_key"], categories=selected_models, ordered=True
    )
    df["_lead_order"] = pd.Categorical(
        df["leadtime_hr"], categories=LEAD_HOURS_ORDER, ordered=True
    )
    df = df.sort_values(["_model_order", "_lead_order", "date_time"])
    df = df.drop(columns=["_model_order", "_lead_order"])

    return df


# CSV HANDLING

def load_or_compute_metrics(region, selected_models, write_csv):
    csv_file = output_path("spatial_metrics", region=region)

    if csv_file.exists() and not write_csv:
        print(f"Loading existing CSV: {csv_file}")

        df = pd.read_csv(csv_file)
        df["date_time"] = pd.to_datetime(df["date_time"])

        required_cols = [
            "model_key",
            "model_name",
            "run_day",
            "leadtime_hr",
            "lead_label",
            "date_time",
            "rmse",
            "mae",
            "mape",
            "r",
            "region",
        ]

        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Existing CSV is missing required columns: {missing_cols}. "
                "Run with --csv to rebuild it."
            )

        df = df[df["region"] == region]
        df = df[df["model_key"].isin(selected_models)]

        if df.empty:
            raise ValueError(
                "Existing CSV was found, but it does not contain rows for the "
                "requested region/models. Run with --csv to rebuild it."
            )

        return df

    if write_csv:
        print("Computing metrics from model/station data and writing CSV...")
    else:
        print("Computing metrics from model/station data without writing CSV...")

    df = compute_all_metrics(
        region=region,
        selected_models=selected_models
    )

    if df.empty:
        print("[Error] No metrics were computed.")
        return df

    if not write_csv:
        return df

    ensure_parent_dir(csv_file)
    df.to_csv(csv_file, index=False)

    print(f"Saved CSV: {csv_file}")

    return df


# PLOTTING

def compute_metric_y_limits(df):
    y_limits = {}

    for metric in METRIC_COLORS:
        manual_limits = Y_LIMITS.get(metric)

        if manual_limits is not None:
            y_limits[metric] = manual_limits
            continue

        values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()

        if values.empty:
            y_limits[metric] = None
            continue

        ymin = values.min()
        ymax = values.max()

        if metric in ["rmse", "mae", "mape"]:
            ymin = 0.0

        if ymax == ymin:
            ymax = ymax + 1.0

        padding = (ymax - ymin) * 0.12

        y_limits[metric] = (ymin, ymax + padding)

    return y_limits


def plot_metric_dashboard(df, region, selected_models):
    df = filter_df_to_plot_window(df)

    if df.empty:
        print("[Error] No rows available inside plot window.")
        return

    n_models = len(selected_models)
    n_metrics = len(METRIC_COLORS)

    fig_width = max(2.8 * n_models, 11)
    fig_height = 9.5

    fig, axes = plt.subplots(
        nrows=n_metrics,
        ncols=n_models,
        figsize=(fig_width, fig_height),
        sharex=True,
        dpi=150
    )

    lead_alpha = {
        "01": 0.35,  # 144h
        "03": 0.50,  # 96h
        "05": 0.65,  # 48h
        "06": 0.80,  # 24h
        "07": 1.00,  # 0h
    }

    axes = np.asarray(axes).reshape(n_metrics, n_models)

    y_limits = compute_metric_y_limits(df)

    plot_start = pd.Timestamp(PLOT_WINDOW["start"])
    plot_end = pd.Timestamp(PLOT_WINDOW["end"])

    for col_idx, model_key in enumerate(selected_models):
        model_df = df[df["model_key"] == model_key]
        model_name = MODELS[model_key]

        for row_idx, metric in enumerate(METRIC_COLORS):
            ax = axes[row_idx, col_idx]

            for day in LEAD_DAYS:
                day_df = model_df[
                    model_df["run_day"].astype(str).str.zfill(2) == day
                ]

                if day_df.empty:
                    continue

                day_df = day_df.sort_values("date_time")

                label, _ = lead_label(day)

                ax.plot(
                    day_df["date_time"],
                    day_df[metric],
                    color=METRIC_COLORS.get(metric, "gray"),
                    linewidth=1.7,
                    alpha=lead_alpha.get(day, 0.95),
                    label=label
                )

            ax.grid(True, alpha=0.3)

            ax.set_xlim(left=plot_start, right=plot_end)

            if y_limits.get(metric) is not None:
                ax.set_ylim(*y_limits[metric])

            ax.xaxis.set_major_locator(DayLocator())
            ax.xaxis.set_major_formatter(DateFormatter("%d"))

            if row_idx == 0:
                ax.set_title(
                    model_name,
                    fontsize=11,
                    fontweight="bold"
                )

            if col_idx == 0:
                ax.set_ylabel(
                    METRIC_INFO[metric]["ylabel"],
                    fontsize=10,
                    fontweight="bold",
                    color="black"
                )
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)

            if row_idx < n_metrics - 1:
                ax.tick_params(labelbottom=False)

    fig.suptitle(
        f"Spatial Error Metrics Across Lead Times - {region}",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )

    fig.supxlabel("January 2025 (UTC)", fontsize=13, y=0.055)

    legend_handles = []

    for day in LEAD_DAYS:
        label, _ = lead_label(day)

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=2.5,
                alpha=lead_alpha.get(day, 1.0),
                label=label
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, 0.005)
    )

    plt.tight_layout(rect=[0.035, 0.075, 1.0, 0.955])
    fig.subplots_adjust(hspace=0.18, wspace=0.08)

    out_file = output_path("spatial_error", region=region)
    ensure_parent_dir(out_file)

    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {out_file}")


# MAIN

def main():
    parser = argparse.ArgumentParser(description="Plot spatial error dashboard with models as columns.")
    parser.add_argument("--region", default=DEFAULT_REGION, choices=list(REGIONS.keys()))
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()), choices=list(MODELS.keys()))
    parser.add_argument("--csv", default=WRITE_CSV, action="store_true", help="Recompute and replace the spatial metrics CSV.")

    args = parser.parse_args()

    region = args.region
    selected_models = resolve_model_order(args.models)

    print(f"Region: {region}")
    print(f"Model order: {selected_models}")
    print(f"Write CSV: {args.csv}")

    df = load_or_compute_metrics(
        region=region,
        selected_models=selected_models,
        write_csv=args.csv,
    )

    plot_metric_dashboard(
        df=df,
        region=region,
        selected_models=selected_models,
    )


if __name__ == "__main__":
    main()
