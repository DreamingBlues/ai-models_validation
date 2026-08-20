# Table 3: Color-coded spatial Error Metrics for California and Los Angeles area. 
# Cell colors indicate relative performance within each region, 
# ORANGE for poorer performance, GREEN for better performance colors have been calibrated separately for both regions. 
# Metrics: RMSE, MAE, MAPE, Correlation r



import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.colorbar import ColorbarBase

from viz_config import (
    DEFAULT_REGIONS,
    LEAD_DAYS,
    LEAD_HOURS_ORDER,
    MODELS,
    PATHS,
    PLOT_WINDOW,
    VARIABLES,
)
from viz_utils import (
    clean_time_index,
    ensure_parent_dir,
    lead_label as forecast_lead_label,
    model_path,
    open_dataset_safe,
    output_path as configured_output_path,
    project_path,
    region_mask,
    trim_to_plot_window,
)

LEAD_COLORS = {
    144: "#eef5fb",
    96: "#dcecf7",
    48: "#c7dff1",
    24: "#a9cbe5",
    0: "#8eb7d8",
}

METRICS = ["RMSE", "MAE", "MAPE", "Correlation"]
DISPLAY_METRICS = ["RMSE", "MAE", "MAPE", "R"]

BETTER_LOW = {
    "RMSE": True,
    "MAE": True,
    "MAPE": True,
    "Correlation": False,
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "good_bad",
    [
        (0.0, "#91cf60"),  # better
        (0.5, "#fee08b"),
        (1.0, "#fc8d59"),  # worse
    ],
)


# HELPERS

def lead_label(hours):
    hours = int(hours)
    return "0h" if hours == 0 else f"{hours}h"


def format_value(value):
    if pd.isna(value):
        return "--"

    return f"{value:.2f}"


def metric_score(value, metric, metric_ranges):
    vmin, vmax = metric_ranges[metric]

    if (
        pd.isna(value)
        or pd.isna(vmin)
        or pd.isna(vmax)
        or vmax == vmin
    ):
        return 0.5

    raw = (value - vmin) / (vmax - vmin)

    if BETTER_LOW[metric]:
        score = raw
    else:
        score = 1.0 - raw

    return min(max(score, 0.0), 1.0)


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

    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce",
    )

    return df.loc[
        (df["date_time"] >= start)
        & (df["date_time"] <= end)
    ].copy()


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
        ds.close()

        print(f"[Warning] No stations found in region {region}")
        return None

    if var_name not in subset:
        ds.close()

        print(
            f"[Warning] Variable {var_name} "
            "not in station dataset"
        )

        return None

    print(
        f"Loaded {subset.sizes['station']} "
        f"stations for {region}"
    )

    return subset


def load_model_dataset(nc_path, var_name):
    if not nc_path.exists():
        print(f"[Warning] File not found: {nc_path}")
        return None

    ds = open_dataset_safe(nc_path)

    if var_name not in ds:
        ds.close()
        raise KeyError(f"{var_name} not found in {nc_path}")

    return ds


# METRIC COMPUTATION

def compute_gridcell_metrics_timeseries(
    ds_model,
    ds_stations,
    model_var,
    station_var,
):
    grid_lat = ds_model.latitude.values
    grid_lon = ds_model.longitude.values

    if grid_lat.ndim == 1 and grid_lon.ndim == 1:
        grid_lon_2d, grid_lat_2d = np.meshgrid(
            grid_lon,
            grid_lat,
        )

    else:
        grid_lat_2d = grid_lat
        grid_lon_2d = grid_lon

    flat_lats = grid_lat_2d.ravel()
    flat_lons = grid_lon_2d.ravel()

    grid_points = np.column_stack(
        (
            flat_lats,
            flat_lons,
        )
    )

    tree = cKDTree(grid_points)

    station_lats = ds_stations.latitude.values
    station_lons = ds_stations.longitude.values

    station_points = np.column_stack(
        (
            station_lats,
            station_lons,
        )
    )

    _, grid_indices = tree.query(
        station_points,
        k=1,
    )

    timestep_values = {}

    unique_grid_indices = np.unique(grid_indices)

    for grid_index in unique_grid_indices:
        member_mask = grid_indices == grid_index

        cell_stations = ds_stations.isel(
            station=member_mask,
        )

        obs_series = (
            cell_stations[station_var]
            .mean(
                dim="station",
                skipna=True,
            )
            .to_series()
        )

        model_series = (
            ds_model[model_var]
            .sel(
                latitude=flat_lats[grid_index],
                longitude=flat_lons[grid_index],
                method="nearest",
            )
            .to_series()
        )

        obs_series = clean_time_index(obs_series)
        model_series = clean_time_index(model_series)

        obs_series = trim_to_plot_window(obs_series)
        model_series = trim_to_plot_window(model_series)

        common_times = obs_series.index.intersection(
            model_series.index
        )

        for time in common_times:
            model_value = scalar_value(
                model_series.loc[time]
            )

            observed_value = scalar_value(
                obs_series.loc[time]
            )

            if (
                pd.notna(model_value)
                and pd.notna(observed_value)
            ):
                if time not in timestep_values:
                    timestep_values[time] = {
                        "model_vals": [],
                        "obs_vals": [],
                    }

                timestep_values[time][
                    "model_vals"
                ].append(model_value)

                timestep_values[time][
                    "obs_vals"
                ].append(observed_value)

    results = []

    for time in sorted(timestep_values):
        model_values = np.asarray(
            timestep_values[time]["model_vals"],
            dtype=float,
        )

        observed_values = np.asarray(
            timestep_values[time]["obs_vals"],
            dtype=float,
        )

        valid = (
            np.isfinite(model_values)
            & np.isfinite(observed_values)
        )

        model_values = model_values[valid]
        observed_values = observed_values[valid]

        if len(model_values) == 0:
            continue

        errors = model_values - observed_values

        rmse = np.sqrt(
            np.mean(errors ** 2)
        )

        mae = np.mean(
            np.abs(errors)
        )

        valid_mape = np.abs(observed_values) > 1e-6

        if valid_mape.sum() > 0:
            mape = (
                np.mean(
                    np.abs(
                        errors[valid_mape]
                        / observed_values[valid_mape]
                    )
                )
                * 100.0
            )

        else:
            mape = np.nan

        if (
            len(model_values) >= 2
            and np.nanstd(model_values) > 0
            and np.nanstd(observed_values) > 0
        ):
            correlation = np.corrcoef(
                model_values,
                observed_values,
            )[0, 1]

        else:
            correlation = np.nan

        results.append(
            {
                "time": time,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "r": correlation,
                "n_grid_cells": len(model_values),
            }
        )

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .set_index("time")
        .sort_index()
    )


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
            lead_name, lead_hours = forecast_lead_label(day)
            model_file = model_path(model_key, day)

            print(f"Loading Day {day}: {model_file}")

            ds_model = load_model_dataset(
                model_file,
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

            for time, row in metrics_df.iterrows():
                records.append(
                    {
                        "model_key": model_key,
                        "model_name": model_name,
                        "run_day": day,
                        "leadtime_hr": lead_hours,
                        "lead_label": lead_name,
                        "date_time": pd.Timestamp(time).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "rmse": row["rmse"],
                        "mae": row["mae"],
                        "mape": row["mape"],
                        "r": row["r"],
                        "n_grid_cells": row["n_grid_cells"],
                        "region": region,
                        "variable": VARIABLES["model_wind"],
                        "units": VARIABLES["units"],
                    }
                )

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

def load_or_compute_metrics(
    region,
    selected_models,
    write_csv,
):
    csv_file = configured_output_path(
        "spatial_metrics",
        region=region,
    )

    if csv_file.exists() and not write_csv:
        print(f"Loading existing CSV: {csv_file}")

        df = pd.read_csv(csv_file)

        required_columns = [
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

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Existing CSV is missing required "
                f"columns: {missing_columns}. "
                "Run with --csv to rebuild it."
            )

        df["date_time"] = pd.to_datetime(
            df["date_time"],
            errors="coerce",
        )

        df = df.loc[
            (df["region"] == region)
            & (
                df["model_key"].isin(
                    selected_models
                )
            )
        ].copy()

        if df.empty:
            raise ValueError(
                "Existing CSV was found, but it does "
                "not contain rows for the requested "
                "region/models. Run with --csv to "
                "rebuild it."
            )

        return df

    if write_csv:
        print(
            "Computing metrics and replacing the "
            "spatial metrics CSV..."
        )

    else:
        print(
            f"Spatial metrics CSV not found: "
            f"{csv_file}"
        )

        print(
            "Computing metrics without writing a CSV..."
        )

    df = compute_all_metrics(
        region=region,
        selected_models=selected_models,
    )

    if df.empty:
        raise ValueError(
            f"No spatial metrics were computed "
            f"for region {region}."
        )

    if write_csv:
        ensure_parent_dir(csv_file)

        df.to_csv(
            csv_file,
            index=False,
        )

        print(f"Saved CSV: {csv_file}")

    return df


def summarize_spatial_metrics(
    df,
    selected_models,
):
    df = filter_df_to_plot_window(df)

    if df.empty:
        raise ValueError(
            "No spatial metric rows are available "
            "inside the plot window. Run with --csv "
            "to rebuild the spatial metrics CSV."
        )

    df["run_day"] = (
        df["run_day"]
        .astype(str)
        .str.zfill(2)
    )

    df["leadtime_hr"] = pd.to_numeric(
        df["leadtime_hr"],
        errors="coerce",
    )

    for metric in [
        "rmse",
        "mae",
        "mape",
        "r",
    ]:
        df[metric] = pd.to_numeric(
            df[metric],
            errors="coerce",
        )

    summary = (
        df.groupby(
            [
                "region",
                "model_key",
                "model_name",
                "run_day",
                "leadtime_hr",
                "lead_label",
            ],
            as_index=False,
            dropna=False,
        )[
            [
                "rmse",
                "mae",
                "mape",
                "r",
            ]
        ]
        .mean()
    )

    summary = summary.rename(
        columns={
            "leadtime_hr": "Lead_Hours",
            "rmse": "RMSE",
            "mae": "MAE",
            "mape": "MAPE",
            "r": "Correlation",
        }
    )

    summary = summary.dropna(
        subset=["Lead_Hours"]
    )

    summary["Lead_Hours"] = (
        summary["Lead_Hours"]
        .astype(int)
    )

    lead_rank = {
        lead: index
        for index, lead in enumerate(
            LEAD_HOURS_ORDER
        )
    }

    model_rank = {
        model: index
        for index, model in enumerate(
            selected_models
        )
    }

    summary["_lead_rank"] = (
        summary["Lead_Hours"]
        .map(lead_rank)
    )

    summary["_model_rank"] = (
        summary["model_key"]
        .map(model_rank)
    )

    summary = summary.sort_values(
        [
            "_model_rank",
            "_lead_rank",
        ]
    )

    summary = summary.drop(
        columns=[
            "_model_rank",
            "_lead_rank",
        ]
    )

    return summary.reset_index(drop=True)


def get_metric_ranges(df_region):
    ranges = {}

    for metric in METRICS:
        values = np.sort(df_region[metric].dropna().unique())

        if len(values) >= 3:
            vmin = values[1]
            vmax = values[-2]
        elif len(values) == 2:
            vmin, vmax = values[0], values[1]
        elif len(values) == 1:
            vmin = vmax = values[0]
        else:
            vmin = vmax = np.nan

        ranges[metric] = (vmin, vmax)

    return ranges


# TABLE DRAWING

def draw_metric_table(
    ax,
    df_model,
    metric_ranges,
    show_header=True,
):
    ax.axis("off")

    rows = []

    for _, row in df_model.iterrows():
        lead = int(row["Lead_Hours"])

        rows.append(
            [lead_label(lead)]
            + [
                format_value(row[metric])
                for metric in METRICS
            ]
        )

    if show_header:
        table = ax.table(
            cellText=rows,
            colLabels=[
                "Lead",
                *DISPLAY_METRICS,
            ],
            cellLoc="center",
            loc="center",
            colWidths=[
                0.16,
                0.14,
                0.14,
                0.14,
                0.14,
            ],
        )

        header_offset = 1

    else:
        table = ax.table(
            cellText=rows,
            cellLoc="center",
            loc="center",
            colWidths=[
                0.16,
                0.14,
                0.14,
                0.14,
                0.14,
            ],
        )

        header_offset = 0

    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.45)

    header_color = "white"
    edge_color = to_rgba("black", 0.22)

    if show_header:
        for column in range(5):
            cell = table[(0, column)]

            cell.set_facecolor(header_color)

            cell.set_text_props(
                color="black",
                weight="bold",
            )

            cell.set_edgecolor(edge_color)
            cell.set_linewidth(1)

    for index, (_, row) in enumerate(
        df_model.iterrows()
    ):
        table_row = index + header_offset
        lead = int(row["Lead_Hours"])

        lead_cell = table[(table_row, 0)]

        lead_cell.set_facecolor(
            LEAD_COLORS.get(
                lead,
                "#b7b7b7",
            )
        )

        lead_cell.set_text_props(
            color="black",
            weight="600",
        )

        lead_cell.set_edgecolor(edge_color)
        lead_cell.set_linewidth(1)

        for column, metric in enumerate(
            METRICS,
            start=1,
        ):
            score = metric_score(
                value=row[metric],
                metric=metric,
                metric_ranges=metric_ranges,
            )

            cell = table[
                (table_row, column)
            ]

            cell.set_facecolor(
                HEATMAP_CMAP(score)
            )

            cell.set_edgecolor(edge_color)
            cell.set_linewidth(1)

    return table


def plot_side_by_side_tables(
    region_dfs,
    selected_models,
):
    nrows = len(selected_models)
    ncols = len(DEFAULT_REGIONS)

    metric_ranges_by_region = {
        region: get_metric_ranges(
            region_dfs[region]
        )
        for region in DEFAULT_REGIONS
    }

    fig = plt.figure(
        figsize=(
            9.8,
            1.65 * nrows + 1.6,
        ),
        dpi=300,
    )

    height_ratios = (
        [1.18]
        + [1.0] * (nrows - 1)
    )

    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols + 1,
        width_ratios=[
            0.30,
            1.0,
            1.0,
        ],
        height_ratios=height_ratios,
        hspace=0.18,
        wspace=0.08,
    )

    for row_index, model_key in enumerate(
        selected_models
    ):
        model_name = MODELS[model_key]

        label_ax = fig.add_subplot(
            gs[row_index, 0]
        )

        label_ax.axis("off")

        label_ax.text(
            1.02,
            0.5,
            model_name,
            ha="right",
            va="center",
            fontsize=11.5,
            fontweight="bold",
        )

        for column_index, region in enumerate(
            DEFAULT_REGIONS
        ):
            ax = fig.add_subplot(
                gs[
                    row_index,
                    column_index + 1,
                ]
            )

            if row_index == 0:
                ax.set_title(
                    region,
                    fontsize=13,
                    fontweight="bold",
                    pad=10,
                )

            df_region = region_dfs[region]

            df_model = df_region.loc[
                df_region["model_key"]
                == model_key
            ].copy()

            if df_model.empty:
                ax.axis("off")

                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

                continue

            draw_metric_table(
                ax=ax,
                df_model=df_model,
                metric_ranges=(
                    metric_ranges_by_region[
                        region
                    ]
                ),
                show_header=(
                    row_index == 0
                ),
            )

    fig.suptitle(
        "Spatial Error Metrics by Lead Time "
        "— Grid-Cell Comparison",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )

    colorbar_axis = fig.add_axes(
        [
            0.38,
            0.045,
            0.28,
            0.018,
        ]
    )

    colorbar = ColorbarBase(
        colorbar_axis,
        cmap=HEATMAP_CMAP.reversed(),
        norm=Normalize(
            vmin=0.0,
            vmax=1.0,
        ),
        orientation="horizontal",
    )

    colorbar.set_ticks(
        [
            0.0,
            1.0,
        ]
    )

    colorbar.set_ticklabels(
        [
            "Worse",
            "Better",
        ]
    )

    fig.text(
        0.5,
        0.075,
        "Color scale is relative "
        "within each region",
        ha="center",
        va="center",
        fontsize=9,
    )

    plt.subplots_adjust(
        left=0.045,
        right=0.985,
        top=0.915,
        bottom=0.13,
    )

    output_path = (
        project_path(PATHS["figs"])
        / "tables"
        / "spatial_error_metrics_CA_LA.png"
    )

    ensure_parent_dir(output_path)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved table figure: {output_path}"
    )


# MAIN

def main():
    parser = argparse.ArgumentParser(description="Create side-by-side CA/LA spatial error metric tables.")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()), choices=list(MODELS.keys()))
    parser.add_argument("--csv", action="store_true", help="Recompute and save spatial metrics.")

    args = parser.parse_args()

    selected_models = args.models

    print(f"Models: {selected_models}")
    print(f"Regions: {DEFAULT_REGIONS}")

    region_dfs = {}

    for region in DEFAULT_REGIONS:
        timestep_df = load_or_compute_metrics(
            region=region,
            selected_models=selected_models,
            write_csv=args.csv,
        )

        region_dfs[region] = summarize_spatial_metrics(
            df=timestep_df,
            selected_models=selected_models,
        )

    plot_side_by_side_tables(
        region_dfs=region_dfs,
        selected_models=selected_models,
    )


if __name__ == "__main__":
    main()
