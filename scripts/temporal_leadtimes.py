# Side-by-side lead-time wind speed forecast comparison
# Layout: Model label | Wind Speed label | California plots | Los Angeles plots
# Abtin Olaee 2026

import argparse
import sys

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter, DayLocator
from matplotlib.lines import Line2D

from viz_config import (
    DEFAULT_REGIONS,
    LEAD_DAYS,
    MODELS,
    PATHS,
    PLOT_WINDOW,
    TEMPORAL_KEY_COLUMNS,
    VARIABLES,
)
from viz_utils import (
    clean_time_index,
    ensure_parent_dir,
    lead_label,
    model_path,
    open_dataset_safe,
    output_path as configured_output_path,
    project_path,
    region_mask,
    trim_to_plot_window,
    upsert_csv_columns,
)

LEAD_COLORS = {
    "01": "#c6dbef",  # 144h
    "03": "#9ecae1",  # 96h
    "05": "#6baed6",  # 48h
    "06": "#2171b5",  # 24h
    "07": "#08306b",  # 0h
}


def load_situ_series(region):
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


# =============================================================================
# DATA PREP
# =============================================================================

def build_plot_data(regions, selected_models):
    """
    Output:
        data[region]["situ"] = observed regional mean
        data[region]["models"][model_key][day] = model regional mean
        region_ymax[region] = max y value for that region only
    """

    data = {}
    region_ymax = {}

    for region in regions:
        situ_series, station_coords = load_situ_series(region)

        if situ_series is None or not station_coords:
            continue

        data[region] = {
            "situ": situ_series,
            "models": {},
        }

        ymax = np.nanmax(situ_series.values) if not situ_series.empty else 0.0

        for model_key in selected_models:
            data[region]["models"][model_key] = {}

            for day in LEAD_DAYS:
                model_series = load_model_series(
                    model_key=model_key,
                    day=day,
                    station_coords=station_coords,
                )

                if model_series is None or model_series.empty:
                    continue

                data[region]["models"][model_key][day] = model_series
                ymax = max(ymax, np.nanmax(model_series.values))

        region_ymax[region] = ymax

    return data, region_ymax


def update_temporal_csv(data, regions, selected_models, write_csv):
    if not write_csv:
        print("--csv not set. Skipping CSV output.")
        return

    for region in regions:
        if region not in data:
            continue

        records = []
        situ_series = data[region]["situ"]

        for model_key in selected_models:
            model_name = MODELS[model_key]
            model_data = data[region]["models"].get(model_key, {})

            for day, model_series in model_data.items():
                lead_name, lead_hours = lead_label(day)
                common_times = model_series.index.intersection(situ_series.index)

                if len(common_times) == 0:
                    continue

                model = model_series.loc[common_times].astype(float)
                obs = situ_series.loc[common_times].astype(float)

                valid = np.isfinite(model.values) & np.isfinite(obs.values)
                model = model.loc[valid]
                obs = obs.loc[valid]

                if model.empty or obs.empty:
                    continue

                for t in model.index:
                    records.append({
                        "region": region,
                        "model_key": model_key,
                        "model_name": model_name,
                        "run_day": day,
                        "leadtime_hr": lead_hours,
                        "lead_label": lead_name,
                        "date_time": pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S"),
                        "model_wind_speed": float(model.loc[t]),
                        "observed_wind_speed": float(obs.loc[t]),
                    })

        if not records:
            print(f"[Warning] No CSV rows calculated for {region}")
            continue

        csv_path = configured_output_path("temporal_metrics", region=region)
        rows_written = upsert_csv_columns(csv_path, records, TEMPORAL_KEY_COLUMNS)

        print(f"Updated CSV with {rows_written} lead-time rows: {csv_path}")


# =============================================================================
# PLOTTING
# =============================================================================

def plot_side_by_side(data, region_ymax, regions, selected_models):
    nrows = len(selected_models)
    ncols = len(regions)

    fig = plt.figure(
        figsize=(12.5, 2.75 * nrows),
        dpi=350,
    )

    # Layout:
    # col 0 = model labels
    # col 1 = California plots
    # col 2 = Los Angeles plots
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols + 1,
        width_ratios=[0.23, 0.5, 0.5],
        hspace=0.32,
        wspace=0.30,
    )

    plot_axes = {}

    for row_idx, model_key in enumerate(selected_models):
        model_name = MODELS[model_key]

        # Model-name label column
        label_ax = fig.add_subplot(gs[row_idx, 0])
        label_ax.axis("off")
        label_ax.text(
            0.98,
            0.5,
            model_name,
            ha="right",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

        for col_idx, region in enumerate(regions):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])

            plot_axes[(row_idx, col_idx)] = ax

            if row_idx == 0:
                ax.set_title(
                    region,
                    fontsize=13,
                    fontweight="bold",
                    pad=10,
                )

            if region not in data:
                ax.axis("off")
                continue

            model_series_by_day = data[region]["models"].get(model_key, {})
            situ_series = data[region]["situ"]

            if not model_series_by_day:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.grid(True, alpha=0.3)
                continue

            for day in LEAD_DAYS:
                if day not in model_series_by_day:
                    continue

                label, _ = lead_label(day)

                ax.plot(
                    model_series_by_day[day].index,
                    model_series_by_day[day].values,
                    color=LEAD_COLORS.get(day, "gray"),
                    linewidth=1.7,
                    alpha=0.95,
                    label=label,
                )

            ax.plot(
                situ_series.index,
                situ_series.values,
                color="black",
                linestyle="--",
                linewidth=2.0,
                zorder=10,
                label="Synoptic (Truth)",
            )

            # Separate y-axis scale by region.
            ymax = region_ymax.get(region, 0.0)
            if ymax > 0:
                ax.set_ylim(0, ymax * 1.12)

            ax.set_xlim(
                pd.Timestamp(PLOT_WINDOW["start"]),
                pd.Timestamp(PLOT_WINDOW["end"]),
            )

            ax.grid(True, alpha=0.3)

            ax.xaxis.set_major_locator(DayLocator())
            ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))

    # One shared y-axis label on the left side of the plot area
    fig.text(
        0.235,
        0.5,
        f"Wind Speed ({VARIABLES['units']})",
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
    )

    fig.suptitle(
        "10-m Wind Speed Forecasts by Lead Time — Regional Mean Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    fig.supxlabel(
        "Date (UTC)",
        fontsize=12,
        y=0.055,
    )

    legend_handles = []

    for day in LEAD_DAYS:
        label, _ = lead_label(day)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=LEAD_COLORS.get(day, "gray"),
                linewidth=2.5,
                label=label,
            )
        )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="Synoptic (Truth)",
        )
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.autofmt_xdate(rotation=35, ha="right")

    plt.subplots_adjust(
        left=0.05,
        right=0.99,
        top=0.95,
        bottom=0.10,
    )

    out_path = configured_output_path("temporal_leadtimes")
    ensure_parent_dir(out_path)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create side-by-side CA/LA lead-time wind speed comparison."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Models to process, in the order given."
    )
    
    parser.add_argument(
        "--csv",
        default=False,
        action="store_true",
        help="Save CSV output."
    )


    args = parser.parse_args()

    selected_models = args.models
    regions = DEFAULT_REGIONS

    print(f"Models: {selected_models}")
    print(f"Regions: {regions}")

    data, region_ymax = build_plot_data(
        regions=regions,
        selected_models=selected_models,
    )

    if not data:
        print("[Error] No data loaded. Exiting.")
        sys.exit(1)

    update_temporal_csv(
        data=data,
        regions=regions,
        selected_models=selected_models,
        write_csv=args.csv,
    )

    plot_side_by_side(
        data=data,
        region_ymax=region_ymax,
        regions=regions,
        selected_models=selected_models,
    )


if __name__ == "__main__":
    main()
