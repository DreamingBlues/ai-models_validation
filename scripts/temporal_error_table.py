# Table 2: Color-coded temporal Error Metrics for California and Los Angeles area. 
# Cell colors indicate relative performance within each region, 
# ORANGE for poorer performance, GREEN for better performance colors have been calibrated separately for both regions. 
# Metrics: RMSE, MAE, MAPE, Correlation r

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.colorbar import ColorbarBase

from viz_config import (
    DEFAULT_REGIONS,
    LEAD_DAYS,
    LEAD_HOURS_ORDER,
    MAPE_MIN_OBS,
    MODELS,
    TEMPORAL_KEY_COLUMNS,
    VARIABLES,
)
from viz_utils import (
    ensure_parent_dir,
    lead_label as forecast_lead_label,
    load_model_series,
    load_station_series,
    output_path,
    upsert_csv_columns,
)

LEAD_COLORS = {
    144: "#eef5fb",
    96:  "#dcecf7",
    48:  "#c7dff1",
    24:  "#a9cbe5",
    0:   "#8eb7d8",
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

    if pd.isna(value) or pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        return 0.5

    raw = (value - vmin) / (vmax - vmin)

    if BETTER_LOW[metric]:
        score = raw
    else:
        score = 1.0 - raw

    return min(max(score, 0.0), 1.0)


def compute_series_metrics(model_series, obs_series):
    common_times = model_series.index.intersection(obs_series.index)

    if len(common_times) <= 10:
        return {
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAPE": np.nan,
            "Correlation": np.nan,
        }

    model = model_series.loc[common_times].astype(float)
    obs = obs_series.loc[common_times].astype(float)

    valid = np.isfinite(model.values) & np.isfinite(obs.values)
    model = model.loc[valid]
    obs = obs.loc[valid]

    if len(model) <= 10:
        return {
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAPE": np.nan,
            "Correlation": np.nan,
        }

    err = model - obs

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    mape_mask = np.abs(obs.values) > MAPE_MIN_OBS
    if np.any(mape_mask):
        mape = float(np.mean(np.abs(err.values[mape_mask] / obs.values[mape_mask])) * 100.0)
    else:
        mape = np.nan

    if len(model) < 2 or model.nunique() <= 1 or obs.nunique() <= 1:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(model.values, obs.values)[0, 1])

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Correlation": correlation,
    }


def load_region_metrics(region, selected_models):
    obs_series, station_coords = load_station_series(region)

    if obs_series is None or not station_coords:
        raise ValueError(f"No station data found for {region}")

    records = []

    for model_key in selected_models:
        model_name = MODELS[model_key]

        for day in LEAD_DAYS:
            lead_name, lead_hours = forecast_lead_label(day)
            model_series = load_model_series(
                model_key=model_key,
                day=day,
                station_coords=station_coords,
            )

            if model_series is None or model_series.empty:
                continue

            metrics = compute_series_metrics(model_series, obs_series)

            records.append({
                "region": region,
                "model_key": model_key,
                "model_name": model_name,
                "run_day": day,
                "leadtime_hr": lead_hours,
                "lead_label": lead_name,
                "date_time": "",
                **metrics,
            })

    if not records:
        raise ValueError(f"No metrics could be computed for {region}")

    df = pd.DataFrame(records)

    df = df.rename(columns={"leadtime_hr": "Lead_Hours"})

    lead_rank = {lead: i for i, lead in enumerate(LEAD_HOURS_ORDER)}
    model_rank = {model: i for i, model in enumerate(selected_models)}

    df["Lead_Hours"] = df["Lead_Hours"].astype(int)
    df["_lead_rank"] = df["Lead_Hours"].map(lead_rank)
    df["_model_rank"] = df["model_key"].map(model_rank)

    df = df.sort_values(["_model_rank", "_lead_rank"])
    df = df.drop(columns=["_model_rank", "_lead_rank"]).reset_index(drop=True)

    return df


def update_temporal_csv(region_dfs, write_csv):
    if not write_csv:
        print("--csv not set. Skipping CSV output.")
        return

    for region, df_region in region_dfs.items():
        records = []

        for _, row in df_region.iterrows():
            records.append({
                "region": region,
                "model_key": row["model_key"],
                "model_name": row["model_name"],
                "run_day": row["run_day"],
                "leadtime_hr": int(row["Lead_Hours"]),
                "lead_label": row["lead_label"],
                "date_time": "",
                "rmse": row["RMSE"],
                "mae": row["MAE"],
                "mape": row["MAPE"],
                "correlation": row["Correlation"],
            })

        if not records:
            print(f"[Warning] No metric rows calculated for {region}")
            continue

        csv_path = output_path("temporal_metrics", region=region)
        rows_written = upsert_csv_columns(csv_path, records, TEMPORAL_KEY_COLUMNS)

        print(f"Updated CSV with {rows_written} temporal metric rows: {csv_path}")

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

def draw_metric_table(ax, df_model, metric_ranges, show_header=True):
    ax.axis("off")

    rows = []

    for _, row in df_model.iterrows():
        lead = int(row["Lead_Hours"])
        rows.append(
            [lead_label(lead)] + [format_value(row[m]) for m in METRICS]
        )

    if show_header:
        table = ax.table(
            cellText=rows,
            colLabels=["Lead", *DISPLAY_METRICS],
            cellLoc="center",
            loc="center",
            colWidths=[0.16, 0.14, 0.14, 0.14, 0.14],
        )
        header_offset = 1
    else:
        table = ax.table(
            cellText=rows,
            cellLoc="center",
            loc="center",
            colWidths=[0.16, 0.14, 0.14, 0.14, 0.14],
        )
        header_offset = 0

    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.45)

    header_color = "white"
    edge_color = to_rgba("black", 0.22)

    if show_header:
        for c in range(5):
            cell = table[(0, c)]
            cell.set_facecolor(header_color)
            cell.set_text_props(color="black", weight="bold")
            cell.set_edgecolor(edge_color)
            cell.set_linewidth(1)

    for i, (_, row) in enumerate(df_model.iterrows()):
        r = i + header_offset
        lead = int(row["Lead_Hours"])

        lead_cell = table[(r, 0)]
        lead_cell.set_facecolor(LEAD_COLORS.get(lead, "#b7b7b7"))
        lead_cell.set_text_props(color="black", weight="600")
        lead_cell.set_edgecolor(edge_color)
        lead_cell.set_linewidth(1)

        for c, metric in enumerate(METRICS, start=1):
            score = metric_score(
                value=row[metric],
                metric=metric,
                metric_ranges=metric_ranges,
            )

            cell = table[(r, c)]
            cell.set_facecolor(HEATMAP_CMAP(score))
            cell.set_edgecolor(edge_color)
            cell.set_linewidth(1)

    return table


def plot_side_by_side_tables(region_dfs, selected_models):
    nrows = len(selected_models)
    ncols = len(DEFAULT_REGIONS)

    metric_ranges_by_region = {
        region: get_metric_ranges(region_dfs[region])
        for region in DEFAULT_REGIONS
    }

    fig = plt.figure(
        figsize=(9.8, 1.65 * nrows + 1.6),
        dpi=300,
    )

    height_ratios = [1.18] + [1.0] * (nrows - 1)

    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols + 1,
        width_ratios=[0.30, 1.0, 1.0],
        height_ratios=height_ratios,
        hspace=0.18,
        wspace=0.08,
    )
    for row_idx, model_key in enumerate(selected_models):
        model_name = MODELS[model_key]

        label_ax = fig.add_subplot(gs[row_idx, 0])
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

        for col_idx, region in enumerate(DEFAULT_REGIONS):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])

            if row_idx == 0:
                ax.set_title(
                    region,
                    fontsize=13,
                    fontweight="bold",
                    pad=10,
                )

            df_region = region_dfs[region]
            df_model = df_region[df_region["model_key"] == model_key].copy()

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
                metric_ranges=metric_ranges_by_region[region],
                show_header=(row_idx == 0),
            )

    fig.suptitle(
        "Temporal Error Metrics by Lead Time — Regional Mean Comparison",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )

    cax = fig.add_axes([0.38, 0.045, 0.28, 0.018])

    cb = ColorbarBase(
        cax,
        cmap=HEATMAP_CMAP.reversed(),
        norm=Normalize(vmin=0.0, vmax=1.0),
        orientation="horizontal",
    )

    cb.set_ticks([0.0, 1.0])
    cb.set_ticklabels(["Worse", "Better"])

    fig.text(
        0.5,
        0.075,
        "Color scale is relative within each region",
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

    output_file = output_path("temporal_error_table")
    ensure_parent_dir(output_file)

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved table figure: {output_file}")


# MAIN

def main():
    parser = argparse.ArgumentParser(description="Create side-by-side CA/LA temporal error metric tables.")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()), choices=list(MODELS.keys()))
    parser.add_argument("--csv", action="store_true", help="Save computed temporal metrics to CSV.")

    args = parser.parse_args()

    selected_models = args.models

    print(f"Models: {selected_models}")
    print(f"Regions: {DEFAULT_REGIONS}")

    region_dfs = {
        region: load_region_metrics(region, selected_models)
        for region in DEFAULT_REGIONS
    }

    update_temporal_csv(
        region_dfs=region_dfs,
        write_csv=args.csv,
    )

    plot_side_by_side_tables(
        region_dfs=region_dfs,
        selected_models=selected_models,
    )


if __name__ == "__main__":
    main()
