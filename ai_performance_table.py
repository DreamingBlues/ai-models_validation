#!/usr/bin/env python3
# Abtin Olaee 2025
"""
Build comparison table for multiple models (FCNet2, FCNet3, GraphCast, Aurora)
with a green–yellow–red heatmap based on global min/max of each metric,
plus a legend at the bottom.

Each CSV is expected to have columns:
Model, Run_Day, Lead_Hours, RMSE, MAE, MAPE, Correlation
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
from pathlib import Path


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
region = "LA"
AURORA_CSV     = f"/shome/u014930890/PGE_Projects/aurora_new/metrics/aurora_leadtime_metrics_{region}.csv"
FCNET2_CSV     = f"/shome/u014930890/PGE_Projects/FCNet2/metrics/fcn2_leadtime_metrics_{region}.csv"
FCNET3_CSV     = f"/shome/u014930890/PGE_Projects/FCNet3/metrics/fcnet3_leadtime_metrics_{region}.csv"
GRAPHCAST_CSV  = f"/shome/u014930890/PGE_Projects/graphcast/metrics/graphcast_leadtime_metrics_{region}.csv"

OUT_FIG        = f"/shome/u014930890/PGE_Projects/figs/ai_performance_heatmap_{region}.png"

# Desired row order (hours)
LEAD_ORDER = [144, 96, 48, 24, 0]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def load_and_order(path):
    df = pd.read_csv(path)
    cat = pd.Categorical(df["Lead_Hours"], LEAD_ORDER, ordered=True)
    df = df.assign(Lead_Hours=cat).sort_values("Lead_Hours").reset_index(drop=True)
    return df


def lead_label(hours):
    h = int(hours)
    return "Current" if h == 0 else f"{h}hr"


def main():
    # Load data for each model
    df_fc2 = load_and_order(FCNET2_CSV)
    df_fc3 = load_and_order(FCNET3_CSV)
    df_gc  = load_and_order(GRAPHCAST_CSV)
    df_au  = load_and_order(AURORA_CSV)

    # Keep order: FCNet2, FCNet3, GraphCast, Aurora
    dfs = [df_fc2, df_fc3, df_gc, df_au]
    model_titles = [
        df_fc2["Model"].iloc[0],
        df_fc3["Model"].iloc[0],
        df_gc["Model"].iloc[0],
        df_au["Model"].iloc[0].split()[0],  # "Aurora"
    ]

    # ------------------------------------------------------------------
    # Global min / max per metric across ALL models
    # ------------------------------------------------------------------
    all_df = pd.concat(dfs, ignore_index=True)

    metric_minmax = {
        "RMSE": (all_df["RMSE"].min(), all_df["RMSE"].max()),
        "MAE":  (all_df["MAE"].min(),  all_df["MAE"].max()),
        "MAPE": (all_df["MAPE"].min(), all_df["MAPE"].max()),
        "Corr": (all_df["Correlation"].min(), all_df["Correlation"].max()),
    }

    # Green → Yellow → Red (bright)
    cmap = LinearSegmentedColormap.from_list(
        "green_yellow_red",
        [
            (0.0, (0.0, 1.0, 0.0, 1.0)),   # green (best)
            (0.5, (1.0, 1.0, 0.0, 1.0)),   # yellow (neutral)
            (1.0, (1.0, 0.0, 0.0, 1.0)),   # red (worst)
        ],
    )

    def value_to_color(val, metric, better_low=True):
        vmin, vmax = metric_minmax[metric]
        if vmax == vmin:
            score = 0.5
        else:
            norm = (val - vmin) / (vmax - vmin)
            score = norm if better_low else 1.0 - norm
        score = min(max(score, 0.0), 1.0)
        return cmap(score)

    # ------------------------------------------------------------------
    # Build table data
    # ------------------------------------------------------------------
    # Assume all dfs share the same Lead_Hours ordering/length
    n_rows = len(dfs[0])

    rows = []
    for i in range(n_rows):
        lead_str = lead_label(dfs[0].iloc[i]["Lead_Hours"])
        row_cells = [lead_str]
        for df in dfs:
            r = df.iloc[i]
            row_cells.extend([
                f"{r.RMSE:.2f}",
                f"{r.MAE:.2f}",
                f"{r.MAPE:.2f}",
                f"{r.Correlation:.2f}",
                "",   # spacer after each model; last spacer will be removed later
            ])
        rows.append(row_cells)

    # Remove trailing spacer for last model
    for row in rows:
        row.pop()  # remove last "" per row

    # Column labels: lead + for each model: RMSE, MAE, MAPE, Corr (+ spacer except last)
    col_labels = ["Lead Times"]
    for m_idx in range(len(dfs)):
        col_labels.extend(["RMSE", "MAE", "MAPE", "Corr"])
        if m_idx != len(dfs) - 1:
            col_labels.append("")  # spacer

    # ------------------------------------------------------------------
    # Create figure and table
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )

    n_cols = len(col_labels)

    # Column widths (slimmer version)
    col_widths = {}
    col_widths[0] = 0.09  # lead labels (slightly slimmer than before)

    # slimmer metric/spacer widths
    metric_width = 0.045
    spacer_width = 0.010

    col_idx = 1
    for m_idx in range(len(dfs)):
        for _ in range(4):
            col_widths[col_idx] = metric_width
            col_idx += 1
        if m_idx != len(dfs) - 1:
            col_widths[col_idx] = spacer_width
            col_idx += 1

    for (r, c), cell in table.get_celld().items():
        if c in col_widths:
            cell.set_width(col_widths[c])

    # Layout colors
    header_color = "#2f75b5"      # dark blue
    row_label_color = "#2f75b5"
    spacer_color = "#b7b7b7"

    # Style header row
    for c in range(n_cols):
        cell = table[(0, c)]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", weight="bold")

    # Style left lead column
    for r in range(1, n_rows + 1):
        lead_cell = table[(r, 0)]
        lead_cell.set_facecolor(row_label_color)
        lead_cell.set_text_props(color="white", weight="bold")

    # Identify spacer columns
    spacer_cols = []
    col = 1
    for m_idx in range(len(dfs)):
        col += 4  # skip metrics
        if m_idx != len(dfs) - 1:
            spacer_cols.append(col)
            col += 1

    for r in range(1, n_rows + 1):
        for sc in spacer_cols:
            cell = table[(r, sc)]
            cell.set_facecolor(spacer_color)
            cell.set_edgecolor(spacer_color)

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    # slimmer horizontal padding
    table.scale(0.85, 1.6)

    # ------------------------------------------------------------------
    # Apply heatmap colors to metric cells
    # ------------------------------------------------------------------
    metric_names = ["RMSE", "MAE", "MAPE", "Corr"]
    better_low_flags = {"RMSE": True, "MAE": True, "MAPE": True, "Corr": False}

    col = 1
    col_metric_df = {}  # col -> (metric_name, df_index)
    for df_idx in range(len(dfs)):
        for m_name in metric_names:
            col_metric_df[col] = (m_name, df_idx)
            col += 1
        if df_idx != len(dfs) - 1:
            col += 1  # spacer

    for row_idx in range(n_rows):     # index in dfs
        r = row_idx + 1               # table row (header is 0)
        for c in range(1, n_cols):
            if c in spacer_cols:
                continue
            if c not in col_metric_df:
                continue
            metric, df_idx = col_metric_df[c]
            df = dfs[df_idx]
            row_src = df.iloc[row_idx]

            if metric == "Corr":
                val = row_src["Correlation"]
            else:
                val = row_src[metric]

            color = value_to_color(val, metric,
                                   better_low=better_low_flags[metric])
            cell = table[(r, c)]
            cell.set_facecolor(color)

    # ------------------------------------------------------------------
    # Add model group titles above table
    # ------------------------------------------------------------------
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()

    # For each model, find first and last metric column indices
    model_col_ranges = []
    col = 1
    for m_idx in range(len(dfs)):
        start_col = col
        end_col = col + 3  # RMSE..Corr
        model_col_ranges.append((start_col, end_col))
        col = end_col + 1
        if m_idx != len(dfs) - 1:
            col += 1  # skip spacer

    header_bbox = table[(0, 0)].get_window_extent(renderer)
    y = inv.transform((0, header_bbox.y1 + 10))[1]

    for m_idx, (start_col, end_col) in enumerate(model_col_ranges):
        bbox_start = table[(0, start_col)].get_window_extent(renderer)
        bbox_end   = table[(0, end_col)].get_window_extent(renderer)
        x = inv.transform(
            ((bbox_start.x0 + bbox_end.x1) / 2.0, 0)
        )[0]
        ax.text(
            x, y, model_titles[m_idx],
            ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )

    # ------------------------------------------------------------------
    # Add legend (colorbar) at the bottom
    # ------------------------------------------------------------------
    fig.subplots_adjust(bottom=0.22)

    cax = fig.add_axes([0.15, 0.06, 0.7, 0.04])
    norm = Normalize(vmin=0.0, vmax=1.0)
    cb = ColorbarBase(cax, cmap=cmap.reversed(), norm=norm, orientation="horizontal")

    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["Worse", "Neutral", "Better"])


    cax.set_xlabel(
        "Relative performance per metric across all models ",
        labelpad=4,
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    Path(OUT_FIG).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()