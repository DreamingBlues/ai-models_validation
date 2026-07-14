# Station Network Map Generator

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from mpl_toolkits.basemap import Basemap

from viz_config import (
    DEFAULT_REGION,
    NETWORK_COLORS,
    NETWORK_NAMES,
    NETWORK_PLOT_ORDER,
    PATHS,
    REGIONS,
)
from viz_utils import (
    ensure_parent_dir,
    open_dataset_safe,
    output_path as configured_output_path,
    project_path,
    region_mask as configured_region_mask,
)


# GEOJSON HELPERS
def _iter_polygons_from_geojson(geojson_obj):
    def polygon_from_coords(coords):
        exterior = [(float(x), float(y)) for x, y in coords[0]]
        holes = [[(float(x), float(y)) for x, y in ring] for ring in coords[1:]]
        return exterior, holes

    for feat in geojson_obj.get("features", []):
        geom = feat.get("geometry", feat)

        if geom["type"] == "Polygon":
            yield polygon_from_coords(geom["coordinates"])

        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                yield polygon_from_coords(poly)


def apply_geojson_mask(df, geojson_path):
    """
    Keep only stations inside the California GeoJSON polygon.
    """

    geojson_path = project_path(geojson_path)

    with open(geojson_path) as f:
        gj = json.load(f)

    points = np.column_stack((df["longitude"].values, df["latitude"].values))
    mask = np.zeros(len(df), dtype=bool)

    for exterior, holes in _iter_polygons_from_geojson(gj):
        path = MplPath(exterior)
        inside = path.contains_points(points)

        for hole in holes:
            inside &= ~MplPath(hole).contains_points(points)

        mask |= inside

    return df.loc[mask].copy()


# HELPERS
def load_station_subset(path, region):
    """
    Load station metadata, mask to California GeoJSON, then filter by region.
    """

    path = project_path(path)
    ds = open_dataset_safe(path)

    df = pd.DataFrame(
        {
            "station": ds["station"].values.astype(str),
            "latitude": ds["latitude"].values,
            "longitude": ds["longitude"].values,
            "elevation": ds["elevation"].values,
            "network": ds["network"].values.astype(int),
        }
    )

    ds.close()

    df["network_name"] = df["network"].map(NETWORK_NAMES)

    df = apply_geojson_mask(df, PATHS["geojson"])

    mask = configured_region_mask(
        df["latitude"].values,
        df["longitude"].values,
        region,
    )

    return df.loc[mask].copy()


def get_region_figsize(minlon, maxlon, minlat, maxlat, base_width=7):
    """
    Keep width fixed for wide regions and scale height down.

    Longitude span is corrected by cos(latitude), so LA becomes shorter
    instead of wider.
    """

    mean_lat = np.deg2rad((minlat + maxlat) / 2)

    lon_span = abs(maxlon - minlon) * np.cos(mean_lat)
    lat_span = abs(maxlat - minlat)

    aspect = lon_span / lat_span

    if aspect >= 1:
        width = base_width
        height = base_width / aspect
    else:
        width = base_width * aspect
        height = base_width

    return width, height


def get_axis_ticks(region, minlon, maxlon, minlat, maxlat):
    """
    Use coarse labels for CA and finer labels for zoomed-in regions.
    """

    if region == "CA":
        lat_step = 2.0
        lon_step = 2.0
    else:
        lat_step = 0.25
        lon_step = 0.25

    lat_start = np.ceil(minlat / lat_step) * lat_step
    lat_end = np.floor(maxlat / lat_step) * lat_step

    lon_start = np.ceil(minlon / lon_step) * lon_step
    lon_end = np.floor(maxlon / lon_step) * lon_step

    lat_ticks = np.arange(lat_start, lat_end + lat_step, lat_step)
    lon_ticks = np.arange(lon_start, lon_end + lon_step, lon_step)

    return lat_ticks, lon_ticks


def print_station_summary(df_region, region):
    print(f"\nRegion: {region}")
    print(f"Total stations: {len(df_region)}")

    counts = (
        df_region.groupby(["network", "network_name"])
        .size()
        .reset_index(name="count")
        .sort_values("network")
    )

    print("\nStations by network:")
    print(counts.to_string(index=False))


def plot_station_map(df_region, region):
    minlon, maxlon, minlat, maxlat = REGIONS[region]

    figsize = get_region_figsize(minlon, maxlon, minlat, maxlat, base_width=7)

    fig = plt.figure(figsize=figsize, dpi=150)

    m = Basemap(
        projection="merc",
        epsg=4326,
        llcrnrlon=minlon,
        llcrnrlat=minlat,
        urcrnrlon=maxlon,
        urcrnrlat=maxlat,
        resolution="i",
    )

    # Terrain-style background, less busy than satellite
    try:
        m.arcgisimage(
            server="http://server.arcgisonline.com/arcgis",
            service="World_Shaded_Relief",
            xpixels=1400,
            verbose=False,
        )
    except Exception:
        m.shadedrelief(scale=0.35)

    # Boundaries
    m.drawcoastlines(color="0.25", linewidth=0.6)
    m.drawstates(color="0.45", linewidth=0.5)
    m.drawcountries(color="0.45", linewidth=0.5)

    # Axis labels only, no visible grid lines
    lat_ticks, lon_ticks = get_axis_ticks(region, minlon, maxlon, minlat, maxlat)

    m.drawparallels(
        lat_ticks,
        labels=[1, 0, 0, 0],
        fontsize=10,
        color=(0, 0, 0, 0),
        textcolor="black",
        linewidth=0.001,
        dashes=[1, 1],
    )

    m.drawmeridians(
        lon_ticks,
        labels=[0, 0, 0, 1],
        fontsize=10,
        color=(0, 0, 0, 0),
        textcolor="black",
        linewidth=0.001,
        dashes=[1, 1],
    )

    for i, network_id in enumerate(NETWORK_PLOT_ORDER):
        group = df_region[df_region["network"] == network_id]

        if len(group) == 0:
            continue

        x, y = m(group["longitude"].values, group["latitude"].values)

        m.scatter(
            x,
            y,
            c=NETWORK_COLORS.get(network_id, "black"),
            s=20,
            marker="o",
            alpha=0.9,
            linewidths=0,
            zorder=10 + i,
            label=f"{NETWORK_NAMES.get(network_id, 'Unknown')} ({len(group)})",
        )

    plt.title(f"Station Network Map: {region}", fontsize=15)

    plt.legend(
        loc="lower left",
        fontsize=8,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.6",
        markerscale=1.5,
    )

    plt.tight_layout()

    out_path = configured_output_path("network_station_map", region=region)
    ensure_parent_dir(out_path)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


# MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region",
        type=str,
        default=DEFAULT_REGION,
        choices=list(REGIONS.keys()),
        help="Region to process. Default: LA."
    )

    args = parser.parse_args()

    print(f"Loading stations for {args.region}...")

    df_region = load_station_subset(PATHS["station"], args.region)

    print_station_summary(df_region, args.region)

    print("Generating map...")
    plot_station_map(df_region, args.region)


if __name__ == "__main__":
    main()
