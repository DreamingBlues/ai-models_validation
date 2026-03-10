#!/usr/bin/env python3
# Gridded Error Map Generator (Absolute & Relative)

import argparse
import pathlib
import sys
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.basemap import Basemap
from scipy.spatial import cKDTree

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "situ_path": "./sensorData/station_data.nc",
    "model_path": "./{model_abr}/processed_data/{model_abr}_processed_CA_Day{day}.nc",
    "output_path": "./figs/animations/",
}

REGION_BOUNDS = {
    "CA": (-124.5, -114.0, 32.5, 42.0),
    "BA": (-122.80, -121.50, 36.85, 38.25),
    "LA": (-119.0, -117.0, 33.5, 35.0),
    "SD": (-117.40, -116.85, 32.50, 33.316),
}

UNITS = {
    "air_temp": "°C",
    "wind_speed": "m/s",
    "wind_direction": "deg"
}

MODELS = {
    "ifs":       {"model_abr": "ifs",       "model_name": "IFS HRES"},
    "fcn2":      {"model_abr": "fcn2",      "model_name": "FCNetv2-small"},
    "fcn3":      {"model_abr": "fcn3",      "model_name": "FCNetv3 Deterministic"},
    "aurora":    {"model_abr": "aurora",    "model_name": "Aurora"},
    "graphcast": {"model_abr": "graphcast", "model_name": "GraphCast"},
    "nbm":       {"model_abr": "nbm",       "model_name": "NBM"}
}

# ==============================================================================
# HELPERS
# ==============================================================================
def load_station_subset(path, region, var_name):
    """
    Loads station data, filters by region box.
    
    Args:
        path (str): path to station sensor data .nc file
        region (str): region case identifier (e.g. 'CA', 'LA')
        var_name (str): variable to be calculated (e.g. 'wind_speed')

    Returns:
        subset (xarray.Dataset): stations within region filtered by space
    """

    # Safety Checks
    if not os.path.exists(path):
        print(f"[Error] Station NetCDF not found: {path}")
        return None
    try:
        ds = xr.open_dataset(path, engine="h5netcdf")
    except Exception as e:
        print(f"[Error] Could not open Station NetCDF: {e}")
        return None
    if var_name not in ds:
        print(f"[Error] Variable '{var_name}' not found in station file.")
        return None
    if region not in REGION_BOUNDS:
        print(f"[Error] Region '{region}' not defined.")
        return None

    # Get region coordinates
    minlon, maxlon, minlat, maxlat = REGION_BOUNDS[region]
    
    # assign boolean mask 
    mask = (
        (ds['latitude'].values >= minlat) & (ds['latitude'].values <= maxlat) &
        (ds['longitude'].values >= minlon) & (ds['longitude'].values <= maxlon)
    )

    # Save appropriate stations only 
    subset = ds.isel(station=mask)
    
    if subset.sizes['station'] == 0:
        print(f"[Warning] No stations found within {region} bounds.")
        return None
        
    return subset


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    # Step 0: Setup 
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=str, required=True)
    parser.add_argument("--region", type=str, default="CA")
    parser.add_argument("--var", type=str, required=True, choices=["air_temp", "wind_speed", "wind_direction"])
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--metric", type=str, required=True, choices=["abs", "rel"], 
                        help="'abs' for RMSE/MAE/Bias, 'rel' for MAPE/% Bias")
    args = parser.parse_args()
    
    # Look up the model info
    m_info = MODELS[args.model]
    model_abr = m_info["model_abr"]
    model_name = m_info["model_name"]
    metric_type = args.metric

    # Unpack and store user inputs 
    day, region, var_name = args.day, args.region, args.var
    minlon, maxlon, minlat, maxlat = REGION_BOUNDS[region]
    unit_label = UNITS.get(var_name, "units")


    # Step 1: Load Data
    print(f"Loading data for {region}...")
    # filter station data by region and store
    ds_stations = load_station_subset(CONFIG["situ_path"], region, var_name)
    if ds_stations is None: sys.exit(1)

    # open model file
    model_file = CONFIG["model_path"].format(model_abr=model_abr, day=args.day)
    try: ds_model = xr.open_dataset(model_file)
    except: sys.exit(1)

    if var_name not in ds_model: sys.exit(1)


    # Step 2: Identify Stations within Grid Cells
    print("Binning stations to model grid...")

    # convert model coordinates from 1d to 2d grid
    grid_lon_2d, grid_lat_2d = np.meshgrid(ds_model.longitude.values, ds_model.latitude.values)
    # assign model grid coordinates into pairs
    flat_lats = grid_lat_2d.ravel()
    flat_lons = grid_lon_2d.ravel()
    grid_points = np.column_stack((flat_lats, flat_lons))

    # organize coordinates into k-dimensional tree
    tree = cKDTree(grid_points)

    # assign station coordinates into pairs
    st_lats = ds_stations.latitude.values
    st_lons = ds_stations.longitude.values
    st_points = np.column_stack((st_lats, st_lons))

    # Assign station to model grid cell
    _, grid_indices = tree.query(st_points, k=1)
    unique_grid_indices = np.unique(grid_indices)



    # Step 3: Align station with their grid cell
    print("Computing temporal data for all grid cells...")
    cell_data = {}
    # loop through grid cells containing stations
    for idx in unique_grid_indices:
        # specify which stations are located in the selected grid cell
        member_mask = (grid_indices == idx)

        # select stations within specific cell
        # average out the station's obs for each time step
        obs_s = ds_stations.isel(station=member_mask)[var_name].mean(dim="station", skipna=True).to_series()
        # ensure timezone Naive format
        obs_s.index = pd.to_datetime(obs_s.index).tz_localize(None)
        
        # select specific cell
        mod_s = ds_model[var_name].sel(latitude=flat_lats[idx], longitude=flat_lons[idx], method="nearest").to_series()
        mod_s.index = pd.to_datetime(mod_s.index).tz_localize(None)
        
        # store values
        cell_data[idx] = {'obs': obs_s, 'model': mod_s}

    # Find common timesteps
    t_model = pd.to_datetime(ds_model.time.values).tz_localize(None)
    t_obs = pd.to_datetime(ds_stations.time.values).tz_localize(None)
    common_times = t_model.intersection(t_obs).sort_values()



    # Step 4: Calculate Performance Metric for Each step/frame of Map
    print(f"Generating {metric_type.upper()} performance metrics for {len(common_times)} frames...")
    frames_data, valid_times = [], []
    
    # Initialize arrays based on metric type
    if metric_type == "abs":
        rmse_per_frame, mae_per_frame, bias_per_frame = [], [], []
    else:
        regional_pct_bias = []

    # loop through each time step
    for t in common_times:
        error_grid_flat = np.full(flat_lats.shape, np.nan)
        
        # temp accumulators for this frame
        if metric_type == "abs":
            frame_sq_errors, frame_abs_errors, frame_raw_errors = [], [], []
        else:
            frame_mape_vals, frame_diffs, frame_obs_sum = [], [], []

        # loop through each grid cell
        for idx in unique_grid_indices:
            obs_s = cell_data[idx]['obs']
            mod_s = cell_data[idx]['model']
            
            # check for missing timesteps
            if t in obs_s.index and t in mod_s.index:
                o_val = obs_s.loc[t]
                m_val = mod_s.loc[t]
                
                # ensure not calcualted with NaN observation
                if pd.notna(o_val) and pd.notna(m_val):
                    diff = m_val - o_val
                    
                    if metric_type == "abs":
                        sq_err = diff**2
                        error_grid_flat[idx] = sq_err
                        frame_sq_errors.append(sq_err)
                        frame_abs_errors.append(abs(diff))
                        frame_raw_errors.append(diff)
                        
                    elif metric_type == "rel" and abs(o_val) > 1e-3:
                        mape_val = (abs(diff) / abs(o_val)) * 100
                        error_grid_flat[idx] = mape_val
                        frame_mape_vals.append(mape_val)
                        frame_diffs.append(diff)
                        frame_obs_sum.append(o_val)
        

        # calculate and store error metrics
        if metric_type == "abs" and frame_sq_errors:
            frames_data.append(error_grid_flat.reshape(grid_lat_2d.shape))
            valid_times.append(t)
            rmse_per_frame.append(np.sqrt(np.mean(frame_sq_errors)))
            mae_per_frame.append(np.mean(frame_abs_errors))
            bias_per_frame.append(np.mean(frame_raw_errors))
            
        elif metric_type == "rel" and frame_mape_vals:
            frames_data.append(error_grid_flat.reshape(grid_lat_2d.shape))
            valid_times.append(t)
            obs_sum = np.sum(frame_obs_sum)
            pct_bias = (np.sum(frame_diffs) / obs_sum) * 100 if obs_sum != 0 else 0
            regional_pct_bias.append(pct_bias)

    if not frames_data:
        print("No overlapping data found. Exiting.")
        sys.exit(0)

    # Limits and Formatting based on metric type
    percentile = 98 if metric_type == "abs" else 95
    vmax_locked = np.nanpercentile(frames_data, percentile)
    cmap_label = f"Square Error ({var_name})" if metric_type == "abs" else "MAPE (%)"

    # Step 5: Build the Stacked Visual
    print("Stitching Map Animation...")
    fig, (ax_map, ax_ts) = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), dpi=200, gridspec_kw={'height_ratios': [2, 1]})

    # Step 5a: Build Geo-Map animation
    # define map plot function
    m = Basemap(projection="merc", epsg=4326, ax=ax_map,
                llcrnrlon=minlon, llcrnrlat=minlat,
                urcrnrlon=maxlon, urcrnrlat=maxlat, resolution="i")
    try:
        # download picture from internet
        m.arcgisimage(server="http://server.arcgisonline.com/arcgis", service="World_Imagery", verbose=False)
    except:
        # draw map outline if cannot connect
        m.drawcoastlines(color="gray")
        m.drawmapboundary(fill_color='white')

    # convert coordinates from Raw GPS Degrees to Flat Cartesian X/Y Map Coordinates
    x_grid, y_grid = m(grid_lon_2d, grid_lat_2d)
    x_st, y_st = m(st_lons, st_lats)

    # plot Station dots
    m.scatter(x_st, y_st, c='blue', s=5, zorder=5, alpha=0.3)

    # Initialize Heatmap with Frame 0
    hm = m.pcolormesh(x_grid, y_grid, frames_data[0], cmap='RdYlGn_r', alpha=0.6, shading='nearest', vmin=0, vmax=vmax_locked)
    # define sidebar
    plt.colorbar(hm, fraction=0.046, pad=0.04, ax=ax_map).set_label(cmap_label)
    # Initialize the Title Object which will update dynamically
    title_obj = ax_map.set_title("")

    # Step 5b: Build timeseries plot (Bottom Panel)
    # plot RMSE, MAE, and Bias    
    if metric_type == "abs":
        ax_ts.plot(valid_times, rmse_per_frame, color='tab:red', lw=1.8, label='RMSE', alpha=0.9)
        ax_ts.plot(valid_times, mae_per_frame, color='tab:orange', lw=1.5, label='MAE', alpha=0.8)
        ax_ts.plot(valid_times, bias_per_frame, color='tab:green', lw=1.5, ls='--', label='Bias', alpha=0.8)
        ax_ts.set_ylabel(f"{unit_label}", fontsize=10)
    else:
        ax_ts.plot(valid_times, regional_pct_bias, color='tab:green', lw=1.5, label='Percentage Bias')
        ax_ts.set_ylabel("Regional Bias (%)")

    # zero line for Bias reference
    ax_ts.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax_ts.legend(loc="upper right", fontsize='small', ncol=3 if metric_type == "abs" else 1)
    ax_ts.grid(True, alpha=0.3)

    # initialize the moving vertical "time indicator"
    time_marker = ax_ts.axvline(x=valid_times[0], color='black', linestyle='--', linewidth=1.5, zorder=10)

    # Step 6: Render and Save GIF
    def update(i):
        """
        param:
        i = frame index
        """
        # Update the 2D grid array
        hm.set_array(frames_data[i].ravel())

        # update timeseries plot
        t_curr = valid_times[i]
        time_marker.set_xdata([t_curr, t_curr]) # move vertical line
        
        # Update the Title Text
        t_str = t_curr.strftime('%Y-%m-%d %H:%M UTC') if metric_type == "abs" else t_curr.strftime('%H:%M UTC')
        
        if metric_type == "abs":
            title_obj.set_text(f"{model_name} | {region} Gridded Error: {var_name} \n{t_str} | Regional RMSE: {rmse_per_frame[i]:.2f}")
        else:
            avg_mape = np.nanmean(frames_data[i])
            title_obj.set_text(f"{model_name} | {region} | {var_name} | {t_str}\nAvg MAPE: {avg_mape:.1f}% | % Bias: {regional_pct_bias[i]:.1f}%")
        
        return hm, title_obj, time_marker 

    print(f"Rendering Animation to .GIF... ")
    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(valid_times), 
        blit=False    
    )
    
    out_path = pathlib.Path(CONFIG["output_path"]) / f"{model_abr}_{metric_type}_err_{region}_{day}_{var_name}.gif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save using Pillow (FPS=2 is 2 frames per second)
    anim.save(out_path, writer=PillowWriter(fps=2))
    plt.close()
    ds_stations.close()
    ds_model.close()
    print(f"Successfully saved Animation: {out_path}")

if __name__ == "__main__":
    main()