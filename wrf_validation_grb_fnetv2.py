#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:48:32 2024

@author: cdwehrke
wg: comments by Werner
"""

# Main script used for WRF validation. Contains multiple functions that will save pickle files once complete.

# Output pickle files will have time/space interpolated WRF data at every station in the domain.
# "case_name_processed_data.pkl" will have time series of WRF and measurements for every single station
# "case_name_wrfmean_data.pkl" is a single domain averaged time series of WRF data (at stations locations) for each variable
# "case_name_situmean_data.pkl" is a single domain averaged time series of station data for each variable

# To run, you will need:
# - A JSON with weather data from the Synoptic Weather API (run synreq.py),
# - A csv with weather variable names from Synoptic and how to calculate those in WRF (provided as synoptic_varlist.csv). 
# This determines which variables are validated. Full list of available variables can be found here: https://demos.synopticdata.com/variables/index.html
# - Model output files, with one frame per output file (grib or netcdf).

# The code also includes some plotting at the end, but can be turned on/off with PLOTTING flag set to True or False.
# california: minlon = -124.409591, minlat = 32.534156, maxlon = -114.131211, maxlat = 42.009518

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import scipy.stats
import glob
import xarray as xr
import cartopy.crs as ccrs
import metpy
import metpy.calc as mpcalc
import warnings
from metpy.units import units
import pickle
import json
import os
import warnings
import datetime
from metpy.calc import wind_components
from datetime import datetime, timedelta
from metpy.io import metar
from metpy.plots.declarative import (BarbPlot, ContourPlot, FilledContourPlot, MapPanel,
                                     PanelContainer, PlotObs)
import earthkit.data as ekd
from mpl_toolkits.basemap import Basemap

# set the csv reference file here. each model needs it"s own csv
# things to note: 
## use only available variables
## change variables from u10 to ["10u"] etc. as needed
var_ref = pd.read_csv("fourcastnetv2-small/synoptic_varlist_fnetv2_era5.csv", engine="python")
path2obs = "sensorData/timeseries" # Synoptic API JSON
path2model = "fourcastnetv2-small/fourcastnetv2-small.grib"  # Path to single grb with all timesteps
freq = "6h" # frequency of wrfoutput files
case_name = "fourcastv2net" #identifier name of output pickle files. Will be saved in same directory that script is run.

PLOTTING=True

def load_fnetv2(path2model):
    """Load fourcastnetv2 model output file.

    Args:
        path2model (str): path to model output file

    Returns:
        xarray.Dataset: open xarray dataset of model output
    """    
    params = ["2t", "r", "10u", "10v", "sp", "z"]
    # Load GRIB information using earthkit.data
    ds = ekd.from_source("file", path2model)
    # Select dataset for all variables and convert to xarray
    xds = ds.sel(param=params).to_xarray(engine="earthkit")
    # Calculate valid time coordinate
    reference_date = pd.to_datetime(str(xds.attrs["date"]), format="%Y%m%d")
    reference_time = pd.Timedelta(minutes=xds.attrs.get("time", 0))
    reference_datetime = reference_date + reference_time
    valid_time = reference_datetime + xds["step"]
    xds = xds.assign_coords(valid_time=valid_time)
    return xds

def to_lon180(lon, right_closed=False):
    """
    Return longitudes in [-180, 180] by default (or (-180, 180] if right_closed=True).
    Does nothing if input is already within [-180, 180].
    """
    a = np.asarray(lon)
    # If everything already in range, return as-is
    if np.nanmin(a) >= -180 and np.nanmax(a) <= 180:
        return lon
    out = ((a + 180) % 360) - 180  # -> [-180, 180)
    if right_closed:
        # flip -180 to +180 for (-180, 180]
        out = np.where(out == -180, 180.0, out)
    return out

def data_loading(path2obs, path2model):
    """Loads in synoptic data and model output files.

    Args:
        path2obs (str): path to observations
        path2model (str): path to model output files
        cwd (str): _description_

    Returns:
        _type_: _description_
    """    
    # 1) Loading model data and synoptic data
    # loading in model data
    model = load_fnetv2(path2model)
    
    # loading in output from synoptic json req.
    js = json.load(open(path2obs))
    obs = {}
    # reformatting synoptic data into pandas dataframes
    for site in js["STATION"]:
        obs[site["STID"]] = pd.DataFrame.from_dict(site["OBSERVATIONS"])
        obs[site["STID"]].set_index(pd.to_datetime(obs[site["STID"]]["date_time"]), inplace=True)
        obs[site["STID"]].attrs = {k: site[k] for k in set(list(site.keys())) - set("OBSERVATIONS")}

    # 2) Process data to get model data at each station location and time
    for number, site in enumerate(pd.Series(obs.keys())):
        print("\rloading data... "+ str(np.round((number/len(obs.keys()))*100, 3))+"%", end="\r")
        
        # fix naming issues in synoptic data
        uncorr = obs[site]
        df = {}
        for i in uncorr.keys():
            c = i
            if i.__contains__("_set_1") == True:
                c = i.replace("_set_1", "")
            if i.__contains__("_set_1d") == True:
                c = i.replace("_set_1d", "")
            df[c] = uncorr[i]
            df = pd.DataFrame.from_dict(df)
        df.attrs = uncorr.attrs

        # lat,lon of each station needed for point selection of model data
        lon = float(df.attrs["LONGITUDE"])
        lat = float(df.attrs["LATITUDE"])
        hgt = float(df.attrs["ELEVATION"])

        # list of variables available in synoptic data
        varlist = []
        for i in df.keys():
            varlist.append(i)

        # only calculate model variables for data we also have obs for
        data = {}
        # creating key in data for this station
        data[df.attrs["STID"]] = {"situ": [], "model": []}
        # appending sub dictionary with wx data
        data[df.attrs["STID"]]["situ"] = df

        # working on the model data                
        for count, step in enumerate(model.step.data):
            # selecting time step
            model_t = model.sel(step=step)
            # selecting closest grid point to station
            if count == 0:
                # get 1d arrays of lat/lon
                lat1d = np.asarray(model.latitude)
                lon1d = to_lon180(np.asarray(model.longitude))
                # distance using Haversine with broadcasting (no huge meshgrid needed)
                rlat = np.deg2rad(lat1d)[:, None]       # (ny, 1)
                rlon = np.deg2rad(lon1d)[None, :]       # (1, nx)
                rlat0 = np.deg2rad(lat)                 # scalar
                rlon0 = np.deg2rad(lon)                 # scalar
                dlat = rlat - rlat0
                dlon = rlon - rlon0
                a = np.sin(dlat/2.0)**2 + np.cos(rlat)*np.cos(rlat0)*np.sin(dlon/2.0)**2  # (ny, nx)
                yloc, xloc = np.unravel_index(np.nanargmin(a), a.shape)
                
            # selecting only that grid point
            model_t = model_t.isel(longitude=xloc, latitude=yloc)
            
            # creating temporary dataframe for this time step
            tempdf = {}
            tempdf["date_time"] = pd.to_datetime(model_t.valid_time.data, utc=True)
            # calculating model variables based on formulas in var_ref (from csv)
            for variable in var_ref["variable"]:
                if variable in varlist:
                    exec("tempdf[variable] =" + var_ref["formula"][list(var_ref["variable"]).index(variable)])
            # adding to dataframe
            tempdf = pd.DataFrame(tempdf, index=[0])
            tempdf = tempdf.set_index(tempdf["date_time"])

            # appending to data dictionary
            if count == 0:
                data[df.attrs["STID"]]["model"] = tempdf
            else:
                tempdf = pd.concat([data[(df.attrs["STID"], (lon, lat))]["model"], tempdf])
                data[df.attrs["STID"]]["model"] = tempdf

    # 3) Checkpoint resulting data
    with open(osp.join(os.getcwd(), f"{case_name}_data.pkl"), "wb") as f:
        pickle.dump(data, f)

    return data

#%% DATA PROCESSING

def data_processing(data,cwd,freq):

    #effectively just resampling the data and making it pretty.
    print("\n")
    processed = {}
    for count, key in enumerate(data.keys()):
        print("\rprocessing data... "+ str(np.round((count/len(data.keys()))*100,3))+"%",end="\r")
        #clipping to wrf time range (safest)
        data[key]["situ"] = data[key]["situ"].set_index(pd.to_datetime(data[key]["situ"].index,utc=True))
        time_wrf = pd.date_range(
            np.maximum(data[key]["wrf"]["date_time"][0],data[key]["situ"].index[0]),
            np.minimum(data[key]["wrf"]["date_time"][-1],data[key]["situ"].index[-1]),
            freq=freq
        )
        processed[key] = {"situ":[],"wrf":[]}
        #resampling to freq (defined manually below)
        data_slice = data[key]["situ"].loc[time_wrf[0]:time_wrf[-1]]
        data_slice = data_slice.loc[~data_slice.index.duplicated(keep="first")]
        processed[key]["situ"] = data_slice.rolling(freq).mean(numeric_only=True).interpolate(kind="linear").resample(freq).mean()
        data_slice = data[key]["wrf"].loc[time_wrf[0]:time_wrf[-1]]
        data_slice = data_slice.loc[~data_slice.index.duplicated(keep="first")]
        processed[key]["wrf"] = data_slice.rolling(freq).mean(numeric_only=True).interpolate(kind="linear").resample(freq).mean()
        
    #saving
    with open(cwd+"/"+case_name+"_processed_data.pkl", "wb") as g:
        pickle.dump(processed, g)

    return processed

#%% DATA EVALUATION

def data_evaluation(processed,cwd, freq):

    print("\n")

    evaluated = {}

    for key in processed.keys():

        try:

            processed[key]["wrf"] = processed[key]["wrf"].tz_localize(tz="utc")
            processed[key]["situ"] = processed[key]["situ"].tz_localize(tz="utc")

            time_wrf = pd.date_range(
                processed[key]["situ"].index[0],
                processed[key]["situ"].index[-1],
                freq=freq
            )

            processed[key]["situ"] = ((processed[key]["situ"].loc[time_wrf[0]:time_wrf[-1]]))

            processed[key]["wrf"] = ((processed[key]["wrf"].loc[time_wrf[0]:time_wrf[-1]]))

        except:
            continue


    for key,count in zip(processed.keys(),np.arange(0,len(processed.keys()))):

        try:

            evaluated[key] = {}

            print("\revaluating data statistics... "+ str(np.round((count/len(processed.keys()))*100,3))+"%",end="\r")

            for variable in processed[key]["wrf"].keys():
                if variable == "date_time":
                    continue
                else:
                    evaluated[key][variable] = {
                        "rmse": [
                            rmse(
                                processed[key]["wrf"][variable],
                                processed[key]["situ"][variable]
                            )
                        ],
                        "pearson":[
                            pearson(
                                processed[key]["wrf"][variable],
                                processed[key]["situ"][variable]
                            )
                        ]
                    }
        except:
            continue

    concat_wrf = {}
    concat_situ = {}
    mean_wrf = {}
    mean_situ = {}
    for key in processed.keys():
        for variable in processed[key]["wrf"].keys():
            if variable == "date_time":
                continue
            else:
                concat_wrf[variable] = pd.DataFrame()
                concat_situ[variable] = pd.DataFrame()

    print("\n")

    for key,count in zip(processed.keys(),np.arange(0,len(processed.keys()))):
        print("\rspatially concatenating data... "+ str(np.round((count/len(processed.keys()))*100,3))+"%",end="\r")
        for variable in processed[key]["wrf"].keys():
            if variable == "date_time":
                continue
            else:
                concat_wrf[variable] = pd.concat([concat_wrf[variable],processed[key]["wrf"][variable]],axis=1)
                concat_situ[variable] = pd.concat([concat_situ[variable],processed[key]["situ"][variable]],axis=1)


    for variable in concat_wrf.keys():
        mean_wrf[variable] = concat_wrf[variable].mean(axis=1).fillna(0).sort_index()    #((concat_wrf[variable].sum(axis=1,numeric_only=True))/len(concat_wrf[variable].columns)).sort_index()
        mean_situ[variable] = concat_situ[variable].mean(axis=1).fillna(0).sort_index()  #((concat_situ[variable].sum(axis=1,numeric_only=True))/len(concat_situ[variable].columns)).sort_index()

    with open(cwd+"/"+case_name+"_evaluated_data.pkl", "wb") as a:
        pickle.dump(evaluated, a)

    with open(cwd+"/"+case_name+"_wrfmean_data.pkl", "wb") as b:
        pickle.dump(mean_wrf, b)

    with open(cwd+"/"+case_name+"_situmean_data.pkl", "wb") as c:
        pickle.dump(mean_situ, c)
        
    return mean_wrf, mean_situ, evaluated


#%%WRAPPING IT UP

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def pearson(predictions, targets):
    return scipy.stats.pearsonr(predictions,targets)[0]

def the_whole_enchilada(path2obs,path2model,cwd,freq,):
    try:
        mean_wrf, mean_situ, evaluated = data_evaluation(pd.read_pickle(cwd+"/"+case_name+"_processed_data.pkl"),cwd, freq)
    except:
        try:
            mean_wrf, mean_situ, evaluated = data_evaluation(data_processing(pd.read_pickle(cwd+"/"+case_name+"_data.pkl"),cwd,freq),cwd, freq)
        except:
            mean_wrf, mean_situ, evaluated = data_evaluation(data_processing(data_assimilation(path2obs,path2model,cwd),cwd,freq),cwd, freq)
    return mean_wrf, mean_situ, evaluated


if __name__ == "__main__":
    cwd = os.getcwd()
    mean_wrf, mean_insitu, stats = the_whole_enchilada(path2obs,path2model,cwd,freq)

    #%%

    if PLOTTING == True:
        data = pd.read_pickle(cwd+"/"+case_name+"_processed_data.pkl")
        minlon = -124.409591
        minlat = 32.534156
        maxlon = -114.131211
        maxlat = 42.009518
        print("\n")

        time = mean_wrf["air_temp"].index
        for var,count in zip(mean_wrf.keys(), np.arange(0,len(mean_wrf.keys()))):
            if var=="fuel_moisture":
                mean_wrf[var] = mean_wrf[var]*100
            mean_wrf[var] = mean_wrf[var].sort_index()
            mean_insitu[var] = mean_insitu[var].sort_index()
            print("\rplotting data... "+ str(np.round((count/len(mean_wrf.keys()))*100,3))+"%",end="\r")
            try:
                plt.figure(figsize=(10,5),dpi=175)
                plt.plot(time,mean_wrf[var],c="dodgerblue",label="Prediction")
                plt.plot(time,mean_insitu[var],c="red",label="Synoptic Measurement")
                plt.ylabel(var)
                plt.xlabel("Time (UTC)")
                plt.xticks(rotation=45)
                plt.title(var+case_name+"\nTime Series RMSE: %s"%np.round(rmse(mean_wrf[var],mean_insitu[var]),2)+"\nTime Series Pearson: %s"%np.round(pearson(mean_wrf[var],mean_insitu[var]),2))
                plt.ylabel(var)
                plt.legend(fontsize="x-large")
                #plt2.legend(fontsize="x-large")
                plt.tight_layout()
                plt.savefig(case_name+"_timeSeries_"+var+".png")
            except:
                continue
            finally:
                try:
                    #plt.figure(figsize=(7,7),dpi=150)
                    plt.figure(figsize=(10,7),dpi=150)
                    line = [mean_wrf[var].max(),mean_insitu[var].max(),mean_wrf[var].min(),mean_insitu[var].min()]
                    plt.xlim(np.min(line),np.max(line))
                    plt.ylim(np.min(line),np.max(line))
                    ax = plt.gca()
                    ax.plot([0, 1], [0, 1], transform=ax.transAxes,linestyle="--",color="black", label="f(x) = x")
                    plt.plot(np.arange(np.min(line),np.max(line)), np.poly1d(np.polyfit(mean_wrf[var], mean_insitu[var], 1))(np.arange(np.min(line),np.max(line))),linestyle="-",color="black", label = "f(x) = "+str(np.poly1d(np.polyfit(mean_wrf[var], mean_insitu[var], 1)))[2:])
                    plt.scatter(mean_wrf[var],mean_insitu[var],c="black")
                    plt.xlabel("Model Predictions")
                    plt.ylabel("Observed")
                    plt.title(var+"\nScatterplot RMSE: %s"%np.round(rmse(mean_wrf[var],mean_insitu[var]),2)+"\nScatterplot Pearson Correlation: %s"%np.round(pearson(mean_wrf[var],mean_insitu[var]),2))
                    plt.legend(["Prediction, Measurement"],  fontsize="x-large")
                    plt.savefig(case_name+"_scatterPlot_"+var+".png")
                except:
                    continue
                finally:
                    if var == "wind_speed" or var =="wind_direction":
                        plt.figure(figsize=(10,10),dpi=150)
                        #you usually have to tweak the epsg and corners to look good
                        map = Basemap(projection="merc",
                                        resolution="l",
                                        epsg = 4326,
                                        urcrnrlon=maxlon,
                                        llcrnrlat=minlat,
                                        llcrnrlon=minlon,
                                        urcrnrlat=maxlat)
                        #tc background
                        map.arcgisimage(server="http://server.arcgisonline.com/arcgis",service="World_Imagery",verbose= False)
                        #map.drawcoastlines()
                        x = []
                        y = []
                        c = []
                        names = []
                        data = pd.read_pickle(cwd+"/"+case_name+"_processed_data.pkl")
                        for name, coords in data.keys():
                            try:
                                x = (map(coords[0],coords[1])[0])
                                y = (map(coords[0],coords[1])[1])
                                u_wrf, v_wrf = wind_components(data[(name,coords)]["wrf"]["wind_speed"].mean() * units("m/s"), data[(name,coords)]["wrf"]["wind_direction"].mean() * units.deg)
                                u_situ, v_situ = wind_components(data[(name,coords)]["situ"]["wind_speed"].mean() * units("m/s"), data[(name,coords)]["situ"]["wind_direction"].mean() * units.deg)
                                map.barbs(x, y,u_wrf*1.94384, v_wrf*1.94384, color="dodgerblue",length=8, alpha=1)#, label="WRF Wind Avg.")
                                map.barbs(x, y,u_situ*1.94384, v_situ*1.94384, color="red",length=8, alpha=1)#, label="Station Wind Avg.")
                            except:
                                continue
                        #map.barbs(x, y,u_situ, v_situ, color="red",length=5, alpha=0.5)
                        plt.title("Average Station vs. WRF Wind Barb")
                        lons = [-124.409591, -114.131211]
                        lats = [32.534156, 42.009518]
                        #lons = [-114.131211, 42.009518]
                        #lats = [-124.409591, 32.534156]
                        x, y = map(lons, lats)
                        cb = map.scatter(x,y,c=c,s=75, cmap="rainbow")
                        #map.scatter(map(-156.67803, 20.87913)[0], map(-156.67803, 20.87913)[1],c="red",s=75,vmin=0.5,vmax=1, marker="X", label = "Lahaina")
                        import matplotlib.patches as mpatches
                        red_patch = mpatches.Patch(color="red", label="Station Wind Avg.")
                        blue_patch = mpatches.Patch(color="dodgerblue", label="WRF Wind Avg.")
                        plt.legend(handles=[red_patch, blue_patch])
                        # plt.show()
                        plt.legend(["Prediction, Measurement"],  fontsize="x-large")
                        plt.colorbar(cb, fraction=0.033, pad=0.04,label="RMSE")
                        plt.savefig(case_name+"_WRFWindBarb_"+var+".png")

                    plt.figure(figsize=(10,10),dpi=150)
                    #you usually have to tweak the epsg and corners to look good
                    map = Basemap(projection="merc",
                                    resolution="l",
                                    epsg = 4326,
                                    urcrnrlon=maxlon,
                                    llcrnrlat=minlat,
                                    llcrnrlon=minlon,
                                    urcrnrlat=maxlat)
                    #tc background
                    map.arcgisimage(server="http://server.arcgisonline.com/arcgis",service="World_Imagery",verbose= False)
                    #map.drawcoastlines()
                    x = []
                    y = []
                    c = []
                    names = []
                    for name, coords in stats.keys():
                        try:
                            c.append(stats[(name,coords)][var]["rmse"])
                            x.append(map(coords[0],coords[1])[0])
                            y.append(map(coords[0],coords[1])[1])
                            names.append(name)
                        except:
                            continue
                        
                    plt.title("Temporally Averaged RMSE values of %s" %var)
                    #lons = [-114.131211, 42.009518]
                    #lats = [-124.409591, 32.534156]
                    #x, y = map(lons, lats)
                    cb = map.scatter(x,y,c=c,s=75, cmap="rainbow")
                    #map.scatter(map(-156.67803, 20.87913)[0], map(-156.67803, 20.87913)[1],c="red",s=75,vmin=0.5,vmax=1, marker="X", label = "Lahaina")
                    plt.legend(["Prediction, Measurement"],  fontsize="x-large")
                    plt.colorbar(cb, fraction=0.033, pad=0.04,label="RMSE")
                    plt.savefig("spatial_"+case_name+"_"+var+".png")
