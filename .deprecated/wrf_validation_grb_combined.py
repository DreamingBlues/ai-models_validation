#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:48:32 2024

@author: cdwehrke
wg: comments by Werner
"""

#Main script used for model validation. Contains multiple functions that will save pickle files once complete.

#Output pickle files will have time/space interpolated model data at every station in the domain.
# "case_name_processed_data.pkl" will have time series of model and measurements for every single station
# "case_name_modelmean_data.pkl" is a single domain averaged time series of model data (at stations locations) for each variable
# "case_name_situmean_data.pkl" is a single domain averaged time series of station data for each variable


# To run, you will need:

# - A JSON with weather data from the Synoptic Weather API (run synreq.py),

# - A csv with weather variable names from Synoptic and how to calculate those in model (provided as synoptic_varlist.csv). 
## - This determines which variables are validated. Full list of available variables can be found here: https://demos.synopticdata.com/variables/index.html

## - model output files, with one frame per output file.

# The code also includes some plotting at the end, but can be turned on/off with PLOTTING flag set to True or False.

# wg: for an HPC environment it is recommended to create a new conda and use conda not pip, as basemap dependencies are tricky to install
# wg: conda activate .aiModel
# wg: conda activate .gribPlot
# wg: todo run on just pip
#-97.810572,27.198243,-96.169168,28.711714 texas
#-157.44914,20.171745,-155.92085,21.59452 hawaii
#-118.86246,33.819416,-118.332855,34.25749 socal
# california: minlon = -124.409591, minlat = 32.534156, maxlon = -114.131211, maxlat = 42.009518



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
import os.path as osp
import warnings
import datetime
from metpy.calc import wind_components
from datetime import datetime, timedelta
from metpy.io import metar
from metpy.plots.declarative import (BarbPlot, ContourPlot, FilledContourPlot, MapPanel,
                                     PanelContainer, PlotObs)
import earthkit.data as ekd
from mpl_toolkits.basemap import Basemap



# Pick the model you want to run
case_name = 'aurora_0.25_pre'  
"""     
 options: 'fourcastv2net'
          'fourcastv2netHRES'
          'fourcastv2netEra5'
          'aurora_0.25_pre'
          'aurora_0.25_fine'
          'aurora_0.1_fine'
"""
path2model = "/shome/u014930890/PGE Projects/aurora_10day/data/aurora-2.5-pretrained_1.grib"

# Consistent settings
path2obs = 'sensorData/timeseries'   # Synoptic API JSON
freq     = '6h'                       # frequency of model/output files


PLOTTING=False
EVALUATE=False

def load_model(case_name, path2model):
    """
    Load model data and variable reference configuration.

    Args:
        case_name (str): model case identifier (e.g., 'aurora_0.25_pre', 'fourcastv2net')
        path2model (str): path to model GRIB file

    Returns:
        xds (xarray.Dataset): model output with valid_time coordinates
        var_ref (pandas.DataFrame): variable name to formula mapping from CSV
    """

    # pick the correct CSV for formulas
    ## PATH NAMES ARE TEMPORARY  UPDATE FOR SPECIFC CASE
    match case_name:
        case 'fourcastv2net' | 'fourcastv2netEra5':
            var_ref_path = 'fourcastnetv2-small/synoptic_varlist_fnetv2_era5.csv'
        case 'fourcastv2netHRES':
            var_ref_path = 'fourcastnetv2-small/synoptic_varlist_fnetv2_ifs_hres.csv'
        case 'aurora_0.25_pre' | 'aurora_0.25_fine' | 'aurora_0.1_fine':
            var_ref_path = 'aurora/synoptic_varlist_aurora.csv'
        case _:
            raise ValueError(f"Unknown case_name='{case_name}'.")
    var_ref = pd.read_csv(var_ref_path, engine='python')

    # Only HRES differs in params, uses 'gh' instead of 'z'
    if case_name == 'fourcastv2netHRES':
        params = ['2t', 'r', '10u', '10v', 'sp', 'gh']
    else:
        # shared params, Aurora, FCNetv2, ERA5
        params = ['2t', 'r', '10u', '10v', 'sp', 'z']


    # open GRIB into xarray and select variables
    ds  = ekd.from_source("file", path2model)
    xds = ds.sel(param=params).to_xarray()

    # HRES-only: convert geopotential height -> geopotential
    if case_name == 'fourcastv2netHRES':
        if 'gh' not in xds:
            raise KeyError("Missing 'gh' for HRES")
        xds = xds.assign(z=xds['gh'] * 9.80665)

    # Time handling
    if case_name == 'fourcastv2netEra5':
        # ERA5 uses forecast_reference_time → derive step
        xds = xds.assign_coords(valid_time=xds.forecast_reference_time)
        step = [wt-xds.valid_time.data[0] for wt in xds.valid_time.data]
        xds = xds.rename({'forecast_reference_time': 'step'}).assign_coords(step=step)
    else:
        # Everyone else: reference datetime + step
        ref_date = pd.to_datetime(str(xds.attrs['date']), format='%Y%m%d')
        ref_time = pd.Timedelta(minutes=int(xds.attrs.get('time', 0)))
        ref_datetime   = (ref_date + ref_time).to_datetime64()
        valid_time = ref_datetime + xds['step']
        xds = xds.assign_coords(valid_time=valid_time)

    return xds, var_ref


def to_lon180(lon, right_closed: bool = False):
    """
    Convert longitude values to [-180, 180] range.

    Args:
        lon (array-like): longitude values to convert
        right_closed (bool): if True, use (-180, 180] range instead of [-180, 180]

    Returns:
        array-like: longitude values in [-180, 180] or (-180, 180] range
    """
    a = np.asarray(lon)
    # quick return if already in range
    if np.nanmin(a) >= -180 and np.nanmax(a) <= 180:
        return lon
    # Main transformation
    out = ((a + 180) % 360) - 180
    if right_closed:  # Edge inclusion handling
        out = np.where(out == -180, 180.0, out)
    return out


def data_loading(path2obs, case_name, path2model):
    """
    Load observation and model data, interpolate model to station locations.

    Args:
        path2obs (str): path to synoptic observation JSON file
        case_name (str): model case identifier for output naming
        path2model (str): path to model GRIB file

    Returns:
        data (dict): dictionary with keys (station_name, (lon, lat)) containing 'situ' and 'model' dataframes
    """

    # 1) Loading model data and synoptic data
    model, var_ref = load_model(case_name, path2model)

    # loading in output from synoptic json req.
    js = json.load(open(path2obs))
    obs = {}
    # reformatting synoptic data into pandas dataframes
    for site in js['STATION']:
        obs[site['STID']] = pd.DataFrame.from_dict(site['OBSERVATIONS'])
        obs[site['STID']] = obs[site['STID']].set_index(pd.to_datetime(obs[site['STID']]['date_time']))
        obs[site['STID']].attrs = {k: site[k] for k in set(list(site.keys())) - set('OBSERVATIONS')}

    # Initialize data dictionary BEFORE the loop
    data = {}
    
    # 2) Process data to get model data at each station location and time
    for number, site in enumerate(pd.Series(obs.keys())):
        print("\rloading data... "+ str(np.round((number/len(obs.keys()))*100, 3))+"%", end="\r")

        # fix naming issues in synoptic data
        uncorr = obs[site]
        df = {}
        for i in uncorr.keys():
            c = i
            if i.__contains__('_set_1')==True:
                c = i.replace('_set_1','')
            if i.__contains__('_set_1d')==True:
                c = i.replace('_set_1d','')
            df[c] = uncorr[i]
            df = pd.DataFrame.from_dict(df)
        df.attrs = uncorr.attrs

        # lat,lon of each station needed for point selection of model data
        lon = float(df.attrs['LONGITUDE'])
        lat = float(df.attrs['LATITUDE'])
        hgt = float(df.attrs['ELEVATION'])

        # list of variables available in synoptic data
        varlist = []
        for i in df.keys():
            varlist.append(i)

        # Key format: (STID, (lon, lat))
        data[(df.attrs["STID"], (lon, lat))] = {"situ": [], "model": []}
        data[(df.attrs["STID"], (lon, lat))]["situ"] = df

        # working on the model data                
        for count, step in enumerate(model.step.data):
            # selecting time step
            model_t = model.sel(step=step)

            # selecting closest grid point to station
            if count == 0:
                # get 1d arrays of lat/lon
                lat1d = np.asarray(model.latitude)
                lon1d = to_lon180(np.asarray(model.longitude))
                # distance using Haversine with broadcasting
                rlat = np.deg2rad(lat1d)[:, None]
                rlon = np.deg2rad(lon1d)[None, :]
                rlat0 = np.deg2rad(lat)
                rlon0 = np.deg2rad(lon)
                dlat = rlat - rlat0
                dlon = rlon - rlon0
                a = np.sin(dlat/2.0)**2 + np.cos(rlat)*np.cos(rlat0)*np.sin(dlon/2.0)**2
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
                data[(df.attrs["STID"], (lon, lat))]["model"] = tempdf
            else:
                data[(df.attrs["STID"], (lon, lat))]["model"] = pd.concat([
                    data[(df.attrs["STID"], (lon, lat))]["model"], tempdf
                ])

    # 3) Checkpoint resulting data
    with open(osp.join(os.getcwd(), f"{case_name}_data.pkl"), "wb") as f:
        pickle.dump(data, f)

    return data

#%% DATA PROCESSING Updated
cwd = os.getcwd()
def data_processing(data, cwd, freq):
    """
    Resample and align model and observation data to common time grid.

    Args:
        data (dict): output from data_loading with station data
        cwd (str): current working directory for saving output
        freq (str): resampling frequency (e.g., '6h')

    Returns:
        processed (dict): dictionary with resampled 'situ' and 'model' dataframes per station
    """
    print('\n')
    processed = {}

    # iterate through stations
    for key, count in zip(data.keys(), np.arange(0, len(data.keys()))):
        print('\rprocessing data... '+ str(np.round((count/len(data.keys()))*100, 3))+'%', end="\r")
        
        # normalize situ index to datetime UTC
        data[key]['situ'] = data[key]['situ'].set_index(
            pd.to_datetime(data[key]['situ'].index, utc=True)
        )

        # guards to skip broken stations
        if (
            'model' not in data[key]
            or data[key]['model'] is None
            or data[key]['situ'].empty
            or data[key]['model'].empty
            or 'date_time' not in data[key]['model']
        ):
            print(f"\nSkipping {key} (missing or empty data)")
            continue

        # compute overlapping time observations
        start = np.maximum(
            data[key]['model']['date_time'].iloc[0],
            data[key]['situ'].index.min()
        )
        end = np.minimum(
            data[key]['model']['date_time'].iloc[-1],
            data[key]['situ'].index.max()
        )
        # second layer of guards to prevent non-overlapping time ranges
        if start >= end:
            print(f"\nSkipping {key} (no overlap between model and situ times)")
            continue

        # rebuild regular time grid
        time_model = pd.date_range(start, end, freq=freq)
        # catches cases where start and end are identical
        if len(time_model) == 0:
            print(f"\nSkipping {key} (empty time range)")
            continue

        # initialize ouput slots
        processed[key] = {'situ': [], 'model': []}
        
        # Resampling situ data to freq
        data_slice = data[key]['situ'].loc[time_model[0]:time_model[-1]]
        # drop duplicate timestamps
        data_slice = data_slice.loc[~data_slice.index.duplicated(keep='first')]
        processed[key]['situ'] = (
            data_slice.rolling(freq)
            .mean(numeric_only=True)
            .interpolate(kind='linear')    # interpolate linearly to fill small gaps
            .resample(freq)
            .mean()
        )
        
        # Resampling model data to freq
        data_slice = data[key]['model'].loc[time_model[0]:time_model[-1]]
        data_slice = data_slice.loc[~data_slice.index.duplicated(keep='first')]
        processed[key]['model'] = (
            data_slice.rolling(freq)
            .mean(numeric_only=True)
            .interpolate(kind='linear')
            .resample(freq)
            .mean()
        )
        
    # Saving
    with open(cwd + '/' + case_name + '_processed_data.pkl', 'wb') as g:
        pickle.dump(processed, g)

    return processed


#%% DATA EVALUATION
def data_evaluation(processed,cwd, freq):

    print('\n')

    evaluated = {}

    for key in processed.keys():

        try:

            processed[key]['model'] = processed[key]['model'].tz_localize(tz='utc')
            processed[key]['situ'] = processed[key]['situ'].tz_localize(tz='utc')

            time_model = pd.date_range(processed[key]['situ'].index[0],
                                     processed[key]['situ'].index[-1],
                                     freq=freq)


            processed[key]['situ'] = ((processed[key]['situ'].loc[time_model[0]:time_model[-1]]))

            processed[key]['model'] = ((processed[key]['model'].loc[time_model[0]:time_model[-1]]))

        except:
            continue


    for key,count in zip(processed.keys(),np.arange(0,len(processed.keys()))):

        try:

            evaluated[key] = {}

            print('\revaluating data statistics... '+ str(np.round((count/len(processed.keys()))*100,3))+'%',end="\r")

            for variable in processed[key]['model'].keys():
                if variable == 'date_time':
                    continue
                else:
                    evaluated[key][variable] = {'rmse':[rmse(processed[key]['model'][variable],
                                                             processed[key]['situ'][variable])],

                                                'pearson':[pearson(processed[key]['model'][variable],
                                                           processed[key]['situ'][variable])]
                                                    }
        except:
            continue

    concat_model = {}

    concat_situ = {}

    mean_model = {}

    mean_situ = {}

    for key in processed.keys():

        for variable in processed[key]['model'].keys():

            if variable == 'date_time':
                continue
            else:

                concat_model[variable] = pd.DataFrame()

                concat_situ[variable] = pd.DataFrame()

    print('\n')

    for key,count in zip(processed.keys(),np.arange(0,len(processed.keys()))):

        print('\rspatially concatenating data... '+ str(np.round((count/len(processed.keys()))*100,3))+'%',end="\r")

        for variable in processed[key]['model'].keys():

            if variable == 'date_time':
                continue

            else:

                concat_model[variable] = pd.concat([concat_model[variable],processed[key]['model'][variable]],axis=1)

                concat_situ[variable] = pd.concat([concat_situ[variable],processed[key]['situ'][variable]],axis=1)


    for variable in concat_model.keys():
        mean_model[variable] = concat_model[variable].mean(axis=1)#.fillna(0).sort_index()#((concat_model[variable].sum(axis=1,numeric_only=True))/len(concat_model[variable].columns)).sort_index()
        mean_situ[variable] = concat_situ[variable].mean(axis=1)#.fillna(0).sort_index()#((concat_situ[variable].sum(axis=1,numeric_only=True))/len(concat_situ[variable].columns)).sort_index()

    with open(cwd+'/'+case_name+'_evaluated_data.pkl', 'wb') as a:
        pickle.dump(evaluated, a)

    with open(cwd+'/'+case_name+'_modelmean_data.pkl', 'wb') as b:
        pickle.dump(mean_model, b)

    with open(cwd+'/'+case_name+'_situmean_data.pkl', 'wb') as c:
        pickle.dump(mean_situ, c)


    return mean_model, mean_situ, evaluated


#%%WRAPPING IT UP

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def pearson(predictions, targets):
    return scipy.stats.pearsonr(predictions,targets)[0]

def the_whole_enchilada(path2obs, path2model, cwd, freq, evaluate=True):
    """
    Main pipeline function that handles loading, processing, and optionally evaluation.

    Args:
        path2obs (str): path to observation data (Synoptic JSON)
        path2model (str): path to model GRIB file
        cwd (str): current working directory
        freq (str): frequency for resampling (e.g., '6h')
        evaluate (bool): if True, run evaluation. If False, only ensure processed data exists.

    Returns:
        tuple: mean_model, mean_situ, evaluated if evaluate=True
        None: if evaluate=False
    """
    processed_path = f"{cwd}/{case_name}_processed_data.pkl"
    data_path = f"{cwd}/{case_name}_data.pkl"
    
    if evaluate:
        # Run full evaluation pipeline
        print("Starting full evaluation pipeline...")

        try:
            # find and evaluate processed data
            print(f"Attempting to load processed data from {processed_path}...")
            mean_model, mean_situ, evaluated = data_evaluation(
                pd.read_pickle(processed_path), cwd, freq
            )
            print("Successfully evaluated processed data.")

        except Exception as e1:
            print(f"Processed data not found or invalid: {e1}")
            try:
                # if processed data not present, process raw data then evaluate
                print(f"Attempting to process raw data from {data_path}...")
                processed = data_processing(pd.read_pickle(data_path), cwd, freq)
                print("Raw data processed. Running evaluation...")
                mean_model, mean_situ, evaluated = data_evaluation(processed, cwd, freq)
                print("Successfully evaluated newly processed data.")

            except Exception as e2:
                print(f"Raw data not found or invalid: {e2}")
                try:
                    # if raw data not pressent load everything fresh, then process + evaluate
                    print(f"Loading model data from {path2model}...")
                    raw = data_loading(path2obs, case_name, path2model)
                    print("Data loaded successfully. Beginning processing and evaluation...")
                    processed = data_processing(raw, cwd, freq)
                    mean_model, mean_situ, evaluated = data_evaluation(processed, cwd, freq)
                    print("Full load + process + evaluation complete.")
                except Exception as e3:
                    print(f"Full evaluation pipeline failed: {e3}")
                    return None

        return mean_model, mean_situ, evaluated
    
    else: ### Only ensure processed data exists, skip evaluation
        if os.path.exists(processed_path):
            # if processed data available, skip all together
            print(f"Found existing {processed_path}, skipping processing")
        else:
            try:
                if os.path.exists(data_path):
                    # If raw data exists, just process it
                    print(f"Found existing {data_path}, running data processing...")
                    data_processing(pd.read_pickle(data_path), cwd, freq)
                else:
                    # load data from fresh, and process + evaluate
                    print("No processed or raw data found, running full loading + processing")
                    data_processing(data_loading(path2obs, case_name, path2model), cwd, freq)
            except Exception as e:
                print(f"Error during processing: {e}")
        return None


#%% RUN ME

if EVALUATE == True:
    mean_model, mean_insitu, stats = the_whole_enchilada(path2obs, path2model, cwd, freq, evaluate=True)
else:
    the_whole_enchilada(path2obs, path2model, cwd, freq, evaluate=False)

#%%

if PLOTTING == True:
    data = pd.read_pickle(cwd+'/'+case_name+'_processed_data.pkl')

    minlon = -124.409591
    minlat = 32.534156
    maxlon = -114.131211
    maxlat = 42.009518
    print('\n')

    time = mean_model['air_temp'].index

    for var,count in zip(mean_model.keys(),np.arange(0,len(mean_model.keys()))):

        if var=='fuel_moisture':
            mean_model[var] = mean_model[var]*100

        mean_model[var] = mean_model[var].sort_index()

        mean_insitu[var] = mean_insitu[var].sort_index()


        print('\rplotting data... '+ str(np.round((count/len(mean_model.keys()))*100,3))+'%',end="\r")

        try:

            plt.figure(figsize=(10,5),dpi=175)
            plt.plot(time,mean_model[var],c='dodgerblue',label='Prediction')
            plt.plot(time,mean_insitu[var],c='red',label='Synoptic Measurement')
            plt.ylabel(var)
            plt.xlabel('Time (UTC)')
            plt.xticks(rotation=45)
            plt.title(var+case_name+'\nTime Series RMSE: %s'%np.round(rmse(mean_model[var],mean_insitu[var]),2)+'\nTime Series Pearson: %s'%np.round(pearson(mean_model[var],mean_insitu[var]),2))
            plt.ylabel(var)
            plt.legend(fontsize="x-large")
            #plt2.legend(fontsize="x-large")
            plt.tight_layout()
            plt.savefig('fnetv2timeSeries'+var+'.png')

        except:
            continue

        finally:

            try:

                #plt.figure(figsize=(7,7),dpi=150)
                plt.figure(figsize=(10,7),dpi=150)

                line = [mean_model[var].max(),mean_insitu[var].max(),mean_model[var].min(),mean_insitu[var].min()]

                plt.xlim(np.min(line),np.max(line))

                plt.ylim(np.min(line),np.max(line))

                ax = plt.gca()

                ax.plot([0, 1], [0, 1], transform=ax.transAxes,linestyle='--',color='black', label='f(x) = x')

                plt.plot(np.arange(np.min(line),np.max(line)), np.poly1d(np.polyfit(mean_model[var], mean_insitu[var], 1))(np.arange(np.min(line),np.max(line))),linestyle='-',color='black', label = 'f(x) = '+str(np.poly1d(np.polyfit(mean_model[var], mean_insitu[var], 1)))[2:])

                plt.scatter(mean_model[var],mean_insitu[var],c='black')

                plt.xlabel('model Predictions')

                plt.ylabel('Observed')

                plt.title(var+'\nScatterplot RMSE: %s'%np.round(rmse(mean_model[var],mean_insitu[var]),2)+'\nScatterplot Pearson Correlation: %s'%np.round(pearson(mean_model[var],mean_insitu[var]),2))

                plt.legend(["Prediction, Measurement"],  fontsize="x-large")

                plt.savefig(case_name+'scatterPlot'+var+'.png')
            except:
                continue
            
            finally:

                if var == 'wind_speed' or var =='wind_direction':


                    plt.figure(figsize=(10,10),dpi=150)

                    #you usually have to tweak the epsg and corners to look good



                    map = Basemap(projection='merc',
                                    resolution='l',
                                    epsg = 4326,
                                    urcrnrlon=maxlon,
                                    llcrnrlat=minlat,
                                    llcrnrlon=minlon,
                                    urcrnrlat=maxlat)



                    #tc background

                    map.arcgisimage(server='http://server.arcgisonline.com/arcgis',service='World_Imagery',verbose= False)

                    #map.drawcoastlines()

                    x = []
                    y = []
                    c = []
                    names = []

                    data = pd.read_pickle(cwd+'/'+case_name+'_processed_data.pkl')

                    for name, coords in data.keys():
                        try:
                            lon = float(data[name]['situ'].attrs['LONGITUDE'])
                            lat = float(data[name]['situ'].attrs['LATITUDE'])
                            x, y = map(lon, lat)

                            u_model, v_model = wind_components(data[(name,coords)]['model']['wind_speed'].mean() * units('m/s'), data[(name,coords)]['model']['wind_direction'].mean() * units.deg)

                            u_situ, v_situ = wind_components(data[(name,coords)]['situ']['wind_speed'].mean() * units('m/s'), data[(name,coords)]['situ']['wind_direction'].mean() * units.deg)

                            map.barbs(x, y,u_model*1.94384, v_model*1.94384, color='dodgerblue',length=8, alpha=1)#, label='model Wind Avg.')

                            map.barbs(x, y,u_situ*1.94384, v_situ*1.94384, color='red',length=8, alpha=1)#, label='Station Wind Avg.')

                        except:
                            continue



                    #map.barbs(x, y,u_situ, v_situ, color='red',length=5, alpha=0.5)

                    plt.title('Average Station vs. model Wind Barb')

                    lons = [-124.409591, -114.131211]
                    lats = [32.534156, 42.009518]
                    
                    #lons = [-114.131211, 42.009518]
                    #lats = [-124.409591, 32.534156]
                    x, y = map(lons, lats)
                    
                    cb = map.scatter(x,y,c=c,s=75, cmap='rainbow')
                    #map.scatter(map(-156.67803, 20.87913)[0], map(-156.67803, 20.87913)[1],c='red',s=75,vmin=0.5,vmax=1, marker='X', label = 'Lahaina')
                    import matplotlib.patches as mpatches

                    red_patch = mpatches.Patch(color='red', label='Station Wind Avg.')
                    blue_patch = mpatches.Patch(color='dodgerblue', label='model Wind Avg.')
                    plt.legend(handles=[red_patch, blue_patch])

                    # plt.show()
                    plt.legend(["Prediction, Measurement"],  fontsize="x-large")

                    plt.colorbar(cb, fraction=0.033, pad=0.04,label='RMSE')
                    plt.savefig(case_name+'modelWindBarb'+var+'.png')



                plt.figure(figsize=(10,10),dpi=150)

                #you usually have to tweak the epsg and corners to look good
                map = Basemap(projection='merc',
                                resolution='l',
                                epsg = 4326,
                                urcrnrlon=maxlon,
                                llcrnrlat=minlat,
                                llcrnrlon=minlon,
                                urcrnrlat=maxlat)

                #tc background

                map.arcgisimage(server='http://server.arcgisonline.com/arcgis',service='World_Imagery',verbose= False)

                #map.drawcoastlines()

                x = []
                y = []
                c = []
                names = []

                for name, coords in stats.keys():
                    try:

                        lon = float(data[name]['situ'].attrs['LONGITUDE'])
                        lat = float(data[name]['situ'].attrs['LATITUDE'])
                        c.append(stats[name][var]['rmse'])
                        X, Y = map(lon, lat)
                        x.append(X); y.append(Y)
                        names.append(name)

                    except:
                        continue

                plt.title('Temporally Averaged RMSE values of %s' %var)
                #lons = [-114.131211, 42.009518]
                #lats = [-124.409591, 32.534156]

                #x, y = map(lons, lats)

                cb = map.scatter(x,y,c=c,s=75, cmap='rainbow')
                #map.scatter(map(-156.67803, 20.87913)[0], map(-156.67803, 20.87913)[1],c='red',s=75,vmin=0.5,vmax=1, marker='X', label = 'Lahaina')
                plt.legend(["Prediction, Measurement"],  fontsize="x-large")

                plt.colorbar(cb, fraction=0.033, pad=0.04,label='RMSE')
                plt.savefig(case_name+var+'.png')
