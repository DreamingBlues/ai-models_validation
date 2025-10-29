#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:48:32 2024

@author: cdwehrke
wg: comments by Werner
"""

#Main script used for WRF validation. Contains multiple functions that will save pickle files once complete.

#Output pickle files will have time/space interpolated WRF data at every station in the domain.
# "case_name_processed_data.pkl" will have time series of WRF and measurements for every single station
# "case_name_wrfmean_data.pkl" is a single domain averaged time series of WRF data (at stations locations) for each variable
# "case_name_situmean_data.pkl" is a single domain averaged time series of station data for each variable


# To run, you will need:

# - A JSON with weather data from the Synoptic Weather API (run synreq.py),

# - A csv with weather variable names from Synoptic and how to calculate those in WRF (provided as synoptic_varlist.csv). 
## - This determines which variables are validated. Full list of available variables can be found here: https://demos.synopticdata.com/variables/index.html

## - WRF output files, with one frame per output file.

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
import warnings
import datetime
from metpy.calc import wind_components
from datetime import datetime, timedelta
from metpy.io import metar
from metpy.plots.declarative import (BarbPlot, ContourPlot, FilledContourPlot, MapPanel,
                                     PanelContainer, PlotObs)
import earthkit.data as ekd
from mpl_toolkits.basemap import Basemap



# Consistent settings (shared by all models) 
path2obs = 'sensorData/timeseries'   # Synoptic API JSON
freq     = '6h'                       # frequency of WRF/output files
cwd      = os.getcwd()

# Pick the model you want to run
case_name = 'aurora_0.25_pre'  
"""     
 options: 'fourcastv2net'
          'fourcastv2netHRES'
          'fourcastv2netEra5'
          'aurora_0.25_pre'
          'aurora_0.25_fine'
          'aurora_0.1_fine
"""
# Switch: per-model file/variable settings
## PATH NAMES ARE TEMPORARY  UPDATE FOR SPECIFC CASE
match case_name:
    case 'fourcastv2net':
        var_ref_path = 'fourcastnetv2-small/synoptic_varlist_fnetv2_era5.csv'
        path2wrf     = 'fourcastnetv2-small.grib'
    case 'fourcastv2netHRES':
        var_ref_path = 'fourcastnetv2-small/synoptic_varlist_fnetv2_ifs_hres.csv'
        path2wrf     = 'fourcastnetv2-small_ifs_hres.grib'
    case 'fourcastv2netEra5':
        var_ref_path = 'fourcastnetv2-small/synoptic_varlist_fnetv2_era5.csv'
        path2wrf     = 'fourcastnetv2-small_era5.grib'
    case 'aurora_0.25_pre':
        var_ref_path = 'aurora/synoptic_varlist_aurora.csv'
        path2wrf     = '/shome/u014930890/PGE Projects/aurora_10day/data/aurora-2.5-pretrained_1.grib'
    case 'aurora_0.25_fine':
        var_ref_path = 'aurora/synoptic_varlist_aurora.csv'
        path2wrf     = 'aurora-2.5-fine.grib'
    case 'aurora_0.1_fine':
        var_ref_path = 'aurora/synoptic_varlist_aurora.csv'
        path2wrf     = 'aurora-1.0-fine.grib'
    case _:
        raise ValueError(
            f"Unknown case_name='{case_name}'. "
            "Valid: fourcastv2net, fourcastv2netHRES, fourcastv2netEra5, aurora_0.25_pre"
        )
# Initialize from the selected case 
var_ref = pd.read_csv(var_ref_path, engine='python')

## Consistent settings
path2obs = 'sensorData/timeseries' #Synoptic API JSON
freq = '6h' #frequency of wrfoutput files



PLOTTING=False
EVALUATE=False



#%% DATA ASSIMILATION
def data_assimilation(path2obs, path2wrf,cwd):

    # shared params, Aurora, FCNetv2, ERA5
    params = ['2t', 'r', '10u', '10v', 'sp', 'z']

    # Only HRES differs in params, uses 'gh' instead of 'z'
    if case_name == 'fourcastv2netHRES':
        params = ['2t', 'r', '10u', '10v', 'sp', 'gh']

    # Read GRIB and select variables
    ds   = ekd.from_source("file", path2wrf)
    wrf1 = ds.sel(param=params).to_xarray()

    # HRES-only: convert geopotential height -> geopotential
    if case_name == 'fourcastv2netHRES':
        wrf1['z'] = wrf1['gh'] * 9.80665  # g [m/s^2]

    # Time handling
    if case_name == 'fourcastv2netEra5':
        # ERA5 uses forecast_reference_time → derive step
        wrf1 = wrf1.assign_coords(valid_time=wrf1.forecast_reference_time)
        step = [wt-wrf1.valid_time.data[0] for wt in wrf1.valid_time.data]
        wrf1 = wrf1.rename({'forecast_reference_time': 'step'}).assign_coords(step=step)
    else:
        # Everyone else: reference datetime + step
        ref_date = pd.to_datetime(str(wrf1.attrs['date']), format='%Y%m%d')
        ref_time = pd.Timedelta(minutes=int(wrf1.attrs.get('time', 0)))
        ref_datetime   = (ref_date + ref_time).to_datetime64()
        valid_time = ref_datetime + wrf1['step']
        wrf1 = wrf1.assign_coords(valid_time=valid_time)

    data = {}

    #loading in output from synoptic json req.
    js = json.load(open(path2obs))

    master = {}

    #when i made this code, i was using an output from one of Angel's scripts. this emulates the output and saved me a lot of time in recoding ? what does this do?
    for site in js['STATION']:

        master[site['STID']] = pd.DataFrame.from_dict(site['OBSERVATIONS'])
        master[site['STID']] = master[site['STID']].set_index(pd.to_datetime(master[site['STID']]['date_time']))
        master[site['STID']].attrs = {k: site[k] for k in set(list(site.keys())) - set('OBSERVATIONS')}

    #starting actual processing
    for site,number in zip(pd.Series(master.keys()),range(len(pd.Series(master.keys())))):

        print('\rassimilating data... '+ str(np.round((number/len(master.keys()))*100,3))+'%',end="\r")

        uncorr = master[site]

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


        # lat,lon of each station needed for point selection of wrf data
        lon = float(df.attrs['LONGITUDE'])
        lat = float(df.attrs['LATITUDE'])
        hgt = float(df.attrs['ELEVATION'])

        varlist = []

        for i in df.keys():
            varlist.append(i)

        #this allows me to only calculate wrf variables for data we also have obs for


        # creating key in data for this station

        data[(df.attrs['STID'],(lon,lat))] = {'situ':[],'wrf':[]}

        # appending sub dictionary with wx data

        data[(df.attrs['STID'],(lon,lat))]['situ'] = df

        #WX data complete, moving onto wrf...

        # working on the grib file                

        for step, count in zip(wrf1.step.data, np.arange(len(wrf1.step.data))):

            wrf = wrf1.sel(step=step)

            # grabbing wrf file
            if count == 0:
                #squeeze bc time variable is 0, assuming 1 time output per file
                longitude = (wrf.longitude + 180.) % 360 - 180.
                latitude = wrf.latitude
                
                # thx stack. this is selecting wrf data at one point
                abslat = np.abs(latitude-lat)

                abslon = np.abs(longitude-lon)

                c = abslon + abslat

                try:
                    ([xloc], [yloc]) = np.where(c == np.min(c))
                except:
                    (xloc, yloc) = (np.where(c == np.min(c))[0][0], np.where(c == np.min(c))[0][1])

            wrf = wrf.isel(longitude=xloc, latitude=yloc)
            
            tempdf ={}

            tempdf['date_time'] = pd.to_datetime(wrf.valid_time.data, utc=True)

            #calculating wrf variables based on formulas in var_ref (an excel sheet i made)
            for variable, name in zip(var_ref['variable'],var_ref['description']):

                if variable in varlist:
                    #print(variable)
                    exec('tempdf[variable] =' + var_ref['formula'][list(var_ref['variable']).index(variable)])


            tempdf = pd.DataFrame(tempdf,index=[0])

            tempdf = tempdf.set_index(tempdf['date_time'])

            if count ==0:
                data[(df.attrs['STID'],(lon,lat))]['wrf'] = tempdf
            else:
                data[(df.attrs['STID'],(lon,lat))]['wrf'] = pd.concat([data[(df.attrs['STID'],(lon,lat))]['wrf'],tempdf])
            #done with wrf

    #saving, as this can take a while to run
    with open(cwd+'/'+case_name+'_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    return data

#%% DATA PROCESSING

def data_processing(data,cwd,freq):

    #effectively just resampling the data and making it pretty.
    print('\n')

    processed = {}

    for key,count in zip(data.keys(),np.arange(0,len(data.keys()))):

        print('\rprocessing data... '+ str(np.round((count/len(data.keys()))*100,3))+'%',end="\r")
        #clipping to wrf time range (safest)
        data[key]['situ'] = data[key]['situ'].set_index(pd.to_datetime(data[key]['situ'].index, utc=True))

        # --- guards ---
        if (
            'wrf' not in data[key]
            or data[key]['wrf'] is None
            or data[key]['situ'].empty
            or data[key]['wrf'].empty
            or 'date_time' not in data[key]['wrf']
        ):
            print(f"Skipping {key} (missing or empty data)")
            continue

        start = np.maximum(
            data[key]['wrf']['date_time'].iloc[0],
            data[key]['situ'].index.min()
        )
        end = np.minimum(
            data[key]['wrf']['date_time'].iloc[-1],
            data[key]['situ'].index.max()
        )

        if start >= end:
            print(f"Skipping {key} (no overlap between WRF and situ times)")
            continue

        time_wrf = pd.date_range(start, end, freq=freq)
        if len(time_wrf) == 0:
            print(f"Skipping {key} (empty time range)")
            continue
        # --- end guards ---


        processed[key] = {'situ':[],'wrf':[]}
        #resampling to freq (defined manually below)
        data_slice = data[key]['situ'].loc[time_wrf[0]:time_wrf[-1]]
        data_slice = data_slice.loc[~data_slice.index.duplicated(keep='first')]
        processed[key]['situ'] = data_slice.rolling(freq).mean(numeric_only=True).interpolate(kind='linear').resample(freq).mean()
        data_slice = data[key]['wrf'].loc[time_wrf[0]:time_wrf[-1]]
        data_slice = data_slice.loc[~data_slice.index.duplicated(keep='first')]
        processed[key]['wrf'] = data_slice.rolling(freq).mean(numeric_only=True).interpolate(kind='linear').resample(freq).mean()
        
    #saving
    with open(cwd+'/'+case_name+'_processed_data.pkl', 'wb') as g:
        pickle.dump(processed, g)

    return processed

#%% DATA EVALUATION

def data_evaluation(processed,cwd, freq):

    print('\n')

    evaluated = {}

    for key in processed.keys():

        try:

            processed[key]['wrf'] = processed[key]['wrf'].tz_localize(tz='utc')
            processed[key]['situ'] = processed[key]['situ'].tz_localize(tz='utc')

            time_wrf = pd.date_range(processed[key]['situ'].index[0],
                                     processed[key]['situ'].index[-1],
                                     freq=freq)


            processed[key]['situ'] = ((processed[key]['situ'].loc[time_wrf[0]:time_wrf[-1]]))

            processed[key]['wrf'] = ((processed[key]['wrf'].loc[time_wrf[0]:time_wrf[-1]]))

        except:
            continue


    for key,count in zip(processed.keys(),np.arange(0,len(processed.keys()))):

        try:

            evaluated[key] = {}

            print('\revaluating data statistics... '+ str(np.round((count/len(processed.keys()))*100,3))+'%',end="\r")

            for variable in processed[key]['wrf'].keys():
                if variable == 'date_time':
                    continue
                else:
                    evaluated[key][variable] = {'rmse':[rmse(processed[key]['wrf'][variable],
                                                             processed[key]['situ'][variable])],

                                                'pearson':[pearson(processed[key]['wrf'][variable],
                                                           processed[key]['situ'][variable])]
                                                    }
        except:
            continue

    concat_wrf = {}

    concat_situ = {}

    mean_wrf = {}

    mean_situ = {}

    for key in processed.keys():

        for variable in processed[key]['wrf'].keys():

            if variable == 'date_time':
                continue
            else:

                concat_wrf[variable] = pd.DataFrame()

                concat_situ[variable] = pd.DataFrame()

    print('\n')

    for key,count in zip(processed.keys(),np.arange(0,len(processed.keys()))):

        print('\rspatially concatenating data... '+ str(np.round((count/len(processed.keys()))*100,3))+'%',end="\r")

        for variable in processed[key]['wrf'].keys():

            if variable == 'date_time':
                continue

            else:

                concat_wrf[variable] = pd.concat([concat_wrf[variable],processed[key]['wrf'][variable]],axis=1)

                concat_situ[variable] = pd.concat([concat_situ[variable],processed[key]['situ'][variable]],axis=1)


    for variable in concat_wrf.keys():
        mean_wrf[variable] = concat_wrf[variable].mean(axis=1)#.fillna(0).sort_index()#((concat_wrf[variable].sum(axis=1,numeric_only=True))/len(concat_wrf[variable].columns)).sort_index()
        mean_situ[variable] = concat_situ[variable].mean(axis=1)#.fillna(0).sort_index()#((concat_situ[variable].sum(axis=1,numeric_only=True))/len(concat_situ[variable].columns)).sort_index()

    with open(cwd+'/'+case_name+'_evaluated_data.pkl', 'wb') as a:
        pickle.dump(evaluated, a)

    with open(cwd+'/'+case_name+'_wrfmean_data.pkl', 'wb') as b:
        pickle.dump(mean_wrf, b)

    with open(cwd+'/'+case_name+'_situmean_data.pkl', 'wb') as c:
        pickle.dump(mean_situ, c)


    return mean_wrf, mean_situ, evaluated


#%%WRAPPING IT UP

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def pearson(predictions, targets):
    return scipy.stats.pearsonr(predictions,targets)[0]

def the_whole_enchilada(path2obs,path2wrf,cwd,freq,):

    try:
        mean_wrf, mean_situ, evaluated = data_evaluation(pd.read_pickle(cwd+'/'+case_name+'_processed_data.pkl'),cwd, freq)

    except:

        try:

            mean_wrf, mean_situ, evaluated = data_evaluation(data_processing(pd.read_pickle(cwd+'/'+case_name+'_data.pkl'),cwd,freq),cwd, freq)

        except:

            mean_wrf, mean_situ, evaluated = data_evaluation(data_processing(data_assimilation(path2obs,path2wrf,cwd),cwd,freq),cwd, freq)

    return mean_wrf, mean_situ, evaluated

#%% RUN ME


if EVALUATE == True:
    mean_wrf, mean_insitu, stats = the_whole_enchilada(path2obs, path2wrf, cwd, freq)

else:
    # Skip evaluation: only make sure processed data exists
    processed_path = f"{cwd}/{case_name}_processed_data.pkl"
    data_path = f"{cwd}/{case_name}_data.pkl"

    if os.path.exists(processed_path):
        print(f"Found existing {processed_path}, skipping processing")

    else:
        try:
            if os.path.exists(data_path):
                print(f"Found existing {data_path}, running data processing...")
                data_processing(pd.read_pickle(data_path), cwd, freq)
            else:
                print("No processed or raw data found, running full assimilation + processing")
                data_processing(data_assimilation(path2obs, path2wrf, cwd), cwd, freq)
        except Exception as e:
            print(f"Error during processing: {e}")


#%%

if PLOTTING == True:
    data = pd.read_pickle(cwd+'/'+case_name+'_processed_data.pkl')

    minlon = -124.409591
    minlat = 32.534156
    maxlon = -114.131211
    maxlat = 42.009518
    print('\n')

    time = mean_wrf['air_temp'].index

    for var,count in zip(mean_wrf.keys(),np.arange(0,len(mean_wrf.keys()))):

        if var=='fuel_moisture':
            mean_wrf[var] = mean_wrf[var]*100

        mean_wrf[var] = mean_wrf[var].sort_index()

        mean_insitu[var] = mean_insitu[var].sort_index()


        print('\rplotting data... '+ str(np.round((count/len(mean_wrf.keys()))*100,3))+'%',end="\r")

        try:

            plt.figure(figsize=(10,5),dpi=175)
            plt.plot(time,mean_wrf[var],c='dodgerblue',label='Prediction')
            plt.plot(time,mean_insitu[var],c='red',label='Synoptic Measurement')
            plt.ylabel(var)
            plt.xlabel('Time (UTC)')
            plt.xticks(rotation=45)
            plt.title(var+case_name+'\nTime Series RMSE: %s'%np.round(rmse(mean_wrf[var],mean_insitu[var]),2)+'\nTime Series Pearson: %s'%np.round(pearson(mean_wrf[var],mean_insitu[var]),2))
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

                line = [mean_wrf[var].max(),mean_insitu[var].max(),mean_wrf[var].min(),mean_insitu[var].min()]

                plt.xlim(np.min(line),np.max(line))

                plt.ylim(np.min(line),np.max(line))

                ax = plt.gca()

                ax.plot([0, 1], [0, 1], transform=ax.transAxes,linestyle='--',color='black', label='f(x) = x')

                plt.plot(np.arange(np.min(line),np.max(line)), np.poly1d(np.polyfit(mean_wrf[var], mean_insitu[var], 1))(np.arange(np.min(line),np.max(line))),linestyle='-',color='black', label = 'f(x) = '+str(np.poly1d(np.polyfit(mean_wrf[var], mean_insitu[var], 1)))[2:])

                plt.scatter(mean_wrf[var],mean_insitu[var],c='black')

                plt.xlabel('WRF Predictions')

                plt.ylabel('Observed')

                plt.title(var+'\nScatterplot RMSE: %s'%np.round(rmse(mean_wrf[var],mean_insitu[var]),2)+'\nScatterplot Pearson Correlation: %s'%np.round(pearson(mean_wrf[var],mean_insitu[var]),2))

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

                            x = (map(coords[0],coords[1])[0])
                            y = (map(coords[0],coords[1])[1])

                            # x.append(map(coords[0],coords[1])[0])
                            # y.append(map(coords[0],coords[1])[1])

                            u_wrf, v_wrf = wind_components(data[(name,coords)]['wrf']['wind_speed'].mean() * units('m/s'), data[(name,coords)]['wrf']['wind_direction'].mean() * units.deg)

                            u_situ, v_situ = wind_components(data[(name,coords)]['situ']['wind_speed'].mean() * units('m/s'), data[(name,coords)]['situ']['wind_direction'].mean() * units.deg)

                            map.barbs(x, y,u_wrf*1.94384, v_wrf*1.94384, color='dodgerblue',length=8, alpha=1)#, label='WRF Wind Avg.')

                            map.barbs(x, y,u_situ*1.94384, v_situ*1.94384, color='red',length=8, alpha=1)#, label='Station Wind Avg.')

                        except:
                            continue



                    #map.barbs(x, y,u_situ, v_situ, color='red',length=5, alpha=0.5)

                    plt.title('Average Station vs. WRF Wind Barb')

                    lons = [-124.409591, -114.131211]
                    lats = [32.534156, 42.009518]
                    
                    #lons = [-114.131211, 42.009518]
                    #lats = [-124.409591, 32.534156]
                    x, y = map(lons, lats)
                    
                    cb = map.scatter(x,y,c=c,s=75, cmap='rainbow')
                    #map.scatter(map(-156.67803, 20.87913)[0], map(-156.67803, 20.87913)[1],c='red',s=75,vmin=0.5,vmax=1, marker='X', label = 'Lahaina')
                    import matplotlib.patches as mpatches

                    red_patch = mpatches.Patch(color='red', label='Station Wind Avg.')
                    blue_patch = mpatches.Patch(color='dodgerblue', label='WRF Wind Avg.')
                    plt.legend(handles=[red_patch, blue_patch])

                    # plt.show()
                    plt.legend(["Prediction, Measurement"],  fontsize="x-large")

                    plt.colorbar(cb, fraction=0.033, pad=0.04,label='RMSE')
                    plt.savefig(case_name+'WRFWindBarb'+var+'.png')



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

                        c.append(stats[(name,coords)][var]['rmse'])
                        x.append(map(coords[0],coords[1])[0])
                        y.append(map(coords[0],coords[1])[1])
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
