#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:52:27 2024

@author: tolul
"""

import itertools
import matplotlib.pyplot as plt
import sys
import os
import fnmatch
from iono_filtering import get_iono_filt
import glob
import re
import multiprocessing
import numpy as np
import math
from netCDF4 import Dataset
import pandas as pd
import datetime
import time
from scipy.signal import butter, filtfilt
import ephem
import yaml
from copy import deepcopy
from scipy.interpolate import griddata
import xarray as xr
import traceback
import logging

from memory_profiler import profile
import signal


def filter_hp(data, lowcut=1 / 60 / 60 / 24 / 30 / 2, fs=1 / 60 / 60 / 24 / 10):
    """
    High pass filter at two months, for data : array 1D with a resolution of 10 days
    """

    b, a = butter(20, lowcut, btype="hp", fs=1 / 60 / 60 / 24 / 10)
    y = filtfilt(b, a, data)
    return y


def max_consecutive_nan(arr):
    """
    Returns a float corresponding to the maximum number of consecutive missing data in the 1D array 
    """

    max_count = 0
    current_count = 0
    for value in arr:
        if np.isnan(value):
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    return max_count


def datetime_to_julian(time):
    """
    Converts datetime to julian time (days since 1950-01-01).
    time is not an array, it is a single value.
    """

    epoch = datetime.datetime(1950, 1, 1)
    julian_day = ephem.julian_date(time) - ephem.julian_date(epoch)
    return julian_day



def fillna_interptrack(dfsol):
    """
    Fills missing data by interpolating along the track. If there is still missing data, it means one entire cycle was missing for all the points of the track, so we interpolate along the time. 
    dfsol is a dataframe of dimensions : (nb_cycles,nb_pts_track)
    """

    if dfsol.isnull().values.any():
        dfsol=dfsol.interpolate(axis=1, limit_direction='both')
    if dfsol.isnull().values.any():
        dfsol=dfsol.interpolate(axis=0, limit_direction='both')
    return dfsol

def create_file_nc(output_filename, list_corr, list_df, filename_trend):                
    """
    Creates a netcdf file containing the σ values of Jason-2, Jason-3, and of the whole period (Jason-1+Jason-2+Jason-3).  
    """

    position = list_df[0].index                                                                                        
    time_tot_days = list_df[0].columns                                                                                           
                                                                                                                        
    f=Dataset(filename_trend)                                                                       
    time_tot_months = f.variables['time'][:]                                                                                     
    sla = f.variables['sla'][:]                                                                                
    f.close()                                                                                                           
    df_sla = pd.DataFrame(sla, columns=time_tot_months)                                                          
    df_sla = df_sla.interpolate(axis=1, limit_direction='both')                                         
                                                                                                                        
    with Dataset(output_filename, 'w') as nc :                                                                          
        dx = nc.createDimension('dx', size = len(position))                                                             
        dt =  nc.createDimension('dt', size=len(time_tot_days)) 
        dtj2 =  nc.createDimension('dtj2', size=np.shape(list_df[1])[1]) 
        dtj3 =  nc.createDimension('dtj3', size=np.shape(list_df[2])[1]) 
        dmonths = nc.createDimension('dmonths', size=len(time_tot_months))                                                       
                                                                                                                        
        vx = nc.createVariable('position', 'f', dimensions = ('dx'))                                                    
        vx.description = 'index of points along the track'                                                        
        vx[:] = position                                                                                                
                                                                                                                        
        vt = nc.createVariable('time', 'f', dimensions = ('dt'))                                                        
        vt.unit = 'CNES Julian day (days since 01-01-1950)'                                                             
        vt[:] = time_tot_days                                                                                                     
                                                                                                                        
        vmonths = nc.createVariable('time_months', 'f', dimensions = ('dmonths'))                                       
        vmonths.unit = 'CNES Julian day (days since 01-01-1950)'                                                        
        vmonths[:] = time_tot_months                                                                                             
                                                                                                                        
        vsla = nc.createVariable('sea_level_anomaly', 'f8', dimensions = ('dx', 'dmonths'))                             
        vsla.units = 'm'                                                                                                
        vsla.description = "The sla are monthly averaged and annual and semi-annual cycles are removed.sla = altitude of satellite - 20 Hz Ku band ALES corrected altimeter range (Passaro et al. 2014) - altimeter ionospheric correction on Ku band (From dual-frequency altimeter range measurements) - model dry tropospheric correction(From ECMWF model) - GPD+ wet tropospheric correction (Fernandes et al. 2016)- sea state bias correction in Ku band (ALES retracking, Passaro et al. 2014) - solid earth tide height (From RADS, tide potential model, Cartwright and Taylor 1971, Cartwright and Eden 1973) - geocentric ocean tide (FES 2014 from RADS, Carrere et al. 2012)- geocentric pole tide height (Wahr 1985) - Atmospheric correction (From RADS, Carrere and Lyard 2003) - X-TRACK mean sea surface (Birol et al. 2017) - ICE_6G_D GIA (Peltier et al. 2015). Each corrective term is edited following Birol et al. 2017."
        vsla[:] = df_sla                                                                                       

        vcorr =  nc.createVariable(list_corr[0], 'f8', dimensions=('dx', 'dt'))                                    
        vcorr.units = 'm²' 
        vcorr.comment = "High frequency noise variance of the geophysical corrections on period from Jason-1 to Jason-3."
        vcorr[:] = list_df[0] 

        vcorrj2 =  nc.createVariable(list_corr[1], 'f8', dimensions=('dx', 'dtj2')) 
        vcorrj2.comment = "High frequency noise variance of the geophysical corrections on Jason-2 mission."
        vcorrj2.units = 'm²'                                                                                           
        vcorrj2[:] = list_df[1] 

        vcorrj3 =  nc.createVariable(list_corr[2], 'f8', dimensions=('dx', 'dtj3'))                                    
        vcorrj3.comment = "High frequency noise variance of the geophysical corrections on Jason-3 mission."
        vcorrj3.units = 'm²'                                                                                           
        vcorrj3[:] = list_df[2]      



def interpolate_nearest(raw_data, raw_lats, new_lats):
    """
    Interpolates raw data (array 1D), with coordinates raw_lats (array 1D) on new_lats (array 1D) with method nearest. If there is not enough data to interpolate (less than 10 points between new_lats.min() and new_lats.max()), there is no interpolation, the data is put to NaN. This prevents overly abrupt interpolation. 
    """

    if len(raw_lats[(raw_lats<=new_lats.max()) & (raw_lats>=new_lats.min())]) >=10:
        raw_data = np.array(raw_data)
        raw_lats = np.array(raw_lats)
        new_lats = np.array(new_lats)

        interpolated_data = griddata(raw_lats, raw_data, new_lats, method="nearest")
    else :
        interpolated_data = new_lats.copy()
        interpolated_data[:]=np.nan

    return interpolated_data

def get_df_corr_track(filename_rr, correction, filename_time, filename_trend):
    """
    Returns a dataframe containing the formatted correction with dimensions : (nb_cycles,npts_track).
    nb_cycles corresponds to the number of cycles available for the mision in the round robin data : around 111 cycles (3 years)
    """

    f = Dataset(filename_trend, 'r')
    lat_trend = f.variables['lat'][:]
    f.close()
    f = Dataset(filename_time, 'r')
    nbcycles = len(f.variables['cycle'][:])
    f.close()
    lat_trend = np.tile(lat_trend, (nbcycles, 1))

    g = Dataset(filename_rr,'r')
    time_raw=g.variables['time'][:]
    time_float=[float(i) for i in time_raw]
    yref=int(g.variables['time'].units[19:23])
    mref=int(g.variables['time'].units[24:26])
    dref=int(g.variables['time'].units[27:29])
    href=int(g.variables['time'].units[30:32])
    minref=int(g.variables['time'].units[33:35])
    sref=int(g.variables['time'].units[36:38])
    usref=int(g.variables['time'].units[39:45])
    tref=datetime.datetime(yref, mref, dref, href, minref, sref, usref)
    time = np.array([tref + datetime.timedelta(microseconds=t) for t in time_float])
    time_julian = np.array([datetime_to_julian(tref + datetime.timedelta(microseconds=t)) for t in time_float])

    cycle=g.variables['cycles_indexes'][:]

    lat=g.variables['LATITUDE'][:]
    lon=g.variables['LONGITUDE'][:]
    corr=g.variables[correction][:]
    if correction == 'MEAN_SEA_SURFACE.MODEL.CNESCLS21.CTOH_J3' or correction == 'MSS_SIO.CTOH_J3':
        ellips_corr=g.variables['ellipsoid_correction'][:]
        corr=corr-ellips_corr
    g.close()

    df=pd.DataFrame({'corr' : corr}, index=cycle) # df has only one column
    df['cumcount'] = df.groupby(level=0).cumcount()
    # We use the variable cycle to transform the dataframe into a 2D one, with cycles in index and points of the track in columns
    df = df.groupby(level=0).apply(lambda x: pd.Series(x['corr'].values)).unstack()
    df.columns = range(1, df.shape[1] + 1)

    dflat=pd.DataFrame({'lat' : lat}, index=cycle)
    dflat['cumcount'] = dflat.groupby(level=0).cumcount()
    dflat = dflat.pivot_table(index=dflat.index, columns='cumcount', values='lat', aggfunc='first', fill_value=None)
    dflat.columns = range(1, dflat.shape[1] + 1)

    dftime=pd.DataFrame({'time' : time}, index=cycle)
    dftime['cumcount'] = dftime.groupby(level=0).cumcount()
    dftime = dftime.pivot_table(index=dftime.index, columns='cumcount', values='time', aggfunc='first', fill_value=None)
    time=dftime.to_numpy()[:,0]

    # These corrections are treated separately as they need to be filtered before use
    if correction == 'IONOSPHERIC_CORRECTION.ALTI' or correction == 'IONOSPHERIC_CORRECTION.ALTI.RTK_ADAPTIVE':
        dflon=pd.DataFrame({'lon' : lon}, index=cycle)
        dflon['cumcount'] = dflon.groupby(level=0).cumcount()
        dflon = dflon.pivot_table(index=dflon.index, columns='cumcount', values='lon', aggfunc='first', fill_value=None)
        dflon.columns = range(1, dflon.shape[1] + 1)

        dftimej=pd.DataFrame({'time' : time_julian}, index=cycle)
        dftimej['cumcount'] = dftimej.groupby(level=0).cumcount()
        dftimej = dftimej.pivot_table(index=dftimej.index, columns='cumcount', values='time', aggfunc='first', fill_value=None)
        timej=dftimej.to_numpy()[:,0]
        df=get_iono_filt(df, dflat, dflon, dftimej)

    # We keep only the points of the track corresponding to the ones of the filename_trend by interpolating in latitude with nearest function
    lat_trend=np.tile(lat_trend, (len(df), 1))
    dflat_trend=pd.DataFrame(lat_trend)

    interpolated_data = np.array([interpolate_nearest(df.iloc[i], dflat.iloc[i], dflat_trend.iloc[i]) for i in range(df.shape[0])])
    df_interpolated = pd.DataFrame(interpolated_data)
    df_interpolated.index=time
    return df_interpolated


def get_df_ionoxt_trace(filename_ionoxt, filename_rr):
    """
    Returns a dataframe containing the formatted ionospheric X-TRACK correction with dimensions : (nb_cycles,npts_track).
    nb_cycles corresponds to the number of cycles available for the mision in the round robin data : around 111 cycles (3 years).
    It is a separate function because this correction is not on the same file as the others, and has not the same format.
    """

    f=Dataset(filename_ionoxt, 'r')
    iono=f.variables['iono'][:]
    mask=f.variables['mask'][:]
    time_raw=f.variables['time'][:]                                                                                         
    time_float=[float(i) for i in time_raw]                                                                                   
    yref=int(f.variables['time'].units[19:23])                                                                          
    mref=int(f.variables['time'].units[24:26])                                                                          
    dref=int(f.variables['time'].units[27:29])                                                                          
    href=int(f.variables['time'].units[30:32])                                                                          
    minref=int(f.variables['time'].units[33:35])                                                                        
    sref=int(f.variables['time'].units[36:38])                                                                          
    usref=int(f.variables['time'].units[39:45])                                                                         
    tref=datetime.datetime(yref, mref, dref, href, minref, sref, usref)                                                 
    time_iono = np.array([tref + datetime.timedelta(microseconds=t) for t in time_float])
    f.close()  
    
    iono=np.where(mask,iono, np.nan)
    df_ionoxt=pd.DataFrame(iono.transpose(), index=time_iono)  
    df_ionoxt=fillna_interptrack(df_ionoxt)
    
    g = Dataset(filename_rr,'r')                                                                        
    time_raw=g.variables['time'][:]                                                                                          
    time_float=[float(i) for i in time_raw]                                                                                   
    yref=int(g.variables['time'].units[19:23])                                                                          
    mref=int(g.variables['time'].units[24:26])                                                                          
    dref=int(g.variables['time'].units[27:29])                                                                          
    href=int(g.variables['time'].units[30:32])                                                                          
    minref=int(g.variables['time'].units[33:35])                                                                        
    sref=int(g.variables['time'].units[36:38])                                                                          
    usref=int(g.variables['time'].units[39:45])                                                                         
    tref=datetime.datetime(yref, mref, dref, href, minref, sref, usref)                                                 
    time = np.array([tref + datetime.timedelta(microseconds=t) for t in time_float])     
    cycle=g.variables['cycles_indexes'][:]
    g.close()
    dftime=pd.DataFrame({'time' : time}, index=cycle)                                                                    
    dftime['cumcount'] = dftime.groupby(level=0).cumcount()                                                             
    dftime = dftime.pivot_table(index=dftime.index, columns='cumcount', values='time', aggfunc='first', fill_value=None) 
    time_rr=dftime.to_numpy()[:,0]
    
    df_ionoxtinterp = (
        df_ionoxt
        .reindex(df_ionoxt.index.union(time_rr))  
        .sort_index()
        .interpolate(method="time")          
        .loc[time_rr]               
    )

    return df_ionoxtinterp


def filter_hp_df(df):
    """
    High pass filter at two months on a dataframe (time, npts_track).
    If a point contains too much missing data, it is put to NaN.
    """

    df_filt = df.copy()
    for col in df:
        point_track = df[col]
        if max_consecutive_nan(point_track) <= 3:
            point_track = point_track.interpolate()
            df_filt[col] = filter_hp(point_track)
        else:
            df_filt[col] = np.nan
    return df_filt


def vardiff_2a2(list_df):
    """
    Computes the median of variances of pair-wise differences in the elements of the list.
    list contains dataframes of filtered SLA.
    """

    init_shape = np.shape(list_df[0])
    arrays = np.stack([df.values for df in list_df])
    combinations = itertools.combinations(range(arrays.shape[0]), 2)
    combinations = np.array(list(combinations))
    vardiff = (arrays[combinations[:, 0]] - arrays[combinations[:, 1]]).var(axis=1)
    varmed = np.nanmedian(vardiff, axis=0) 
    return varmed


def outliers_treatment(dfsol, correction):
    """
    Returns a dataframe of the edited correction.
    """

    if correction == 'SEA_STATE_BIAS' :
        dfsol[dfsol==0]=np.nan
        dfsol[dfsol<-0.5]=np.nan
        dfsol[dfsol>0.02]=np.nan
    elif correction == 'MEAN_SEA_SURFACE' :
        if (dfsol.max()-dfsol.min()).max() > 1:
            med=dfsol.median().mean()
            dfsol[dfsol<med-1]=med
            dfsol[dfsol>med+1]=med
    elif correction == 'IONO':
        dfsol[dfsol==0]=np.nan
        dfsol[dfsol<-0.6]=np.nan                                                                                        
        dfsol[dfsol>0.02]=np.nan
    elif correction == 'WET':                                                                                            
        dfsol[dfsol<-0.5]=np.nan                                                                                          
        dfsol[dfsol>=0]=np.nan
    elif correction == 'DRY':                                                                                            
        dfsol[dfsol<-2.5]=np.nan                                                                                          
        dfsol[dfsol>-1.9]=np.nan
    return dfsol

def get_variance_corr_rr_sla_fillna(filename_rr, filename_ionoxt, jason, filename_time, filename_trend):
    """
    Returns a dataframe of dimension (npts_track, nt_mission) containing σ, the high frequency noise value related to the geophysical corrections. 
    nt_mission corresponds to the number of cycles of the mission, which is given by the variable jason (either 'J2' or 'J3'). 
    """

    f = Dataset(filename_time, 'r')
    jason_num=f.variables['jason'][:]
    time_raw=f.variables['time'][:]
    time_raw=time_raw[0] #the dimensions of time_raw are (npts_track,nb_cycles). Time varies only along the cycles, so we take the time of the first point
    time_float=[float(i) for i in time_raw]
    tref=datetime.datetime(1950,1,1)
    time_tot = np.array([tref + datetime.timedelta(days=t) for t in time_float])
    f.close()

    if jason == 'J2':
        list_orbit=["ORBIT.ALTI.POE_GDR_E"]
        list_range=["range_20hz_ales_2025"]
        list_ssb=['SEA_STATE_BIAS.ALTI.NON_PARAMETRIC_RTK_ADAPTIVE', 'sea_state_bias_20hz_ales_2025', 'SEA_STATE_BIAS.ALTI.NON_PARAMETRIC']
        list_tide=["OCEAN_TIDE_FES22","OCEAN_TIDE_HEIGHT.MODEL.FES14B","OCEAN_TIDE_HEIGHT.MODEL.GOT4V10","tide_EOT20"]
        list_wet=["WET_TROPOSPHERIC_CORRECTION.RAD","WET_TROPOSPHERIC_CORRECTION.GPD_PLUS","WET_TROPOSPHERIC_CORRECTION.MODEL.ECMWF_GAUSS"]
        list_dry=["DRY_TROPOSPHERIC_CORRECTION.MODEL.ECMWF_GAUSS","DRY_TROPO_NCEP","DRY_TROPO_ERA"]
        list_dac=['DAC_RADS']
        list_iono=["IONO_XTRACK","IONOSPHERIC_CORRECTION.MODEL.GIM"]
        list_mss=["MEAN_SEA_SURFACE.MODEL.CNESCLS15","MEAN_SEA_SURFACE.MODEL.CNESCLS21.CTOH_J2","MSS_SIO.CTOH_J2"]
    if jason == 'J3':
        list_orbit=['ORBIT.ALTI.CNES_POE_F']
        list_range=['range_20hz_ales_2025']
        list_ssb=['SEA_STATE_BIAS.ALTI', 'sea_state_bias_20hz_ales_2025', 'SEA_STATE_BIAS.ALTI.ADAPTIVE_UPDATE']
        list_tide=['OCEAN_TIDE_FES22', 'OCEAN_TIDE_HEIGHT.MODEL.FES14B', 'OCEAN_TIDE_HEIGHT.MODEL.GOT4V10', 'tide_EOT20']
        list_wet=['WET_TROPOSPHERIC_CORRECTION.RAD', 'WET_TROPOSPHERIC_CORRECTION.GPD_PLUS_J3', 'WET_TROPOSPHERIC_CORRECTION.MODEL.ECMWF']
        list_dry=['DRY_TROPOSPHERIC_CORRECTION.MODEL.ECMWF', 'DRY_TROPO_ERA', 'DRY_TROPO_NCEP']
        list_dac=['DAC_RADS']
        list_iono=['IONO_XTRACK', 'IONOSPHERIC_CORRECTION.MODEL.GIM']
        list_mss=['MEAN_SEA_SURFACE.MODEL.CNESCLS15', 'MEAN_SEA_SURFACE.MODEL.CNESCLS21.CTOH_J3', 'MSS_SIO.CTOH_J3']

    list_df_orbit=list_orbit.copy()
    list_df_range=list_range.copy()
    list_df_ssb=list_ssb.copy()
    list_df_tide=list_tide.copy()
    list_df_wet=list_wet.copy()
    list_df_dry=list_dry.copy()
    list_df_dac=list_dac.copy()
    list_df_iono=list_iono.copy()
    list_df_mss=list_mss.copy()

    #for each correction, we collect all the possible algorithms and format them under dataframes with dimensions : (nb_cycles,npts_track)
    for i in range(len(list_orbit)):
        var=list_orbit[i]
        list_df_orbit[i]=get_df_corr_track(filename_rr, var, filename_time, filename_trend)
        list_df_orbit[i]=fillna_interptrack(list_df_orbit[i])

    for i in range(len(list_range)):
        var=list_range[i]
        list_df_range[i]=get_df_corr_track(filename_rr, var, filename_time, filename_trend)
        list_df_range[i]=fillna_interptrack(list_df_range[i])


    for i in range(len(list_ssb)):
        var=list_ssb[i]
        list_df_ssb[i]=get_df_corr_track(filename_rr, var, filename_time, filename_trend)
        list_df_ssb[i]=outliers_treatment(list_df_ssb[i], 'SEA_STATE_BIAS')
        list_df_ssb[i]=fillna_interptrack(list_df_ssb[i])


    for i in range(len(list_tide)):
        var=list_tide[i]
        list_df_tide[i]=get_df_corr_track(filename_rr, var, filename_time, filename_trend)
        list_df_tide[i]=fillna_interptrack(list_df_tide[i])


    for i in range(len(list_wet)):
        var=list_wet[i]
        list_df_wet[i]=get_df_corr_track(filename_rr, var, filename_time, filename_trend)
        list_df_wet[i]=outliers_treatment(list_df_wet[i], 'WET')
        list_df_wet[i]=fillna_interptrack(list_df_wet[i])


    for i in range(len(list_dry)):
        var=list_dry[i]
        list_df_dry[i]=get_df_corr_track(filename_rr, var, filename_time, filename_trend)
        list_df_dry[i]=outliers_treatment(list_df_dry[i], 'DRY')
        list_df_dry[i]=fillna_interptrack(list_df_dry[i])


    for i in range(len(list_dac)):
        var=list_dac[i]
        list_df_dac[i]=get_df_corr_track(filename_rr, var, filename_time, filename_trend)
        list_df_dac[i]=fillna_interptrack(list_df_dac[i])


    for i in range(len(list_iono)):
        var=list_iono[i]
        if var == 'IONO_XTRACK' :
            list_df_iono[i]=get_df_ionoxt_trace(filename_ionoxt, filename_rr)
            list_df_iono[i]=fillna_interptrack(list_df_iono[i])
        else :
            list_df_iono[i]=get_df_corr_track(filename_rr, var, filename_time, filename_trend)
            list_df_iono[i]=outliers_treatment(list_df_iono[i], 'IONO')
            list_df_iono[i]=fillna_interptrack(list_df_iono[i])


    for i in range(len(list_mss)):
        var=list_mss[i]
        list_df_mss[i]=get_df_corr_track(filename_rr, var, filename_time, filename_trend)
        list_df_mss[i]=outliers_treatment(list_df_mss[i], 'MEAN_SEA_SURFACE')
        list_df_mss[i]=fillna_interptrack(list_df_mss[i])

    # list_sla contains all the possible SLAs obtained by varying the corrections. list_sla is a list of dataframes : (nb_sla,nb_cycles,npts_track)
    list_sla = [o-r-s-t-w-dr-da-i-m for o, r, s, t, w, dr, da, i, m in itertools.product(list_df_orbit, list_df_range, list_df_ssb, list_df_tide, list_df_wet, list_df_dry, list_df_dac  , list_df_iono, list_df_mss)]
    arr_sla=np.array(list_sla)

    # We filter each SLA with a high pass filter of two months
    list_sla_hf = [filter_hp_df(df_sla) for df_sla in list_sla]
    arr_slahf=np.array(list_sla_hf) 

    # We compute the median of the variances of pairwise differences in the filtered SLAs
    # varmed dimensions : (npts_track)
    varmed = vardiff_2a2(list_sla_hf)

    if jason == "J2":
        time = time_tot[(jason_num == 1) | (jason_num == 2)]
    elif jason == "J3":
        time = time_tot[jason_num == 3]

    time_julian = [datetime_to_julian(t) for t in time]
    
    # We propagate the varmed value of each point on all the cycles of the mission. For Jason 2 mission, we propagate also this value on the cycles of Jason 1.
    # df varmed dimensions : (nb_cycles, npts_track)
    df_varmed=pd.DataFrame(index=time_julian, columns=np.arange(np.shape(list_sla[0])[1]))
    for i in range(np.shape(df_varmed)[1]):
        df_varmed[i] = varmed[i]
    return df_varmed.transpose() # df_varmed dimensions : (npts_track, nb_cycles)


def get_variance_all_rr(filename_time, filename_trend, filename_rr_J2, filename_rr_J3, filename_ionoxt, output_variance_file, region, track, station):
    """
    Concatenates the σ values of each mission under a dataframe of dimensions (npts_track, nt_tot). 
    Writes a netcdf file containing the σ values of Jason-2, Jason-3, and of the whole period (Jason-2 extended to Jason-1 and concatenated with Jason-3). 
    """

    df_sla_J1J2 = get_variance_corr_rr_sla_fillna(filename_rr_J2, filename_ionoxt, 'J2', filename_time, filename_trend)
    df_sla_J3 = get_variance_corr_rr_sla_fillna(filename_rr_J3, filename_ionoxt, 'J3', filename_time, filename_trend)
    df_sla = df_sla_J1J2.join(df_sla_J3)
    list_corr = ['sla_hp', 'sla_hp_J2', 'sla_hp_J3']
    list_df = [df_sla, df_sla_J1J2, df_sla_J3]

    #Creation of the variance file
    os.makedirs(output_variance_file, exist_ok=True)
    create_file_nc(f"{output_variance_file}/hfnoise_{region}_{track:03d}_{station:02d}.nc", list_corr, list_df, filename_trend)
    return (list_corr, list_df)

get_variance_all_rr('../../data_hfnoise/ctoh.sla.ref.HF.J1+J2+J3.cci_R13.177.nc', '../../data_hfnoise/ESACCI-SEALEVEL-IND-MSLTR-MERGED-R13_JA_177_07-20260915-v3.1.nc', '../../data_hfnoise/sla_track177_J2.nc', '../../data_hfnoise/sla_track177_J3.nc', '../../data_hfnoise/ESACCI-SEALEVEL-IND-CORR-MERGED-R13_JA_177_07-20260915-v3.1_correction.nc', '../../output_hfnoise/', 'R13', 177,7)
