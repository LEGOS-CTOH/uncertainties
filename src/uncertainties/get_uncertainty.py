#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:52:27 2024

@author: tolul
"""

import matplotlib.pyplot as plt
import sys
import os
import fnmatch
import rsl_track
import glob
import re
import numpy as np
import math
from netCDF4 import Dataset
import pandas as pd
import yaml
from copy import deepcopy
import xarray as xr



def get_filename_trend(trend_files, region, track, station):
    """
    Returns the name of the filename in the trend directory that corresponds to the region, track and station selected.
    """

    pattern = f"*{region}_*_{track:03d}_{station:02d}*v3.1.nc"
    for file in os.listdir(trend_files):
        if fnmatch.fnmatch(file, pattern):
            return file
    return None


def min_dist_grid(list_lon, list_lat, lon_ref, lat_ref):
    """
    Returns two floats representing the indices of the lon/lat couple closest to lon_ref/lat_ref.                        
    imin is the index for the longitude, jmin for the latitude. 
    """
    
    tab_dist = np.zeros((len(list_lon), len(list_lat)))
    for i in range(len(list_lon)):
        for j in range(len(list_lat)):
            dist = haversine(list_lon[i], list_lat[j], lon_ref, lat_ref)
            tab_dist[i, j] = dist
    imin, jmin = np.unravel_index(np.argmin(tab_dist), tab_dist.shape)
    return imin, jmin


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance (in km) between two points on the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in km
    return c * r


def get_variable_77853(dossier, variable, filename="77853.nc"):
    """
    Returns the wanted variable in the file 77853.nc
    The file 77853.nc has been created by Prandi et al (2021). It contains (among other things) gridded and global values of GIA drift error and WTC low frequency noise variance.
    """

    f = Dataset(dossier + filename, "r")
    var = np.ma.filled(f.variables[variable][:], np.nan)
    f.close()
    return var


def get_gia_track(trend_files, yaml_files, region, track, station):
    """
    Collects the value of GIA drift error (among the gridded ones computed by Prandi et al. 2021) associated to the selected station.
    """

    gia_grid = get_variable_77853(yaml_files, "gia_drift")
    longrid = get_variable_77853(yaml_files, "longitude")
    latgrid = get_variable_77853(yaml_files, "latitude")
    filename_trend = get_filename_trend(trend_files, region, track, station)

    f = Dataset(trend_files + filename_trend, "r")
    lon = f.variables["lon"][0]
    lat = f.variables["lat"][0]
    f.close()

    idx_i, idx_j = min_dist_grid(longrid, latgrid, lon, lat)
    gia = gia_grid[idx_i, idx_j]
    return gia

def get_wtc_track(trend_files, yaml_files, region, track, station):                                          
    """
    Collects the value of WTC low frequency noise variance (among the gridded ones computed by Prandi et al. 2021) associated to the selected station.
    """

    wtc_grid = get_variable_77853(yaml_files, "wtc_noise")                                                            
    longrid = get_variable_77853(yaml_files, "longitude")                                                             
    latgrid = get_variable_77853(yaml_files, "latitude")                                                              
    filename_trend = get_filename_trend(trend_files, region, track, station)                                
                                                                                                                        
    f = Dataset(trend_files + filename_trend, "r")                                                             
    lon = f.variables["lon"][0]                                                                                         
    lat = f.variables["lat"][0]                                                                                         
    f.close()                                                                                                           
                                                                                                                        
    idx_i, idx_j = min_dist_grid(longrid, latgrid, lon, lat)                                                          
    wtc = wtc_grid[idx_i, idx_j]                                                                                        
    return wtc

def create_yaml(
    trend_files,
    region,
    track,
    station,
    variance_files,
    yaml_files,
    uncertainty_files,
):
    """
    Creates a yaml file containing the main information about each error source.
    """

    liste_corr=['sla_hp']
    with open(yaml_files + "rsl_template.yaml") as f:
        r = yaml.safe_load(f)

    r2 = deepcopy(r)
    r2["input"]["filename"] = (
        f"{variance_files+region}/hfnoise_{region}_{track:03d}_{station:02d}.nc"
    )
    r2["hfnoise_template"]["source"] = r2["input"]["filename"]
    r2["errors"] = []
    for corr in liste_corr:
        r2["errors"].append(corr)
        r2[corr] = r2["hfnoise_template"].copy()
        r2[corr]["variable"] = corr

        if "hp" in corr: #hp : it has been filtered with a high pass 
            r2[corr]["timescale"] = 0.16  # 2months
            r2[corr]["type"] = "noise"

    del r2["hfnoise_template"]
    r2["errors"].append("gia_drift")
    r2["errors"].append("wtc_noise") 
    gia = get_gia_track(trend_files, yaml_files, region, track, station)
    # Conversion of GIA from mm to m
    gia = 0.0001 * gia
    gia = float(gia)
    wtc = get_wtc_track(trend_files, yaml_files, region, track, station)                                     
    # Conversion of WTC from mm² to m²                                                                                                
    wtc = 0.0000001 * wtc                                                                                                  
    wtc = float(wtc)
    r2["gia_drift"]["value"] = gia
    r2["wtc_noise"]["value"] = wtc
    r2["errors"].append("orbit")
    r2["orbit"]["value"] = 0.00033
    r2["errors"].append("j1_j2_bias")
    r2["errors"].append("j2_j3_bias")
    r2["j1_j2_bias"]["value"] = 0.000002
    r2['j1_j2_bias']['timing']=2009.07
    r2["j2_j3_bias"]["value"] = 0.000002 
    r2['j2_j3_bias']['timing']=2016.84
    os.makedirs(uncertainty_files + region, exist_ok=True)
    r2["output"] = (
        f"{uncertainty_files+region}/matvar_ci_{region}_{track:03d}_{station:02d}.nc"
    )
    os.makedirs(yaml_files + region, exist_ok=True)
    yamlfn = f"{yaml_files+region}/rsl_{region}_{track:03d}_{station:02d}.yaml"
    with open(yamlfn, "w") as f:
        yaml.dump(r2, f)

    return r2


def add_trendci_to_file(
    trend_files, uncertainty_files, trend_uncertainty_files, region, track, station
):
    """
    Creates a netcdf file in trend_uncertainty_files containing the same variables as trend_file, but with the confidence interval from uncertainty_files added.
    """

    filename = get_filename_trend(trend_files, region, track, station)
    f = Dataset(
        f"{uncertainty_files+region}/matvar_ci_{region}_{track:03d}_{station:02d}.nc",
        "r",
    )
    ci = f.variables["trend_ci"][:]
    f.close()

    os.makedirs(trend_uncertainty_files + region, exist_ok=True)

    data = xr.open_dataset(trend_files + filename)

    #removal of the least square error associated to the trends (it is a simpler way to estimate uncertainties)
    data = data.drop_vars("local_sla_trend_error")                                                                      

    data['u_local_sla_trend']=xr.DataArray(
    # Conversion of the confidence interval from m to mm
    data=ci*1000,
    dims=["nbpoints"],
    coords=dict(
       nbpoints=(["nbpoints"], np.arange(0,len(ci))+1),
    ),
    name="u_local_sla_trend",
    attrs=dict(long_name='Uncertainty in local sea level trend estimation at the 90% confidence level', units='mm/year')
    )
    data["nbpoints"] = data.nbpoints.assign_attrs(units="count", long_name="points number")
    data['distance_to_coast']=data.distance_to_coast.assign_attrs(distance_to_coast_min=float(data.distance_to_coast.attrs['distance_to_coast_min'])*1000)
    data['local_sla_trend']=data.local_sla_trend.assign_attrs(comment="Sea level trends computed from X-TRACK/ALES monthly sea level anomaly between 2002-01-01 and 2021-06-30, corrected from GIA ICE_6G_D (Peltier et al. 2015). Uncertainty components described by the unc_comps attribute", unc_comps="u_local_sla_trend")
    data=data.assign_attrs(source=data.attrs['source']+', CNES database')

    data.to_netcdf(trend_uncertainty_files + region + "/" + filename)



def treat1stat(trend_files,variance_files,yaml_files,uncertainty_files,trend_uncertainty_files,region,track,station):
    """
    Creates the yaml file of errors for the selected station, then computes the error variance covariance matrix, derive the uncertainties and save them in a netcdf file.
    """

    # Creation of a yaml file : "rsl_{region}_{trace:03d}_{station:02d}.yaml" in the yaml_files repertory
    # prerequisites : rsl_template.yaml have to be in the yaml_files repertory
    create_yaml(
        trend_files,
        region,
        track,
        station,
        variance_files,
        yaml_files,
        uncertainty_files,
    )
        
    # Creation of a netcdf file : "matvar_ci_{region}_{track:03d}_{station:02d}.nc" in the uncertainty_files repertory
    # It is an intermediate file of uncertainties that contains the confidence intervals and the error variance covariance matrix sigma
    rsl_track.main(
        f"{yaml_files+region}/rsl_{region}_{track:03d}_{station:02d}.yaml", region, track, station, trend_files
    )
        
    # Creation of the final netcdf file : "ESACCI-SEALEVEL-IND-MSLTR-MERGED-{region}_JA_{track:03d}_{station:02d}-*-v3.1.nc" in the trend_uncertainty_files repertory
    add_trendci_to_file(
        trend_files,
        uncertainty_files,
        trend_uncertainty_files,
        region,
        track,
        station,
        )


def main(trend_files,variance_files,yaml_files,uncertainty_files, trend_uncertainty_files):
    """
    Creates one uncertainty netcdf file per station.
    """

    for f in glob.glob(variance_files+'*/*.nc'):                                                                        
        track = int(re.split("_|-", f)[-2])                                                                             
        station = int(re.split("_|-", f)[-1].replace('.nc',''))                                                                           
        region = re.split("_|-", f)[-3] 
        treat1stat(trend_files, variance_files, yaml_files, uncertainty_files, trend_uncertainty_files, region, track, station)


#############################################################################################################################
#Launching of the code
#############################################################################################################################

main("../../data_uncertainties/trends/",              
    "../../data_uncertainties/variances/",           
    "../../data_uncertainties/yaml/",                
    "../../data_uncertainties/intermediate_uncertainties/",
    "../../output_uncertainties/")

