#!/usr/bin/env python
# coding: utf-8

"""
    perform the analysis of rmsl uncertainties
"""

import argparse
import logging
import yaml

import numpy as np
import os
import fnmatch
import pandas as pd

from io_track import loader, writer
from error_track import Error
from covariance_track import Covariance
from inversion_track import inversion_ols_any_degree as els
from permonth import monthlymean
from permonth import julian_to_years


def get_filename_trend(trend_files, region, track, station):
    """                                                                                                                 
    Returns the name of the filename in the trend directory that corresponds to the region, track and station wanted.   
    """

    pattern = f"*{region}_*_{track:03d}_{station:02d}*v3.1.nc"
    for file in os.listdir(trend_files):
        if fnmatch.fnmatch(file, pattern):
            return file
    return None

def treatment(conf, region, track, station, trend_files):
    liste_trend=[]
    filename_trend=get_filename_trend(trend_files, region, track, station)
    # load data
    logging.info("loading data from %s" % conf["input"]["filename"])
    position, t, z = loader(conf["input"]["filename"], conf["input"]["variable"])
    # convert time/sla to desired units
    tconv = julian_to_years(t)
    zconv = z * conf["z_fact"]
    #print('time and sla loaded')

    # init errors
    logging.info("init errors")
    err_dict = {}
    for err in conf["errors"]:
        #print('err : ', err)
        logging.debug("init error %s" % err)
        err_dict[err] = Error(conf[err])

    # variables to store results
    nx, nt = len(position), len(t) #nx, ny, nt = len(x), len(y), len(t)
    #ng = nx * ny
    trend, trendci = np.full((nx), np.nan, dtype=float), np.full((nx), np.nan, dtype=float
    )
    accel, accelci = np.full((nx), np.nan, dtype=float), np.full(
        (nx), np.nan, dtype=float)

    covariances = np.full((nx, nt, nt), np.nan, dtype=float)

    # loop through track
    logging.info("going through the track...")
    cnt = 1
    for iposition, cposition in enumerate(position):
        logging.debug("processing cell %d/%d" % (cnt, nx))
        #print('iposition : ', iposition, ' cposition : ', cposition)

        # extract the time series
        y_vals = zconv[iposition, :]

        # create & populate the error covariance matrix
        covar = Covariance(tconv)
        for err in conf["errors"]:
            c_err = err_dict[err]
            if c_err.type == "bias":
                covar.add_bias(c_err.value(cposition), c_err.timing)
            if c_err.type == "noise":
                covar.add_noise(c_err.value(iposition), c_err.timescale)
            if c_err.type == "drift":
                covar.add_drift(c_err.value(iposition))
        covariances[iposition, :, :] = covar.omega
        tconv_m, covar_m = monthlymean(tconv, covar.omega, trend_files, filename_trend)
        
        # perform a linear fit & store results 
        beta_hat, var_beta_hat = els(tconv_m, y_vals, covar_m, deg=1)
        trend[iposition] = beta_hat[-1]
        liste_trend.append(trend[iposition])
        trendci[iposition] = var_beta_hat[-1] * conf["ci_fact"]

        # perform quadratic fit & store accelerations
        beta_hat, var_beta_hat = els(tconv_m, y_vals, covar_m, deg=2)
        # acceleration = twice quadratic coeff
        accel[iposition] = beta_hat[-1] * 2.0
        accelci[iposition] = var_beta_hat[-1] * conf["ci_fact"]
        cnt += 1
    
    # were done, need to write output to a file
    dvars_2d = {}   
    dvars_3d = {}

    dvars_2d["trend"] = {
        "data": np.ma.array(trend, mask=np.isnan(trend)),
        "units": "%s/%s" % (conf["output_z_unit"], conf["output_t_unit"]),
        "description": "Sea level anomaly trend",
    }
    dvars_2d["trend_ci"] = {
        "data": np.ma.array(trendci, mask=np.isnan(trendci)),
        "units": "%s/%s" % (conf["output_z_unit"], conf["output_t_unit"]),
        "description": "Confidence interval on sea level anomaly trend",
    }
    dvars_3d["covariance"] = {
        "data": covariances,
        "units": "%s*%s" % (conf["output_z_unit"], conf["output_z_unit"]),
        "description": "Error variance covariance matrix Sigma",
    }

    writer(conf["output"], position, t, dvars_2d, dvars_3d)

def main(yaml_file, region, track, station, trend_files):
    f=open(yaml_file, "r")
    conf = yaml.safe_load(f)
    f.close()
    treatment(conf, region, track, station, trend_files)

