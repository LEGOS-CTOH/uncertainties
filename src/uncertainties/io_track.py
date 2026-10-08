#!/usr/bin/env python
# coding: utf-8

'''
    I/O utilities
'''

from netCDF4 import Dataset
import numpy as np

def loader(fname, 
    varname, 
    position='position',
    time='time'):
    '''
    reads data from netCDF file available from SEANOE
    '''
    with Dataset(fname, 'r') as nc:
        #longitudes = nc.variables[lon][:]
        #latitudes = nc.variables[lat][:]
        positions = nc.variables[position][:]
        dates = nc.variables[time][:]
        sla = nc.variables[varname][:]
    #return longitudes, latitudes, dates, sla.filled(np.nan)
    return positions, dates, sla.filled(np.nan)

def writer(fname, 
    position,
    time,
    d1d,
    d3d):
    ''' write the output to a NetCDF file '''
    with Dataset(fname, 'w') as nc:
        
        # create dimensions
        dx = nc.createDimension('x', size=len(position))
        dt = nc.createDimension('t', size=len(time))
        
        # lat/lon
        vx = nc.createVariable('points', 'f', dimensions=('x'))
        vx.description = 'index of points along the track'
        vx[:] = position
        
        vt = nc.createVariable('time', 'f', dimensions=('t'))
        vt.unit = 'CNES Julian day (days since 01-01-1950)'
        vt[:] = time
        
        # variables 1D                                                                     
        for v in d1d:                                                                      
            ncv = nc.createVariable(v, 'f8', dimensions=('x'))                             
            ncv.units = d1d[v]['units']                                                    
            ncv.description = d1d[v]['description']                                        
            ncv[:] = d1d[v]['data']

        # variables 3D 
        for v in d3d:
            ncv = nc.createVariable(v, 'f8', dimensions=('x','t','t'))
            ncv.units = d3d[v]['units']
            ncv.description = d3d[v]['description']
            ncv[:] = d3d[v]['data']
        
