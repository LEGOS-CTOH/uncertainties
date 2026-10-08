import numpy as np
import xarray as xr
import pandas as pd
import ephem
import datetime
from astropy.time import Time


def years_to_datetime(year):
    """
    From decimal years to datetime
    """

    yr = int(year)
    days = int((year - int(year)) * 365.25)
    #print(days)
    base_date = pd.to_datetime(f'{yr}-01-01')
    return base_date + pd.DateOffset(days=days)

def julian_to_years(t):
    """
    From days since 1950/01/01 to decimal years
    """

    tref=datetime.datetime(1950,1,1)
    time_date = np.array([tref + datetime.timedelta(days=float(i)) for i in t])
    timey=Time(time_date).decimalyear
    return timey

def datetime_to_years(dt):
    """
    From datetime to decimal years
    """

    yr=dt.year
    to=datetime.datetime(dt.year, 1, 1)
    d=(dt-to).days
    return yr+d/365


def monthlymean(tconv, covar, trend_files, filename_trend):
    """
    Returns the covariance matrix covar monthly averaged, and the corresponding time in decimal years.
    tconv is in decimal years
    """

    #creation of a template dataframe filled with NaNs. The index is the time from the monthly averaged trends
    f=xr.open_dataset(trend_files+filename_trend)
    f.swap_dims({'nbmonths':'time'})
    a=f.resample(time='M').mean()
    dft_trend = pd.DataFrame(a['time'])
    dft_trend.index=a['time']
    dft_trend[0]=np.nan
    
    df_time=pd.DataFrame({'time':tconv})
    df_time['time'] = df_time['time'].apply(years_to_datetime)
    
    # 2021/07/01 is not supposed to appear, as the selected period for this study is 2002/01/01 - 2021/06/30.
    # If it appears, it is because of the time conversions from decimal to datetime.
    # I remove it by replacing it by 2021/06/30
    if df_time.iloc[-1]['time'] == datetime.datetime(2021,7,1):   
        df_time.iloc[-1]['time']=datetime.datetime(2021,6,30)
    
    time=df_time['time'].to_numpy()
    df_covar=pd.DataFrame(covar, index=time, columns=time)
    df_covar=df_covar.resample('M').mean()
    df_covar=df_covar.transpose()
    df_covar=df_covar.resample('M').mean()
    df_covar=pd.merge(df_covar, dft_trend, left_index=True, right_index=True, how='outer')
    
    #removing of the last column which is coming from dft
    df_covar=df_covar.drop(df_covar.columns[-1], axis=1)
    df_covar=df_covar.interpolate()
    df_covar=df_covar.transpose()
    df_covar=pd.merge(df_covar, dft_trend, left_index=True, right_index=True, how='outer')
    
    #removing of the last row which is coming from dft
    df_covar=df_covar.drop(df_covar.columns[-1], axis=1)
    df_covar=df_covar.interpolate()
    time=df_covar.index
    time_yrs=[datetime_to_years(t) for t in time]
    covar_m=df_covar.values
    
    
    # In case of missing first cycles, the corresponding first rows and columns of the covariance matrix have been filled with NaNs
    # These NaNs prevent the uncertainty calculation, so we fill them by propagating the value of the first complete row in diagonal (because the matrix is diagonal)
    if np.isnan(covar_m[0,0]):
        if np.isnan(covar_m[1,1]):                                                                                            
            if np.isnan(covar_m[2,2]):          
                if np.isnan(covar_m[3,3]):                                                                                          
                    covar_m[3, 3:233]=covar_m[4,4:234]                                                                                
                    covar_m[3,233]=covar_m[4,233]                                                                               
                    covar_m[3:233,3]=covar_m[4:234,4]                                                                           
                    covar_m[233,3]=covar_m[233,4]
                covar_m[2, 2:233]=covar_m[3,3:234]                                                                                
                covar_m[2,233]=covar_m[3,233]                                                                               
                covar_m[2:233,2]=covar_m[3:234,3]                                                                           
                covar_m[233,2]=covar_m[233,3]
            covar_m[1, 1:233]=covar_m[2,2:234]
            covar_m[1,233]=covar_m[2,233]                                                                                                                                           
            covar_m[1:233,1]=covar_m[2:234,2]                                                                                                                                   
            covar_m[233,1]=covar_m[233,2]
        covar_m[0,0:233]=covar_m[1,1:234]
        covar_m[0,233]=covar_m[1,233]
        covar_m[0:233,0]=covar_m[1:234,1]
        covar_m[233,0]=covar_m[233,1]
    return time_yrs, covar_m

