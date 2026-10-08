#!/usr/bin/env python
# coding: utf-8

# In[52]:


from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as pyplot
from future.utils import lzip
import datetime
import pandas as pd
import ephem


# In[17]:


def MAD_filter(signal, time, corr, ws, tdelta, plot, COEFF=3, LEFT_EXT_SIZE=None,
               RIGHT_EXT_SIZE=None):
    """
    Filter the signal using a Median Absolute Deviation technique
    signal: array of values from which we compute the median
    time: corresponding time
    ws: size of the window in number of points
    tdelta: time delta between two measures
    ws*tdelta: time size of the window
    COEFF: coeff*mad gives the shift between moving median and high and low threshold, default is 3
    RETURNS signal with outliers replaced by median signal values @ that points,
    returns also the median signal and the index of outliers
    """
    
    # Check input sizes
    # =================
    npts = len(signal)
    if len(time) != npts:
        print('MAD_filter: x and y do not have the same size')
        return (signal)

    # Define arrays
    # =============
    med = np.zeros(npts)
    high = np.zeros(npts)  # for plot
    low = np.zeros(npts)  # for plot
    res = np.copy(signal)  # for plot

    # Get the overall median
    # ======================
    gmed = np.median(signal)
    # Full time length of the signal
    # ==============================
    timespan = time[-1] - time[0]

    # Loop over points in the signal
    # To compute a running median
    # ==============================
    ind = np.arange(npts, dtype = int)
    for pt in range(npts):
        n = 1
        wt = []

        # Ensure we have enough points locally to compute
        # a correct median. Else enlarge the window
        # ===============================================
        while len(wt) < ws and n * ws * tdelta < timespan:
            dif = np.abs(time - time[pt])
            wt = ind[dif < n * ws * tdelta]
            n += 1

        n -= 1

        if len(wt) > 0:

            # local signal and its median
            # ===========================
            windw = signal[wt]
            loc_med = np.median(windw)

            # Isolated values with local median very different
            # from global one are replaced by global median #adapter a la duree/taille des
            # traces?ex iono cycle6 J1213NEA
            # =================================================
            if n >= 2 and (loc_med / gmed > 3. / 2 or loc_med / gmed < 2. / 3):
                # print('test loc median / global median activated')
                loc_med = gmed

            med[pt] = loc_med
        else:
            # Just in case there are no points -> global median
            # But empty corrections are usually not filtered...
            # =================================================
            med[pt] = gmed

    # The thresold we will apply is computed from
    # the median of (signal minus running median)
    # -> Global Median Absolute Deviation
    # ===========================================
    shift_med = np.abs(signal - med)
    gmad = np.median(shift_med)

    # In some cases points are aligned in blocks of values
    # This leads to a zero MAD. In such case, use the mean
    # of distance to median and larger coeff (usually it's already a
    # smoothed corr without much outliers)
    # ======================================================
    if gmad == 0:
        # gmad = np.std(shift_med,ddof=1)
        gmad = np.mean(shift_med)
        COEFF = COEFF * 2.

    # Filter 'bad' values: replace them by the running
    # median at that point
    # ================================================
    ww = np.ravel(np.where(np.abs(signal - med) > COEFF * gmad))
    res[ww] = med[ww]

    # High and low tresholds (for plot)
    # =================================
    high = med + COEFF * gmad
    low = med - COEFF * gmad


    return (res, med, ww, gmad)


# In[18]:


def MEAN_filter(signal, time, corr, ws, tdelta, plot, MIRROR=None, RETURN_EXTENDED=None):
    """
    signal: array of values from which we compute the mean
    time: corresponding time
    ws: size of the window in number of points
    tdelta: time delta between two measures
    ws*tdelta: time size of the window

    """

    if len(time) != len(signal):
        print('MEAN_filter: x and y do not have the same size')
        return (signal)

    if MIRROR == 1:
        sig_extended, tim_extended, left_ext_size = sig_extend_left(signal, time, ws, tdelta,
                                                                    SIMPLE=1)
        sig_extended, tim_extended, right_ext_size = sig_extend_right(sig_extended, tim_extended,
                                                                      ws, tdelta, SIMPLE=1)
    elif MIRROR == 2:
        sig_extended, tim_extended, left_ext_size = sig_extend_left(signal, time, ws, tdelta)
        sig_extended, tim_extended, right_ext_size = sig_extend_right(sig_extended, tim_extended,
                                                                      ws, tdelta)
    else:
        sig_extended = np.copy(signal)
        tim_extended = np.copy(time)

    npts = len(sig_extended)
    mean = np.array(sig_extended * 0)
    timespan = time[-1] - time[0]
    mean_all = np.mean(signal)

    ind = np.arange(npts, dtype = int)
    for pt in range(npts):
        n = 1
        wt = []

        # ensure we have enough points to compute a correct mean
        while len(wt) < ws / 2 and n * ws * tdelta < timespan:
            dif = np.abs(tim_extended - tim_extended[pt])
            wt = ind[dif < n * ws * tdelta]
            n += 1


        if len(wt) > 0:
            # local signal and mean
            windw = sig_extended[wt]  # [sig_extended[u] for u in wt]
            loc_mean = np.mean(windw)

            # Running mean @pt is the local mean
            mean[pt] = loc_mean
        else:
            mean[pt] = mean_all

    # Remove extensions
    if MIRROR is not None and RETURN_EXTENDED is None:
        mean = mean[left_ext_size:]
        mean = mean[:len(mean) - right_ext_size]
        time_out = tim_extended[left_ext_size:]
        time_out = time_out[:len(time_out) - right_ext_size]
    else:
        time_out = tim_extended

    # Plots
    if plot == 2:
        # from matplotlib import pyplot
        # from pylab import rcParams
        # rcParams['figure.figsize'] = 17, 11
        pyplot.plot(time, signal, "yo")
        pyplot.plot(time_out, mean, "g+")
        pyplot.title(corr)
        pyplot.show()

    return (mean, time_out)


# In[19]:


def binom_array(n):
     """
     Compute a binomial coefficients array for Bezier computation
     """
     cn = np.empty(n)
     for i in range(0, n):
         cn[i] = binom(n - 1, i)
     return (cn)


# In[20]:


def binom(n, k):
     """
     A fast way to calculate binomial coefficients by A. Dalke
     """
     if 0 <= k <= n:
         ntok = 1
         ktok = 1
         for t in range(1, min(k, n - k) + 1):
             ntok *= n
             ktok *= t
             n -= 1
         return ntok // ktok
     else:
         return 0


# In[21]:


def sig_extend_left(signal, time, ws, tdelta, SIMPLE=None):
    """
    Extension of the signal by mirroring the left side
    We use either simple mirroring (if SIMPLE != None) or
    inverse mirroring. In this case the mirroring is done from
    a pivot point which abscissa is 0 and ordinate is the mean within the local window
    The extension of the mirrored part is given by ws*tdelta
    """
    # wt = np.ravel(np.where(np.abs(time-time[0]) < ws*tdelta))
    wt = [u for u, v in enumerate(time) if np.abs(v - time[0]) < ws * tdelta]
    if len(wt) < 3:
        #print_msg('signal too short for mirroring left', MSG_DEBUG)
        return (signal, time, 0)
    windw = signal[wt]  # [signal[u] for u in wt]#signal[wt]
    meanw = np.mean(windw)
    # stdw = np.std(windw)

    # if (np.abs(windw[0] - meanw)) > 3.*stdw or SIMPLE != None:
    if SIMPLE is not None:
        # simple mirroring
        vextension = windw[1:][::-1]
        textension = -1. * time[wt[1:]][::-1] + 2. * time[wt[0]]
        signal = np.concatenate((vextension, signal))
        time = np.concatenate((textension, time))

    else:
        # inverse mirroring
        # vextension = -1.*windw[1:][::-1] + 2.* windw[wt[0]]
        vextension = -1. * windw[1:][::-1] + 2. * meanw
        textension = -1. * time[wt[1:]][::-1] + 2. * time[wt[0]]
        signal = np.concatenate((vextension, signal))
        time = np.concatenate((textension, time))

    return (signal, time, len(wt) - 1)


# In[22]:


def sig_extend_right(signal, time, ws, tdelta, SIMPLE=None):
    """
    Extension of the signal by mirroring the right side
    We use either simple mirroring (if SIMPLE != None) or
    inverse mirroring. In this case the mirroring is done from
    a pivot point which abscissa is the one from the last point of serie
    and ordinate is the mean within the local window
    The extension of the mirrored part is given by ws*tdelta
    """
    ls = len(signal)
    # wt = np.ravel(np.where(np.abs(time-time[ls-1]) < ws*tdelta))
    wt = [u for u, v in enumerate(time) if np.abs(v - time[ls - 1]) < ws * tdelta]
    if len(wt) < 3:
        #print_msg('signal too short for mirroring right', MSG_DEBUG)
        return (signal, time, 0)
    windw = signal[wt]  # [signal[u] for u in wt]#signal[wt]
    meanw = np.mean(windw)
    # stdw = np.std(windw)

    # if (np.abs(windw[-1] - meanw)) > 3.*stdw or SIMPLE != None:
    if SIMPLE is not None:
        # simple mirroring
        vextension = windw[:-1][::-1]
        textension = -1. * time[wt[:-1]][::-1] + 2. * time[ls - 1]
        signal = np.concatenate((signal, vextension), 0)
        time = np.concatenate((time, textension), 0)

    else:
        # inverse mirroring
        # vextension = -1.*windw[:-1][::-1] + 2.* signal[ls-1]
        vextension = -1. * windw[:-1][::-1] + 2. * meanw
        textension = -1. * time[wt[:-1]][::-1] + 2. * time[ls - 1]
        signal = np.concatenate((signal, vextension), 0)
        time = np.concatenate((time, textension), 0)

    return (signal, time, len(wt) - 1)


# In[23]:


def reduce_TS_length(data, time, max_len, borders_len):
     """
     Length of the resulting array (run_mean) does not need to be too
long, and should
     not for binomial coefficient calculation (bezier, max 467?). Reduce
it if necessary
     by removing 1 point out of 3 in a while loop until length is below
max_len (but don't modify
     the extremas: borders_len pts)
     """

     #print_msg("Reducing size of time serie", MSG_DEBUG)
     while len(data) > max_len:
         data = [y for x, y in enumerate(data) if
                 x % 3 != 0 or x < borders_len or x > len(data) -
borders_len]
         time = [y for x, y in enumerate(time) if
                 x % 3 != 0 or x < borders_len or x > len(time) -
borders_len]
     data = np.array(data)
     time = np.array(time)

     return (data, time)


# In[24]:


def bezier_interpolate_4(valid_arr_vals, valid_arr_time, cn):
    """
    From the flag array choose the valid values (unflagged) as
    control points to build a bezier parametric curve from start
    to end point. Compute the value for nvalid+dt parameter 't' which value
    is between 0 and 1. cn are binomial coefficients (see binom())
    v2 ->  dtlen has been suppressed
    v3 -> mask0 is given in input
    v4 -> data and time as arrays in unput
    """

    # Get valid points
    nvalid = len(valid_arr_time)

    # Build the points array
    vpoints = lzip(valid_arr_time, valid_arr_vals)

    # Compute the Bezier curve for nvalid points + dt
    xBezier = np.empty(nvalid)
    yBezier = np.empty(nvalid)

    for nn in range(0, nvalid):
        point_t = Bezier(vpoints, float(nn) / (nvalid - 1), cn)
        xBezier[nn] = point_t[0]
        yBezier[nn] = point_t[1]

    return (xBezier, yBezier)


# In[25]:


def Bezier(points, t, cn):
    """
    equivalent to C function bezier_compute
    Computes the x and y values (pt) at parameter t
    of the bezier curve defined by 'points'
    (control points). cn are binomial coefficients (see binom())
    """

    nvalid = len(points)
    order = nvalid - 1

    pt = [0, 0]
    if t == 0:
        pt[0] = points[0][0]
        pt[1] = points[0][1]
    elif t == 1:
        pt[0] = points[nvalid - 1][0]
        pt[1] = points[nvalid - 1][1]
    else:
        for i in range(0, nvalid):
            temp = cn[i] * t ** i * (1 - t) ** (order - i)
            pt[0] += temp * points[i][0]
            pt[1] += temp * points[i][1]

    return (pt)


# In[26]:


def interpolate_corr(obs, obs_times, flag, flag_perm, new_times, new_vals, CONTINUE=None, ALL=None):
    """
    Interpolate at obs_times the values in new_vals if they are flagged (flag array) but not
    permanently flagged
    (perm_flag array). Return an updated version of flag, and interpolated values.
    """
    fill_value=99.9999
    mask = fill_value  # 99.9999

    # Check lengths
    if not len(obs) == len(obs_times) == len(flag) == len(flag_perm):
        print('Interpolate_corr: input arrays have different sizes (obs/obs_times/flags)')
    else:
        npts = len(flag)

    # Copy of the corrections values from obs array
    i_obs_vals = np.copy(obs)

    # Copy of the flag array
    i_flag = np.copy(flag)

    # If ALL is set, interpolate at every obs_times
    # (but not permanently flagged)
    if ALL is not None:
        flag = 1 + np.zeros(npts)

    lnewvals = len(new_vals)

    # Loop over these times
    for i in range(npts):
        if flag_perm[i] == 0 and flag[i] != 0:
            i_flag[i] = 0
            z = dinterpolate_tz(new_vals, new_times,
                                mask, lnewvals, obs_times[i], CONTINUE)
            i_obs_vals[i] = z
            if z == mask:
                i_flag[i] = 1

    return (i_obs_vals, i_flag)


# In[27]:


def dinterpolate_tz(h, t, z, n, time, CONTINUE=None):
    """
    Equivalent to C function dinterpolate_tz: linear interpolation of 'h' at 'time' within 't' array
    """

    # if time is out of bounds
    # modif230915NF: constant value if out of bounds
    if CONTINUE is not None:
        if time < t[0]:
            return (h[0])
        if time > t[-1]:
            return (h[-1])
    else:
        if time < t[0] or time > t[-1]:
            return (z)

    # not enough points
    if n < 2:
        return (h[0])

    # initialization
    k1 = 0
    k2 = n - 1
    k = 0

    # Search two closest neighbours of time
    while k2 - k1 > 1:
        k = int(0.5 * (k2 + k1))

        if time > t[k]:
            k1 = k
            continue

        if time < t[k]:
            k2 = k
            continue

        # If time is part of t array: no interpolation needed
        if time == t[k]:
            if h[k] == z:
                return (z)
            else:
                return (h[k])

    # If neighbouring data equal fill value, return fill value
    if h[k1] == z or h[k2] == z:
        return (z)

    # Interpolate
    r = (t[k2] - time) / (t[k2] - t[k1])

    z = r * h[k1] + (1. - r) * h[k2]

    return (z)


# In[84]:


def iono_editing(data, flag_perm, dflag, tdelta=1.019577/86400, PLOT=False):
    """==============================
       IONOSPHERIC CORRECTION EDITING
       ==============================
       ionospheric correction (bi_freq) is basically a noisy time serie with
       rather slow variations. For the bi-frequence, the editing is done in four steps:
       * Median Absolute deviation filtering (or Hampel type) to eliminate outliers
       * Running average without outliers to smooth signal
       * Bezier curve build from running average to smooth further and to keep
       continuity of the slope in case of land mask
       * Interpolation at EVERY point (flagged or not): the bezier smothed curve replaces the
       original signal
       In the case of a modeled ionospheric path delay the editing is different (GFO/RA2/SRL):
       * Continuity filter to remove outliers
       * Interpolation at the position of those outliers
       
    """

    # Define some masks and params
    # ============================
    
    
    #flag0_perm = np.ravel([flag_perm == 0])
    #flag0_iono = np.ravel([dflag['iono'] == 0])
    #mask = (flag0_perm & flag0_iono)
    mask = flag_perm == 0
 
    # We don't use values which were already flagged
    # ==============================================
    sig = data['iono'][mask]
    tim = data['cnestime'][mask]
    lats = data['latitude'][mask]

    # Index correspondence between data and masked data
    # =================================================
    match_ind = np.ravel(np.where(mask))
    #npts = len(data['cnestime'])
    npts = len(match_ind) # FL correction pour time local
    
    # Check wether the iono correction is from model
    # or computed from bi-freq altimeter: the editing
    # is not done the same way.
    # ===============================================

    # Change the size of the window (for running mean or MAD filter):
    # larger if the local time is between 0 and 6h
    # Only very approx local time needed here
    # ======================================================
    #print('matchind : ', match_ind)
    mean_time = data['cnestime'][match_ind][npts // 2]
    #mean_time = data['cnestime'][npts // 2]
    day_time = mean_time - int(mean_time)
    mean_lon = data['longitude'][match_ind][npts // 2]
    #mean_lon = data['longitude'][npts // 2]
    loc_time = (day_time * 24 + mean_lon * (24. / 360.)) % 24
    add_frames = 0
    if loc_time < 6:
        add_frames = 5

    # Define some constants
    # =====================
    window_size_iono = 20. + add_frames
    tresh_coeff_iono = 3.5
    # Compute MAD filtered signal (filtered out values are replaced by running median), return
    # also running median signal plus the index of new flagged values(ww)
    # ==================================================================================
    madf_sig, run_med, ww, MAD = MAD_filter(sig, tim, 'iono', window_size_iono, tdelta, PLOT,
                                            COEFF=tresh_coeff_iono)

    # update dflag for new outliers
    # =============================
    #if len(ww) > 0:
    #    matched_ww = match_ind[ww]
    #    dflag['iono'][matched_ww] = 1

    # Running mean over MAD filtered signal (reversed mirror on the edges)
    # ====================================================================
    run_mean, tim = MEAN_filter(madf_sig, tim, 'iono', window_size_iono, tdelta, PLOT,
                                MIRROR=2)  # ,RETURN_EXTENDED=1)

    # Reduce size of time serie if necessary
    # ======================================
    if len(run_mean) > 400:
        run_mean, tim = reduce_TS_length(run_mean, tim, 400, 20)

    # Use Bezier curves to smooth final signal
    # and keep continuity when land (island) leads to gaps in signal
    # Then interpolate at all positions the parametric curve obtained
    # ===============================================================
    nvalid = len(tim)
    cn = binom_array(nvalid)
    dflag_new=dflag.copy()
    if nvalid > 1:
        bezier_times, bezier_vals = bezier_interpolate_4(run_mean, tim, cn)
        iono_filt, dflag_new['iono'] = interpolate_corr(data['iono'], data['cnestime'],
                                                               dflag['iono'], flag_perm,
                                                               bezier_times, bezier_vals,
                                                               CONTINUE=1, ALL=1)
    else:
        print('Info: no valid point for interpolation of iono')
        #intcorrs['iono'] = np.copy(data['iono'])
        dflag_new['iono'] = np.copy(dflag['iono'])



    return iono_filt 



def get_iono_filt(dfiono, dflat, dflon, dftime):
    tdelta=1.019577/86400
    PLOT=False
    df_filt=dfiono.copy()
    iono=dfiono.to_numpy()
    lat=dflat.to_numpy()
    lon=dflon.to_numpy()
    time=dftime.to_numpy()
    for cycle in range(len(iono)):
        try :
            #print('cycle : ', cycle)
            data={'iono':iono[cycle], 'latitude':lat[cycle], 'longitude':lon[cycle], 'cnestime':time[cycle]}
            #print('data : ', data)
            #print('ici dftps : ', dftps)
            #print('data[cnestime] : ', data['cnestime'])
            flag_perm = np.isnan(iono[cycle]).astype(int)
            dflag={'iono':flag_perm}
            df_filt.iloc[cycle]=iono_editing(data, flag_perm, dflag)
        except :
            pass
    return df_filt

