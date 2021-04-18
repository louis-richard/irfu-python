#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

"""resample.py
@author: Louis Richard
"""

import bisect
import warnings
import numpy as np
import xarray as xr

from scipy import interpolate


def _guess_sampling_frequency(ref_time):
    r"""Compute sampling frequency of the time line.

    Parameters
    ----------
    ref_time : np.ndarray
        Time array

    Returns
    -------
    sfy : float
        Sampling frequency.

    """

    n_data = len(ref_time)

    sfy1 = 1 / (ref_time[1] - ref_time[0])
    sfy = None
    not_found = True

    if n_data == 2:
        sfy = sfy1
        not_found = False

    cur, max_try = [2, 10]

    while not_found and cur <= n_data and cur - 3 < max_try:
        sfy = 1 / (ref_time[cur] - ref_time[cur - 1])

        if np.absolute(sfy - sfy1) < sfy * .001:
            not_found = False

            sfy = (sfy + sfy1) / 2
            break

        sfy = sfy1
        cur += 1

    if not_found:
        raise RuntimeError(
            "Cannot guess sampling frequency. Tried {:d} times".format(
                max_try))

    return sfy


def _average(inp_time, inp_data, ref_time, thresh, dt2):
    """Resample inp_data to timeline of ref_time, using half-window of dt2.
    Points above std*tresh are excluded. thresh=0 turns off this option.

    Parameters
    ----------
    inp_time : np.ndarray
        Original timeline of the input.

    inp_data : np.ndarray
        Time series data.

    ref_time : np.ndarray
        Reference timeline.

    thresh : float
        Threshold

    dt2 : float
        Size of the half-window.


    Returns
    -------
    out_data : np.ndarray
        Resampled data.

    """

    try:
        out_data = np.zeros((len(ref_time), inp_data.shape[1]))
    except IndexError:
        inp_data = inp_data[:, None]
        out_data = np.zeros((len(ref_time), inp_data.shape[1]))

    for i, ref_t in enumerate(ref_time):
        idx_l = bisect.bisect_left(inp_time, ref_t - dt2)
        idx_r = bisect.bisect_right(inp_time, ref_t + dt2)

        idx = np.arange(idx_l, idx_r)

        if idx.size == 0:
            out_data[i, ...] = np.nan
        else:
            if thresh:
                std_ = np.std(inp_data[idx, ...], axis=0)
                mean_ = np.mean(inp_data[idx, ...], axis=0)

                assert any(np.isnan(std_))

                for j, stdd in enumerate(std_):
                    if not np.isnan(stdd):
                        idx_r = bisect.bisect_right(
                            inp_data[idx, j + 1] - mean_[j], thresh * stdd)
                        if idx_r:
                            out_data[i, j + 1] = np.mean(
                                inp_data[idx[idx_r], j + 1], axis=0)
                        else:
                            out_data[i, j + 1] = np.nan
                    else:
                        out_data[i, ...] = np.nan

            else:
                out_data[i, ...] = np.mean(inp_data[idx, ...], axis=0)

    if out_data.shape[1] == 1:
        out_data = out_data[:, 0]

    return out_data


def resample(inp, ref, method="", f_s=None, window=None, thresh=0):
    """
    Resample inp to the time line of ref. If sampling of X is more than two
    times higher than Y, we average X, otherwise we interpolate X.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series to resample.

    ref : xarray.DataArray
        Reference time line.

    method : str
        Method of interpolation "spline", "linear" etc.
        (default "linear") if method is given then interpolate
        independent of sampling.

    f_s : float
        Sampling frequency of the Y signal, 1/window.

    window : int or float or np.ndarray
        Length of the averaging window, 1/fsample.

    thresh : float
        Points above STD*THRESH are disregarded for averaging

    Returns
    -------
    out : xarray.DataArray
        Resampled input to the reference time line using the selected method.


    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2015-10-30T05:15:20.000", "2015-10-30T05:16:20.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field and electric field

    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("E_gse_edp_fast_l2", tint, mms_id)

    Resample magnetic field to electric field sampling

    >>> b_xyz = pyrf.resample(b_xyz, e_xyz)

    """

    flag_do = "check"

    if method:
        flag_do = "interpolation"

    if f_s is not None:
        sfy = f_s
    elif window is not None:
        sfy = 1 / window
    else:
        sfy = None

    inp_time = inp.time.data.view("i8") * 1e-9
    ref_time = ref.time.data.view("i8") * 1e-9

    if flag_do == "check":
        if len(ref_time) > 1:
            if not sfy:
                sfy = _guess_sampling_frequency(ref_time)

            if len(inp_time) / (inp_time[-1] - inp_time[0]) > 2 * sfy:
                flag_do = "average"
                warnings.warn("Using averages in resample", UserWarning)
            else:
                flag_do = "interpolation"
        else:
            flag_do = "interpolation"

    assert flag_do in ["average", "interpolation"]

    if flag_do == "average":
        assert not method, "cannot mix interpolation and averaging flags"

        if not sfy:
            sfy = _guess_sampling_frequency(ref_time)

        out_data = _average(inp_time, inp.data, ref_time, thresh, .5 / sfy)

    else:
        if not method:
            method = "linear"

        # If time series agree, no interpolation is necessary.
        if len(inp_time) == len(ref_time) and all(inp_time == ref_time):
            out_data = inp.data.copy()
            coord = [ref.coords["time"].data]

            if len(inp.coords) > 1:
                for k in inp.dims[1:]:
                    coord.append(inp.coords[k].data)

            out = xr.DataArray(out_data, coords=coord, dims=inp.dims,
                               attrs=inp.attrs)

            return out

        tck = interpolate.interp1d(inp_time, inp.data, kind=method, axis=0,
                                   fill_value="extrapolate")
        out_data = tck(ref_time)

    coord = [ref.coords["time"]]

    if len(inp.coords) > 1:
        for k in inp.dims[1:]:
            coord.append(inp.coords[k].data)

    out = xr.DataArray(out_data, coords=coord, dims=inp.dims, attrs=inp.attrs)

    return out
