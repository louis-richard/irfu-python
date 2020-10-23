#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fft_bandpass.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from ..pyrf import calc_fs, ts_scalar, ts_vec_xyz


def fft_bandpass(inp=None, f_min=None, f_max=None):
    """Perform simple bandpass using FFT - returns fields between with ``f_min`` < f < ``f_max``.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series to be bandpassed filtered.

    f_min : float or int
        Minimum frequency of filter, f < ``f_min`` are removed.

    f_max : float or int
        Maximum frequency of filter, f > ``f_max`` are removed.

    Returns
    -------
    out : xarray.DataArray
        Time series of the bandpassed filtered data.

    Notes
    -----
    Can be some spurius effects near boundary. Can take longer interval then use tlim to remove.

    Examples
    --------
    >>> from pyrfu import mms
    >>> b_xyz_bp = mms.fft_bandpass(e_xyz, f_min, f_max)

    """

    assert inp is not None and isinstance(inp, xr.DataArray)
    assert f_min is not None and isinstance(f_min, (float, int))
    assert f_max is not None and isinstance(f_max, (float, int))

    inp_time, inp_data = [inp.time.data, inp.data]

    # Make sure number of elements is an even number, if odd remove last element to make an even
    # number
    n_els = len(inp_data)

    if n_els % 2:
        inp_time = inp_time[:-1]
        inp_data = inp_data[:-1, :]

        n_els = len(inp_data)

    try:
        num_fields = inp_data.shape[1]
    except IndexError:
        num_fields = 1
        inp_data = inp_data[:, np.newaxis]

    # Set NaN values to zero so FFT works
    idx_nans = np.isnan(inp_data)
    inp_data[idx_nans] = 0.0

    # Bandpass filter field data
    fs = calc_fs(inp)
    fn = fs / 2
    f = np.linspace(-fn, fn, n_els)

    # FFT and remove frequencies
    for nn in range(num_fields):
        inp_temp = np.fft.fft(inp_data[:, nn])
        inp_temp = np.fft.fftshift(inp_temp)

        inp_temp[np.abs(f) < f_min] = 0
        inp_temp[np.abs(f) > f_max] = 0

        inp_temp = np.fft.ifftshift(inp_temp)
        inp_data[:, nn] = np.fft.ifft(inp_temp)

    # Put back original NaNs
    inp_data[idx_nans] = np.nan

    # Return data in the same format as input
    if num_fields == 1:
        out = ts_scalar(inp_time, inp_data, attrs=inp.attrs)
    elif num_fields == 3:
        out = ts_vec_xyz(inp_time, inp_data, attrs=inp.attrs)
    else:
        raise ValueError("Invalid shape")

    return out
