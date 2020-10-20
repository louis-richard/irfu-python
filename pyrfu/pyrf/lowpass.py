#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lowpass.py

@author : Louis RICHARD
"""

import xarray as xr

from scipy import signal


def lowpass(inp=None, f_cut=None, fhz=None):
    """Filter the data through low or highpass filter with max frequency f_cut and subtract from the original.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input variable.

    f_cut : float
        Cutoff frequency.

    fhz : float
        Sampling frequency.

    Returns
    -------
    out : xarray.DataArray
        Time series of the filter data.

    """

    if inp is None or f_cut is None or fhz is None:
        raise ValueError("lowpass requires at least 3 arguments")

    if not isinstance(inp, xr.DataArray):
        raise TypeError("inp must be a DataArray")

    if not isinstance(f_cut, float):
        raise TypeError("f_cut must be a float")

    if not isinstance(fhz, float):
        raise TypeError("fhz must be a float")

    data = inp.data

    # Remove trend
    data_detrend = signal.detrend(data, type='linear')
    rest = data - data_detrend

    # Elliptic filter
    f_nyq, rp, rs, n = [fhz / 2, 0.1, 60, 4]

    [b, a] = signal.ellip(n, rp, rs, f_cut / f_nyq, output='ba')

    # Filter data
    out_data = signal.filtfilt(b, a, data_detrend) + rest

    out = xr.DataArray(out_data, coords=inp.coords, dims=inp.dims)

    return out
