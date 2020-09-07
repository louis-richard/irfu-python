#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lowpass.py

@author : Louis RICHARD
"""

import xarray as xr

from scipy import signal


def lowpass(inp=None, fcut=None, fhz=None):
    """
    Filter the data through low or highpas filter with max frequency fcut and subtract from the original

    Parameters :
        inp : DataArray
            Time series of the input variable

        fcut : float
            Cutoff frequency

        fhz : float
            Sampling frequency

    Returns :
        out : DataArray
            Time series of the filter data

    """

    if inp is None or fcut is None or fhz is None:
        raise ValueError("lowpass requires at least 3 arguments")

    if not isinstance(inp, xr.DataArray):
        raise TypeError("inp must be a DataArray")

    if not isinstance(fcut, float):
        raise TypeError("fcut must be a float")

    if not isinstance(fhz, float):
        raise TypeError("fhz must be a float")

    data = inp.data

    # Remove detrend
    data_detrend = signal.detrend(data, type='linear')
    rest = data - data_detrend

    # Elliptic filter
    fnyq, rp, rs, n = [fhz / 2, 0.1, 60, 4]

    [b, a] = signal.ellip(n, rp, rs, fcut / fnyq, output='ba')

    # Filter data
    outdata = signal.filtfilt(b, a, data_detrend) + rest

    out = xr.DataArray(outdata, coords=inp.coords, dims=inp.dims)

    return out
