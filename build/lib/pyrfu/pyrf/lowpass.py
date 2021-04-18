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

"""lowpass.py
@author: Louis Richard
"""

import xarray as xr

from scipy import signal


def lowpass(inp, f_cut, fhz):
    """Filter the data through low or highpass filter with max
    frequency f_cut and subtract from the original.

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

    data = inp.data

    # Remove trend
    data_detrend = signal.detrend(data, type='linear')
    rest = data - data_detrend

    # Elliptic filter
    f_nyq, r_pass, r_stop, order = [fhz / 2, 0.1, 60, 4]

    elliptic_filter = signal.ellip(order, r_pass, r_stop, f_cut / f_nyq,
                                   output='ba')

    # Filter data
    out_data = signal.filtfilt(elliptic_filter[0], elliptic_filter[1],
                               data_detrend)

    out = xr.DataArray(out_data + rest, coords=inp.coords, dims=inp.dims)

    return out
