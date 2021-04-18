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

"""increments.py
@author: Louis Richard
"""

import numpy as np
import xarray as xr

from scipy.stats import kurtosis


def increments(inp, scale=10):
    """Returns the increments of a time series.

    .. math:: y = |x_i - x_{i+s}|

    where :math:`s` is the scale.

    Parameters
    ----------
    inp : xarray.DataArray
        Input time series.

    scale : int
        Scale at which to compute the increments.

    Returns
    -------
    kurt : ndarray
        kurtosis of the increments, one per product, using the Fisher's
        definition (0 value for a normal distribution).

    result : xarray.DataArray
        An xarray containing the time series increments, one per
        product in the original time series.

    """

    if inp.data.ndim == 1:
        data = inp.data[:, np.newaxis]
    else:
        data = inp.data

    delta_inp = np.abs((data[scale:, :] - data[:-scale, :]))

    result = np.array(delta_inp)

    time, cols = [inp.coords[dim].data for dim in inp.dims]

    result = xr.DataArray(result, coords=[time[0:len(delta_inp)], cols],
                          dims=inp.dims, attrs=inp.attrs)

    kurt = kurtosis(result, axis=0, fisher=False)

    return kurt, result
