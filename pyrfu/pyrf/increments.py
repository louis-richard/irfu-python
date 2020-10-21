#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
increments.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from scipy.stats import kurtosis


def increments(x=None, scale=10):
    """Returns the increments of a time series.

    .. math:: y = |x_i - x_{i+s}|

    where :math:`s` is the scale.

    Parameters
    ----------
    x : xarray.DataArray
        Input time series.

    scale : int
        Scale at which to compute the increments.

    Returns
    -------
    kurt : numpy.ndarray
        kurtosis of the increments, one per product, using the Fisher's
        definition (0 value for a normal distribution).

    result : xarray.DataArray
        An xarray containing the time series increments, one per
        product in the original time series.

    """
    assert x is not None and isinstance(x, xr.DataArray)
    assert isinstance(scale, int)

    data = x.data

    if len(data.shape) == 1:
        data = data[:, np.newaxis]
    else:
        pass

    f = np.abs((data[scale:, :] - data[:-scale, :]))

    result = np.array(f)

    t, c = x.dims

    time = x.coords[t].data
    cols = x.coords[c].data

    result = xr.DataArray(result, coords=[time[0:len(f)], cols], dims=[t, c], attrs=x.attrs)

    kurt = kurtosis(result, axis=0, fisher=False)

    return kurt, result
