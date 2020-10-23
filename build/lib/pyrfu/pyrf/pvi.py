#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pvi.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr


def pvi(x=None, scale=10):
    """Returns the PVI of a time series.

    .. math::

        y = \\frac{|x_i - x_{i+s}|^2}{<|x_i - x_{i+s}|^2>}

    where :math:`s` is the scale.

    Parameters
    ----------
    x : xarray.DataArray
        Input time series.

    scale : int
        Scale at which to compute the PVI.

    Returns
    -------
    values : xarray.DataArray
        An xarray containing the pvi of the original time series.

    """

    assert x is not None and isinstance(x, xr.DataArray)
    assert isinstance(scale, int)

    data = x.data

    if len(data.shape) == 1:
        data = data[:, np.newaxis]
    else:
        pass

    f = np.abs((data[scale:, :] - data[:-scale, :]))
    f2 = np.sum(f ** 2, axis=1)
    sigma = np.mean(f2)
    result = np.array(f2 / sigma)

    t = x.dims[0]
    time = x.coords[t].data

    result = xr.DataArray(result, coords=[time[0:len(f)]], dims=[t], attrs=x.attrs)

    result.attrs["units"] = "dimensionless"

    return result
