#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

from scipy.stats import kurtosis

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def increments(inp, scale: int = 10):
    r"""Returns the increments of a time series.

    .. math:: y = |x_i - x_{i+s}|

    where :math:`s` is the scale.

    Parameters
    ----------
    inp : xarray.DataArray
        Input time series.
    scale : int, Optional
        Scale at which to compute the increments. Default is 10.

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

    delta_inp = data[scale:, :] - data[:-scale, :]

    result = np.array(delta_inp)

    cols = [inp.coords[dim].data for dim in inp.dims]

    if inp.data.ndim == 1:
        result = xr.DataArray(np.squeeze(result),
                              coords=[cols[0][0:len(delta_inp)]],
                              dims=inp.dims, attrs=inp.attrs)
    else:
        result = xr.DataArray(np.squeeze(result),
                              coords=[cols[0][0:len(delta_inp)], *cols[1:]],
                              dims=inp.dims, attrs=inp.attrs)

    kurt = kurtosis(result, axis=0, fisher=False)

    return kurt, result
