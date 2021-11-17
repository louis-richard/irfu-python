#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def pvi(inp, scale: int = 10):
    r"""Returns the PVI of a time series.

    .. math::

        y = \frac{|x_i - x_{i+s}|^2}{<|x_i - x_{i+s}|^2>}

    where :math:`s` is the scale.

    Parameters
    ----------
    inp : xarray.DataArray
        Input time series.
    scale : int, Optional
        Scale at which to compute the PVI. Default is 10.

    Returns
    -------
    values : xarray.DataArray
        An xarray containing the pvi of the original time series.

    """

    if len(inp.data.shape) == 1:
        data = inp.data[:, np.newaxis]
    else:
        data = inp.data

    delta_inp = np.abs((data[scale:, :] - data[:-scale, :]))
    delta_inp2 = np.sum(delta_inp ** 2, axis=1)
    sigma = np.mean(delta_inp2)
    result = np.array(delta_inp2 / sigma)

    time = inp.coords[inp.dims[0]].data

    result = xr.DataArray(result, coords=[time[0:len(delta_inp)]],
                          dims=[inp.dims[0]], attrs=inp.attrs)

    result.attrs["units"] = "dimensionless"

    return result
