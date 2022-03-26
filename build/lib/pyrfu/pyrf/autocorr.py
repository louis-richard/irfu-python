#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #rd party imports
import numpy as np
import xarray as xr

# Local imports
from .calc_dt import calc_dt

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2022"
__license__ = "MIT"
__version__ = "2.3.12"
__status__ = "Prototype"


def autocorr(inp, maxlags: int = None, normed: bool = True):
    r"""Compute the autocorrelation function

    Parameters
    ----------
    inp : xarray.DataArray
        Input time series.
    maxlags : int, Optional
        Maximum lag in number of points. Default is None (i.e., len(inp) - 1).
    normed : bool, Optional
        Flag to normalize the corelation.

    Returns
    -------
    out : xarray.DataArray
        Autocorrelation function

    """

    x = np.atleast_2d(inp.data.copy())
    n = len(inp)

    if maxlags is None:
        maxlags = n - 1

    if maxlags >= n or maxlags < 1:
        raise ValueError(f"maxlags must be None or strictly positive < {n:d}")

    lags = np.arange(-maxlags, maxlags + 1)
    lags = lags * calc_dt(inp)

    out_data = np.zeros_like(x)

    for i in range(x.shape[1]):
        correls = np.correlate(x[:, i], x[:, i], mode="full")

        if normed:
            correls /= np.sqrt(np.dot(x[:, i], x[:, i]) ** 2)

        correls = correls[n - 1 - maxlags:n + maxlags]
        out_data[:, i] = correls[lags >= 0]

    if len(inp.shape) == 1:
        out = xr.DataArray(np.squeeze(out_data), coords=[lags[lags >= 0]],
                           dims=["lag"])
    elif len(inp.shape) == 2:
        out = xr.DataArray(out_data,
                           coords=[lags[lags >= 0], inp[inp.dims[1]].data],
                           dims=["lag", inp.dims[1]])
    else:
        raise ValueError("invalid shape!!")

    return out
