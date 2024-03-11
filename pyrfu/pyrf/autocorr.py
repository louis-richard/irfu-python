#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .calc_dt import calc_dt

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def autocorr(inp, maxlags: int = None, normed: bool = True):
    r"""Compute the autocorrelation function

    Parameters
    ----------
    inp : xarray.DataArray
        Input time series (scalar of vector).
    maxlags : int, Optional
        Maximum lag in number of points. Default is None (i.e., len(inp) - 1).
    normed : bool, Optional
        Flag to normalize the correlation.

    Returns
    -------
    out : xarray.DataArray
        Autocorrelation function

    """

    # Check input type
    assert isinstance(inp, xr.DataArray), "inp must be a xarray.DataArray"

    # Check input dimension (scalar or vector)
    assert inp.ndim < 3, "inp must be a scalar or a vector"

    if inp.ndim == 1:
        x = inp.data[:, None]
    else:
        x = inp.data

    n_t = len(inp)

    if maxlags is None:
        maxlags = n_t - 1

    if maxlags >= n_t or maxlags < 1:
        raise ValueError(f"maxlags must be None or strictly positive < {n_t:d}")

    lags = np.linspace(-float(maxlags), float(maxlags), 2 * maxlags + 1, dtype=int)
    lags = lags * calc_dt(inp)

    out_data = np.zeros((maxlags + 1, x.shape[1]))

    for i in range(x.shape[1]):
        correls = np.correlate(x[:, i], x[:, i], mode="full")

        if normed:
            correls /= np.sqrt(np.dot(x[:, i], x[:, i]) ** 2)

        correls = correls[n_t - 1 - maxlags : n_t + maxlags]
        out_data[:, i] = correls[lags >= 0]

    if inp.ndim == 1:
        out = xr.DataArray(
            np.squeeze(out_data),
            coords=[lags[lags >= 0]],
            dims=["lag"],
        )
    else:
        out = xr.DataArray(
            out_data,
            coords=[lags[lags >= 0], inp[inp.dims[1]].data],
            dims=["lag", inp.dims[1]],
        )

    return out
