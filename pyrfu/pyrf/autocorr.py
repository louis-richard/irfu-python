#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Optional

# 3rd party imports
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

# Local imports
from pyrfu.pyrf.calc_dt import calc_dt

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def autocorr(
    inp: DataArray, maxlags: Optional[int] = None, normed: Optional[bool] = True
) -> DataArray:
    r"""Compute the autocorrelation function.

    Parameters
    ----------
    inp : DataArray
        Input time series (scalar of vector).
    maxlags : int, Optional
        Maximum lag in number of points. Default is None (i.e., len(inp) - 1).
    normed : bool, Optional
        Flag to normalize the correlation.

    Returns
    -------
    DataArray
        Autocorrelation function

    Raises
    ------
    TypeError
        If inp is not a xarray.DataArray
    ValueError
        If inp is not a scalar or a vector
    ValueError
        If maxlags is not None or strictly positive < len(inp)

    """
    # Check input type
    if not isinstance(inp, DataArray):
        raise TypeError("inp must be a DataArray")

    # Check input dimension (scalar or vector)
    if inp.ndim > 2:
        raise ValueError("inp must be a scalar or a vector")

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
        coords = [lags[lags >= 0]]
        dims = ["lag"]
    else:
        coords = [lags[lags >= 0], inp[inp.dims[1]].data]
        dims = ["lag", str(inp.dims[1])]

    return xr.DataArray(np.squeeze(out_data), coords=coords, dims=dims)
