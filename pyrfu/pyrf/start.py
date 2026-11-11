#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party import
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def start(inp):
    r"""Gives the first time of the time series.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series.

    Returns
    -------
    out : float
        First time, in seconds since 1970-01-01 (unix time).

    Raises
    ------
    TypeError
        If inp is not a xarray.DataArray.
    ValueError
        If the first dimension of inp is not time.

    """

    # Check input type
    if not isinstance(inp, xr.DataArray):
        raise TypeError("inp must be a xarray.DataArray")

    # Make sure this is a time series
    if list(inp.dims)[0] != "time":
        raise ValueError("inp must be a time series")

    # From the time in ns (in us it was 1000 times too small)
    out = inp.time.data[0].astype("datetime64[ns]").astype(np.int64) * 1e-9

    return out
