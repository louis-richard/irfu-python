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


def end(inp):
    """Gives the last time of the time series in unix format.

    Parameters
    ----------
    inp : xarray.DataArray or xarray.Dataset
        Time series of the input variable.

    Returns
    -------
    out : float
        Value of the last time in unix format.

    Raises
    ------
    TypeError
        If inp is not a xarray.DataArray or a xarray.Dataset.

    """

    if not isinstance(inp, (xr.DataArray, xr.Dataset)):
        raise TypeError("inp must be a xarray.DataArray or xarray.Dataset")

    # From the time in ns (in us it was 1000 times too small)
    out = inp.time.data[-1].astype("datetime64[ns]").astype(np.int64) / 1e9

    return out
