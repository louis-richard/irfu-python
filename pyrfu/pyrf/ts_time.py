#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def ts_time(time, attrs: dict = None):
    r"""Creates time line in DataArray.

    Parameters
    ----------
    time : ndarray
        Input time line.

    Returns
    -------
    out : xarray.DataArray
        Time series of the time line.

    """

    assert isinstance(time, np.ndarray)

    if time.dtype == np.float64:
        time = (time * 1e9).astype("datetime64[ns]")
    elif time.dtype == "datetime64[ns]":
        pass
    else:
        raise TypeError("time must be in unix (float64) or numpy.datetime64")

    out = xr.DataArray(time, coords=[time], dims=["time"], attrs=attrs)

    return out
