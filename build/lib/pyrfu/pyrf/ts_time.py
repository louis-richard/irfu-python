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


def ts_time(time):
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

    time = (time * 1e9).astype("datetime64[ns]")

    out = xr.DataArray(time, coords=[time], dims=["time"])

    return out
