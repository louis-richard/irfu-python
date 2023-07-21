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


def ts_scalar(time, data, attrs: dict = None):
    r"""Create a time series containing a 0th order tensor

    Parameters
    ----------
    time : numpy.ndarray
        Array of times.
    data : numpy.ndarray
        Data corresponding to the time list.
    attrs : dict, Optional
        Attributes of the data list.

    Returns
    -------
    out : xarray.DataArray
        0th order tensor time series.

    """

    # Check input type
    assert isinstance(time, np.ndarray), "time must be a numpy.ndarray"
    assert isinstance(data, np.ndarray), "data must be a numpy.ndarray"

    # Check input shape must be (n, )
    assert data.ndim == 1, "Input must be a scalar"
    assert len(time) == len(data), "Time and data must have the same length"

    if attrs is None or not isinstance(attrs, dict):
        attrs = {}

    out = xr.DataArray(data, coords=[time[:]], dims="time", attrs=attrs)
    out.attrs["TENSOR_ORDER"] = 0

    return out
