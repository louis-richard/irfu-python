#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ts_scalar.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr


def ts_scalar(t=None, data=None, attrs=None):
    """Create a time series containing a 0th order tensor

    Parameters
    ----------
    t : numpy.ndarray
        Array of times.

    data : numpy.ndarray
        Data corresponding to the time list.

    attrs : dict
        Attributes of the data list.

    Returns
    -------
    out : xarray.DataArray
        0th order tensor time series.

    """

    assert t is not None and isinstance(t, np.ndarray)
    assert data is not None and isinstance(data, np.ndarray)

    if data.ndim != 1:
        raise TypeError("Input must be a scalar")

    if len(t) != len(data):
        raise IndexError("Time and data must have the same length")

    flag_attrs = True

    if attrs is None:
        flag_attrs = False

    out = xr.DataArray(data, coords=[t[:]], dims="time")

    if flag_attrs:
        out.attrs = attrs

    out.attrs["TENSOR_ORDER"] = 0

    return out
