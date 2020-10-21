#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ts_vec_xyz.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr


def ts_vec_xyz(t=None, data=None, attrs=None):
    """Create a time series containing a 1st order tensor.

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
        1st order tensor time series.

    """

    assert t is not None and isinstance(t, np.ndarray)
    assert data is not None and isinstance(data, np.ndarray)
    assert data.ndim == 2 and data.shape[1] == 3

    if len(t) != len(data):
        raise IndexError("Time and data must have the same length")

    flag_attrs = True

    if attrs is None:
        flag_attrs = False

    out = xr.DataArray(data, coords=[t[:], ["x", "y", "z"]], dims=["time", "comp"])

    if flag_attrs:
        out.attrs = attrs

        out.attrs["TENSOR_ORDER"] = 1
    else:
        out.attrs["TENSOR_ORDER"] = 1

    return out
