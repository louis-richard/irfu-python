#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def ts_vec_xyz(time, data, attrs: dict = None):
    r"""Create a time series containing a 1st order tensor.

    Parameters
    ----------
    time : ndarray
        Array of times.
    data : ndarray
        Data corresponding to the time list.
    attrs : dict, Optional
        Attributes of the data list.

    Returns
    -------
    out : xarray.DataArray
        1st order tensor time series.

    """

    assert data.ndim == 2 and data.shape[1] == 3
    assert len(time) == len(data), "Time and data must have the same length"

    if attrs is None:
        attrs = {}

    out = xr.DataArray(data, coords=[time[:], ["x", "y", "z"]],
                       dims=["time", "comp"], attrs=attrs)

    out.attrs["TENSOR_ORDER"] = 1

    return out
