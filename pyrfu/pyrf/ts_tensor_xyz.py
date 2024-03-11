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


def ts_tensor_xyz(time, data, attrs: dict = None):
    r"""Create a time series containing a 2nd order tensor.

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
        2nd order tensor time series.

    """

    # Check input type
    assert isinstance(time, np.ndarray), "time must be a numpy.ndarray"
    assert isinstance(data, np.ndarray), "data must be a numpy.ndarray"

    # Check input shape must be (n, 3, 3)
    assert data.ndim == 3 and data.shape[1:] == (3, 3)
    assert len(time) == len(data), "Time and data must have the same length"

    if attrs is None or not isinstance(attrs, dict):
        attrs = {}

    out = xr.DataArray(
        data,
        coords=[time[:], ["x", "y", "z"], ["x", "y", "z"]],
        dims=["time", "comp_h", "comp_v"],
        attrs=attrs,
    )

    out.attrs["TENSOR_ORDER"] = 2

    return out
