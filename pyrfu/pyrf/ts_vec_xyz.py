#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Mapping, Optional, Union

# 3rd party imports
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from xarray.core.dataarray import DataArray

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def ts_vec_xyz(
    time: NDArray[np.datetime64],
    data: NDArray[Union[np.float32, np.float64]],
    attrs: Optional[Mapping[str, object]] = None,
) -> DataArray:
    r"""Create a time series containing a 1st order tensor.

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
    out : DataArray
        1st order tensor time series.

    Raises
    ------
    TypeError
        If time or data is not a numpy.ndarray or if attrs is not a dict.
    ValueError
        If data does not have shape (n, 3) or if time and data do not have the same
        length.

    """
    # Check input type
    if not isinstance(time, np.ndarray):
        raise TypeError("time must be a numpy.ndarray")

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")

    # Check input shape must be (n, 3)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("data must have shape (n, 3)")

    # Check input length
    if len(time) != len(data):
        raise ValueError("Time and data must have the same length")

    if attrs is None:
        attrs = {"TENSOR_ORDER": 1}
    elif isinstance(attrs, dict):
        attrs["TENSOR_ORDER"] = 1
    else:
        raise TypeError("attrs must be a dict")

    out: DataArray = xr.DataArray(
        data,
        coords=[time[:], ["x", "y", "z"]],
        dims=["time", "comp"],
        attrs=attrs,
    )

    return out
