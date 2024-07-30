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


def ts_tensor_xyz(
    time: NDArray[np.datetime64],
    data: NDArray[Union[np.float32, np.float64]],
    attrs: Optional[Mapping[str, object]] = None,
) -> DataArray:
    r"""Create a time series containing a 2nd order tensor.

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
    DataArray
        2nd order tensor time series.

    Raises
    ------
    TypeError
        If time or data is not a numpy.ndarray.
    ValueError
        * If time and data do not have the same length.
        * If data does not have shape (n, 3, 3).

    """

    # Check input type
    if not isinstance(time, np.ndarray):
        raise TypeError("time must be a numpy.ndarray")

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")

    # Check data and time have the same length
    if len(time) != len(data):
        raise ValueError("Time and data must have the same length")

    # Check input shape must be (n, 3, 3)
    if data.ndim != 3 or data.shape[1:] != (3, 3):
        raise ValueError("data must have shape (n, 3, 3)")

    if attrs is None or not isinstance(attrs, dict):
        attrs = {}

    out = xr.DataArray(
        data,
        coords=[time[:], ["x", "y", "z"], ["x", "y", "z"]],
        dims=["time", "rcomp", "ccomp"],
        attrs=attrs,
    )

    out.attrs["TENSOR_ORDER"] = 2

    return out
