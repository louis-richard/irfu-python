#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Any

# 3rd party imports
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from xarray import DataArray


def ts_tensor_xyz(
    time: NDArray[np.datetime64],
    data: NDArray[np.float32 | np.float64],
    attrs: dict[str, Any] | None = None,
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
        If time or data is not a numpy.ndarray, or attrs not a dict.
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

    # As ts_scalar and ts_vec_xyz (attrs that were not a dict were replaced
    # by {}); a copy, the caller's dict is unchanged
    if attrs is None:
        attrs = {}
    elif isinstance(attrs, dict):
        attrs = dict(attrs)
    else:
        raise TypeError("attrs must be a dict")

    out = xr.DataArray(
        data,
        coords=[time[:], ["x", "y", "z"], ["x", "y", "z"]],
        dims=["time", "rcomp", "ccomp"],
        attrs=attrs,
    )

    out.attrs["TENSOR_ORDER"] = 2

    return out
