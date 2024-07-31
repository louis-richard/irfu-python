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
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

NDArrayFloats = NDArray[Union[np.float32, np.float64]]


def ts_spectr(
    time: NDArray[np.datetime64],
    energy: NDArrayFloats,
    data: NDArrayFloats,
    comp_name: Optional[str] = None,
    attrs: Optional[Mapping[str, object]] = None,
) -> DataArray:
    r"""Create a time series containing a spectrum

    Parameters
    ----------
    time : numpy.ndarray
        Array of times.
    energy : numpy.ndarray
        Y value of the spectrum (energies, frequencies, etc.)
    data : numpy.ndarray
        Data of the spectrum.
    attrs : dict, Optional
        Attributes of the data list.

    Returns
    -------
    out : xarray.DataArray
        Time series of a spectrum

    """

    # Check input type
    if not isinstance(time, np.ndarray):
        raise TypeError("time must be a numpy.ndarray")

    if not isinstance(energy, np.ndarray):
        raise TypeError("time must be a numpy.ndarray")

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")

    # Check input shape must be (n, m, )
    if data.ndim != 2:
        raise ValueError("Input must be a spectrum")

    if len(time) != data.shape[0]:
        raise ValueError("Shape mismatch. Time and data must have the same length")

    if len(energy) != data.shape[1]:
        raise ValueError("Shape mismatch. Energy and data must have the same length")

    if comp_name is None:
        comp_name = "energy"

    if attrs is None or not isinstance(attrs, dict):
        attrs = {}

    out: DataArray = xr.DataArray(
        data, coords=[time, energy], dims=["time", comp_name], attrs=attrs
    )
    out.attrs["TENSOR_ORDER"] = 0

    return out
