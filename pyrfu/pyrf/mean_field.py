#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Tuple, Union

# 3rd party imports
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from xarray.core.dataarray import DataArray

# Local imports
from pyrfu.pyrf.ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"

NDArrayFloats = NDArray[Union[np.float32, np.float64]]


def mean_field(inp: DataArray, deg: int) -> Tuple[DataArray, DataArray]:
    r"""Estimate the mean and wave fields.

    The mean field is computed by fitting a polynomial of degree `deg` to the
    input data. The wave field is then computed as the difference between the
    input data and the mean field.

    Parameters
    ----------
    inp : DataArray
        Input data.
    deg : int
        Degree of the polynomial.

    Returns
    -------
    Tuple
        Mean field and wave field.

    Raises
    ------
    TypeError
        If input is not a xarray.DataArray.

    """
    # Checking input
    if not isinstance(inp, xr.DataArray):
        raise TypeError("Input must be a xarray.DataArray")

    # Extracting time and data
    time: NDArray[np.datetime64] = inp.time.data
    data: NDArray[np.float64] = inp.data.astype(np.float64)  # force to double precision
    time_ints: NDArray[np.uint16] = np.arange(len(time), dtype=np.uint16)

    # Preallocating output
    inp_mean: NDArray[np.float64] = np.zeros_like(data, dtype=np.float64)
    inp_wave: NDArray[np.float64] = np.zeros_like(data, dtype=np.float64)

    for i in range(data.shape[1]):
        # Polynomial fit
        polynomial_coeffs: NDArray[np.float64] = np.polyfit(time_ints, data[:, i], deg)

        # Computing mean and wave field
        inp_mean[:, i] = np.polyval(polynomial_coeffs, time_ints)
        inp_wave[:, i] = data[:, i] - inp_mean[:, i]

    # Time series
    inp_mean_ts: DataArray = ts_vec_xyz(inp.time.data, inp_mean)
    inp_wave_ts: DataArray = ts_vec_xyz(inp.time.data, inp_wave)

    return inp_mean_ts, inp_wave_ts
