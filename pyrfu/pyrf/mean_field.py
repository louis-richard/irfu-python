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
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"

NDArrayFloats = NDArray[Union[np.float32, np.float64]]


def mean_field(inp: DataArray, deg: int) -> Tuple[DataArray, DataArray]:
    r"""Estimate the mean and wave fields.

    The mean field is computed by fitting a polynomial of degree `deg` in time
    to each component of the input data (NaNs are ignored in the fit). The wave
    field is then computed as the difference between the input data and the
    mean field.

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

    # Extracting time (in seconds since the first sample) and data
    time: NDArray[np.datetime64] = inp.time.data
    data: NDArray[np.float64] = inp.data.astype(np.float64)  # force to double precision
    time_sec: NDArray[np.float64] = (time - time[0]) / np.timedelta64(1, "s")

    # Preallocating output
    inp_mean: NDArray[np.float64] = np.full_like(data, np.nan, dtype=np.float64)

    for i in range(data.shape[1]):
        # Polynomial fit (the time is mapped onto [-1, 1], so that the fit stays
        # well-conditioned for long time series and high degrees)
        idx = np.isfinite(data[:, i])

        if np.sum(idx) <= deg:
            continue

        polynomial = np.polynomial.Polynomial.fit(time_sec[idx], data[idx, i], deg)

        # Computing mean field
        inp_mean[:, i] = polynomial(time_sec)

    # Wave field
    inp_wave: NDArray[np.float64] = data - inp_mean

    # Time series
    inp_mean_ts: DataArray = ts_vec_xyz(inp.time.data, inp_mean)
    inp_wave_ts: DataArray = ts_vec_xyz(inp.time.data, inp_wave)

    return inp_mean_ts, inp_wave_ts
