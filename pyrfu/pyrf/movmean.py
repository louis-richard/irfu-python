#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Any, Optional

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


def movmean(inp: DataArray, window_size: Optional[int] = None) -> DataArray:
    r"""Computes running average of the inp over npts points.

    Parameters
    ----------
    inp : DataArray
        Time series of the input variable.
    window_size : int, Optional
        Number of points to average over. Default is a 4-point running average.

    Returns
    -------
    DataArray
        Time series of the input variable averaged over npts points.

    Raises
    ------
    TypeError
        If inp is not a DataArray.
    ValueError
        If window_size is smaller than 2 or larger than the length of the data.

    Notes
    -----
    Works also with 3D skymap distribution.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]

    Spacecraft index

    >>> mms_id = 1

    Load ion pressure tensor

    >>> p_xyz_i = mms.get_data("Pi_gse_fpi_brst_l2", tint, mms_id)

    Running average the pressure tensor over 10s

    >>> fs = pyrf.calc_fs(p_xyz_i)
    >>> p_xyz_i = pyrf.movmean(p_xyz_i, int(10 * fs))

    """

    # Checks if input is a DataArray
    if not isinstance(inp, xr.DataArray):
        raise TypeError("Input must be a xarray.DataArray")

    # Gets input data and time
    time: NDArray[np.datetime64] = inp.time.data
    inp_data: NDArray[np.float32] = inp.data

    # Checks if window_size is defined
    if window_size is None:
        window_size = 2
    elif window_size < 2 or window_size > len(time):
        raise ValueError(
            "Window size must be larger than 2 and smaller than the length of the data."
        )

    # Checks if window_size is odd. If not, makes it odd.
    if window_size % 2:
        window_size -= 1

    # Preallocates output
    out_dat: NDArray[np.float32] = np.empty(
        (len(time) - window_size, *inp_data.shape[1:]), dtype=np.float32
    )

    # Computes moving average
    cum_sum: NDArray[np.float32] = np.cumsum(inp_data, axis=0)
    out_dat[...] = (
        cum_sum[window_size:, ...] - cum_sum[:-window_size, ...]
    ) / window_size

    # Gets coordinates
    coords: list[NDArray[Any]] = [
        time[int(window_size / 2) : -int(window_size / 2)],
        *[inp.coords[k].data for k in inp.dims[1:]],
    ]

    # Output in DataArray type
    out: DataArray = xr.DataArray(
        out_dat, coords=coords, dims=inp.dims, attrs=inp.attrs
    )

    return out
