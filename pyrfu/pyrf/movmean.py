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
    r"""Computes running average of the inp over window_size points.

    Parameters
    ----------
    inp : DataArray
        Time series of the input variable.
    window_size : int, Optional
        Number of points to average over. Default is a 2-point running average.

    Returns
    -------
    DataArray
        Time series of the input variable averaged over window_size points,
        at the times of the input where a full window is available. NaNs are
        ignored in the averages (NaN only if the whole window is NaN).

    Raises
    ------
    TypeError
        If inp is not a DataArray.
    ValueError
        If window_size is smaller than 2 or larger than the length of the data.

    Notes
    -----
    Works also with 3D skymap distribution. For an odd window_size the window is
    centred on each time; for an even window_size it extends one point further
    after it than before it.

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
    inp_data: NDArray[Any] = inp.data

    # Checks if window_size is defined
    if window_size is None:
        window_size = 2
    elif window_size < 2 or window_size > len(time):
        raise ValueError(
            "Window size must be at least 2 and at most the length of the data."
        )

    # Cumulative sums (with a leading zero) of the finite values and of their
    # number, so that NaNs are ignored rather than propagated to all the later
    # samples. Double precision (or complex) to limit round-off in the sums.
    is_finite: NDArray[np.bool_] = np.isfinite(inp_data)
    out_dtype = np.result_type(inp_data.dtype, np.float64)
    values: NDArray[Any] = np.where(is_finite, inp_data, 0).astype(out_dtype)
    zeros: NDArray[Any] = np.zeros((1, *inp_data.shape[1:]), dtype=out_dtype)
    cum_sum: NDArray[Any] = np.concatenate([zeros, np.cumsum(values, axis=0)])
    cum_cnt: NDArray[Any] = np.concatenate(
        [zeros.real.astype(np.int64), np.cumsum(is_finite, axis=0)]
    )

    # Sums and numbers of finite values over the windows [i, i + window_size)
    win_sum: NDArray[Any] = cum_sum[window_size:, ...] - cum_sum[:-window_size, ...]
    win_cnt: NDArray[Any] = cum_cnt[window_size:, ...] - cum_cnt[:-window_size, ...]

    # Computes moving average (NaN where the whole window is NaN)
    with np.errstate(invalid="ignore", divide="ignore"):
        out_dat: NDArray[Any] = np.where(win_cnt > 0, win_sum / win_cnt, np.nan)

    # Gets coordinates (time of the centre of the window, one point after the
    # centre for even windows)
    i_start = (window_size - 1) // 2
    coords: list[NDArray[Any]] = [
        time[i_start : i_start + len(out_dat)],
        *[inp.coords[k].data for k in inp.dims[1:]],
    ]

    # Output in DataArray type
    out: DataArray = xr.DataArray(
        out_dat, coords=coords, dims=inp.dims, attrs=dict(inp.attrs)
    )

    return out
