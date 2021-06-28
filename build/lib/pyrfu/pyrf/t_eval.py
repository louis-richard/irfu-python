#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import bisect

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def t_eval(inp, times):
    r"""Evaluates the input time series at the target time.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series if the input to evaluate.
    times : ndarray
        Times at which the input will be evaluated.

    Returns
    -------
    out : xarray.DataArray
        Time series of the input at times t.

    """

    idx = np.zeros(len(times))

    for i, time in enumerate(times):
        idx[i] = bisect.bisect_left(inp.time.data, time)

    idx = idx.astype(int)

    if inp.ndim == 2:
        out = xr.DataArray(inp.data[idx, :], coords=[times, inp.comp],
                           dims=["time", "comp"])
    else:
        out = xr.DataArray(inp.data[idx], coords=[times], dims=["time"])

    return out
